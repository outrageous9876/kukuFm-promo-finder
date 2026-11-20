"""Simple Streamlit page to browse promo results or run a new show.  Start with:  streamlit run app.py"""
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

import rank
from promo_finder import clock, describe
from run import run_show

load_dotenv()
st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Microdrama promo finder")


def result_folders():
    """outputs/ and any show folder inside it that has a checked/ folder."""
    root = Path("outputs")
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    return [p for p in candidates if Path(p, "checked").is_dir()]


def load_candidates(folder):
    """All candidates of a folder, best first: by rank from ranked.json (written by rank.py)
    when it exists, then the unranked ones (e.g. REJECTED) by weighted score."""
    ranked_file = Path(folder, "ranked.json")
    ranked = json.loads(ranked_file.read_text(encoding="utf-8")) if ranked_file.exists() else []
    by_id = {r["id"]: r for r in ranked}
    rows = []
    for path in sorted(Path(folder, "checked").glob("episode_*.json")):
        episode = int(path.stem.split("_")[1])
        for r in json.loads(path.read_text(encoding="utf-8")):
            cid = f"E{episode:02d}-C{r['clip']}"
            rows.append(by_id.get(cid, dict(r, episode=episode, id=cid)))
    return sorted(rows, key=lambda r: (r.get("rank", float("inf")), -r.get("weighted_score", 0)))


load_episodes = st.cache_data(rank.load_episodes)  # same loader as rank.py, cached for the app


def script_line(r, key, episodes):
    """The actual script text of the event r[key]: 'SPEAKER: line' or 'ACTION: text'."""
    events = episodes[r["episode"]]["events"]
    n = r.get(key)
    if not (isinstance(n, int) and 0 <= n < len(events)):
        return ""
    e = events[n]
    return f"{e['speaker'] if e['type'] == 'DIALOGUE' else e['type']}: {e['text']}"


def to_table(rows, episodes):
    """The columns shown in the table (rank and judge_reason only when ranking exists)."""
    ranked = any("rank" in r for r in rows)
    return pd.DataFrame([{
        **({"rank": r.get("rank")} if ranked else {}),
        "episode": r["episode"], "clip": r.get("clip"),
        "start": clock(r["start_time"]) if "start_time" in r else "",
        "end": clock(r["end_time"]) if "end_time" in r else "",
        "duration": r.get("duration"), "score": r.get("weighted_score"), "status": r["status"],
        "hook_line": script_line(r, "hook_event", episodes),
        "cliffhanger_line": script_line(r, "cliffhanger_event", episodes),
        "hook (AI summary)": r.get("hook"), "cliffhanger (AI summary)": r.get("cliffhanger"),
        "why_it_works": r.get("why_it_works"),
        **({"judge_reason": r.get("judge_reason")} if ranked else {}),
        "warnings": "; ".join(([r["reason"]] if r["reason"] else []) + r["warnings"]),
    } for r in rows])


def show_script(r, episodes):
    """The real script lines of one clip, with hook and cliffhanger highlighted."""
    if "start_time" not in r:
        st.info("This candidate has no valid boundaries, so there is no script to show.")
        return
    events = episodes[r["episode"]]["events"]
    st.markdown(f"**Episode {r['episode']}, clip {r['clip']}**: {r['status']} | "
                f"{clock(r['start_time'])} -> {clock(r['end_time'])} ({r['duration']}s)")
    st.caption(f"New viewer needs to know: {r.get('standalone_context')}")
    for i in range(r["start_event"], r["end_event"] + 1):
        line = describe(i, events[i])
        if i == r.get("hook_event"):
            st.success(f"HOOK  {line}")
        elif i == r.get("cliffhanger_event"):
            st.error(f"CLIFFHANGER  {line}")
        else:
            st.text(line)


def table_with_script(rows, episodes, key):
    """A clickable table; selecting a row shows that clip's script below it."""
    if not rows:
        st.write("Nothing to show.")
        return
    picked = st.dataframe(to_table(rows, episodes), hide_index=True, on_select="rerun",
                          selection_mode="single-row", key=key, alt="Promo candidates")
    if picked.selection.rows:
        show_script(rows[picked.selection.rows[0]], episodes)
    else:
        st.caption("Click a row to see the script lines of that clip.")


# ---------- Sidebar: choose existing results, or run a new show ----------
st.sidebar.header("Results")
mode = st.sidebar.radio("Source", ["Existing results", "Upload a ZIP"])
api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input(
    "Gemini API key (not saved)", type="password")
folder = None

if mode == "Existing results":
    folders = result_folders()
    if folders:
        folder = st.sidebar.selectbox("Results folder", folders, format_func=str)
    else:
        st.sidebar.write("No results in outputs/ yet.")
else:
    upload = st.sidebar.file_uploader("Show ZIP (screenplay .txt files)", type="zip")
    if upload and st.sidebar.button("Run", disabled=not api_key):
        bar = st.progress(0.0, text="Starting...")
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp, upload.name)
            zip_path.write_bytes(upload.getvalue())
            folder = run_show(zip_path, genai.Client(api_key=api_key), name=Path(upload.name).stem,
                              on_progress=lambda done, total, n, status: bar.progress(
                                  done / total, text=f"Episode {n}: {status[:60]} ({done}/{total})"))
        st.session_state["ran"] = str(folder)
    if not api_key:
        st.sidebar.warning("Add GEMINI_API_KEY to .env or type it above to run.")
    folder = st.session_state.get("ran")

if not folder:
    st.stop()

# ---------- Main page ----------
rows = load_candidates(folder)
episodes = load_episodes(str(folder))
statuses = [r["status"] for r in rows]
cols = st.columns(5)
cols[0].metric("Episodes", len({r["episode"] for r in rows}))
cols[1].metric("Candidates", len(rows))
for col, status in zip(cols[2:], ["ACCEPTED", "NEEDS_REVIEW", "REJECTED"]):
    col.metric(status, statuses.count(status))

all_tab, unsure_tab = st.tabs(["All candidates", "Rejected / needs review"])
with all_tab:
    if not any("rank" in r for r in rows):
        st.caption("Sorted by weighted score. Scores are the model's self-ratings and don't "
                   "separate clips well; use rank.")
    usable = [r for r in rows if r["status"] != "REJECTED"]  # REJECTED only in the other tab
    left, right = st.columns(2)
    pick_status = left.multiselect("Status", ["ACCEPTED", "NEEDS_REVIEW"])
    pick_episode = right.multiselect("Episode", sorted({r["episode"] for r in usable}))
    shown = [r for r in usable if (not pick_status or r["status"] in pick_status)
             and (not pick_episode or r["episode"] in pick_episode)]
    table_with_script(shown, episodes, key="all")
    st.download_button("Download CSV", to_table(shown, episodes).to_csv(index=False),
                       "promos.csv", "text/csv")
    st.download_button("Download JSON", json.dumps(shown, indent=2, ensure_ascii=False),
                       "promos.json", "application/json")
with unsure_tab:
    table_with_script([r for r in rows if r["status"] != "ACCEPTED"], episodes, key="unsure")
