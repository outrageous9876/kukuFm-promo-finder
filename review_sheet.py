"""Build outputs/review_sheet.csv for a manual quality check (no API calls).   python review_sheet.py"""
import csv
import json
import random
import sys
from pathlib import Path

import rank
from promo_finder import clock

SEED = 42           # fixed, so the same results give the same sample
SAMPLE = 5          # clips per random group
OPENING = 3.0       # seconds counted as the opening
REVIEW_COLUMNS = ["hook_ok", "cliffhanger_ok", "standalone_ok", "boundaries_ok", "verdict", "note"]


def lines(c, events, include_camera):
    """The clip's script lines as '12.3s SPEAKER: text', seconds from the clip start."""
    out = []
    for e in events[c["start_event"]:c["end_event"] + 1]:
        if e["type"] == "CAMERA" and not include_camera:
            continue
        who = e["speaker"] if e["type"] == "DIALOGUE" else e["type"]
        out.append((e["start"] - c["start_time"], f"{e['start'] - c['start_time']:.1f}s {who}: {e['text']}"))
    return out


def row(group, c, events):
    """One sheet row: facts about the clip, its real script, and empty columns for the reviewer."""
    every_line = lines(c, events, include_camera=True)
    spoken = [text for _, text in lines(c, events, include_camera=False)]
    return {
        "group": group, "rank": c.get("rank", ""), "clip_id": c["id"], "episode": c["episode"],
        "start": clock(c["start_time"]), "end": clock(c["end_time"]), "duration": c["duration"],
        "status": c["status"],
        "warnings": "; ".join(([c["reason"]] if c["reason"] else []) + c["warnings"]),
        "opening_3s": "\n".join(text for offset, text in every_line if offset <= OPENING),
        "ending": "\n".join(spoken[-3:]),
        "full_script": "\n".join(spoken),
        **{column: "" for column in REVIEW_COLUMNS},
    }


def build(out_dir="outputs"):
    ranked = json.loads(Path(out_dir, "ranked.json").read_text(encoding="utf-8"))
    rank_of = {r["id"]: r["rank"] for r in ranked}
    everything = []  # every checked clip that has real boundaries, with its rank if it has one
    for path in sorted(Path(out_dir, "checked").glob("episode_*.json")):
        episode = int(path.stem.split("_")[1])
        for r in json.loads(path.read_text(encoding="utf-8")):
            cid = f"E{episode:02d}-C{r['clip']}"
            if "start_time" in r:
                everything.append(dict(r, episode=episode, id=cid, **({"rank": rank_of[cid]} if cid in rank_of else {})))

    pick = random.Random(SEED)
    top10 = sorted((c for c in everything if "rank" in c), key=lambda c: c["rank"])[:10]
    taken = {c["id"] for c in top10}
    pools = {"accepted": [c for c in everything if c["status"] == "ACCEPTED" and c.get("rank", 0) > 30],
             "review": [c for c in everything if c["status"] == "NEEDS_REVIEW" and c["id"] not in taken],
             "rejected": [c for c in everything if c["status"] == "REJECTED"]}
    groups = [("top10", top10)] + [(name, pick.sample(pool, min(SAMPLE, len(pool))))
                                   for name, pool in pools.items()]

    episodes = rank.load_episodes(out_dir)
    rows = [row(name, c, episodes[c["episode"]]["events"]) for name, clips in groups for c in clips]
    with open(Path(out_dir, "review_sheet.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {Path(out_dir, 'review_sheet.csv')}: " + ", ".join(f"{n} {len(c)}" for n, c in groups))
    return rows


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "outputs")
