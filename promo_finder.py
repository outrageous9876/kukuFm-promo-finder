"""Step 3: ask Gemini for promo clips (as event numbers), then check and score them in plain Python."""
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from parser import load_all

load_dotenv()  # reads GEMINI_API_KEY and GEMINI_MODEL from .env
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
MIN_SECONDS, MAX_SECONDS = 20, 60
HOOK_WINDOW = 3.0         # the hook must start within 3s of the clip start
CLIFF_WINDOW = 5.0        # at most 5s of clip may remain after the cliffhanger ends
SILENT_WINDOW = 3.0       # look for a silent ending in the last 3s
MAX_OVERLAP = 0.5         # two clips may share at most half of the shorter one
PAUSE_SECONDS = 4
MAX_TRIES = 3
# Weight of each 1-10 model score in the final 0-100 score.
WEIGHTS = {"hook_score": 25, "cliffhanger_score": 25, "standalone_score": 20,
           "emotion_score": 15, "narrative_score": 10, "curiosity_score": 5}

PROMPT = """You are a promo editor for a microdrama app. Below is one episode as numbered events.
Pick 1 to 3 promo clips of {min_s}-{max_s} seconds that would make a viewer who has never
seen the show stop scrolling, and then NEED to watch the next episode.
Return fewer candidates rather than weak ones.

Rules:
- hook_event is the event within the FIRST 3 SECONDS of the clip that makes a scrolling
  viewer stop. It is NOT the most dramatic moment of the clip overall. It can be a line of
  dialogue or a visual action (a slap, a fall, a reveal). Slow establishing or walking
  shots are NOT hooks.
- Choose start_event so that the hook lands in the first 3 seconds. A short line of setup
  before the hook is fine if it makes the hook land.
- cliffhanger_event is the moment that leaves an urgent unanswered question.
  End the clip right after it.
- Every clip MUST be {min_s}-{max_s} seconds long.
  duration = end time of end_event minus start time of start_event.
  Compute it for every clip and return it as estimated_duration (a number of seconds).
  Only return clips whose estimated_duration is {min_s}-{max_s}.
- A promo is NOT the whole episode. Do not extend a clip to the end of the episode just to
  include its final cliffhanger.
- Pick a compact sequence: hook -> minimum context -> escalation -> cliffhanger.
  Hook in the first 3 seconds, cut right after the cliffhanger.
- If a scene can't make a compelling {min_s}-{max_s} second promo, return fewer clips rather
  than stretching it.
- Refer to events only by their [number]. Do NOT write timestamps anywhere
  (estimated_duration is a number of seconds, not a timestamp).

Return only JSON: a list of objects with these keys:
start_event, end_event, hook_event, cliffhanger_event (ints), estimated_duration (number),
hook, cliffhanger, standalone_context (what a new viewer needs to know), why_it_works (one line),
and these scores (ints 1-10, be strict):
hook_score, cliffhanger_score, standalone_score (standalone clarity),
emotion_score (emotional intensity), narrative_score (narrative clarity),
curiosity_score (social curiosity: would people comment or share?).

EPISODE LOGLINE: {logline}

EVENTS:
{events}"""

# Follow-up message (same conversation) for clips Python rejected only for their duration.
REPAIR_PROMPT = """These clips are outside {min_s}-{max_s} seconds, measured from the real timestamps:
{problems}
Give replacement clips for them, each {min_s}-{max_s} seconds long.
duration = end time of end_event minus start time of start_event.
Keep the hook in the first 3 seconds and cut right after the cliffhanger.
Return only JSON in the same schema (a list). Return an empty list if no clip of
{min_s}-{max_s} seconds works."""


def clock(seconds):
    """62.5 -> '01:02.50' (same style as the scripts)."""
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def describe(i, event):
    """One event as a numbered line, e.g. [3] 00:05.00-00:07.50 DIALOGUE ANNA (Angry): "Get out!" """
    who = f" {event['speaker']} ({event['emotion']})" if event["speaker"] else ""
    return f"[{i}] {clock(event['start'])}-{clock(event['end'])} {event['type']}{who}: \"{event['text']}\""


class QuotaExhausted(Exception):
    """The model's quota is used up (429). Retrying won't help, so the whole run stops."""


def is_quota_error(error):
    """True for a 429 quota error from Gemini."""
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "exceeded your current quota" in text


API_CALLS = 0  # every request sent to Gemini (including retries), for the end-of-run report


def build_prompt(episode):
    """The first message for one episode: rules + logline + numbered events."""
    events = "\n".join(describe(i, e) for i, e in enumerate(episode["events"]))
    return PROMPT.format(min_s=MIN_SECONDS, max_s=MAX_SECONDS,
                         logline=episode["logline"], events=events)


def ask_model(client, contents, tries=MAX_TRIES):
    """Send a prompt (or a whole conversation) to Gemini and return the raw text reply.
    Retries other errors (e.g. 503 high demand); a quota error stops at once."""
    global API_CALLS
    config = types.GenerateContentConfig(response_mime_type="application/json")
    for attempt in range(1, tries + 1):
        try:
            API_CALLS += 1
            reply = client.models.generate_content(model=MODEL, contents=contents, config=config)
            json.loads(reply.text)  # make sure it is valid JSON, else retry
            return reply.text
        except Exception as error:
            print(f"  attempt {attempt} failed: {str(error)[:120]}")
            if is_quota_error(error):
                raise QuotaExhausted(str(error)) from error  # no retry: it would waste calls
            if attempt == tries:
                raise
            time.sleep(PAUSE_SECONDS * attempt)  # wait a bit longer each time


def weighted_score(result):
    """Turn the six 1-10 model scores into one 0-100 score (missing or bad scores count as 0)."""
    total = 0
    for key, weight in WEIGHTS.items():
        score = result.get(key)
        if isinstance(score, int) and 1 <= score <= 10:
            total += score / 10 * weight
    return round(total, 1)


def valid(n, events):
    """True if n is a real event number."""
    return isinstance(n, int) and 0 <= n < len(events)


def span(events, first, last):
    """Real (start, end, duration) of the events first..last."""
    clip = events[first:last + 1]
    start, end = min(e["start"] for e in clip), max(e["end"] for e in clip)
    return start, end, round(end - start, 2)


def check(candidate, events):
    """Add real timestamps, add warnings, and set the status. Never moves the clip boundaries."""
    r = dict(candidate, warnings=[], status="ACCEPTED", reason=None)
    r["weighted_score"] = weighted_score(r)
    start, end = r.get("start_event"), r.get("end_event")
    hook, cliff = r.get("hook_event"), r.get("cliffhanger_event")

    # 1. The clip boundaries themselves must be real and in order.
    if not (valid(start, events) and valid(end, events)) or end < start:
        r["status"], r["reason"] = "REJECTED", "start/end events missing, invalid or out of order"
        return r

    # 2. Real timestamps of the model's boundaries (we never move them).
    r["start_time"], r["end_time"], r["duration"] = span(events, start, end)

    # 3. Hook: must be a real event inside the clip, starting within 3s of the clip start.
    if not (valid(hook, events) and start <= hook <= end):
        r["status"], r["reason"] = "NEEDS_REVIEW", "hook_event missing or not inside the clip"
    else:
        r["hook_offset"] = round(events[hook]["start"] - r["start_time"], 2)
        if r["hook_offset"] > HOOK_WINDOW:
            r["status"], r["reason"] = "NEEDS_REVIEW", f"hook starts {r['hook_offset']}s into clip"

    # 4. Duration rule: outside 20-60s is rejected (this overrides NEEDS_REVIEW).
    if not MIN_SECONDS <= r["duration"] <= MAX_SECONDS:
        r["status"], r["reason"] = "REJECTED", f"duration {r['duration']}s is outside {MIN_SECONDS}-{MAX_SECONDS}s"

    # 5. Cliffhanger (structure only): inside the clip and ending at most 5s before the clip end.
    if not (valid(cliff, events) and start <= cliff <= end):
        r["warnings"].append("cliffhanger_event missing or not inside the clip")
    else:
        r["cliffhanger_remaining"] = round(r["end_time"] - events[cliff]["end"], 2)
        if r["cliffhanger_remaining"] > CLIFF_WINDOW:
            r["warnings"].append(f"clip runs {r['cliffhanger_remaining']}s after the cliffhanger ends")

    # 6. Silent ending: the last 3 seconds hold only CAMERA events.
    ending = [e for e in events[start:end + 1] if e["end"] > r["end_time"] - SILENT_WINDOW]
    if all(e["type"] == "CAMERA" for e in ending):
        r["warnings"].append("last 3s are CAMERA only (possible silent thinking shot)")
    return r


def reject_overlaps(results):
    """If two clips share more than half of the shorter one, reject the lower weighted score."""
    alive = [r for r in results if r["status"] != "REJECTED"]
    kept = []
    for r in sorted(alive, key=lambda r: r["weighted_score"], reverse=True):  # best first
        for k in kept:
            shared = min(r["end_time"], k["end_time"]) - max(r["start_time"], k["start_time"])
            if shared > MAX_OVERLAP * min(r["duration"], k["duration"]):
                r["status"], r["reason"] = "REJECTED", f"overlaps clip {k['clip']}"
                break
        if r["status"] != "REJECTED":
            kept.append(r)


def show(r, events):
    """Print one checked candidate for a human to read."""
    source = "repair" if r.get("repaired") else "first pass"
    print(f"\n  clip {r['clip']} ({source}): {r['status']}" + (f" ({r['reason']})" if r["reason"] else ""))
    if "start_time" in r:
        print(f"    clip: events {r['start_event']}-{r['end_event']}  "
              f"{clock(r['start_time'])} -> {clock(r['end_time'])} ({r['duration']}s real, "
              f"model estimated {r.get('estimated_duration')}s)")
        print(f"    hook_offset {r.get('hook_offset')}s, cliffhanger_remaining {r.get('cliffhanger_remaining')}s")
        for label in ("hook", "cliffhanger"):
            n = r.get(f"{label}_event")
            print(f"    {label + ':':<13}{describe(n, events[n]) if valid(n, events) else n}")
    for warning in r["warnings"]:
        print(f"    WARNING: {warning}")
    print("    scores: " + ", ".join(f"{k.replace('_score', '')} {r.get(k)}" for k in WEIGHTS)
          + f"  -> weighted {r['weighted_score']}/100")


def number_episodes(parsed):
    """{episode number: episode}. Uses the number in 'EPISODE NUMBER'; falls back to file order."""
    episodes = {}
    for position, (episode, _) in enumerate(parsed, 1):
        digits = re.findall(r"\d+", episode["number"])
        n = int(digits[0]) if digits else position
        episodes[position if n in episodes else n] = episode  # avoid clashes on odd shows
    return episodes


def repair(client, n, prompt, raw, results, events, out_dir):
    """One follow-up call for clips rejected only for their duration. Returns the new clips."""
    too_long_or_short = [(i, r) for i, r in enumerate(results, 1)
                         if r["status"] == "REJECTED" and r["reason"].startswith("duration")]
    if not too_long_or_short:
        return []
    problems = "\n".join(f"clip {i}: events {r['start_event']}-{r['end_event']} is {r['duration']}s, "
                         f"must be {MIN_SECONDS}-{MAX_SECONDS}s" for i, r in too_long_or_short)
    print(f"  repair: asking for replacements of {len(too_long_or_short)} clip(s)")
    conversation = [  # the same conversation: our prompt, the model's reply, then our follow-up
        types.Content(role="user", parts=[types.Part(text=prompt)]),
        types.Content(role="model", parts=[types.Part(text=raw)]),
        types.Content(role="user", parts=[types.Part(text=REPAIR_PROMPT.format(
            min_s=MIN_SECONDS, max_s=MAX_SECONDS, problems=problems))]),
    ]
    try:
        repair_raw = ask_model(client, conversation, tries=1)  # at most 1 extra call per episode
    except QuotaExhausted:
        raise  # quota is gone: stop the whole run
    except Exception as error:  # repair failed: keep the rejected clips and move on
        print(f"  repair failed: {str(error)[:120]}")
        return []
    Path(out_dir, "raw", f"episode_{n:02d}_repair.json").write_text(repair_raw, encoding="utf-8")
    return [dict(check(c, events), repaired=True) for c in json.loads(repair_raw)]


def process(client, n, episode, out_dir="outputs"):
    """Ask the model about one episode, check its answer, repair durations once, save both files."""
    print(f"\n=== Episode {n} ({len(episode['events'])} events) ===")
    prompt = build_prompt(episode)
    raw = ask_model(client, prompt)
    Path(out_dir, "raw", f"episode_{n:02d}.json").write_text(raw, encoding="utf-8")
    results = [dict(check(c, episode["events"]), repaired=False) for c in json.loads(raw)]
    results += repair(client, n, prompt, raw, results, episode["events"], out_dir)  # rejected ones stay
    for number, result in enumerate(results, 1):
        result["clip"] = number
    reject_overlaps(results)
    Path(out_dir, "checked", f"episode_{n:02d}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for result in results:
        show(result, episode["events"])


def run_all(client, episodes, wanted, out_dir="outputs", on_progress=None):
    """Process each wanted episode with checkpointing; save and return the run log.
    on_progress(done, total, n, status) is called after each episode (used by the app)."""
    Path(out_dir, "raw").mkdir(parents=True, exist_ok=True)
    Path(out_dir, "checked").mkdir(parents=True, exist_ok=True)
    print(f"Model: {MODEL}")
    log = {"model": MODEL, "date": datetime.now().isoformat(timespec="seconds"), "episodes": {}}
    for done, n in enumerate(wanted, 1):
        if Path(out_dir, "checked", f"episode_{n:02d}.json").exists():  # checkpoint: already done
            log["episodes"][n] = "skipped (already done)"
        else:
            try:
                process(client, n, episodes[n], out_dir)
                log["episodes"][n] = "ok"
            except QuotaExhausted as error:  # quota used up: log it and stop the whole run
                print(f"  Episode {n}: QUOTA EXHAUSTED for {MODEL}. Stopping the run.")
                log["episodes"][n] = f"failed: quota exhausted: {error}"
                log["stopped_early"] = f"quota exhausted at episode {n}"
                break
            except Exception as error:  # log the failure and carry on with the next episode
                print(f"  Episode {n} FAILED: {str(error)[:150]}")
                log["episodes"][n] = f"failed: {error}"
            time.sleep(PAUSE_SECONDS)  # be gentle with the free-tier rate limit
        if on_progress:
            on_progress(done, len(wanted), n, log["episodes"][n])

    log["api_calls"] = API_CALLS
    Path(out_dir, "run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    failed = [n for n, status in log["episodes"].items() if status.startswith("failed")]
    print(f"\nProcessed {len(log['episodes']) - len(failed)}/{len(wanted)} episodes. Failed: {failed or 'none'}")
    print(f"API calls this run (including retries and repairs): {API_CALLS}")
    if "stopped_early" in log:
        print(f"STOPPED EARLY: {log['stopped_early']}. Episodes after it were not run.")
    return log


def summary(out_dir="outputs"):
    """Count statuses, warnings and reasons over every checked file."""
    statuses, problems, repaired = Counter(), Counter(), Counter()
    files = sorted(Path(out_dir, "checked").glob("episode_*.json"))
    for path in files:
        for r in json.loads(path.read_text(encoding="utf-8")):
            statuses[r["status"]] += 1
            if r.get("repaired"):
                repaired[r["status"]] += 1
            # Replace numbers like " 4.2s" with " Xs" so similar messages are counted together
            # (only after a space, so the fixed limit "20-60s" stays readable).
            for text in r["warnings"] + ([r["reason"]] if r["reason"] else []):
                problems[re.sub(r"(?<=\s)[\d.]+s\b", "Xs", re.sub(r"clip \d+", "clip X", text))] += 1
    print(f"\nEpisodes with results: {len(files)}")
    print(f"Total candidates: {sum(statuses.values())}")
    for status, count in statuses.most_common():
        print(f"  {status}: {count}")
    print("Most common warnings / reasons:")
    for text, count in problems.most_common(8):
        print(f"  {count:>3}  {text}")
    fixed = repaired["ACCEPTED"] + repaired["NEEDS_REVIEW"]
    print(f"Clips from the repair call: {sum(repaired.values())} "
          f"({fixed} fixed = ACCEPTED or NEEDS_REVIEW, {repaired['REJECTED']} still rejected)")


if __name__ == "__main__":
    # Usage: python promo_finder.py 1 25 50   or   python promo_finder.py all
    episodes = number_episodes(load_all())
    wanted = sorted(episodes) if sys.argv[1:] == ["all"] else [int(n) for n in sys.argv[1:]]
    run_all(genai.Client(), episodes, wanted)  # genai.Client() reads GEMINI_API_KEY from the environment
    summary()
