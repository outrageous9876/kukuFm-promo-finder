"""Step 4: rank promo candidates with a tournament judged by Gemini.   python rank.py outputs/"""
import csv
import json
import math
import random
import sys
import types
from collections import Counter
from pathlib import Path

from google import genai

import promo_finder as pf  # reuses ask_model (same model, retries and quota stop) and clock
from parser import load_all

GROUP_SIZE = 8      # about this many clips per group
FINALISTS = 2       # top clips per group that go to the final round
SEED = 42           # fixed shuffle, so the same input gives the same groups
OPENING_SECONDS = 3.0  # events starting this soon after the clip start are the "opening"

JUDGE_PROMPT = """You are a skeptical social media editor choosing which microdrama promo clip to
post first on TikTok/Reels for a US audience. The viewers have never heard of this show.

Below are {n} clips. Each shows only what a viewer would hear and see, with the seconds since
the clip started: first its OPENING (everything in the first 3 seconds, including camera shots),
then the REST OF CLIP (dialogue, actions, sound effects).

Most viewers decide in the first 3 seconds. A clip whose OPENING is quiet, slow or only setup
must rank below clips with a strong opening, no matter how good its ending is.
Judge the opening first.

Rank ALL the clips from best to worst. Judge each one on:
1. Does the opening (first 3 seconds) stop the scroll?
2. Does the ending make you NEED to watch the next episode?
3. Does it make sense with zero context?
Be critical: most clips are not great, and the differences matter.

Return only JSON: {{"ranking": [{{"id": "<clip id>", "reason": "<one line>"}}, ...]}}
Use every clip ID exactly once.

CLIPS:
{cards}"""


class CountingModels:
    """Wraps client.models only to add up the token usage the API reports.
    It keeps the whole client: if the client object is garbage-collected, its connection closes."""
    def __init__(self, client):
        self.client, self.tokens = client, Counter()

    def generate_content(self, **kwargs):
        reply = self.client.models.generate_content(**kwargs)
        usage = getattr(reply, "usage_metadata", None)
        if usage:
            self.tokens["input"] += usage.prompt_token_count or 0
            self.tokens["output"] += usage.candidates_token_count or 0
            self.tokens["total"] += usage.total_token_count or 0
        return reply


def load_episodes(out_dir):
    """Parsed scripts: saved by run.py as episodes.json, or (for outputs/) parsed from data/."""
    saved = Path(out_dir, "episodes.json")
    if saved.exists():
        return {int(n): ep for n, ep in json.loads(saved.read_text(encoding="utf-8")).items()}
    return pf.number_episodes(load_all("data"))


def load_candidates(out_dir):
    """All ACCEPTED and NEEDS_REVIEW candidates, each with an ID like 'E01-C1'."""
    candidates = []
    for path in sorted(Path(out_dir, "checked").glob("episode_*.json")):
        episode = int(path.stem.split("_")[1])
        for r in json.loads(path.read_text(encoding="utf-8")):
            if r["status"] != "REJECTED":
                candidates.append(dict(r, episode=episode, id=f"E{episode:02d}-C{r['clip']}"))
    return candidates


def clip_card(c, events):
    """What a stranger would see (no scores, no AI reasons, no logline).
    OPENING = every event starting in the first 3s, CAMERA included (the only visual info).
    REST OF CLIP = dialogue, ACTION and SOUND FX lines after that (CAMERA skipped to save tokens)."""
    opening, rest = ["OPENING (first 3s):"], ["REST OF CLIP:"]
    for e in events[c["start_event"]:c["end_event"] + 1]:
        offset = e["start"] - c["start_time"]
        who = e["speaker"] if e["type"] == "DIALOGUE" else e["type"]
        line = f"  {offset:5.1f}s {who}: {e['text']}"
        if offset <= OPENING_SECONDS:
            opening.append(line)
        elif e["type"] != "CAMERA":
            rest.append(line)
    return "\n".join([f"ID {c['id']} ({c['duration']}s)"] + opening + rest)


def parse_ranking(raw, ids):
    """[(id, reason), ...] if the reply ranks every ID exactly once, else None."""
    try:
        items = json.loads(raw)["ranking"]
        ranking = [(item["id"], str(item.get("reason", ""))) for item in items]
    except (KeyError, TypeError, ValueError):
        return None
    return ranking if sorted(i for i, _ in ranking) == sorted(ids) else None


def judge(client, group, cards, out_dir, name):
    """Ask the judge to rank one group. Retry once if the IDs are wrong, then fall back."""
    ids = [c["id"] for c in group]
    prompt = JUDGE_PROMPT.format(n=len(group), cards="\n\n".join(cards[i] for i in ids))
    for attempt in (1, 2):
        raw = pf.ask_model(client, prompt)
        Path(out_dir, "raw", f"ranking_{name}_try{attempt}.json").write_text(raw, encoding="utf-8")
        ranking = parse_ranking(raw, ids)
        if ranking:
            return ranking
        print(f"  {name}: judge reply did not rank every ID exactly once (try {attempt})")
    print(f"  {name}: FALLBACK to weighted score")
    best_first = sorted(group, key=lambda c: c["weighted_score"], reverse=True)
    return [(c["id"], "fallback: ranked by weighted score (judge reply invalid)") for c in best_first]


def rank_show(out_dir, client=None, seed=SEED, save_dir=None):
    """Rank every usable candidate in out_dir; save ranked.json and final_rankings.csv
    in save_dir (default: out_dir). seed changes how clips are shuffled into groups."""
    save_dir = Path(save_dir or out_dir)
    client = client or genai.Client()  # reads GEMINI_API_KEY from .env
    counting = CountingModels(client)
    client = types.SimpleNamespace(models=counting)
    calls_before = pf.API_CALLS

    candidates = load_candidates(out_dir)
    if not candidates:
        print("No ACCEPTED or NEEDS_REVIEW candidates to rank.")
        return []
    episodes = load_episodes(out_dir)
    cards = {c["id"]: clip_card(c, episodes[c["episode"]]["events"]) for c in candidates}
    by_id = {c["id"]: c for c in candidates}
    Path(save_dir, "raw").mkdir(parents=True, exist_ok=True)

    # Round 1: shuffle with a fixed seed and split into groups of about GROUP_SIZE.
    shuffled = candidates[:]
    random.Random(seed).shuffle(shuffled)
    n_groups = math.ceil(len(shuffled) / GROUP_SIZE)
    groups = [shuffled[i::n_groups] for i in range(n_groups)]  # deal like cards: even sizes
    print(f"Ranking {len(candidates)} candidates in {n_groups} group(s) with {pf.MODEL}")

    group_place, reasons, finalists = {}, {}, []
    for g, group in enumerate(groups, 1):
        ranking = judge(client, group, cards, save_dir, f"round_1_group_{g:02d}")
        for place, (cid, reason) in enumerate(ranking, 1):
            group_place[cid], reasons[cid] = place, reason
        finalists += [cid for cid, _ in ranking[:FINALISTS]]

    # Round 2: the top clips of every group, ranked the same way (not needed for one group).
    if n_groups > 1:
        final = judge(client, [by_id[i] for i in finalists], cards, save_dir, "round_2_final")
        top = [cid for cid, _ in final]
        reasons.update(dict(final))  # finalists keep the final-round reason
    else:
        top = []

    # Final order: final round first, then the rest by group place, then weighted score.
    rest = sorted((c for c in candidates if c["id"] not in top),
                  key=lambda c: (group_place[c["id"]], -c["weighted_score"]))
    order = [by_id[i] for i in top] + rest
    ranked = [dict(c, rank=place, judge_reason=reasons[c["id"]], final_round=c["id"] in top)
              for place, c in enumerate(order, 1)]

    Path(save_dir, "ranked.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    with open(Path(save_dir, "final_rankings.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "episode", "start", "end", "duration", "hook", "cliffhanger",
                         "why_it_works", "judge_reason", "status", "warnings"])
        for r in ranked:
            writer.writerow([r["rank"], r["episode"], pf.clock(r["start_time"]), pf.clock(r["end_time"]),
                             r["duration"], r.get("hook"), r.get("cliffhanger"), r.get("why_it_works"),
                             r["judge_reason"], r["status"],
                             "; ".join(([r["reason"]] if r["reason"] else []) + r["warnings"])])

    print("\nTOP 10")
    for r in ranked[:10]:
        print(f"{r['rank']:>3}. {r['id']}  {pf.clock(r['start_time'])}-{pf.clock(r['end_time'])} "
              f"({r['duration']}s) {r['status']}\n       {r['judge_reason']}")
    print(f"\nAPI calls for ranking: {pf.API_CALLS - calls_before}")
    print(f"Tokens: {dict(counting.tokens) if counting.tokens else 'not reported by the API'}")
    print(f"Saved {Path(save_dir, 'ranked.json')} and {Path(save_dir, 'final_rankings.csv')}")
    return ranked


if __name__ == "__main__":
    # Usage: python rank.py outputs/                         (seed 42, saves into outputs/)
    #        python rank.py outputs/ 7 outputs/ranking_seed7  (other seed, saves elsewhere)
    args = sys.argv[1:]
    rank_show(args[0] if args else "outputs", seed=int(args[1]) if len(args) > 1 else SEED,
              save_dir=args[2] if len(args) > 2 else None)
