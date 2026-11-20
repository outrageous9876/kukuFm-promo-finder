# Microdrama promo finder

Reads microdrama screenplays and finds 20–60 second promo clips that hook a viewer in the
first 3 seconds, end on a cliffhanger and make sense to someone who has never seen the show.

How it works:
1. `parser.py` turns each screenplay `.txt` into timed events (camera, action, sound, dialogue).
2. `promo_finder.py` sends each episode to Gemini as numbered events. The model picks clips
   **by event number only**. It never writes timestamps, so it can't invent them.
3. Plain Python then looks up the real timestamps and checks the clips: duration 20–60s, hook
   within 3s of the start, cliffhanger near the end, silent endings, and overlaps between clips.
   It also works out a weighted score. Each clip ends up `ACCEPTED`, `NEEDS_REVIEW` or `REJECTED`.
   Python checks structure only. Whether a moment is actually gripping is still the model's
   judgment (and the weighted score is built from the model's own scores).
   If clips are rejected only for their length, one follow-up call asks the model for
   replacements, and Python checks those the same way.
4. `rank.py` ranks the usable clips with a tournament. A separate "skeptical TikTok editor"
   judge sees only the script lines (opening first) and ranks groups of about 8, then the
   group winners are ranked again in a final round. This is used instead of the self-scores,
   which barely differ between clips.

## Setup

```
pip install -r requirements.txt
copy .env.example .env      (macOS/Linux: cp .env.example .env)
```

Put your key in `.env`:
```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
```

**Swap the model:** change the `GEMINI_MODEL` line in `.env`. Nothing else needs to change.

## Run

Whole pipeline for a show (a ZIP or a folder of `.txt` screenplays):
```
python run.py path/to/show.zip
```
It unzips to a temporary folder, parses, finds promos for every episode and ranks them.
If you run it again it skips episodes that are already done, so a run stopped by an error or a
rate limit picks up where it left off. A quota error (429) stops the run at once instead of
wasting calls. To rank again on its own: `python rank.py outputs/<show name>`.

The app (browse existing results without API calls, or upload a ZIP and run it):
```
streamlit run app.py
```
If `.env` has no key, the app asks for it in a password box. The key is never saved.

## Output files

Each show gets its own folder, `outputs/<show name>/`:

| File | What it is |
|---|---|
| `raw/episode_XX.json` | The exact reply from the model |
| `raw/episode_XX_repair.json` | The exact reply to the length-repair follow-up (only if one was needed) |
| `raw/ranking_*.json` | The exact replies from the ranking judge |
| `checked/episode_XX.json` | Candidates with real timestamps, checks, status and score |
| `ranked.json` / `final_rankings.csv` | The final ranking, with the judge's reason for each clip |
| `run_log.json` | Model name, date, API calls and status of each episode (ok / skipped / failed) |
| `episodes.json` | The parsed scripts, used by the app to show script lines |

## Results

On the 50-episode test show, with `gemini-3.5-flash-lite`:

- **50 / 50 episodes** processed, **119 candidates**: 69 accepted, 13 need review, 37 rejected.
- **82 clips ranked** (accepted + needs review).
- **Precision@10 = 0.8** from a manual review of the top 10: 8 ready to post, 2 maybe, 0 bad.
- About **1.5 API calls per episode** (including the length repair), and about **13 minutes**
  for the full show.

## Known limitations

- **Endings sometimes cut a line early.** A clip can stop one line before its real cliffhanger.
  Python checks timing, not what the next line says.
- **The ranking order varies between shuffles.** With a different group shuffle (seed 7 vs 42),
  only 5 of the top 10 stayed in the top 10. Treat the order as a shortlist, not a final answer.
- **Screenplay ≠ final footage.** Timings and content come from the script; the edited episode
  may differ.
- **Single-episode, continuous clips only.** A promo is one unbroken stretch of one episode.
  It can't cut separate moments together or combine episodes.

## Data

The screenplays are confidential and are **not** in this repository. `data/`, `outputs/`,
`*.zip` and `.env` are all in `.gitignore`.
