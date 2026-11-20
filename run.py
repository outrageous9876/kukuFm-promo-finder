"""Whole pipeline for one show:  python run.py path/to/show.zip   (or a folder of .txt scripts)"""
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

from google import genai

from parser import load_all
from promo_finder import number_episodes, run_all, summary
from rank import rank_show


def show_folder(name, out_root="outputs"):
    """Each show gets its own results folder, e.g. outputs/my_show/."""
    return Path(out_root, re.sub(r"[^\w-]+", "_", name).strip("_"))


def run_show(source, client, name=None, out_root="outputs", on_progress=None):
    """Unzip (if needed) -> parse -> find promos -> rank. Returns the results folder."""
    source = Path(source)
    out_dir = show_folder(name or source.stem, out_root)
    with tempfile.TemporaryDirectory() as tmp:  # deleted automatically at the end
        folder = source
        if source.suffix.lower() == ".zip":
            zipfile.ZipFile(source).extractall(tmp)
            folder = tmp
        episodes = number_episodes(load_all(folder))
    if not episodes:
        raise ValueError(f"No .txt screenplays found in {source}")

    # Save the parsed scripts so the app can show the real script lines later.
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir, "episodes.json").write_text(json.dumps(episodes, ensure_ascii=False), encoding="utf-8")

    log = run_all(client, episodes, sorted(episodes), out_dir, on_progress)

    if "stopped_early" in log:  # quota is gone, so ranking would fail too
        print("Skipping ranking: the run stopped early. Run again later to finish and rank.")
    else:
        rank_show(out_dir, client)
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python run.py path/to/show.zip_or_folder")
    out_dir = run_show(sys.argv[1], genai.Client())  # reads GEMINI_API_KEY from .env
    summary(out_dir)
    print(f"\nResults saved in {out_dir}/  (checked/, raw/, run_log.json, episodes.json)")
