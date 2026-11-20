"""Step 1: turn screenplay .txt files into lists of timed events."""
import re
import sys
from pathlib import Path

# Block headers that are NOT a character speaking. Anything else is dialogue.
DIRECTION_TYPES = {"CAMERA", "ACTION", "SOUND FX"}

# **NAME (Emotion):**  or  **NAME:**
HEADER_RE = re.compile(r"^\*\*(.+?)(?:\s*\((.*)\))?:\*\*$")
# Start: (00:11.30) | End: (00:13.10)
TIME_RE = re.compile(r"^Start:\s*\(([\d:.]+)\)\s*\|\s*End:\s*\(([\d:.]+)\)$")
# Header fields like "EPISODE NUMBER: Episode 1"
FIELD_RE = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES):\s*(.*)$")
# Layout lines we expect and simply skip: SCENE 01, Scene heading:, --- SCRIPT START ---, ...
LAYOUT_RE = re.compile(r"^(SCENE \d+|Scene (heading|start|end):.*|---.*---|Actions and dialogue:)$", re.I)


def to_seconds(stamp):
    """'01:02.50' -> 62.5  (also handles '1:01:02.50' with hours)."""
    seconds = 0.0
    for part in stamp.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def parse_episode(path):
    """Return (episode dict, list of lines that could not be parsed)."""
    episode = {"file": path.name, "number": "", "logline": "", "notes": "", "events": []}
    errors = []
    current = None  # the event whose text we are currently reading

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        field = FIELD_RE.match(line)
        header = HEADER_RE.match(line)
        time = TIME_RE.match(line)

        if field:
            key = {"EPISODE NUMBER": "number", "EPISODE LOGLINE": "logline",
                   "CONTINUITY NOTES": "notes"}[field.group(1)]
            episode[key] = field.group(2)
        elif header:
            name = header.group(1).strip()
            is_dialogue = name.upper() not in DIRECTION_TYPES
            current = {
                "type": "DIALOGUE" if is_dialogue else name.upper(),
                "speaker": name if is_dialogue else None,
                "emotion": header.group(2) if is_dialogue else None,
                "start": None, "end": None, "text": "",
            }
            episode["events"].append(current)
        elif time and current and current["start"] is None:
            current["start"] = to_seconds(time.group(1))
            current["end"] = to_seconds(time.group(2))
        elif not line or LAYOUT_RE.match(line):
            continue  # blank or layout line: nothing to store
        elif line.startswith("Start:"):
            errors.append((line_no, raw))  # looked like a timestamp but didn't match
        elif current and current["start"] is not None:
            current["text"] = (current["text"] + " " + line).strip().strip('"')
        else:
            errors.append((line_no, raw))  # text with no event to belong to

    # Drop events that never got a timestamp, and report them.
    for event in episode["events"]:
        if event["start"] is None:
            errors.append((0, f"event without timestamp: {event['type']} {event['speaker'] or ''}"))
    episode["events"] = [e for e in episode["events"] if e["start"] is not None]
    episode["duration"] = max((e["end"] for e in episode["events"]), default=0.0)
    return episode, errors


def load_all(folder="data"):
    """Parse every .txt file in the folder (and its subfolders), skipping exact duplicates."""
    paths = sorted(Path(folder).rglob("*.txt"))  # rglob = also look in subfolders
    print(f"Files found: {len(paths)}")
    seen = {}  # file contents -> name of the first file that had them
    results = []
    for path in paths:
        content = path.read_bytes()
        if content in seen:
            print(f"WARNING: skipping {path.name} (exact duplicate of {seen[content]})")
            continue
        seen[content] = path.name
        results.append(parse_episode(path))
    return results


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    results = load_all(folder)
    numbers = {episode["number"] for episode, _ in results}
    print(f"Episodes parsed: {len(results)} ({len(numbers)} distinct episode numbers)\n")
    total_errors = 0
    for episode, errors in results:
        print(f"{episode['number']:<12} {len(episode['events']):>4} events  "
              f"{episode['duration']:>7.1f}s  {episode['file']}")
        for line_no, text in errors:
            print(f"    FAILED line {line_no}: {text}")
        total_errors += len(errors)
    print(f"\nTotal events: {sum(len(e['events']) for e, _ in results)}")
    print(f"Lines that failed to parse: {total_errors}")
