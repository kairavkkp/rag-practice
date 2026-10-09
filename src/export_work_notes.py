#!/usr/bin/env python3
"""
Export Apple Notes to Markdown files with date front matter, ready for a RAG project.

Exports every folder listed in FOLDERS below, skipping handwritten notes and any
note whose title is in EXCLUDE_TITLES.

Each note's date comes from its title if the title is exactly YYYY-MM-DD,
otherwise from the note's creation date. Files are named:
  2024-03-18.md                          title was the date
  2024-10-30_efs-kinesis-action-items.md other titles: creation date + title

Output:
  data/raw/<Folder>.jsonl        backup, one line per note read from Notes
  data/notes/<Folder>/*.md       one Markdown file per note

Usage:
  python3 export_notes.py            # export all folders (resumes where it left off)
  python3 export_notes.py --rebuild  # rebuild the .md files from the backups, no Notes app
  python3 export_notes.py --fresh    # forget earlier progress and read everything again

Each batch is saved as soon as it is read. If a run is interrupted, the next run
skips notes that are already saved and carries on.

Only uses the Python standard library. If tqdm is installed (pip install tqdm)
it shows a tqdm progress bar, otherwise a simple counter.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

# Notes folders to export (exact names, case-sensitive)
FOLDERS = ["Work Notes", "Lighthouz AI"]

# Notes to leave out, by exact title (e.g. notes containing keys or personal notes)
EXCLUDE_TITLES = [
    "Dashydash OpenAI keys",
]

# ---------------------------------------------------------------------------
# 1. Talk to the Notes app (AppleScript, run through osascript)
# ---------------------------------------------------------------------------
# AppleScript is used because JavaScript automation (JXA) is unreliable on recent macOS.
# Fields are separated with ASCII control characters that never appear in notes.
US, RS = "\x1f", "\x1e"

# Step A: list every note's id + title in the folder.
AS_LIST = """
on run argv
  set folderName to item 1 of argv
  set US to character id 31
  set RS to character id 30
  tell application "Notes"
    set f to folder folderName
    set theIds to id of every note of f
    set theNames to name of every note of f
  end tell
  set out to ""
  repeat with i from 1 to count of theIds
    set out to out & (item i of theIds) & US & (item i of theNames) & RS
  end repeat
  return out
end run
"""

# Step B: read the full contents of a small batch of notes by id.
AS_BATCH = """
on isoDate(d)
  return (d as «class isot» as string)
end isoDate

on run argv
  set US to character id 31
  set RS to character id 30
  set out to ""
  tell application "Notes"
    repeat with theId in argv
      set theId to theId as text
      try
        set n to note id theId
        set nTitle to name of n
        set nBody to body of n
        set nCreated to my isoDate(creation date of n)
        set nModified to my isoDate(modification date of n)
        set nAtt to (count of attachments of n) as text
        set out to out & theId & US & "OK" & US & nTitle & US & nCreated & US & nModified & US & nAtt & US & nBody & RS
      on error errMsg
        set out to out & theId & US & "ERROR" & US & errMsg & RS
      end try
    end repeat
  end tell
  return out
end run
"""

BATCH_SIZE = 10

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class SimpleBar:
    """Fallback progress display if tqdm isn't installed."""

    def __init__(self, total):
        self.total, self.done = total, 0
        self.update(0)

    def update(self, n):
        self.done += n
        print(f"\rRead {self.done}/{self.total} notes", end="", flush=True)

    def close(self):
        print()


def osa(script, args):
    r = subprocess.run(
        ["osascript", "-e", script, *args], capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return [rec for rec in r.stdout.rstrip("\n").split(RS) if rec.strip()]


def parse_batch(records):
    out = []
    for rec in records:
        parts = rec.split(US, 6)
        if len(parts) >= 2 and parts[1] == "OK" and len(parts) == 7:
            nid, _, title, created, modified, att, body = parts
            out.append(
                {
                    "id": nid,
                    "title": title,
                    "created": created,
                    "modified": modified,
                    "attachments": int(att or 0),
                    "html": body,
                }
            )
        else:
            out.append({"id": parts[0], "error": parts[2] if len(parts) > 2 else rec})
    return out


def open_notes():
    # Make sure Notes is open (osascript fails with error -600 if it isn't running)
    print("Opening Notes...")
    subprocess.run(["open", "-g", "-a", "Notes"])
    time.sleep(5)


def list_folder(folder):
    print(f'\nLooking for the "{folder}" folder...')
    try:
        records = osa(AS_LIST, [folder])
    except RuntimeError as e:
        sys.exit(
            f"osascript failed:\n{e}\n\nIf it says it can't get the folder, check the exact name "
            "(case-sensitive). If it says 'Not authorized', allow your terminal under "
            "System Settings > Privacy & Security > Automation > Notes."
        )
    pairs = [rec.split(US, 1) for rec in records]
    return [p[0] for p in pairs], [p[1] if len(p) > 1 else "" for p in pairs]


# ---------------------------------------------------------------------------
# 2. Convert the note HTML into plain Markdown
# ---------------------------------------------------------------------------
class NoteHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
        self.list_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in ("ul", "ol"):
            self.list_depth += 1
        elif tag == "li":
            self.out.append("\n" + "  " * max(self.list_depth - 1, 0) + "- ")
        elif tag == "br":
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag == "img":
            self.out.append("[image]")

    def handle_endtag(self, tag):
        if tag == "li" and self.out and self.out[-1] == "\n":
            self.out.pop()  # Notes ends many items with <br>; it isn't a blank line
        elif tag in ("h1", "h2", "h3", "div", "p", "tr"):
            self.out.append("\n")
        elif tag in ("ul", "ol"):
            self.list_depth -= 1
            self.out.append("\n")

    def handle_data(self, data):
        # Whitespace with a newline is HTML source formatting between tags
        # (e.g. "</li>\n<li>"), not note content; Notes marks real blank lines with <br>
        if not data.strip() and "\n" in data:
            return
        self.out.append(data)


def html_to_markdown(html, title):
    parser = NoteHTML()
    parser.feed(html or "")
    text = "".join(parser.out).replace("\xa0", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    # Apple puts the title as the first line of the body; drop it (we store it separately).
    # Sometimes it is split over several <h1> tags: "# 202", "# 3-0", "# 2-1", "# 7".
    want = title.replace(" ", "")
    got, i = "", 0
    while i < len(lines) and got != want:
        piece = lines[i].strip().lstrip("#").replace(" ", "")
        if piece and not want.startswith(got + piece):
            break
        got += piece
        i += 1
    if want and got == want:
        while i < len(lines) and lines[i].strip() in ("", "#"):
            i += 1
        lines = lines[i:]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Filters: date title and handwriting
# ---------------------------------------------------------------------------
TITLE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MIN_TYPED_CHARS = 30


def title_date(title):
    """Return the date if the title is exactly YYYY-MM-DD and a real date, else None."""
    t = title.strip()
    if not TITLE_RE.match(t):
        return None
    try:
        return datetime.strptime(t, "%Y-%m-%d").date()
    except ValueError:  # e.g. 2024-13-45
        return None


def is_handwritten(note, body):
    """Notes doesn't expose a 'handwritten' flag to scripts, so this is a heuristic:
    the note has a drawing/attachment/image but almost no typed text."""
    has_visual = note.get("attachments", 0) > 0 or "<img" in (note.get("html") or "")
    typed = body.replace("[image]", "").strip()
    return has_visual and len(typed) < MIN_TYPED_CHARS


# ---------------------------------------------------------------------------
# 4. Write one Markdown file per note, as soon as it is read
# ---------------------------------------------------------------------------
class Writer:
    def __init__(self, notes_dir):
        self.notes_dir = notes_dir
        self.used_names = {p.name for p in notes_dir.glob("*.md")}
        self.written = 0
        self.restored = 0
        self.skipped = {
            "excluded by title": [],
            "looks handwritten": [],
            "empty": [],
            "locked or unreadable": [],
        }
        self.mismatches = []

    def write(self, n):
        if n["title"].strip() in EXCLUDE_TITLES:
            self.skipped["excluded by title"].append(n["title"])
            return
        body = html_to_markdown(n["html"], n["title"])
        if is_handwritten(n, body):
            self.skipped["looks handwritten"].append(n["title"])
            return
        if not body:
            self.skipped["empty"].append(n["title"])
            return

        created = datetime.fromisoformat(n["created"].replace("Z", "+00:00")).date()
        note_date = title_date(n["title"])
        if note_date is not None:
            date_source = "title"
            base = note_date.isoformat()
            if abs((note_date - created).days) > 7:
                self.mismatches.append((n["title"], created))
        else:
            date_source = "created"
            note_date = created
            slug = (
                re.sub(r"[^a-z0-9]+", "-", n["title"].lower())
                .strip("-")[:50]
                .rstrip("-")
            )
            base = f"{note_date.isoformat()}_{slug or 'note'}"
            # The title carries meaning here, so keep it in the text for search
            body = f"# {n['title'].strip()}\n\n{body}"

        name = f"{base}.md"
        i = 2
        while name in self.used_names:
            name = f"{base}_{i}.md"
            i += 1
        self.used_names.add(name)

        front = "\n".join(
            [
                "---",
                f"date: {note_date.isoformat()}",
                f"title: {json.dumps(n['title'], ensure_ascii=False)}",
                f"folder: {json.dumps(n['folder'], ensure_ascii=False)}",
                f"created: {n['created']}",
                f"modified: {n['modified']}",
                f"date_source: {date_source}",
                "---",
            ]
        )
        (self.notes_dir / name).write_text(f"{front}\n\n{body}\n", encoding="utf-8")
        self.written += 1

    def summary(self, folder, found_line=""):
        new = self.written - self.restored
        print(f"\n=== {folder} ===")
        if found_line:
            print(found_line)
        print(
            f"{self.written} notes in {self.notes_dir}/ "
            f"({self.restored} from earlier runs, {new} new in this run)"
        )
        for reason, titles in self.skipped.items():
            if titles:
                print(f"\nSkipped {len(titles)} ({reason}):")
                for t in titles[:30]:
                    print(f"  - {t}")
                if len(titles) > 30:
                    print(f"  ... and {len(titles) - 30} more")
        if self.mismatches:
            print(
                f"\n{len(self.mismatches)} notes where the title date is >7 days from the creation "
                "date (fine if you wrote them later):"
            )
            for t, c in self.mismatches[:20]:
                print(f"  - {t} (created {c})")


def read_raw(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):  # older single-JSON export
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def clear_notes(notes_dir):
    for old in notes_dir.glob("*.md"):
        old.unlink()


def safe_name(folder):
    return folder.replace("/", "-").strip()


def migrate_old_layout(out):
    """Earlier versions wrote data/raw/notes_raw.jsonl and data/notes/*.md (flat).
    Split that backup per folder so nothing has to be read from Notes again."""
    legacy = out / "raw" / "notes_raw.jsonl"
    if not legacy.exists():
        return
    by_folder = {}
    for n in read_raw(legacy):
        by_folder.setdefault(n.get("folder", "Work Notes"), []).append(n)
    for folder, notes in by_folder.items():
        with (out / "raw" / f"{safe_name(folder)}.jsonl").open(
            "a", encoding="utf-8"
        ) as f:
            for n in notes:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
    legacy.rename(legacy.with_name("notes_raw.jsonl.migrated"))
    for old in (out / "notes").glob("*.md"):
        old.unlink()
    print(
        f"Moved {sum(map(len, by_folder.values()))} saved notes into the per-folder layout."
    )


def export_folder(folder, out, rebuild_only):
    notes_dir = out / "notes" / safe_name(folder)
    raw_path = out / "raw" / f"{safe_name(folder)}.jsonl"
    notes_dir.mkdir(parents=True, exist_ok=True)

    # Rewrite .md files for everything already saved (fast, keeps files in sync with the backup)
    clear_notes(notes_dir)
    writer = Writer(notes_dir)
    saved = read_raw(raw_path) if raw_path.exists() else []
    for n in saved:
        writer.write(n)
    writer.restored = writer.written
    if rebuild_only:
        writer.summary(folder)
        return

    done = {n["id"] for n in saved}
    ids, titles = list_folder(folder)
    id_to_title = dict(zip(ids, titles))
    wanted = [i for i, t in zip(ids, titles) if t.strip() not in EXCLUDE_TITLES]
    writer.skipped["excluded by title"] = [
        t for t in titles if t.strip() in EXCLUDE_TITLES
    ]
    todo = [i for i in wanted if i not in done]
    print(
        f"Found {len(ids)} notes; {len(ids) - len(wanted)} excluded by title; "
        f"{len(wanted) - len(todo)} already saved. Reading {len(todo)} now..."
    )

    bar = tqdm(total=len(todo), unit="note") if tqdm else SimpleBar(len(todo))
    with raw_path.open("a", encoding="utf-8") as raw_f:
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start : start + BATCH_SIZE]
            try:
                results = parse_batch(osa(AS_BATCH, batch))
            except RuntimeError as e:
                results = [{"id": i, "error": str(e)} for i in batch]
            # Stop early only if nothing has ever worked (a real problem, not just a few locked notes)
            if (
                start == 0
                and not saved
                and len(results) >= 3
                and all("error" in r for r in results)
            ):
                bar.close()
                sys.exit(
                    f"\nEvery note in the first batch failed, so stopping early. First error:\n"
                    f"{results[0]['error']}"
                )
            for r in results:
                if "error" in r:  # not saved, so the next run retries it
                    writer.skipped["locked or unreadable"].append(
                        id_to_title.get(r["id"], r["id"])
                    )
                    continue
                r["folder"] = folder
                raw_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                writer.write(r)
            raw_f.flush()
            bar.update(len(batch))
    bar.close()
    writer.summary(folder, f"Found {len(ids)} notes in the folder.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild the .md files from the saved backups without using Notes",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="delete earlier progress and read every folder from Notes again",
    )
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "notes").mkdir(parents=True, exist_ok=True)
    migrate_old_layout(out)

    if args.fresh:
        for folder in FOLDERS:
            (out / "raw" / f"{safe_name(folder)}.jsonl").unlink(missing_ok=True)
    if not args.rebuild:
        open_notes()
    for folder in FOLDERS:
        export_folder(folder, out, args.rebuild)


if __name__ == "__main__":
    main()
