import json
from pathlib import Path

NOTES_DIR = Path("data/notes")
OUT = Path("data/index/chunks.jsonl")
MAX_WORDS = 250  # adjust after looking at your stats


def load_note(path):
    """Return {"file", "folder", "date", "title", "body"}.
    The file is: '---', header lines like 'date: 2024-03-18', '---', then the body.
    "file" is "<folder>/<filename>", since filenames repeat across folders."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Split on whole '---' lines, so a title containing '---' can't break it
    if not text.startswith("---\n") or "\n---\n" not in text:
        raise ValueError(
            f"File {path} does not have the expected format with '---' separators."
        )
    header, body = text[len("---\n") :].split("\n---\n", 1)

    meta = {}
    for line in header.splitlines():
        key, _, value = line.partition(":")  # values like 'created' contain colons
        meta[key.strip()] = value.strip()
    folder = json.loads(meta["folder"])  # title and folder are JSON-quoted

    return {
        "file": f"{folder}/{path.name}",
        "folder": folder,
        "date": meta["date"],
        "title": json.loads(meta["title"]),
        "body": body.strip(),
    }


def split_blocks(body):
    """Split the body into blocks. A heading line ('#') starts a new block,
    a blank line ends one. Consecutive list lines ('- ') stay in the same block."""


def chunk_note(note):
    """Pack blocks into chunks of at most MAX_WORDS words, in order.
    Never split a block. A block longer than MAX_WORDS becomes its own chunk."""


def main():
    # for each note: load -> split into blocks -> chunk
    # write one JSON line per chunk to OUT:
    # {"chunk_id": "Work Notes/2024-03-18.md#0", "file": "Work Notes/2024-03-18.md",
    #  "folder": "Work Notes", "title": "2024-03-18", "date": "2024-03-18",
    #  "position": 0, "text": "..."}
    ...


if __name__ == "__main__":
    main()
