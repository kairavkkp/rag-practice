import json
from pathlib import Path

NOTES_DIR = Path("data/notes")
OUT = Path("data/index/chunks.jsonl")
MAX_WORDS = 250  # adjust after looking at your stats


def load_note(path):
    """Return {"file", "date", "title", "body"}.
    The file is: '---', header lines like 'date: 2024-03-18', '---', then the body.
    Hint: text.split("---", 2) gives you ['', header, body]."""


def split_blocks(body):
    """Split the body into blocks. A heading line ('#') starts a new block,
    a blank line ends one. Consecutive list lines ('- ') stay in the same block."""


def chunk_note(note):
    """Pack blocks into chunks of at most MAX_WORDS words, in order.
    Never split a block. A block longer than MAX_WORDS becomes its own chunk."""


def main():
    # for each note: load -> split into blocks -> chunk
    # write one JSON line per chunk to OUT:
    # {"chunk_id": "2024-03-18.md#0", "file": "2024-03-18.md",
    #  "date": "2024-03-18", "position": 0, "text": "..."}
    ...


if __name__ == "__main__":
    main()
