import json
import re
from pathlib import Path

from transformers import AutoTokenizer

NOTES_DIR = Path("data/notes")
OUT = Path("data/index/chunks.jsonl")
MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 200  # packing target, header included; tune with the eval
MODEL_MAX_TOKENS = 512  # bge-small silently truncates anything longer

# Machine noise: UUIDs, and any 40+ character run without spaces (base64, hashes, URLs,
# file names, minified JSON). Prose words are never that long.
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
LONG_ID = re.compile(r"\S{40,}")
JSON_KEY = re.compile(r'^"[^"]+"\s*:')
MARKUP_TAG = re.compile(r"</?[A-Za-z]")

_tokenizer = None


def count_tokens(text):
    """Tokens as the embedding model sees them, without [CLS]/[SEP]."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL)
    return len(_tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])


def context_header(note):
    """Prefixed to every chunk at embed time and in the prompt."""
    return f"Date: {note['date']} | Folder: {note['folder']} | Note: {note['title']}"


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


def strip_ids(line):
    return LONG_ID.sub("[id]", UUID.sub("[id]", line))


def is_noise(line):
    """True for lines that carry no meaning for an embedding: REPL/JSON/HTML dumps, and
    lines that are mostly IDs. Short IDs like INVPLAT-1899 or PPD-1991 are kept."""
    s = line.strip().lstrip("-*").strip()
    if not s:
        return False
    if s.startswith((">>>", "...", "{", "}")) or JSON_KEY.match(s):
        return True
    if len(MARKUP_TAG.findall(s)) >= 3:  # pasted HTML/XML
        return True
    cleaned = strip_ids(s)
    words = re.findall(r"[A-Za-z]{2,}", cleaned.replace("[id]", ""))
    return cleaned != s and len(words) < 3


def make_segments(lines):
    """Group lines so a run of noise lines becomes one segment. Each segment keeps
    its raw lines and a clean version: a placeholder for noise, IDs replaced otherwise."""
    segments = []
    for line in lines:
        noise = is_noise(line)
        if noise and segments and segments[-1]["noise"]:
            segments[-1]["raw"].append(line)
        else:
            segments.append({"noise": noise, "raw": [line]})
    for seg in segments:
        first = seg["raw"][0]
        if seg["noise"]:
            indent = first[: len(first) - len(first.lstrip())]
            n = len(seg["raw"])
            seg["clean"] = f"{indent}[{n} line{'s' if n > 1 else ''} of IDs/data]"
        else:
            seg["clean"] = strip_ids(first)
        seg["raw"] = "\n".join(seg["raw"])
        seg["tokens"] = count_tokens(seg["clean"])
    return segments


def split_units(body):
    """Split the body into units: a non-indented line (top-level list item, paragraph
    line or heading) plus the indented lines under it. Consecutive noise lines stay in
    one unit, so a list of 69 UUIDs collapses into one placeholder.
    Returns [{"lines": [...], "blank_before": bool}]."""
    units, blank = [], False
    for line in body.splitlines():
        if not line.strip():
            blank = True
            continue
        top_level = not line[0].isspace()
        continues_noise = (
            units and not blank and is_noise(line) and is_noise(units[-1]["lines"][-1])
        )
        if units and not blank and (not top_level or continues_noise):
            units[-1]["lines"].append(line)
        else:
            units.append({"lines": [line], "blank_before": blank})
        blank = False
    return units


def chunk_note(note):
    """Pack units into chunks of about CHUNK_TOKENS tokens (cleaned text + header), in
    order. A unit bigger than the budget is split between its segments.
    Returns [{"text": raw text, "embed_text": cleaned text}]."""
    budget = CHUNK_TOKENS - count_tokens(context_header(note))

    # Items to pack: whole units, or the segments of a unit that is too big
    items = []
    for unit in split_units(note["body"]):
        segments = make_segments(unit["lines"])
        if sum(s["tokens"] for s in segments) <= budget:
            segments = [
                {
                    "raw": "\n".join(s["raw"] for s in segments),
                    "clean": "\n".join(s["clean"] for s in segments),
                    "tokens": sum(s["tokens"] for s in segments),
                }
            ]
        for i, seg in enumerate(segments):
            items.append({**seg, "blank_before": unit["blank_before"] and i == 0})

    chunks, current, used = [], [], 0
    for item in items:
        if current and used + item["tokens"] > budget:
            chunks.append(current)
            current, used = [], 0
        current.append(item)
        used += item["tokens"]
    if current:
        chunks.append(current)

    def join(chunk, key):
        out = chunk[0][key]
        for item in chunk[1:]:
            out += ("\n\n" if item["blank_before"] else "\n") + item[key]
        return out

    return [{"text": join(c, "raw"), "embed_text": join(c, "clean")} for c in chunks]


def main():
    # for each note: load -> split into units -> chunk
    # write one JSON line per chunk to OUT:
    # {"chunk_id": "Work Notes/2024-03-18.md#0", "file": "Work Notes/2024-03-18.md",
    #  "folder": "Work Notes", "title": "2024-03-18", "date": "2024-03-18",
    #  "position": 0, "text": "<raw>", "embed_text": "<cleaned>", "n_tokens": 87}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_notes, tokens = 0, []
    with open(OUT, "w", encoding="utf-8") as f:
        for path in sorted(NOTES_DIR.glob("*/*.md")):
            note = load_note(path)
            n_notes += 1
            for position, chunk in enumerate(chunk_note(note)):
                # n_tokens is exactly what the model gets: header + text + [CLS]/[SEP]
                n = count_tokens(f"{context_header(note)}\n{chunk['embed_text']}") + 2
                record = {
                    "chunk_id": f"{note['file']}#{position}",
                    "file": note["file"],
                    "folder": note["folder"],
                    "title": note["title"],
                    "date": note["date"],
                    "position": position,
                    "text": chunk["text"],
                    "embed_text": chunk["embed_text"],
                    "n_tokens": n,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                tokens.append(n)
    tokens.sort()
    print(
        f"{n_notes} notes -> {len(tokens)} chunks in {OUT}\n"
        f"tokens/chunk: median {tokens[len(tokens) // 2]}, "
        f"p90 {tokens[int(len(tokens) * 0.9)]}, max {tokens[-1]}; "
        f"over {MODEL_MAX_TOKENS} (truncated): {sum(t > MODEL_MAX_TOKENS for t in tokens)}"
    )


if __name__ == "__main__":
    main()
