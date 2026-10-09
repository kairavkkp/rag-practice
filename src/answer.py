"""Answer a question from the notes with a local LLM (Ollama), citing sources.

    python src/answer.py "What's the status of PPD-1294?"
    python src/answer.py "What did I fix in the audit lambda?" --folder "Lighthouz AI"

Embed small, return big: retrieve chunks, then give the LLM each matched note in full.
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from build_chunks import NOTES_DIR, context_header, load_note
from search import RerankIndex

LOG = Path("data/logs/queries.jsonl")
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM = "qwen3.5:9b"
K = 5  # chunks to retrieve; their notes become the context

SYSTEM = """You answer questions about the user's work notes.
Use only the numbered notes provided. After every fact, cite its note like [1] or [2][3].
If the notes don't contain the answer, reply exactly: I don't know. The notes don't cover this.
Notes are dated; when facts changed over time, prefer the newest note and say so.
Be brief."""


def build_context(results):
    """One numbered source per note (several chunks can share a note), in rank order."""
    sources = []
    for _, chunk in results:
        if chunk["file"] not in [s["file"] for s in sources]:
            sources.append(load_note(NOTES_DIR / chunk["file"]))
    context = "\n\n".join(
        f"[{i}] {context_header(n)}\n{n['body']}" for i, n in enumerate(sources, 1)
    )
    return sources, context


def ask_llm(question, context):
    payload = {
        "model": LLM,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Notes:\n\n{context}\n\nQuestion: {question}"},
        ],
        "stream": False,
        "think": False,  # qwen thinks by default; slower and not needed for lookups
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        OLLAMA_URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["message"]["content"]


def log(entry):
    """One JSON line per question. Lives under data/ (git-ignored): it holds note content."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def answer(question, index, folder=None):
    """Retrieve, build the context, ask the LLM, log it. Returns the log entry."""
    t0 = time.perf_counter()
    results = index.search(question, k=K, folder=folder)
    t1 = time.perf_counter()
    sources, context = build_context(results)
    text = ask_llm(question, context)
    t2 = time.perf_counter()
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "folder": folder,
        "retrieved": [{"chunk_id": c["chunk_id"], "score": round(s, 4)} for s, c in results],
        "sources": [n["file"] for n in sources],
        "answer": text,
        "seconds": {"retrieve": round(t1 - t0, 2), "llm": round(t2 - t1, 2)},
        "model": LLM,
    }
    log(entry)
    return entry


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--folder", help='only search one folder, e.g. "Work Notes"')
    args = ap.parse_args()
    result = answer(" ".join(args.question), RerankIndex(), args.folder)
    print(result["answer"] + "\n")
    for i, f in enumerate(result["sources"], 1):
        print(f"[{i}] {f}")
    print(f"\n{result['seconds']}")
