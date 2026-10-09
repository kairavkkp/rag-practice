"""Answer a question from the notes with a local LLM (Ollama), citing sources.

    python src/answer.py "What's the status of PPD-1294?"

Embed small, return big: retrieve chunks, then give the LLM each matched note in full.
"""

import json
import sys
import urllib.request

from build_chunks import NOTES_DIR, context_header, load_note
from search import RerankIndex

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


def answer(question, index):
    sources, context = build_context(index.search(question, k=K))
    return ask_llm(question, context), sources


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    text, sources = answer(question, RerankIndex())
    print(text + "\n")
    for i, n in enumerate(sources, 1):
        print(f"[{i}] {n['file']}")
