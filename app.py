"""Streamlit UI to ask questions and inspect retrieval.

    streamlit run app.py      (from the repo root, venv active)
"""

import sys
import urllib.error
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from answer import K, answer  # noqa: E402
from build_chunks import NOTES_DIR, load_note  # noqa: E402
from search import RerankIndex  # noqa: E402

MODES = {
    "rerank": "hybrid top 20, re-scored by a cross-encoder (best)",
    "hybrid": "dense + BM25 merged with reciprocal rank fusion",
    "bm25": "keyword match: exact IDs, names, dates",
    "dense": "embedding similarity: meaning, paraphrases",
}
FOLDERS = ["All", "Work Notes", "Lighthouz AI"]

# Demo questions from eval/questions.json, labelled with what each one shows
SAMPLES = [
    ("Exact ticket ID", "What was INVPLAT-1899?"),
    ("Changed over time", "What's the status of PPD-1294?"),
    ("Paraphrased", "How does G2 decide whether a new customer is safe to factor?"),
    ("Spread across days", "How did the Load ID regex work evolve in late 2025?"),
    ("Date in the question", "Which PRs did I move to prod on 2024-06-05?"),
    ("Not in the notes", "Who is on the PagerDuty on-call rotation?"),
]


def use_sample(question):
    st.session_state.question = question
    st.session_state.auto_ask = True


@st.cache_resource(show_spinner="Loading models...")
def load_indexes():
    """Load once per server; the rerank index already holds the other three."""
    r = RerankIndex()
    return {"rerank": r, "hybrid": r.hybrid, "bm25": r.hybrid.bm25, "dense": r.hybrid.dense}


st.set_page_config(page_title="Notes RAG", layout="wide")
st.title("Ask my work notes")

with st.sidebar:
    mode = st.radio("Retrieval", list(MODES), format_func=lambda m: f"{m}: {MODES[m]}")
    folder = st.selectbox("Folder", FOLDERS)
    folder = None if folder == "All" else folder
    retrieval_only = st.checkbox("Retrieval only (skip the LLM)", help="Fast; for comparing modes")

indexes = load_indexes()
index = indexes[mode]
chunks_by_id = {c["chunk_id"]: c for c in indexes["dense"].chunks}

st.caption("Try a sample question (each shows a different retrieval challenge):")
cols = st.columns(3)
for i, (label, sample) in enumerate(SAMPLES):
    cols[i % 3].button(
        label, help=sample, on_click=use_sample, args=(sample,), use_container_width=True
    )

with st.form("ask"):
    question = st.text_input("Question", key="question", placeholder="Ask anything about the notes")
    submitted = st.form_submit_button("Ask")

# A sample click fills the box and asks right away
if (submitted or st.session_state.pop("auto_ask", False)) and question.strip():
    if retrieval_only:
        results = index.search(question, k=K, folder=folder)
        retrieved = [{"chunk_id": c["chunk_id"], "score": s} for s, c in results]
    else:
        try:
            with st.spinner("Retrieving and asking the LLM..."):
                result = answer(question, index, folder)
        except urllib.error.URLError:
            st.error("Can't reach Ollama at localhost:11434. Start it with `ollama serve`.")
            st.stop()
        retrieved = result["retrieved"]
        st.subheader("Answer")
        st.markdown(result["answer"])
        st.caption(
            f"retrieve {result['seconds']['retrieve']} s · LLM {result['seconds']['llm']} s"
            f" · {result['model']}"
        )
        st.subheader("Sources given to the LLM")
        for i, f in enumerate(result["sources"], 1):
            with st.expander(f"[{i}] {f}"):
                st.text(load_note(NOTES_DIR / f)["body"])

    st.subheader(f"Retrieved chunks ({mode})")
    if not retrieved:
        st.info("Nothing retrieved.")
    for rank, r in enumerate(retrieved, 1):
        chunk = chunks_by_id[r["chunk_id"]]
        with st.expander(f"{rank}. {r['chunk_id']}  ·  score {r['score']:.4f}  ·  {chunk['date']}"):
            raw, embedded = st.tabs(["raw text (BM25, LLM)", "embed_text (vector)"])
            raw.text(chunk["text"])
            embedded.text(chunk["embed_text"])
