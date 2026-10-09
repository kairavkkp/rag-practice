"""Embed every chunk into data/index/embeddings.npy (row i = line i of chunks.jsonl)."""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from build_chunks import MODEL, OUT as CHUNKS, context_header

EMBEDDINGS = CHUNKS.parent / "embeddings.npy"


def load_chunks():
    with open(CHUNKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    # Same header the prompt will use, so the vector knows the date/folder/title
    texts = [f"{context_header(c)}\n{c['embed_text']}" for c in chunks]
    model = SentenceTransformer(MODEL)
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    assert len(vectors) == len(chunks)
    np.save(EMBEDDINGS, vectors)
    print(f"{len(chunks)} chunks -> {vectors.shape} in {EMBEDDINGS}")


if __name__ == "__main__":
    main()
