"""Dense retrieval: brute-force cosine similarity over embeddings.npy.

    python src/search.py "what was INVPLAT-1899?"
"""

import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from build_chunks import MODEL
from embed import EMBEDDINGS, load_chunks

# bge v1.5: short queries match passages better with this prefix (passages get none)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def normalize(m):
    """Scale vectors to length 1, so a dot product is the cosine similarity."""
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class DenseIndex:
    def __init__(self):
        self.chunks = load_chunks()
        self.vectors = normalize(np.load(EMBEDDINGS))
        assert len(self.vectors) == len(self.chunks), "re-run embed.py after build_chunks.py"
        self.model = SentenceTransformer(MODEL)

    def search(self, query, k=5):
        """Return the top-k chunks as [(score, chunk)], best first."""
        q = normalize(self.model.encode(QUERY_PREFIX + query))
        scores = self.vectors @ q  # cosine similarity with every chunk at once
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]


if __name__ == "__main__":
    index = DenseIndex()
    for score, chunk in index.search(" ".join(sys.argv[1:]) or "GL Code"):
        print(f"{score:.3f}  {chunk['chunk_id']}")
