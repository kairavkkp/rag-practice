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


class HybridIndex:
    """Dense + BM25, merged with reciprocal rank fusion (RRF).
    RRF ignores the raw scores (cosine and BM25 aren't comparable) and uses ranks:
    score = sum over lists of 1 / (RRF_K + rank). Chunks high in both lists win."""

    RRF_K = 60  # standard default from the RRF paper
    DEPTH = 20  # how many results to take from each list before fusing

    def __init__(self):
        from bm25 import BM25Index

        self.dense, self.bm25 = DenseIndex(), BM25Index()

    def search(self, query, k=5):
        scores, chunks = {}, {}
        for results in (self.dense.search(query, self.DEPTH), self.bm25.search(query, self.DEPTH)):
            for rank, (_, chunk) in enumerate(results, 1):
                cid = chunk["chunk_id"]
                scores[cid] = scores.get(cid, 0) + 1 / (self.RRF_K + rank)
                chunks[cid] = chunk
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [(scores[cid], chunks[cid]) for cid in top]


class RerankIndex:
    """Hybrid retrieval for recall, then a cross-encoder for precision.
    Dense/BM25 score query and chunk separately; a cross-encoder reads them together
    in one pass, so it judges "does this chunk answer this question?" much better.
    Too slow for every chunk, fine for CANDIDATES of them."""

    RERANKER = "BAAI/bge-reranker-base"
    CANDIDATES = 20

    def __init__(self):
        from sentence_transformers import CrossEncoder

        from build_chunks import context_header

        self.hybrid = HybridIndex()
        self.model = CrossEncoder(self.RERANKER)
        self.header = context_header

    def search(self, query, k=5):
        candidates = [c for _, c in self.hybrid.search(query, self.CANDIDATES)]
        # The cleaned text plus header, like the embedder sees (and it fits in 512 tokens)
        pairs = [(query, f"{self.header(c)}\n{c['embed_text']}") for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])[:k]
        return [(float(s), c) for s, c in ranked]


if __name__ == "__main__":
    index = RerankIndex()
    for score, chunk in index.search(" ".join(sys.argv[1:]) or "GL Code"):
        print(f"{score:.3f}  {chunk['chunk_id']}")
