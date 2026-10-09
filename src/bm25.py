"""Keyword retrieval: BM25 over the raw chunk text (plus its context header).

    python src/bm25.py "what was INVPLAT-1899?"
"""

import math
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from build_chunks import context_header
from embed import load_chunks

K1 = 1.5  # how fast repeats of a term stop adding score
B = 0.75  # how much long chunks are penalised (0 = not at all, 1 = fully)

# Words joined by - _ . / : stay one token: INVPLAT-1899, s3:GetObject, PPD-1294
TOKEN = re.compile(r"[a-z0-9]+(?:[-_./:][a-z0-9]+)*")

# Dropped from queries. Notes are terse bullets, so words like "does" or "what" are rare
# in them and IDF would rate them as important (e.g. "does" scored higher than "winfactor").
STOPWORDS = set(
    """a an the and or of to in on at by for with from about as into
    what which who whom whose when where why how
    is are was were be been being am do does did done have has had
    i me my we our you your they them their it its this that these those there
    any some can could would should will shall may might""".split()
)


def tokenize(text):
    """Lowercase tokens. A compound like 'invplat-1899' is also split into its parts,
    so 'ticket 1899' still matches and the full ID scores extra when it matches."""
    tokens = []
    for tok in TOKEN.findall(text.lower()):
        tokens.append(tok)
        parts = re.split(r"[-_./:]", tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return tokens


class BM25Index:
    def __init__(self):
        self.chunks = load_chunks()
        # Index the header too, so dates, folder and title are searchable
        docs = [tokenize(f"{context_header(c)}\n{c['text']}") for c in self.chunks]
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean())
        n = len(docs)

        # postings: term -> [(chunk index, term count in that chunk)]
        self.postings = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))

        # Rare terms matter more. This IDF variant never goes negative.
        self.idf = {
            term: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1)
            for term, p in self.postings.items()
        }

    def search(self, query, k=5):
        """Return the top-k chunks as [(score, chunk)], best first."""
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(tokenize(query)) - STOPWORDS:
            for i, tf in self.postings.get(term, []):
                length_norm = 1 - B + B * self.doc_len[i] / self.avg_len
                scores[i] += self.idf[term] * tf * (K1 + 1) / (tf + K1 * length_norm)
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top if scores[i] > 0]


if __name__ == "__main__":
    index = BM25Index()
    for score, chunk in index.search(" ".join(sys.argv[1:]) or "GL Code"):
        print(f"{score:6.2f}  {chunk['chunk_id']}")
