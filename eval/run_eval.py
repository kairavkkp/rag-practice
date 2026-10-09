"""Retrieval eval over eval/questions.json.

    python eval/run_eval.py --mode hybrid --name hybrid-rrf

Metrics, per answerable question (sources are note files, so a chunk hit = its file):
  hit@5     an expected source is in the top 5
  recall@5  share of expected sources in the top 5 (matters for multi-day questions)
  rank      position of the first expected source in the top 20 (- if missing)
Unanswerable questions have no sources; their top score is printed so we can later pick a
threshold for "I don't know". Results are saved to eval/results/<name>.json (ids and
scores only, no note text) to compare runs.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bm25 import BM25Index  # noqa: E402
from search import DenseIndex, HybridIndex, RerankIndex  # noqa: E402

MODES = {"dense": DenseIndex, "bm25": BM25Index, "hybrid": HybridIndex, "rerank": RerankIndex}

QUESTIONS = Path("eval/questions.json")
RESULTS = Path("eval/results")
K, DEPTH = 5, 20


def unique_files(results):
    """Ranked note files from ranked chunks (a note can have several chunks)."""
    files = []
    for _, chunk in results:
        if chunk["file"] not in files:
            files.append(chunk["file"])
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="latest")
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text())
    index = MODES[args.mode]()
    rows = []
    for q in questions:
        results = index.search(q["question"], k=DEPTH)
        files = unique_files(results)
        sources = set(q["sources"])
        rank = next((i + 1 for i, f in enumerate(files) if f in sources), None)
        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "answerable": bool(sources),
                "hit": rank is not None and rank <= K,
                "recall": len(sources & set(files[:K])) / len(sources) if sources else None,
                "rank": rank,
                "top_score": results[0][0] if results else 0.0,
                "top_files": files[:K],
            }
        )

    print(f"{'id':>3}  {'category':<13}{'hit@5':>6}{'recall':>8}{'rank':>6}{'top':>7}")
    for r in rows:
        if r["answerable"]:
            hit, rec, rank = ("yes" if r["hit"] else "NO"), f"{r['recall']:.2f}", r["rank"] or "-"
        else:
            hit, rec, rank = "", "", ""
        print(f"{r['id']:>3}  {r['category']:<13}{hit:>6}{rec:>8}{rank:>6}{r['top_score']:>7.3f}")

    answerable = [r for r in rows if r["answerable"]]
    by_cat = defaultdict(list)
    for r in answerable:
        by_cat[r["category"]].append(r["hit"])
    print(
        f"\nhit@5 {sum(r['hit'] for r in answerable)}/{len(answerable)}"
        f" | mean recall@5 {sum(r['recall'] for r in answerable) / len(answerable):.2f}"
        f" | by category: "
        + ", ".join(f"{c} {sum(h)}/{len(h)}" for c, h in by_cat.items())
    )
    found = [r["top_score"] for r in answerable if r["hit"]]
    unans = [r["top_score"] for r in rows if not r["answerable"]]
    if found and unans:
        print(f"top score: answerable hits min {min(found):.3f} | unanswerable max {max(unans):.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"saved {RESULTS / args.name}.json")


if __name__ == "__main__":
    main()
