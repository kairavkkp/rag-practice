# rag-practice

A learning project: build a RAG (retrieval-augmented generation) pipeline **by hand** over my own
work notes exported from Apple Notes. The goal is to understand every stage, not to ship fast.

## How to work with me

- **You write the code; I review it.** Write pipeline stages directly, keep them small and
  readable, and explain non-obvious choices so I can review and learn from them.
- **No RAG frameworks.** No LangChain, LlamaIndex, Haystack or vector DB clients. Plain Python and
  NumPy, plus the embedding model and the LLM. I implement cosine similarity, BM25, rank fusion,
  reranking, etc. myself.
- **Every time you change a file, give me** a git-log-style entry (conventional commit subject +
  bullet points). No diff needed.
- Keep explanations short and concrete. Run code to check claims where possible.
- Measure before improving: every new pipeline layer is judged against `eval/questions.json`.
- **Optimise for understanding, not eval score.** One technique per small change: explain the
  idea, show which questions moved and why (including failures), then move on. Don't over-tune.

## Environment

- macOS 27. Python venv at the repo root; run things with `python` (venv active), always from
  the repo root (paths like `data/...` are relative to it).
- LLM: Ollama `qwen3.5:9b` at `localhost:11434` (M1 Pro, 16 GB).
- Repo path: `~/Documents/repos/rag-practice`
- Privacy: these are real work notes. Prefer local models (sentence-transformers, Ollama).
  Never commit `data/`. Don't print large amounts of note content into logs or commit messages.

## Layout

```
export_work_notes.py           Apple Notes -> Markdown exporter (done, see below)
data/                          git-ignored
  raw/<Folder>.jsonl           raw backup from Notes, one JSON line per note
  notes/<Folder>/*.md          one Markdown file per note  <- pipeline input
  index/chunks.jsonl           built by src/build_chunks.py (derived, safe to delete)
  index/embeddings.npy         one vector per chunk, same order as chunks.jsonl
src/build_chunks.py            loader + noise cleaner + token-based chunker
src/embed.py                   chunks.jsonl -> embeddings.npy
src/bm25.py                    BM25Index: hand-written BM25 over raw text + header
src/search.py                  DenseIndex, HybridIndex (RRF), RerankIndex (cross-encoder)
src/answer.py                  rerank -> whole notes -> Ollama answer with [n] citations
eval/questions.json            15 test questions
eval/run_eval.py               retrieval eval: hit@5, recall@5, rank (--name <run>)
eval/results/<run>.json        saved eval runs (ids and scores only)
```

Two folders, from two different jobs, so some answers depend on the folder:

| Folder | Notes | Median words | Max words | Over 300 words |
|---|---|---|---|---|
| Work Notes | 435 | 70 | 553 | 3 |
| Lighthouz AI | 201 | 125 | 1568 | 29 |

## Note file format

```
---
date: 2024-03-18
title: "2024-03-18"
folder: "Work Notes"
created: 2024-03-18T10:00:00
modified: 2024-03-18T10:05:00
date_source: title
---

<body in Markdown>
```

- `title` and `folder` are JSON-quoted strings: parse them with `json.loads`.
- Parse header lines with `line.partition(":")` (values like `created` contain colons).
- `date` comes from the title if the title is exactly `YYYY-MM-DD` (`date_source: title`),
  otherwise from the creation date (`date_source: created`).
- Notes without a date title are named `<date>_<title-slug>.md`, and their body starts with
  `# <title>`.
- A note's identity is `<folder>/<filename>`; filenames repeat across folders.
- Load notes with `Path("data/notes").glob("*/*.md")`.

## Exporter (`export_work_notes.py`)

- `python export_work_notes.py`: exports all folders in `FOLDERS`, resumes where it left off.
- `--rebuild`: regenerates the `.md` files from `data/raw/*.jsonl` without opening Notes
  (use after changing a filter).
- `--fresh`: deletes the backups and re-reads everything (~26 min for both folders).
- Config constants at the top: `FOLDERS`, `EXCLUDE_TITLES`, and `MIN_TYPED_CHARS` for the
  handwriting heuristic (attachment + fewer than 30 typed characters = handwritten, skipped).
- macOS 27 quirks learned the hard way:
  - JXA (JavaScript for Automation) is broken: use **AppleScript** only.
  - `name of container of note` fails: don't ask Notes for a note's folder.
  - Error -600 "Application isn't running" means a stuck Notes process: `killall Notes`, reopen.
  - After interrupting a run, run `killall Notes` before starting again.

## Pipeline decisions so far

- **Chunking:** sized in **tokens** (bge tokenizer), not words: these notes average ~1.8
  tokens/word and IDs/code go much higher. Unit = a top-level line plus the indented lines
  under it (one task with its sub-items). Pack units in order up to `CHUNK_TOKENS = 200`
  (header included); a unit over budget is split between its lines. Never merge across notes:
  each chunk comes from one note and one date. Nothing may exceed 512 tokens (silent truncation).
- **Embed clean, keep raw:** ~23% of tokens are machine noise (UUIDs, base64, JSON, REPL
  output, pasted HTML). `embed_text` replaces noise runs with `[N lines of IDs/data]` and long
  strings with `[id]`; short ticket IDs (INVPLAT-1899) are kept. `text` stays raw for BM25 and
  the prompt.
- **Embed small, return big:** retrieve chunks; give the LLM the whole raw note (median ~130
  tokens), or the chunk plus neighbours for very large notes. BM25 runs over raw chunk `text`
  so it ranks the same units as vector search and the two can be fused.
- **Chunk record:**
  `{"chunk_id": "<folder>/<file>#<position>", "file": "<folder>/<file>", "folder", "title",
  "date", "position", "text", "embed_text", "n_tokens"}`
- **Embedding model:** `BAAI/bge-small-en-v1.5` (512-token limit fits 250-word chunks plus header).
  Don't use `all-MiniLM-L6-v2` with 250-word chunks: it silently truncates at ~190 words.
- **Context header:** at embed time and in the prompt, prefix each chunk with
  `Date: <date> | Folder: <folder> | Note: <title>`.
- **Storage:** brute-force cosine similarity over a NumPy array (~900 chunks). No vector DB.
  Parked idea: Postgres + pgvector, only if we outgrow files (100k+ chunks, several users,
  concurrent writes, or SQL-heavy metadata filtering). Not needed at this size.

## Roadmap

Based on the "production RAG" architecture: router -> query transform -> hybrid search ->
reranker -> compression -> LLM with forced citations, with evals and logging underneath.
Retrieval is the hard part; tune every stage separately.

1. [x] Export notes from Apple Notes
2. [ ] Write `eval/questions.json`: 10-15 questions across exact-match IDs, paraphrased wording,
   facts that changed over time, answers spread across days, and unanswerable.
   `sources` = list of `<folder>/<file>`.
3. [x] `src/build_chunks.py`: `load_note` (reviewed), `split_blocks`, `chunk_note`, `main`.
   Check: every note has a chunk, no empty chunks, spot-read 3 random chunks.
4. [x] Embed chunks, cosine top-k retrieval, prompt with citations, answer.
5. [x] Baseline eval: retrieval hit rate (an expected source in the top 5).
   `baseline-dense`: hit@5 9/13, mean recall@5 0.56. Misses: exact IDs (#1, #8), a name
   (#11), one paraphrase (#5, rank 14). Unanswerable top scores (max 0.642) overlap answerable
   hits (min 0.621), so a score threshold can't detect "no answer".
6. [ ] Add one layer at a time, re-running the eval after each:
   - [x] BM25 keyword search + reciprocal rank fusion (exact IDs and names).
     bm25 11/13 (recall 0.76), hybrid RRF 10/13 (0.63): fusion lets dense drag down exact-ID
     hits. Learned: IDF is collection-relative (query stopwords needed); no stemming.
   - [x] reranking: hybrid top 20 -> `BAAI/bge-reranker-base` -> best 5. **13/13, recall
     0.92**, every first hit at rank 1. Reranker scores still can't flag unanswerable.
   - citation / "I don't know" prompt (unanswerable questions)
   - date awareness, prefer newest (facts that changed)
   - query rewrite, HyDE, multi-query (paraphrased questions)
   - metadata filter by folder
   - logging every query, retrieval and answer to JSONL
   - router and compression last (likely unnecessary at this size)

Update the checkboxes in this file as stages are completed.