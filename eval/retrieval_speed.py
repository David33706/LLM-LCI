#!/usr/bin/env python3
"""
retrieval_speed.py - Retrieval-latency benchmark over the eval set (no LLM).

WHY SEPARATE FROM tools/ask.py
------------------------------
tools/ask.py reports ONE end-to-end query the way a user experiences it (and the
LLM dominates that time, and is machine-dependent). This script reports the
RETRIEVAL latency alone, aggregated over the whole eval set - the number that
pairs with the Hit@k / MRR table and is near-stable across machines. No LLM, so
it runs in seconds.

WHAT IT REPORTS
---------------
Per-query hybrid retrieval time over N repeats of the eval questions, summarised
as median / p95 / mean / min / max. Median + p95 are the honest pair for
right-skewed latency. Warm-up runs are discarded.

Standalone like bm25_eval.py: it does NOT touch evaluate.py's scoring logic.

Defaults = the validated config: weighted fusion, alpha=0.4, candidate_k=100, top_k=10.

RUN FROM THE REPO ROOT:
  python eval/retrieval_speed.py --eval-set my_questions.csv
  python eval/retrieval_speed.py --eval-set my_questions.csv --repeats 5
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# same message as step3_orchestrate.EVAL_SET_MISSING (this script does not import step3)
EVAL_SET_MISSING = ("ERROR: eval set not found: {}. The project's 50-question eval set is not published; "
                    "pass --eval-set with your own .xlsx (sheet \"Questions\") or .csv with columns "
                    "id,type,question,source_doi (optional: valid_dois, expected_answer, source_location).")


def load_questions(path):
    """Read questions from .xlsx/.xlsm or .csv; pick the column whose header contains 'question'."""
    p = Path(path)
    if not p.exists():
        print(EVAL_SET_MISSING.format(path))
        sys.exit(1)

    if p.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
        except ImportError:
            print("ERROR: openpyxl is required to read .xlsx. Run: pip install openpyxl")
            sys.exit(1)
        wb = load_workbook(p, read_only=True, data_only=True)
        ws = wb["Questions"] if "Questions" in wb.sheetnames else wb.active
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    else:  # csv
        import csv
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.reader(f)]

    if not rows:
        print("ERROR: eval set is empty.")
        sys.exit(1)

    header = [("" if h is None else str(h)).strip().lower() for h in rows[0]]
    q_idx = next((i for i, h in enumerate(header) if "question" in h), None)
    if q_idx is None:
        print(f"ERROR: no 'question' column found. Headers seen: {header}")
        sys.exit(1)

    questions = []
    for r in rows[1:]:
        if r is None or q_idx >= len(r) or r[q_idx] is None:
            continue
        q = str(r[q_idx]).strip()
        if q:
            questions.append(q)
    if not questions:
        print("ERROR: found a 'question' column but no question rows.")
        sys.exit(1)
    return questions


def main():
    ap = argparse.ArgumentParser(description="Hybrid retrieval-latency benchmark (no LLM).")
    ap.add_argument("--eval-set", default="eval/LCA_RAG_eval_set_v3_50.xlsx",
                    help=".xlsx or .csv with a 'question' column (default: the private v3_50 workbook, not published).")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chroma-dir", default="chroma_db")
    ap.add_argument("--collection", default="papers")
    ap.add_argument("--bm25-dir", default="bm25_index/papers")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--fusion", choices=["weighted", "rrf"], default="weighted")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3, help="Timed passes over the question set.")
    ap.add_argument("--warmup", type=int, default=2, help="Warm-up queries to discard.")
    args = ap.parse_args()

    questions = load_questions(args.eval_set)
    print(f"Loaded {len(questions)} questions from {args.eval_set}")

    try:
        from hybrid import build_hybrid_retriever
    except ImportError:
        print("ERROR: import of build_hybrid_retriever failed.")
        print("Run from the repo root: python eval/retrieval_speed.py --eval-set ...")
        sys.exit(1)

    retriever = build_hybrid_retriever(
        chroma_dir=args.chroma_dir,
        chroma_collection=args.collection,
        embedding_model=args.embedding_model,
        bm25_dir=args.bm25_dir,
        llm_model=None,                 # retrieval only
        fusion=args.fusion,
        alpha=args.alpha,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )

    # preflight: chroma non-empty (a missing BM25 index raises a clear error on first query)
    try:
        n = retriever.vector_store.count()
    except Exception as e:
        print(f"ERROR: could not open Chroma: {e}")
        sys.exit(1)
    if n == 0:
        print(f"ERROR: Chroma collection '{args.collection}' is empty (corpus not on this machine).")
        sys.exit(1)

    cfg = (f", alpha={args.alpha}" if args.fusion == "weighted" else f", rrf_k={args.rrf_k}")
    print(f"chroma_db: {n:,} chunks | hybrid {args.fusion}{cfg}, top_k={args.top_k}\n")

    # warm-up (discarded)
    for i in range(max(0, args.warmup)):
        retriever.query(questions[i % len(questions)], top_k=args.top_k, generate_answer=False)

    # timed passes
    per_query_ms = []
    for _ in range(max(1, args.repeats)):
        for q in questions:
            t0 = time.perf_counter()
            retriever.query(q, top_k=args.top_k, generate_answer=False)
            per_query_ms.append((time.perf_counter() - t0) * 1000.0)

    per_query_ms.sort()
    n_obs = len(per_query_ms)
    median = statistics.median(per_query_ms)
    p95 = per_query_ms[min(n_obs - 1, int(round(0.95 * (n_obs - 1))))]
    mean = statistics.fmean(per_query_ms)

    print(f"Hybrid retrieval latency over {len(questions)} questions x {args.repeats} repeats "
          f"= {n_obs} observations:")
    print(f"  median : {median:8.1f} ms")
    print(f"  p95    : {p95:8.1f} ms")
    print(f"  mean   : {mean:8.1f} ms")
    print(f"  min    : {per_query_ms[0]:8.1f} ms")
    print(f"  max    : {per_query_ms[-1]:8.1f} ms")
    print("\nNote: retrieval is near machine-independent; end-to-end time is dominated")
    print("by the LLM (see tools/ask.py) and varies by machine.")


if __name__ == "__main__":
    main()