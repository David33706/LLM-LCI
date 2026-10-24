#!/usr/bin/env python3
"""
tools/ask.py — one free-text question against the LCA librarian.

Locate ("which study contains X?") -> retrieval only: no GPU, runs anywhere
the index folders exist. Assume ("what was assumed?") -> full reader path:
needs the vLLM server (RUNBOOK) or the one-shot wrapper ./ask.sh.

Imports step3_orchestrate's parser, pipeline and orchestrator — shared, never
copied, so ask.py cannot drift from the measured eval path.

  python tools/ask.py "Which study contains the LCI for constructed wetlands?"
  python tools/ask.py "What CO2 capture efficiency is assumed for MEA capture?"
  python tools/ask.py "..." --mode locate --task1-topn 5
"""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)  # step3 paths (chroma_db, elsevier_xml, ...) are repo-root-relative

import requests
import step3_orchestrate as s3

LOCATE_RE = re.compile(r"\bwhich\s+(stud(y|ies)|papers?)\b", re.I)
EXPECTED_CHUNKS = 290_247


def preflight_index(args):
    try:
        import chromadb
        n = (chromadb.PersistentClient(path=args.chroma_dir)
             .get_collection(args.collection).count())
    except Exception as e:
        sys.exit(f"ERROR: vector index not openable at '{args.chroma_dir}' "
                 f"({type(e).__name__}: {e}).\n"
                 f"Copy chroma_db/ and bm25_index/ from a working machine (RUNBOOK).")
    if n == 0:
        sys.exit("ERROR: vector index is EMPTY — retrieval would silently return "
                 "nothing. Copy chroma_db/ from a working machine (RUNBOOK).")
    if n == EXPECTED_CHUNKS:
        print(f"vector store: {n:,} chunks")
    else:
        print(f"vector store: {n:,} chunks (frozen corpus = {EXPECTED_CHUNKS:,}) "
              f"— fine if papers were added via add_papers.py")


def preflight_server(args):
    base = args.url.split("/v1/")[0]
    try:
        r = requests.get(base + "/v1/models", timeout=5)
        r.raise_for_status()
    except Exception:
        sys.exit(
            "ERROR: assumption questions need the reader server (GPU); none is "
            f"reachable at {base}.\n"
            "  option 1 (interactive): start it on a GPU node — bash serve_vllm.sh\n"
            "  option 2 (one-shot):    ./ask.sh \"your question\"  from the login node")
    # client/server mismatches give wrong results silently
    try:
        served = {m["id"]: m for m in r.json()["data"]}
    except Exception as e:
        sys.exit(f"ERROR: unexpected /v1/models reply from {base} "
                 f"({type(e).__name__}: {e}) — is this a vLLM server?")
    if args.model not in served:
        sys.exit(f"ERROR: server at {base} serves {sorted(served)}, but --model is "
                 f"'{args.model}'. Pass --model to match the server's SERVED_NAME.")
    srv_len = served[args.model].get("max_model_len")
    if srv_len is not None and srv_len != args.ctx:
        sys.exit(f"ERROR: server max_model_len is {srv_len}, but --ctx is {args.ctx}. "
                 f"Pass --ctx {srv_len} (client --ctx must equal server MAX_LEN).")


def main():
    ap = s3.build_parser()
    ap.add_argument("question", help="free-text question (quote it in the shell)")
    ap.add_argument("--mode", choices=["auto", "locate", "assume"], default="auto",
                    help="locate = which-study, retrieval only; assume = reader; "
                         "auto = locate iff the question asks 'which study/paper'")
    args = ap.parse_args()
    s3.resolve_args(args)

    mode = args.mode
    if mode == "auto":
        mode = "locate" if LOCATE_RE.search(args.question) else "assume"

    preflight_index(args)
    if mode == "assume":
        preflight_server(args)

    q = {"id": "", "type": "TASK1" if mode == "locate" else "TASK2",
         "question": args.question, "source_doi": "", "expected_answer": ""}

    print("=" * 84)
    label = "locate (retrieval only)" if mode == "locate" else f"assume (reader: {args.model})"
    print(f"ASK  |  {label}")
    print(f"Q: {args.question}")
    pipeline = s3.build_pipeline(args)
    titles = s3.load_titles(args.wos)
    s3.orchestrate(q, pipeline, titles, args, verbose=True)


if __name__ == "__main__":
    main()