#!/usr/bin/env python3
"""
build_bm25_index.py - Build a BM25 keyword index from an existing Chroma collection.

WHY THIS EXISTS
---------------
Stage 2 adds hybrid retrieval = dense (existing Chroma) + sparse (BM25 keyword).
BM25 needs the chunk TEXT to build its index. That text is already stored in
Chroma (the `documents` field), so this script pulls it out once, tokenizes it,
and saves a BM25 index to disk. Build it once; the HybridRetriever loads it at
query time. No re-reading XML, no re-embedding.

It reads the SAME chunks the dense retriever uses, in the SAME order, and stores
each chunk's id + doi + section_title alongside, so a BM25 hit maps back to the
exact same chunk identity the dense side returns. That alignment is what lets
hybrid.py fuse the two rankings correctly (validated config: weighted fusion,
alpha=0.4 on the dense side; RRF is the other, non-default option).

USAGE (from the repo root: --chroma-dir and --out are relative to the cwd)
-----
  # Build from the MiniLM 'papers' collection (the dense baseline)
  python ingest/build_bm25_index.py --collection papers --out bm25_index/papers

  # The output dir holds the bm25s index + a chunk-metadata sidecar (ids/dois).

REQUIRES
--------
  pip install bm25s
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Build a BM25 index from a Chroma collection.")
    ap.add_argument("--collection", default="papers", help="Chroma collection to read chunk text from")
    ap.add_argument("--chroma-dir", default="chroma_db", help="Chroma persistence directory")
    ap.add_argument("--out", default=None,
                    help="Output dir for the BM25 index (default: bm25_index/<collection>)")
    ap.add_argument("--page", type=int, default=5000, help="Chroma read page size")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("bm25_index") / args.collection

    # --- imports with clear errors ---
    try:
        import bm25s
    except ImportError:
        print("ERROR: bm25s is not installed. Run: pip install bm25s")
        sys.exit(1)
    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb is not installed.")
        sys.exit(1)

    # --- open the collection ---
    if not os.path.isdir(args.chroma_dir):   # PersistentClient would create an empty store
        print(f"ERROR: --chroma-dir not found: {os.path.abspath(args.chroma_dir)} "
              f"(run from the repo root)")
        sys.exit(1)
    client = chromadb.PersistentClient(path=args.chroma_dir)
    try:
        coll = client.get_collection(args.collection)
    except Exception as e:
        print(f"ERROR: could not open collection '{args.collection}' in {args.chroma_dir}: {e}")
        sys.exit(1)

    total = coll.count()
    if total == 0:
        print(f"ERROR: collection '{args.collection}' is empty.")
        sys.exit(1)
    print(f"Collection '{args.collection}' has {total} chunks. Pulling text...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- pull ALL chunk text + ids + metadata, in stable order, paged ---
    # We keep ids and dois aligned with the corpus list index, so a BM25 hit at
    # position j maps to corpus_ids[j] / corpus_dois[j].
    corpus_texts = []
    corpus_ids = []
    corpus_dois = []
    corpus_sections = []

    t0 = time.time()
    for offset in range(0, total, args.page):
        got = coll.get(
            include=["documents", "metadatas"],
            limit=args.page,
            offset=offset,
        )
        docs = got.get("documents", []) or []
        metas = got.get("metadatas", []) or []
        ids = got.get("ids", []) or []
        for k in range(len(docs)):
            txt = docs[k] or ""
            meta = metas[k] if k < len(metas) else {}
            corpus_texts.append(txt)
            corpus_ids.append(ids[k] if k < len(ids) else None)
            corpus_dois.append((meta or {}).get("doi"))
            corpus_sections.append((meta or {}).get("section_title"))
        print(f"  pulled {min(offset + args.page, total)}/{total}")

    print(f"Pulled {len(corpus_texts)} chunks in {time.time()-t0:.1f}s. Tokenizing + indexing...")

    # --- tokenize and build the BM25 index ---
    # bm25s.tokenize with its defaults: lowercases, keeps tokens of 2+ word chars
    # (token_pattern r"(?u)\b\w\w+\b", so 1-char tokens are dropped), and in the
    # installed bm25s 0.3.9 REMOVES English stopwords (stopwords="english" is the
    # default). The query side (hybrid.py) uses the same defaults, so they match.
    t1 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    print(f"Indexed in {time.time()-t1:.1f}s.")

    # --- save index + the aligned id/doi sidecar ---
    index_path = out_dir / "bm25s"
    retriever.save(str(index_path))
    sidecar = {
        "collection": args.collection,
        "chroma_dir": args.chroma_dir,
        "count": len(corpus_texts),
        "ids": corpus_ids,
        "dois": corpus_dois,
        "sections": corpus_sections,
    }
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f)

    # Also stash the raw texts so the retriever can return chunk text without
    # re-reading Chroma. (Kept separate from the bm25s index files.)
    with open(out_dir / "texts.json", "w", encoding="utf-8") as f:
        json.dump(corpus_texts, f)

    print("\n" + "=" * 60)
    print("BM25 INDEX BUILT")
    print(f"  source collection : {args.collection}")
    print(f"  chunks indexed    : {len(corpus_texts)}")
    print(f"  output dir        : {out_dir}")
    print(f"    - bm25s/        (the BM25 index)")
    print(f"    - chunks.json   (aligned ids / dois / sections)")
    print(f"    - texts.json    (chunk texts)")
    print("=" * 60)


if __name__ == "__main__":
    main()