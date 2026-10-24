#!/usr/bin/env python3
"""
ingest_dir.py — Batch-ingest local Elsevier XML files into ChromaDB.

WHY THIS EXISTS
---------------
rag.py's built-in `ingest` re-fetches each paper from the Elsevier API over the
network. You already have all the XML on disk in elsevier_xml/, so this script
reads straight from the folder — no network, no API key, no rate limits. It
reuses rag.py's pipeline unchanged (same chunker, embedder, Chroma store), so
whatever you ingest here is identical to what `ingest` would produce.

USAGE (from the repo root)
-----
  # Full build into an EMPTY collection (wipe first: RUNBOOK §3b step 1)
  python ingest/ingest_dir.py --xml-dir elsevier_xml --collection papers

  # Resume after interruption (skips DOIs already in the collection)
  python ingest/ingest_dir.py --xml-dir elsevier_xml --collection papers --resume

  # Dry run: see what would be ingested, embed nothing
  python ingest/ingest_dir.py --xml-dir elsevier_xml --dry-run

GUARDS (fail loudly, exit 1)
------
  - Without --resume, refuses to write into a collection that already has
    records: prose chunk IDs contain a random uuid4, so re-ingesting appends
    orphan chunks instead of overwriting.
  - With --resume, exits if the existing DOIs cannot be read.
  - Exits 1 after the summary if any file errored (the collection is incomplete).

OUTPUT
------
  - Builds/updates the Chroma collection at --chroma-dir.
  - Writes an ingest log CSV (one row per paper: doi, status, chunks_added, reason).
"""

import argparse
import csv

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def filename_to_doi(filename: str) -> str:
    """Reverse the save_xml_to_file naming: '10.1016_j.afres.2025.100716.xml'
    -> '10.1016/j.afres.2025.100716'.

    save_xml_to_file replaced '/' '\\' ':' with '_'. In practice the only one
    that appears in Elsevier DOIs is '/', and it's always the FIRST underscore
    (the '10.1016/' publisher prefix). Replacing just the first underscore
    reconstructs the DOI without corrupting the rest of the suffix."""
    stem = filename[:-4] if filename.lower().endswith(".xml") else filename
    return stem.replace("_", "/", 1)


def existing_dois_in_collection(vector_store) -> Set[str]:
    """Return the set of DOIs already present in the Chroma collection, so
    --resume can skip them. Reads metadata in pages to avoid loading documents."""
    seen: Set[str] = set()
    try:
        vector_store._ensure_collection()
        coll = vector_store._collection
        total = coll.count()
        if total == 0:
            return seen
        page = 5000
        for offset in range(0, total, page):
            got = coll.get(include=["metadatas"], limit=page, offset=offset)
            for meta in got.get("metadatas", []) or []:
                if meta and meta.get("doi"):
                    seen.add(str(meta["doi"]))
    except Exception as e:
        print(f"ERROR: --resume could not read existing DOIs from the collection: {e}")
        print("  Refusing to continue: ingesting without the skip list would append "
              "duplicate (orphan) prose chunks.")
        sys.exit(1)
    return seen


def main():
    ap = argparse.ArgumentParser(description="Batch-ingest local XML files into ChromaDB.")
    ap.add_argument("--xml-dir", default="elsevier_xml", help="Folder of .xml files")
    ap.add_argument("--chroma-dir", default="chroma_db", help="Chroma persistence directory")
    ap.add_argument("--collection", default="papers", help="Chroma collection name")
    ap.add_argument("--embedding-model",
                    default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Embedding model (default: MiniLM, the indexed model)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip DOIs already present in the collection")
    ap.add_argument("--limit", type=int, default=0,
                    help="Ingest at most N files (0 = all). Useful for a quick test.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be ingested; do not embed or write")
    ap.add_argument("--chunk-size", type=int, default=1200,
                    help="Chunker max size in CHARACTERS (default 1200, ~250 tokens)")
    ap.add_argument("--overlap", type=int, default=200,
                    help="Chunk overlap in characters (default 200)")
    ap.add_argument("--log", default=None,
                    help="Path to ingest log CSV (default: logs/ingest_<collection>_<ts>.csv)")
    args = ap.parse_args()

    xml_dir = Path(args.xml_dir)
    if not xml_dir.is_dir():
        print(f"ERROR: --xml-dir not found: {xml_dir}")
        sys.exit(1)

    xml_files = sorted(xml_dir.glob("*.xml"))
    if args.limit:
        xml_files = xml_files[:args.limit]
    if not xml_files:
        print(f"ERROR: no .xml files in {xml_dir}")
        sys.exit(1)
    print(f"Found {len(xml_files)} XML files in {xml_dir}")

    # Build the pipeline from rag.py (embedder=MiniLM by default, no LLM needed for ingest)
    try:
        from rag import build_default_pipeline
    except ImportError as e:
        print(f"ERROR: could not import rag.py from {_ROOT}: {e}")
        raise

    pipeline = build_default_pipeline(
        chroma_dir=args.chroma_dir,
        chroma_collection=args.collection,
        embedding_model=args.embedding_model,
        llm_model=None,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    skip: Set[str] = set()
    if args.resume:
        print("Reading existing DOIs for --resume ...")
        skip = existing_dois_in_collection(pipeline.vector_store)
        print(f"  {len(skip)} DOIs already in '{args.collection}' will be skipped")
    elif not args.dry_run:
        existing = pipeline.vector_store.count()
        if existing:
            print(f"ERROR: collection '{args.collection}' already has {existing} records; "
                  f"prose chunk IDs are random (uuid4) so re-ingesting appends orphans. "
                  f"Wipe first (RUNBOOK §3b step 1) or pass --resume.")
            sys.exit(1)

    if args.dry_run:
        print("\n--- DRY RUN (nothing will be embedded) ---")
        would = [f for f in xml_files if filename_to_doi(f.name) not in skip]
        for f in would[:20]:
            print(f"  would ingest: {filename_to_doi(f.name)}")
        if len(would) > 20:
            print(f"  ... and {len(would) - 20} more")
        print(f"\nTotal that would be ingested: {len(would)} (skipping {len(xml_files) - len(would)})")
        return

    # Log setup
    if args.log:
        log_path = Path(args.log)
    else:
        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path("logs") / f"ingest_{args.collection}_{ts}.csv"

    counts = {"success": 0, "skipped": 0, "no_text": 0, "no_chunks": 0, "error": 0}
    total_chunks = 0
    t0 = time.time()

    with open(log_path, "w", newline="", encoding="utf-8") as lf:
        w = csv.writer(lf)
        w.writerow(["doi", "status", "chunks_added", "reason", "file"])

        for i, path in enumerate(xml_files, start=1):
            doi = filename_to_doi(path.name)

            if doi in skip:
                counts["skipped"] += 1
                w.writerow([doi, "skipped", 0, "already_in_collection", path.name])
                continue

            try:
                xml_text = path.read_text(encoding="utf-8", errors="ignore")
                # ingest_xml accepts a raw XML string (extract_structured_sections parses it)
                result = pipeline.ingest_xml(doi=doi, xml_input=xml_text)
                status = result.get("status")
                if status == "success":
                    counts["success"] += 1
                    n = result.get("chunks_added", 0)
                    total_chunks += n
                    w.writerow([doi, "success", n, "", path.name])
                else:
                    reason = result.get("reason", "unknown")
                    bucket = "no_text" if reason == "no_text_extracted" else \
                             "no_chunks" if reason == "no_chunks_created" else "error"
                    counts[bucket] += 1
                    w.writerow([doi, "failed", 0, reason, path.name])
            except Exception as exc:
                counts["error"] += 1
                w.writerow([doi, "error", 0, str(exc)[:200], path.name])

            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                print(f"  [{i}/{len(xml_files)}] "
                      f"ok={counts['success']} skip={counts['skipped']} "
                      f"notext={counts['no_text']} err={counts['error']} "
                      f"chunks={total_chunks} ({rate:.1f} files/s)")

    elapsed = time.time() - t0
    try:
        final_vectors = pipeline.vector_store.count()
    except Exception:
        final_vectors = "?"

    print("\n" + "=" * 60)
    print(f"INGEST COMPLETE  ({elapsed:.0f}s)")
    print(f"  collection      : {args.collection}  @  {args.chroma_dir}")
    print(f"  embedding model : {args.embedding_model}")
    print(f"  success         : {counts['success']}")
    print(f"  skipped(resume) : {counts['skipped']}")
    print(f"  no text/sections: {counts['no_text']}")
    print(f"  no chunks       : {counts['no_chunks']}")
    print(f"  errors          : {counts['error']}")
    print(f"  chunks added    : {total_chunks}")
    print(f"  vectors in store: {final_vectors}")
    print(f"  log             : {log_path}")
    print("=" * 60)
    if counts["error"]:
        print("Some files errored — check the 'reason' column in the log.")
        print(f"ERROR: {counts['error']} files errored — the collection is incomplete; see {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()