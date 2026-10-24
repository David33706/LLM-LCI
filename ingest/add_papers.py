#!/usr/bin/env python3
"""
ingest/add_papers.py — add NEW Elsevier XMLs to all three indexes, safely.

Point it at a folder of new XMLs. It:
  1. derives each DOI from the filename (10.1016_j.x.y.z.xml -> 10.1016/j.x.y.z),
  2. SKIPS DOIs already in the vector store (so the README wipe-first rule is
     never triggered — only genuinely new papers are appended; note the uuid
     wipe-rule applies to PROSE chunks from ingest_dir.py — table records use
     deterministic ids + upsert and are re-ingest-safe by construction),
  3. runs the SAME canonical builders as a full rebuild, scoped to a staging
     dir with only the new files: ingest_dir.py --resume -> tables ->
     index_tables.py -> build_bm25_index.py (full BM25 rebuild); any child
     exiting non-zero aborts the run (run_step),
  4. copies the XMLs into elsevier_xml/ and the table xlsx into tables_xml/
     so future FULL rebuilds include them,
  5. asserts these counts and aborts (non-zero exit) on any mismatch:
       - preflight: elsevier_xml/, tables_xml/ and chroma_db/ exist,
         paper_list2_2025.csv exists, and every --redo DOI has a file in
         --xml-dir (checked before any write);
       - after prose ingest: every new DOI has > 0 records in the store
         (the prose delta itself is printed, not asserted);
       - after table indexing: table records added == tables extracted
         (index_tables.py builds one record per xlsx sheet; sheet names are
         deduped, so the two are equal by construction);
       - after the BM25 rebuild: BM25 chunk count == store count.

  python ingest/add_papers.py --xml-dir new_papers/
  python ingest/add_papers.py --xml-dir new_papers/ --redo 10.1016/j.x.y.z
      (--redo: a previous add crashed mid-paper — delete that DOI's records
       and re-ingest it from the folder)

NOTE: chroma_db/, bm25_index/, elsevier_xml/, tables_xml/ are gitignored and
per-machine. Run this on EVERY machine that keeps an index (Mac AND cluster),
or rsync those four folders after running it once.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tables.xml_tables import process_single_xml  # confirmed per-file API


def filename_to_doi(xml_path: Path) -> str:
    # download convention: doi.replace('/', '_') + '.xml'  (utils/elsevier.py)
    return xml_path.stem.replace("_", "/", 1)


def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(_ROOT / "chroma_db"))
    return client.get_collection("papers")


def doi_record_count(col, doi: str) -> int:
    res = col.get(where={"doi": doi}, include=[])
    return len(res.get("ids", []))


def run_step(cmd: list, label: str):
    print(f"\n--- {label}: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(_ROOT))
    if r.returncode != 0:
        sys.exit(f"\nABORT: '{label}' exited {r.returncode}. Nothing after this "
                 f"step ran. Fix the cause, then re-run with --redo for any "
                 f"partially ingested DOI listed above.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", required=True,
                    help="folder containing the NEW .xml files to add")
    ap.add_argument("--redo", default="",
                    help="comma-separated DOIs to delete and re-ingest "
                         "(recovery after a crashed previous add)")
    args = ap.parse_args()

    src = Path(args.xml_dir)
    files = sorted(src.glob("*.xml"))
    if not files:
        sys.exit(f"ABORT: no .xml files in {src}")

    # preflight (before any write): the canonical dirs must exist
    for d in ("elsevier_xml", "tables_xml", "chroma_db"):
        if not (_ROOT / d).is_dir():
            print(f"ERROR: {_ROOT / d} is not a directory. add_papers writes into "
                  f"the existing per-machine index; nothing was changed.")
            sys.exit(1)
    if not (_ROOT / "paper_list2_2025.csv").is_file():
        print(f"ERROR: {_ROOT / 'paper_list2_2025.csv'} not found; index_tables.py "
              f"needs it for titles and would abort after prose was ingested.")
        sys.exit(1)

    redo = {d.strip() for d in args.redo.split(",") if d.strip()}
    missing_redo = sorted(redo - {filename_to_doi(f) for f in files})
    if missing_redo:
        print(f"ERROR: --redo DOI(s) with no file in {src}: {', '.join(missing_redo)}. "
              f"Nothing was deleted.")
        sys.exit(1)

    col = get_collection()
    n_start = col.count()
    print(f"vector store before: {n_start:,} records")

    todo, skipped, bad = [], [], []
    for f in files:
        doi = filename_to_doi(f)
        if not doi.startswith("10."):
            bad.append(f.name)
            continue
        existing = doi_record_count(col, doi)
        if doi in redo:
            if existing:
                col.delete(where={"doi": doi})
                print(f"  redo: deleted {existing} records for {doi}")
            todo.append((f, doi))
        elif existing:
            skipped.append((doi, existing))
        else:
            todo.append((f, doi))

    for name in bad:
        print(f"  WARNING: '{name}' does not look like a DOI-named file — skipped")
    for doi, n in skipped:
        print(f"  already indexed ({n} records), skipping: {doi}")
    if not todo:
        print(f"\nNothing to add. Store unchanged at {col.count():,}.")
        return

    print(f"\nto ingest: {len(todo)} paper(s)")
    staging = Path(tempfile.mkdtemp(prefix=".add_papers_", dir=str(_ROOT)))
    stage_xml = staging / "xml"
    stage_tab = staging / "tables"
    stage_xml.mkdir()
    stage_tab.mkdir()
    for f, _ in todo:
        shutil.copy2(f, stage_xml / f.name)

    # -- 1. prose chunks (canonical script, scoped to staging) --------------
    n0 = col.count()
    run_step([sys.executable, "ingest/ingest_dir.py",
              "--xml-dir", str(stage_xml), "--collection", "papers",
              "--resume"],     # the store is non-empty; ingest_dir refuses without it
             "ingest prose")
    n1 = col.count()
    prose_added = n1 - n0
    print(f"prose chunks added: {prose_added:,} ({n0:,} -> {n1:,})")
    for f, doi in todo:
        c = doi_record_count(col, doi)
        if c == 0:
            sys.exit(f"ABORT: {doi} produced 0 records — XML parse problem in "
                     f"{f.name}? After fixing, re-run with --redo {doi}")
        print(f"  {doi}: {c} records")

    # -- 2. tables: extract per file, then canonical indexer ----------------
    n_tables = 0
    for f, doi in todo:
        r = process_single_xml(stage_xml / f.name, stage_tab)
        n_tables += r.get("tables", 0)
        print(f"  tables in {doi}: {r.get('tables', 0)} ({r['status']})")
    if n_tables:
        run_step([sys.executable, "ingest/index_tables.py", "--all",
                  "--xlsx-dir", str(stage_tab), "--xml-dir", str(stage_xml),
                  "--collection", "papers", "--wos", "paper_list2_2025.csv"],
                 "index tables")
    n2 = col.count()
    tab_added = n2 - n1
    print(f"table records added: {tab_added:,} ({n1:,} -> {n2:,})")
    if n_tables and tab_added == 0:
        sys.exit("ABORT: tables were extracted but 0 table records landed in "
                 "the store. "
                 "This points at an xlsx/DOI filename mismatch in staging; "
                 "inspect the staging dir left behind, then re-run with "
                 "--redo <doi>.")
    if tab_added != n_tables:
        sys.exit(f"ABORT: {n_tables} tables were extracted but {tab_added} table "
                 f"records landed in the store (expected equal). Inspect the "
                 f"staging dir left behind ({staging}), then re-run with "
                 f"--redo for the affected DOIs.")

    # -- 3. make the canonical dirs complete for future FULL rebuilds -------
    for f, doi in todo:
        dst = _ROOT / "elsevier_xml" / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        x = stage_tab / f"{f.stem}.xlsx"
        if x.exists():
            shutil.copy2(x, _ROOT / "tables_xml" / x.name)

    # -- 4. BM25 full rebuild (cheap; avoids partial-index divergence) ------
    run_step([sys.executable, "ingest/build_bm25_index.py",
              "--collection", "papers", "--out", "bm25_index/papers"],
             "rebuild BM25")
    import json
    with open(_ROOT / "bm25_index" / "papers" / "chunks.json",
              encoding="utf-8") as fh:
        bm25_n = len(json.load(fh)["ids"])
    if bm25_n != n2:
        sys.exit(f"ABORT: BM25 has {bm25_n:,} chunks but the store has {n2:,} "
                 f"— rebuild did not complete; re-run build_bm25_index.py")

    shutil.rmtree(staging)

    # -- report -------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{len(files)} XML found | {len(todo)} ingested | {len(skipped)} "
          f"already indexed | {len(bad)} bad names")
    print(f"records: {n_start:,} -> {n2:,}  "
          f"(prose +{prose_added:,}, tables +{tab_added:,})")
    print(f"BM25 rebuilt: {bm25_n:,} chunks")
    print("REMINDERS:")
    print("  * OPTIONAL: append new papers (Authors/Title/DOI) to "
          "paper_list2_2025.csv — titles then show in locate mode and enrich "
          "table-record text (re-run with --redo <doi> to re-embed with title)")
    print("  * run this same command on the OTHER machine (Mac/cluster), or "
          "rsync chroma_db/ bm25_index/ elsevier_xml/ tables_xml/")
    print("=" * 72)


if __name__ == "__main__":
    main()