#!/usr/bin/env python3
"""
validate_eval_set.py — READ-ONLY structural check of the frozen eval set.

The eval workbook and its gold labels (incl. valid_dois) are a load-bearing wall
(RUNBOOK §8). This script asserts the eval set still has the expected shape, so a
stale copy (e.g. one without valid_dois) or an accidental edit fails loudly instead
of silently changing the measurement. It never writes the file.

Default: the full workbook eval/LCA_RAG_eval_set_v3_50.xlsx (also expected_answer,
source_location, notes). Given a .csv export of the same 50 questions (id, type,
question, source_doi, valid_dois), it runs the checks that apply to those columns.
The eval set is kept in the author's private archive and not published; a missing
file is an ERROR (exit 1).

Run from the repo root:
  python eval/validate_eval_set.py
  python eval/validate_eval_set.py --eval-set path/to/eval_workbook.xlsx

Prints OK lines; on any mismatch prints "ERROR: ..." lines and exits 1.
"""
import argparse
import csv
import os
import re
import sys

import openpyxl

# Expected values: the frozen v3_50 workbook (sheet `Questions`) and
# Prof. Lam's valid_dois reply of 2026-07-16. Change only with an approved proposal.
SHEET = "Questions"
COLUMNS = ["id", "question", "expected_answer", "source_doi", "source_location",
           "type", "notes", "valid_dois"]
CSV_COLUMNS = ["id", "type", "question", "source_doi", "valid_dois"]   # csv export of the same 50
N_ROWS = 50
TASK2_IDS = set(range(1, 39))                      # 38 TASK2: ids 1-38
TASK1_IDS = set(range(39, 51))                     # 12 TASK1: ids 39-50
BLANK_LOCATION_IDS = set(range(31, 39)) | TASK1_IDS  # TASK2 31-38 and all TASK1
VALID_DOIS_IDS = {1, 11, 14, 16, 20, 21, 28, 31, 33, 34, 35, 38}
N_VALID_PAIRS = 22
N_VALID_DISTINCT = 21
# same message as step3_orchestrate.EVAL_SET_MISSING (this script does not import step3)
EVAL_SET_MISSING = ("ERROR: eval set not found: {}. The project's 50-question eval set is not published; "
                    "pass --eval-set with your own .xlsx (sheet \"Questions\") or .csv with columns "
                    "id,type,question,source_doi (optional: valid_dois, expected_answer, source_location).")


def blank(v):
    return v is None or str(v).strip() == ""


def main():
    ap = argparse.ArgumentParser(description="Read-only check of the v3_50 eval set (.csv or .xlsx).")
    ap.add_argument("--eval-set", default="eval/LCA_RAG_eval_set_v3_50.xlsx")
    args = ap.parse_args()
    if not os.path.exists(args.eval_set):
        print(EVAL_SET_MISSING.format(args.eval_set))
        sys.exit(1)
    is_csv = args.eval_set.lower().endswith(".csv")
    columns = CSV_COLUMNS if is_csv else COLUMNS

    errors = []

    def check(ok, ok_msg, err_msg):
        if ok:
            print(f"OK: {ok_msg}")
        else:
            errors.append(err_msg)
            print(f"ERROR: {err_msg}")
        return ok

    if is_csv:
        try:
            with open(args.eval_set, newline="", encoding="utf-8-sig") as f:
                rows = [tuple(r) for r in csv.reader(f)]
        except Exception as e:
            print(f"ERROR: could not open {args.eval_set}: {e}")
            sys.exit(1)
        print(f"eval set: {args.eval_set}")
        if not rows:
            print("ERROR: csv is empty")
            sys.exit(1)
    else:
        try:
            wb = openpyxl.load_workbook(args.eval_set, read_only=True, data_only=True)
        except Exception as e:
            print(f"ERROR: could not open {args.eval_set}: {e}")
            sys.exit(1)
        print(f"eval set: {args.eval_set}")

        if not check(SHEET in wb.sheetnames, f"sheet '{SHEET}' present",
                     f"no sheet '{SHEET}' (sheets: {wb.sheetnames})"):
            sys.exit(1)
        rows = list(wb[SHEET].iter_rows(values_only=True))
        wb.close()
        if not rows:
            print(f"ERROR: sheet '{SHEET}' is empty")
            sys.exit(1)

    hdr = [("" if h is None else str(h).strip().lower()) for h in rows[0]]
    while hdr and hdr[-1] == "":
        hdr.pop()
    if not check(hdr == columns, f"columns = {columns}",
                 f"columns {hdr} != expected {columns}"):
        print("ERROR: column mismatch; later checks skipped")
        sys.exit(1)

    data = [dict(zip(columns, r)) for r in rows[1:]
            if r is not None and not all(blank(v) for v in r[:len(columns)])]
    check(len(data) == N_ROWS, f"{N_ROWS} question rows", f"{len(data)} question rows, expected {N_ROWS}")

    # ids: unique ints 1..N_ROWS
    ids = []
    for d in data:
        v = d["id"]
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if is_csv and isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())                     # csv cells are text
        if not isinstance(v, int) or isinstance(v, bool):
            errors.append(f"id {d['id']!r} is not an integer")
            print(f"ERROR: id {d['id']!r} is not an integer")
            continue
        ids.append(v)
        d["id"] = v
    check(len(ids) == len(set(ids)), "ids unique",
          f"duplicate ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    check(set(ids) == set(range(1, N_ROWS + 1)), f"ids = 1..{N_ROWS}",
          f"ids differ from 1..{N_ROWS}: missing {sorted(set(range(1, N_ROWS + 1)) - set(ids))}, "
          f"extra {sorted(set(ids) - set(range(1, N_ROWS + 1)))}")
    by_id = {d["id"]: d for d in data if isinstance(d["id"], int)}

    # types
    t2 = {i for i, d in by_id.items() if str(d["type"] or "").strip() == "TASK2"}
    t1 = {i for i, d in by_id.items() if str(d["type"] or "").strip() == "TASK1"}
    other = sorted(set(by_id) - t1 - t2)
    check(t2 == TASK2_IDS, f"{len(TASK2_IDS)} TASK2 (ids 1-38)",
          f"TASK2 ids differ: missing {sorted(TASK2_IDS - t2)}, extra {sorted(t2 - TASK2_IDS)}")
    check(t1 == TASK1_IDS, f"{len(TASK1_IDS)} TASK1 (ids 39-50)",
          f"TASK1 ids differ: missing {sorted(TASK1_IDS - t1)}, extra {sorted(t1 - TASK1_IDS)}")
    check(not other, "no other type values",
          f"ids with type not TASK1/TASK2: {other}")

    # required cells
    no_q = sorted(i for i, d in by_id.items() if blank(d["question"]))
    no_doi = sorted(i for i, d in by_id.items() if blank(d["source_doi"]))
    check(not no_q, "every row has a question", f"rows without question: {no_q}")
    check(not no_doi, "every row has a source_doi", f"rows without source_doi: {no_doi}")

    # source_location blank exactly for TASK2 31-38 and all TASK1 (workbook only; not in the csv)
    if not is_csv:
        blank_loc = {i for i, d in by_id.items() if blank(d["source_location"])}
        check(blank_loc == BLANK_LOCATION_IDS,
              f"source_location blank exactly for ids 31-50 ({len(BLANK_LOCATION_IDS)} rows)",
              f"source_location blank set differs: unexpectedly blank {sorted(blank_loc - BLANK_LOCATION_IDS)}, "
              f"unexpectedly filled {sorted(BLANK_LOCATION_IDS - blank_loc)}")

    # valid_dois
    with_valid = {i for i, d in by_id.items() if not blank(d["valid_dois"])}
    check(with_valid == VALID_DOIS_IDS,
          f"valid_dois filled exactly on qids {sorted(VALID_DOIS_IDS)}",
          f"valid_dois qids differ: missing {sorted(VALID_DOIS_IDS - with_valid)}, "
          f"extra {sorted(with_valid - VALID_DOIS_IDS)}")
    pairs = []
    for i in sorted(with_valid):
        for doi in re.split(r"[;,]", str(by_id[i]["valid_dois"])):
            if doi.strip():
                pairs.append((i, doi.strip()))
    distinct = {doi.lower() for _, doi in pairs}
    bad = [(i, doi) for i, doi in pairs if not doi.startswith("10.")]
    check(len(pairs) == N_VALID_PAIRS, f"{N_VALID_PAIRS} (qid, DOI) valid_dois pairs",
          f"{len(pairs)} valid_dois pairs, expected {N_VALID_PAIRS}")
    check(len(distinct) == N_VALID_DISTINCT, f"{N_VALID_DISTINCT} distinct valid_dois DOIs",
          f"{len(distinct)} distinct valid_dois DOIs, expected {N_VALID_DISTINCT}")
    check(not bad, "every valid_dois DOI starts with '10.'",
          f"valid_dois entries not starting with '10.': {bad}")

    if errors:
        print(f"\nERROR: {len(errors)} check(s) failed for {args.eval_set}")
        sys.exit(1)
    print(f"\nOK: {args.eval_set} matches the frozen eval-set shape")


if __name__ == "__main__":
    main()
