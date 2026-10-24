#!/usr/bin/env python3
"""compare_judges.py — score one or more judged_*.xlsx files against human labels.

Agreement convention: Y<->MATCH and N<->no count; PARTIAL never counts as agreement.
A (qid, doi) label with no row (or an empty judge_match) in a judged file is reported
as "missing in judged file" and excluded from agreement (not counted as agreement or disagreement).
First --judged file is the flip baseline; flips and the watchlist are keyed by (qid, doi).

Needs the expert-labelled answers workbook (a `valid?` column). It and the judged files
of past runs are kept in the author's private archive (not published: they contain
verbatim paper text or licensed data).

Run from the repo root:
  python eval/compare_judges.py --labels <expert review xlsx> \
      --judged eval_runs/judged_expected_..._OLDPROMPT.xlsx eval_runs/..._25NEW.xlsx eval_runs/..._35NEW.xlsx
"""
import argparse
import os
import sys
from collections import Counter
import openpyxl


def load(path, vcol):
    ws = openpyxl.load_workbook(path).active
    rows = list(ws.iter_rows(values_only=True))
    hdr = list(rows[0])
    out = {}
    for r in rows[1:]:
        if not r or r[0] is None:
            continue
        d = dict(zip(hdr, r))
        out[(str(d["qid"]), str(d["doi"]))] = str(d.get(vcol) or "").strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="xlsx with human valid? column")
    ap.add_argument("--judged", nargs="+", required=True)
    ap.add_argument("--watch", default="5,6,10,12,28")
    args = ap.parse_args()

    if not args.labels.lower().endswith((".xlsx", ".xlsm")) or not os.path.exists(args.labels):
        sys.exit("ERROR: this needs the expert-labelled answers workbook (valid? column), "
                 f"which is not published. Got --labels {args.labels}.")
    for f in args.judged:
        if not os.path.exists(f):
            sys.exit(f"ERROR: judged file not found: {f}")
    lam = load(args.labels, "valid?")
    if not lam or not any(lam.values()):
        sys.exit(f"ERROR: {args.labels} has no filled `valid?` labels; this needs the "
                 "expert-labelled answers workbook, which is not published.")
    keys = sorted(lam, key=lambda k: int(k[0]))
    print(f"labels: {len(keys)} rows | {dict(Counter(lam[k].upper() for k in keys))}")

    verdicts = {}
    for f in args.judged:
        jv = load(f, "judge_match")
        name = f.split("/")[-1].replace("judged_expected_", "").replace(".xlsx", "")
        row, ct, fp, fn, errs, nmatch, missing = {}, Counter(), [], [], [], 0, []
        for k in keys:
            L = lam[k].upper()
            if k not in jv or not str(jv[k]).strip():
                row[k] = "MISSING"
                missing.append(k[0])
                continue
            m = jv[k].upper()
            m = "MATCH" if m.startswith("MATCH") else ("ERR" if m.startswith("ERR") else "no")
            row[k] = m
            if m == "MATCH":
                nmatch += 1
            if m == "ERR":
                errs.append(k[0])
                continue
            ct[(L, m)] += 1
            if L == "N" and m == "MATCH":
                fp.append(k[0])
            if L == "Y" and m == "no":
                fn.append(k[0])
        verdicts[name] = row
        agree = ct[("Y", "MATCH")] + ct[("N", "no")]
        n_lab = len(keys) - len(missing)
        print(f"\n{name}\n  judge-MATCH {nmatch}/{n_lab} | agreement with labels {agree}/{n_lab}")
        print(f"  missing in judged file: {len(missing)}" + (f" {missing}" if missing else ""))
        print(f"  FP (judge MATCH, human N): {fp}")
        print(f"  FN (judge no,   human Y): {fn}")
        if errs:
            print(f"  ERRORS: {errs}")

    names = list(verdicts)
    base = verdicts[names[0]]
    n_per_q = Counter(k[0] for k in keys)
    label = lambda k: f"Q{k[0]}" if n_per_q[k[0]] == 1 else f"Q{k[0]} {k[1]}"   # doi only if qid repeats
    print(f"\nFLIPS vs {names[0]}:")
    for n in names[1:]:
        fl = [(k, base[k], verdicts[n][k]) for k in keys if verdicts[n][k] != base[k]]
        s = ", ".join(f"{label(k)} {a}->{b}" for k, a, b in fl)
        print(f"  {n}: {s or 'none'}")

    print("\nwatchlist:")
    for q in args.watch.split(","):
        q = q.strip()
        wkeys = [k for k in keys if k[0] == q]
        if not wkeys:
            cells = " | ".join(f"{n}=?" for n in names)
            print(f"  Q{q}: human=? | {cells}")
        for k in wkeys:
            cells = " | ".join(f"{n}={verdicts[n][k]}" for n in names)
            print(f"  {label(k)}: human={lam[k]} | {cells}")


if __name__ == "__main__":
    main()