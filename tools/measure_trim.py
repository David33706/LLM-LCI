#!/usr/bin/env python3
"""
measure_trim.py — validate section trimming BEFORE trusting it in an eval run.

For each gold paper in the eval set:
  - tokens full vs. trimmed, and whether the trimmed feed fits the reader budget
    (ctx - 1500 - 500 = 129,072 at the adopted 131,072 ctx; --ctx to change)
  - which section titles the trimmer DROPPED
  - whether the question's source_location still matches a KEPT section title
    (if it matches only a DROPPED one -> "LOCATION-AT-RISK" = the keep-list is
     wrong for that paper; fix it before running the eval; if it matches NO
     section title, e.g. "Table 3" -> "UNMATCHED (not verified)")

The gold-location check needs source_location, which only the full eval workbook
has (kept in the author's private archive; not published: it contains verbatim
paper text or licensed data). On a questions file without source_location (e.g. your
own .csv) the token measurement runs and the location check is reported as NOT CHECKED.

No GPU, no server needed. Run from the repo root:
  python tools/measure_trim.py
"""

import argparse
import os
import re
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.elsevier import extract_structured_sections
# reuse the EXACT same rules Step 3 applies — one source of truth
from step3_orchestrate import (keep_section, count_tokens, doi_to_filename, norm_doi,
                                render_tables_text, build_parser, load_questions)

# same math as step3_orchestrate: budget = ctx - max_tokens - 500, from step3's own defaults
_S3_DEFAULTS = build_parser().parse_args([])


def location_tokens(loc: str) -> set:
    """'P3 2.2.1.2. System boundary' -> {'system', 'boundary'} (strip page/numbering)."""
    loc = re.sub(r"\bP\d+\b", " ", str(loc or ""))
    loc = re.sub(r"\b[\d.]+\b", " ", loc)
    return {w.lower() for w in re.findall(r"[A-Za-z]{3,}", loc)}


def title_matches(loc_toks: set, titles) -> bool:
    for t in titles:
        t_toks = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", t or "")}
        # subset, or all but one location token present (old line parsed as loc <= (t | loc): always True)
        if loc_toks and (loc_toks <= t_toks or len(loc_toks & t_toks) >= max(1, len(loc_toks) - 1)):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval/LCA_RAG_eval_set_v3_50.xlsx")
    ap.add_argument("--xml-dir", default="elsevier_xml")
    ap.add_argument("--tables", action="store_true",
                    help="also measure the feed WITH rendered table content appended")
    ap.add_argument("--ctx", type=int, default=_S3_DEFAULTS.ctx,
                    help="reader max-model-len (default: step3's adopted %(default)s)")
    args = ap.parse_args()
    budget = args.ctx - _S3_DEFAULTS.max_tokens - 500

    questions = load_questions(args.eval_set)   # .csv or .xlsx, same loader as step3
    has_locations = any(q.get("source_location") for q in questions)

    # unique gold papers (several questions can share one)
    papers = {}
    for q in questions:
        d = str(q["source_doi"]).strip()
        papers.setdefault(norm_doi(d), {"doi": d, "locations": []})
        if q.get("source_location"):
            papers[norm_doi(d)]["locations"].append(
                (q.get("id"), str(q["source_location"])))

    print(f"{len(questions)} questions -> {len(papers)} unique gold papers | budget {budget} tok\n")
    if args.tables:
        print(f"{'DOI':<38}{'full':>8}{'trim':>8}{'+tab':>8}{'fits':>6}{'ntab':>6}  dropped sections")
    else:
        print(f"{'DOI':<38}{'full':>8}{'trim':>8}{'saved':>7}{'fits':>6}  dropped sections")
    print("-" * 110)

    fulls, trims, over, missing, at_risk = [], [], 0, 0, []
    unmatched, n_loc_unchecked = [], 0   # locations matching NO section title / on missing XMLs
    for key, info in papers.items():
        p = Path(args.xml_dir) / doi_to_filename(info["doi"])
        if not p.exists():
            print(f"{info['doi']:<38}{'XML MISSING':>8}")
            missing += 1
            n_loc_unchecked += len(info["locations"])
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        sections = extract_structured_sections(raw)

        kept_titles, dropped_titles, kept_text, full_text = [], [], [], []
        for s in sections:
            t = (s.get("text") or "").strip()
            if not t:
                continue
            full_text.append(t)
            if keep_section(s):
                kept_titles.append(s.get("section_title", ""))
                kept_text.append(t)
            else:
                dropped_titles.append(s.get("section_title", ""))

        n_full = count_tokens("\n\n".join(full_text))
        n_trim = count_tokens("\n\n".join(kept_text))
        drop_str = "; ".join(dropped_titles[:5]) + (" ..." if len(dropped_titles) > 5 else "")
        if args.tables:
            tbl = render_tables_text(p)
            ntab = (tbl.count("\n## ") + 1) if tbl else 0
            n_feed = count_tokens("\n\n".join(kept_text) + ("\n\n# TABLES\n\n" + tbl if tbl else ""))
        else:
            ntab, n_feed = 0, n_trim
        fulls.append(n_full); trims.append(n_feed)
        fits = n_feed <= budget
        if not fits:
            over += 1
        if args.tables:
            print(f"{info['doi']:<38}{n_full:>8}{n_trim:>8}{n_feed:>8}{'Y' if fits else 'NO':>6}{ntab:>6}  {drop_str}")
        else:
            saved = f"-{(1 - n_trim / n_full):.0%}" if n_full else "-"
            print(f"{info['doi']:<38}{n_full:>8}{n_trim:>8}{saved:>7}{'Y' if fits else 'NO':>6}  {drop_str}")

        # gold-location survival: does each source_location still match a KEPT title?
        for qid, loc in info["locations"]:
            toks = location_tokens(loc)
            if not toks:
                unmatched.append((qid, info["doi"], loc))
                continue
            in_kept = title_matches(toks, kept_titles)
            in_dropped = title_matches(toks, dropped_titles)
            if in_dropped and not in_kept:
                at_risk.append((qid, info["doi"], loc))
            elif not in_kept and not in_dropped:
                unmatched.append((qid, info["doi"], loc))

    print("-" * 110)
    if not fulls:
        print(f"ERROR: no gold XMLs found in {args.xml_dir} ({missing} missing; "
              "corpus not on this machine? see RUNBOOK §3c/§3b)")
        sys.exit(1)
    if fulls:
        fulls.sort(); trims.sort()
        med = lambda a: a[len(a) // 2]
        print(f"\nSUMMARY  ({len(fulls)} papers measured, {missing} XML missing)")
        lbl = "trimmed+tables" if args.tables else "trimmed"
        print(f"  median tokens : full {med(fulls):>6}  ->  {lbl} {med(trims):>6}")
        print(f"  max tokens    : full {fulls[-1]:>6}  ->  {lbl} {trims[-1]:>6}")
        print(f"  over budget after trim ({budget}): {over}/{len(fulls)}")
    if at_risk:
        print(f"\n  *** LOCATION-AT-RISK ({len(at_risk)}): gold location matches only a DROPPED section ***")
        for qid, doi, loc in at_risk:
            print(f"    Q{qid}: {doi} -> '{loc}'")
        print("  -> extend KEEP_PATTERN in step3_orchestrate.py with the relevant keyword, re-run this script.")
    if unmatched:
        print(f"\n  UNMATCHED (not verified) ({len(unmatched)}): gold location matches NO section title "
              f"(e.g. a table label), so trimming was not checked for it:")
        for qid, doi, loc in unmatched:
            print(f"    Q{qid}: {doi} -> '{loc}'  UNMATCHED (not verified)")
    if n_loc_unchecked:
        print(f"\n  NOT CHECKED: {n_loc_unchecked} gold location(s) on papers whose XML is missing.")
    if not has_locations:
        print("\n  gold-location check: NOT CHECKED — this eval set has no source_location "
              "(only the full eval workbook has it; it is not published).")
    elif not at_risk and not unmatched and not n_loc_unchecked:
        print("\n  gold-location check: OK — every source_location matches a kept section (or has none listed).")
    elif not at_risk:
        print("\n  gold-location check: no location matches only a DROPPED section, but "
              f"{len(unmatched) + n_loc_unchecked} location(s) were not verified (see above).")


if __name__ == "__main__":
    main()