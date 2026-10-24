#!/usr/bin/env python3
"""
measure_paper_tokens.py — how long are your papers, in tokens?

The single biggest design input for full-paper-feed is: how many tokens is a
typical full paper? That decides which models are candidates (32K vs 128K
context) and how much "lost in the middle" should worry you. This measures
YOUR corpus, not a literature average.
"""

import argparse, glob, os, statistics, sys
from pathlib import Path


def build_token_counter():
    """Prefer the real Qwen2.5 tokenizer; degrade gracefully if offline."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        return "qwen2.5", lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    except Exception as e:
        print(f"  [tokenizer] Qwen2.5 unavailable ({type(e).__name__}); trying tiktoken")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return "tiktoken-cl100k", lambda s: len(enc.encode(s))
    except Exception as e:
        print(f"  [tokenizer] tiktoken unavailable ({type(e).__name__}); using chars/4")
    return "chars/4 heuristic", lambda s: max(1, len(s) // 4)


def _local(name):
    """'ce:para' -> 'para' so we're robust to namespace-prefix variation."""
    return name.split(":")[-1] if name else ""


def extract_texts(xml_path):
    """Return (body_only, refs_text, mode). mode exposes how text was found."""
    from bs4 import BeautifulSoup
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml-xml")

    abstract = "\n".join(
        t.get_text(" ", strip=True)
        for t in soup.find_all(lambda x: getattr(x, "name", None) and _local(x.name) == "abstract")
    )

    sections = soup.find_all(lambda x: getattr(x, "name", None) and _local(x.name) == "sections")
    if sections:
        body = "\n".join(s.get_text(" ", strip=True) for s in sections)
        mode = "structured"
    else:
        raw = soup.find(lambda x: getattr(x, "name", None) and _local(x.name) == "rawtext")
        if raw is not None:
            body, mode = raw.get_text(" ", strip=True), "rawtext"
        else:
            body, mode = soup.get_text(" ", strip=True), "wholedoc_fallback"

    refs = "\n".join(
        b.get_text(" ", strip=True)
        for b in soup.find_all(lambda x: getattr(x, "name", None) and _local(x.name) == "bibliography")
    )
    return (abstract + "\n" + body).strip(), refs, mode


def pct(s, p):
    if not s:
        return 0
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return int(s[lo] + (s[hi] - s[lo]) * (k - lo))


def distribution_block(label, vals):
    if not vals:
        print(f"\n{label}: no data"); return
    s = sorted(vals)
    print(f"\n{label}  (n={len(s)})")
    print(f"  min {s[0]:>7,}   p10 {pct(s,10):>7,}   p25 {pct(s,25):>7,}   "
          f"median {pct(s,50):>7,}   p75 {pct(s,75):>7,}")
    print(f"  p90 {pct(s,90):>7,}   p95 {pct(s,95):>7,}   p99 {pct(s,99):>7,}   "
          f"max {s[-1]:>7,}   mean {int(statistics.mean(s)):>7,}")
    print("  fits in context window:")
    for w in (8000, 16000, 32000, 64000, 128000):
        n = sum(1 for v in s if v <= w)
        print(f"    <= {w:>7,} tok : {n:>5}/{len(s)}  ({n/len(s):6.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", default="elsevier_xml")
    ap.add_argument("--csv", default=None, help="Optional per-paper output")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
    print(f"Found {len(files)} XML files in {args.xml_dir}")
    if not files:
        print("  Nothing to measure. Check the path — are you in the repo root?")
        sys.exit(1)

    name, count_tokens = build_token_counter()
    print(f"  token counting method: {name}")

    body_tokens, full_tokens, rows = [], [], []
    modes = {"structured": 0, "rawtext": 0, "wholedoc_fallback": 0}
    failed = 0

    try:
        from tqdm import tqdm
        it = tqdm(files, desc="Measuring")
    except Exception:
        it = files

    for path in it:
        try:
            body, refs, mode = extract_texts(path)
            if not body.strip():
                failed += 1; continue
            bt = count_tokens(body)
            ft = count_tokens(body + "\n" + refs) if refs else bt
            body_tokens.append(bt); full_tokens.append(ft)
            modes[mode] = modes.get(mode, 0) + 1
            rows.append((Path(path).name, bt, ft, mode))
        except Exception:
            failed += 1

    print("\n=== what was measured ===")
    print(f"  parsed OK           : {len(body_tokens)}")
    print(f"    structured body   : {modes.get('structured',0)}")
    print(f"    rawtext fallback  : {modes.get('rawtext',0)}")
    print(f"    wholedoc fallback : {modes.get('wholedoc_fallback',0)}  (counts suspect)")
    print(f"  empty / failed      : {failed}")

    distribution_block("BODY ONLY (no references) — tokens/paper", body_tokens)
    distribution_block("WITH REFERENCES — tokens/paper", full_tokens)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "tokens_body_only", "tokens_with_refs", "extract_mode"])
            w.writerows(rows)
        print(f"\n  per-paper counts written to {args.csv}")


if __name__ == "__main__":
    main()