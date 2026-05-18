# code to extract text from pdf files
# that will be fed to the model + store raw text and a token count estimate

# it also matche the pdf file with the corresponding row in the xlsx file to get the label

"""
prepare_corpus.py  —  Steps 1 & 2
==================================
Step 1: Load ground-truth XLSX, extract titles from every PDF in
        data/papers/, and match each PDF to a ground-truth row using
        multi-strategy fuzzy matching with a confidence score.

Step 2: Extract full text from each matched (and unmatched) PDF using
        PyMuPDF, compute token estimates, and save everything to a
        single structured JSON file ready for the inference pipeline.

Output
------
data/corpus.json          — main structured output (one record per PDF)
data/corpus_review.csv    — human-readable match report for manual review
                            (flag any match below MATCH_THRESHOLD here)
"""

import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAPERS_DIR      = Path("data/papers")
GROUND_TRUTH    = Path("data/ground_truth.xlsx")   # adjust filename as needed
OUTPUT_JSON     = Path("data/corpus.json")
OUTPUT_REVIEW   = Path("data/corpus_review.csv")

# Column names in the XLSX  (adjust if yours differ)
GT_COL_CHECKED  = "Checked"
GT_COL_AUTHOR   = "Lead author"
GT_COL_YEAR     = "Year"
GT_COL_TITLE    = "Title"

# Fuzzy match threshold (0–100).  Matches below this are flagged as
# UNMATCHED and written to the review CSV for manual inspection.
MATCH_THRESHOLD = 80

# Approximate tokens-per-character for academic English.
# 1 token ≈ 4 chars is the standard rule of thumb.
CHARS_PER_TOKEN = 4

# Maximum number of characters to keep from the first page when
# heuristically extracting the paper title.
TITLE_EXTRACTION_CHARS = 600

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lower-case, strip accents, collapse whitespace, remove punctuation."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Step 1a  —  Extract title candidate from a PDF
# ---------------------------------------------------------------------------

def extract_title_from_pdf(pdf_path: Path) -> str:
    """
    Multi-strategy title extraction.  Returns the best candidate string,
    or an empty string if nothing usable is found.

    Strategy order (first non-empty result wins):
      1. PDF metadata /Title field  (most reliable when present)
      2. Largest-font text on page 1  (common for typeset journals)
      3. First non-trivial line(s) of page 1 text  (fallback)
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"  [WARN] Cannot open {pdf_path.name}: {exc}")
        return ""

    # --- Strategy 1: PDF metadata ---
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if meta_title and len(meta_title) > 10 and not meta_title.lower().startswith("microsoft"):
        doc.close()
        return meta_title

    if not doc.page_count:
        doc.close()
        return ""

    page = doc[0]

    # --- Strategy 2: Largest-font span on page 1 ---
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        spans = []
        for block in blocks:
            if block.get("type") != 0:        # text blocks only
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if len(txt) > 8 and size > 0:
                        spans.append((size, txt))
        if spans:
            # Sort by font size descending; collect spans that are
            # within 1 pt of the largest — they form the title line(s).
            spans.sort(key=lambda x: -x[0])
            max_size = spans[0][0]
            title_parts = [s for s in spans if abs(s[0] - max_size) <= 1.5]
            # Deduplicate while preserving order
            seen = set()
            unique_parts = []
            for _, t in title_parts:
                if t not in seen:
                    seen.add(t)
                    unique_parts.append(t)
            candidate = " ".join(unique_parts).strip()
            if len(candidate) > 15:
                doc.close()
                return candidate
    except Exception:
        pass  # fall through to strategy 3

    # --- Strategy 3: First meaningful text lines from page 1 ---
    raw = page.get_text("text")
    doc.close()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    # Skip very short lines (page numbers, journal name headers, etc.)
    meaningful = [l for l in lines if len(l) > 20]
    if meaningful:
        # Take up to 3 lines  (titles occasionally span lines)
        candidate = " ".join(meaningful[:3])
        return candidate[:TITLE_EXTRACTION_CHARS]

    return ""


# ---------------------------------------------------------------------------
# Step 1b  —  Match extracted titles to ground-truth rows
# ---------------------------------------------------------------------------

def match_to_ground_truth(
    extracted_title: str,
    gt_df: pd.DataFrame,
    pdf_name: str,
) -> dict:
    """
    Returns a dict with keys:
        gt_index      : int | None   — row index in gt_df
        gt_title      : str | None
        gt_label      : str | None   — 'Y' or 'N'
        gt_year       : int | None
        gt_author     : str | None
        match_score   : float        — 0–100
        match_status  : str          — 'MATCHED' | 'UNMATCHED' | 'NO_TITLE'
        match_strategy: str
    """
    empty = dict(
        gt_index=None, gt_title=None, gt_label=None,
        gt_year=None, gt_author=None,
        match_score=0.0, match_status="NO_TITLE", match_strategy="none",
    )

    if not extracted_title or len(extracted_title.strip()) < 8:
        return empty

    norm_extracted = normalise(extracted_title)
    gt_titles_norm = [normalise(t) for t in gt_df[GT_COL_TITLE].fillna("")]

    # --- Primary: token_set_ratio (robust to word-order differences) ---
    result = process.extractOne(
        norm_extracted,
        gt_titles_norm,
        scorer=fuzz.token_set_ratio,
        score_cutoff=0,
    )
    best_score_tsr = result[1] if result else 0
    best_idx_tsr   = result[2] if result else None

    # --- Secondary: partial_ratio (good when one title is a substring) ---
    result2 = process.extractOne(
        norm_extracted,
        gt_titles_norm,
        scorer=fuzz.partial_ratio,
        score_cutoff=0,
    )
    best_score_pr = result2[1] if result2 else 0
    best_idx_pr   = result2[2] if result2 else None

    # Pick whichever strategy scored higher
    if best_score_tsr >= best_score_pr:
        best_score  = best_score_tsr
        best_idx    = best_idx_tsr
        strategy    = "token_set_ratio"
    else:
        best_score  = best_score_pr
        best_idx    = best_idx_pr
        strategy    = "partial_ratio"

    if best_idx is None or best_score < MATCH_THRESHOLD:
        return {**empty,
                "match_score": round(best_score, 2),
                "match_status": "UNMATCHED",
                "match_strategy": strategy}

    row = gt_df.iloc[best_idx]
    return dict(
        gt_index      = int(best_idx),
        gt_title      = str(row[GT_COL_TITLE]),
        gt_label      = str(row[GT_COL_CHECKED]).strip().upper(),
        gt_year       = int(row[GT_COL_YEAR]) if pd.notna(row.get(GT_COL_YEAR)) else None,
        gt_author     = str(row[GT_COL_AUTHOR]) if pd.notna(row.get(GT_COL_AUTHOR)) else None,
        match_score   = round(float(best_score), 2),
        match_status  = "MATCHED",
        match_strategy= strategy,
    )


# ---------------------------------------------------------------------------
# Step 2  —  Full text extraction
# ---------------------------------------------------------------------------

def extract_full_text(pdf_path: Path) -> tuple[str, int, int, list[str]]:
    """
    Extract all text from a PDF using PyMuPDF.
    Returns (full_text, page_count, estimated_tokens, warnings).

    Uses pdfplumber as a fallback for pages where PyMuPDF returns empty
    text (some journal PDFs have quirky encoding on specific pages).
    """
    import pdfplumber  # imported here so the module is optional

    warnings = []
    pages_text = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        return "", 0, 0, [f"Cannot open: {exc}"]

    page_count = doc.page_count
    empty_pages_fitz = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages_text.append(text)
        else:
            empty_pages_fitz.append(i)
            pages_text.append(None)   # placeholder
    doc.close()

    # Fallback: pdfplumber for pages that came up empty under fitz
    if empty_pages_fitz:
        warnings.append(
            f"PyMuPDF returned empty text on {len(empty_pages_fitz)} page(s) "
            f"({empty_pages_fitz[:5]}{'...' if len(empty_pages_fitz)>5 else ''}); "
            "retrying with pdfplumber."
        )
        try:
            with pdfplumber.open(str(pdf_path)) as plumb:
                for idx in empty_pages_fitz:
                    if idx < len(plumb.pages):
                        fallback = (plumb.pages[idx].extract_text() or "").strip()
                        pages_text[idx] = fallback if fallback else ""
        except Exception as exc:
            warnings.append(f"pdfplumber fallback failed: {exc}")
            for idx in empty_pages_fitz:
                if pages_text[idx] is None:
                    pages_text[idx] = ""

    # Replace any remaining None (shouldn't happen, but be safe)
    pages_text = [t if t is not None else "" for t in pages_text]

    full_text = "\n\n".join(t for t in pages_text if t)
    est_tokens = estimate_tokens(full_text)

    # Warn if the paper looks suspiciously short after extraction
    if est_tokens < 500:
        warnings.append(
            f"Very short extraction: only ~{est_tokens} tokens total — "
            "may be a scanned/image-only PDF."
        )

    return full_text, page_count, est_tokens, warnings


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # --- Validate paths ---
    if not PAPERS_DIR.exists():
        sys.exit(f"ERROR: Papers directory not found: {PAPERS_DIR}")
    if not GROUND_TRUTH.exists():
        sys.exit(f"ERROR: Ground truth file not found: {GROUND_TRUTH}")

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"ERROR: No PDF files found in {PAPERS_DIR}")

    print(f"Found {len(pdf_files)} PDF(s) in {PAPERS_DIR}")

    # --- Load ground truth ---
    print(f"Loading ground truth: {GROUND_TRUTH}")
    try:
        gt_df = pd.read_excel(GROUND_TRUTH, dtype=str)
    except Exception as exc:
        sys.exit(f"ERROR: Cannot read XLSX: {exc}")

    required_cols = [GT_COL_CHECKED, GT_COL_AUTHOR, GT_COL_YEAR, GT_COL_TITLE]
    missing = [c for c in required_cols if c not in gt_df.columns]
    if missing:
        sys.exit(
            f"ERROR: XLSX is missing columns: {missing}\n"
            f"Found columns: {list(gt_df.columns)}"
        )

    print(f"Ground truth rows: {len(gt_df)}")
    print(f"  Y (relevant): {(gt_df[GT_COL_CHECKED].str.upper()=='Y').sum()}")
    print(f"  N (not relevant): {(gt_df[GT_COL_CHECKED].str.upper()=='N').sum()}")

    # --- Process each PDF ---
    records = []
    review_rows = []

    for pdf_path in pdf_files:
        print(f"\n--- {pdf_path.name} ---")

        # Step 1: extract title + match
        print("  [1] Extracting title...")
        extracted_title = extract_title_from_pdf(pdf_path)
        print(f"      Extracted: '{extracted_title[:80]}{'...' if len(extracted_title)>80 else ''}'")

        print("  [1] Matching to ground truth...")
        match = match_to_ground_truth(extracted_title, gt_df, pdf_path.name)
        print(
            f"      Status: {match['match_status']}  "
            f"Score: {match['match_score']:.1f}  "
            f"Strategy: {match['match_strategy']}"
        )
        if match["match_status"] == "MATCHED":
            print(f"      GT title: '{str(match['gt_title'])[:80]}'")
            print(f"      GT label: {match['gt_label']}  |  Author: {match['gt_author']}  |  Year: {match['gt_year']}")

        # Step 2: full text extraction
        print("  [2] Extracting full text...")
        full_text, page_count, est_tokens, warnings = extract_full_text(pdf_path)
        for w in warnings:
            print(f"      [WARN] {w}")
        print(f"      Pages: {page_count}  |  Est. tokens: {est_tokens:,}")

        record = {
            # --- file info ---
            "pdf_filename"      : pdf_path.name,
            "pdf_path"          : str(pdf_path),
            # --- title extraction ---
            "extracted_title"   : extracted_title,
            # --- ground truth match ---
            "match_status"      : match["match_status"],
            "match_score"       : match["match_score"],
            "match_strategy"    : match["match_strategy"],
            "gt_index"          : match["gt_index"],
            "gt_title"          : match["gt_title"],
            "gt_label"          : match["gt_label"],       # 'Y', 'N', or None
            "gt_year"           : match["gt_year"],
            "gt_author"         : match["gt_author"],
            # --- text extraction ---
            "page_count"        : page_count,
            "estimated_tokens"  : est_tokens,
            "extraction_warnings": warnings,
            "full_text"         : full_text,
        }
        records.append(record)

        review_rows.append({
            "pdf_filename"    : pdf_path.name,
            "extracted_title" : extracted_title[:120],
            "match_status"    : match["match_status"],
            "match_score"     : match["match_score"],
            "match_strategy"  : match["match_strategy"],
            "gt_title"        : match["gt_title"],
            "gt_label"        : match["gt_label"],
            "gt_year"         : match["gt_year"],
            "gt_author"       : match["gt_author"],
            "page_count"      : page_count,
            "estimated_tokens": est_tokens,
            "warnings"        : "; ".join(warnings),
        })

    # --- Save outputs ---
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving corpus JSON → {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at"    : datetime.now().isoformat(),
                "papers_dir"      : str(PAPERS_DIR),
                "ground_truth_file": str(GROUND_TRUTH),
                "match_threshold" : MATCH_THRESHOLD,
                "total_pdfs"      : len(records),
                "matched"         : sum(1 for r in records if r["match_status"] == "MATCHED"),
                "unmatched"       : sum(1 for r in records if r["match_status"] == "UNMATCHED"),
                "no_title"        : sum(1 for r in records if r["match_status"] == "NO_TITLE"),
                "records"         : records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saving review CSV  → {OUTPUT_REVIEW}")
    review_df = pd.DataFrame(review_rows)
    review_df = review_df.sort_values("match_score", ascending=True)  # worst matches first
    review_df.to_csv(OUTPUT_REVIEW, index=False)

    # --- Summary ---
    matched   = sum(1 for r in records if r["match_status"] == "MATCHED")
    unmatched = sum(1 for r in records if r["match_status"] == "UNMATCHED")
    no_title  = sum(1 for r in records if r["match_status"] == "NO_TITLE")
    token_counts = [r["estimated_tokens"] for r in records if r["estimated_tokens"] > 0]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total PDFs processed : {len(records)}")
    print(f"  MATCHED              : {matched}")
    print(f"  UNMATCHED            : {unmatched}  ← review these in {OUTPUT_REVIEW}")
    print(f"  NO_TITLE             : {no_title}   ← title extraction failed")
    if token_counts:
        print(f"  Token estimates      : min={min(token_counts):,}  "
              f"median={sorted(token_counts)[len(token_counts)//2]:,}  "
              f"max={max(token_counts):,}")
        over_32k = sum(1 for t in token_counts if t > 32_000)
        if over_32k:
            print(f"  PDFs > 32K tokens    : {over_32k}  "
                  "← Qwen2.5-7B will need truncation for these")
    print("=" * 60)

    if unmatched + no_title > 0:
        print(
            "\n[ACTION REQUIRED] Some PDFs could not be matched automatically.\n"
            f"Open {OUTPUT_REVIEW} (sorted by match_score ascending) and:\n"
            "  1. Verify or correct the 'gt_title' column manually.\n"
            "  2. The inference script in step 3 will skip unmatched PDFs\n"
            "     unless you patch corpus.json with the correct gt_index.\n"
            "  Alternatively, set MATCH_THRESHOLD lower (currently "
            f"{MATCH_THRESHOLD}) if matches look correct but scored just below."
        )


if __name__ == "__main__":
    main()