"""
tables/xml_tables.py
--------------------
Extract structured tables from Elsevier XML files (CALS format).

Usage:
    # Extract from all XMLs, save to tables_xml/
    python -m tables.xml_tables --input-dir elsevier_xml --output-dir tables_xml

    # Single file test
    python -m tables.xml_tables --input-file elsevier_xml/10.1016_j.afres.2025.100716.xml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup, FeatureNotFound, Tag


# ── XML Parsing ────────────────────────────────────────────────

def _parse_xml(content: str) -> BeautifulSoup:
    """Parse XML with best available parser."""
    for parser in ["lxml-xml", "xml"]:
        try:
            return BeautifulSoup(content, parser)
        except FeatureNotFound:
            continue
    # Fallback to lxml html parser (handles most XML)
    return BeautifulSoup(content, "lxml")


def _cell_text(entry: Tag) -> str:
    """
    Extract text from a table cell, handling scientific notation markup.

    Handles:
      <sup loc="post">-1</sup>  → ^{-1}
      <ce:inf loc="post">2</ce:inf>  → ₂ (or _2 for plain text)
      <italic>H</italic>  → H
    """
    if entry is None:
        return ""

    # Process inline markup before extracting text
    # Handle superscripts: <sup>x</sup> → ^{x}
    for sup in entry.find_all("sup"):
        sup.replace_with(f"^{{{sup.get_text()}}}")

    # Handle subscripts: <ce:inf> or <inf> → _{x}
    for inf in entry.find_all(["ce:inf", "inf"]):
        inf.replace_with(f"_{{{inf.get_text()}}}")

    # Handle cross-references (just extract the text)
    for xref in entry.find_all(["ce:cross-ref", "cross-ref"]):
        xref.replace_with(xref.get_text())

    text = entry.get_text(" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Post-processing ────────────────────────────────────────────

# Unicode maps for common scientific notation
_SUPERSCRIPT_MAP = str.maketrans(
    "0123456789+-abcdefghijklmnoprstuvwxyz",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ",
)
_SUPERSCRIPT_CHARS = set("0123456789+-abcdefghijklmnoprstuvwxyz")
_SUBSCRIPT_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Special superscript replacements
_SUPERSCRIPT_SPECIAL = {
    "◦": "°",
    "°": "°",
    ",": ",",
    "#": "#",
}


def _clean_cell(text: str) -> str:
    """
    Clean a single cell value:
      - Collapse notation spacing: ^{ –4 } → ^{-4}
      - Convert subscripts to Unicode: CO _{ 2 } → CO₂
      - Convert superscripts to Unicode: m ^{ 3 } → m³, H ^{ + } → H⁺
      - Standardize dashes in numeric contexts: – and − → -
      - Clean up scientific notation: 5.3 × 10 ^{ –4 } → 5.3 × 10⁻⁴
    """
    if not text or not isinstance(text, str):
        return text

    # Standardize dash characters to regular hyphen-minus in braces
    text = text.replace("–", "-").replace("−", "-").replace("‐", "-")

    # Remove space before subscript/superscript markers (artifact of XML extraction)
    text = re.sub(r"\s+_\{", "_{", text)
    text = re.sub(r"\s+\^\{", "^{", text)

    # Clean subscript notation: _{ 2 } → ₂, _{ l2 } → keep as _{l2}
    def _sub_repl(m):
        inner = m.group(1).strip()
        # Pure numeric/simple subscripts → Unicode
        if all(c in "0123456789" for c in inner):
            return inner.translate(_SUBSCRIPT_MAP)
        # Chemical formula subscripts like "l2" → keep readable
        converted = ""
        for c in inner:
            if c in "0123456789":
                converted += c.translate(_SUBSCRIPT_MAP)
            else:
                converted += c
        return converted

    text = re.sub(r"_\{\s*([^}]+?)\s*\}", _sub_repl, text)

    # Clean superscript notation: ^{ -4 } → ⁻⁴, ^{ a } → ᵃ, ^{ ◦ } → °
    def _sup_repl(m):
        inner = m.group(1).strip()
        # Special single-char replacements (degree, comma, hash)
        if inner in _SUPERSCRIPT_SPECIAL:
            return _SUPERSCRIPT_SPECIAL[inner]
        # All chars translatable to Unicode superscripts?
        if all(c in _SUPERSCRIPT_CHARS for c in inner):
            return inner.translate(_SUPERSCRIPT_MAP)
        # Mixed but mostly translatable (e.g., "⁻1" already partially converted)
        # → strip braces, keep content readable
        return inner

    text = re.sub(r"\^\{\s*([^}]+?)\s*\}", _sup_repl, text)

    # Clean up spacing around × in scientific notation
    text = re.sub(r"\s*×\s*", " × ", text)

    # Collapse any double spaces
    text = re.sub(r"  +", " ", text).strip()

    return text


def _postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all cells in a DataFrame and handle structural issues:
      - Apply _clean_cell to every string value
      - Forward-fill first column (handles merged/spanning cells)
      - Drop pure separator rows (all NaN/empty except one "Outputs"/"Inputs" cell)
    """
    # Clean all cells
    cleaned = df.copy()
    for idx in range(len(cleaned.columns)):
        cleaned.iloc[:, idx] = cleaned.iloc[:, idx].apply(
            lambda x: _clean_cell(str(x)) if pd.notna(x) and isinstance(x, str) and str(x).strip() else x
        )

    # Also clean column headers
    cleaned.columns = [_clean_cell(str(c)) for c in cleaned.columns]

    # Forward-fill first column (common in tables with spanning process names)
    first = cleaned.iloc[:, 0]
    if first.isna().any() or (first == "").any():
        cleaned.iloc[:, 0] = first.replace("", pd.NA).ffill()

    return cleaned


# ── Table Extraction ───────────────────────────────────────────

def extract_tables_from_xml(xml_path: Path) -> list[dict]:
    """
    Extract all tables from an Elsevier XML file.

    Returns list of dicts:
        {
            "label": "Table 1",
            "caption": "Description...",
            "headers": ["col1", "col2", ...],
            "data": [["val1", "val2", ...], ...],
            "df": pd.DataFrame,
            "source_file": "10.1016_j.xxx.xml",
        }
    """
    content = xml_path.read_text(encoding="utf-8", errors="ignore")
    soup = _parse_xml(content)

    results = []
    tables = soup.find_all("ce:table")
    if not tables:
        # Try without namespace prefix
        tables = soup.find_all("table")

    for table_tag in tables:
        # Label (e.g., "Table 1")
        label_tag = table_tag.find("ce:label") or table_tag.find("label")
        label = label_tag.get_text(strip=True) if label_tag else ""

        # Caption
        caption_tag = table_tag.find("ce:caption") or table_tag.find("caption")
        caption = ""
        if caption_tag:
            para = caption_tag.find("ce:simple-para") or caption_tag.find("simple-para")
            if para:
                caption = para.get_text(" ", strip=True)
            else:
                caption = caption_tag.get_text(" ", strip=True)
        caption = re.sub(r"\s+", " ", caption).strip()
        caption = _clean_cell(caption)

        # Find tgroup (contains the actual table structure)
        tgroup = table_tag.find("tgroup")
        if tgroup is None:
            continue

        # Headers
        headers = []
        thead = tgroup.find("thead")
        if thead:
            header_rows = thead.find_all("row")
            if header_rows:
                # Use last header row as column names (handles multi-row headers)
                last_header = header_rows[-1]
                headers = [_cell_text(e) for e in last_header.find_all("entry")]

        # Data rows
        data = []
        tbody = tgroup.find("tbody")
        if tbody:
            for row in tbody.find_all("row"):
                cells = [_cell_text(e) for e in row.find_all("entry")]
                data.append(cells)

        if not data and not headers:
            continue

        # Build DataFrame
        ncols = int(tgroup.get("cols", 0)) or max(
            len(headers),
            max((len(r) for r in data), default=0),
        )

        # Pad headers if needed
        if len(headers) < ncols:
            headers += [f"col_{i+1}" for i in range(len(headers), ncols)]

        # Pad/trim rows
        padded_data = []
        for row in data:
            if len(row) < ncols:
                row = row + [""] * (ncols - len(row))
            elif len(row) > ncols:
                row = row[:ncols]
            padded_data.append(row)

        df = pd.DataFrame(padded_data, columns=headers[:ncols])

        # Post-process: clean notation, forward-fill, etc.
        df = _postprocess_df(df)
        # Update headers to match cleaned columns
        headers = list(df.columns)

        results.append({
            "label": label,
            "caption": caption,
            "headers": headers[:ncols],
            "data": padded_data,
            "df": df,
            "source_file": xml_path.name,
        })

    return results


# ── Batch Export ───────────────────────────────────────────────

def export_tables_to_excel(
    tables: list[dict],
    output_path: Path,
) -> Path:
    """Save extracted tables to a single Excel file (one sheet per table)."""
    with pd.ExcelWriter(output_path) as writer:
        for i, t in enumerate(tables):
            sheet_name = f"{t['label'] or f'Table_{i+1}'}"[:31]
            # Deduplicate sheet names
            existing = writer.sheets if hasattr(writer, 'sheets') else {}
            if sheet_name in existing:
                sheet_name = f"{sheet_name[:28]}_{i}"[:31]

            t["df"].to_excel(writer, sheet_name=sheet_name, index=False)
    return output_path


def process_single_xml(xml_path: Path, output_dir: Path) -> dict:
    """Process one XML file, return stats."""
    tables = extract_tables_from_xml(xml_path)
    if not tables:
        return {"file": xml_path.name, "tables": 0, "status": "no_tables"}

    doi_stem = xml_path.stem
    out_path = output_dir / f"{doi_stem}.xlsx"
    export_tables_to_excel(tables, out_path)

    return {
        "file": xml_path.name,
        "tables": len(tables),
        "status": "exported",
        "output": str(out_path),
        "labels": [t["label"] for t in tables],
    }


def batch_extract(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Extract tables from all XMLs in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(input_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in {input_dir}")

    log = []
    exported = 0
    total_tables = 0

    for i, xml_path in enumerate(xml_files):
        result = process_single_xml(xml_path, output_dir)
        log.append(result)

        if result["status"] == "exported":
            exported += 1
            total_tables += result["tables"]
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(xml_files)}] {exported} files exported, {total_tables} tables")

    no_tables = sum(1 for r in log if r["status"] == "no_tables")

    print(f"\n{'='*60}")
    print(f"XML files processed: {len(xml_files)}")
    print(f"  With tables:       {exported}")
    print(f"  Without tables:    {no_tables}")
    print(f"  Total tables:      {total_tables}")
    print(f"Output directory:    {output_dir}")

    return pd.DataFrame(log)


# ── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract tables from Elsevier XML files"
    )
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory of XML files")
    parser.add_argument("--input-file", type=Path, default=None,
                        help="Single XML file to process")
    parser.add_argument("--output-dir", type=Path, default=Path("tables_xml"),
                        help="Output directory for Excel files")
    args = parser.parse_args()

    if args.input_file:
        # Single file mode — print to stdout
        tables = extract_tables_from_xml(args.input_file)
        print(f"\n{args.input_file.name}: {len(tables)} tables\n")
        for t in tables:
            print(f"### {t['label']}: {t['caption']}")
            print(t["df"].to_markdown(index=False))
            print()
        if tables:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out = args.output_dir / f"{args.input_file.stem}.xlsx"
            export_tables_to_excel(tables, out)
            print(f"Saved to {out}")

    elif args.input_dir:
        log_df = batch_extract(args.input_dir, args.output_dir)
        log_path = args.output_dir / "extraction_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"Log saved to {log_path}")

    else:
        parser.error("Provide either --input-dir or --input-file")


if __name__ == "__main__":
    main()