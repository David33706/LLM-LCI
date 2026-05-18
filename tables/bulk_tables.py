from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from langchain_core.output_parsers import StrOutputParser

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extract import ConverterExtractor
from critique_agent import (
    build_table_markdown,
    critique_prompt,
    extract_table_context,
    load_critic,
    resolve_device_map,
)

MAX_EXCEL_FILENAME_LEN = 80
MAX_SHEET_NAME_LEN = 31
TABLE_KEYWORD_PATTERN = re.compile(
    r"\b(input|inputs|output|outputs|inventory|inventories|lci)\b",
    flags=re.IGNORECASE,
)
TABLE_TITLE_LINE_PATTERN = re.compile(r"^\s*table\s*\d+\b[^\n]*", flags=re.IGNORECASE | re.MULTILINE)


def sanitize_name(name: str) -> str:
    """Return a filesystem-safe stem with compact spacing."""
    cleaned = re.sub(r"[\\/:*?\"<>|]", " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "paper"


def shorten_stem(name: str, max_len: int = MAX_EXCEL_FILENAME_LEN) -> str:
    safe = sanitize_name(name)
    return safe[:max_len].rstrip(" .") or "paper"


def unique_path(path: Path) -> Path:
    """Avoid overwriting by appending a numeric suffix when needed."""
    if not path.exists():
        return path

    base = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{base}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def safe_sheet_name(index: int, used: set[str]) -> str:
    base = f"Table_{index + 1}"[:MAX_SHEET_NAME_LEN]
    if base not in used:
        used.add(base)
        return base

    counter = 1
    while True:
        suffix = f"_{counter}"
        trimmed = base[: MAX_SHEET_NAME_LEN - len(suffix)] + suffix
        if trimmed not in used:
            used.add(trimmed)
            return trimmed
        counter += 1


def extract_markdown_block(text: str) -> str:
    fenced = re.search(r"```markdown\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _split_md_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    parts = re.split(r"(?<!\\)\|", row)
    return [p.replace(r"\|", "|").strip() for p in parts]


def markdown_table_to_dataframe(
    markdown_text: str,
    expected_ncols: int,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Parse critique markdown back into a DataFrame while preserving original schema width."""
    block = extract_markdown_block(markdown_text)
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        raise ValueError("No markdown table rows detected in critique output.")

    data_rows: list[list[str]] = []
    for line in table_lines[1:]:
        # Skip markdown separator rows like |---|:---:|---|
        cells = _split_md_row(line)
        if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        data_rows.append(cells)

    if not data_rows:
        raise ValueError("No data rows found in critique markdown table.")

    fixed_rows: list[list[str]] = []
    for row in data_rows:
        if len(row) < expected_ncols:
            row = row + [""] * (expected_ncols - len(row))
        elif len(row) > expected_ncols:
            row = row[: expected_ncols - 1] + [" | ".join(row[expected_ncols - 1 :])]
        fixed_rows.append(row)

    cols = [str(c) for c in expected_columns]
    if len(cols) < expected_ncols:
        cols += [f"col_{i + 1}" for i in range(len(cols), expected_ncols)]
    else:
        cols = cols[:expected_ncols]

    return pd.DataFrame(fixed_rows, columns=cols)


def extract_title_like_line(raw_context: str) -> str:
    """Return caption-like text using only the 100 chars before each Table line."""
    for match in TABLE_TITLE_LINE_PATTERN.finditer(raw_context):
        title_line = match.group(0).strip()
        prefix = raw_context[max(0, match.start() - 100):match.start()]
        prefix = " ".join(prefix.split())
        if prefix:
            return f"{prefix} {title_line}"
        return title_line
    return ""


def is_relevant_lci_table(raw_context: str) -> tuple[bool, str]:
    """Filter using title/caption-like line keywords to reduce compute overhead."""
    title_hint = extract_title_like_line(raw_context)
    return bool(TABLE_KEYWORD_PATTERN.search(title_hint)), title_hint


def export_tables_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    extractor: ConverterExtractor,
    critique_chain=None,
    use_keyword_filter: bool = True,
) -> tuple[Path | None, int, int, int]:
    doc = extractor.convert(str(pdf_path))
    tables = list(getattr(doc, "tables", []) or [])
    if not tables:
        return None, 0, 0, 0

    out_stem = shorten_stem(pdf_path.stem)
    out_path = unique_path(output_dir / f"{out_stem}.xlsx")

    selected_tables: list[tuple[int, object, str, str]] = []
    ignored_non_lci = 0
    for i, table in enumerate(tables):
        raw_context = extract_table_context(pdf_path, table)
        if use_keyword_filter:
            keep, title_hint = is_relevant_lci_table(raw_context)
        else:
            keep = True
            title_hint = ""
        if keep:
            selected_tables.append((i, table, raw_context, title_hint))
        else:
            ignored_non_lci += 1

    if not selected_tables:
        return None, len(tables), 0, ignored_non_lci

    used_sheet_names: set[str] = set()
    with pd.ExcelWriter(out_path) as writer:
        for i, table, raw_context, title_hint in selected_tables:
            original_df: pd.DataFrame = table.export_to_dataframe(doc=doc)
            df = original_df.copy()

            if critique_chain is not None:
                ocr_table = build_table_markdown(table, doc)
                corrected = critique_chain.invoke(
                    {"raw_context": raw_context, "ocr_table": ocr_table}
                )
                try:
                    df = markdown_table_to_dataframe(
                        corrected,
                        expected_ncols=original_df.shape[1],
                        expected_columns=[str(c) for c in original_df.columns],
                    )
                except Exception as exc:
                    print(f"  [WARN] Table {i + 1}: critique parse failed, using original extraction ({exc})")

            if title_hint:
                print(f"  [KEEP] Table {i + 1}: {title_hint[:120]}")

            sheet = safe_sheet_name(i, used_sheet_names)
            df.to_excel(writer, sheet_name=sheet, index=False)

    return out_path, len(tables), len(selected_tables), ignored_non_lci


def collect_pdfs(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(input_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all Docling-detected tables from PDFs and export one Excel file per paper."
    )
    parser.add_argument("--input_dir", type=Path, help="Folder containing PDF files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables_exports"),
        help="Folder where Excel files are written (default: tables_exports)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process PDFs in nested folders.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model name used for table critique.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/work/lamlab/huggingface/models",
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        choices=[0, 1],
        default=None,
        help="Force critique model loading on a single GPU (0 or 1). If omitted, uses auto device_map.",
    )
    parser.add_argument(
        "--no-critique",
        action="store_true",
        help="Disable LLM-based table rectification and export raw extracted tables only.",
    )
    parser.add_argument(
        "--no-title-keyword-filter",
        action="store_true",
        help="Disable title keyword matching and process all detected tables.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"ERROR: input_dir does not exist or is not a folder: {input_dir}")

    pdf_files = collect_pdfs(input_dir, recursive=args.recursive)
    if not pdf_files:
        raise SystemExit(f"ERROR: No PDF files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = ConverterExtractor()

    critique_chain = None
    if not args.no_critique:
        print("[INFO] Loading critique model...")
        critic = load_critic(
            model_name=args.model,
            cache_dir=args.cache_dir,
            device_map=resolve_device_map(args.gpu_index),
        )
        critique_chain = critique_prompt | critic | StrOutputParser()

    print(f"[INFO] Found {len(pdf_files)} PDF(s) in {input_dir}")
    print(f"[INFO] Writing Excel outputs to {output_dir}")
    print(f"[INFO] Critique mode: {'OFF (raw tables)' if args.no_critique else 'ON (rectified tables)'}")
    print(f"[INFO] Title keyword filter: {'OFF (all tables)' if args.no_title_keyword_filter else 'ON'}")

    exported = 0
    skipped = 0
    ignored_tables_total = 0
    selected_tables_total = 0
    failed = 0

    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{idx}/{len(pdf_files)}] {pdf_path.name}")
        try:
            out_path, total_tables, selected_tables, ignored_non_lci = export_tables_from_pdf(
                pdf_path,
                output_dir,
                extractor,
                critique_chain=critique_chain,
                use_keyword_filter=not args.no_title_keyword_filter,
            )
            ignored_tables_total += ignored_non_lci
            selected_tables_total += selected_tables
            if out_path is None:
                skipped += 1
                if total_tables == 0:
                    print("  [SKIP] No tables detected.")
                else:
                    print("  [SKIP] Tables detected but none matched keyword/title filter.")
                continue
            exported += 1
            print(f"  [OK] {out_path.name}")
            print(f"  [INFO] Selected tables: {selected_tables}/{total_tables} | Ignored by filter: {ignored_non_lci}")
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PDFs processed  : {len(pdf_files)}")
    print(f"Exported workbooks: {exported}")
    print(f"Skipped (no tables): {skipped}")
    print(f"Tables selected   : {selected_tables_total}")
    print(f"Tables ignored by keyword filter: {ignored_tables_total}")
    print(f"Failed            : {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
