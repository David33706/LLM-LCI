"""
screening/merge_wos.py
----------------------
Merge Web of Science export batches (.xls) into a single CSV.
Validates column consistency, abstract coverage, DOI duplicates,
and reports a summary before saving.

Usage:
    python -m screening.merge_wos --input-dir "../Test sets/WoS_2025" --output "../Test sets/WoS_2025/WoS_2025_merged.csv"
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


def discover_files(input_dir: str) -> list:
    """Find all .xls files in the input directory."""
    pattern = str(Path(input_dir) / "*.xls")
    files = sorted(glob.glob(pattern))
    return files


def validate_and_load(files: list) -> list:
    """Load each file, validate columns, report per-file stats."""
    dfs = []
    ref_cols = None

    print(f"{'File':<30s} {'Rows':>6s} {'Abstracts':>10s} {'DOIs':>6s} {'DupDOIs':>8s} {'Cols':>5s}")
    print("-" * 75)

    for f in files:
        name = Path(f).name
        df = pd.read_excel(f, engine="xlrd")

        if "Abstract" not in df.columns:
            print(f"  ERROR: {name} missing 'Abstract' column — was it exported with Full Record?")
            sys.exit(1)

        abstracts = df["Abstract"].notna().sum()
        dois = df["DOI"].notna().sum() if "DOI" in df.columns else 0
        dup_dois = df["DOI"].duplicated().sum() if "DOI" in df.columns else 0

        print(f"{name:<30s} {len(df):>6d} {abstracts:>10d} {dois:>6d} {dup_dois:>8d} {len(df.columns):>5d}")

        # Check column consistency
        if ref_cols is None:
            ref_cols = set(df.columns)
        elif set(df.columns) != ref_cols:
            missing = ref_cols - set(df.columns)
            extra = set(df.columns) - ref_cols
            print(f"  WARNING: column mismatch in {name}!")
            if missing:
                print(f"    Missing: {missing}")
            if extra:
                print(f"    Extra: {extra}")

        dfs.append(df)

    return dfs


def merge_and_dedup(dfs: list) -> pd.DataFrame:
    """Concatenate dataframes and deduplicate by DOI (preserving no-DOI rows)."""
    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)

    if "DOI" in merged.columns:
        has_doi = merged[merged["DOI"].notna()]
        no_doi = merged[merged["DOI"].isna()]
        has_doi = has_doi.drop_duplicates(subset="DOI", keep="first")
        merged = pd.concat([has_doi, no_doi], ignore_index=True)

    after = len(merged)
    removed = before - after

    return merged, before, after, removed


def main():
    parser = argparse.ArgumentParser(description="Merge WoS export batches into a single CSV")
    parser.add_argument("--input-dir", required=True, help="Directory containing .xls export files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    # Discover files
    files = discover_files(args.input_dir)
    if not files:
        print(f"No .xls files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} files in {args.input_dir}\n")

    # Validate and load
    dfs = validate_and_load(files)

    # Merge and dedup
    print()
    merged, before, after, removed = merge_and_dedup(dfs)

    # Summary
    abstracts = merged["Abstract"].notna().sum()
    dois = merged["DOI"].notna().sum() if "DOI" in merged.columns else 0
    no_doi = len(merged) - dois
    missing_abstract = len(merged) - abstracts

    print(f"{'='*75}")
    print(f"Total rows before dedup:  {before}")
    print(f"Duplicates removed:       {removed}")
    print(f"Final row count:          {after}")
    print(f"Abstracts present:        {abstracts}/{after} ({missing_abstract} missing)")
    print(f"DOIs present:             {dois}/{after} ({no_doi} missing)")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()