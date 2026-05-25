"""
screening/download_xml.py
-------------------------
Download full-text XML from Elsevier for papers classified as LCA-relevant.

Usage:
    python -m screening.download_xml
    python -m screening.download_xml --output-dir elsevier_xml --delay 0.5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path so utils.elsevier is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.elsevier import batch_fetch_elsevier_xml


SCREENED_FILES = [
    "screening/results/List_1_screened.csv",
    "screening/results/List_2_screened.csv",
]


def collect_dois(file_list: list) -> list:
    """Read screened CSVs, filter to LCA_prediction==1, deduplicate DOIs."""
    all_dois = set()
    for path in file_list:
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        positive = df[df["LCA_prediction"] == 1]
        dois = positive["DOI"].dropna().unique()
        print(f"  {path}: {len(dois)} LCA-relevant papers with DOIs")
        all_dois.update(dois)

    return sorted(all_dois)


def main():
    parser = argparse.ArgumentParser(description="Download Elsevier XML for LCA-relevant papers")
    parser.add_argument("--output-dir", default="elsevier_xml", help="Directory to save XML files")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per DOI")
    parser.add_argument("--all-publishers", action="store_true",
                        help="Try all DOIs, not just Elsevier (10.1016)")
    args = parser.parse_args()

    print("Collecting DOIs from screened results...")
    all_dois = collect_dois(SCREENED_FILES)
    print(f"Total unique DOIs: {len(all_dois)}")

    if not args.all_publishers:
        dois = [d for d in all_dois if d.startswith("10.1016/")]
        skipped = len(all_dois) - len(dois)
        print(f"Elsevier DOIs (10.1016): {len(dois)}")
        print(f"Non-Elsevier skipped: {skipped}")
    else:
        dois = all_dois

    print()

    if not dois:
        print("No DOIs found. Run the screening first.")
        sys.exit(1)

    results = batch_fetch_elsevier_xml(
        doi_list=dois,
        output_dir=args.output_dir,
        delay=args.delay,
        max_retries=args.max_retries,
    )

    # Summary
    success = sum(1 for v in results.values() if v is not None)
    failed = sum(1 for v in results.values() if v is None)
    print(f"\n{'='*60}")
    print(f"Done. {success} downloaded, {failed} failed (non-Elsevier or unavailable).")
    print(f"XML files saved to: {args.output_dir}/")

    # Save a log of what worked and what didn't
    log_path = Path(args.output_dir) / "download_log.csv"
    log = pd.DataFrame([
        {"DOI": doi, "status": "success" if path else "failed", "path": path or ""}
        for doi, path in results.items()
    ])
    log.to_csv(log_path, index=False)
    print(f"Download log saved to: {log_path}")


if __name__ == "__main__":
    main()