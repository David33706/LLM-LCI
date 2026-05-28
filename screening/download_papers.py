"""
screening/download_papers.py
-----------------------------
Download full-text papers from Elsevier API.
  - OA papers  → XML (view=FULL)
  - Non-OA     → PDF fallback

Usage (test on existing 719 DOIs first):
    python -m screening.download_papers

Usage (full WoS 2025 set):
    python -m screening.download_papers \
        --input screening/results/WoS_2025_screened.csv

Compares with previous download_log.csv if present.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────

API_KEY = os.getenv("ELSEVIER_API_KEY") or "0dfa537c1b3c5439061a39391dde2c0e"
BASE_URL = "https://api.elsevier.com/content/article/doi"

DEFAULT_INPUTS = [
    "screening/results/List_1_screened.csv",
    "screening/results/List_2_screened.csv",
]


# ── Helpers ────────────────────────────────────────────────────

def doi_to_filename(doi: str, ext: str) -> str:
    """Convert DOI to a safe filename."""
    return doi.replace("/", "_").replace("\\", "_").replace(":", "_") + ext


def try_xml(doi: str, timeout: int = 30) -> requests.Response | None:
    """Try downloading full XML (works for OA papers only with CCL key)."""
    url = f"{BASE_URL}/{doi}?view=FULL&APIKey={API_KEY}"
    try:
        r = requests.get(url, headers={"Accept": "application/xml"}, timeout=timeout)
        if r.status_code == 200:
            return r
    except requests.RequestException:
        pass
    return None


def try_pdf(doi: str, timeout: int = 60) -> requests.Response | None:
    """Download PDF, reject single-page cover pages."""
    url = f"{BASE_URL}/{doi}?APIKey={API_KEY}"
    try:
        r = requests.get(url, headers={"Accept": "application/pdf"}, timeout=timeout)
        if r.status_code == 200 and len(r.content) > 1000:
            # Reject 1-page PDFs (cover/abstract pages, not full papers)
            pages = r.content.count(b"/Type /Page") - r.content.count(b"/Type /Pages")
            if pages > 1:
                return r
    except requests.RequestException:
        pass
    return None


def collect_dois(file_list: list[str]) -> list[str]:
    """Read screened CSVs → LCA_prediction==1 → Elsevier DOIs."""
    all_dois = set()
    for path in file_list:
        if not Path(path).exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path, low_memory=False)
        pos = df[df["LCA_prediction"] == 1]
        dois = pos["DOI"].dropna().unique()
        els = [d for d in dois if str(d).startswith("10.1016/")]
        print(f"  {path}: {len(els)} Elsevier LCA-relevant DOIs")
        all_dois.update(els)
    return sorted(all_dois)


# ── Main download logic ───────────────────────────────────────

def download_all(
    dois: list[str],
    xml_dir: str = "elsevier_xml",
    pdf_dir: str = "elsevier_pdf",
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    For each DOI:
      1. If XML already exists on disk → skip (status=xml_cached)
      2. If PDF already exists on disk → skip (status=pdf_cached)
      3. Try XML view=FULL → if 200, save XML
      4. If XML fails → try PDF → if 200, save PDF
      5. If both fail → status=failed
    """
    os.makedirs(xml_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    log = []

    for doi in tqdm(dois, desc="Downloading"):
        fname_xml = doi_to_filename(doi, ".xml")
        fname_pdf = doi_to_filename(doi, ".pdf")
        path_xml = os.path.join(xml_dir, fname_xml)
        path_pdf = os.path.join(pdf_dir, fname_pdf)

        # 1. Already have XML?
        if os.path.exists(path_xml) and os.path.getsize(path_xml) > 1000:
            log.append({"DOI": doi, "format": "xml", "status": "cached", "path": path_xml})
            continue

        # 2. Already have PDF?
        if os.path.exists(path_pdf) and os.path.getsize(path_pdf) > 1000:
            log.append({"DOI": doi, "format": "pdf", "status": "cached", "path": path_pdf})
            continue

        # 3. Try XML
        r = try_xml(doi)
        if r is not None:
            with open(path_xml, "wb") as f:
                f.write(r.content)
            log.append({"DOI": doi, "format": "xml", "status": "downloaded", "path": path_xml})
            time.sleep(delay)
            continue

        # 4. XML failed → try PDF
        r = try_pdf(doi)
        if r is not None:
            with open(path_pdf, "wb") as f:
                f.write(r.content)
            log.append({"DOI": doi, "format": "pdf", "status": "downloaded", "path": path_pdf})
            time.sleep(delay)
            continue

        # 5. Both failed
        log.append({"DOI": doi, "format": "none", "status": "failed", "path": ""})
        time.sleep(delay)

    return pd.DataFrame(log)


# ── Comparison with old log ────────────────────────────────────

def compare_with_old(new_log: pd.DataFrame, old_log_path: str):
    """Show improvement over previous download attempt."""
    if not os.path.exists(old_log_path):
        return

    old = pd.read_csv(old_log_path)
    old_success = set(old[old["status"] == "success"]["DOI"])
    old_failed = set(old[old["status"] == "failed"]["DOI"])

    new_has = set(new_log[new_log["status"] != "failed"]["DOI"])
    new_failed = set(new_log[new_log["status"] == "failed"]["DOI"])

    rescued = old_failed & new_has
    still_failed = old_failed & new_failed

    print(f"\n{'='*60}")
    print("Comparison with previous download_log.csv:")
    print(f"  Previously successful: {len(old_success)}")
    print(f"  Previously failed:     {len(old_failed)}")
    print(f"  Now rescued (PDF):     {len(rescued)}")
    print(f"  Still failed:          {len(still_failed)}")
    if still_failed:
        print(f"  Still-failed DOIs:     {list(still_failed)[:5]}...")


# ── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Elsevier papers (XML + PDF fallback)"
    )
    parser.add_argument(
        "--input", nargs="+", default=None,
        help="Screened CSV file(s). Default: test lists."
    )
    parser.add_argument("--xml-dir", default="elsevier_xml")
    parser.add_argument("--pdf-dir", default="elsevier_pdf")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    input_files = args.input or DEFAULT_INPUTS

    print("Collecting DOIs...")
    dois = collect_dois(input_files)
    print(f"Total Elsevier DOIs to process: {len(dois)}\n")

    if not dois:
        print("No DOIs found. Run screening first.")
        sys.exit(1)

    log_df = download_all(
        dois,
        xml_dir=args.xml_dir,
        pdf_dir=args.pdf_dir,
        delay=args.delay,
    )

    # Summary
    xml_count = len(log_df[log_df["format"] == "xml"])
    pdf_count = len(log_df[log_df["format"] == "pdf"])
    failed_count = len(log_df[log_df["status"] == "failed"])
    cached_count = len(log_df[log_df["status"] == "cached"])
    downloaded_count = len(log_df[log_df["status"] == "downloaded"])

    print(f"\n{'='*60}")
    print(f"Total:      {len(log_df)}")
    print(f"  XML:      {xml_count}  (cached: {len(log_df[(log_df['format']=='xml') & (log_df['status']=='cached')])},"
          f" new: {len(log_df[(log_df['format']=='xml') & (log_df['status']=='downloaded')])})")
    print(f"  PDF:      {pdf_count}  (cached: {len(log_df[(log_df['format']=='pdf') & (log_df['status']=='cached')])},"
          f" new: {len(log_df[(log_df['format']=='pdf') & (log_df['status']=='downloaded')])})")
    print(f"  Failed:   {failed_count}")
    print(f"  Coverage: {xml_count + pdf_count}/{len(log_df)}"
          f" ({(xml_count + pdf_count) / len(log_df) * 100:.1f}%)")

    # Save log
    log_path = os.path.join(args.xml_dir, "download_log_v2.csv")
    log_df.to_csv(log_path, index=False)
    print(f"\nLog saved to: {log_path}")

    # Compare with old
    old_log_path = os.path.join(args.xml_dir, "download_log.csv")
    compare_with_old(log_df, old_log_path)


if __name__ == "__main__":
    main()