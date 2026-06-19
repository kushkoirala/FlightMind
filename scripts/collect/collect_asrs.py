"""
Download NASA ASRS (Aviation Safety Reporting System) data.

ASRS is the world's largest repository of voluntary, confidential aviation
safety reports from pilots, controllers, mechanics, and dispatchers.

The database allows exports of up to 10,000 records per query.
Reports contain rich free-text narratives ideal for LLM training.

Strategy: Download via the ASRS database search API in batches.
Also check for GitHub mirrors with pre-scraped data.

Output: data/raw/asrs/
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "asrs"

# GitHub mirror of ASRS data (pre-scraped)
GITHUB_MIRROR = "https://github.com/orangejulius/asrs-data"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/orangejulius/asrs-data/master"

# ASRS search endpoint
ASRS_SEARCH = "https://asrs.arc.nasa.gov/search/database.html"


def download_github_mirror(output_dir: Path):
    """Try to download ASRS data from the GitHub mirror."""
    mirror_dir = output_dir / "github_mirror"
    mirror_dir.mkdir(exist_ok=True)

    print("Checking GitHub mirror for pre-scraped ASRS data...")

    # Try to clone the repo
    clone_url = f"{GITHUB_MIRROR}.git"
    if (mirror_dir / ".git").exists():
        print("  [SKIP] GitHub mirror already cloned")
        return mirror_dir

    try:
        import subprocess
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(mirror_dir)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print(f"  [OK]   Cloned ASRS mirror")
            return mirror_dir
        else:
            print(f"  [WARN] Git clone failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"  [WARN] Could not clone mirror: {e}")

    return None


def download_asrs_batch(start_date: str, end_date: str, output_dir: Path,
                        batch_name: str) -> dict:
    """
    Download a batch of ASRS reports for a date range.

    The ASRS web interface allows CSV export. We'll construct the query
    and download the results.

    Note: The ASRS search may require manual interaction for large downloads.
    This function documents the process and handles what can be automated.
    """
    out_file = output_dir / f"asrs_{batch_name}.csv"
    if out_file.exists():
        print(f"  [SKIP] {batch_name} (already exists)")
        return {"batch": batch_name, "status": "skipped"}

    # ASRS doesn't have a clean REST API - downloads require form submission
    # Document the manual process
    print(f"  [INFO] ASRS batch {batch_name} ({start_date} to {end_date})")
    print(f"         Manual download required from:")
    print(f"         {ASRS_SEARCH}")
    print(f"         Set date range: {start_date} to {end_date}")
    print(f"         Export as CSV, save to: {out_file}")

    return {"batch": batch_name, "status": "manual_required", "date_range": [start_date, end_date]}


def process_asrs_csvs(input_dir: Path, output_dir: Path) -> dict:
    """Process any downloaded ASRS CSV files and extract narratives."""
    import csv

    narr_dir = output_dir / "narratives"
    narr_dir.mkdir(exist_ok=True)

    stats = {"files_processed": 0, "total_reports": 0, "total_chars": 0}

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("  No ASRS CSV files found to process.")
        return stats

    for csv_file in csv_files:
        print(f"  Processing {csv_file.name}...")
        try:
            with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats["total_reports"] += 1

                    # ASRS CSVs typically have narrative columns
                    narrative = ""
                    for col in reader.fieldnames:
                        col_lower = col.lower()
                        if "narrative" in col_lower or "synopsis" in col_lower:
                            if row.get(col):
                                narrative += row[col] + "\n"

                    stats["total_chars"] += len(narrative)

            stats["files_processed"] += 1
        except Exception as e:
            print(f"    Error: {e}")

    print(f"  Processed {stats['files_processed']} files")
    print(f"  Reports: {stats['total_reports']:,}")
    print(f"  Narrative text: {stats['total_chars']:,} chars")
    return stats


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("NASA ASRS Data Collector")
    print(f"{'=' * 60}")
    print(f"Output: {RAW_DIR}")
    print()

    # Phase 1: Try GitHub mirror
    mirror = download_github_mirror(RAW_DIR)

    # Phase 2: Document manual download batches
    print("\nASRS Manual Download Guide:")
    print("-" * 40)
    batches = []
    # Split into year-long batches (ASRS limits exports to 10K records)
    for year in range(2000, 2026):
        batch = download_asrs_batch(
            f"{year}-01-01", f"{year}-12-31",
            RAW_DIR, f"{year}"
        )
        batches.append(batch)

    # Phase 3: Process any existing CSVs
    print("\nProcessing existing ASRS CSV files...")
    stats = process_asrs_csvs(RAW_DIR, RAW_DIR)

    # Save manifest
    manifest = {
        "source": "NASA Aviation Safety Reporting System (ASRS)",
        "url": ASRS_SEARCH,
        "github_mirror": GITHUB_MIRROR,
        "note": "ASRS requires manual CSV export from web interface (10K records per batch)",
        "batches": batches,
        "stats": stats,
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"ASRS collection initialized.")
    print(f"Note: Most ASRS data requires manual download from the web interface.")
    print(f"Follow the batch guide above to download year-by-year.")


if __name__ == "__main__":
    main()
