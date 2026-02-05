"""
Download historical METAR/TAF data from Iowa Environmental Mesonet (IEM).

IEM maintains the most comprehensive free archive of METAR observations
worldwide. Data is available via a simple HTTP API.

Strategy: Download METAR data for major US airports over multiple years.
This gives the model exposure to weather encoding/decoding patterns.

Output: data/raw/metar/
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "metar"

# IEM METAR download API
IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Major US airports to collect METAR data from
# Covers diverse weather patterns across the country
STATIONS = [
    # Major hubs
    "KJFK", "KLAX", "KORD", "KATL", "KDEN",
    "KDFW", "KSFO", "KLAS", "KMIA", "KSEA",
    # Regional variety
    "KBOS", "KPHX", "KMSP", "KSTL", "KMCI",
    "KSLC", "KPDX", "KBWI", "KIAD", "KIAH",
    # GA airports
    "KHUT", "KICT", "KCOS", "KAPA", "KFFZ",
    "KGAI", "KCDW", "KPAO", "KRVS", "KVNY",
]

# Date range for collection
START_YEAR = 2020
END_YEAR = 2025


def download_station_year(station: str, year: int, output_dir: Path) -> dict:
    """Download one year of METAR data for a station."""
    filename = f"{station}_{year}.csv"
    dest = output_dir / filename

    if dest.exists() and dest.stat().st_size > 100:
        return {"station": station, "year": year, "status": "skipped",
                "size": dest.stat().st_size}

    params = {
        "station": station,
        "data": "all",
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "yes",
        "elev": "yes",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",  # METAR + SPECI
        "year1": year, "month1": 1, "day1": 1,
        "year2": year, "month2": 12, "day2": 31,
    }

    try:
        resp = requests.get(IEM_BASE, params=params, timeout=120)
        resp.raise_for_status()

        # IEM returns CSV with header
        content = resp.text
        if len(content) < 100 or "ERROR" in content[:200]:
            return {"station": station, "year": year, "status": "empty"}

        dest.write_text(content, encoding="utf-8")
        lines = content.count("\n")
        return {"station": station, "year": year, "status": "ok",
                "observations": lines - 1, "size": len(content)}

    except Exception as e:
        return {"station": station, "year": year, "status": "error", "error": str(e)}


def extract_raw_metar_strings(csv_dir: Path, output_dir: Path) -> dict:
    """Extract raw METAR strings from downloaded CSVs for text training."""
    import csv

    metar_file = output_dir / "all_metars.txt"
    stats = {"total_observations": 0, "total_chars": 0}

    with open(metar_file, "w", encoding="utf-8") as out:
        for csv_path in sorted(csv_dir.glob("*.csv")):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # IEM CSV has a 'metar' column with the raw METAR string
                        metar = row.get("metar", "")
                        if metar and metar != "M":
                            out.write(metar + "\n")
                            stats["total_observations"] += 1
                            stats["total_chars"] += len(metar)
            except Exception:
                pass

    return stats


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_dir = RAW_DIR / "csv"
    csv_dir.mkdir(exist_ok=True)

    print("METAR Data Collector (Iowa Environmental Mesonet)")
    print(f"{'=' * 60}")
    print(f"Stations: {len(STATIONS)}")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Output: {RAW_DIR}")
    print()

    # Build task list
    tasks = []
    for station in STATIONS:
        for year in range(START_YEAR, END_YEAR + 1):
            tasks.append((station, year))

    print(f"Downloading {len(tasks)} station-year combinations...")
    print(f"(Rate-limited to avoid overloading IEM servers)")
    print()

    results = []
    # Use modest parallelism to be polite to IEM servers
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(download_station_year, s, y, csv_dir): (s, y)
            for s, y in tasks
        }
        for i, future in enumerate(as_completed(futures)):
            result = futures[future]
            r = future.result()
            results.append(r)

            if r["status"] == "ok":
                obs = r.get("observations", 0)
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(tasks)}] {r['station']} {r['year']}: {obs:,} obs")

            # Small delay to be respectful
            time.sleep(0.2)

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    total_obs = sum(r.get("observations", 0) for r in results if r["status"] == "ok")
    print(f"\nDownloaded: {ok}, Skipped: {skipped}, Total obs: {total_obs:,}")

    # Extract raw METAR strings
    print("\nExtracting raw METAR strings...")
    metar_stats = extract_raw_metar_strings(csv_dir, RAW_DIR)
    print(f"  Total METAR strings: {metar_stats['total_observations']:,}")
    print(f"  Total text: {metar_stats['total_chars']:,} chars")

    # Save manifest
    manifest = {
        "source": "Iowa Environmental Mesonet (IEM) METAR Archive",
        "url": "https://mesonet.agron.iastate.edu/request/download.phtml",
        "stations": STATIONS,
        "date_range": f"{START_YEAR}-{END_YEAR}",
        "download_results": {
            "ok": ok, "skipped": skipped, "total_observations": total_obs,
        },
        "metar_stats": metar_stats,
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"METAR collection complete.")


if __name__ == "__main__":
    main()
