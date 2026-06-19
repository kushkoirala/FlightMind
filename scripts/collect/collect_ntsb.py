"""
Download and process NTSB accident/incident data.

Two data sources:
1. NTSB bulk CSV (avall.zip) — structured data for all events 1982-present
2. Zenodo full-text corpus — extracted narrative text from NTSB PDFs (2016-2023)

The structured data gives us fields like: event date, location, aircraft make/model,
injury severity, weather, phase of flight, probable cause, etc.

The full-text corpus gives us the detailed narrative investigation reports.

Output: data/raw/ntsb/
  ├── avall/           (structured CSV data)
  ├── narratives/      (full-text reports from Zenodo)
  └── manifest.json
"""

import os
import sys
import json
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ntsb"

# NTSB bulk download URLs
NTSB_AVALL_URL = "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5Cavall.zip"
NTSB_PRE82_URL = "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5CPRE1982.zip"

# Zenodo full-text corpus (7,462 reports, 2016-2023)
ZENODO_RECORD = "17096333"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  [SKIP] {desc} (already exists: {dest.stat().st_size / 1e6:.1f} MB)")
        return True

    print(f"  [GET]  {desc}...")
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(dest, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {desc}") as pbar:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"  [OK]   {desc} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {desc}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_ntsb_bulk(output_dir: Path):
    """Download NTSB bulk CSV data."""
    zip_dir = output_dir / "zip"
    csv_dir = output_dir / "avall"
    zip_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading NTSB bulk data...")

    # Download avall.zip (1982-present)
    avall_zip = zip_dir / "avall.zip"
    if download_file(NTSB_AVALL_URL, avall_zip, "avall.zip (1982-present)"):
        if not any(csv_dir.glob("*.csv")):
            print("  Extracting avall.zip...")
            with zipfile.ZipFile(avall_zip, "r") as zf:
                zf.extractall(csv_dir)
            print(f"  Extracted to {csv_dir}")

    # Download PRE1982.zip
    pre82_zip = zip_dir / "PRE1982.zip"
    if download_file(NTSB_PRE82_URL, pre82_zip, "PRE1982.zip (pre-1982)"):
        pre82_dir = output_dir / "pre1982"
        pre82_dir.mkdir(exist_ok=True)
        if not any(pre82_dir.glob("*.csv")):
            print("  Extracting PRE1982.zip...")
            with zipfile.ZipFile(pre82_zip, "r") as zf:
                zf.extractall(pre82_dir)


def extract_narratives_from_mdb(output_dir: Path) -> dict:
    """Extract narrative text from NTSB MDB (Access) files using pyodbc."""
    import pyodbc

    stats = {"total_events": 0, "with_narrative": 0, "total_chars": 0}
    narr_dir = output_dir / "narratives"
    narr_dir.mkdir(exist_ok=True)

    # Process each MDB file
    for mdb_file in [output_dir / "avall" / "avall.mdb",
                     output_dir / "pre1982" / "PRE1982.MDB"]:
        if not mdb_file.exists():
            print(f"  [SKIP] {mdb_file.name} not found")
            continue

        print(f"  Processing {mdb_file.name}...")
        try:
            conn_str = (
                r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
                f"DBQ={mdb_file};"
            )
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()

            # List tables
            tables = [t.table_name for t in cursor.tables(tableType="TABLE")]
            print(f"    Tables: {', '.join(tables[:10])}")

            # Look for the events table (typically named 'events' or 'Events')
            events_table = None
            for t in tables:
                if t.lower() in ("events", "event", "avall"):
                    events_table = t
                    break
            if not events_table:
                # Use first table
                events_table = tables[0] if tables else None

            if events_table:
                # Get column names
                cursor.execute(f"SELECT TOP 1 * FROM [{events_table}]")
                columns = [desc[0] for desc in cursor.description]
                print(f"    Columns ({len(columns)}): {', '.join(columns[:15])}...")

                # Identify narrative columns
                narr_cols = [c for c in columns if any(
                    kw in c.lower() for kw in
                    ["narr", "narrative", "cause", "findings", "synopsis"]
                )]
                id_cols = [c for c in columns if any(
                    kw in c.lower() for kw in ["ev_id", "ntsb_no", "event_id"]
                )]
                id_col = id_cols[0] if id_cols else columns[0]

                print(f"    ID column: {id_col}")
                print(f"    Narrative columns: {narr_cols if narr_cols else '(none found)'}")

                # Count total rows
                cursor.execute(f"SELECT COUNT(*) FROM [{events_table}]")
                total = cursor.fetchone()[0]
                stats["total_events"] += total
                print(f"    Total events: {total:,}")

                # Extract narratives if columns exist
                if narr_cols:
                    select_cols = f"[{id_col}], " + ", ".join(f"[{c}]" for c in narr_cols)
                    cursor.execute(f"SELECT {select_cols} FROM [{events_table}]")

                    batch = []
                    for row in cursor:
                        narrative_parts = []
                        for i, col in enumerate(narr_cols):
                            val = row[i + 1]
                            if val and str(val).strip():
                                narrative_parts.append(f"## {col}\n{str(val).strip()}")

                        if narrative_parts:
                            stats["with_narrative"] += 1
                            text = f"# Event: {row[0]}\n\n" + "\n\n".join(narrative_parts)
                            stats["total_chars"] += len(text)
                            batch.append(text)

                    # Write narratives to a single file
                    if batch:
                        out_file = narr_dir / f"{mdb_file.stem}_narratives.txt"
                        out_file.write_text(
                            "\n\n---\n\n".join(batch), encoding="utf-8"
                        )
                        print(f"    Wrote {len(batch):,} narratives to {out_file.name}")

            # Also check for a Narratives table
            for t in tables:
                if "narr" in t.lower() and t != events_table:
                    print(f"    Found additional table: {t}")
                    try:
                        cursor.execute(f"SELECT TOP 1 * FROM [{t}]")
                        cols = [desc[0] for desc in cursor.description]
                        cursor.execute(f"SELECT COUNT(*) FROM [{t}]")
                        cnt = cursor.fetchone()[0]
                        print(f"      Columns: {', '.join(cols[:10])}")
                        print(f"      Rows: {cnt:,}")

                        # Extract all text from this table
                        text_cols = [c for c in cols if any(
                            kw in c.lower() for kw in
                            ["narr", "text", "cause", "finding", "synopsis"]
                        )]
                        if text_cols:
                            cursor.execute(f"SELECT {', '.join(f'[{c}]' for c in text_cols)} FROM [{t}]")
                            narr_texts = []
                            for row in cursor:
                                parts = [str(v).strip() for v in row if v and str(v).strip()]
                                if parts:
                                    narr_texts.append("\n".join(parts))
                                    stats["total_chars"] += sum(len(p) for p in parts)
                                    stats["with_narrative"] += 1

                            if narr_texts:
                                out_file = narr_dir / f"{mdb_file.stem}_{t}_narratives.txt"
                                out_file.write_text(
                                    "\n\n---\n\n".join(narr_texts), encoding="utf-8"
                                )
                                print(f"      Wrote {len(narr_texts):,} narratives")
                    except Exception as e:
                        print(f"      Error reading {t}: {e}")

            conn.close()

        except Exception as e:
            print(f"    Error processing {mdb_file.name}: {e}")

    # Also process Zenodo CSV if present
    zenodo_csv = output_dir / "zenodo" / "final_reports_2016-23_cons_2024-12-24.csv"
    if zenodo_csv.exists():
        print(f"\n  Processing Zenodo CSV ({zenodo_csv.stat().st_size / 1e6:.1f} MB)...")
        import csv
        try:
            with open(zenodo_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                zen_texts = []
                for row in reader:
                    # Combine all text columns
                    text_parts = []
                    for col, val in row.items():
                        if val and len(val) > 50:
                            text_parts.append(f"## {col}\n{val}")
                    if text_parts:
                        zen_texts.append("\n\n".join(text_parts))
                        stats["total_chars"] += sum(len(p) for p in text_parts)

                print(f"    Zenodo reports: {len(zen_texts):,}")
                stats["with_narrative"] += len(zen_texts)
        except Exception as e:
            print(f"    Error processing Zenodo CSV: {e}")

    print(f"\n  Summary:")
    print(f"  Events: {stats['total_events']:,}")
    print(f"  With narratives: {stats['with_narrative']:,}")
    print(f"  Total narrative text: {stats['total_chars']:,} chars (~{stats['total_chars'] // 4:,} tokens)")
    return stats


def download_zenodo_fulltext(output_dir: Path):
    """Download full-text NTSB reports from Zenodo."""
    zen_dir = output_dir / "zenodo"
    zen_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading Zenodo full-text corpus...")
    try:
        resp = requests.get(ZENODO_API, timeout=30)
        resp.raise_for_status()
        record = resp.json()

        files = record.get("files", [])
        print(f"  Found {len(files)} files in Zenodo record {ZENODO_RECORD}")

        for file_info in files:
            filename = file_info["key"]
            url = file_info["links"]["self"]
            size = file_info["size"]
            dest = zen_dir / filename
            download_file(url, dest, f"{filename} ({size / 1e6:.1f} MB)")

    except Exception as e:
        print(f"  [FAIL] Could not access Zenodo API: {e}")
        print(f"  Manual download: https://zenodo.org/records/{ZENODO_RECORD}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("NTSB Data Collector")
    print(f"{'=' * 60}")
    print(f"Output: {RAW_DIR}")
    print()

    # Phase 1: NTSB bulk CSV
    download_ntsb_bulk(RAW_DIR)

    # Phase 2: Zenodo full-text
    download_zenodo_fulltext(RAW_DIR)

    # Phase 3: Extract narrative text from MDB/CSV files
    print("\nExtracting narratives...")
    stats = extract_narratives_from_mdb(RAW_DIR)

    # Save manifest
    manifest = {
        "source": "NTSB Aviation Accident Data",
        "urls": {
            "bulk_csv": "https://www.ntsb.gov/safety/data/Pages/Data_Stats.aspx",
            "zenodo": f"https://zenodo.org/records/{ZENODO_RECORD}",
        },
        "stats": stats,
    }
    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"NTSB collection complete.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
