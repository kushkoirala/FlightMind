"""
FlightMind Data Cleaning Pipeline
==================================
Reads raw data from each source, cleans and normalizes text,
and outputs JSONL files to data/cleaned/ (one per source).

Each output line: {"text": "...", "source": "...", "domain": "...", "chars": N}

Usage:
    python scripts/process/clean_all.py              # clean all sources
    python scripts/process/clean_all.py --source ntsb  # clean one source
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (except newlines) to single spaces,
    collapse 3+ newlines to 2, strip trailing whitespace per line."""
    # Replace tabs and other horizontal whitespace with spaces
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip trailing whitespace per line
    text = re.sub(r" +\n", "\n", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_page_numbers(text: str) -> str:
    """Remove standalone page numbers (arabic or roman) on their own line."""
    # Lines that are just a number (possibly with whitespace)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # Lines that are just roman numerals
    text = re.sub(r"(?m)^\s*[ivxlcdm]+\s*$", "", text, flags=re.IGNORECASE)
    return text


def is_useful_text(text: str, min_chars: int = 100) -> bool:
    """Filter out documents that are too short to be useful."""
    return len(text.strip()) >= min_chars


def write_jsonl(docs: list[dict], output_path: Path):
    """Write documents to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(docs):,} documents to {output_path}")


# ---------------------------------------------------------------------------
# Source-specific cleaners
# ---------------------------------------------------------------------------

def clean_ntsb() -> list[dict]:
    """Clean NTSB accident narratives.

    Raw format: single large file with --- separators between events.
    Each event block contains factual narrative + probable cause (sometimes
    duplicated). We deduplicate and keep the longest unique version.
    """
    print("\n[NTSB] Cleaning accident narratives...")
    narratives_dir = RAW_DIR / "ntsb" / "narratives"
    docs = []

    for txt_file in sorted(narratives_dir.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8", errors="replace")
        blocks = raw.split("\n---\n")

        for block in blocks:
            block = block.strip()
            if not block or block == "import":
                continue

            # Deduplicate: NTSB data often has the same narrative twice
            # (full + condensed). Split into paragraphs and keep unique ones.
            paragraphs = [p.strip() for p in block.split("\n\n") if p.strip()]
            seen = set()
            unique_paragraphs = []
            for p in paragraphs:
                # Use first 200 chars as dedup key (catches near-duplicates)
                key = p[:200].lower().strip()
                if key not in seen:
                    seen.add(key)
                    unique_paragraphs.append(p)

            text = "\n\n".join(unique_paragraphs)
            text = normalize_whitespace(text)

            if is_useful_text(text, min_chars=200):
                docs.append({
                    "text": text,
                    "source": "ntsb",
                    "domain": "accident_reports",
                    "chars": len(text),
                })

    print(f"  [NTSB] {len(docs):,} documents, {sum(d['chars'] for d in docs):,} chars")
    return docs


def clean_handbooks() -> list[dict]:
    """Clean FAA handbooks extracted from PDFs.

    Each handbook is a single large text file. We split into chapters
    based on common patterns, remove page number artifacts, and normalize.
    """
    print("\n[FAA Handbooks] Cleaning...")
    txt_dir = RAW_DIR / "faa_handbooks" / "txt"
    docs = []

    handbook_names = {
        "phak": "Pilot's Handbook of Aeronautical Knowledge",
        "afh": "Airplane Flying Handbook",
        "ifh": "Instrument Flying Handbook",
        "iph": "Instrument Procedures Handbook",
        "hfh": "Helicopter Flying Handbook",
        "gfh": "Glider Flying Handbook",
        "wbh": "Weight and Balance Handbook",
        "aah": "Advanced Avionics Handbook",
        "rmh": "Risk Management Handbook",
        "awh": "Aviation Weather Handbook",
        "amt_airframe": "Aviation Maintenance Technician - Airframe",
    }

    for txt_file in sorted(txt_dir.glob("*.txt")):
        stem = txt_file.stem.lower()
        name = handbook_names.get(stem, stem.upper())
        raw = txt_file.read_text(encoding="utf-8", errors="replace")

        # Remove page numbers
        text = remove_page_numbers(raw)

        # Split into chapters: look for "Chapter N" or "Chapter" on its own line
        # Also look for section headers like "Section 1" or all-caps lines
        chapter_pattern = re.compile(
            r"(?m)^(Chapter\s+\d+[\.\s].*?)$", re.IGNORECASE
        )
        splits = chapter_pattern.split(text)

        if len(splits) > 2:
            # Pair up: [preamble, header1, content1, header2, content2, ...]
            # First element is preamble (table of contents, etc.)
            chunks = []
            preamble = splits[0].strip()
            if len(preamble) > 500:
                chunks.append(f"{name}\n\n{preamble}")

            for i in range(1, len(splits), 2):
                header = splits[i].strip() if i < len(splits) else ""
                content = splits[i + 1].strip() if i + 1 < len(splits) else ""
                if content:
                    chunks.append(f"{name} - {header}\n\n{content}")
        else:
            # No chapter splits found - keep as one large document
            # but split into ~10K char chunks for reasonable doc sizes
            text = normalize_whitespace(text)
            chunk_size = 10_000
            chunks = []
            for start in range(0, len(text), chunk_size):
                # Find nearest paragraph break after chunk_size
                end = start + chunk_size
                if end < len(text):
                    # Look for paragraph break near the end
                    break_pos = text.rfind("\n\n", start + chunk_size // 2, end + 2000)
                    if break_pos > start:
                        end = break_pos
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(f"{name}\n\n{chunk}")

        for chunk in chunks:
            chunk = normalize_whitespace(chunk)
            if is_useful_text(chunk, min_chars=200):
                docs.append({
                    "text": chunk,
                    "source": "faa_handbooks",
                    "domain": "training_material",
                    "chars": len(chunk),
                })

    print(f"  [FAA Handbooks] {len(docs):,} documents, {sum(d['chars'] for d in docs):,} chars")
    return docs


def clean_regulations() -> list[dict]:
    """Clean 14 CFR regulations.

    Format: sections delimited by === § X.Y - Title ===
    Each section becomes one document.
    """
    print("\n[14 CFR] Cleaning regulations...")
    txt_file = RAW_DIR / "faa_regulations" / "txt" / "14cfr_combined.txt"
    docs = []

    if not txt_file.exists():
        print("  [14 CFR] File not found, skipping")
        return docs

    raw = txt_file.read_text(encoding="utf-8", errors="replace")

    # Split on section headers
    section_pattern = re.compile(r"(?m)^===\s*(§\s*[\d\.]+\s*-\s*.+?)\s*===$")
    splits = section_pattern.split(raw)

    for i in range(1, len(splits), 2):
        header = splits[i].strip()
        content = splits[i + 1].strip() if i + 1 < len(splits) else ""
        if not content:
            continue

        text = f"14 CFR {header}\n\n{content}"
        text = normalize_whitespace(text)

        if is_useful_text(text, min_chars=50):
            docs.append({
                "text": text,
                "source": "14cfr",
                "domain": "regulations",
                "chars": len(text),
            })

    print(f"  [14 CFR] {len(docs):,} documents, {sum(d['chars'] for d in docs):,} chars")
    return docs


def clean_metar() -> list[dict]:
    """Clean METAR/TAF observations.

    Raw: one METAR string per line. Group into station-day batches
    for training (each batch ~24 observations = 1 day at one station).
    Add a context header so the model learns METAR format.
    """
    print("\n[METAR] Cleaning weather observations...")
    metar_file = RAW_DIR / "metar" / "all_metars.txt"
    docs = []

    if not metar_file.exists():
        print("  [METAR] File not found, skipping")
        return docs

    # Group METARs by station-day
    station_day_groups: dict[str, list[str]] = {}
    line_count = 0

    with open(metar_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or len(line) < 20:
                continue
            line_count += 1

            # METAR format: ICAO DDHHMM Z ...
            parts = line.split()
            if len(parts) < 3:
                continue

            station = parts[0]
            timestamp = parts[1]  # DDHHMM Z
            day = timestamp[:2] if len(timestamp) >= 2 else "00"
            key = f"{station}_{day}"

            if key not in station_day_groups:
                station_day_groups[key] = []
            station_day_groups[key].append(line)

    # Create documents from groups
    for key, metars in station_day_groups.items():
        station = key.split("_")[0]
        text = f"METAR observations for {station}:\n\n" + "\n".join(metars)

        docs.append({
            "text": text,
            "source": "metar",
            "domain": "weather",
            "chars": len(text),
        })

    print(f"  [METAR] {len(docs):,} documents from {line_count:,} observations, "
          f"{sum(d['chars'] for d in docs):,} chars")
    return docs


def clean_wikipedia() -> list[dict]:
    """Clean Wikipedia aviation articles.

    Each article is already a separate text file. Filter out stubs
    and very short articles.
    """
    print("\n[Wikipedia] Cleaning aviation articles...")
    articles_dir = RAW_DIR / "wikipedia_aviation" / "articles"
    docs = []

    if not articles_dir.exists():
        print("  [Wikipedia] Articles directory not found, skipping")
        return docs

    for txt_file in sorted(articles_dir.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8", errors="replace")

        # Remove Wikipedia reference markers like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", raw)
        # Remove "See also", "References", "External links" sections
        text = re.sub(
            r"(?m)^(See also|References|External links|Notes|Further reading|Bibliography)\s*\n.*",
            "",
            text,
            flags=re.DOTALL,
        )
        text = normalize_whitespace(text)

        # Filter stubs (< 500 chars after cleaning)
        if is_useful_text(text, min_chars=500):
            docs.append({
                "text": text,
                "source": "wikipedia",
                "domain": "encyclopedia",
                "chars": len(text),
            })

    print(f"  [Wikipedia] {len(docs):,} documents, {sum(d['chars'] for d in docs):,} chars")
    return docs


def clean_hf_datasets() -> list[dict]:
    """Clean HuggingFace datasets (kathleenge/aviation, ATC transcripts).

    These are stored as Arrow DatasetDict objects. Must select splits first.
    """
    print("\n[HF Datasets] Cleaning...")
    docs = []

    from datasets import load_from_disk

    # kathleenge/aviation dataset - single 'text' column, many empty rows
    aviation_dir = RAW_DIR / "aircraft_performance" / "aviation_dataset"
    if aviation_dir.exists():
        try:
            ds_dict = load_from_disk(str(aviation_dir))
            # It's a DatasetDict - get first available split
            split_name = list(ds_dict.keys())[0]
            ds = ds_dict[split_name]
            count = 0
            for row in ds:
                text = row.get("text", "")
                if not isinstance(text, str):
                    continue
                text = normalize_whitespace(text)
                if is_useful_text(text, min_chars=50):
                    docs.append({
                        "text": text,
                        "source": "hf_aviation",
                        "domain": "aircraft_data",
                        "chars": len(text),
                    })
                    count += 1
            print(f"  [HF aviation] {count:,} documents from {len(ds):,} rows")
        except Exception as e:
            print(f"  [HF aviation] Error: {e}")

    # ATC transcripts - DatasetDict with 'text' column
    atc_dir = RAW_DIR / "atc_transcripts"
    if atc_dir.exists():
        for ds_dir in sorted(atc_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            try:
                ds_dict = load_from_disk(str(ds_dir))
                count = 0
                for split_name in ds_dict:
                    ds = ds_dict[split_name]
                    for row in ds:
                        text = row.get("text", "")
                        if not isinstance(text, str) or len(text) < 10:
                            continue
                        text = normalize_whitespace(text)
                        if text:
                            docs.append({
                                "text": text,
                                "source": "atc_transcripts",
                                "domain": "communications",
                                "chars": len(text),
                            })
                            count += 1
                print(f"  [ATC {ds_dir.name}] {count:,} documents")
            except Exception as e:
                print(f"  [ATC {ds_dir.name}] Error: {e}")

    print(f"  [HF Datasets] Total: {len(docs):,} documents, "
          f"{sum(d['chars'] for d in docs):,} chars")
    return docs


def clean_openap() -> list[dict]:
    """Extract aircraft performance data from OpenAP as natural language.

    OpenAP stores aircraft data as YAML files and engine data as CSV.
    We convert each aircraft's performance data into readable text.
    """
    print("\n[OpenAP] Extracting aircraft performance data...")
    import csv

    try:
        import yaml
    except ImportError:
        print("  [OpenAP] PyYAML not installed, installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
        import yaml

    openap_dir = RAW_DIR / "aircraft_performance" / "openap"
    openap_data_dir = openap_dir / "openap" / "data"
    docs = []

    if not openap_data_dir.exists():
        print(f"  [OpenAP] Data directory not found at {openap_data_dir}")
        return docs

    # Read aircraft YAML files
    aircraft_dir = openap_data_dir / "aircraft"
    if aircraft_dir.exists():
        for yml_file in sorted(aircraft_dir.glob("*.yml")):
            try:
                data = yaml.safe_load(yml_file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                text = format_aircraft_yaml(data)
                if is_useful_text(text, min_chars=100):
                    docs.append({
                        "text": text,
                        "source": "openap",
                        "domain": "aircraft_performance",
                        "chars": len(text),
                    })
            except Exception as e:
                print(f"  [OpenAP] Error reading {yml_file.name}: {e}")

    # Read engine CSV
    engine_csv = openap_data_dir / "engine" / "engines.csv"
    if engine_csv.exists():
        with open(engine_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = format_engine_csv_row(row)
                if is_useful_text(text, min_chars=100):
                    docs.append({
                        "text": text,
                        "source": "openap",
                        "domain": "engine_performance",
                        "chars": len(text),
                    })

    print(f"  [OpenAP] {len(docs):,} documents, {sum(d['chars'] for d in docs):,} chars")
    return docs


def format_aircraft_yaml(data: dict) -> str:
    """Convert OpenAP aircraft YAML to readable text."""
    name = data.get("aircraft", data.get("name", "Unknown"))
    lines = [f"Aircraft Performance Data: {name}"]

    field_labels = {
        "mtow": "Maximum Takeoff Weight (kg)",
        "mlw": "Maximum Landing Weight (kg)",
        "oew": "Operating Empty Weight (kg)",
        "mfc": "Maximum Fuel Capacity (kg)",
        "vmo": "Maximum Operating Speed (kts)",
        "mmo": "Maximum Mach Number",
        "ceiling": "Service Ceiling (m)",
    }

    for key, label in field_labels.items():
        if key in data and data[key] is not None:
            lines.append(f"{label}: {data[key]}")

    if "pax" in data and isinstance(data["pax"], dict):
        pax = data["pax"]
        lines.append(f"Passenger Capacity: {pax.get('low', '?')}-{pax.get('max', '?')}")

    if "fuselage" in data and isinstance(data["fuselage"], dict):
        f = data["fuselage"]
        lines.append(f"Fuselage: length {f.get('length', '?')}m, "
                      f"width {f.get('width', '?')}m, height {f.get('height', '?')}m")

    if "wing" in data and isinstance(data["wing"], dict):
        w = data["wing"]
        lines.append(f"Wing: area {w.get('area', '?')}m², span {w.get('span', '?')}m, "
                      f"sweep {w.get('sweep', '?')}°")

    if "cruise" in data and isinstance(data["cruise"], dict):
        c = data["cruise"]
        lines.append(f"Cruise: Mach {c.get('mach', '?')} at {c.get('height', '?')}m, "
                      f"range {c.get('range', '?')}km")

    if "engine" in data and isinstance(data["engine"], dict):
        e = data["engine"]
        lines.append(f"Engines: {e.get('number', '?')}x {e.get('default', '?')} "
                      f"({e.get('type', '?')}, {e.get('mount', '?')}-mounted)")
        if "options" in e and isinstance(e["options"], dict):
            for variant, eng in e["options"].items():
                lines.append(f"  {variant}: {eng}")

    if "drag" in data and isinstance(data["drag"], dict):
        d = data["drag"]
        lines.append(f"Drag: CD0={d.get('cd0', '?')}, k={d.get('k', '?')}, "
                      f"Oswald efficiency={d.get('e', '?')}")

    return "\n".join(lines)


def format_engine_csv_row(row: dict) -> str:
    """Convert OpenAP engine CSV row to readable text."""
    name = row.get("name", "Unknown")
    mfr = row.get("manufacturer", "Unknown")
    etype = {"TF": "Turbofan", "MTF": "Mixed Turbofan", "TP": "Turboprop"}.get(
        row.get("type", ""), row.get("type", ""))

    lines = [f"Engine: {name} by {mfr} ({etype})"]

    if row.get("max_thrust"):
        lines.append(f"Maximum Thrust: {row['max_thrust']} N")
    if row.get("bpr"):
        lines.append(f"Bypass Ratio: {row['bpr']}")
    if row.get("pr"):
        lines.append(f"Pressure Ratio: {row['pr']}")
    if row.get("cruise_thrust"):
        lines.append(f"Cruise Thrust: {row['cruise_thrust']} N")
    if row.get("cruise_sfc"):
        lines.append(f"Cruise SFC: {row['cruise_sfc']} kg/(N·s)")
    if row.get("cruise_mach"):
        lines.append(f"Cruise Mach: {row['cruise_mach']}")
    if row.get("cruise_alt"):
        lines.append(f"Cruise Altitude: {row['cruise_alt']} ft")
    if row.get("fuel_lto"):
        lines.append(f"LTO Fuel Flow: {row['fuel_lto']} kg")

    # Emission indices
    ei_fields = [("ei_nox_to", "NOx (takeoff)"), ("ei_co_to", "CO (takeoff)"),
                 ("ei_hc_to", "HC (takeoff)")]
    ei_parts = []
    for field, label in ei_fields:
        if row.get(field) and row[field] != "0.0":
            ei_parts.append(f"{label}: {row[field]} g/kg")
    if ei_parts:
        lines.append("Emission Indices: " + ", ".join(ei_parts))

    return "\n".join(lines)


def clean_fineweb_sample() -> list[dict]:
    """Clean FineWeb-EDU sample (general knowledge data).

    Stored as JSONL with 'text' field.
    """
    print("\n[FineWeb-EDU] Cleaning sample...")
    fw_dir = RAW_DIR / "fineweb_edu_sample"
    docs = []

    if not fw_dir.exists():
        print("  [FineWeb-EDU] Not found, skipping")
        return docs

    for jsonl_file in sorted(fw_dir.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = row.get("text", "")
                    text = normalize_whitespace(text)
                    if is_useful_text(text, min_chars=200):
                        docs.append({
                            "text": text,
                            "source": "fineweb_edu",
                            "domain": "general",
                            "chars": len(text),
                        })
                except json.JSONDecodeError:
                    continue

    print(f"  [FineWeb-EDU] {len(docs):,} documents, {sum(d['chars'] for d in docs):,} chars")
    return docs


# ---------------------------------------------------------------------------
# NTSB CSV/Zenodo narratives
# ---------------------------------------------------------------------------

def clean_ntsb_csv() -> list[dict]:
    """Clean NTSB narratives from Zenodo CSV download."""
    print("\n[NTSB CSV] Checking for Zenodo CSV narratives...")
    csv_dir = RAW_DIR / "ntsb" / "narratives_csv"
    docs = []

    if not csv_dir.exists():
        return docs

    import csv

    for csv_file in sorted(csv_dir.glob("*.csv")):
        try:
            with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try common column names for narrative text
                    text = ""
                    for col in ["narr_accp", "narr_accf", "narr_cause", "narrative",
                                "report_narrative", "factual_narrative"]:
                        if col in row and row[col]:
                            text += row[col] + "\n\n"

                    text = normalize_whitespace(text)
                    if is_useful_text(text, min_chars=200):
                        docs.append({
                            "text": text,
                            "source": "ntsb_csv",
                            "domain": "accident_reports",
                            "chars": len(text),
                        })
        except Exception as e:
            print(f"  [NTSB CSV] Error reading {csv_file.name}: {e}")

    if docs:
        print(f"  [NTSB CSV] {len(docs):,} documents, {sum(d['chars'] for d in docs):,} chars")
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CLEANERS = {
    "ntsb": clean_ntsb,
    "ntsb_csv": clean_ntsb_csv,
    "handbooks": clean_handbooks,
    "regulations": clean_regulations,
    "metar": clean_metar,
    "wikipedia": clean_wikipedia,
    "hf_datasets": clean_hf_datasets,
    "openap": clean_openap,
    "fineweb": clean_fineweb_sample,
}


def main():
    parser = argparse.ArgumentParser(description="FlightMind Data Cleaning Pipeline")
    parser.add_argument("--source", choices=list(CLEANERS.keys()),
                        help="Clean only this source (default: all)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print statistics without writing files")
    args = parser.parse_args()

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    sources = [args.source] if args.source else list(CLEANERS.keys())
    total_docs = 0
    total_chars = 0

    print("=" * 60)
    print("FlightMind Data Cleaning Pipeline")
    print("=" * 60)

    for source in sources:
        cleaner = CLEANERS[source]
        docs = cleaner()

        if docs and not args.stats_only:
            output_path = CLEANED_DIR / f"{source}.jsonl"
            write_jsonl(docs, output_path)

        total_docs += len(docs)
        total_chars += sum(d["chars"] for d in docs)

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_docs:,} documents, {total_chars:,} chars (~{total_chars // 4:,} tokens)")
    print("=" * 60)

    # Write summary
    if not args.stats_only:
        summary = {
            "total_documents": total_docs,
            "total_chars": total_chars,
            "estimated_tokens": total_chars // 4,
            "sources_processed": sources,
        }
        summary_path = CLEANED_DIR / "cleaning_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
