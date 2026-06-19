"""
Download 14 CFR (Federal Aviation Regulations) in XML format.

Source: GovInfo bulk data repository
The full Title 14 of the Code of Federal Regulations is available as
structured XML from the Government Publishing Office (GPO).

This gives us the complete text of all FAR parts:
  Part 1    - Definitions
  Part 23   - Normal Category Airplanes
  Part 25   - Transport Category Airplanes
  Part 27   - Normal Category Rotorcraft
  Part 33   - Aircraft Engines
  Part 35   - Propellers
  Part 43   - Maintenance
  Part 61   - Certification: Pilots
  Part 67   - Medical Standards
  Part 71   - Airspace
  Part 91   - General Operating Rules
  Part 97   - Instrument Approaches
  Part 107  - Small UAS
  Part 119  - Air Carrier Certification
  Part 121  - Air Carrier Operations
  Part 135  - Commuter/On-Demand Operations
  Part 141  - Pilot Schools
  ... and many more

Output: data/raw/faa_regulations/
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
import requests
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "faa_regulations"

# GovInfo bulk XML for Title 14
# Format: https://www.govinfo.gov/bulkdata/CFR/{year}/title-14
GOVINFO_BASE = "https://www.govinfo.gov/bulkdata/CFR"
CFR_YEAR = 2025  # Most recent available
CFR_TITLE = 14

# Title 14 has multiple volumes
VOLUMES = [1, 2, 3, 4, 5]


def download_cfr_xml(year: int, title: int, volume: int, output_dir: Path) -> Path:
    """Download a CFR XML volume from GovInfo."""
    filename = f"CFR-{year}-title{title}-vol{volume}.xml"
    url = f"{GOVINFO_BASE}/{year}/title-{title}/{filename}"
    dest = output_dir / filename

    if dest.exists():
        print(f"  [SKIP] {filename} (exists)")
        return dest

    print(f"  [GET]  {filename}...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [OK]   {filename} ({size_mb:.1f} MB)")
        return dest
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  [SKIP] {filename} (not found, trying {year - 1})")
            # Try previous year
            return download_cfr_xml(year - 1, title, volume, output_dir)
        print(f"  [FAIL] {filename}: {e}")
        return None
    except Exception as e:
        print(f"  [FAIL] {filename}: {e}")
        return None


def extract_text_from_cfr_xml(xml_path: Path) -> list[dict]:
    """Extract regulation text from CFR XML."""
    sections = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # CFR XML structure: CFRDOC > TITLE > CHAPTER > SUBCHAPTER > PART > SUBPART > SECTION
        # We want to extract section-level text
        for elem in root.iter():
            tag = elem.tag

            if tag == "SECTION":
                sectno = ""
                subject = ""
                text_parts = []

                for child in elem:
                    if child.tag == "SECTNO":
                        sectno = (child.text or "").strip()
                    elif child.tag == "SUBJECT":
                        subject = (child.text or "").strip()
                    elif child.tag == "P":
                        p_text = "".join(child.itertext()).strip()
                        if p_text:
                            text_parts.append(p_text)

                if sectno and text_parts:
                    sections.append({
                        "section": sectno,
                        "subject": subject,
                        "text": "\n".join(text_parts),
                    })

            elif tag == "PART":
                # Extract part-level info
                for child in elem:
                    if child.tag == "HD" and child.text:
                        pass  # Part heading

    except ET.ParseError as e:
        print(f"    XML parse error in {xml_path.name}: {e}")

    return sections


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    xml_dir = RAW_DIR / "xml"
    txt_dir = RAW_DIR / "txt"
    xml_dir.mkdir(exist_ok=True)
    txt_dir.mkdir(exist_ok=True)

    print("14 CFR Regulations Collector")
    print(f"{'=' * 60}")
    print(f"Output: {RAW_DIR}")
    print(f"Year: {CFR_YEAR}, Title: {CFR_TITLE}")
    print()

    # Phase 1: Download XML volumes
    print("Phase 1: Downloading CFR XML volumes...")
    xml_files = []
    for vol in VOLUMES:
        result = download_cfr_xml(CFR_YEAR, CFR_TITLE, vol, xml_dir)
        if result:
            xml_files.append(result)

    # Also try eCFR API as fallback
    print(f"\nDownloaded {len(xml_files)} XML volumes.")

    # Phase 2: Extract text
    print("\nPhase 2: Extracting regulation text...")
    all_sections = []
    for xml_file in xml_files:
        print(f"  Processing {xml_file.name}...")
        sections = extract_text_from_cfr_xml(xml_file)
        all_sections.extend(sections)
        print(f"    Extracted {len(sections)} sections")

    # Write combined text
    combined_path = txt_dir / "14cfr_combined.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        for sec in all_sections:
            f.write(f"=== {sec['section']} - {sec['subject']} ===\n")
            f.write(sec["text"])
            f.write("\n\n")

    total_chars = sum(len(s["text"]) for s in all_sections)

    # Save manifest
    manifest = {
        "source": "14 CFR - Code of Federal Regulations, Title 14 (Aeronautics and Space)",
        "year": CFR_YEAR,
        "url": f"{GOVINFO_BASE}/{CFR_YEAR}/title-{CFR_TITLE}",
        "volumes_downloaded": len(xml_files),
        "sections_extracted": len(all_sections),
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 4,
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Sections: {len(all_sections):,}")
    print(f"Total text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")


if __name__ == "__main__":
    main()
