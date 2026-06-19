"""
Download FAA Handbooks and extract text.

These are freely available PDFs from the FAA website covering:
- Pilot's Handbook of Aeronautical Knowledge (PHAK)
- Airplane Flying Handbook (AFH)
- Instrument Flying Handbook (IFH)
- Instrument Procedures Handbook (IPH)
- Aviation Weather Handbook
- Weight and Balance Handbook
- Risk Management Handbook
- Advanced Avionics Handbook
- Aviation Instructor's Handbook
- AMT Handbooks (General, Airframe, Powerplant)
- Helicopter, Glider, Balloon handbooks
- And more

Output: data/raw/faa_handbooks/*.txt (one file per handbook)
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "faa_handbooks"

# FAA Handbook URLs - chapter-level PDFs
# Source: https://www.faa.gov/regulations_policies/handbooks_manuals/aviation
HANDBOOKS = {
    "phak": {
        "name": "Pilot's Handbook of Aeronautical Knowledge",
        "faa_id": "FAA-H-8083-25C",
        "url": "https://www.faa.gov/sites/faa.gov/files/2022-03/pilot_handbook.pdf",
    },
    "ifh": {
        "name": "Instrument Flying Handbook",
        "faa_id": "FAA-H-8083-15B",
        "url": "https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/FAA-H-8083-15B.pdf",
    },
    "iph": {
        "name": "Instrument Procedures Handbook",
        "faa_id": "FAA-H-8083-16B",
        "url": "https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/instrument_procedures_handbook/FAA-H-8083-16B.pdf",
    },
    "awh": {
        "name": "Aviation Weather Handbook",
        "faa_id": "FAA-H-8083-28A",
        "url": "https://www.faa.gov/sites/faa.gov/files/FAA-H-8083-28A_FAA_Web.pdf",
    },
    "wbh": {
        "name": "Weight and Balance Handbook",
        "faa_id": "FAA-H-8083-1B",
        "url": "https://www.faa.gov/sites/faa.gov/files/2023-09/Weight_Balance_Handbook.pdf",
    },
    "rmh": {
        "name": "Risk Management Handbook",
        "faa_id": "FAA-H-8083-2A",
        "url": "https://www.faa.gov/sites/faa.gov/files/2022-06/risk_management_handbook_2A.pdf",
    },
    "aah": {
        "name": "Advanced Avionics Handbook",
        "faa_id": "FAA-H-8083-6",
        "url": "https://www.govinfo.gov/content/pkg/GOVPUB-TD4-PURL-gpo46261/pdf/GOVPUB-TD4-PURL-gpo46261.pdf",
    },
    "hfh": {
        "name": "Helicopter Flying Handbook",
        "faa_id": "FAA-H-8083-21B",
        "url": "https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/faa-h-8083-21.pdf",
    },
    "gfh": {
        "name": "Glider Flying Handbook",
        "faa_id": "FAA-H-8083-13A",
        "url": "https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/glider_handbook/faa-h-8083-13a.pdf",
    },
    "amt_airframe": {
        "name": "AMT Handbook - Airframe",
        "faa_id": "FAA-H-8083-31B",
        "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/FAA-H-8083-31B_Aviation_Maintenance_Technician_Handbook.pdf",
    },
}


def download_pdf(key: str, info: dict, output_dir: Path) -> dict:
    """Download a single handbook PDF."""
    pdf_path = output_dir / f"{key}.pdf"

    if pdf_path.exists():
        print(f"  [SKIP] {info['name']} (already downloaded)")
        return {"key": key, "status": "skipped", "path": str(pdf_path)}

    print(f"  [GET]  {info['name']}...")
    try:
        resp = requests.get(info["url"], timeout=120, stream=True)
        resp.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  [OK]   {info['name']} ({size_mb:.1f} MB)")
        return {"key": key, "status": "downloaded", "path": str(pdf_path), "size_mb": size_mb}
    except Exception as e:
        print(f"  [FAIL] {info['name']}: {e}")
        return {"key": key, "status": "failed", "error": str(e)}


def extract_text(pdf_path: Path, txt_path: Path) -> int:
    """Extract text from PDF using pdfplumber. Returns char count."""
    try:
        import pdfplumber
    except ImportError:
        print("  pdfplumber not installed. Run: pip install pdfplumber")
        return 0

    if txt_path.exists() and txt_path.stat().st_size > 0:
        return txt_path.stat().st_size

    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    full_text = "\n\n".join(text_parts)
    txt_path.write_text(full_text, encoding="utf-8")
    return len(full_text)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_dir = RAW_DIR / "pdf"
    txt_dir = RAW_DIR / "txt"
    pdf_dir.mkdir(exist_ok=True)
    txt_dir.mkdir(exist_ok=True)

    print(f"FAA Handbooks Collector")
    print(f"{'=' * 60}")
    print(f"Output: {RAW_DIR}")
    print(f"Handbooks: {len(HANDBOOKS)}")
    print()

    # Phase 1: Download PDFs
    print("Phase 1: Downloading PDFs...")
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(download_pdf, key, info, pdf_dir): key
            for key, info in HANDBOOKS.items()
        }
        for future in as_completed(futures):
            results.append(future.result())

    downloaded = sum(1 for r in results if r["status"] in ("downloaded", "skipped"))
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\nDownloaded: {downloaded}, Failed: {failed}")

    # Phase 2: Extract text
    print("\nPhase 2: Extracting text from PDFs...")
    stats = {}
    for key, info in HANDBOOKS.items():
        pdf_path = pdf_dir / f"{key}.pdf"
        txt_path = txt_dir / f"{key}.txt"
        if pdf_path.exists():
            chars = extract_text(pdf_path, txt_path)
            stats[key] = {"name": info["name"], "chars": chars}
            print(f"  {info['name']}: {chars:,} chars")

    # Save manifest
    manifest = {
        "source": "FAA Handbooks",
        "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation",
        "handbooks": {k: {**v, "stats": stats.get(k, {})} for k, v in HANDBOOKS.items()},
        "total_chars": sum(s["chars"] for s in stats.values()),
    }
    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_chars = manifest["total_chars"]
    print(f"\n{'=' * 60}")
    print(f"Total text extracted: {total_chars:,} chars (~{total_chars // 4:,} tokens)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
