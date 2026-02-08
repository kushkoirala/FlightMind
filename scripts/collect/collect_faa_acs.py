"""
Download FAA Advisory Circulars (ACs) related to aviation operations.

FAA Advisory Circulars provide guidance and information on aviation topics.
They cover aircraft design, maintenance, operations, airworthiness, etc.

Uses the FAA's document library search to find and download ACs.

Output: data/raw/faa_acs/
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "faa_acs"

# Key FAA Advisory Circulars - manually curated list of most useful ones
# These are freely available on the FAA website
AC_LIST = [
    # Operations
    {"id": "AC 91-73B", "name": "Parts 91 and 135 Single-Pilot Procedures During Taxi Operations",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_91-73B.pdf"},
    {"id": "AC 91-74B", "name": "Pilot Guide: Flight in Icing Conditions",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_91-74B.pdf"},
    {"id": "AC 00-6B", "name": "Aviation Weather",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_00-6B.pdf"},
    {"id": "AC 00-45H", "name": "Aviation Weather Services",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_00-45H_CHG_1_2.pdf"},
    {"id": "AC 00-24C", "name": "Thunderstorms",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_00-24C.pdf"},
    {"id": "AC 61-67C", "name": "Stall and Spin Awareness Training",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_61-67C.pdf"},
    {"id": "AC 61-98D", "name": "Currency Requirements and Guidance for the Flight Review",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_61-98D.pdf"},
    {"id": "AC 90-48D", "name": "Pilots' Role in Collision Avoidance",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_90-48D.pdf"},
    {"id": "AC 120-92B", "name": "Safety Management Systems for Aviation Service Providers",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-92B.pdf"},
    {"id": "AC 20-117", "name": "Hazards Following Ground Deicing and Ground Operations in Conditions Conducive to Aircraft Icing",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_20-117.pdf"},
    # Maintenance & Airworthiness
    {"id": "AC 43.13-1B", "name": "Acceptable Methods, Techniques, and Practices - Aircraft Inspection and Repair",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43.13-1B_w-chg1.pdf"},
    {"id": "AC 43.13-2B", "name": "Acceptable Methods, Techniques, and Practices - Aircraft Alterations",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/ac_43_13-2b.pdf"},
    {"id": "AC 43-4A", "name": "Corrosion Control for Aircraft",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43-4A.pdf"},
    # Airports
    {"id": "AC 150/5300-13B", "name": "Airport Design",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/150-5300-13B-Airport-Design.pdf"},
    # ATC & Airspace
    {"id": "AC 90-66B", "name": "Non-Towered Airport Flight Operations",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_90-66B.pdf"},
    {"id": "AC 91-70B", "name": "Oceanic and International Operations",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_91-70B_CHG_1.pdf"},
]


def download_ac_pdf(ac: dict, pdf_dir: Path) -> dict:
    """Download a single AC PDF."""
    safe_id = ac["id"].replace("/", "-").replace(" ", "_")
    pdf_path = pdf_dir / f"{safe_id}.pdf"

    if pdf_path.exists():
        return {"id": ac["id"], "status": "skipped", "path": str(pdf_path)}

    try:
        resp = requests.get(ac["url"], timeout=120, stream=True,
                            headers={"User-Agent": "FlightMind/0.1"})
        resp.raise_for_status()

        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        return {"id": ac["id"], "status": "downloaded", "path": str(pdf_path),
                "size_mb": round(size_mb, 1)}

    except Exception as e:
        return {"id": ac["id"], "status": "failed", "error": str(e)}


def extract_text(pdf_path: Path, txt_path: Path) -> int:
    """Extract text from PDF. Returns char count."""
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return txt_path.stat().st_size

    try:
        import pdfplumber
    except ImportError:
        print("  pdfplumber not installed. Run: pip install pdfplumber")
        return 0

    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        print(f"  Error extracting {pdf_path.name}: {e}")
        return 0

    full_text = "\n\n".join(text_parts)
    txt_path.write_text(full_text, encoding="utf-8")
    return len(full_text)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_dir = RAW_DIR / "pdf"
    txt_dir = RAW_DIR / "txt"
    pdf_dir.mkdir(exist_ok=True)
    txt_dir.mkdir(exist_ok=True)

    print("FAA Advisory Circulars Collector")
    print(f"{'=' * 60}")
    print(f"Advisory Circulars: {len(AC_LIST)}")
    print(f"Output: {RAW_DIR}")
    print()

    # Phase 1: Download PDFs
    print("Phase 1: Downloading AC PDFs...")
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(download_ac_pdf, ac, pdf_dir): ac
            for ac in AC_LIST
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = result["status"]
            ac_id = result["id"]
            if status == "downloaded":
                print(f"  [OK]   {ac_id} ({result.get('size_mb', '?')} MB)")
            elif status == "failed":
                print(f"  [FAIL] {ac_id}: {result.get('error', '?')}")
            else:
                print(f"  [SKIP] {ac_id}")

    downloaded = sum(1 for r in results if r["status"] in ("downloaded", "skipped"))
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\nDownloaded: {downloaded}, Failed: {failed}")

    # Phase 2: Extract text
    print("\nPhase 2: Extracting text from PDFs...")
    total_chars = 0
    for ac in AC_LIST:
        safe_id = ac["id"].replace("/", "-").replace(" ", "_")
        pdf_path = pdf_dir / f"{safe_id}.pdf"
        txt_path = txt_dir / f"{safe_id}.txt"

        if pdf_path.exists():
            chars = extract_text(pdf_path, txt_path)
            if chars > 0:
                print(f"  {ac['id']}: {chars:,} chars")
            total_chars += chars

    # Save manifest
    manifest = {
        "source": "FAA Advisory Circulars",
        "url": "https://www.faa.gov/regulations_policies/advisory_circulars",
        "advisory_circulars": AC_LIST,
        "total_downloaded": downloaded,
        "total_failed": failed,
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 4,
    }
    (RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print(f"Total text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")


if __name__ == "__main__":
    main()
