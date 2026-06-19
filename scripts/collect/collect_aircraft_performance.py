"""
Download aircraft performance and specification data.

Sources:
1. FAA Aircraft Registry (bulk CSV) — 300K+ registered aircraft
2. FAA Aircraft Characteristics Database — dimensions, approach speeds
3. OpenAP (TU Delft) — aerodynamic models for 100+ aircraft types
4. Kaggle datasets — Aircraft Bluebook performance data

Output: data/raw/aircraft_performance/
"""

import os
import sys
import json
import zipfile
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "aircraft_performance"

# FAA Aircraft Registry
FAA_REGISTRY_URL = "https://registry.faa.gov/database/ReleasableAircraft.zip"

# OpenAP GitHub
OPENAP_REPO = "https://github.com/junzis/openap"


def download_faa_registry(output_dir: Path) -> dict:
    """Download the FAA Aircraft Registry bulk data."""
    zip_path = output_dir / "ReleasableAircraft.zip"
    csv_dir = output_dir / "faa_registry"

    if csv_dir.exists() and any(csv_dir.glob("*.txt")):
        print("  [SKIP] FAA Registry (already extracted)")
        return {"status": "skipped"}

    print("  [GET]  FAA Aircraft Registry...")
    try:
        resp = requests.get(FAA_REGISTRY_URL, timeout=300, stream=True)
        resp.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  [OK]   Downloaded ({size_mb:.1f} MB)")

        # Extract
        csv_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(csv_dir)
        print(f"  [OK]   Extracted to {csv_dir}")

        # Count records
        master_file = csv_dir / "MASTER.txt"
        if master_file.exists():
            lines = sum(1 for _ in open(master_file, encoding="latin-1"))
            print(f"  [OK]   {lines:,} aircraft registrations")
            return {"status": "ok", "records": lines}

        return {"status": "ok"}
    except Exception as e:
        print(f"  [FAIL] FAA Registry: {e}")
        return {"status": "error", "error": str(e)}


def download_openap(output_dir: Path) -> dict:
    """Clone or download OpenAP aircraft performance data."""
    openap_dir = output_dir / "openap"

    if openap_dir.exists() and any(openap_dir.iterdir()):
        print("  [SKIP] OpenAP (already downloaded)")
        return {"status": "skipped"}

    print("  [GET]  OpenAP (TU Delft aircraft performance models)...")
    try:
        import subprocess
        result = subprocess.run(
            ["git", "clone", "--depth", "1", f"{OPENAP_REPO}.git", str(openap_dir)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print(f"  [OK]   Cloned OpenAP")
            return {"status": "ok"}
        else:
            print(f"  [WARN] Git clone failed, trying pip install...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "openap"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                print(f"  [OK]   Installed openap via pip")
                return {"status": "ok", "method": "pip"}
            else:
                print(f"  [FAIL] Could not install OpenAP")
                return {"status": "error", "error": result.stderr[:200]}
    except Exception as e:
        print(f"  [FAIL] OpenAP: {e}")
        return {"status": "error", "error": str(e)}


def extract_openap_data(output_dir: Path) -> dict:
    """Extract aircraft data from OpenAP into text format."""
    try:
        import openap

        txt_path = output_dir / "openap_aircraft_data.txt"
        if txt_path.exists():
            return {"status": "skipped"}

        lines = []
        # Get list of available aircraft
        try:
            from openap import prop
            aircraft_list = prop.available_aircraft()
        except Exception:
            aircraft_list = []

        for ac in aircraft_list:
            try:
                aircraft = openap.prop.aircraft(ac)
                lines.append(f"=== {ac.upper()} ===")
                for key, value in aircraft.items():
                    lines.append(f"  {key}: {value}")
                lines.append("")
            except Exception:
                pass

        if lines:
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  [OK]   Extracted {len(aircraft_list)} aircraft from OpenAP")
            return {"status": "ok", "aircraft_count": len(aircraft_list)}

        return {"status": "no_data"}
    except ImportError:
        print("  [SKIP] OpenAP not installed, skipping extraction")
        return {"status": "skipped"}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Aircraft Performance Data Collector")
    print(f"{'=' * 60}")
    print(f"Output: {RAW_DIR}")
    print()

    results = {}

    # FAA Registry
    print("1. FAA Aircraft Registry")
    results["faa_registry"] = download_faa_registry(RAW_DIR)

    # OpenAP
    print("\n2. OpenAP Aircraft Performance Models")
    results["openap_download"] = download_openap(RAW_DIR)
    results["openap_extract"] = extract_openap_data(RAW_DIR)

    # Save manifest
    manifest = {
        "source": "Aircraft Performance Data",
        "components": {
            "faa_registry": {
                "description": "FAA Aircraft Registration Database",
                "url": FAA_REGISTRY_URL,
            },
            "openap": {
                "description": "OpenAP Aircraft Performance (TU Delft)",
                "url": OPENAP_REPO,
            },
        },
        "results": results,
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Aircraft performance data collection complete.")


if __name__ == "__main__":
    main()
