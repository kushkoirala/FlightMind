"""
Master data collection script.

Runs all individual collectors in sequence with progress tracking.
Each collector is independent and can be re-run safely (idempotent).

Usage:
    python scripts/collect/collect_all.py              # Run all collectors
    python scripts/collect/collect_all.py --only faa   # Run only FAA collectors
    python scripts/collect/collect_all.py --skip metar  # Skip METAR collection
"""

import sys
import json
import time
import argparse
import importlib
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "collect"))

COLLECTORS = [
    {
        "name": "faa_handbooks",
        "module": "collect_faa_handbooks",
        "description": "FAA Handbooks (PHAK, AFH, IFH, etc.)",
        "tags": ["faa"],
    },
    {
        "name": "faa_regulations",
        "module": "collect_faa_regulations",
        "description": "14 CFR Federal Aviation Regulations (XML)",
        "tags": ["faa"],
    },
    {
        "name": "ntsb",
        "module": "collect_ntsb",
        "description": "NTSB accident reports and narratives",
        "tags": ["safety"],
    },
    {
        "name": "asrs",
        "module": "collect_asrs",
        "description": "NASA ASRS voluntary safety reports",
        "tags": ["safety"],
    },
    {
        "name": "metar",
        "module": "collect_metar",
        "description": "Historical METAR/TAF weather observations",
        "tags": ["weather"],
    },
    {
        "name": "hf_datasets",
        "module": "collect_hf_datasets",
        "description": "HuggingFace aviation datasets (ATC, performance)",
        "tags": ["hf"],
    },
    {
        "name": "wikipedia",
        "module": "collect_wikipedia",
        "description": "Wikipedia aviation articles",
        "tags": ["general"],
    },
]


def run_collector(collector: dict) -> dict:
    """Run a single collector and return results."""
    name = collector["name"]
    module_name = collector["module"]

    print(f"\n{'=' * 70}")
    print(f"  COLLECTOR: {name}")
    print(f"  {collector['description']}")
    print(f"{'=' * 70}\n")

    start = time.time()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - start
        return {"name": name, "status": "ok", "elapsed_s": round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ERROR in {name}: {e}")
        return {"name": name, "status": "error", "error": str(e),
                "elapsed_s": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(description="Run all data collectors")
    parser.add_argument("--only", nargs="+", help="Only run these collectors")
    parser.add_argument("--skip", nargs="+", help="Skip these collectors")
    parser.add_argument("--tag", help="Only run collectors with this tag")
    args = parser.parse_args()

    collectors = COLLECTORS

    if args.only:
        collectors = [c for c in collectors if c["name"] in args.only]
    if args.skip:
        collectors = [c for c in collectors if c["name"] not in args.skip]
    if args.tag:
        collectors = [c for c in collectors if args.tag in c["tags"]]

    print(f"AviationLM Data Collection Pipeline")
    print(f"{'=' * 70}")
    print(f"Collectors to run: {len(collectors)}")
    for c in collectors:
        print(f"  - {c['name']}: {c['description']}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    results = []
    for collector in collectors:
        result = run_collector(collector)
        results.append(result)

    # Summary
    print(f"\n\n{'=' * 70}")
    print(f"  COLLECTION SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        status = "OK" if r["status"] == "ok" else "FAILED"
        print(f"  [{status:6s}] {r['name']:25s} ({r['elapsed_s']:.1f}s)")

    total_time = sum(r["elapsed_s"] for r in results)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n  Total: {ok}/{len(results)} succeeded in {total_time:.0f}s")

    # Save run log
    log = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    log_path = PROJECT_ROOT / "data" / "collection_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
