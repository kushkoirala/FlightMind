"""
Download aviation datasets from HuggingFace.

Covers:
- ATC transcripts (ATCO2, UWB-ATCC, jacktol/atc-dataset)
- Aircraft performance data
- Aviation safety datasets

Also downloads general-purpose pretraining data:
- FineWeb-EDU sample for general knowledge

Output: data/raw/atc_transcripts/  and  data/raw/aircraft_performance/
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# HuggingFace datasets to download
DATASETS = {
    "atc_transcripts": {
        "datasets": [
            {
                "name": "uwb_atcc",
                "repo": "Jzuluaga/uwb_atcc",
                "description": "UWB Air Traffic Control Communications corpus (~138 hours)",
            },
            {
                "name": "atco2_1h",
                "repo": "Jzuluaga/atco2_corpus_1h",
                "description": "ATCO2 test set (1 hour, gold annotations)",
            },
            {
                "name": "atc_dataset",
                "repo": "jacktol/atc-dataset",
                "description": "Combined ATC transcriptions for ASR fine-tuning",
            },
            {
                "name": "atcosim",
                "repo": "Jzuluaga/atcosim_corpus",
                "description": "ATC simulation corpus",
            },
        ],
        "output_dir": "data/raw/atc_transcripts",
    },
    "aircraft_performance": {
        "datasets": [
            {
                "name": "aviation_dataset",
                "repo": "kathleenge/aviation",
                "description": "General aviation dataset",
            },
        ],
        "output_dir": "data/raw/aircraft_performance",
    },
}


def download_hf_dataset(repo: str, output_dir: Path, name: str) -> dict:
    """Download a dataset from HuggingFace using the datasets library."""
    dest = output_dir / name

    if dest.exists() and any(dest.iterdir()):
        print(f"  [SKIP] {repo} (already downloaded)")
        return {"repo": repo, "status": "skipped"}

    dest.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        print(f"  [GET]  {repo}...")
        ds = load_dataset(repo, trust_remote_code=True)

        # Save to disk
        ds.save_to_disk(str(dest))

        # Count samples
        total = 0
        if hasattr(ds, "num_rows"):
            total = sum(ds.num_rows.values()) if isinstance(ds.num_rows, dict) else ds.num_rows
        else:
            for split in ds:
                total += len(ds[split])

        print(f"  [OK]   {repo}: {total:,} samples")
        return {"repo": repo, "status": "ok", "samples": total, "path": str(dest)}

    except ImportError:
        print(f"  [FAIL] datasets library not installed. Run: pip install datasets")
        return {"repo": repo, "status": "error", "error": "datasets not installed"}
    except Exception as e:
        print(f"  [FAIL] {repo}: {e}")
        return {"repo": repo, "status": "error", "error": str(e)}


def download_fineweb_edu_sample(output_dir: Path) -> dict:
    """
    Download a small sample of FineWeb-EDU for local testing.
    The full dataset (1.3T tokens) will be downloaded directly on the cloud GPU node.
    """
    dest = output_dir / "fineweb_edu_sample"

    if dest.exists() and any(dest.iterdir()):
        print(f"  [SKIP] FineWeb-EDU sample (already downloaded)")
        return {"status": "skipped"}

    dest.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        print(f"  [GET]  FineWeb-EDU sample (first 10K rows for testing)...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        # Take first 10K samples for local testing
        samples = []
        for i, sample in enumerate(ds):
            if i >= 10000:
                break
            samples.append(sample)

        # Save as JSONL
        jsonl_path = dest / "sample_10k.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        total_chars = sum(len(s.get("text", "")) for s in samples)
        print(f"  [OK]   FineWeb-EDU sample: {len(samples):,} docs, {total_chars:,} chars")
        return {"status": "ok", "samples": len(samples), "total_chars": total_chars}

    except Exception as e:
        print(f"  [FAIL] FineWeb-EDU sample: {e}")
        return {"status": "error", "error": str(e)}


def main():
    print("HuggingFace Dataset Collector")
    print(f"{'=' * 60}")
    print()

    all_results = {}

    for category, config in DATASETS.items():
        output_dir = PROJECT_ROOT / config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Category: {category}")

        for ds_info in config["datasets"]:
            result = download_hf_dataset(ds_info["repo"], output_dir, ds_info["name"])
            all_results[ds_info["name"]] = result

    # Download FineWeb-EDU sample for local testing
    print(f"\nGeneral pretraining data (sample for testing):")
    fineweb_dir = PROJECT_ROOT / "data" / "raw"
    fineweb_result = download_fineweb_edu_sample(fineweb_dir)
    all_results["fineweb_edu_sample"] = fineweb_result

    # Save manifest
    manifest = {
        "source": "HuggingFace Datasets",
        "datasets": all_results,
    }

    manifest_path = PROJECT_ROOT / "data" / "raw" / "hf_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"HuggingFace collection complete.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
