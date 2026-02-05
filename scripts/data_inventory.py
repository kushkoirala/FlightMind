"""
Data Inventory - Assess all collected data and estimate total tokens.

Scans data/raw/ and produces a summary report.
Handles both plain text files and HuggingFace Arrow datasets.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def dir_size_mb(directory: Path) -> float:
    """Get total size of directory in MB."""
    total = 0
    for f in directory.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def estimate_hf_dataset_chars(ds_path: Path) -> int:
    """Estimate text chars in a HuggingFace Arrow dataset."""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(ds_path))
        total = 0

        splits = ds.keys() if hasattr(ds, "keys") else {"train": ds}
        if not hasattr(ds, "keys"):
            splits = {"data": ds}

        for split_name, split in (ds.items() if hasattr(ds, "items") else [(None, ds)]):
            text_col = None
            for c in ["text", "transcript", "transcription", "sentence", "utterance"]:
                if c in split.column_names:
                    text_col = c
                    break
            if text_col:
                n = min(500, len(split))
                sample = split.select(range(n))
                sample_chars = sum(len(str(t)) for t in sample[text_col] if t)
                total += int(sample_chars / n * len(split))
        return total
    except Exception:
        return 0


def main():
    print("=" * 70)
    print("  AviationLM Data Inventory")
    print("=" * 70)
    print()

    inventory = {}

    # 1. FAA Handbooks
    hb_dir = RAW_DIR / "faa_handbooks"
    if hb_dir.exists():
        txt_dir = hb_dir / "txt"
        txt_chars = 0
        txt_count = 0
        if txt_dir.exists():
            for f in txt_dir.glob("*.txt"):
                txt_chars += f.stat().st_size
                txt_count += 1

        inventory["faa_handbooks"] = {
            "description": "FAA Handbooks (PHAK, IFH, AWH, etc.)",
            "documents": txt_count,
            "total_chars": txt_chars,
            "estimated_tokens": txt_chars // 4,
            "size_mb": round(dir_size_mb(hb_dir), 1),
        }
        print(f"  FAA Handbooks:      {txt_count:>5} docs     {txt_chars:>12,} chars  ~{txt_chars // 4:>10,} tokens")

    # 2. 14 CFR Regulations
    reg_dir = RAW_DIR / "faa_regulations"
    if reg_dir.exists():
        reg_chars = 0
        for f in reg_dir.rglob("*.txt"):
            reg_chars += f.stat().st_size

        inventory["faa_regulations"] = {
            "description": "14 CFR Federal Aviation Regulations",
            "total_chars": reg_chars,
            "estimated_tokens": reg_chars // 4,
            "size_mb": round(dir_size_mb(reg_dir), 1),
        }
        print(f"  14 CFR Regs:            1 corpus   {reg_chars:>12,} chars  ~{reg_chars // 4:>10,} tokens")

    # 3. NTSB
    ntsb_dir = RAW_DIR / "ntsb"
    if ntsb_dir.exists():
        manifest_path = ntsb_dir / "manifest.json"
        ntsb_chars = 0
        ntsb_narratives = 0
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stats = manifest.get("stats", {})
            ntsb_chars = stats.get("total_chars", 0)
            ntsb_narratives = stats.get("with_narrative", 0)

        # Also check narrative text files on disk
        narr_dir = ntsb_dir / "narratives"
        if narr_dir.exists():
            narr_chars = sum(f.stat().st_size for f in narr_dir.rglob("*.txt"))
            if narr_chars > ntsb_chars:
                ntsb_chars = narr_chars

        inventory["ntsb"] = {
            "description": "NTSB Accident Reports & Narratives",
            "narratives": ntsb_narratives,
            "total_chars": ntsb_chars,
            "estimated_tokens": ntsb_chars // 4,
            "size_mb": round(dir_size_mb(ntsb_dir), 1),
        }
        print(f"  NTSB:           {ntsb_narratives:>6,} reports   {ntsb_chars:>12,} chars  ~{ntsb_chars // 4:>10,} tokens")

    # 4. METAR
    metar_dir = RAW_DIR / "metar"
    if metar_dir.exists():
        metar_chars = 0
        metar_files = 0
        for f in metar_dir.rglob("*.txt"):
            metar_chars += f.stat().st_size
            metar_files += 1

        inventory["metar"] = {
            "description": "Historical METAR/TAF Weather Observations",
            "text_files": metar_files,
            "total_chars": metar_chars,
            "estimated_tokens": metar_chars // 4,
            "size_mb": round(dir_size_mb(metar_dir), 1),
        }
        print(f"  METAR:           {metar_files:>5} files    {metar_chars:>12,} chars  ~{metar_chars // 4:>10,} tokens")

    # 5. ATC Transcripts (HuggingFace Arrow datasets)
    atc_dir = RAW_DIR / "atc_transcripts"
    if atc_dir.exists():
        atc_chars = 0
        atc_datasets = 0
        for ds_dir in sorted(atc_dir.iterdir()):
            if ds_dir.is_dir():
                chars = estimate_hf_dataset_chars(ds_dir)
                atc_chars += chars
                atc_datasets += 1

        inventory["atc_transcripts"] = {
            "description": "ATC Transcripts (HuggingFace: ATCO2, UWB-ATCC, etc.)",
            "datasets": atc_datasets,
            "total_chars": atc_chars,
            "estimated_tokens": atc_chars // 4,
            "size_mb": round(dir_size_mb(atc_dir), 1),
        }
        print(f"  ATC Transcripts: {atc_datasets:>5} datasets  {atc_chars:>12,} chars  ~{atc_chars // 4:>10,} tokens")

    # 6. Aviation Knowledge Dataset (kathleenge/aviation in HF Arrow)
    av_ds_dir = RAW_DIR / "aircraft_performance" / "aviation_dataset"
    if av_ds_dir.exists():
        av_chars = estimate_hf_dataset_chars(av_ds_dir)

        inventory["aviation_knowledge_hf"] = {
            "description": "Aviation Knowledge (kathleenge/aviation HF dataset)",
            "total_chars": av_chars,
            "estimated_tokens": av_chars // 4,
        }
        print(f"  Aviation HF DS:      388K rows    {av_chars:>12,} chars  ~{av_chars // 4:>10,} tokens")

    # 7. Aircraft Performance (OpenAP, FAA Registry)
    ap_dir = RAW_DIR / "aircraft_performance"
    if ap_dir.exists():
        # Count only text/json in the openap directory, excluding the HF dataset
        ap_chars = 0
        for f in ap_dir.rglob("*"):
            if f.is_file() and f.suffix in (".txt", ".csv", ".json", ".py", ".yaml"):
                # Skip HF Arrow dataset files
                if "aviation_dataset" not in str(f):
                    ap_chars += f.stat().st_size

        inventory["aircraft_performance"] = {
            "description": "Aircraft Performance Data (OpenAP, FAA Registry)",
            "total_chars": ap_chars,
            "estimated_tokens": ap_chars // 4,
            "size_mb": round(dir_size_mb(ap_dir), 1),
        }
        print(f"  Aircraft Perf:       OpenAP+FAA   {ap_chars:>12,} chars  ~{ap_chars // 4:>10,} tokens")

    # 8. FineWeb-EDU Sample (JSONL)
    fw_dir = RAW_DIR / "fineweb_edu_sample"
    if fw_dir.exists():
        fw_chars = 0
        fw_count = 0
        jsonl_path = fw_dir / "sample_10k.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    fw_chars += len(doc.get("text", ""))
                    fw_count += 1

        inventory["fineweb_edu"] = {
            "description": "FineWeb-EDU General Knowledge Sample",
            "documents": fw_count,
            "total_chars": fw_chars,
            "estimated_tokens": fw_chars // 4,
            "size_mb": round(dir_size_mb(fw_dir), 1),
        }
        print(f"  FineWeb-EDU:     {fw_count:>5} docs     {fw_chars:>12,} chars  ~{fw_chars // 4:>10,} tokens")

    # 9. Wikipedia
    wiki_dir = RAW_DIR / "wikipedia_aviation"
    if wiki_dir.exists():
        wiki_chars = 0
        wiki_articles = 0
        articles_dir = wiki_dir / "articles"
        if articles_dir.exists():
            for f in articles_dir.glob("*.txt"):
                wiki_chars += f.stat().st_size
                wiki_articles += 1

        inventory["wikipedia"] = {
            "description": "Wikipedia Aviation Articles",
            "articles": wiki_articles,
            "total_chars": wiki_chars,
            "estimated_tokens": wiki_chars // 4,
            "size_mb": round(dir_size_mb(wiki_dir), 1),
        }
        status = f"{wiki_articles:,} articles" if wiki_articles else "(downloading)"
        print(f"  Wikipedia:      {status:>14s}  {wiki_chars:>12,} chars  ~{wiki_chars // 4:>10,} tokens")

    # Summary
    print()
    print("-" * 70)

    total_chars = sum(v.get("total_chars", 0) for v in inventory.values())
    total_tokens = total_chars // 4
    total_size_mb = 0
    for v in inventory.values():
        total_size_mb += v.get("size_mb", 0)

    fw_tokens = inventory.get("fineweb_edu", {}).get("estimated_tokens", 0)
    aviation_tokens = total_tokens - fw_tokens

    print(f"  TOTAL:                              {total_chars:>12,} chars  ~{total_tokens:>10,} tokens")
    print(f"  Disk usage: {total_size_mb:>10.1f} MB")
    print()
    print(f"  Aviation-specific tokens:  ~{aviation_tokens:>12,}")
    print(f"  General knowledge (local): ~{fw_tokens:>12,} (10K doc sample)")
    print(f"  General knowledge (full):  FineWeb-EDU will be streamed during training")
    print()

    # Model size implications
    print("  Model sizing (Chinchilla 20:1 rule, aviation data only):")
    for name, params in [("d8 (88M)", 88e6), ("d12 (184M)", 184e6),
                         ("d16 (332M)", 332e6), ("d20 (561M)", 561e6)]:
        needed = int(params * 20)
        have_pct = min(100, (aviation_tokens / needed) * 100)
        print(f"    {name}: needs {needed / 1e9:.1f}B tokens, have {have_pct:.1f}% for aviation midtraining")

    print()
    print("  Note: General pretraining uses FineWeb-EDU (1.3T tokens, streamed).")
    print("  Aviation data is used for midtraining (Phase 2) where ~200M tokens")
    print("  is substantial for domain adaptation of a pretrained model.")
    print()
    print("=" * 70)

    # Save inventory
    inventory_path = PROJECT_ROOT / "data" / "inventory.json"
    inventory["_summary"] = {
        "total_chars": total_chars,
        "estimated_tokens": total_tokens,
        "total_size_mb": round(total_size_mb, 1),
        "aviation_tokens": aviation_tokens,
        "general_tokens_local": fw_tokens,
    }
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"  Inventory saved to: {inventory_path}")


if __name__ == "__main__":
    main()
