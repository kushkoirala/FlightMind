"""
Download NASA Technical Reports related to aviation.

NASA Technical Reports Server (NTRS) provides free access to
NASA's research output. Aviation-relevant topics include:
- Aerodynamics, flight dynamics, propulsion
- Aviation safety (ASAP, CAST studies)
- Air traffic management (NextGen)
- Weather hazards, icing, turbulence
- Human factors in aviation
- Unmanned aircraft systems (UAS)

Uses the NTRS public API (no authentication required).

Output: data/raw/nasa_ntrs/
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nasa_ntrs"

NTRS_API = "https://ntrs.nasa.gov/api"
HEADERS = {
    "User-Agent": "FlightMind/0.1 (aviation LLM research; educational/non-commercial)",
}

# Search queries to cover aviation topics
SEARCH_QUERIES = [
    "aviation safety",
    "flight dynamics",
    "aerodynamics",
    "air traffic management",
    "aircraft performance",
    "aircraft design",
    "flight control",
    "aviation weather",
    "aircraft icing",
    "wind shear",
    "wake turbulence",
    "human factors aviation",
    "pilot workload",
    "unmanned aircraft",
    "general aviation",
    "instrument landing system",
    "aircraft engine",
    "aircraft structures",
    "flight simulation",
    "NextGen airspace",
    "collision avoidance",
    "aviation noise",
    "composite materials aircraft",
    "fly-by-wire",
    "stability and control",
    "boundary layer",
    "computational fluid dynamics aircraft",
    "aircraft fuel efficiency",
    "aircraft maintenance",
    "fatigue crack aircraft",
]

MAX_PER_QUERY = 200  # Results per search query
MAX_TOTAL = 5000


def search_ntrs(query: str, page: int = 1, page_size: int = 25) -> dict:
    """Search NTRS for documents matching a query."""
    params = {
        "q": query,
        "page": page,
        "pageSize": page_size,
    }

    try:
        resp = requests.get(f"{NTRS_API}/citations/search",
                            params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # Try alternate endpoint format
        try:
            resp = requests.get(f"https://ntrs.nasa.gov/api/citations/search",
                                params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            pass
    return {}


def get_document_detail(doc_id: str) -> dict:
    """Get full details for a specific document."""
    try:
        resp = requests.get(f"{NTRS_API}/citations/{doc_id}",
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        pass
    return {}


def format_report(doc: dict) -> str:
    """Format a NASA technical report as training text."""
    lines = []

    title = doc.get("title", "")
    lines.append(f"# {title}")
    lines.append("")

    # Metadata
    doc_id = doc.get("id", "")
    if doc_id:
        lines.append(f"NASA Technical Report: {doc_id}")

    report_number = doc.get("reportNumber", "")
    if report_number:
        lines.append(f"Report Number: {report_number}")

    pub_date = doc.get("publicationDate", "")
    if pub_date:
        lines.append(f"Date: {pub_date}")

    # Authors
    authors = doc.get("authorAffiliations", [])
    if authors:
        author_names = []
        for a in authors:
            meta = a.get("meta", {})
            name = meta.get("author", {}).get("name", "")
            if name:
                author_names.append(name)
        if author_names:
            lines.append(f"Authors: {', '.join(author_names)}")

    # Center
    center = doc.get("center", {})
    if center:
        center_name = center.get("name", "") if isinstance(center, dict) else str(center)
        if center_name:
            lines.append(f"NASA Center: {center_name}")

    # Subject categories
    subjects = doc.get("subjectCategories", [])
    if subjects:
        lines.append(f"Subjects: {', '.join(subjects)}")

    # Keywords
    keywords = doc.get("keywords", [])
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords[:20])}")

    lines.append("")

    # Abstract
    abstract = doc.get("abstract", "")
    if abstract:
        lines.append("## Abstract")
        lines.append("")
        lines.append(abstract)
        lines.append("")

    # Description (sometimes longer than abstract)
    description = doc.get("description", "")
    if description and description != abstract:
        lines.append("## Description")
        lines.append("")
        lines.append(description)
        lines.append("")

    return "\n".join(lines)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = RAW_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    print("NASA Technical Reports Server (NTRS) Collector")
    print(f"{'=' * 60}")
    print(f"Search queries: {len(SEARCH_QUERIES)}")
    print(f"Max per query: {MAX_PER_QUERY}")
    print(f"Output: {RAW_DIR}")
    print()

    all_docs = {}  # id -> doc dict
    total_chars = 0

    # Phase 1: Search and collect document metadata
    print("Phase 1: Searching NTRS...")

    for qi, query in enumerate(SEARCH_QUERIES):
        if len(all_docs) >= MAX_TOTAL:
            break

        query_count = 0
        for page in range(1, (MAX_PER_QUERY // 25) + 2):
            data = search_ntrs(query, page=page, page_size=25)

            results = data.get("results", [])
            if not results:
                break

            for item in results:
                doc_id = str(item.get("id", ""))
                if doc_id and doc_id not in all_docs:
                    all_docs[doc_id] = item
                    query_count += 1

            if len(results) < 25:
                break

            time.sleep(0.5)

        print(f"  [{qi+1}/{len(SEARCH_QUERIES)}] \"{query}\" +{query_count} "
              f"(total: {len(all_docs)})")
        time.sleep(0.5)

    print(f"\nCollected {len(all_docs)} unique documents")

    # Phase 2: Format and save
    print(f"\nPhase 2: Formatting and saving reports...")
    saved = 0
    skipped = 0

    for doc_id, doc in all_docs.items():
        text = format_report(doc)

        if len(text) < 200:  # Skip very short entries
            skipped += 1
            continue

        # Sanitize filename
        title = doc.get("title", doc_id)
        safe_name = title[:150]
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']:
            safe_name = safe_name.replace(ch, "_")
        safe_name = safe_name.strip().strip(".")

        txt_path = reports_dir / f"{doc_id}_{safe_name}.txt"
        txt_path.write_text(text, encoding="utf-8")
        total_chars += len(text)
        saved += 1

    # Save manifest
    manifest = {
        "source": "NASA Technical Reports Server (NTRS)",
        "url": "https://ntrs.nasa.gov",
        "search_queries": SEARCH_QUERIES,
        "documents_found": len(all_docs),
        "documents_saved": saved,
        "documents_skipped": skipped,
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 4,
    }
    (RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print(f"Reports saved: {saved:,}")
    print(f"Total text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")


if __name__ == "__main__":
    main()
