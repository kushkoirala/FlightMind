"""
Download aviation-related Wikipedia articles.

Uses the Wikipedia API to fetch articles in aviation categories:
- Aircraft by type (fixed-wing, rotorcraft, etc.)
- Airlines
- Airports
- Aviation accidents
- Aerodynamics concepts
- Aviation terminology
- Avionics
- Flight instruments
- Navigation systems
- Air traffic control
- Aviation regulations
- Aerospace engineering

Output: data/raw/wikipedia_aviation/
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# Fix Windows console encoding for non-ASCII characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wikipedia_aviation"

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "FlightMind/0.1 (https://github.com/kushkoirala/FlightMind; aviation LLM research project)",
}

# Categories to crawl (including subcategories)
SEED_CATEGORIES = [
    "Aviation",
    "Aircraft",
    "Aerodynamics",
    "Avionics",
    "Air_traffic_control",
    "Aviation_accidents_and_incidents",
    "Airports",
    "Flight_instruments",
    "Aircraft_engines",
    "Aerospace_engineering",
    "Aviation_safety",
    "Aviation_meteorology",
    "Instrument_flight_rules",
    "Visual_flight_rules",
    "Cessna_aircraft",
    "Boeing_aircraft",
    "Airbus_aircraft",
    "Military_aircraft",
    "General_aviation",
    "Flight_training",
]

# Maximum depth for subcategory crawl
MAX_DEPTH = 2
# Maximum articles per category
MAX_ARTICLES_PER_CAT = 200
# Maximum total articles
MAX_TOTAL_ARTICLES = 10000


def get_category_members(category: str, cmtype: str = "page",
                         limit: int = 500) -> list[dict]:
    """Get members of a Wikipedia category."""
    members = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": cmtype,
        "cmlimit": min(limit, 500),
        "format": "json",
    }

    while True:
        try:
            resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            members.extend(data.get("query", {}).get("categorymembers", []))

            if "continue" in data and len(members) < limit:
                params["cmcontinue"] = data["continue"]["cmcontinue"]
                time.sleep(0.1)
            else:
                break
        except Exception as e:
            print(f"    Error fetching category {category}: {e}")
            break

    return members[:limit]


def get_article_text(title: str) -> str:
    """Get plain text extract of a Wikipedia article."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
    }

    try:
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id != "-1":
                return page.get("extract", "")
    except Exception:
        pass
    return ""


def crawl_categories(seeds: list[str], max_depth: int = 2) -> set[str]:
    """Recursively crawl categories to find all article titles."""
    all_titles = set()
    visited_cats = set()
    queue = [(cat, 0) for cat in seeds]

    while queue and len(all_titles) < MAX_TOTAL_ARTICLES:
        category, depth = queue.pop(0)
        if category in visited_cats:
            continue
        visited_cats.add(category)

        # Get articles in this category
        articles = get_category_members(category, "page", MAX_ARTICLES_PER_CAT)
        new_titles = {a["title"] for a in articles} - all_titles
        all_titles.update(new_titles)

        if new_titles:
            print(f"  [{len(all_titles):5d}] Category:{category} +{len(new_titles)} articles")

        # Get subcategories if not too deep
        if depth < max_depth:
            subcats = get_category_members(category, "subcat", 50)
            for sc in subcats:
                cat_name = sc["title"].replace("Category:", "")
                if cat_name not in visited_cats:
                    queue.append((cat_name, depth + 1))

        time.sleep(0.1)  # Rate limiting

    return all_titles


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    articles_dir = RAW_DIR / "articles"
    articles_dir.mkdir(exist_ok=True)

    print("Wikipedia Aviation Article Collector")
    print(f"{'=' * 60}")
    print(f"Seed categories: {len(SEED_CATEGORIES)}")
    print(f"Max depth: {MAX_DEPTH}")
    print(f"Max articles: {MAX_TOTAL_ARTICLES:,}")
    print(f"Output: {RAW_DIR}")
    print()

    # Phase 1: Crawl categories to find articles
    print("Phase 1: Discovering articles via category crawl...")
    titles = crawl_categories(SEED_CATEGORIES, MAX_DEPTH)
    print(f"\nDiscovered {len(titles):,} unique articles")

    # Save title list
    titles_path = RAW_DIR / "article_titles.json"
    titles_path.write_text(json.dumps(sorted(titles), indent=2), encoding="utf-8")

    # Phase 2: Download article text
    print(f"\nPhase 2: Downloading article text...")
    total_chars = 0
    downloaded = 0
    failed = 0

    for i, title in enumerate(sorted(titles)):
        # Sanitize filename (remove all chars invalid on Windows)
        safe_name = title
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            safe_name = safe_name.replace(ch, "_")
        # Truncate long filenames (Windows max path component is 255)
        if len(safe_name) > 200:
            safe_name = safe_name[:200]
        txt_path = articles_dir / f"{safe_name}.txt"

        if txt_path.exists() and txt_path.stat().st_size > 0:
            total_chars += txt_path.stat().st_size
            downloaded += 1
            continue

        text = get_article_text(title)
        if text and len(text) > 100:  # Skip stubs
            txt_path.write_text(f"# {title}\n\n{text}", encoding="utf-8")
            total_chars += len(text)
            downloaded += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(titles)}] Downloaded: {downloaded}, "
                  f"Total: {total_chars:,} chars")

        time.sleep(0.2)  # Rate limiting

    # Save manifest
    manifest = {
        "source": "Wikipedia Aviation Articles",
        "seed_categories": SEED_CATEGORIES,
        "max_depth": MAX_DEPTH,
        "articles_discovered": len(titles),
        "articles_downloaded": downloaded,
        "articles_failed": failed,
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 4,
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Articles: {downloaded:,} downloaded, {failed} failed")
    print(f"Total text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")


if __name__ == "__main__":
    main()
