"""
Download SKYbrary aviation safety articles.

SKYbrary (skybrary.aero) is a comprehensive aviation safety knowledge base
maintained by EUROCONTROL. It contains thousands of articles on:
- Aviation safety, human factors, operational procedures
- Aircraft types, airports, navigation aids
- Regulations (ICAO, EASA, FAA)
- Accident/incident case studies
- Weather phenomena, ATC procedures

SKYbrary uses Drupal (not MediaWiki). We crawl via their XML sitemap.

Output: data/raw/skybrary/
"""

import os
import sys
import json
import time
import re
import requests
from pathlib import Path
from xml.etree import ElementTree

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "skybrary"

SKYBRARY_BASE = "https://skybrary.aero"
SITEMAP_INDEX = "https://skybrary.aero/sitemap.xml"
HEADERS = {
    "User-Agent": "FlightMind/0.1 (aviation LLM research; educational/non-commercial)",
}

# Content paths we want (skip airport pages - they're just ICAO codes with little text)
WANTED_PREFIXES = ["/articles/", "/accidents-and-incidents/"]


def get_sitemap_urls() -> list[str]:
    """Parse all sitemap pages to collect article URLs."""
    all_urls = []

    # First get the sitemap index
    print("  Fetching sitemap index...")
    resp = requests.get(SITEMAP_INDEX, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # Parse sitemap index to find sub-sitemaps
    root = ElementTree.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    sub_sitemaps = []
    for sitemap in root.findall("sm:sitemap", ns):
        loc = sitemap.find("sm:loc", ns)
        if loc is not None:
            sub_sitemaps.append(loc.text)

    print(f"  Found {len(sub_sitemaps)} sub-sitemaps")

    # Parse each sub-sitemap
    for i, sitemap_url in enumerate(sub_sitemaps):
        try:
            resp = requests.get(sitemap_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)

            page_urls = []
            for url_elem in root.findall("sm:url", ns):
                loc = url_elem.find("sm:loc", ns)
                if loc is not None:
                    url = loc.text
                    # Filter to wanted content types
                    path = url.replace(SKYBRARY_BASE, "")
                    if any(path.startswith(p) for p in WANTED_PREFIXES):
                        page_urls.append(url)

            all_urls.extend(page_urls)
            print(f"  Sitemap {i+1}/{len(sub_sitemaps)}: +{len(page_urls)} articles "
                  f"(total: {len(all_urls)})")

            time.sleep(0.3)
        except Exception as e:
            print(f"  Error on sitemap {i+1}: {e}")

    return all_urls


def extract_text_from_html(html: str) -> str:
    """Extract main article text from SKYbrary HTML page."""
    # Find the main content area
    # SKYbrary uses Drupal, content is usually in article or field--name-body

    # Try to extract the article body
    body_match = re.search(
        r'<div[^>]*class="[^"]*field--name-body[^"]*"[^>]*>(.*?)</div>\s*</div>',
        html, re.DOTALL
    )
    if not body_match:
        # Fallback: look for article content
        body_match = re.search(
            r'<article[^>]*>(.*?)</article>',
            html, re.DOTALL
        )
    if not body_match:
        # Another fallback: look for main content
        body_match = re.search(
            r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>\s*(?:</div>|\s*<div[^>]*class="[^"]*field)',
            html, re.DOTALL
        )

    if not body_match:
        # Last resort: grab everything between <main> tags
        body_match = re.search(r'<main[^>]*>(.*?)</main>', html, re.DOTALL)

    if not body_match:
        return ""

    text = body_match.group(1)

    # Clean HTML to plain text
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Convert headers
    text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n## \1\n', text, flags=re.DOTALL)
    # Convert paragraphs
    text = re.sub(r'<p[^>]*>', '\n', text)
    text = re.sub(r'</p>', '\n', text)
    # Convert list items
    text = re.sub(r'<li[^>]*>', '\n- ', text)
    # Convert line breaks
    text = re.sub(r'<br\s*/?>', '\n', text)
    # Remove all remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Clean whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def get_article(url: str) -> tuple[str, str]:
    """Fetch a SKYbrary article and extract its title and text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        html = resp.text

        # Extract title
        title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
        title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else ""

        # Extract text
        text = extract_text_from_html(html)

        return title, text
    except Exception:
        return "", ""


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    articles_dir = RAW_DIR / "articles"
    articles_dir.mkdir(exist_ok=True)

    print("SKYbrary Aviation Safety Article Collector")
    print(f"{'=' * 60}")
    print(f"Output: {RAW_DIR}")
    print()

    # Phase 1: Get all article URLs from sitemap
    print("Phase 1: Discovering articles via sitemap...")
    urls = get_sitemap_urls()
    print(f"\n  Found {len(urls)} articles + accident reports")

    # Save URL list
    (RAW_DIR / "article_urls.json").write_text(
        json.dumps(urls, indent=2), encoding="utf-8"
    )

    # Phase 2: Download and extract content
    print(f"\nPhase 2: Downloading article content...")
    total_chars = 0
    downloaded = 0
    skipped = 0

    for i, url in enumerate(urls):
        # Create filename from URL slug
        slug = url.replace(SKYBRARY_BASE, "").strip("/").replace("/", "_")
        safe_name = slug
        for ch in [':', '*', '?', '"', '<', '>', '|']:
            safe_name = safe_name.replace(ch, "_")
        if len(safe_name) > 200:
            safe_name = safe_name[:200]

        txt_path = articles_dir / f"{safe_name}.txt"

        if txt_path.exists() and txt_path.stat().st_size > 0:
            total_chars += txt_path.stat().st_size
            downloaded += 1
            continue

        title, text = get_article(url)

        if text and len(text) > 100:
            header = f"# {title}\n\n" if title else ""
            txt_path.write_text(f"{header}{text}", encoding="utf-8")
            total_chars += len(text)
            downloaded += 1
        else:
            skipped += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(urls)}] Downloaded: {downloaded}, "
                  f"Skipped: {skipped}, Chars: {total_chars:,}")

        time.sleep(0.5)  # Be polite

    # Save manifest
    manifest = {
        "source": "SKYbrary Aviation Safety Knowledge Base",
        "url": "https://skybrary.aero",
        "maintainer": "EUROCONTROL",
        "urls_discovered": len(urls),
        "articles_downloaded": downloaded,
        "articles_skipped": skipped,
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 4,
    }
    (RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print(f"Articles: {downloaded:,} downloaded, {skipped} skipped")
    print(f"Total text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")


if __name__ == "__main__":
    main()
