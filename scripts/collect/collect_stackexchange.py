"""
Download Aviation StackExchange Q&A data.

Aviation StackExchange (aviation.stackexchange.com) has ~30K questions
with expert-written answers covering:
- Regulations, procedures, aerodynamics, weather
- Aircraft systems, navigation, ATC communications
- Pilot training, safety, operations

Uses the public Stack Exchange API (no auth needed, 300 req/day;
with API key: 10,000/day).

Output: data/raw/aviation_stackexchange/
"""

import os
import sys
import json
import time
import gzip
import requests
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "aviation_stackexchange"

SE_API = "https://api.stackexchange.com/2.3"
SITE = "aviation"

# Tags to focus on (high-quality content)
PRIORITY_TAGS = [
    "aerodynamics", "regulations", "faa-regulations", "safety",
    "aircraft-systems", "navigation", "weather", "atc",
    "flight-training", "instrument-flight-rules", "cessna-172",
    "general-aviation", "aircraft-performance", "engine",
    "airspace", "airport", "flight-planning", "procedures",
    "boeing-737", "airbus-a320", "takeoff", "landing",
    "stall", "turbulence", "icing", "fuel", "weight-and-balance",
]


def fetch_questions(page: int = 1, pagesize: int = 100, tagged: str = None,
                    min_score: int = 1) -> dict:
    """Fetch a page of questions with answers from the API."""
    params = {
        "page": page,
        "pagesize": pagesize,
        "order": "desc",
        "sort": "votes",
        "site": SITE,
        "filter": "withbody",  # Include body text
        "min": min_score,
    }
    if tagged:
        params["tagged"] = tagged

    resp = requests.get(f"{SE_API}/questions", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data


def fetch_answers(question_ids: list[int]) -> dict:
    """Fetch answers for a batch of question IDs."""
    ids_str = ";".join(str(qid) for qid in question_ids)
    params = {
        "order": "desc",
        "sort": "votes",
        "site": SITE,
        "filter": "withbody",
        "pagesize": 100,
    }

    resp = requests.get(f"{SE_API}/questions/{ids_str}/answers",
                        params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def html_to_text(html: str) -> str:
    """Simple HTML tag stripping for SE content."""
    import re
    # Remove HTML tags but keep text
    text = re.sub(r'<pre><code>(.*?)</code></pre>', r'\n```\n\1\n```\n', html, flags=re.DOTALL)
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<p>', '\n', text)
    text = re.sub(r'<li>', '\n- ', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def format_qa(question: dict, answers: list[dict]) -> str:
    """Format a Q&A pair as training text."""
    lines = []
    lines.append(f"## Question: {question['title']}")
    if question.get('tags'):
        lines.append(f"Tags: {', '.join(question['tags'])}")
    lines.append(f"Score: {question.get('score', 0)}")
    lines.append("")
    lines.append(html_to_text(question.get('body', '')))
    lines.append("")

    # Sort answers by score
    sorted_answers = sorted(answers, key=lambda a: a.get('score', 0), reverse=True)

    for i, ans in enumerate(sorted_answers[:3]):  # Top 3 answers
        is_accepted = ans.get('is_accepted', False)
        label = "Accepted Answer" if is_accepted else f"Answer {i+1}"
        lines.append(f"### {label} (Score: {ans.get('score', 0)})")
        lines.append("")
        lines.append(html_to_text(ans.get('body', '')))
        lines.append("")

    return "\n".join(lines)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    qa_dir = RAW_DIR / "questions"
    qa_dir.mkdir(exist_ok=True)

    print("Aviation StackExchange Collector")
    print(f"{'=' * 60}")
    print(f"Output: {RAW_DIR}")
    print()

    all_questions = {}  # id -> question dict
    total_chars = 0
    api_calls = 0

    # Phase 1: Fetch top questions across all tags
    print("Phase 1: Fetching top-voted questions...")

    # First, get top questions overall (not filtered by tag)
    for page in range(1, 51):  # Up to 5000 questions (50 pages x 100)
        try:
            data = fetch_questions(page=page, pagesize=100, min_score=0)
            api_calls += 1

            items = data.get("items", [])
            if not items:
                break

            for q in items:
                qid = q["question_id"]
                if qid not in all_questions:
                    all_questions[qid] = q

            quota = data.get("quota_remaining", "?")
            print(f"  Page {page}: +{len(items)} questions "
                  f"(total: {len(all_questions)}, quota: {quota})")

            if not data.get("has_more", False):
                break

            # Rate limiting - SE API has backoff field
            backoff = data.get("backoff", 0)
            time.sleep(max(0.5, backoff))

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            time.sleep(5)
            break

    print(f"\nCollected {len(all_questions)} unique questions")

    # Phase 2: Fetch answers in batches of 30
    print("\nPhase 2: Fetching answers...")
    question_ids = list(all_questions.keys())
    all_answers = {}  # question_id -> [answers]

    for i in range(0, len(question_ids), 30):
        batch = question_ids[i:i+30]
        try:
            data = fetch_answers(batch)
            api_calls += 1

            for ans in data.get("items", []):
                qid = ans["question_id"]
                if qid not in all_answers:
                    all_answers[qid] = []
                all_answers[qid].append(ans)

            quota = data.get("quota_remaining", "?")
            if (i // 30 + 1) % 10 == 0:
                print(f"  Batch {i//30+1}/{len(question_ids)//30+1} "
                      f"(answers for {len(all_answers)} questions, quota: {quota})")

            backoff = data.get("backoff", 0)
            time.sleep(max(0.5, backoff))

        except Exception as e:
            print(f"  Error on batch {i//30+1}: {e}")
            time.sleep(5)

    print(f"  Fetched answers for {len(all_answers)} questions")

    # Phase 3: Format and save
    print("\nPhase 3: Formatting and saving Q&A pairs...")
    saved = 0
    for qid, question in all_questions.items():
        answers = all_answers.get(qid, [])

        # Skip questions with no answers
        if not answers:
            continue

        text = format_qa(question, answers)
        if len(text) < 200:  # Skip very short Q&As
            continue

        # Save as text file
        safe_title = question.get("title", str(qid))
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            safe_title = safe_title.replace(ch, "_")
        safe_title = safe_title[:150]

        txt_path = qa_dir / f"{qid}_{safe_title}.txt"
        txt_path.write_text(text, encoding="utf-8")
        total_chars += len(text)
        saved += 1

    # Save manifest
    manifest = {
        "source": "Aviation StackExchange",
        "url": "https://aviation.stackexchange.com",
        "api_calls": api_calls,
        "questions_fetched": len(all_questions),
        "questions_with_answers": len(all_answers),
        "questions_saved": saved,
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 4,
    }
    (RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print(f"Questions saved: {saved:,}")
    print(f"Total text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")
    print(f"API calls used: {api_calls}")


if __name__ == "__main__":
    main()
