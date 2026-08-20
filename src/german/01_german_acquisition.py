"""
01_german_acquisition.py
========================
Task 1 of 7: German Federal Law Acquisition Pipeline
Acquires BGB, ZPO, and GG from gesetze-im-internet.de.

NOTE: The site uses a ToC page + individual paragraph sub-pages.
This script crawls the ToC to collect paragraph URLs, then fetches each
paragraph page, parses the content, and saves structured JSON records.

Usage:
    python 01_german_acquisition.py            # Full acquisition (all 3 laws)
    python 01_german_acquisition.py --test     # Test mode (BGB first 5 paragraphs)
    python 01_german_acquisition.py --laws BGB # Only acquire BGB
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

BASE_URL = "https://www.gesetze-im-internet.de"

LAWS: Dict[str, Dict] = {
    "BGB": {
        "toc_url": "https://www.gesetze-im-internet.de/bgb/",
        "label": "Buergerliches Gesetzbuch",
        "json_file": "bgb_paragraphs.json",
    },
    "ZPO": {
        "toc_url": "https://www.gesetze-im-internet.de/zpo/",
        "label": "Zivilprozessordnung",
        "json_file": "zpo_paragraphs.json",
    },
    "GG": {
        "toc_url": "https://www.gesetze-im-internet.de/gg/",
        "label": "Grundgesetz",
        "json_file": "gg_articles.json",
    },
}

OUTPUT_DIR = Path("data/german")
STRUCTURED_DIR = OUTPUT_DIR / "structured"

DELAY_SECONDS: float = 0.8
MAX_RETRIES: int = 3
BACKOFF_FACTOR: float = 2.0
REQUEST_TIMEOUT: int = 20
TEST_LIMIT: int = 5  # number of paragraphs to fetch in test mode

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LegalNLP-Research/1.0; "
        "+mailto:research@example.org)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "de,en;q=0.9",
}

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("german_acquisition.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("german_acquisition")

# ─── HTTP HELPERS ──────────────────────────────────────────────────────────────

def fetch_url(url: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """
    Fetches HTML content with retry and exponential backoff.

    Args:
        url: Target URL.
        retries: Maximum number of attempts.

    Returns:
        HTML string or None on permanent failure.
    """
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            time.sleep(DELAY_SECONDS)
            return response.text

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status == 429:
                wait = BACKOFF_FACTOR ** attempt * 5
                log.warning("Rate-limited (429). Waiting %.1fs.", wait)
                time.sleep(wait)
            elif status == 404:
                log.debug("Not found (404): %s", url)
                return None
            else:
                log.warning("HTTP %s on attempt %d for %s", status, attempt, url)
                time.sleep(BACKOFF_FACTOR ** attempt)

        except requests.exceptions.ConnectionError as e:
            log.warning("Connection error (attempt %d): %s", attempt, e)
            time.sleep(BACKOFF_FACTOR ** attempt)

        except requests.exceptions.Timeout:
            log.warning("Timeout on attempt %d: %s", attempt, url)
            time.sleep(BACKOFF_FACTOR ** attempt)

        except requests.exceptions.RequestException as e:
            log.error("Unrecoverable error: %s", e)
            return None

    log.error("All %d attempts failed: %s", retries, url)
    return None


def save_json(records: List[Dict], filepath: Path) -> None:
    """Saves a list of dicts to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info("Saved %d records to %s", len(records), filepath)

# ─── TOC CRAWLING ─────────────────────────────────────────────────────────────

def collect_paragraph_urls(toc_url: str, law_code: str) -> List[str]:
    """
    Parses the Table of Contents page and collects all paragraph sub-page URLs.
    gesetze-im-internet.de serves individual paragraphs as separate HTML pages
    linked from the ToC (e.g., __1.html, __2.html, ...).

    Args:
        toc_url: URL of the law's table of contents.
        law_code: Short law code for logging.

    Returns:
        List of absolute URLs to individual paragraph pages.
    """
    log.info("[%s] Fetching ToC: %s", law_code, toc_url)
    html = fetch_url(toc_url)
    if html is None:
        log.error("[%s] Could not fetch ToC. Aborting.", law_code)
        return []

    soup = BeautifulSoup(html, "lxml")
    base = toc_url.rstrip("/")
    base_dir = "/".join(base.split("/")[:-1]) if not base.endswith("/") else base

    urls = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        # Paragraph pages match patterns like: __1.html  or  ___3_bis_6.html
        if href.endswith(".html") and not href.startswith("http") and "__" in href:
            if href.startswith("__"):
                full_url = toc_url.rstrip("/") + "/" + href
            else:
                full_url = BASE_URL + "/" + law_code.lower() + "/" + href
            if full_url not in urls:
                urls.append(full_url)

    log.info("[%s] Collected %d paragraph URLs from ToC.", law_code, len(urls))
    return urls

# ─── PARAGRAPH PAGE PARSER ─────────────────────────────────────────────────────

def parse_paragraph_page(
    html: str, url: str, law_code: str
) -> Optional[Dict]:
    """
    Parses a single paragraph sub-page from gesetze-im-internet.de.
    The page structure is:
        <div class="jnhtml">
            <div class="jnheader">
                <h3><span class="jnenbez">§ 1</span>
                    <span class="jnentitel">Beginn der Rechtsfaehigkeit</span>
                </h3>
            </div>
            <div class="jurAbsatz">full text...</div>
        </div>

    Args:
        html: Raw HTML of the paragraph page.
        url: URL of the page (stored for reference).
        law_code: Short law code.

    Returns:
        A structured dict or None if parsing fails.
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract paragraph number (§ X or Art. X)
    para_num = ""
    jnenbez = soup.find(class_="jnenbez")
    if jnenbez:
        para_num = jnenbez.get_text(strip=True)

    # Extract paragraph title
    para_title = ""
    jnentitel = soup.find(class_="jnentitel")
    if jnentitel:
        para_title = jnentitel.get_text(strip=True)

    # Extract full text (all jurAbsatz divs)
    absatz_divs = soup.find_all(class_="jurAbsatz")
    if not absatz_divs:
        # Fallback: get all text inside jnhtml
        jnhtml = soup.find(class_="jnhtml")
        full_text = jnhtml.get_text(separator="\n", strip=True) if jnhtml else ""
    else:
        full_text = "\n".join(
            div.get_text(separator="\n", strip=True) for div in absatz_divs
        ).strip()

    if not full_text and not para_num:
        return None

    # Build stable paragraph_id from URL filename
    filename = url.rstrip("/").split("/")[-1].replace(".html", "")
    para_id = f"{law_code.lower()}-{filename}"

    return {
        "law": law_code,
        "paragraph_id": para_id,
        "paragraph_number": para_num,
        "paragraph_title": para_title,
        "text": full_text,
        "url": url,
    }

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def acquire_law(
    law_code: str,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Full acquisition pipeline for a single law:
    1. Fetch ToC to get all paragraph URLs.
    2. Fetch and parse each paragraph page.
    3. Return list of structured records.

    Args:
        law_code: Law code (BGB, ZPO, or GG).
        limit: Maximum number of paragraphs to fetch (for testing).

    Returns:
        List of paragraph record dicts.
    """
    cfg = LAWS[law_code]
    paragraph_urls = collect_paragraph_urls(cfg["toc_url"], law_code)

    if not paragraph_urls:
        return []

    if limit:
        paragraph_urls = paragraph_urls[:limit]
        log.info("[%s] TEST MODE: limiting to %d paragraphs.", law_code, limit)

    records: List[Dict] = []

    for i, url in enumerate(paragraph_urls, 1):
        html = fetch_url(url)
        if html is None:
            log.warning("[%s] Skipping (fetch failed): %s", law_code, url)
            continue

        record = parse_paragraph_page(html, url, law_code)
        if record:
            records.append(record)
        else:
            log.debug("[%s] Skipping (parse returned None): %s", law_code, url)

        if i % 100 == 0:
            log.info("[%s] Progress: %d / %d paragraphs fetched.", law_code, i, len(paragraph_urls))

    log.info("[%s] Acquisition complete: %d records.", law_code, len(records))
    return records


def run_acquisition(laws_to_process: Optional[List[str]] = None) -> None:
    """Full pipeline: acquire all specified laws and save JSON."""
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    target_laws = laws_to_process or list(LAWS.keys())
    grand_total = 0

    for law_code in target_laws:
        log.info("=" * 60)
        log.info("Starting: %s — %s", law_code, LAWS[law_code]["label"])
        log.info("=" * 60)

        records = acquire_law(law_code)
        grand_total += len(records)

        json_path = STRUCTURED_DIR / LAWS[law_code]["json_file"]
        save_json(records, json_path)

    log.info("=" * 60)
    log.info("All laws complete. Grand total: %d records.", grand_total)
    log.info("Output directory: %s", STRUCTURED_DIR.resolve())

# ─── TEST FUNCTION ─────────────────────────────────────────────────────────────

def test_parsing() -> None:
    """Downloads the first TEST_LIMIT paragraphs of BGB and verifies structure."""
    log.info("Running test mode: BGB first %d paragraphs.", TEST_LIMIT)

    records = acquire_law("BGB", limit=TEST_LIMIT)

    if not records:
        log.error("Test FAILED: no records returned.")
        return

    required_fields = ["law", "paragraph_id", "paragraph_number", "text", "url"]
    sample = records[0]
    missing = [f for f in required_fields if f not in sample or not sample[f]]
    if missing:
        log.error("Test FAILED: missing/empty fields: %s", missing)
    else:
        log.info("Test PASSED. All required fields present.")

    log.info("\n--- Sample records ---")
    for r in records:
        log.info(
            "  %-15s | %-30s | text_len=%d",
            r["paragraph_number"],
            r["paragraph_title"][:30],
            len(r["text"]),
        )

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="German Law Acquisition Pipeline — BGB, ZPO, GG"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: fetch only first 5 BGB paragraphs.",
    )
    parser.add_argument(
        "--laws",
        nargs="+",
        choices=["BGB", "ZPO", "GG"],
        default=None,
        help="Laws to download (default: all three).",
    )
    args = parser.parse_args()

    if args.test:
        test_parsing()
    else:
        run_acquisition(laws_to_process=args.laws)
