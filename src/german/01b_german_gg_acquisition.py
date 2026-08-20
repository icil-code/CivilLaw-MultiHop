"""
01b_german_gg_acquisition.py
============================
GG (Grundgesetz) special acquisition script.
GG uses a SINGLE-PAGE format (all articles in one HTML file),
unlike BGB/ZPO which use individual sub-pages per paragraph.

Usage:
    python 01b_german_gg_acquisition.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

GG_FULL_URL = "https://www.gesetze-im-internet.de/gg/BJNR000010949.html"
OUTPUT_PATH = Path("data/german/structured/gg_articles.json")
DELAY_SECONDS = 1.0
REQUEST_TIMEOUT = 30

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
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("gg_acquisition.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("gg_acquisition")

# ─── FETCH ─────────────────────────────────────────────────────────────────────

def fetch_url(url: str) -> Optional[str]:
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            time.sleep(DELAY_SECONDS)
            return resp.text
        except requests.exceptions.RequestException as e:
            log.warning("Attempt %d failed: %s", attempt, e)
            time.sleep(2.0 ** attempt)
    return None

# ─── PARSER ────────────────────────────────────────────────────────────────────

def parse_gg_fullpage(html: str) -> List[Dict]:
    """
    Parses the GG full-page HTML (BJNR000010949.html).
    All articles are in <div id="BJNR...BJNE..."> elements.
    Each div contains:
      - class="jnenbez": Article number (e.g., "Art 1")
      - class="jnentitel": Article title
      - class="jurAbsatz": Article text (one or more divs)

    Skips:
      - Footnote divs (IDs ending with "_FNS")
      - Divs with no paragraph number AND no meaningful text

    Args:
        html: Raw HTML of the full GG page.

    Returns:
        List of article record dicts.
    """
    soup = BeautifulSoup(html, "lxml")
    records: List[Dict] = []
    current_abschnitt = ""

    all_elements = soup.find_all(["h2", "h3", "div"])

    for element in all_elements:
        if not hasattr(element, "name"):
            continue

        # Track section headings (Abschnitte)
        if element.name in ("h2", "h3"):
            text = element.get_text(strip=True)
            if text and len(text) < 200:
                current_abschnitt = text
            continue

        # Only process norm divs with BJNE IDs
        el_id = element.get("id", "")
        el_class = element.get("class", [])

        is_norm_div = (
            ("BJNE" in el_id or "jnhtml" in el_class)
            and element.name == "div"
        )
        if not is_norm_div:
            continue

        # Skip footnote divs
        if el_id.endswith("_FNS"):
            continue

        # Extract article number
        jnenbez = element.find(class_="jnenbez")
        para_num = jnenbez.get_text(strip=True) if jnenbez else ""

        # Extract article title
        jnentitel = element.find(class_="jnentitel")
        para_title = jnentitel.get_text(strip=True) if jnentitel else ""

        # Extract text
        absatz_divs = element.find_all(class_="jurAbsatz")
        if absatz_divs:
            full_text = "\n".join(
                d.get_text(separator="\n", strip=True) for d in absatz_divs
            ).strip()
        else:
            working = BeautifulSoup(str(element), "lxml")
            for tag in working.find_all(class_=["jnenbez", "jnentitel", "jnheader"]):
                tag.decompose()
            full_text = working.get_text(separator="\n", strip=True)

        # Skip records with no paragraph number AND no meaningful text
        if not para_num and not full_text:
            continue

        # Build stable ID from element ID or paragraph number
        id_suffix = el_id if el_id else para_num.replace(" ", "_").replace(".", "")
        paragraph_id = f"gg-{id_suffix}"

        records.append({
            "law": "GG",
            "paragraph_id": paragraph_id,
            "paragraph_number": para_num,
            "paragraph_title": para_title,
            "abschnitt": current_abschnitt,
            "text": full_text,
            "url": f"{GG_FULL_URL}#{el_id}",
        })

    return records

# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting GG (Grundgesetz) acquisition...")
    log.info("URL: %s", GG_FULL_URL)

    html = fetch_url(GG_FULL_URL)
    if html is None:
        log.error("Failed to fetch GG page. Aborting.")
        sys.exit(1)

    records = parse_gg_fullpage(html)
    log.info("Parsed %d GG article records.", len(records))

    # Preview first 5
    log.info("\n--- Sample records ---")
    for r in records[:5]:
        log.info(
            "  %-15s | %-35s | text_len=%d",
            r["paragraph_number"],
            r["paragraph_title"][:35],
            len(r["text"]),
        )

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info("Saved to: %s", OUTPUT_PATH.resolve())
