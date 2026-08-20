"""
02_german_lsu_transformation.py
================================
Task 2 of 7: German Legal Semantic Unit (LSU) Transformation Pipeline

Transforms acquired German law paragraphs (BGB, ZPO, GG) into LSUs
following the identical schema as the Turkish instantiation, enabling
direct cross-jurisdictional comparison in an AI & Law journal paper.

Usage:
    python 02_german_lsu_transformation.py
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

INPUT_DIR = Path("data/german/structured")
OUTPUT_DIR = Path("data/german/lsus")
OUTPUT_FILE = OUTPUT_DIR / "german_lsus.json"

YEAR = "2024"
LANGUAGE = "de"
JURISDICTION = "DE"
DEFAULT_VALID_FROM = f"{YEAR}-01-01"
REPEALED_VALID_TO = f"{int(YEAR) - 1}-12-31"

INPUT_FILES: Dict[str, Path] = {
    "BGB": INPUT_DIR / "bgb_paragraphs.json",
    "ZPO": INPUT_DIR / "zpo_paragraphs.json",
    "GG":  INPUT_DIR / "gg_articles.json",
}

REQUIRED_LSU_FIELDS = [
    "lsu_id",
    "law_code",
    "article_number",
    "hierarchy_level",
    "provision_text",
    "valid_from",
    "amendment_status",
    "language",
    "jurisdiction",
]

PROGRESS_INTERVAL = 500

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("german_lsu_transformation.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("german_lsu")

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def load_paragraphs(law_code: str, filepath: Path) -> List[Dict[str, Any]]:
    """
    Loads a JSON file of acquired paragraphs for a given law.

    Args:
        law_code: Short law code (BGB, ZPO, GG) for logging.
        filepath: Path to the JSON file.

    Returns:
        List of paragraph dicts. Empty list if file is missing or invalid.
    """
    if not filepath.exists():
        log.error("[%s] Input file not found: %s", law_code, filepath)
        return []

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        log.info("[%s] Loaded %d paragraphs from %s", law_code, len(data), filepath)
        return data
    except json.JSONDecodeError as e:
        log.error("[%s] JSON parse error in %s: %s", law_code, filepath, e)
        return []


def normalise_paragraph_number(raw: str) -> str:
    """
    Normalises a raw paragraph number string for use in LSU IDs.
    Removes special characters and whitespace, keeping digits and letters.

    Args:
        raw: Raw paragraph number (e.g., "§ 100", "Art 1", "§ 3 bis 6").

    Returns:
        Normalised string (e.g., "100", "1", "3_bis_6").

    Examples:
        >>> normalise_paragraph_number("§ 100")
        '100'
        >>> normalise_paragraph_number("Art 1")
        '1'
        >>> normalise_paragraph_number("§ 3 bis 6")
        '3_bis_6'
    """
    # Remove § sign, Art., Artikel, leading/trailing whitespace
    cleaned = re.sub(r"[§Aa]rt\.?|Artikel", "", raw, flags=re.IGNORECASE).strip()
    # Replace spaces with underscores (for ranges like "3 bis 6")
    cleaned = re.sub(r"\s+", "_", cleaned)
    # Remove any remaining non-alphanumeric except underscores and hyphens
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", cleaned)
    return cleaned if cleaned else "unknown"


def build_lsu_id(law_code: str, paragraph_number: str, raw_id: str) -> str:
    """
    Constructs an Akoma Ntoso-compatible IRI for a given paragraph.

    Format:
        Legislation (BGB, ZPO): /akn/de/act/{law_code}/{year}/paragraph/{number}
        Constitution (GG):      /akn/de/act/gg/{year}/article/{number}

    For GG, the BJNE element ID (e.g., BJNR000010949BJNE001700314) stored
    in paragraph_id is used as a stable unique suffix when paragraph_number
    is empty.

    Args:
        law_code: Short law code (BGB, ZPO, GG).
        paragraph_number: Human-readable paragraph/article number string.
        raw_id: The raw paragraph_id field from the source data (fallback).

    Returns:
        Akoma Ntoso-compatible IRI string.
    """
    law_lower = law_code.lower()
    norm_num = normalise_paragraph_number(paragraph_number)

    # For GG: prefer the BJNE element ID which is stable and unique
    if law_code == "GG":
        # raw_id format: "gg-BJNR000010949BJNE001700314"
        bjne_part = raw_id.replace("gg-", "").strip()
        if bjne_part and bjne_part != "unknown":
            return f"/akn/de/act/gg/{YEAR}/article/{bjne_part}"
        # fallback to normalised paragraph number (e.g. "Eingangsformel")
        if norm_num and norm_num != "unknown":
            return f"/akn/de/act/gg/{YEAR}/article/{norm_num}"
        return f"/akn/de/act/gg/{YEAR}/article/unknown"

    # For BGB/ZPO: use the normalised paragraph number
    if not norm_num or norm_num == "unknown":
        fallback = re.sub(r"[^a-zA-Z0-9_\-]", "", raw_id.split("-", 1)[-1])
        norm_num = fallback if fallback else "unknown"

    return f"/akn/de/act/{law_lower}/{YEAR}/paragraph/{norm_num}"



def is_repealed(para: Dict[str, Any]) -> bool:
    """
    Determines whether a paragraph has been repealed (weggefallen).
    A paragraph is considered repealed if its text is empty or contains
    the German legal term "weggefallen" (lapsed/repealed).

    Args:
        para: Paragraph dict from acquired data.

    Returns:
        True if the paragraph is repealed, False otherwise.
    """
    text = para.get("text", "").strip().lower()
    title = para.get("paragraph_title", "").strip().lower()
    return (
        not text
        or "weggefallen" in text
        or "weggefallen" in title
        or text in {"(weggefallen)", "[weggefallen]"}
    )


def build_hierarchy_level(
    para: Dict[str, Any], law_code: str
) -> Dict[str, Optional[str]]:
    """
    Constructs the hierarchy_level dict from raw paragraph metadata.

    For BGB/ZPO, maps buch/abschnitt/titel to book/chapter/title.
    For GG, all structural fields are set to null (GG has no Bücher/Abschnitte).

    Args:
        para: Paragraph dict from acquired data.
        law_code: Short law code (BGB, ZPO, GG).

    Returns:
        Dictionary with keys: book, chapter, title, paragraph.
    """
    para_num = para.get("paragraph_number", "").strip() or None

    if law_code == "GG":
        return {
            "book": None,
            "chapter": para.get("abschnitt", "").strip() or None,
            "title": None,
            "paragraph": para_num,
        }

    return {
        "book":      para.get("buch", "").strip() or None,
        "chapter":   para.get("abschnitt", "").strip() or None,
        "title":     para.get("titel", "").strip() or None,
        "paragraph": para_num,
    }


def create_lsu(para: Dict[str, Any], law_code: str) -> Dict[str, Any]:
    """
    Transforms a single raw paragraph dict into a Legal Semantic Unit (LSU).
    The output schema is identical to the Turkish LSU instantiation, enabling
    direct cross-jurisdictional comparison.

    Args:
        para: Raw paragraph dict from the acquired data.
        law_code: Short law code (BGB, ZPO, or GG).

    Returns:
        LSU dict following the canonical schema.
    """
    paragraph_number = para.get("paragraph_number", "").strip()
    paragraph_title  = para.get("paragraph_title", "").strip()
    raw_id           = para.get("paragraph_id", "")
    text             = para.get("text", "").strip()
    source_url       = para.get("url", "").strip()

    repealed = is_repealed(para)

    lsu_id = build_lsu_id(law_code, paragraph_number, raw_id)
    hierarchy = build_hierarchy_level(para, law_code)

    # Article number: prefer paragraph_title if it gives the title, else para number
    article_number = paragraph_number if paragraph_number else raw_id

    return {
        "lsu_id":           lsu_id,
        "law_code":         law_code,
        "article_number":   article_number,
        "paragraph_title":  paragraph_title or None,
        "hierarchy_level":  hierarchy,
        "provision_text":   text,
        "decision_number":  None,
        "court_name":       None,
        "decision_date":    None,
        "valid_from":       DEFAULT_VALID_FROM,
        "valid_to":         REPEALED_VALID_TO if repealed else None,
        "amendment_status": "repealed" if repealed else "in_force",
        "source_url":       source_url or None,
        "language":         LANGUAGE,
        "jurisdiction":     JURISDICTION,
    }


# ─── VALIDATION ────────────────────────────────────────────────────────────────

def validate_lsus(lsus: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validates the full LSU list for schema compliance.

    Checks:
        1. All required fields are present and non-null where required.
        2. All LSU IDs are unique.
        3. Law codes are in the expected set {BGB, ZPO, GG}.
        4. All language values are "de".
        5. All jurisdiction values are "DE".

    Args:
        lsus: List of LSU dicts to validate.

    Returns:
        Tuple of (passed: bool, errors: List[str]).
    """
    errors: List[str] = []
    seen_ids: Set[str] = set()
    allowed_law_codes = {"BGB", "ZPO", "GG"}

    for i, lsu in enumerate(lsus):
        lsu_id = lsu.get("lsu_id", f"<index {i}>")

        # Check required fields
        for field in REQUIRED_LSU_FIELDS:
            if field not in lsu:
                errors.append(f"{lsu_id}: missing field '{field}'")
            elif lsu[field] is None and field in ("lsu_id", "law_code", "language", "jurisdiction"):
                errors.append(f"{lsu_id}: required field '{field}' is null")

        # Check ID uniqueness
        if lsu_id in seen_ids:
            errors.append(f"Duplicate lsu_id: {lsu_id}")
        else:
            seen_ids.add(lsu_id)

        # Check law code
        law_code = lsu.get("law_code", "")
        if law_code not in allowed_law_codes:
            errors.append(f"{lsu_id}: unexpected law_code '{law_code}'")

        # Check language
        if lsu.get("language") != "de":
            errors.append(f"{lsu_id}: expected language='de', got '{lsu.get('language')}'")

        # Check jurisdiction
        if lsu.get("jurisdiction") != "DE":
            errors.append(f"{lsu_id}: expected jurisdiction='DE', got '{lsu.get('jurisdiction')}'")

    passed = len(errors) == 0
    return passed, errors


def print_summary(lsus: List[Dict[str, Any]], passed: bool, errors: List[str]) -> None:
    """
    Prints a structured summary of the LSU creation run.

    Args:
        lsus: The full list of created LSUs.
        passed: Whether validation passed.
        errors: List of validation error messages.
    """
    by_law: Dict[str, int] = {}
    by_status: Dict[str, int] = {}

    for lsu in lsus:
        code = lsu.get("law_code", "UNKNOWN")
        status = lsu.get("amendment_status", "unknown")
        by_law[code] = by_law.get(code, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1

    log.info("")
    log.info("=" * 55)
    log.info("  German LSU Creation Summary")
    log.info("=" * 55)
    log.info("  Total LSUs    : %d", len(lsus))
    for code in ("BGB", "ZPO", "GG"):
        log.info("  %-12s  : %d", code, by_law.get(code, 0))
    log.info("  In force      : %d", by_status.get("in_force", 0))
    log.info("  Repealed      : %d", by_status.get("repealed", 0))
    log.info("-" * 55)

    if passed:
        log.info("  Validation    : PASSED (all fields present, all IDs unique)")
    else:
        log.warning("  Validation    : FAILED (%d errors)", len(errors))
        for err in errors[:20]:   # print first 20 errors max
            log.warning("    - %s", err)
        if len(errors) > 20:
            log.warning("    ... and %d more errors.", len(errors) - 20)

    log.info("=" * 55)
    log.info("  Output        : %s", OUTPUT_FILE.resolve())
    log.info("=" * 55)


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_transformation() -> None:
    """
    Main LSU transformation pipeline.

    Steps:
        1. Load raw paragraphs for BGB, ZPO, and GG.
        2. Transform each paragraph into a canonical LSU.
        3. Resolve duplicate LSU IDs (append counter suffix).
        4. Validate the full LSU list.
        5. Save consolidated JSON output.
        6. Print summary.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_lsus: List[Dict[str, Any]] = []
    seen_ids: Dict[str, int] = {}  # id -> occurrence count (for deduplication)

    for law_code, filepath in INPUT_FILES.items():
        log.info("=" * 55)
        log.info("Processing: %s", law_code)
        log.info("=" * 55)

        paragraphs = load_paragraphs(law_code, filepath)
        if not paragraphs:
            log.warning("[%s] No paragraphs loaded. Skipping.", law_code)
            continue

        law_lsus: List[Dict[str, Any]] = []

        for i, para in enumerate(paragraphs):
            try:
                lsu = create_lsu(para, law_code)
            except Exception as e:
                log.warning(
                    "[%s] Error transforming paragraph at index %d: %s",
                    law_code, i, e
                )
                continue

            # Resolve duplicate IDs deterministically
            base_id = lsu["lsu_id"]
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                lsu["lsu_id"] = f"{base_id}-dup{seen_ids[base_id]}"
                log.warning(
                    "[%s] Duplicate lsu_id resolved: %s -> %s",
                    law_code, base_id, lsu["lsu_id"]
                )
            else:
                seen_ids[base_id] = 1

            law_lsus.append(lsu)

            total_so_far = len(all_lsus) + len(law_lsus)
            if total_so_far % PROGRESS_INTERVAL == 0:
                log.info("Progress: %d LSUs created so far...", total_so_far)

        log.info(
            "[%s] Transformed %d / %d paragraphs into LSUs.",
            law_code, len(law_lsus), len(paragraphs)
        )
        all_lsus.extend(law_lsus)

    log.info("Total LSUs before validation: %d", len(all_lsus))

    # Validate
    passed, errors = validate_lsus(all_lsus)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_lsus, f, ensure_ascii=False, indent=2)

    # Summary
    print_summary(all_lsus, passed, errors)


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_transformation()
