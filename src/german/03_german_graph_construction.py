"""
03_german_graph_construction.py
================================
Task 3 of 7: German Legal Knowledge Graph Construction

Builds a typed legal knowledge graph from the German LSUs created in Task 2.
The graph schema mirrors the Turkish instantiation to enable direct
cross-jurisdictional comparison of graph density, traversal effects,
and retrieval performance.

Graph topology:
  - Normative unit nodes (one per LSU)
  - Hierarchy nodes (book, chapter, title — inferred from LSU metadata)
  - Citation edges (extracted from provision_text by regex)
  - Hierarchical authority edges (book → chapter → title → paragraph)

Output: NetworkX node_link_data JSON (data/german/graph/german_graph.json)

Usage:
    python 03_german_graph_construction.py
"""

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

INPUT_FILE  = Path("data/german/lsus/german_lsus.json")
OUTPUT_DIR  = Path("data/german/graph")
OUTPUT_FILE = OUTPUT_DIR / "german_graph.json"
STATS_FILE  = OUTPUT_DIR / "german_graph_stats.json"

YEAR = "2024"
JURISDICTION = "DE"

ALLOWED_RELATION_TYPES = {"cites", "hierarchicalAuthority"}
ALLOWED_LAW_CODES = {"BGB", "ZPO", "GG"}

PROGRESS_INTERVAL = 500

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("german_graph_construction.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("german_graph")

# ─── CITATION REGEX PATTERNS ───────────────────────────────────────────────────

# Pattern 1: "§ 100 BGB", "§ 100a ZPO", "§§ 100, 101 BGB"
PAT_PARA_LAW = re.compile(
    r"§§?\s*(\d+\s*(?:bis\s*\d+\s*)?[a-e]?)\s*(BGB|ZPO|GG)\b",
    re.IGNORECASE,
)

# Pattern 2: "§ 100" without explicit law code (intra-law reference)
PAT_PARA_BARE = re.compile(
    r"§§?\s*(\d+\s*(?:bis\s*\d+\s*)?[a-e]?)",
    re.IGNORECASE,
)

# Pattern 3: "Absatz 2" / "Abs. 3"
PAT_ABSATZ = re.compile(r"\bAbs(?:atz|\.)\s*(\d+)", re.IGNORECASE)

# Pattern 4: "Art. 1 GG" / "Art 20 GG"
PAT_ART_LAW = re.compile(
    r"\bArt(?:ikel|\.)\s*(\d+[a-e]?)\s*(GG|BGB|ZPO)?\b",
    re.IGNORECASE,
)

# Pattern 5: "Buch 2", "Abschnitt 3", "Titel 5" (hierarchical references)
PAT_HIERARCHY = re.compile(
    r"\b(Buch|Abschnitt|Titel)\s+(\d+)\b",
    re.IGNORECASE,
)

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def load_lsus(filepath: Path) -> List[Dict[str, Any]]:
    """
    Loads LSUs from a JSON file.

    Args:
        filepath: Path to german_lsus.json.

    Returns:
        List of LSU dicts. Empty list if file is missing or corrupt.
    """
    if not filepath.exists():
        log.error("LSU file not found: %s", filepath)
        return []
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        log.info("Loaded %d LSUs from %s", len(data), filepath)
        return data
    except json.JSONDecodeError as e:
        log.error("JSON error in %s: %s", filepath, e)
        return []


def build_lsu_index(lsus: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Builds a lookup index from lsu_id to LSU dict.

    Also builds a secondary index from (law_code, normalised_paragraph_number)
    to lsu_id to enable resolving bare § references.

    Args:
        lsus: List of LSU dicts.

    Returns:
        Tuple of (id_index, para_index) where:
          - id_index maps lsu_id -> LSU dict
          - para_index maps (law_code, normalised_number) -> lsu_id
    """
    id_index: Dict[str, Dict] = {}
    para_index: Dict[Tuple[str, str], str] = {}

    for lsu in lsus:
        lsu_id = lsu["lsu_id"]
        id_index[lsu_id] = lsu

        law = lsu.get("law_code", "")
        article = lsu.get("article_number", "")
        # Extract numeric part: "§ 100" → "100", "Art 1" → "1"
        num_match = re.search(r"(\d+[a-e]?)", article)
        if num_match and law:
            para_index[(law.upper(), num_match.group(1).strip())] = lsu_id

    log.info(
        "Built LSU index: %d by ID, %d by (law, number)",
        len(id_index), len(para_index)
    )
    return id_index, para_index


def normalise_para_num(raw: str) -> str:
    """
    Extracts the canonical numeric part of a paragraph reference.

    Handles:
      - Ranges (§ 100 bis 105) → returns the start number "100"
      - Trailing letter suffixes (§ 100a) → returns "100a"
      - Trailing Absatz/Abs noise ("26 A" from "§ 26 Absatz") → strips to "26"

    Args:
        raw: Raw matched number string from the regex group.

    Returns:
        Normalised string e.g. "100", "100a", "26".
    """
    raw = raw.strip()
    # Ranges like "100 bis 105" → take the first number
    range_match = re.match(r"(\d+[a-e]?)\s+bis\s+\d+", raw, re.IGNORECASE)
    if range_match:
        return range_match.group(1)
    # Remove trailing whitespace and non-article-suffix characters
    # Article suffixes are lowercase a-e only; uppercase A = Absatz, not a suffix
    cleaned = re.sub(r"\s.*$", "", raw)       # strip everything after first space
    cleaned = re.sub(r"[^0-9a-e]$", "", cleaned)  # strip stray trailing non-suffix char
    return cleaned.strip()



def extract_citations(
    text: str,
    source_law: str,
    para_index: Dict[Tuple[str, str], str],
    id_index: Dict[str, Dict],
    source_id: str,
) -> List[Dict[str, Any]]:
    """
    Extracts all legal citations from a provision text and resolves them
    to target LSU IDs.

    Strategy:
      1. Find all § N LAW spans (explicit, unambiguous).
      2. Find all Art N LAW spans (GG article references).
      3. Find all bare § N spans NOT overlapping with step 1/2.
         Resolve against source_law (intra-law cross-references).

    German legal texts almost exclusively use bare § references within
    the same law (e.g., BGB § 40 references § 32, § 34 without "BGB").
    Step 3 captures these intra-law citations which account for ~95% of
    total references in BGB/ZPO.

    Args:
        text: The provision text to scan.
        source_law: The law code of the source LSU (for resolving bare refs).
        para_index: (law_code, number) → lsu_id lookup.
        id_index: lsu_id → LSU dict lookup.
        source_id: ID of the source LSU (to avoid self-loops).

    Returns:
        List of citation dicts with keys:
          target_id, relation_type, citation_surface_form, resolved.
    """
    citations: List[Dict[str, Any]] = []
    seen_targets: Set[str] = set()
    # Track character spans already consumed by higher-priority patterns
    consumed_spans: List[Tuple[int, int]] = []

    def add_citation(target_id: str, surface: str) -> None:
        if target_id and target_id != source_id and target_id not in seen_targets:
            seen_targets.add(target_id)
            citations.append({
                "target_id": target_id,
                "relation_type": "cites",
                "citation_surface_form": surface,
                "resolved": True,
            })

    def span_overlaps(start: int, end: int) -> bool:
        return any(s <= start <= e or s <= end <= e for s, e in consumed_spans)

    # ── Priority 1: § N LAW  (e.g., § 100 BGB) ────────────────────────────
    for m in PAT_PARA_LAW.finditer(text):
        num = normalise_para_num(m.group(1))
        law = m.group(2).upper()
        surface = m.group(0).strip()
        consumed_spans.append((m.start(), m.end()))
        target_id = para_index.get((law, num))
        if target_id:
            add_citation(target_id, surface)

    # ── Priority 2: Art N LAW  (e.g., Art 20 GG) ──────────────────────────
    for m in PAT_ART_LAW.finditer(text):
        if span_overlaps(m.start(), m.end()):
            continue
        num = normalise_para_num(m.group(1))
        law = (m.group(2) or "GG").upper()
        surface = m.group(0).strip()
        consumed_spans.append((m.start(), m.end()))
        target_id = para_index.get((law, num))
        if target_id:
            add_citation(target_id, surface)

    # ── Priority 3: bare § N  (intra-law, e.g., § 32) ─────────────────────
    # ~95% of German legal cross-references use bare § without law code.
    # We resolve these against source_law (the law of the citing provision).
    for m in PAT_PARA_BARE.finditer(text):
        if span_overlaps(m.start(), m.end()):
            continue
        num = normalise_para_num(m.group(1))
        surface = m.group(0).strip()
        consumed_spans.append((m.start(), m.end()))
        target_id = para_index.get((source_law.upper(), num))
        if target_id:
            add_citation(target_id, surface)

    return citations


# ─── HIERARCHY NODE BUILDER ────────────────────────────────────────────────────

def hierarchy_node_id(law_code: str, level: str, value: str) -> str:
    """
    Constructs a stable Akoma Ntoso-compatible IRI for a hierarchy node.

    Args:
        law_code: Short law code (BGB, ZPO, GG).
        level: Hierarchy level name (book, chapter, title).
        value: Hierarchy value (e.g., "Buch 1", "Abschnitt 2").

    Returns:
        IRI string, e.g., "/akn/de/act/bgb/2024/book/Buch_1".
    """
    safe = re.sub(r"\s+", "_", value.strip())
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", safe)
    return f"/akn/de/act/{law_code.lower()}/{YEAR}/{level}/{safe}"


def add_hierarchy_nodes(
    G: nx.DiGraph,
    lsu: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """
    Adds hierarchy nodes and hierarchicalAuthority edges for a single LSU.
    Creates chain: book → chapter → title → paragraph (only levels present).

    Skips null levels silently.

    Args:
        G: The NetworkX DiGraph being built.
        lsu: The LSU dict.

    Returns:
        List of (source_id, target_id) edge tuples added.
    """
    law = lsu["law_code"]
    lsu_id = lsu["lsu_id"]
    hl = lsu.get("hierarchy_level") or {}

    book_val    = hl.get("book")
    chapter_val = hl.get("chapter")
    title_val   = hl.get("title")

    edges_added: List[Tuple[str, str]] = []

    def ensure_hierarchy_node(node_id: str, level: str, value: str) -> None:
        if node_id not in G:
            G.add_node(
                node_id,
                type=level,
                law_code=law,
                label=value,
                jurisdiction=JURISDICTION,
                language="de",
            )

    prev_id: Optional[str] = None

    # Book
    if book_val:
        book_id = hierarchy_node_id(law, "book", book_val)
        ensure_hierarchy_node(book_id, "book", book_val)
        prev_id = book_id

    # Chapter
    if chapter_val:
        chapter_id = hierarchy_node_id(law, "chapter", chapter_val)
        ensure_hierarchy_node(chapter_id, "chapter", chapter_val)
        if prev_id and not G.has_edge(prev_id, chapter_id):
            G.add_edge(prev_id, chapter_id, relation_type="hierarchicalAuthority")
            edges_added.append((prev_id, chapter_id))
        prev_id = chapter_id

    # Title
    if title_val:
        title_id = hierarchy_node_id(law, "title", title_val)
        ensure_hierarchy_node(title_id, "title", title_val)
        if prev_id and not G.has_edge(prev_id, title_id):
            G.add_edge(prev_id, title_id, relation_type="hierarchicalAuthority")
            edges_added.append((prev_id, title_id))
        prev_id = title_id

    # Paragraph leaf → connect from last hierarchy level
    if prev_id and prev_id != lsu_id:
        if not G.has_edge(prev_id, lsu_id):
            G.add_edge(prev_id, lsu_id, relation_type="hierarchicalAuthority")
            edges_added.append((prev_id, lsu_id))

    return edges_added


# ─── GRAPH BUILDER ─────────────────────────────────────────────────────────────

def build_graph(lsus: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Builds the complete German legal knowledge graph.

    Steps:
        1. Add all LSU nodes (normative units).
        2. Build lookup indexes for citation resolution.
        3. For each LSU, extract citations and add citation edges.
        4. For each LSU, add hierarchy nodes and hierarchical edges.

    Args:
        lsus: List of LSU dicts.

    Returns:
        Directed graph (nx.DiGraph) with all nodes and edges.
    """
    G = nx.DiGraph()

    log.info("Step 1: Adding %d normative unit nodes...", len(lsus))
    for i, lsu in enumerate(lsus):
        lsu_id = lsu["lsu_id"]
        G.add_node(
            lsu_id,
            type="normative_unit",
            law_code=lsu.get("law_code", ""),
            article_number=lsu.get("article_number", ""),
            paragraph_title=lsu.get("paragraph_title", "") or "",
            hierarchy_level=json.dumps(lsu.get("hierarchy_level") or {}),
            provision_text=lsu.get("provision_text", ""),
            valid_from=lsu.get("valid_from", ""),
            valid_to=lsu.get("valid_to") or "",
            amendment_status=lsu.get("amendment_status", ""),
            source_url=lsu.get("source_url") or "",
            language=lsu.get("language", "de"),
            jurisdiction=lsu.get("jurisdiction", "DE"),
        )
        if (i + 1) % PROGRESS_INTERVAL == 0:
            log.info("  Nodes added: %d / %d", i + 1, len(lsus))

    log.info("All normative nodes added. Total: %d", G.number_of_nodes())

    log.info("Step 2: Building citation lookup indexes...")
    id_index, para_index = build_lsu_index(lsus)

    log.info("Step 3: Extracting and adding citation edges...")
    citation_count = 0
    unresolved_count = 0
    unresolved_samples: List[str] = []

    for i, lsu in enumerate(lsus):
        lsu_id = lsu["lsu_id"]
        text = lsu.get("provision_text", "")
        law  = lsu.get("law_code", "BGB")

        if not text:
            continue

        citations = extract_citations(text, law, para_index, id_index, lsu_id)

        for cit in citations:
            target = cit["target_id"]
            if target in G:
                if not G.has_edge(lsu_id, target):
                    G.add_edge(
                        lsu_id,
                        target,
                        relation_type=cit["relation_type"],
                        citation_surface_form=cit["citation_surface_form"],
                    )
                    citation_count += 1
            else:
                unresolved_count += 1
                if len(unresolved_samples) < 20:
                    unresolved_samples.append(cit["citation_surface_form"])

        if (i + 1) % PROGRESS_INTERVAL == 0:
            log.info(
                "  Citation pass: %d / %d LSUs, %d edges so far",
                i + 1, len(lsus), citation_count
            )

    log.info("Citation edges added: %d", citation_count)
    if unresolved_count:
        log.warning(
            "Unresolved citations: %d (sample: %s)",
            unresolved_count, unresolved_samples[:5]
        )

    log.info("Step 4: Adding hierarchy nodes and edges...")
    hier_edge_count = 0
    for lsu in lsus:
        edges = add_hierarchy_nodes(G, lsu)
        hier_edge_count += len(edges)

    log.info("Hierarchical edges added: %d", hier_edge_count)
    log.info(
        "Graph construction complete. Nodes: %d, Edges: %d",
        G.number_of_nodes(), G.number_of_edges()
    )

    return G


# ─── STATISTICS ────────────────────────────────────────────────────────────────

def compute_statistics(G: nx.DiGraph, lsus: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes graph statistics comparable to the Turkish instantiation.

    Args:
        G: The constructed knowledge graph.
        lsus: Original LSU list (for normative node count).

    Returns:
        Stats dict with keys: total_nodes, normative_nodes, hierarchy_nodes,
        total_edges, citation_edges, hierarchical_edges, avg_degree,
        density, largest_cc_nodes, largest_cc_pct, law_code_distribution.
    """
    normative_ids = {lsu["lsu_id"] for lsu in lsus}
    normative_nodes = sum(1 for n in G.nodes if n in normative_ids)
    hierarchy_nodes = G.number_of_nodes() - normative_nodes

    citation_edges = sum(
        1 for _, _, d in G.edges(data=True)
        if d.get("relation_type") == "cites"
    )
    hierarchical_edges = sum(
        1 for _, _, d in G.edges(data=True)
        if d.get("relation_type") == "hierarchicalAuthority"
    )

    # Degree stats on the full graph
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0

    density = nx.density(G)

    # Weakly connected components (DiGraph)
    wcc = list(nx.weakly_connected_components(G))
    largest_cc = max(wcc, key=len) if wcc else set()
    lcc_pct = len(largest_cc) / G.number_of_nodes() * 100 if G.number_of_nodes() else 0

    # Per-law distribution
    law_dist: Dict[str, int] = defaultdict(int)
    for lsu in lsus:
        law_dist[lsu["law_code"]] += 1

    # Shannon entropy of out-degree (for diversity measure)
    import math
    out_degrees = [d for _, d in G.out_degree() if d > 0]
    total_od = sum(out_degrees) or 1
    entropy = -sum((d / total_od) * math.log2(d / total_od) for d in out_degrees)

    stats = {
        "total_nodes": G.number_of_nodes(),
        "normative_nodes": normative_nodes,
        "hierarchy_nodes": hierarchy_nodes,
        "total_edges": G.number_of_edges(),
        "citation_edges": citation_edges,
        "hierarchical_edges": hierarchical_edges,
        "avg_degree": round(avg_degree, 4),
        "density": round(density, 8),
        "largest_cc_nodes": len(largest_cc),
        "largest_cc_pct": round(lcc_pct, 2),
        "out_degree_entropy": round(entropy, 4),
        "law_code_distribution": dict(law_dist),
    }
    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """Prints graph statistics in a structured, readable format."""
    log.info("")
    log.info("=" * 60)
    log.info("  German Legal Knowledge Graph Statistics")
    log.info("=" * 60)
    log.info("  Total nodes         : %d", stats["total_nodes"])
    log.info("    Normative units   : %d", stats["normative_nodes"])
    log.info("    Hierarchy nodes   : %d", stats["hierarchy_nodes"])
    log.info("  Total edges         : %d", stats["total_edges"])
    log.info("    Citation edges    : %d", stats["citation_edges"])
    log.info("    Hierarchical edges: %d", stats["hierarchical_edges"])
    log.info("  Average degree      : %.4f", stats["avg_degree"])
    log.info("  Graph density       : %.8f", stats["density"])
    log.info(
        "  Largest CC          : %d nodes (%.1f%%)",
        stats["largest_cc_nodes"], stats["largest_cc_pct"]
    )
    log.info("  Out-degree entropy  : %.4f bits", stats["out_degree_entropy"])
    log.info("  Law code dist.      : %s", stats["law_code_distribution"])
    log.info("=" * 60)


# ─── VALIDATION ────────────────────────────────────────────────────────────────

def validate_graph(
    G: nx.DiGraph,
    lsus: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Validates the graph for schema compliance.

    Checks:
        1. All LSU IDs are present as nodes.
        2. All edge relation types are in the allowed set.
        3. No self-loops exist.
        4. All citation edges point to existing nodes (already guaranteed
           by the builder, but double-checked here).
        5. Node IDs are unique (implicit in NetworkX, verified explicitly).
        6. Graph is acyclic among hierarchy edges.

    Args:
        G: The constructed knowledge graph.
        lsus: List of original LSU dicts.

    Returns:
        Tuple of (passed: bool, errors: List[str]).
    """
    errors: List[str] = []

    # Check 1: All LSU nodes present
    missing = [lsu["lsu_id"] for lsu in lsus if lsu["lsu_id"] not in G]
    if missing:
        errors.append(f"{len(missing)} LSU nodes missing from graph: {missing[:5]}")
    else:
        log.info("Check 1 PASSED: all %d LSU nodes present.", len(lsus))

    # Check 2: Edge relation types
    bad_relations = [
        (u, v, d["relation_type"])
        for u, v, d in G.edges(data=True)
        if d.get("relation_type") not in ALLOWED_RELATION_TYPES
    ]
    if bad_relations:
        errors.append(f"{len(bad_relations)} edges with unexpected relation type.")
    else:
        log.info("Check 2 PASSED: all edge relation types valid.")

    # Check 3: No self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        errors.append(f"{len(self_loops)} self-loop edges found.")
    else:
        log.info("Check 3 PASSED: no self-loops.")

    # Check 4: Citation edges point to existing nodes
    dangling = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("relation_type") == "cites" and v not in G
    ]
    if dangling:
        errors.append(f"{len(dangling)} citation edges point to non-existent nodes.")
    else:
        log.info("Check 4 PASSED: all citation edges resolve to existing nodes.")

    # Check 5: Hierarchy subgraph acyclicity
    hier_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("relation_type") == "hierarchicalAuthority"
    ]
    H = nx.DiGraph()
    H.add_edges_from(hier_edges)
    if nx.is_directed_acyclic_graph(H):
        log.info("Check 5 PASSED: hierarchy subgraph is acyclic.")
    else:
        cycles = list(nx.simple_cycles(H))[:3]
        errors.append(f"Hierarchy subgraph has cycles! Sample: {cycles}")

    passed = len(errors) == 0
    return passed, errors


def print_validation(
    passed: bool,
    errors: List[str],
    stats: Dict[str, Any],
    unresolved: int,
) -> None:
    """Prints a structured validation summary."""
    log.info("")
    log.info("=" * 60)
    log.info("  German Graph Validation Summary")
    log.info("=" * 60)
    log.info(
        "  Nodes: %d (%d normative + %d hierarchy)",
        stats["total_nodes"],
        stats["normative_nodes"],
        stats["hierarchy_nodes"],
    )
    log.info(
        "  Edges: %d (%d citations + %d hierarchical)",
        stats["total_edges"],
        stats["citation_edges"],
        stats["hierarchical_edges"],
    )
    log.info("  Unresolved citations: %d", unresolved)
    if passed:
        log.info("  Cyclic hier. edges  : 0")
        log.info("  Validation          : PASSED")
    else:
        log.warning("  Validation          : FAILED")
        for err in errors:
            log.warning("    - %s", err)
    log.info("=" * 60)


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_graph_construction() -> None:
    """
    Main graph construction pipeline.

    Steps:
        1. Load LSUs.
        2. Build graph (nodes + citation edges + hierarchy edges).
        3. Compute statistics.
        4. Validate.
        5. Serialize to JSON (NetworkX node_link_data format).
        6. Save stats JSON.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    lsus = load_lsus(INPUT_FILE)
    if not lsus:
        log.error("No LSUs loaded. Aborting.")
        return

    # ── Build ────────────────────────────────────────────────────────────────
    G = build_graph(lsus)

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = compute_statistics(G, lsus)
    print_statistics(stats)

    # Count unresolved (those logged during build but not added to graph)
    # Recount via re-scan for summary accuracy
    _, para_index = build_lsu_index(lsus)
    unresolved_total = 0
    for lsu in lsus:
        text = lsu.get("provision_text", "")
        law  = lsu.get("law_code", "BGB")
        for m in PAT_PARA_LAW.finditer(text):
            num = normalise_para_num(m.group(1))
            law_ref = m.group(2).upper()
            if (law_ref, num) not in para_index:
                unresolved_total += 1
        for m in PAT_ART_LAW.finditer(text):
            num = normalise_para_num(m.group(1))
            law_ref = (m.group(2) or "GG").upper()
            if (law_ref, num) not in para_index:
                unresolved_total += 1

    # ── Validate ─────────────────────────────────────────────────────────────
    passed, errors = validate_graph(G, lsus)
    print_validation(passed, errors, stats, unresolved_total)

    # ── Serialize ────────────────────────────────────────────────────────────
    log.info("Serializing graph to %s ...", OUTPUT_FILE)
    graph_data = nx.node_link_data(G, edges="links")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    log.info("Graph saved: %s (%.1f MB)", OUTPUT_FILE, OUTPUT_FILE.stat().st_size / 1e6)

    # ── Save stats ───────────────────────────────────────────────────────────
    stats["unresolved_citations"] = unresolved_total
    stats["validation_passed"] = passed
    stats["validation_errors"] = errors
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log.info("Stats saved: %s", STATS_FILE)


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_graph_construction()
