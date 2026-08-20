"""
04_german_judicial_decisions.py
================================
Task 4A of 7: German Judicial Decisions — Simulated Acquisition

For cross-jurisdictional validation, we need judicial decisions linked to
the normative units in the graph. Real BGH/BVerfG decisions require legal
database access (Juris, Beck-online). This script generates a structurally
faithful simulated judicial decision corpus:

  - 500 BGH decisions (Bundesgerichtshof — civil, equivalent to Yargıtay)
  - 100 BVerfG decisions (Bundesverfassungsgericht — constitutional, equiv. AYM)
  - 50 BAG decisions (Bundesarbeitsgericht — labour, for ZPO procedural coverage)

Each decision is linked to 1-4 BGB/ZPO/GG articles via INTERPRETS edges,
mirroring the Turkish graph's interpretive layer.

Decisions follow realistic German citation patterns:
  - BGH chamber distribution (ZR civil, StR criminal chambers)
  - BVerfG Senat structure (1. Senat = fundamental rights, 2. Senat = state org.)
  - Temporal distribution 2020–2025
  - Realistic docket numbers (e.g., "VIII ZR 123/23")

Usage:
    python 04_german_judicial_decisions.py
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
from networkx.readwrite import json_graph

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

LSU_FILE   = Path("data/german/lsus/german_lsus.json")
GRAPH_FILE = Path("data/german/graph/german_graph.json")
OUTPUT_DIR = Path("data/german/decisions")
OUTPUT_FILE = OUTPUT_DIR / "german_decisions.json"
GRAPH_OUT   = Path("data/german/graph/german_graph_with_decisions.json")
STATS_OUT   = Path("data/german/graph/german_graph_with_decisions_stats.json")

SEED = 42
random.seed(SEED)

BGH_CHAMBERS = [
    "I ZR", "II ZR", "III ZR", "IV ZR", "V ZR",
    "VI ZR", "VII ZR", "VIII ZR", "IX ZR", "X ZR",
    "I StR", "II StR", "III StR", "IV StR", "V StR",
    "I ZB", "II ZB",
]
BVERFG_CHAMBERS = ["1. Senat", "2. Senat", "1. Kammer", "2. Kammer", "3. Kammer"]
BAG_CHAMBERS = ["1. Senat", "2. Senat", "3. Senat", "4. Senat", "5. Senat",
                "6. Senat", "7. Senat", "8. Senat", "9. Senat", "10. Senat"]

DECISION_SPECS = [
    {"court": "BGH",    "count": 500, "chambers": BGH_CHAMBERS,    "prefix": "BGHZ"},
    {"court": "BVerfG", "count": 100, "chambers": BVERFG_CHAMBERS, "prefix": "BVerfGE"},
    {"court": "BAG",    "count":  50, "chambers": BAG_CHAMBERS,    "prefix": "BAGE"},
]

YEARS = list(range(2020, 2026))

# BGB articles most cited by BGH (civil law hub provisions)
BGH_PRIORITY_ARTICLES = [
    "§ 133", "§ 157", "§ 242", "§ 280", "§ 281", "§ 433",
    "§ 434", "§ 437", "§ 535", "§ 611", "§ 631", "§ 823",
    "§ 826", "§ 929", "§ 985", "§ 1004",
]
# BVerfG priority: GG articles
BVERFG_PRIORITY_ARTICLES = [
    "Art 1", "Art 2", "Art 3", "Art 5", "Art 12",
    "Art 14", "Art 19", "Art 20", "Art 103",
]

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("german_judicial_decisions.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("german_decisions")

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def load_lsus(filepath: Path) -> List[Dict[str, Any]]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def load_graph(filepath: Path) -> nx.DiGraph:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data, directed=True, edges="links")


def make_docket_number(court: str, chamber: str, year: int, seq: int) -> str:
    """
    Generates a realistic German court docket number.

    Examples:
        BGH:    "VIII ZR 123/23"
        BVerfG: "1 BvR 456/22"
        BAG:    "8 AZR 789/24"
    """
    yr_short = str(year)[-2:]
    if court == "BGH":
        return f"{chamber} {seq}/{yr_short}"
    elif court == "BVerfG":
        senat_num = chamber.split(".")[0].strip()
        return f"{senat_num} BvR {seq}/{yr_short}"
    else:  # BAG
        senat_num = chamber.split(".")[0].strip()
        return f"{senat_num} AZR {seq}/{yr_short}"


def pick_cited_lsus(
    court: str,
    all_lsus: List[Dict],
    lsu_by_law: Dict[str, List[Dict]],
    n_citations: int,
) -> List[str]:
    """
    Picks LSU IDs to cite for a given decision, weighted by court type.

    BGH → mostly BGB, some ZPO
    BVerfG → mostly GG, some BGB
    BAG → mostly BGB (§ 611, 631...), ZPO

    Args:
        court: Court code.
        all_lsus: Full LSU list.
        lsu_by_law: Dict mapping law code → list of LSUs.
        n_citations: Number of citations to select.

    Returns:
        List of lsu_id strings.
    """
    pool: List[Dict] = []
    if court == "BGH":
        pool = lsu_by_law.get("BGB", []) * 4 + lsu_by_law.get("ZPO", [])
    elif court == "BVerfG":
        pool = lsu_by_law.get("GG", []) * 5 + lsu_by_law.get("BGB", [])
    else:  # BAG
        pool = lsu_by_law.get("BGB", []) * 3 + lsu_by_law.get("ZPO", []) * 2

    # Filter to in-force only
    pool = [l for l in pool if l.get("amendment_status") == "in_force"]
    if not pool:
        pool = all_lsus

    selected = random.sample(pool, min(n_citations, len(pool)))
    return [l["lsu_id"] for l in selected]


def generate_decision_text(
    court: str, docket: str, year: int, cited_articles: List[str]
) -> str:
    """
    Generates a minimal but structurally realistic decision text stub.

    Args:
        court: Court code.
        docket: Docket number string.
        year: Decision year.
        cited_articles: List of cited article numbers.

    Returns:
        Short German-language decision text stub.
    """
    citations = ", ".join(cited_articles[:4]) if cited_articles else "den einschlaegigen Vorschriften"
    templates = [
        f"Der {court} hat mit Urteil vom {random.randint(1,28)}.{random.randint(1,12)}.{year} "
        f"(Az.: {docket}) entschieden, dass gemaess {citations} die Revision zurueckzuweisen ist.",
        f"In dem Revisionsverfahren {docket} hat der {court} unter Beruecksichtigung von "
        f"{citations} festgestellt, dass die Vorinstanz zutreffend entschieden hat.",
        f"Der {court} ({docket}) hat die Rechtsfrage unter Anwendung von {citations} "
        f"im Sinne der staendigen Rechtsprechung entschieden.",
    ]
    return random.choice(templates)

# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def generate_decisions(lsus: List[Dict]) -> List[Dict]:
    """
    Generates the full simulated judicial decision corpus.

    Returns:
        List of decision dicts with lsu_citations field.
    """
    lsu_by_law: Dict[str, List[Dict]] = {}
    for lsu in lsus:
        law = lsu.get("law_code", "BGB")
        lsu_by_law.setdefault(law, []).append(lsu)

    decisions: List[Dict] = []
    global_seq = 100

    for spec in DECISION_SPECS:
        court = spec["court"]
        log.info("Generating %d %s decisions...", spec["count"], court)

        for i in range(spec["count"]):
            chamber = random.choice(spec["chambers"])
            year    = random.choice(YEARS)
            seq     = global_seq + i
            docket  = make_docket_number(court, chamber, year, seq)
            n_cits  = random.randint(1, 4)
            cited   = pick_cited_lsus(court, lsus, lsu_by_law, n_cits)
            cited_articles = [
                next((l["article_number"] for l in lsus if l["lsu_id"] == c), c)
                for c in cited
            ]
            text = generate_decision_text(court, docket, year, cited_articles)

            decision = {
                "decision_id": f"de-{court.lower()}-{seq}",
                "court_name": court,
                "chamber": chamber,
                "decision_number": docket,
                "decision_date": f"{year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "decision_type": "Urteil" if court != "BVerfG" else "Beschluss",
                "jurisdiction": "DE",
                "language": "de",
                "lsu_citations": cited,          # list of lsu_ids this decision interprets
                "cited_articles": cited_articles,
                "decision_text": text,
                "amendment_status": "in_force",
                "valid_from": f"{year}-01-01",
                "valid_to": None,
            }
            decisions.append(decision)

        global_seq += spec["count"]
        log.info("  -> %d %s decisions generated.", spec["count"], court)

    return decisions


def add_decisions_to_graph(
    G: nx.DiGraph, decisions: List[Dict]
) -> Tuple[int, int]:
    """
    Adds decision nodes and INTERPRETS edges to the existing graph.

    Args:
        G: Existing knowledge graph.
        decisions: List of decision dicts.

    Returns:
        Tuple of (nodes_added, edges_added).
    """
    nodes_added = 0
    edges_added = 0

    for d in decisions:
        did = d["decision_id"]
        if did not in G:
            G.add_node(
                did,
                type="judicial_decision",
                court_name=d["court_name"],
                chamber=d["chamber"],
                decision_number=d["decision_number"],
                decision_date=d["decision_date"],
                decision_type=d["decision_type"],
                jurisdiction=d["jurisdiction"],
                language=d["language"],
                amendment_status=d["amendment_status"],
                valid_from=d["valid_from"],
                valid_to=d.get("valid_to") or "",
                decision_text=d["decision_text"],
            )
            nodes_added += 1

        for lsu_id in d["lsu_citations"]:
            if lsu_id in G and not G.has_edge(did, lsu_id):
                G.add_edge(
                    did,
                    lsu_id,
                    relation_type="interprets",
                    court=d["court_name"],
                )
                edges_added += 1

    return nodes_added, edges_added


def compute_full_stats(G: nx.DiGraph, lsus: List[Dict], decisions: List[Dict]) -> Dict:
    import math

    normative_ids = {l["lsu_id"] for l in lsus}
    decision_ids  = {d["decision_id"] for d in decisions}

    normative_n  = sum(1 for n in G if n in normative_ids)
    decision_n   = sum(1 for n in G if n in decision_ids)
    hierarchy_n  = G.number_of_nodes() - normative_n - decision_n

    cite_edges   = sum(1 for _,_,d in G.edges(data=True) if d.get("relation_type") == "cites")
    hier_edges   = sum(1 for _,_,d in G.edges(data=True) if d.get("relation_type") == "hierarchicalAuthority")
    interp_edges = sum(1 for _,_,d in G.edges(data=True) if d.get("relation_type") == "interprets")

    degrees = [d for _, d in G.degree()]
    avg_deg = sum(degrees) / len(degrees) if degrees else 0

    wcc = list(nx.weakly_connected_components(G))
    lcc = max(wcc, key=len) if wcc else set()
    lcc_pct = len(lcc) / G.number_of_nodes() * 100

    court_dist: Dict[str, int] = {}
    for d in decisions:
        court_dist[d["court_name"]] = court_dist.get(d["court_name"], 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "normative_nodes": normative_n,
        "decision_nodes": decision_n,
        "hierarchy_nodes": hierarchy_n,
        "total_edges": G.number_of_edges(),
        "citation_edges": cite_edges,
        "hierarchical_edges": hier_edges,
        "interprets_edges": interp_edges,
        "avg_degree": round(avg_deg, 4),
        "density": round(nx.density(G), 8),
        "largest_cc_nodes": len(lcc),
        "largest_cc_pct": round(lcc_pct, 2),
        "court_distribution": court_dist,
    }


def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading LSUs and existing graph...")
    lsus = load_lsus(LSU_FILE)
    G    = load_graph(GRAPH_FILE)
    log.info("Loaded: %d LSUs, graph has %d nodes / %d edges",
             len(lsus), G.number_of_nodes(), G.number_of_edges())

    log.info("Generating simulated judicial decisions...")
    decisions = generate_decisions(lsus)
    log.info("Generated %d decisions total.", len(decisions))

    log.info("Adding decisions to graph...")
    nodes_added, edges_added = add_decisions_to_graph(G, decisions)
    log.info("Added %d decision nodes, %d interprets edges.", nodes_added, edges_added)

    # Save decisions JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False, indent=2)
    log.info("Decisions saved: %s", OUTPUT_FILE)

    # Save updated graph
    graph_data = nx.node_link_data(G, edges="links")
    with open(GRAPH_OUT, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    log.info("Updated graph saved: %s", GRAPH_OUT)

    # Stats
    stats = compute_full_stats(G, lsus, decisions)
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info("  German Graph with Judicial Decisions")
    log.info("=" * 60)
    log.info("  Total nodes       : %d", stats["total_nodes"])
    log.info("    Normative       : %d", stats["normative_nodes"])
    log.info("    Decisions       : %d", stats["decision_nodes"])
    log.info("    Hierarchy       : %d", stats["hierarchy_nodes"])
    log.info("  Total edges       : %d", stats["total_edges"])
    log.info("    Citation        : %d", stats["citation_edges"])
    log.info("    Hierarchical    : %d", stats["hierarchical_edges"])
    log.info("    Interprets      : %d", stats["interprets_edges"])
    log.info("  Avg degree        : %.4f", stats["avg_degree"])
    log.info("  Density           : %.8f", stats["density"])
    log.info("  Largest CC        : %d nodes (%.1f%%)",
             stats["largest_cc_nodes"], stats["largest_cc_pct"])
    log.info("  Court dist.       : %s", stats["court_distribution"])
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
