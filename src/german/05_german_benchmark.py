"""
05_german_benchmark.py
======================
Task 4C: German Multi-Hop Retrieval Benchmark Construction

Generates CivilLaw-MultiHop-DE — a structured benchmark of 75 multi-step
retrieval queries over the German legal knowledge graph, mirroring the
Turkish benchmark (N=150) at half scale (proportional to corpus size).

Stratification (mirrors Turkish benchmark proportions):
  - 1-hop:  23 queries (30.7%) — direct provision lookup
  - 2-hop:  37 queries (49.3%) — decision → cited provision
  - 3-hop:  15 queries (20.0%) — hierarchy + citation chain

Each query has:
  - query_text: Natural language legal question (German)
  - hop_stratum: 1, 2, or 3
  - gold_evidence_path: list of lsu_ids forming the evidence chain
  - failure_mode: F1 (temporal), F2 (hierarchy), F3 (multi-hop), F4 (metadata)
  - design_intent: explanation of what the query tests

Usage:
    python 05_german_benchmark.py
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from networkx.readwrite import json_graph

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

GRAPH_FILE     = Path("data/german/graph/german_graph_with_decisions.json")
LSU_FILE       = Path("data/german/lsus/german_lsus.json")
DECISIONS_FILE = Path("data/german/decisions/german_decisions.json")
OUTPUT_DIR     = Path("data/german/benchmark")
OUTPUT_FILE    = OUTPUT_DIR / "CivilLaw_MultiHop_DE_Benchmark.json"
DEVSET_FILE    = OUTPUT_DIR / "CivilLaw_MultiHop_DE_DevSet.json"

SEED = 42
random.seed(SEED)

N_TOTAL  = 75
N_1HOP   = 23
N_2HOP   = 37
N_3HOP   = 15
N_DEVSET = 8

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("german_benchmark.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("german_benchmark")

# ─── QUERY TEMPLATES ───────────────────────────────────────────────────────────

QUERY_TEMPLATES_1HOP = [
    ("Was regelt {article} {law}?",
     "F3", "Direct normative unit lookup — tests basic retrieval"),
    ("Welche Voraussetzungen stellt {article} {law} auf?",
     "F3", "Provision requirement lookup"),
    ("Was besagt {article} {law} zum Thema Haftung?",
     "F4", "Thematic provision lookup with metadata noise potential"),
    ("Welche Rechtsfolgen sieht {article} {law} vor?",
     "F3", "Legal consequence lookup"),
    ("Wie definiert {article} {law} den relevanten Begriff?",
     "F3", "Definition lookup in normative text"),
]

QUERY_TEMPLATES_2HOP = [
    ("Wie hat der {court} {article} {law} ausgelegt?",
     "F2", "Decision → provision interpretation chain"),
    ("Welche BGH-Entscheidung betrifft {article} {law} und was sind deren Voraussetzungen?",
     "F3", "Decision → cited provision multi-hop"),
    ("Was entschied das {court} zu {article} {law} im Jahr {year}?",
     "F1", "Temporal validity check — F1 failure mode"),
    ("Wie legt der {court} den Anwendungsbereich von {article} {law} aus?",
     "F2", "Hierarchical authority interpretation chain"),
    ("Welche Norm hat der {court} in {docket} angewendet?",
     "F3", "Decision to provision reverse lookup"),
]

QUERY_TEMPLATES_3HOP = [
    ("Welche BGB-Norm liegt der {court}-Entscheidung {docket} zugrunde und wie verhält sie sich zu {article2} {law}?",
     "F2", "3-hop: decision → primary norm → related norm"),
    ("Wie verhalten sich {article} {law} und {article2} {law2} zueinander nach der Rechtsprechung des {court}?",
     "F3", "3-hop: norm1 → decision → norm2 citation chain"),
    ("Welche Norm des {law} wird in der {court}-Entscheidung {docket} ausgelegt, und welche weitere Norm des {law2} wird dabei herangezogen?",
     "F1", "3-hop temporal + hierarchy failure mode"),
]

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def load_data():
    lsus = json.load(open(LSU_FILE, encoding="utf-8"))
    decisions = json.load(open(DECISIONS_FILE, encoding="utf-8"))
    with open(GRAPH_FILE, encoding="utf-8") as f:
        G = json_graph.node_link_graph(json.load(f), directed=True, edges="links")
    return lsus, decisions, G


def pick_lsu(lsus: List[Dict], law_code: Optional[str] = None) -> Dict:
    pool = [l for l in lsus if l.get("amendment_status") == "in_force"]
    if law_code:
        pool = [l for l in pool if l.get("law_code") == law_code]
    return random.choice(pool)


def pick_decision(decisions: List[Dict], court: Optional[str] = None) -> Dict:
    pool = decisions
    if court:
        pool = [d for d in decisions if d["court_name"] == court]
    return random.choice(pool)


def make_1hop_query(lsus: List[Dict], idx: int) -> Dict:
    lsu = pick_lsu(lsus)
    tmpl, fm, intent = random.choice(QUERY_TEMPLATES_1HOP)
    law = lsu["law_code"]
    article = lsu["article_number"]
    query_text = tmpl.format(article=article, law=law)
    return {
        "query_id": f"DE-1H-{idx:03d}",
        "hop_stratum": 1,
        "query_text": query_text,
        "gold_evidence_path": [lsu["lsu_id"]],
        "failure_mode": fm,
        "design_intent": intent,
        "law_codes_involved": [law],
    }


def make_2hop_query(lsus: List[Dict], decisions: List[Dict], idx: int) -> Dict:
    decision = pick_decision(decisions)
    cited_ids = decision.get("lsu_citations", [])
    if not cited_ids:
        cited_ids = [pick_lsu(lsus)["lsu_id"]]
    cited_id = random.choice(cited_ids)
    cited_lsu = next((l for l in lsus if l["lsu_id"] == cited_id), None)
    if not cited_lsu:
        cited_lsu = pick_lsu(lsus)
        cited_id = cited_lsu["lsu_id"]

    tmpl, fm, intent = random.choice(QUERY_TEMPLATES_2HOP)
    query_text = tmpl.format(
        court=decision["court_name"],
        article=cited_lsu["article_number"],
        law=cited_lsu["law_code"],
        year=decision["decision_date"][:4],
        docket=decision["decision_number"],
    )
    return {
        "query_id": f"DE-2H-{idx:03d}",
        "hop_stratum": 2,
        "query_text": query_text,
        "gold_evidence_path": [decision["decision_id"], cited_id],
        "failure_mode": fm,
        "design_intent": intent,
        "law_codes_involved": list({cited_lsu["law_code"]}),
        "court": decision["court_name"],
    }


def make_3hop_query(lsus: List[Dict], decisions: List[Dict], idx: int) -> Dict:
    decision = pick_decision(decisions)
    cited_ids = decision.get("lsu_citations", [])
    if len(cited_ids) < 2:
        cited_ids = [pick_lsu(lsus)["lsu_id"], pick_lsu(lsus)["lsu_id"]]
    primary_id = cited_ids[0]
    secondary_id = cited_ids[1] if len(cited_ids) > 1 else pick_lsu(lsus)["lsu_id"]

    lsu1 = next((l for l in lsus if l["lsu_id"] == primary_id), pick_lsu(lsus))
    lsu2 = next((l for l in lsus if l["lsu_id"] == secondary_id), pick_lsu(lsus))

    tmpl, fm, intent = random.choice(QUERY_TEMPLATES_3HOP)
    query_text = tmpl.format(
        court=decision["court_name"],
        article=lsu1["article_number"],
        article2=lsu2["article_number"],
        law=lsu1["law_code"],
        law2=lsu2["law_code"],
        docket=decision["decision_number"],
    )
    return {
        "query_id": f"DE-3H-{idx:03d}",
        "hop_stratum": 3,
        "query_text": query_text,
        "gold_evidence_path": [decision["decision_id"], lsu1["lsu_id"], lsu2["lsu_id"]],
        "failure_mode": fm,
        "design_intent": intent,
        "law_codes_involved": list({lsu1["law_code"], lsu2["law_code"]}),
        "court": decision["court_name"],
    }

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def run_benchmark_construction() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    lsus, decisions, G = load_data()
    log.info("Loaded %d LSUs, %d decisions, graph %d nodes/%d edges",
             len(lsus), len(decisions), G.number_of_nodes(), G.number_of_edges())

    benchmark: List[Dict] = []

    log.info("Generating %d 1-hop queries...", N_1HOP)
    for i in range(N_1HOP):
        benchmark.append(make_1hop_query(lsus, i + 1))

    log.info("Generating %d 2-hop queries...", N_2HOP)
    for i in range(N_2HOP):
        benchmark.append(make_2hop_query(lsus, decisions, i + 1))

    log.info("Generating %d 3-hop queries...", N_3HOP)
    for i in range(N_3HOP):
        benchmark.append(make_3hop_query(lsus, decisions, i + 1))

    random.shuffle(benchmark)
    for i, q in enumerate(benchmark):
        q["index"] = i + 1

    # Dev set (disjoint)
    devset = random.sample(benchmark, N_DEVSET)
    devset_ids = {q["query_id"] for q in devset}
    test_set = [q for q in benchmark if q["query_id"] not in devset_ids]

    # Validate gold paths
    valid = sum(1 for q in benchmark if all(
        nid in G for nid in q["gold_evidence_path"]
    ))
    log.info("Gold paths fully in graph: %d / %d", valid, len(benchmark))

    # Failure mode distribution
    fm_dist: Dict[str, int] = {}
    for q in benchmark:
        fm = q["failure_mode"]
        fm_dist[fm] = fm_dist.get(fm, 0) + 1

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    with open(DEVSET_FILE, "w", encoding="utf-8") as f:
        json.dump(devset, f, ensure_ascii=False, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info("  CivilLaw-MultiHop-DE Benchmark Summary")
    log.info("=" * 60)
    log.info("  Total queries  : %d", len(benchmark))
    log.info("  1-hop          : %d", N_1HOP)
    log.info("  2-hop          : %d", N_2HOP)
    log.info("  3-hop          : %d", N_3HOP)
    log.info("  Dev set        : %d (disjoint)", len(devset))
    log.info("  Test set       : %d", len(test_set))
    log.info("  Gold paths OK  : %d / %d", valid, len(benchmark))
    log.info("  Failure modes  : %s", fm_dist)
    log.info("  Output         : %s", OUTPUT_FILE)
    log.info("=" * 60)


if __name__ == "__main__":
    run_benchmark_construction()
