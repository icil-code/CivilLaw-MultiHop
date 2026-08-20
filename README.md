# CivilLaw-MultiHop: Cross-Jurisdictional Legal Retrieval Benchmark

This repository contains the evaluation benchmarks, knowledge graph data, and retrieval scripts accompanying the double-blind submission **"Knowledge Infrastructure Engineering for Multi-Step Legal Retrieval in Civil-Law Jurisdictions"**.

## Repository Structure

```text
.
├── data/
│   ├── turkish/           # Primary instantiation (Turkish legal system)
│   │   ├── benchmark_150_queries.json
│   │   ├── lsu_corpus.zip
│   │   └── ontology_schema.ttl
│   └── german/            # Cross-jurisdictional validation (German legal system)
│       ├── benchmark/CivilLaw_MultiHop_DE_Benchmark.json
│       ├── graph/german_graph_with_decisions.json
│       └── lsus/german_lsus.json
├── src/
│   ├── turkish/           # Retrieval and traversal scripts for Turkish graph
│   │   ├── evaluate_mrr.py
│   │   └── graph_traversal.py
│   └── german/            # Pipeline for German validation
│       ├── 01_german_acquisition.py
│       ├── 02_german_lsu_transformation.py
│       ├── 03_german_graph_construction.py
│       ├── 04_german_judicial_decisions.py
│       ├── 05_german_benchmark.py
│       └── 06_german_retrieval_ablation.py
├── requirements.txt
└── README.md
```

## Data Assets

- **Legal Semantic Units (LSUs)**: Formatted as Akoma Ntoso-compatible standard JSON representations.
- **Knowledge Graphs**: NetworkX-compatible structural properties (`cites`, `hierarchicalAuthority`, `interprets`).
- **Benchmarks**: 
  - `benchmark_150_queries.json` (N=150) covering the Turkish corpus.
  - `CivilLaw_MultiHop_DE_Benchmark.json` (N=75) covering the German corpus.

## Reproducing the Results

To replicate the German pipeline and ablation study:
```bash
pip install -r requirements.txt
cd src/german
python 06_german_retrieval_ablation.py
```

## License
MIT License.
