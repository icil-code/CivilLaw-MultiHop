# CivilLaw-MultiHop: Benchmark and Baseline Knowledge Infrastructure for Multi-Step Legal Retrieval

This repository contains the benchmark dataset, ontology schema, and evaluation code for the paper:
**"Knowledge Infrastructure Engineering for Multi-Step Legal Retrieval in Civil Law Jurisdictions"**

## Contents
- `data/benchmark_150_queries.json`: Complete dataset of 150 expert-validated multi-hop legal queries with gold standard relevance paths.
- `data/lsu_corpus.json`: Anonymized full dataset of 15,257 Legal Semantic Units (LSUs).
- `data/ontology_schema.ttl`: Semantic graph ontology schema for legal entities and temporal validity.
- `src/`: Baseline evaluation scripts for BM25, Dense RAG, and TL-GraphRAG.

## Citation
If you use this benchmark in your research, please cite our paper.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21923390.svg)](https://doi.org/10.5281/zenodo.21923390)
