import json
import os

# Paths updated to point to the full datasets
BENCHMARK_QUERIES_PATH = os.path.join(os.path.dirname(__file__), '../data/benchmark_150_queries.json')
LSU_CORPUS_PATH = os.path.join(os.path.dirname(__file__), '../data/lsu_corpus.json')

def load_data():
    with open(BENCHMARK_QUERIES_PATH, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} benchmark queries.")
    
    with open(LSU_CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} LSUs.")
    return queries, corpus

if __name__ == '__main__':
    load_data()
