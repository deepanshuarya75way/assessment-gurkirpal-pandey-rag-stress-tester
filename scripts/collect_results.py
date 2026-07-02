# scripts/collect_results.py
# Reads all result JSON files and prints summary table

import json
import os

results_dir = 'results'
files = sorted([f for f in os.listdir(results_dir)
               if f.endswith('.json')])

print(f"{'Config':<45} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
print("="*75)

for filename in files:
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)

    m = data.get('metrics', {})
    chunker = data.get('chunker', '?')
    embedder = data.get('embedder', '?')
    retriever = data.get('retriever', '?')
    reranker = data.get('reranker_depth', None)

    name = f"{chunker}+{embedder}+{retriever}"
    if reranker:
        name += f"+rerank{reranker}"

    print(f"{name:<45} "
          f"{m.get('recall@1',0):>6.4f} "
          f"{m.get('recall@3',0):>6.4f} "
          f"{m.get('recall@5',0):>6.4f} "
          f"{m.get('mrr',0):>6.4f}")