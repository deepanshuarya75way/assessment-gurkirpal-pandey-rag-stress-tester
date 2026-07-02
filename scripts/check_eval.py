import json
import sys
sys.path.append('.')

with open('data/eval/gold_eval_set.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for q in data['eval_set'][:5]:
    print(f"Q: {q['question'][:50]}")
    print(f"   IDs: {q['ground_truth_chunk_ids']}")
    print()