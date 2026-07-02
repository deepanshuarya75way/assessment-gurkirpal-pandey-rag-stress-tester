import json
import os
import sys
sys.path.append('.')

from src.chunking.fixed_size import FixedSizeChunker

# show what eval set has
print("EVAL SET CHUNK IDs (first question):")
with open('data/eval/gold_eval_set.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

for q in eval_data['eval_set'][:5]:
    if q['question_type'] == 'unanswerable':
        continue
    print(f"Q: {q['question'][:50]}")
    print(f"   Source: {q['source_doc']}")
    print(f"   IDs: {q['ground_truth_chunk_ids'][:3]}")
    print()

# show what chunker generates for same document
print("\nCHUNKER GENERATED IDs for monetary_policy_report_2022_apr:")
pages = []
with open('data/processed/monetary_policy_report_2022_apr.jsonl',
          'r', encoding='utf-8') as f:
    for line in f:
        pages.append(json.loads(line))

chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
chunks = chunker.chunk(pages)
print(f"Total chunks: {len(chunks)}")
for c in chunks[:10]:
    print(f"  {c.id}")