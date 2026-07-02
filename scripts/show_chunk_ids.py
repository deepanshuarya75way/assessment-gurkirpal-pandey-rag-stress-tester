import json
import sys
import os
sys.path.append('.')

from src.chunking.fixed_size import FixedSizeChunker

pages = []
with open('data/processed/monetary_policy_report_2024_apr.jsonl', encoding='utf-8') as f:
    for line in f:
        pages.append(json.loads(line))

chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
chunks = chunker.chunk(pages)

print('Sample chunk IDs:')
for c in chunks[:10]:
    print(f'  {c.id} | page {c.metadata["page"]}')