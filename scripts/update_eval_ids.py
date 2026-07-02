# scripts/update_eval_ids.py
import json
import os
import sys
sys.path.append('.')

from src.chunking.fixed_size import FixedSizeChunker

chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)

# build complete lookup of ALL chunk IDs
print("Building chunk ID lookup from all documents...")
lookup = {}

processed_dir = 'data/processed'
for filename in sorted(os.listdir(processed_dir)):
    if not filename.endswith('.jsonl'):
        continue

    filepath = os.path.join(processed_dir, filename)
    pages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))

    chunks = chunker.chunk(pages)

    for chunk in chunks:
        source = os.path.basename(chunk.source_doc)
        page = chunk.metadata['page']
        key = (source, page)
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(chunk.id)

    print(f"  {filename}: {len(chunks)} chunks")

print(f"Total unique page keys: {len(lookup)}")

# load eval set
with open('data/eval/gold_eval_set.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

# update chunk IDs for each question
print("\nUpdating eval set...")
for sample in eval_data['eval_set']:
    if sample['question_type'] == 'unanswerable':
        sample['ground_truth_chunk_ids'] = []
        continue

    source_doc = sample['source_doc']
    source_pages = sample['source_pages']

    chunk_ids = []
    for page in source_pages:
        key = (source_doc, page)
        if key in lookup:
            chunk_ids.extend(lookup[key])

    sample['ground_truth_chunk_ids'] = chunk_ids
    print(f"  Q{sample['id']}: {len(chunk_ids)} chunk IDs found")
    if chunk_ids:
        print(f"    First ID: {chunk_ids[0]}")

# save updated eval set
with open('data/eval/gold_eval_set.json', 'w', encoding='utf-8') as f:
    json.dump(eval_data, f, indent=2, ensure_ascii=False)

print("\nDone! gold_eval_set.json updated!")

# verify
print("\nVerifying...")
with open('data/eval/gold_eval_set.json', 'r', encoding='utf-8') as f:
    verify = json.load(f)

for q in verify['eval_set'][:3]:
    if q['question_type'] == 'unanswerable':
        continue
    print(f"Q: {q['question'][:50]}")
    print(f"   IDs: {q['ground_truth_chunk_ids'][:2]}")