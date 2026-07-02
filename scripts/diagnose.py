# scripts/diagnose.py
# Full project diagnostic - checks everything

import sys
import os
import json
sys.path.append('.')

print("="*60)
print("FULL PROJECT DIAGNOSTIC")
print("="*60)

# ── CHECK 1: Folder Structure ──
print("\n1. FOLDER STRUCTURE")
required = [
    'src/interfaces.py',
    'src/chunking/fixed_size.py',
    'src/chunking/sentence_based.py',
    'src/chunking/paragraph_based.py',
    'src/embeddings/bge_embedder.py',
    'src/embeddings/gemini_embedder.py',
    'src/retrieval/bm25_retriever.py',
    'src/retrieval/dense_retriever.py',
    'src/retrieval/hybrid_retriever.py',
    'scripts/prepare_data.py',
    'scripts/run_eval.py',
    'config/baseline.yaml',
    'data/eval/gold_eval_set.json',
    '.env'
]

for path in required:
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "OK" if (exists and size > 0) else "MISSING/EMPTY"
    print(f"  [{status}] {path} ({size} bytes)")

# ── CHECK 2: Processed Files ──
print("\n2. PROCESSED FILES")
processed_dir = 'data/processed'
jsonl_files = sorted([f for f in os.listdir(processed_dir)
                      if f.endswith('.jsonl')])
print(f"  Found {len(jsonl_files)} JSONL files")
for f in jsonl_files:
    path = os.path.join(processed_dir, f)
    size = os.path.getsize(path)
    print(f"  {f} ({size//1024} KB)")

# ── CHECK 3: Chunk IDs ──
print("\n3. CHUNK ID FORMAT CHECK")
print("  Loading first document and showing chunk IDs...")
from src.chunking.fixed_size import FixedSizeChunker

first_file = os.path.join(processed_dir, jsonl_files[0])
pages = []
with open(first_file, 'r', encoding='utf-8') as f:
    for line in f:
        pages.append(json.loads(line))

chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
chunks = chunker.chunk(pages)
print(f"  First 5 chunk IDs from {jsonl_files[0]}:")
for c in chunks[:5]:
    print(f"    {c.id}")

# ── CHECK 4: Eval Set ──
print("\n4. GOLD EVAL SET")
with open('data/eval/gold_eval_set.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

questions = eval_data['eval_set']
print(f"  Total questions: {len(questions)}")

has_ids = 0
empty_ids = 0
unanswerable = 0

for q in questions:
    if q['question_type'] == 'unanswerable':
        unanswerable += 1
    elif q.get('ground_truth_chunk_ids'):
        has_ids += 1
    else:
        empty_ids += 1

print(f"  With chunk IDs:   {has_ids}")
print(f"  Empty chunk IDs:  {empty_ids}")
print(f"  Unanswerable:     {unanswerable}")

# ── CHECK 5: Sample Chunk ID Match ──
print("\n5. CHUNK ID MATCH CHECK")
print("  Checking if eval chunk IDs match generated chunk IDs...")

all_chunk_ids = set()
for filename in sorted(os.listdir(processed_dir)):
    if not filename.endswith('.jsonl'):
        continue
    filepath = os.path.join(processed_dir, filename)
    pages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))
    chunks = chunker.chunk(pages)
    for c in chunks:
        all_chunk_ids.add(c.id)

print(f"  Total chunk IDs generated: {len(all_chunk_ids)}")

matched = 0
unmatched = 0
for q in questions:
    if q['question_type'] == 'unanswerable':
        continue
    for cid in q.get('ground_truth_chunk_ids', []):
        if cid in all_chunk_ids:
            matched += 1
        else:
            unmatched += 1

print(f"  Eval IDs that MATCH generated IDs:   {matched}")
print(f"  Eval IDs that DO NOT match:          {unmatched}")

if unmatched == 0:
    print("  PERFECT - All eval IDs match!")
else:
    print("  MISMATCH - Eval IDs don't match generated IDs!")
    print("  Need to rerun update_eval_ids.py")

# ── CHECK 6: ENV Keys ──
print("\n6. API KEYS")
from dotenv import load_dotenv
load_dotenv()
gemini = os.environ.get("GEMINI_API_KEY", "")
hf = os.environ.get("HUGGINGFACE_TOKEN", "")
print(f"  GEMINI_API_KEY:     {'SET' if gemini else 'MISSING'}")
print(f"  HUGGINGFACE_TOKEN:  {'SET' if hf else 'MISSING'}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)