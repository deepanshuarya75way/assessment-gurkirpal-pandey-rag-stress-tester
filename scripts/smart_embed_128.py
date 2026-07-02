# scripts/smart_embed_128.py
# Embeds relevant pages with chunk_size=128

import json
import os
import sys
import pickle
import time
sys.path.append('.')

from src.chunking.fixed_size import FixedSizeChunker
from src.interfaces import EmbeddedChunk

RELEVANT_PAGES = {
    'monetary_policy_report_2022_apr': list(range(1, 30)),
    'monetary_policy_report_2022_oct': list(range(1, 30)),
    'monetary_policy_report_2023_apr': list(range(1, 35)),
    'monetary_policy_report_2023_oct': list(range(1, 30)),
    'monetary_policy_report_2024_apr': list(range(1, 30)),
    'economic_survey_2022_23': list(range(40, 80)) + list(range(1, 20)),
    'economic_survey_2023_24': list(range(15, 30)) + list(range(50, 75)),
    'financial_stability_report_2022_dec': list(range(5, 20)),
    'financial_stability_report_2023_dec': list(range(10, 25)),
    'financial_stability_report_2024_jun': list(range(25, 40)),
    'financial_stability_report_2024_dec': list(range(18, 30)),
}

# use chunk_size=128
chunker = FixedSizeChunker(chunk_size=128, chunk_overlap=25)
all_chunks = []

print("Loading relevant pages with chunk_size=128...")
for doc_name, relevant_pages in RELEVANT_PAGES.items():
    filepath = f'data/processed/{doc_name}.jsonl'
    if not os.path.exists(filepath):
        continue

    pages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            page = json.loads(line)
            if page['page'] in relevant_pages:
                pages.append(page)

    chunks = chunker.chunk(pages)
    all_chunks.extend(chunks)
    print(f"  {doc_name}: {len(pages)} pages -> {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")
print(f"Gemini calls needed: {len(all_chunks)}")

if len(all_chunks) > 1000:
    print("WARNING: exceeds daily quota!")
    print("Will embed first 1000 only")
    all_chunks = all_chunks[:1000]

print("\nStarting Gemini embedding...")
from src.embeddings.gemini_embedder import GeminiEmbedder
embedder = GeminiEmbedder()

embedded = []
for i, chunk in enumerate(all_chunks):
    if i % 50 == 0:
        print(f"  Embedding {i+1}/{len(all_chunks)}...")
    try:
        vector = embedder.embed_query(chunk.text)
        embedded.append(EmbeddedChunk(
            chunk=chunk,
            embedding=vector,
            model_name="gemini-embedding-001"
        ))
        time.sleep(0.1)
    except Exception as e:
        print(f"  Error at chunk {i}: {e}")
        time.sleep(3)

cache_file = 'data/processed/cache_smart_gemini_128.pkl'
with open(cache_file, 'wb') as f:
    pickle.dump(embedded, f)

print(f"\nDone! Saved {len(embedded)} embeddings to {cache_file}")