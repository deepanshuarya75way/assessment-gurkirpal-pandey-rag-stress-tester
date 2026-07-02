import json
import os
import sys
sys.path.append('.')
from src.chunking.fixed_size import FixedSizeChunker

relevant_docs = [
    'monetary_policy_report_2022_apr.jsonl',
    'monetary_policy_report_2022_oct.jsonl',
    'monetary_policy_report_2023_apr.jsonl',
    'monetary_policy_report_2023_oct.jsonl',
    'monetary_policy_report_2024_apr.jsonl',
    'economic_survey_2022_23.jsonl',
    'economic_survey_2023_24.jsonl',
    'financial_stability_report_2022_dec.jsonl',
    'financial_stability_report_2023_dec.jsonl',
    'financial_stability_report_2024_jun.jsonl',
    'financial_stability_report_2024_dec.jsonl',
]

chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
total = 0

for filename in relevant_docs:
    filepath = os.path.join('data/processed', filename)
    pages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    chunks = chunker.chunk(pages)
    total += len(chunks)
    print(f"{filename}: {len(chunks)} chunks")

print(f"\nTotal chunks: {total}")
print(f"Gemini calls needed: {total}")
print(f"Daily limit: 1500")
print(f"Days needed: {total//1500 + 1}")