# src/chunking/fixed_size.py
# FIXED SIZE CHUNKER

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, Chunker


class FixedSizeChunker(Chunker):

    def __init__(self, chunk_size=256, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pages: list) -> list:
        all_chunks = []

        for page_data in pages:
            text = page_data["text"]
            words = text.split()

            if len(words) < 10:
                continue

            start = 0
            page_chunk_counter = 0

            while start < len(words):
                end = start + self.chunk_size
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)

                source_name = os.path.basename(page_data["source"])
                source_name = source_name.replace(".pdf", "")

                chunk_id = f"{source_name}_p{page_data['page']}_c{page_chunk_counter}"

                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source_doc=page_data["source"],
                    metadata={
                        "page":     page_data["page"],
                        "doc_type": page_data["doc_type"],
                        "year":     page_data["year"],
                        "chunk_size": self.chunk_size,
                        "strategy": "fixed_size"
                    }
                )

                all_chunks.append(chunk)
                page_chunk_counter += 1

                step = self.chunk_size - self.chunk_overlap
                start += step

                if start >= len(words):
                    break

        return all_chunks


if __name__ == "__main__":
    import json

    test_file = "data/processed/monetary_policy_report_2024_apr.jsonl"
    print(f"Loading: {test_file}")
    pages = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))

    print(f"Loaded {len(pages)} pages")

    chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
    chunks = chunker.chunk(pages)

    print(f"Created {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(c.text.split()) for c in chunks) // len(chunks)} words")

    print("\n-- First 5 chunks --")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Words: {len(chunk.text.split())}")
        print(f"  Page: {chunk.metadata['page']}")
        print(f"  Text preview: {chunk.text[:100]}...")