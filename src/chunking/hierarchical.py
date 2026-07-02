# src/chunking/hierarchical.py
# HIERARCHICAL CHUNKING
# Creates TWO levels of chunks:
# Parent chunks (512 words) - for context
# Child chunks (128 words) - for precise retrieval
#
# How it works:
# 1. Split text into large PARENT chunks (512 words)
# 2. Split each parent into small CHILD chunks (128 words)
# 3. Child chunks linked to parent via parent_id
# 4. Retriever finds CHILD chunk (precise match)
# 5. But returns PARENT chunk (full context)
#
# Best of both worlds:
# Small chunks = precise retrieval
# Large chunks = complete context for LLM

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))))

from src.interfaces import Chunk, Chunker


class HierarchicalChunker(Chunker):
    """
    Creates parent and child chunks.
    Child chunks are indexed for retrieval.
    Parent chunks are returned as context.

    Parameters:
        parent_size: words per parent chunk
        child_size: words per child chunk
        child_overlap: overlap between child chunks
    """

    def __init__(self, parent_size=512,
                 child_size=128, child_overlap=25):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def make_chunks(self, words: list,
                   size: int, overlap: int,
                   prefix: str, page_data: dict,
                   parent_id: str = None) -> list:
        """
        Generic chunk maker for both parent and child
        """
        chunks = []
        counter = 0
        start = 0

        source_name = os.path.basename(
            page_data["source"]).replace(".pdf", "")

        while start < len(words):
            end = start + size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if len(chunk_words) < 5:
                break

            chunk_id = (f"{source_name}"
                       f"_p{page_data['page']}"
                       f"_{prefix}{counter}")

            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                source_doc=page_data["source"],
                metadata={
                    "page":      page_data["page"],
                    "doc_type":  page_data["doc_type"],
                    "year":      page_data["year"],
                    "strategy":  "hierarchical",
                    "level":     "parent" if parent_id is None
                                 else "child",
                    "parent_id": parent_id
                },
                parent_id=parent_id
            )
            chunks.append(chunk)
            counter += 1

            step = size - overlap
            start += step
            if start >= len(words):
                break

        return chunks

    def chunk(self, pages: list) -> list:
        """
        Creates both parent and child chunks.
        Returns ALL chunks (parent + child).
        Child chunks have parent_id set.
        """
        all_chunks = []

        for page_data in pages:
            text = page_data["text"]
            words = text.split()

            if len(words) < 10:
                continue

            # Step 1: Create parent chunks (512 words)
            parents = self.make_chunks(
                words=words,
                size=self.parent_size,
                overlap=self.child_overlap,
                prefix="par",
                page_data=page_data,
                parent_id=None
            )

            # Step 2: Create child chunks (128 words)
            # Each child linked to its parent
            for parent in parents:
                parent_words = parent.text.split()
                children = self.make_chunks(
                    words=parent_words,
                    size=self.child_size,
                    overlap=self.child_overlap,
                    prefix="chi",
                    page_data=page_data,
                    parent_id=parent.id
                )
                all_chunks.extend(children)

            # Also add parents to the collection
            all_chunks.extend(parents)

        return all_chunks

    def get_children_only(self, chunks: list) -> list:
        """Returns only child chunks for indexing"""
        return [c for c in chunks
                if c.metadata.get("level") == "child"]

    def get_parent(self, child: Chunk,
                   all_chunks: list) -> Chunk:
        """Given a child chunk, returns its parent"""
        for chunk in all_chunks:
            if chunk.id == child.parent_id:
                return chunk
        return child  # fallback to child if parent not found


if __name__ == "__main__":
    import json

    test_file = "data/processed/monetary_policy_report_2024_apr.jsonl"
    pages = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))

    chunker = HierarchicalChunker(
        parent_size=512,
        child_size=128,
        child_overlap=25
    )
    chunks = chunker.chunk(pages)

    parents = [c for c in chunks
               if c.metadata['level'] == 'parent']
    children = [c for c in chunks
                if c.metadata['level'] == 'child']

    print(f"Total chunks: {len(chunks)}")
    print(f"Parent chunks: {len(parents)}")
    print(f"Child chunks: {len(children)}")

    print(f"\n-- Sample Parent Chunk --")
    if parents:
        p = parents[0]
        print(f"ID: {p.id}")
        print(f"Words: {len(p.text.split())}")
        print(f"Text: {p.text[:100]}...")

    print(f"\n-- Sample Child Chunk --")
    if children:
        c = children[0]
        print(f"ID: {c.id}")
        print(f"Parent ID: {c.parent_id}")
        print(f"Words: {len(c.text.split())}")
        print(f"Text: {c.text[:100]}...")

    print(f"\n-- Comparison --")
    print(f"Hierarchical children: {len(children)}")
    print(f"Fixed size (256):      260 chunks")
    print(f"Recursive:             279 chunks")