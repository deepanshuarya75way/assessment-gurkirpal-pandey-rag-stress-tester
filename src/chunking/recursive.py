# src/chunking/recursive.py
# RECURSIVE CHARACTER SPLITTING
# LangChain-style chunking
# Tries to split at natural boundaries first

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, Chunker


class RecursiveChunker(Chunker):
    """
    Splits text recursively at natural boundaries.
    Priority order: paragraphs > lines > words > chars

    Parameters:
        chunk_size: target number of words per chunk
        chunk_overlap: words to repeat between chunks
    """

    def __init__(self, chunk_size=256, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # separators in priority order
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str, separators: list) -> list:
        """
        Recursively splits text using separators
        in priority order
        """
        # get current separator
        separator = separators[0]
        remaining = separators[1:]

        # split by current separator
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        # filter empty splits
        splits = [s for s in splits if s.strip()]

        # accumulate splits into chunks
        chunks = []
        current = []
        current_len = 0

        for split in splits:
            split_len = len(split.split())

            # if this split alone exceeds chunk size
            # and we have more separators to try
            if split_len > self.chunk_size and remaining:
                # first save what we have
                if current:
                    chunks.append(
                        separator.join(current))
                    current = []
                    current_len = 0
                # recursively split this piece
                sub_chunks = self.split_text(
                    split, remaining)
                chunks.extend(sub_chunks)

            elif current_len + split_len > self.chunk_size:
                # current chunk is full → save it
                if current:
                    chunks.append(
                        separator.join(current))

                # start new chunk with overlap
                overlap_words = []
                overlap_len = 0
                for prev in reversed(current):
                    prev_len = len(prev.split())
                    if overlap_len + prev_len <= self.chunk_overlap:
                        overlap_words.insert(0, prev)
                        overlap_len += prev_len
                    else:
                        break

                current = overlap_words + [split]
                current_len = overlap_len + split_len

            else:
                current.append(split)
                current_len += split_len

        # save remaining text
        if current:
            chunks.append(separator.join(current))

        return chunks

    def chunk(self, pages: list) -> list:
        """
        Takes list of pages and returns list of Chunks
        """
        all_chunks = []
        chunk_counter = 0

        for page_data in pages:
            text = page_data["text"]

            if len(text.split()) < 10:
                continue

            # recursively split the text
            text_chunks = self.split_text(
                text, self.separators)

            for chunk_text in text_chunks:
                if len(chunk_text.split()) < 5:
                    continue

                source_name = os.path.basename(
                    page_data["source"])
                source_name = source_name.replace(
                    ".pdf", "")
                chunk_id = (f"{source_name}"
                           f"_p{page_data['page']}"
                           f"_r{chunk_counter}")

                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source_doc=page_data["source"],
                    metadata={
                        "page":     page_data["page"],
                        "doc_type": page_data["doc_type"],
                        "year":     page_data["year"],
                        "strategy": "recursive"
                    }
                )
                all_chunks.append(chunk)
                chunk_counter += 1

        return all_chunks


if __name__ == "__main__":
    import json

    test_file = "data/processed/monetary_policy_report_2024_apr.jsonl"
    pages = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))

    chunker = RecursiveChunker(chunk_size=256, chunk_overlap=50)
    chunks = chunker.chunk(pages)

    print(f"Created {len(chunks)} chunks")
    avg = sum(len(c.text.split()) for c in chunks) // len(chunks)
    print(f"Average chunk size: {avg} words")
    print(f"\n-- First 3 chunks --")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Words: {len(chunk.text.split())}")
        print(f"  Text: {chunk.text[:100]}...")

    print(f"\n-- Comparison --")
    print(f"Recursive chunker: {len(chunks)} chunks")
    print(f"Fixed size:        260 chunks")