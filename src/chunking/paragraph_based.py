# src/chunking/paragraph_based.py
# PARAGRAPH BASED CHUNKER
# Splits text at paragraph boundaries
#
# Why paragraphs?
# A paragraph = one complete thought or idea
# Keeping paragraphs together means
# each chunk has one coherent topic
#
# Example:
# Paragraph 1: "About inflation..."     ← one chunk
# Paragraph 2: "About repo rate..."     ← another chunk
# Much better than cutting mid-thought!

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, Chunker


class ParagraphChunker(Chunker):
    """
    Splits text at paragraph boundaries.
    Combines short paragraphs together until
    chunk reaches target size.

    Parameters:
        chunk_size: target number of words per chunk
        min_paragraph_words: skip paragraphs shorter than this
    """

    def __init__(self, chunk_size=256, min_paragraph_words=20):
        self.chunk_size = chunk_size
        self.min_paragraph_words = min_paragraph_words

    def split_into_paragraphs(self, text: str) -> list:
        """
        Splits text into paragraphs.
        Paragraphs are separated by blank lines (\n\n)
        """
        # split on double newlines
        paragraphs = text.split('\n\n')

        # clean and filter
        paragraphs = [p.strip() for p in paragraphs]
        paragraphs = [p for p in paragraphs
                     if len(p.split()) >= self.min_paragraph_words]

        return paragraphs

    def chunk(self, pages: list) -> list:
        """
        Takes a list of pages and returns a list of Chunks
        """
        all_chunks = []
        chunk_counter = 0

        for page_data in pages:
            text = page_data["text"]

            # split into paragraphs
            paragraphs = self.split_into_paragraphs(text)

            if not paragraphs:
                continue

            # accumulate paragraphs into chunks
            current_paragraphs = []
            current_word_count = 0

            for paragraph in paragraphs:
                para_words = len(paragraph.split())

                # if adding this paragraph exceeds chunk size
                # and we already have some content → save current chunk
                if current_word_count + para_words > self.chunk_size \
                        and current_paragraphs:

                    chunk_text = "\n\n".join(current_paragraphs)
                    source_name = os.path.basename(page_data["source"])
                    source_name = source_name.replace(".pdf", "")
                    chunk_id = f"{source_name}_p{page_data['page']}_pc{chunk_counter}"

                    chunk = Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        source_doc=page_data["source"],
                        metadata={
                            "page":         page_data["page"],
                            "doc_type":     page_data["doc_type"],
                            "year":         page_data["year"],
                            "strategy":     "paragraph_based",
                            "n_paragraphs": len(current_paragraphs)
                        }
                    )

                    all_chunks.append(chunk)
                    chunk_counter += 1

                    # start fresh
                    current_paragraphs = []
                    current_word_count = 0

                # add paragraph to current chunk
                current_paragraphs.append(paragraph)
                current_word_count += para_words

            # save remaining paragraphs
            if current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                source_name = os.path.basename(page_data["source"])
                source_name = source_name.replace(".pdf", "")
                chunk_id = f"{source_name}_p{page_data['page']}_pc{chunk_counter}"

                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source_doc=page_data["source"],
                    metadata={
                        "page":         page_data["page"],
                        "doc_type":     page_data["doc_type"],
                        "year":         page_data["year"],
                        "strategy":     "paragraph_based",
                        "n_paragraphs": len(current_paragraphs)
                    }
                )
                all_chunks.append(chunk)
                chunk_counter += 1

        return all_chunks


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_file = "data/processed/monetary_policy_report_2024_apr.jsonl"

    print(f"Loading: {test_file}")
    pages = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))

    print(f"Loaded {len(pages)} pages")

    chunker = ParagraphChunker(chunk_size=256, min_paragraph_words=20)
    chunks = chunker.chunk(pages)

    print(f"Created {len(chunks)} chunks")
    if chunks:
        avg = sum(len(c.text.split()) for c in chunks) // len(chunks)
        print(f"Average chunk size: {avg} words")

    # show first 3 chunks
    print("\n── First 3 chunks ──")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Words: {len(chunk.text.split())}")
        print(f"  Paragraphs: {chunk.metadata['n_paragraphs']}")
        print(f"  Text preview: {chunk.text[:150]}...")

    # compare all three
    print("\n── Comparison So Far ──")
    print(f"Fixed Size Chunker:  260 chunks")
    print(f"Sentence Chunker:    234 chunks")
    print(f"Paragraph Chunker:   {len(chunks)} chunks")