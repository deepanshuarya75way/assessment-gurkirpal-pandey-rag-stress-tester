# src/chunking/sentence_based.py
# SENTENCE BASED CHUNKER
# Splits text at sentence boundaries instead of fixed word counts
#
# Why this is better than fixed size:
# Fixed size might cut mid-sentence like:
# "The repo rate was kept at 6.5% because infla-" ← CUT!
# "-tion remained above target"
#
# Sentence based always cuts at natural boundaries:
# "The repo rate was kept at 6.5% because inflation remained above target."
# ← clean cut here ✅

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, Chunker
import re


class SentenceChunker(Chunker):
    """
    Splits text into chunks at sentence boundaries.
    Accumulates sentences until chunk reaches target size,
    then starts a new chunk with some sentence overlap.

    Parameters:
        chunk_size:    target number of words per chunk
        chunk_overlap: number of sentences to repeat between chunks
    """

    def __init__(self, chunk_size=256, chunk_overlap=2):
        self.chunk_size = chunk_size
        # overlap here is number of SENTENCES not words
        self.chunk_overlap = chunk_overlap

    def split_into_sentences(self, text: str) -> list:
        """
        Splits text into sentences using simple rules.
        Splits at . ! ? followed by space and capital letter.
        """
        # split at sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # clean up each sentence
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return sentences

    def chunk(self, pages: list) -> list:
        """
        Takes a list of pages and returns a list of Chunks
        """
        all_chunks = []
        chunk_counter = 0

        for page_data in pages:
            text = page_data["text"]

            # split page into sentences
            sentences = self.split_into_sentences(text)

            # skip if too few sentences
            if len(sentences) < 2:
                continue

            # accumulate sentences into chunks
            current_sentences = []
            current_word_count = 0
            sentence_index = 0

            while sentence_index < len(sentences):
                sentence = sentences[sentence_index]
                word_count = len(sentence.split())

                # add sentence to current chunk
                current_sentences.append(sentence)
                current_word_count += word_count

                # if chunk is big enough, save it
                if current_word_count >= self.chunk_size:

                    # join sentences into chunk text
                    chunk_text = " ".join(current_sentences)

                    # create chunk id
                    source_name = os.path.basename(page_data["source"])
                    source_name = source_name.replace(".pdf", "")
                    chunk_id = f"{source_name}_p{page_data['page']}_sc{chunk_counter}"

                    # create Chunk object
                    chunk = Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        source_doc=page_data["source"],
                        metadata={
                            "page":       page_data["page"],
                            "doc_type":   page_data["doc_type"],
                            "year":       page_data["year"],
                            "strategy":   "sentence_based",
                            "n_sentences": len(current_sentences)
                        }
                    )

                    all_chunks.append(chunk)
                    chunk_counter += 1

                    # keep last N sentences as overlap for next chunk
                    current_sentences = current_sentences[-self.chunk_overlap:]
                    current_word_count = sum(
                        len(s.split()) for s in current_sentences
                    )

                sentence_index += 1

            # save remaining sentences as final chunk
            if current_sentences and current_word_count > 20:
                chunk_text = " ".join(current_sentences)
                source_name = os.path.basename(page_data["source"])
                source_name = source_name.replace(".pdf", "")
                chunk_id = f"{source_name}_p{page_data['page']}_sc{chunk_counter}"

                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source_doc=page_data["source"],
                    metadata={
                        "page":       page_data["page"],
                        "doc_type":   page_data["doc_type"],
                        "year":       page_data["year"],
                        "strategy":   "sentence_based",
                        "n_sentences": len(current_sentences)
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

    chunker = SentenceChunker(chunk_size=256, chunk_overlap=2)
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
        print(f"  Sentences: {chunk.metadata['n_sentences']}")
        print(f"  Text preview: {chunk.text[:100]}...")

    # compare with fixed size
    print("\n── Comparison ──")
    print(f"Sentence chunker: {len(chunks)} chunks")
    print("Fixed size chunker: 260 chunks")
    print("Sentence chunker respects natural language boundaries!")