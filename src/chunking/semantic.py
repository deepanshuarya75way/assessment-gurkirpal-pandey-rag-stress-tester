# src/chunking/semantic.py
# SEMANTIC CHUNKING
# Splits text where topic CHANGES
# Uses cosine similarity between adjacent sentences
# If similarity drops below threshold → new chunk!

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, Chunker
import re


class SemanticChunker(Chunker):
    """
    Splits text at topic boundaries using embeddings.
    Compares adjacent sentences - when similarity drops
    below threshold, starts a new chunk.

    Parameters:
        threshold: cosine similarity threshold (0.0-1.0)
                   lower = more splits
                   higher = fewer splits
        min_chunk_words: minimum words per chunk
    """

    def __init__(self, threshold=0.85, min_chunk_words=50):
        self.threshold = threshold
        self.min_chunk_words = min_chunk_words
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load BGE model for sentence embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading BGE model for semantic chunking...")
            self.model = SentenceTransformer(
                'BAAI/bge-large-en-v1.5')
            print("Model ready!")
        except Exception as e:
            print(f"Error loading model: {e}")

    def cosine_similarity(self, v1: list, v2: list) -> float:
        """Calculate cosine similarity between two vectors"""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def split_into_sentences(self, text: str) -> list:
        """Split text into sentences"""
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences
                    if len(s.strip()) > 20]
        return sentences

    def chunk(self, pages: list) -> list:
        """
        Takes list of pages and returns list of Chunks
        Splits where cosine similarity drops below threshold
        """
        if not self.model:
            raise ValueError(
                "Model not loaded! Check BGE installation.")

        all_chunks = []
        chunk_counter = 0

        for page_data in pages:
            text = page_data["text"]

            # split into sentences
            sentences = self.split_into_sentences(text)

            if len(sentences) < 2:
                continue

            # embed all sentences at once (efficient)
            embeddings = self.model.encode(
                sentences,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()

            # group sentences into chunks
            current_sentences = [sentences[0]]
            current_embedding = embeddings[0]

            for i in range(1, len(sentences)):
                # compare current sentence to previous
                sim = self.cosine_similarity(
                    embeddings[i-1], embeddings[i])

                # if similarity drops → topic changed → new chunk
                if sim < self.threshold:
                    # save current chunk if big enough
                    chunk_text = " ".join(current_sentences)
                    if len(chunk_text.split()) >= self.min_chunk_words:
                        source_name = os.path.basename(
                            page_data["source"]).replace(".pdf", "")
                        chunk_id = (f"{source_name}"
                                   f"_p{page_data['page']}"
                                   f"_sem{chunk_counter}")

                        chunk = Chunk(
                            id=chunk_id,
                            text=chunk_text,
                            source_doc=page_data["source"],
                            metadata={
                                "page": page_data["page"],
                                "doc_type": page_data["doc_type"],
                                "year": page_data["year"],
                                "strategy": "semantic",
                                "threshold": self.threshold,
                                "n_sentences": len(current_sentences)
                            }
                        )
                        all_chunks.append(chunk)
                        chunk_counter += 1

                    # start new chunk
                    current_sentences = [sentences[i]]
                    current_embedding = embeddings[i]
                else:
                    current_sentences.append(sentences[i])

            # save remaining sentences
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                if len(chunk_text.split()) >= self.min_chunk_words:
                    source_name = os.path.basename(
                        page_data["source"]).replace(".pdf", "")
                    chunk_id = (f"{source_name}"
                               f"_p{page_data['page']}"
                               f"_sem{chunk_counter}")

                    chunk = Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        source_doc=page_data["source"],
                        metadata={
                            "page": page_data["page"],
                            "doc_type": page_data["doc_type"],
                            "year": page_data["year"],
                            "strategy": "semantic",
                            "threshold": self.threshold,
                            "n_sentences": len(current_sentences)
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

    # test with threshold 0.85 (default)
    chunker = SemanticChunker(threshold=0.85)
    chunks = chunker.chunk(pages)

    print(f"\nThreshold 0.85: {len(chunks)} chunks")
    if chunks:
        avg = sum(len(c.text.split())
                 for c in chunks) // len(chunks)
        print(f"Average size: {avg} words")

    print(f"\n-- First 3 chunks --")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Words: {len(chunk.text.split())}")
        print(f"  Sentences: {chunk.metadata['n_sentences']}")
        print(f"  Text: {chunk.text[:100]}...")

    print(f"\n-- Comparison --")
    print(f"Semantic (0.85): {len(chunks)} chunks")
    print(f"Fixed size:      260 chunks")
    print(f"Sentence:        234 chunks")