# src/embeddings/bge_embedder.py
# BGE LARGE EMBEDDER — Free Local Model
#
# BGE = Beijing General Embedding
# Made by BAAI (Beijing Academy of AI)
# Runs 100% on your laptop — no API, no cost, no limits!
#
# We use this for:
# → All development and testing (saves Gemini quota)
# → Bulk embedding of full corpus
# → Comparing against Gemini embeddings in Week 3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, EmbeddedChunk, Embedder


class BGEEmbedder(Embedder):
    """
    Converts text into vectors using BGE-Large model.
    Runs locally — no API key needed!

    Parameters:
        model_name: which BGE model to use
    """

    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self._setup_model()

    def _setup_model(self):
        """
        Loads the BGE model from local cache
        Already downloaded so this is fast!
        """
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading BGE model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("BGE model ready!")
        except Exception as e:
            print(f"Error loading BGE model: {e}")

    def embed_chunks(self, chunks: list) -> list:
        """
        Converts list of Chunks into EmbeddedChunks
        Runs entirely on your laptop!
        """
        if not self.model:
            raise ValueError("BGE model not loaded!")

        embedded_chunks = []
        total = len(chunks)

        # process in batches for efficiency
        batch_size = 32
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch]

            print(f"Embedding batch {i//batch_size + 1}"
                  f"/{(total-1)//batch_size + 1}"
                  f" ({len(batch)} chunks)...")

            # BGE works best with this prefix for documents
            prefixed_texts = [
                f"Represent this sentence: {text}"
                for text in batch_texts
            ]

            # get embeddings
            vectors = self.model.encode(
                prefixed_texts,
                normalize_embeddings=True
            )

            for j, chunk in enumerate(batch):
                embedded_chunk = EmbeddedChunk(
                    chunk=chunk,
                    embedding=vectors[j].tolist(),
                    model_name=self.model_name
                )
                embedded_chunks.append(embedded_chunk)

        print(f"Done! Embedded {len(embedded_chunks)} chunks")
        return embedded_chunks

    def embed_query(self, query: str) -> list:
        """
        Converts a question into a vector
        BGE uses different prefix for queries vs documents
        """
        if not self.model:
            raise ValueError("BGE model not loaded!")

        # BGE uses this prefix for queries
        prefixed_query = f"Represent this question for searching: {query}"

        vector = self.model.encode(
            prefixed_query,
            normalize_embeddings=True
        )

        return vector.tolist()


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("Testing BGE Embedder...")
    print()

    # setup embedder
    embedder = BGEEmbedder()

    if embedder.model:
        # test with sample text
        test_text = "The repo rate was kept unchanged at 6.5 percent"
        print(f"Testing with: '{test_text}'")

        vector = embedder.embed_query(test_text)
        print(f"✅ Embedding works!")
        print(f"Vector dimensions: {len(vector)}")
        print(f"First 5 values: {[round(v, 4) for v in vector[:5]]}")

        # test similarity
        print("\nTesting similarity...")
        text1 = "repo rate monetary policy"
        text2 = "policy rate interest rate"
        text3 = "rainfall monsoon agriculture"

        v1 = embedder.embed_query(text1)
        v2 = embedder.embed_query(text2)
        v3 = embedder.embed_query(text3)

        def cosine_sim(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            mag_a = sum(x**2 for x in a) ** 0.5
            mag_b = sum(x**2 for x in b) ** 0.5
            return dot / (mag_a * mag_b)

        sim_12 = cosine_sim(v1, v2)
        sim_13 = cosine_sim(v1, v3)

        print(f"'{text1}' vs '{text2}': {sim_12:.4f}")
        print(f"'{text1}' vs '{text3}': {sim_13:.4f}")

        if sim_12 > sim_13:
            print("\n✅ BGE understands meaning!")
            print("Works great — and completely free!")
        else:
            print("\n⚠️ Unexpected similarity scores")

        # test on real chunks
        print("\n── Testing on real corpus chunks ──")
        test_file = "data/processed/monetary_policy_report_2024_apr.jsonl"
        pages = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                pages.append(json.loads(line))

        from src.chunking.fixed_size import FixedSizeChunker
        chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
        chunks = chunker.chunk(pages)

        # only embed first 10 chunks for testing
        print(f"Embedding first 10 chunks of {len(chunks)} total...")
        test_chunks = chunks[:10]
        embedded = embedder.embed_chunks(test_chunks)

        print(f"✅ Successfully embedded {len(embedded)} chunks!")
        print(f"Each chunk vector: {len(embedded[0].embedding)} dimensions")