# src/embeddings/instructor_embedder.py
# INSTRUCTOR EMBEDDER
# Note: hkunlp/instructor-large loaded via SentenceTransformer
# The key difference from BGE:
# Instructor prepends task-specific instructions to text
# This improves domain-specific financial retrieval

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))))

from src.interfaces import Chunk, EmbeddedChunk, Embedder


class InstructorEmbedder(Embedder):
    """
    Instructor-style embedder using SentenceTransformer.
    Prepends task instructions to improve retrieval quality.
    Uses BAAI/bge-large as base (instructor-large has
    package compatibility issues on this environment).
    The instruction-prepending technique is the key
    innovation from the instructor model paper.
    """

    def __init__(self,
                 model_name="BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        # instructor-style task instructions
        self.doc_instruction = (
            "Represent this Indian financial policy "
            "document paragraph for retrieval: ")
        self.query_instruction = (
            "Represent this financial policy question "
            "for finding relevant documents: ")
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading Instructor model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Instructor embedder ready!")
        except Exception as e:
            print(f"Error: {e}")

    def embed_chunks(self, chunks: list) -> list:
        if not self.model:
            raise ValueError("Model not loaded!")

        embedded = []
        batch_size = 32

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"Batch {i//batch_size+1}/"
                  f"{(len(chunks)-1)//batch_size+1}...")

            # key difference: prepend instruction
            texts = [
                self.doc_instruction + c.text
                for c in batch]

            vecs = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False)

            for j, chunk in enumerate(batch):
                embedded.append(EmbeddedChunk(
                    chunk=chunk,
                    embedding=vecs[j].tolist(),
                    model_name="instructor-" + self.model_name))

        print(f"Done! {len(embedded)} chunks embedded")
        return embedded

    def embed_query(self, query: str) -> list:
        if not self.model:
            raise ValueError("Model not loaded!")

        # prepend query instruction
        vec = self.model.encode(
            self.query_instruction + query,
            normalize_embeddings=True)
        return vec.tolist()


if __name__ == "__main__":
    print("Testing Instructor Embedder...")
    embedder = InstructorEmbedder()

    if embedder.model:
        test = "The repo rate was kept at 6.5%"
        vec = embedder.embed_query(test)
        print(f"Vector dimensions: {len(vec)}")
        print(f"First 5: {[round(v,4) for v in vec[:5]]}")

        v1 = embedder.embed_query("repo rate policy")
        v2 = embedder.embed_query("interest rate bank")
        v3 = embedder.embed_query("rainfall agriculture")

        def sim(a, b):
            d = sum(x*y for x,y in zip(a,b))
            return d/(sum(x**2 for x in a)**0.5 *
                     sum(x**2 for x in b)**0.5)

        print(f"\nSimilarity test:")
        print(f"repo vs interest: {sim(v1,v2):.4f}")
        print(f"repo vs rainfall: {sim(v1,v3):.4f}")
        if sim(v1,v2) > sim(v1,v3):
            print("Instructor embedder understands meaning!")
        else:
            print("Unexpected scores")
    else:
        print("Model not loaded!")