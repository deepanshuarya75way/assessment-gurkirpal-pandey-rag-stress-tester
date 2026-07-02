# src/embeddings/gemini_embedder.py
# GEMINI EMBEDDER
# Uses Google's gemini-embedding-001 model
# to convert text chunks into vectors
#
# This replaces OpenAI embeddings for our project
# Gemini embedding-001 is top ranked on MTEB benchmark
# and works great for finance/policy documents!

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, EmbeddedChunk, Embedder
from dotenv import load_dotenv
import time

load_dotenv()


class GeminiEmbedder(Embedder):
    """
    Converts text into vectors using Google's Gemini embedding model.

    Parameters:
        model_name: which Gemini embedding model to use
    """

    def __init__(self, model_name="gemini-embedding-001"):
        self.model_name = model_name
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """
        Sets up the Gemini client with API key
        """
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            print("WARNING: GEMINI_API_KEY not found in .env file")
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            print(f"Gemini embedder ready: {self.model_name}")
        except Exception as e:
            print(f"Error setting up Gemini client: {e}")

    def embed_chunks(self, chunks: list) -> list:
        """
        Converts a list of Chunks into EmbeddedChunks
        Sends text to Gemini API and gets vectors back
        """
        if not self.client:
            raise ValueError("Gemini client not set up. Check your API key!")

        embedded_chunks = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"Embedding chunk {i+1}/{total}...")

            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=chunk.text
                )

                vector = result.embeddings[0].values

                embedded_chunk = EmbeddedChunk(
                    chunk=chunk,
                    embedding=vector,
                    model_name=self.model_name
                )
                embedded_chunks.append(embedded_chunk)

                # small delay to avoid rate limiting
                time.sleep(0.1)

            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                time.sleep(2)
                continue

        print(f"Done! Embedded {len(embedded_chunks)} chunks")
        return embedded_chunks

    def embed_query(self, query: str) -> list:
        """
        Converts a question into a vector
        Used at search time
        """
        if not self.client:
            raise ValueError("Gemini client not set up!")

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=query
        )

        return result.embeddings[0].values


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Gemini Embedder...")
    print()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("❌ No GEMINI_API_KEY found in .env file!")
    else:
        print("✅ API key found!")
        embedder = GeminiEmbedder()

        if embedder.client:
            # test with one sample text
            test_text = "The repo rate was kept unchanged at 6.5 percent"
            print(f"Testing with: '{test_text}'")

            vector = embedder.embed_query(test_text)

            print(f"✅ Embedding works!")
            print(f"Vector dimensions: {len(vector)}")
            print(f"First 5 values: {[round(v, 4) for v in vector[:5]]}")

            # test similarity between two texts
            print("\nTesting similarity...")
            text1 = "repo rate monetary policy"
            text2 = "policy rate interest rate"
            text3 = "rainfall monsoon agriculture"

            v1 = embedder.embed_query(text1)
            v2 = embedder.embed_query(text2)
            v3 = embedder.embed_query(text3)

            # cosine similarity function
            def cosine_sim(a, b):
                dot = sum(x*y for x, y in zip(a, b))
                mag_a = sum(x**2 for x in a) ** 0.5
                mag_b = sum(x**2 for x in b) ** 0.5
                return dot / (mag_a * mag_b)

            sim_12 = cosine_sim(v1, v2)
            sim_13 = cosine_sim(v1, v3)

            print(f"'{text1}' vs '{text2}': {sim_12:.4f}")
            print(f"'{text1}' vs '{text3}': {sim_13:.4f}")
            print()
            if sim_12 > sim_13:
                print("✅ Embeddings understand meaning!")
                print("'repo rate' is closer to 'policy rate'")
                print("than to 'rainfall monsoon' — correct!")
            else:
                print("⚠️ Unexpected similarity scores")