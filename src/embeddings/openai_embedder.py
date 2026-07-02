# src/embeddings/openai_embedder.py
# OPENAI EMBEDDER
# Uses OpenAI's text-embedding-3-small model
# to convert text chunks into vectors
#
# This is our BASELINE embedder
# It runs on OpenAI's servers (costs money but very fast)
# Later we'll compare with free local models

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, EmbeddedChunk, Embedder
from dotenv import load_dotenv

# load API key from .env file
load_dotenv()


class OpenAIEmbedder(Embedder):
    """
    Converts text into vectors using OpenAI's embedding model.

    Parameters:
        model_name: which OpenAI embedding model to use
    """

    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """
        Sets up the OpenAI client with API key
        """
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            print("WARNING: OPENAI_API_KEY not found in .env file")
            print("Add your key to .env before using this embedder")
            return

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print(f"OpenAI embedder ready: {self.model_name}")
        except Exception as e:
            print(f"Error setting up OpenAI client: {e}")

    def embed_chunks(self, chunks: list) -> list:
        """
        Converts a list of Chunks into EmbeddedChunks
        Sends text to OpenAI API and gets vectors back
        """
        if not self.client:
            raise ValueError("OpenAI client not set up. Check your API key!")

        embedded_chunks = []

        # process in batches of 100 to be efficient
        batch_size = 100
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch]

            print(f"Embedding batch {i//batch_size + 1}"
                  f"/{(total-1)//batch_size + 1}"
                  f" ({len(batch)} chunks)...")

            try:
                # send to OpenAI API
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts
                )

                # create EmbeddedChunk for each result
                for j, chunk in enumerate(batch):
                    vector = response.data[j].embedding

                    embedded_chunk = EmbeddedChunk(
                        chunk=chunk,
                        embedding=vector,
                        model_name=self.model_name
                    )
                    embedded_chunks.append(embedded_chunk)

            except Exception as e:
                print(f"Error embedding batch: {e}")
                raise

        print(f"Done! Embedded {len(embedded_chunks)} chunks")
        return embedded_chunks

    def embed_query(self, query: str) -> list:
        """
        Converts a question string into a vector
        Used at search time to find similar chunks
        """
        if not self.client:
            raise ValueError("OpenAI client not set up!")

        response = self.client.embeddings.create(
            model=self.model_name,
            input=[query]
        )

        return response.data[0].embedding


# ─────────────────────────────────────────
# TEST — checks setup without using API
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing OpenAI Embedder setup...")
    print()

    # check if API key exists
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key or api_key == "sk-your-key-here":
        print("❌ No API key found!")
        print("Add your OpenAI key to .env file:")
        print('OPENAI_API_KEY=sk-...')
        print()
        print("Once you have the key we can test embedding!")
    else:
        print("✅ API key found!")
        print("Testing with one sample text...")

        embedder = OpenAIEmbedder()

        if embedder.client:
            # test with one simple text
            test_text = "The repo rate was kept unchanged at 6.5 percent"
            vector = embedder.embed_query(test_text)

            print(f"✅ Embedding works!")
            print(f"Text: '{test_text}'")
            print(f"Vector dimensions: {len(vector)}")
            print(f"First 5 values: {vector[:5]}")
        else:
            print("❌ Client setup failed")