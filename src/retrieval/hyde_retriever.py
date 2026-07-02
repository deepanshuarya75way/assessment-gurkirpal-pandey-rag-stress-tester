# src/retrieval/hyde_retriever.py
# HyDE - Hypothetical Document Embeddings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))))

from src.interfaces import Chunk, EmbeddedChunk, RetrievalResult, Retriever
from src.retrieval.dense_retriever import DenseRetriever
from dotenv import load_dotenv

load_dotenv()


class HyDERetriever(Retriever):

    def __init__(self, embedder, collection_name="hyde_chunks"):
        self.embedder = embedder
        self.dense = DenseRetriever(collection_name=collection_name)
        self.llm_client = None
        self._setup_llm()
        print("HyDE retriever ready")

    def _setup_llm(self):
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            self.llm_client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"Error setting up LLM: {e}")

    def generate_hypothetical_answer(self, question):
        if not self.llm_client:
            return question

        prompt = (
            "You are answering a question about Indian "
            "monetary policy and economic reports. Write a "
            "short plausible answer (2-3 sentences) even if "
            "you are not certain of exact figures. This will "
            "be used for document search, not as a final "
            "answer.\n\nQuestion: " + question + "\n\nAnswer:"
        )

        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"HyDE generation failed: {e}, using raw question")
            return question

    def index(self, embedded_chunks):
        self.dense.index(embedded_chunks)

    def retrieve(self, query, top_k=5, query_embedding=None):
        hypothetical = self.generate_hypothetical_answer(query)
        hyde_vector = self.embedder.embed_query(hypothetical)
        result = self.dense.retrieve(
            query, top_k=top_k,
            query_embedding=hyde_vector)
        result.retriever_name = "hyde"
        return result


if __name__ == "__main__":
    import json
    from src.chunking.fixed_size import FixedSizeChunker
    from src.embeddings.gemini_embedder import GeminiEmbedder

    print("Testing HyDE Retriever...")

    test_file = "data/processed/monetary_policy_report_2024_apr.jsonl"
    pages = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))

    chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
    chunks = chunker.chunk(pages)
    print(f"Created {len(chunks)} chunks")

    embedder = GeminiEmbedder()

    test_chunks = chunks[:20]
    print(f"Embedding {len(test_chunks)} chunks...")
    embedded = embedder.embed_chunks(test_chunks)

    retriever = HyDERetriever(embedder)
    retriever.index(embedded)

    query = "What was the repo rate decision?"
    print(f"\nQuery: '{query}'")

    hypothetical = retriever.generate_hypothetical_answer(query)
    print(f"\nHypothetical answer generated:")
    print(f"'{hypothetical[:200]}...'")

    result = retriever.retrieve(query, top_k=3)
    print(f"\n-- HyDE Search Results --")
    for i, chunk in enumerate(result.retrieved_chunks):
        print(f"Result {i+1}: {chunk.id}")
        print(f"  {chunk.text[:80]}...")