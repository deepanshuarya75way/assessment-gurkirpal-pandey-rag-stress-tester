# src/retrieval/hybrid_retriever.py
# HYBRID RETRIEVER - Combines BM25 + Dense

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, EmbeddedChunk, RetrievalResult, Retriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever(Retriever):

    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(collection_name="hybrid_chunks")
        self.chunks = []
        print(f"Hybrid retriever ready (alpha={alpha})")

    def index(self, embedded_chunks: list) -> None:
        print("Building hybrid index...")
        self.chunks = [ec.chunk for ec in embedded_chunks]
        self.bm25.index(embedded_chunks)
        self.dense.index(embedded_chunks)
        print("Hybrid index ready!")

    def normalize_scores(self, scores: list) -> list:
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def retrieve(self, query: str, top_k: int = 5,
                 query_embedding: list = None) -> RetrievalResult:

        if query_embedding is None:
            raise ValueError("query_embedding required!")

        fetch_k = min(top_k * 3, len(self.chunks))

        bm25_result = self.bm25.retrieve(query, top_k=fetch_k)
        dense_result = self.dense.retrieve(
            query, top_k=fetch_k, query_embedding=query_embedding)

        bm25_norm = self.normalize_scores(bm25_result.scores)
        dense_norm = self.normalize_scores(dense_result.scores)

        chunk_scores = {}

        for chunk, score in zip(bm25_result.retrieved_chunks, bm25_norm):
            chunk_scores[chunk.id] = {
                "chunk": chunk,
                "bm25": score,
                "dense": 0.0
            }

        for chunk, score in zip(dense_result.retrieved_chunks, dense_norm):
            if chunk.id in chunk_scores:
                chunk_scores[chunk.id]["dense"] = score
            else:
                chunk_scores[chunk.id] = {
                    "chunk": chunk,
                    "bm25": 0.0,
                    "dense": score
                }

        for chunk_id in chunk_scores:
            d = chunk_scores[chunk_id]["dense"]
            b = chunk_scores[chunk_id]["bm25"]
            chunk_scores[chunk_id]["hybrid"] = (
                self.alpha * d + (1 - self.alpha) * b)

        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x["hybrid"],
            reverse=True)[:top_k]

        retrieved_chunks = [c["chunk"] for c in sorted_chunks]
        scores = [c["hybrid"] for c in sorted_chunks]

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            scores=scores,
            retriever_name=f"hybrid_alpha{self.alpha}")


if __name__ == "__main__":
    import json
    from src.chunking.fixed_size import FixedSizeChunker
    from src.embeddings.bge_embedder import BGEEmbedder

    test_file = "data/processed/monetary_policy_report_2024_apr.jsonl"
    print(f"Loading: {test_file}")
    pages = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))

    chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
    chunks = chunker.chunk(pages)

    embedder = BGEEmbedder()
    embedded = embedder.embed_chunks(chunks[:50])

    retriever = HybridRetriever(alpha=0.6)
    retriever.index(embedded)

    test_queries = [
        "What is the repo rate in 2024?",
        "inflation target RBI",
        "monetary policy stance",
    ]

    print("\n-- Hybrid Search Results --")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_vector = embedder.embed_query(query)
        result = retriever.retrieve(
            query, top_k=3, query_embedding=query_vector)

        for i, (chunk, score) in enumerate(
                zip(result.retrieved_chunks, result.scores)):
            print(f"  Result {i+1} (hybrid score: {score:.4f}):")
            print(f"    Page: {chunk.metadata['page']}")
            print(f"    Text: {chunk.text[:80]}...")

    print("\n-- Retriever Comparison --")
    print("BM25:   keyword only, fast, exact terms")
    print("Dense:  meaning based, finds synonyms")
    print("Hybrid: best of both worlds!")