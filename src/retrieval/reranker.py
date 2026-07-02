# src/retrieval/reranker.py
# CROSS-ENCODER RERANKER
#
# Initial retrieval (BM25/Dense/Hybrid) finds top_k=20
# candidates quickly but roughly.
#
# Reranker then looks at EACH candidate MORE CAREFULLY
# by reading question+chunk TOGETHER (cross-encoder)
# This is slower but much more accurate.
#
# Two-stage retrieval:
# Stage 1: Fast retrieval -> get 20 candidates
# Stage 2: Slow reranking -> pick best 5 from those 20

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))))

from src.interfaces import Chunk, RetrievalResult


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Cross-encoders read query+document TOGETHER
    (unlike bi-encoders which embed separately)
    This makes them more accurate but slower.

    Parameters:
        model_name: HuggingFace cross-encoder model
    """

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            print("Reranker ready!")
        except Exception as e:
            print(f"Error loading reranker: {e}")

    def rerank(self, query: str, result: RetrievalResult,
               top_k: int = 5) -> RetrievalResult:
        """
        Takes initial retrieval results and reranks them
        using cross-encoder scoring.

        Returns top_k best chunks after reranking.
        """
        if not self.model:
            raise ValueError("Reranker not loaded!")

        chunks = result.retrieved_chunks

        if not chunks:
            return result

        # create [query, chunk_text] pairs for cross-encoder
        pairs = [[query, chunk.text] for chunk in chunks]

        # get relevance scores from cross-encoder
        scores = self.model.predict(pairs)

        # sort chunks by new scores
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # take top_k after reranking
        top_chunks = scored_chunks[:top_k]

        new_chunks = [c for c, s in top_chunks]
        new_scores = [float(s) for c, s in top_chunks]

        return RetrievalResult(
            query=query,
            retrieved_chunks=new_chunks,
            scores=new_scores,
            retriever_name=result.retriever_name + "_reranked"
        )


if __name__ == "__main__":
    import json
    from src.chunking.fixed_size import FixedSizeChunker
    from src.embeddings.bge_embedder import BGEEmbedder
    from src.retrieval.dense_retriever import DenseRetriever

    print("Testing Cross-Encoder Reranker...")

    test_file = "data/processed/monetary_policy_report_2024_apr.jsonl"
    pages = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))

    chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
    chunks = chunker.chunk(pages)

    embedder = BGEEmbedder()
    embedded = embedder.embed_chunks(chunks[:50])

    retriever = DenseRetriever()
    retriever.index(embedded)

    query = "What was the repo rate decision?"
    print(f"\nQuery: '{query}'")

    # get initial 10 candidates
    query_vector = embedder.embed_query(query)
    initial_result = retriever.retrieve(
        query, top_k=10, query_embedding=query_vector)

    print(f"\n-- Before Reranking (top 5 of 10) --")
    for i, chunk in enumerate(initial_result.retrieved_chunks[:5]):
        print(f"{i+1}. {chunk.id}: {chunk.text[:60]}...")

    # rerank to get best 5
    reranker = CrossEncoderReranker()
    reranked_result = reranker.rerank(
        query, initial_result, top_k=5)

    print(f"\n-- After Reranking (top 5) --")
    for i, (chunk, score) in enumerate(zip(
            reranked_result.retrieved_chunks,
            reranked_result.scores)):
        print(f"{i+1}. (score={score:.3f}) {chunk.id}")
        print(f"   {chunk.text[:60]}...")