# src/retrieval/bm25_retriever.py
# BM25 RETRIEVER — Keyword Based Search
#
# BM25 = Best Match 25 (a classic search algorithm)
# This is how Google search worked before AI!
#
# How it works:
# Query: "repo rate 2024"
# → looks for chunks containing these exact words
# → ranks by how often words appear
# → returns most matching chunks
#
# Pros: Fast, no AI needed, great for exact terms
# Cons: Misses meaning — "policy rate" won't match "repo rate"
#
# This is why we compare BM25 vs Dense (AI) retrieval!

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, EmbeddedChunk, RetrievalResult, Retriever
from rank_bm25 import BM25Okapi


class BM25Retriever(Retriever):
    """
    Keyword based retriever using BM25 algorithm.
    No embeddings needed — works on raw text.

    This is our keyword baseline to compare
    against dense (vector) retrieval.
    """

    def __init__(self):
        self.bm25 = None
        self.chunks = []      # store chunks for lookup

    def index(self, embedded_chunks: list) -> None:
        """
        Builds the BM25 index from chunks.
        Note: BM25 doesn't need embeddings
        so we just extract the chunks from EmbeddedChunks
        """
        print("Building BM25 index...")

        # extract chunks from embedded chunks
        # BM25 only needs the text, not vectors
        self.chunks = [ec.chunk for ec in embedded_chunks]

        # tokenize each chunk into words
        # BM25 works on lists of words
        tokenized_chunks = [
            chunk.text.lower().split()
            for chunk in self.chunks
        ]

        # build the BM25 index
        self.bm25 = BM25Okapi(tokenized_chunks)

        print(f"BM25 index built with {len(self.chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Searches for chunks matching the query using keywords.

        Returns top_k most relevant chunks.
        """
        if not self.bm25:
            raise ValueError("Index not built! Call index() first.")

        # tokenize query same way as chunks
        tokenized_query = query.lower().split()

        # get BM25 scores for all chunks
        scores = self.bm25.get_scores(tokenized_query)

        # get indices of top_k highest scores
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # get the actual chunks and scores
        retrieved_chunks = [self.chunks[i] for i in top_indices]
        retrieved_scores = [float(scores[i]) for i in top_indices]

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            scores=retrieved_scores,
            retriever_name="bm25"
        )


# ─────────────────────────────────────────
# TEST — works without any API key!
# ─────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys
    sys.path.append('.')

    # load some chunks using our fixed size chunker
    from src.chunking.fixed_size import FixedSizeChunker
    from src.interfaces import EmbeddedChunk

    # load processed pages
    test_file = "data/processed/monetary_policy_report_2024_apr.jsonl"
    print(f"Loading: {test_file}")

    pages = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            pages.append(json.loads(line))

    # chunk the pages
    chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
    chunks = chunker.chunk(pages)
    print(f"Created {len(chunks)} chunks")

    # wrap chunks as EmbeddedChunks
    # BM25 doesn't need real embeddings
    # so we use empty vectors
    embedded_chunks = [
        EmbeddedChunk(
            chunk=chunk,
            embedding=[],     # empty — BM25 doesn't use vectors
            model_name="none"
        )
        for chunk in chunks
    ]

    # build index
    retriever = BM25Retriever()
    retriever.index(embedded_chunks)

    # test queries
    test_queries = [
        "What is the repo rate in 2024?",
        "inflation target RBI",
        "monetary policy stance withdrawal accommodation",
    ]

    print("\n── Search Results ──")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = retriever.retrieve(query, top_k=3)

        for i, (chunk, score) in enumerate(
            zip(result.retrieved_chunks, result.scores)
        ):
            print(f"  Result {i+1} (score: {score:.2f}):")
            print(f"    Page: {chunk.metadata['page']}")
            print(f"    Text: {chunk.text[:80]}...")