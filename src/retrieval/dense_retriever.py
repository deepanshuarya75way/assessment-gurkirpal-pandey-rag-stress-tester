# src/retrieval/dense_retriever.py
# DENSE RETRIEVER - AI Vector Search

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.interfaces import Chunk, EmbeddedChunk, RetrievalResult, Retriever
import chromadb


class DenseRetriever(Retriever):

    def __init__(self, collection_name="rag_chunks"):
        self.collection_name = collection_name
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            get_or_create=True
        )
        print(f"Dense retriever ready: {collection_name}")

    def index(self, embedded_chunks: list) -> None:
        print(f"Indexing {len(embedded_chunks)} chunks...")

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for ec in embedded_chunks:
            ids.append(ec.chunk.id)
            embeddings.append(ec.embedding)
            documents.append(ec.chunk.text)
            metadatas.append({
                "page":     str(ec.chunk.metadata["page"]),
                "doc_type": ec.chunk.metadata["doc_type"],
                "year":     str(ec.chunk.metadata["year"]),
                "source":   ec.chunk.source_doc
            })

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        print(f"Indexed {len(embedded_chunks)} chunks in ChromaDB!")

    def retrieve(self, query: str, top_k: int = 5,
                 query_embedding: list = None) -> RetrievalResult:

        if query_embedding is None:
            raise ValueError("query_embedding required!")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_chunks = []
        scores = []

        for i in range(len(results["ids"][0])):
            chunk = Chunk(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                source_doc=results["metadatas"][0][i]["source"],
                metadata={
                    "page":     results["metadatas"][0][i]["page"],
                    "doc_type": results["metadatas"][0][i]["doc_type"],
                    "year":     results["metadatas"][0][i]["year"],
                }
            )
            retrieved_chunks.append(chunk)
            distance = results["distances"][0][i]
            similarity = 1 - distance
            scores.append(similarity)

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            scores=scores,
            retriever_name="dense_bge"
        )


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
    print(f"Created {len(chunks)} chunks")

    print("Embedding first 50 chunks with BGE...")
    embedder = BGEEmbedder()
    embedded = embedder.embed_chunks(chunks[:50])

    retriever = DenseRetriever()
    retriever.index(embedded)

    test_queries = [
        "What is the repo rate in 2024?",
        "inflation target RBI",
        "monetary policy stance",
    ]

    print("\n-- Dense Search Results --")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_vector = embedder.embed_query(query)
        result = retriever.retrieve(
            query,
            top_k=3,
            query_embedding=query_vector
        )

        for i, (chunk, score) in enumerate(
            zip(result.retrieved_chunks, result.scores)
        ):
            print(f"  Result {i+1} (similarity: {score:.4f}):")
            print(f"    Page: {chunk.metadata['page']}")
            print(f"    Text: {chunk.text[:80]}...")