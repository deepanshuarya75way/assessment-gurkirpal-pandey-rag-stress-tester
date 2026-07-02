# src/interfaces.py
# Shared data structures and abstract base classes
# ALL THREE MODULES depend on this file
# DO NOT change anything here without team agreement

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ─────────────────────────────────────────
# SHARED DATA STRUCTURES
# ─────────────────────────────────────────

@dataclass
class Chunk:
    """Represents one piece of a document after splitting"""
    id: str                  # unique identifier e.g. "doc1_chunk_001"
    text: str                # the actual text content
    source_doc: str          # which PDF this came from
    metadata: dict           # must include: page, doc_type, year
    parent_id: str = None    # only used in hierarchical chunking


@dataclass
class EmbeddedChunk:
    """A chunk that has been converted to a vector"""
    chunk: Chunk             # the original chunk
    embedding: list          # list of floats (the vector)
    model_name: str          # which model created this embedding


@dataclass
class RetrievalResult:
    """What comes back when we search for relevant chunks"""
    query: str               # the original question asked
    retrieved_chunks: list   # list of Chunk objects found
    scores: list             # relevance score for each chunk
    retriever_name: str      # which retriever was used


@dataclass
class EvalSample:
    """One question-answer pair from our gold eval set"""
    question: str                      # the question
    ground_truth_answer: str           # the correct answer
    ground_truth_chunk_ids: list       # which chunks should be retrieved
    source_doc: str                    # which document has the answer
    source_pages: list                 # which pages have the answer


# ─────────────────────────────────────────
# ABSTRACT BASE CLASSES
# ─────────────────────────────────────────

class Chunker(ABC):
    """Every chunking strategy must implement this"""
    @abstractmethod
    def chunk(self, pages: list) -> list:
        """
        Input:  pages = list of dicts with keys:
                text, page, source, doc_type, year
        Output: list of Chunk objects
        """
        pass


class Embedder(ABC):
    """Every embedding model must implement this"""
    @abstractmethod
    def embed_chunks(self, chunks: list) -> list:
        """Convert a list of Chunks into EmbeddedChunks"""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list:
        """Convert a question string into a vector"""
        pass


class Retriever(ABC):
    """Every retrieval strategy must implement this"""
    @abstractmethod
    def index(self, embedded_chunks: list) -> None:
        """Build the search index from embedded chunks"""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """Search the index and return top_k results"""
        pass