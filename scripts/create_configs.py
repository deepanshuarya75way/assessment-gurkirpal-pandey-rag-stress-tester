# scripts/create_configs.py
# Creates all 20 experiment config files

import os
import yaml

configs = {
    # VARY CHUNKER with BM25
    "config/exp01_fixed256_bge_bm25.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "bge",
        "retriever": "bm25",
        "retriever_params": {"top_k": 5},
        "max_docs": 17
    },
    "config/exp02_fixed128_bge_bm25.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 128, "chunk_overlap": 25},
        "embedder": "bge",
        "retriever": "bm25",
        "retriever_params": {"top_k": 5},
        "max_docs": 17
    },
    "config/exp03_fixed512_bge_bm25.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 512, "chunk_overlap": 100},
        "embedder": "bge",
        "retriever": "bm25",
        "retriever_params": {"top_k": 5},
        "max_docs": 17
    },
    "config/exp04_sentence_bge_bm25.yaml": {
        "chunker": "sentence",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 2},
        "embedder": "bge",
        "retriever": "bm25",
        "retriever_params": {"top_k": 5},
        "max_docs": 17
    },
    "config/exp05_paragraph_bge_bm25.yaml": {
        "chunker": "paragraph",
        "chunker_params": {"chunk_size": 256, "min_paragraph_words": 20},
        "embedder": "bge",
        "retriever": "bm25",
        "retriever_params": {"top_k": 5},
        "max_docs": 17
    },
    # VARY RETRIEVER with Gemini
    "config/exp06_fixed256_gemini_dense.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "gemini",
        "retriever": "dense",
        "retriever_params": {"top_k": 5},
        "max_docs": 17
    },
    "config/exp07_fixed256_gemini_hybrid.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 5, "alpha": 0.6},
        "max_docs": 17
    },
    "config/exp08_fixed256_gemini_hyde.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "gemini",
        "retriever": "hyde",
        "retriever_params": {"top_k": 5},
        "max_docs": 17
    },
    # VARY EMBEDDER
    "config/exp09_fixed256_bge_dense.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "bge",
        "retriever": "dense",
        "retriever_params": {"top_k": 5},
        "max_docs": 17
    },
    "config/exp10_fixed256_instructor_bm25.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "instructor",
        "retriever": "bm25",
        "retriever_params": {"top_k": 5},
        "max_docs": 17
    },
    # VARY CHUNK SIZE with Hybrid
    "config/exp11_fixed128_gemini_hybrid.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 128, "chunk_overlap": 25},
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 5, "alpha": 0.6},
        "max_docs": 17
    },
    "config/exp12_fixed512_gemini_hybrid.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 512, "chunk_overlap": 100},
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 5, "alpha": 0.6},
        "max_docs": 17
    },
    "config/exp13_recursive_gemini_hybrid.yaml": {
        "chunker": "recursive",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 5, "alpha": 0.6},
        "max_docs": 17
    },
    # ALPHA VARIATION
    "config/exp14_fixed256_gemini_hybrid_a04.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 5, "alpha": 0.4},
        "max_docs": 17
    },
    "config/exp15_fixed256_gemini_hybrid_a08.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 5, "alpha": 0.8},
        "max_docs": 17
    },
    # RERANKER TESTS
    "config/exp16_fixed256_gemini_hybrid_rerank10.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 10, "alpha": 0.6},
        "reranker_depth": 10,
        "max_docs": 17
    },
    "config/exp17_fixed256_gemini_hybrid_rerank20.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 20, "alpha": 0.6},
        "reranker_depth": 20,
        "max_docs": 17
    },
    "config/exp18_fixed256_gemini_dense_rerank10.yaml": {
        "chunker": "fixed_size",
        "chunker_params": {"chunk_size": 256, "chunk_overlap": 50},
        "embedder": "gemini",
        "retriever": "dense",
        "retriever_params": {"top_k": 10},
        "reranker_depth": 10,
        "max_docs": 17
    },
    # ADVANCED CHUNKERS
    "config/exp19_semantic_gemini_hybrid.yaml": {
        "chunker": "semantic",
        "chunker_params": {"threshold": 0.85, "min_chunk_words": 50},
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 5, "alpha": 0.6},
        "max_docs": 17
    },
    "config/exp20_hierarchical_gemini_hybrid.yaml": {
        "chunker": "hierarchical",
        "chunker_params": {
            "parent_size": 512,
            "child_size": 128,
            "child_overlap": 25
        },
        "embedder": "gemini",
        "retriever": "hybrid",
        "retriever_params": {"top_k": 5, "alpha": 0.6},
        "max_docs": 17
    }
}

os.makedirs("config", exist_ok=True)
for path, cfg in configs.items():
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Created: {path}")

print(f"\nTotal: {len(configs)} configs created!")