# scripts/run_eval.py
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import yaml
import pickle
import argparse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_pages(processed_dir, max_docs=17):
    all_pages = []
    files = sorted([f for f in os.listdir(processed_dir)
                   if f.endswith('.jsonl')])[:max_docs]
    for filename in files:
        filepath = os.path.join(processed_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_pages.append(json.loads(line))
        print(f"Loaded: {filename}")
    print(f"Total pages: {len(all_pages)}")
    return all_pages, files


def load_eval_set(eval_path):
    with open(eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["eval_set"]


def get_chunker(config):
    name = config["chunker"]
    params = config.get("chunker_params", {})
    if name == "fixed_size":
        from src.chunking.fixed_size import FixedSizeChunker
        return FixedSizeChunker(**params)
    elif name == "sentence":
        from src.chunking.sentence_based import SentenceChunker
        return SentenceChunker(**params)
    elif name == "paragraph":
        from src.chunking.paragraph_based import ParagraphChunker
        return ParagraphChunker(**params)
    elif name == "recursive":
        from src.chunking.recursive import RecursiveChunker
        return RecursiveChunker(**params)
    elif name == "semantic":
        from src.chunking.semantic import SemanticChunker
        return SemanticChunker(**params)
    elif name == "hierarchical":
        from src.chunking.hierarchical import HierarchicalChunker
        return HierarchicalChunker(**params)
    else:
        raise ValueError(f"Unknown chunker: {name}")


def get_embedder(config):
    name = config["embedder"]
    if name == "bge":
        from src.embeddings.bge_embedder import BGEEmbedder
        return BGEEmbedder()
    elif name == "gemini":
        from src.embeddings.gemini_embedder import GeminiEmbedder
        return GeminiEmbedder()
    elif name == "instructor":
        from src.embeddings.instructor_embedder import InstructorEmbedder
        return InstructorEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {name}")


def get_retriever(config, embedder=None):
    name = config["retriever"]
    params = config.get("retriever_params", {})
    if name == "bm25":
        from src.retrieval.bm25_retriever import BM25Retriever
        return BM25Retriever()
    elif name == "dense":
        from src.retrieval.dense_retriever import DenseRetriever
        return DenseRetriever()
    elif name == "hybrid":
        from src.retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever(alpha=params.get("alpha", 0.6))
    elif name == "hyde":
        from src.retrieval.hyde_retriever import HyDERetriever
        return HyDERetriever(embedder=embedder)
    else:
        raise ValueError(f"Unknown retriever: {name}")


def extract_doc_name(chunk_id):
    return re.sub(r'_p\d+_[a-z]+\d+$', '', chunk_id)


def doc_recall_at_k(retrieved_ids, source_doc, k):
    correct_doc = source_doc.replace('.pdf', '')
    for chunk_id in retrieved_ids[:k]:
        if extract_doc_name(chunk_id) == correct_doc:
            return 1.0
    return 0.0


def doc_mrr(retrieved_ids, source_doc):
    correct_doc = source_doc.replace('.pdf', '')
    for i, chunk_id in enumerate(retrieved_ids):
        if extract_doc_name(chunk_id) == correct_doc:
            return 1.0 / (i + 1)
    return 0.0


def context_precision(retrieved_ids, ground_truth_ids):
    if not retrieved_ids:
        return 0.0
    correct = set(ground_truth_ids)
    hits = sum(1 for cid in retrieved_ids if cid in correct)
    return hits / len(retrieved_ids)


def get_embedded(config, chunks, embedder):
    retriever_name = config["retriever"]
    embedder_name = config["embedder"]
    chunk_size = config.get(
        "chunker_params", {}).get("chunk_size", 256)

    if retriever_name == "bm25":
        from src.interfaces import EmbeddedChunk
        embedded = [
            EmbeddedChunk(chunk=c, embedding=[],
                         model_name="none")
            for c in chunks]
        print("BM25 mode - skipping embeddings!")
        return embedded

    # pick right cache file
    if embedder_name == "gemini":
        if chunk_size == 128:
            cache = "data/processed/cache_smart_gemini_128.pkl"
        else:
            cache = "data/processed/cache_smart_gemini.pkl"
    else:
        cache = (f"data/processed/cache_"
                f"{config['chunker']}_{embedder_name}.pkl")

    if os.path.exists(cache):
        print(f"Loading cache: {cache}")
        with open(cache, 'rb') as f:
            embedded = pickle.load(f)
        print(f"Loaded {len(embedded)} embeddings!")
        return embedded

    print(f"Embedding {len(chunks)} chunks...")
    embedded = embedder.embed_chunks(chunks)
    with open(cache, 'wb') as f:
        pickle.dump(embedded, f)
    print("Cache saved!")
    return embedded


def run_pipeline(config_path):
    print(f"\n{'='*50}")
    print(f"RAG Pipeline Stress Tester")
    print(f"Config: {config_path}")
    print(f"{'='*50}\n")

    config = load_config(config_path)
    max_docs = config.get("max_docs", 17)
    reranker_depth = config.get("reranker_depth", None)

    print(f"Chunker:   {config['chunker']}")
    print(f"Embedder:  {config['embedder']}")
    print(f"Retriever: {config['retriever']}")
    if reranker_depth:
        print(f"Reranker:  depth={reranker_depth}")
    print()

    pages, loaded_files = load_pages(
        "data/processed", max_docs=max_docs)
    loaded_docs = set(
        f.replace('.jsonl', '.pdf') for f in loaded_files)

    chunker = get_chunker(config)
    chunks = chunker.chunk(pages)

    # for hierarchical, use children only for indexing
    if config["chunker"] == "hierarchical":
        index_chunks = [c for c in chunks
                       if c.metadata.get("level") == "child"]
        print(f"Created {len(chunks)} total chunks "
              f"({len(index_chunks)} children for indexing)")
        chunks = index_chunks
    else:
        print(f"Created {len(chunks)} chunks\n")

    embedder = get_embedder(config)
    embedded = get_embedded(config, chunks, embedder)

    retriever = get_retriever(config, embedder)
    retriever.index(embedded)

    # setup reranker if needed
    reranker = None
    if reranker_depth:
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    eval_samples = load_eval_set(
        "data/eval/gold_eval_set.json")
    relevant = [s for s in eval_samples
                if s['source_doc'] in loaded_docs
                and s['question_type'] != 'unanswerable']

    print(f"\nRunning eval on {len(relevant)} questions...\n")

    results = []
    r1_scores = []
    r3_scores = []
    r5_scores = []
    mrr_scores = []
    cp_scores = []

    top_k = config.get(
        "retriever_params", {}).get("top_k", 5)
    fetch_k = reranker_depth if reranker_depth else top_k

    for sample in relevant:
        query = sample["question"]
        source_doc = sample["source_doc"]
        gt_ids = sample.get("ground_truth_chunk_ids", [])

        query_vector = embedder.embed_query(query)

        if config["retriever"] in ["dense", "hybrid", "hyde"]:
            result = retriever.retrieve(
                query, top_k=fetch_k,
                query_embedding=query_vector)
        else:
            result = retriever.retrieve(query, top_k=fetch_k)

        # apply reranker if configured
        if reranker:
            result = reranker.rerank(query, result, top_k=top_k)

        retrieved_ids = [c.id for c in result.retrieved_chunks]

        r1 = doc_recall_at_k(retrieved_ids, source_doc, 1)
        r3 = doc_recall_at_k(retrieved_ids, source_doc, 3)
        r5 = doc_recall_at_k(retrieved_ids, source_doc, 5)
        mrr = doc_mrr(retrieved_ids, source_doc)
        cp = context_precision(retrieved_ids, gt_ids)

        r1_scores.append(r1)
        r3_scores.append(r3)
        r5_scores.append(r5)
        mrr_scores.append(mrr)
        cp_scores.append(cp)

        results.append({
            "question": query,
            "source_doc": source_doc,
            "question_type": sample["question_type"],
            "retrieved_docs": [
                extract_doc_name(cid)
                for cid in retrieved_ids[:3]],
            "correct_doc": source_doc.replace('.pdf', ''),
            "recall@1": r1,
            "recall@3": r3,
            "recall@5": r5,
            "mrr": mrr,
            "context_precision": cp
        })

        status = "HIT" if r3 > 0 else "MISS"
        print(f"[{status}] {query[:45]}...")

    avg_r1 = sum(r1_scores)/len(r1_scores) if r1_scores else 0
    avg_r3 = sum(r3_scores)/len(r3_scores) if r3_scores else 0
    avg_r5 = sum(r5_scores)/len(r5_scores) if r5_scores else 0
    avg_mrr = sum(mrr_scores)/len(mrr_scores) if mrr_scores else 0
    avg_cp = sum(cp_scores)/len(cp_scores) if cp_scores else 0

    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Config:    {config_path}")
    print(f"Chunker:   {config['chunker']}")
    print(f"Embedder:  {config['embedder']}")
    print(f"Retriever: {config['retriever']}")
    print(f"Questions: {len(relevant)}")
    print(f"Recall@1:  {avg_r1:.4f}")
    print(f"Recall@3:  {avg_r3:.4f}")
    print(f"Recall@5:  {avg_r5:.4f}")
    print(f"MRR:       {avg_mrr:.4f}")
    print(f"Ctx Prec:  {avg_cp:.4f}")

    output = {
        "config": config_path,
        "timestamp": datetime.now().isoformat(),
        "chunker": config["chunker"],
        "embedder": config["embedder"],
        "retriever": config["retriever"],
        "reranker_depth": reranker_depth,
        "questions_evaluated": len(relevant),
        "metrics": {
            "recall@1": avg_r1,
            "recall@3": avg_r3,
            "recall@5": avg_r5,
            "mrr": avg_mrr,
            "context_precision": avg_cp
        },
        "per_question_results": results
    }

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = (f"{config['chunker']}_"
            f"{config['embedder']}_"
            f"{config['retriever']}")
    if reranker_depth:
        name += f"_rerank{reranker_depth}"
    output_file = f"results/{name}_{ts}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {output_file}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/baseline.yaml",
        help="Path to config file")
    args = parser.parse_args()
    run_pipeline(args.config)