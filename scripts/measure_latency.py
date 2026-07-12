# scripts/measure_latency.py
# LATENCY MEASUREMENT
# Measures time taken for each pipeline stage:
# 1. Query embedding latency
# 2. Retrieval latency
# 3. Total per-query latency
# Brief requires "latency per query" in benchmark results

import sys
import os
import json
import time
import pickle
sys.path.append('.')

from dotenv import load_dotenv
load_dotenv()


def measure_latency():
    print("="*55)
    print("LATENCY MEASUREMENT")
    print("Config: Fixed256 + Gemini + Hybrid (best config)")
    print("="*55)

    # load embeddings
    print("\nLoading cached embeddings...")
    with open("data/processed/cache_smart_gemini.pkl", "rb") as f:
        embedded = pickle.load(f)
    print(f"Loaded {len(embedded)} embeddings")

    # setup embedder
    from src.embeddings.gemini_embedder import GeminiEmbedder
    embedder = GeminiEmbedder()

    # setup retriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(alpha=0.6)
    retriever.index(embedded)

    # load eval questions
    with open("data/eval/gold_eval_set.json", "r") as f:
        eval_data = json.load(f)

    questions = [
        q["question"] for q in eval_data["eval_set"]
        if q["question_type"] != "unanswerable"
    ][:10]  # test on 10 questions

    print(f"\nMeasuring latency on {len(questions)} questions...")
    print("-"*55)

    embed_times = []
    retrieve_times = []
    total_times = []

    for i, question in enumerate(questions):
        print(f"\nQ{i+1}: {question[:50]}...")

        # measure total start
        total_start = time.time()

        # Step 1: measure embedding latency
        embed_start = time.time()
        query_vector = embedder.embed_query(question)
        embed_end = time.time()
        embed_ms = (embed_end - embed_start) * 1000

        # Step 2: measure retrieval latency
        retrieve_start = time.time()
        result = retriever.retrieve(
            question, top_k=5,
            query_embedding=query_vector)
        retrieve_end = time.time()
        retrieve_ms = (retrieve_end - retrieve_start) * 1000

        # total latency
        total_ms = (time.time() - total_start) * 1000

        embed_times.append(embed_ms)
        retrieve_times.append(retrieve_ms)
        total_times.append(total_ms)

        print(f"  Embed:    {embed_ms:.1f}ms")
        print(f"  Retrieve: {retrieve_ms:.1f}ms")
        print(f"  Total:    {total_ms:.1f}ms")

    # summary
    print(f"\n{'='*55}")
    print(f"LATENCY SUMMARY (avg over {len(questions)} queries)")
    print(f"{'='*55}")
    print(f"Query Embedding:  {sum(embed_times)/len(embed_times):.1f}ms avg "
          f"(min={min(embed_times):.1f}, max={max(embed_times):.1f})")
    print(f"Retrieval:        {sum(retrieve_times)/len(retrieve_times):.1f}ms avg "
          f"(min={min(retrieve_times):.1f}, max={max(retrieve_times):.1f})")
    print(f"Total per query:  {sum(total_times)/len(total_times):.1f}ms avg "
          f"(min={min(total_times):.1f}, max={max(total_times):.1f})")

    print(f"\nBreakdown:")
    embed_pct = sum(embed_times)/sum(total_times)*100
    retrieve_pct = sum(retrieve_times)/sum(total_times)*100
    print(f"  Embedding:  {embed_pct:.0f}% of total latency")
    print(f"  Retrieval:  {retrieve_pct:.0f}% of total latency")

    # save results
    latency_results = {
        "config": "Fixed256 + Gemini + Hybrid",
        "n_questions": len(questions),
        "avg_embed_ms": round(sum(embed_times)/len(embed_times), 1),
        "avg_retrieve_ms": round(sum(retrieve_times)/len(retrieve_times), 1),
        "avg_total_ms": round(sum(total_times)/len(total_times), 1),
        "min_total_ms": round(min(total_times), 1),
        "max_total_ms": round(max(total_times), 1),
        "per_query": [
            {
                "question": questions[i][:50],
                "embed_ms": round(embed_times[i], 1),
                "retrieve_ms": round(retrieve_times[i], 1),
                "total_ms": round(total_times[i], 1)
            }
            for i in range(len(questions))
        ]
    }

    os.makedirs("results", exist_ok=True)
    with open("results/latency_results.json", "w") as f:
        json.dump(latency_results, f, indent=2)

    print(f"\nSaved: results/latency_results.json")
    return latency_results


if __name__ == "__main__":
    measure_latency()