# scripts/generate_answers.py
# END-TO-END ANSWER GENERATION + EVALUATION
# Full pipeline: Question -> Retrieve -> Generate -> Score

import sys
import os
import json
import pickle
import time
sys.path.append('.')

from dotenv import load_dotenv
load_dotenv()


def load_cached_embeddings():
    """Load pre-computed Gemini embeddings"""
    cache = "data/processed/cache_smart_gemini.pkl"
    if not os.path.exists(cache):
        print("Cache not found! Run smart_embed.py first")
        sys.exit(1)
    with open(cache, 'rb') as f:
        embedded = pickle.load(f)
    print(f"Loaded {len(embedded)} cached embeddings")
    return embedded


def build_retriever(embedded):
    """Build hybrid retriever from cached embeddings"""
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(alpha=0.6)
    retriever.index(embedded)
    return retriever


def generate_answer(question, chunks, client):
    """
    Sends question + retrieved chunks to Gemini
    and gets a generated answer back.
    """
    # format chunks as context
    context = "\n\n".join([
        f"[Document {i+1}]: {chunk.text}"
        for i, chunk in enumerate(chunks)
    ])

    prompt = f"""You are answering questions about Indian monetary policy and economic documents.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "The answer is not available in the provided documents."
Keep your answer concise (2-3 sentences maximum).

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Generation failed: {e}")
        time.sleep(3)
        return "Generation failed due to API error."


def score_with_llm_judge(question, ground_truth,
                          system_answer, client):
    """Score answer quality using Gemini as judge"""
    prompt = f"""You are evaluating whether a system answer correctly answers a question about Indian economic policy.

Question: {question}
Correct Answer: {ground_truth}
System Answer: {system_answer}

Rate the system answer:
5 - Correct and complete
4 - Mostly correct, minor imprecision
3 - Partially correct
2 - Mostly wrong
1 - Wrong or hallucinated

Respond only with JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        print(f"Judge failed: {e}")
        return {"score": 0, "reason": "Error"}


def rouge_l(predicted, ground_truth):
    """ROUGE-L score"""
    pred = predicted.lower().split()
    truth = ground_truth.lower().split()
    if not pred or not truth:
        return 0.0
    m, n = len(pred), len(truth)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if pred[i-1] == truth[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / len(pred)
    r = lcs / len(truth)
    return 2*p*r/(p+r)


def exact_match(predicted, ground_truth):
    """Exact match score"""
    import re
    pred = re.sub(r'[^\w\s]', '', predicted.lower().strip())
    truth = re.sub(r'[^\w\s]', '', ground_truth.lower().strip())
    return 1.0 if pred == truth else 0.0


def run_end_to_end():
    print("="*55)
    print("END-TO-END RAG PIPELINE EVALUATION")
    print("="*55)
    print("Config: Fixed256 + Gemini + Hybrid")
    print()

    # setup Gemini client
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # setup embedder
    from src.embeddings.gemini_embedder import GeminiEmbedder
    embedder = GeminiEmbedder()

    # load embeddings and build retriever
    embedded = load_cached_embeddings()
    retriever = build_retriever(embedded)

    # load eval set
    with open('data/eval/gold_eval_set.json',
              'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    questions = [q for q in eval_data['eval_set']
                if q['question_type'] != 'unanswerable'
                and q.get('ground_truth_chunk_ids')]

    print(f"Evaluating {len(questions)} questions...")
    print()

    results = []
    judge_scores = []
    rouge_scores = []
    em_scores = []

    for i, sample in enumerate(questions):
        question = sample['question']
        ground_truth = sample['ground_truth_answer']

        print(f"Q{i+1}/{len(questions)}: {question[:50]}...")

        # Step 1: Retrieve
        query_vector = embedder.embed_query(question)
        result = retriever.retrieve(
            question, top_k=5,
            query_embedding=query_vector)

        # Step 2: Generate answer
        answer = generate_answer(
            question,
            result.retrieved_chunks,
            client)

        print(f"  Generated: {answer[:80]}...")

        # Step 3: Score
        # ROUGE-L
        rl = rouge_l(answer, ground_truth)

        # Exact Match
        em = exact_match(answer, ground_truth)

        # LLM Judge (use quota wisely - only first 10)
        if i < 10:
            judge = score_with_llm_judge(
                question, ground_truth, answer, client)
            judge_score = judge.get('score', 0)
            judge_reason = judge.get('reason', '')
            time.sleep(1)  # rate limit
        else:
            judge_score = -1
            judge_reason = "skipped to save quota"

        print(f"  ROUGE-L: {rl:.3f} | "
              f"Judge: {judge_score}/5")

        judge_scores.append(judge_score)
        rouge_scores.append(rl)
        em_scores.append(em)

        results.append({
            "question": question,
            "question_type": sample['question_type'],
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "rouge_l": rl,
            "exact_match": em,
            "judge_score": judge_score,
            "judge_reason": judge_reason
        })

        time.sleep(0.5)

    # calculate averages
    valid_judges = [s for s in judge_scores if s > 0]
    avg_rouge = sum(rouge_scores)/len(rouge_scores)
    avg_em = sum(em_scores)/len(em_scores)
    avg_judge = sum(valid_judges)/len(valid_judges) \
        if valid_judges else 0

    print(f"\n{'='*55}")
    print(f"END-TO-END RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"Questions evaluated: {len(results)}")
    print(f"Average ROUGE-L:     {avg_rouge:.4f}")
    print(f"Average Exact Match: {avg_em:.4f}")
    print(f"Average Judge Score: {avg_judge:.2f}/5.0")
    print(f"  (based on first 10 questions)")

    print(f"\nJudge Score Distribution:")
    for score in range(1, 6):
        count = sum(1 for s in valid_judges if s == score)
        bar = "█" * count
        print(f"  {score}/5: {bar} ({count})")

    # save results
    output = {
        "config": "Fixed256 + Gemini + Hybrid",
        "pipeline": "retrieve + generate + score",
        "questions_evaluated": len(results),
        "summary_metrics": {
            "avg_rouge_l": avg_rouge,
            "avg_exact_match": avg_em,
            "avg_judge_score": avg_judge
        },
        "per_question": results
    }

    os.makedirs("results", exist_ok=True)
    with open("results/end_to_end_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: results/end_to_end_results.json")
    return output


if __name__ == "__main__":
    run_end_to_end()