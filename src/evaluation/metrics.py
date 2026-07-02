# src/evaluation/metrics.py
# ADDITIONAL EVALUATION METRICS
# Context Precision, Exact Match, ROUGE-L, LLM-as-judge

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))))

import re


def context_precision(retrieved_ids: list,
                      ground_truth_ids: list) -> float:
    """
    Of the retrieved chunks, what fraction are
    actually relevant (in ground truth)?

    Different from Recall:
    Recall asks "did we find the right chunks?"
    Precision asks "are the chunks we found correct?"
    """
    if not retrieved_ids:
        return 0.0

    correct = set(ground_truth_ids)
    retrieved = set(retrieved_ids)

    relevant_found = len(retrieved & correct)
    return relevant_found / len(retrieved_ids)


def exact_match(predicted_answer: str,
                ground_truth_answer: str) -> float:
    """
    Does the predicted answer EXACTLY match ground truth?
    (after normalizing case and whitespace)

    Very strict metric - rarely 1.0 in practice
    """
    pred = predicted_answer.strip().lower()
    truth = ground_truth_answer.strip().lower()

    # remove punctuation for fairer comparison
    pred = re.sub(r'[^\w\s]', '', pred)
    truth = re.sub(r'[^\w\s]', '', truth)

    return 1.0 if pred == truth else 0.0


def rouge_l(predicted_answer: str,
           ground_truth_answer: str) -> float:
    """
    ROUGE-L: Longest Common Subsequence based metric
    Measures overlap between predicted and ground truth
    Good for measuring partial correctness

    Returns F1 score based on LCS
    """
    pred_tokens = predicted_answer.lower().split()
    truth_tokens = ground_truth_answer.lower().split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    # find longest common subsequence
    m, n = len(pred_tokens), len(truth_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == truth_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(truth_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def llm_judge(question: str, ground_truth: str,
              system_answer: str, client) -> dict:
    """
    Uses Gemini to judge if system answer is correct.
    Returns score (1-5) and reasoning.

    This is the most sophisticated metric -
    uses LLM's own judgment to evaluate quality.
    """
    prompt = f"""You are evaluating whether a system's answer correctly answers a question about Indian economic policy.

Question: {question}
Correct Answer: {ground_truth}
System Answer: {system_answer}

Rate the system answer:
5 - Correct and complete. Matches the ground truth in all key facts.
4 - Mostly correct. Minor imprecision or omission but not misleading.
3 - Partially correct. Gets the main point but misses important details.
2 - Mostly wrong. A relevant fact or two but largely incorrect.
1 - Wrong or hallucinated. Contradicts or ignores the ground truth.

Respond only with JSON: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.strip()
        # clean markdown code blocks if present
        text = text.replace("```json", "").replace("```", "")

        import json
        result = json.loads(text)
        return result
    except Exception as e:
        print(f"LLM judge failed: {e}")
        return {"score": 0, "reason": "Error during judging"}


if __name__ == "__main__":
    print("Testing Evaluation Metrics...")

    # test context precision
    retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    ground_truth = ["chunk2", "chunk4"]
    cp = context_precision(retrieved, ground_truth)
    print(f"\nContext Precision test:")
    print(f"Retrieved: {retrieved}")
    print(f"Ground truth: {ground_truth}")
    print(f"Context Precision: {cp:.4f}")
    print(f"(2 of 5 retrieved are relevant = 0.4)")

    # test exact match
    pred = "The repo rate is 6.5 percent"
    truth = "The repo rate is 6.5 percent"
    em = exact_match(pred, truth)
    print(f"\nExact Match test (identical):")
    print(f"Score: {em}")

    pred2 = "Repo rate stands at 6.5%"
    em2 = exact_match(pred2, truth)
    print(f"\nExact Match test (different wording):")
    print(f"Score: {em2}")

    # test ROUGE-L
    rl = rouge_l(pred2, truth)
    print(f"\nROUGE-L test (different wording):")
    print(f"Pred: '{pred2}'")
    print(f"Truth: '{truth}'")
    print(f"ROUGE-L F1: {rl:.4f}")

    # test LLM judge
    print(f"\nTesting LLM-as-judge...")
    from google import genai
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    question = "What was the repo rate in April 2024?"
    ground_truth_ans = "The RBI kept repo rate unchanged at 6.5%"
    system_ans = "The repo rate was maintained at 6.5 percent"

    judge_result = llm_judge(
        question, ground_truth_ans, system_ans, client)
    print(f"Judge result: {judge_result}")