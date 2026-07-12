# scripts/judge_variance.py
# LLM JUDGE VARIANCE ANALYSIS
# Brief requirement: Run judge TWICE on same questions
# Measure how much scores vary between runs
# This quantifies noise in our eval signal

import sys
import os
import json
import time
import pickle
sys.path.append('.')

from dotenv import load_dotenv
load_dotenv()


def run_judge(question, ground_truth,
              system_answer, client, run_num):
    """Run LLM judge and return score"""
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
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt)
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "")
        result = json.loads(text)
        print(f"  Run {run_num}: score={result['score']}/5 "
              f"— {result['reason'][:50]}...")
        return result
    except Exception as e:
        print(f"  Run {run_num}: failed — {e}")
        return {"score": -1, "reason": "error"}


def measure_variance():
    print("="*55)
    print("LLM JUDGE VARIANCE ANALYSIS")
    print("Running judge TWICE on same questions")
    print("to measure evaluation noise")
    print("="*55)

    # load end-to-end results
    results_file = "results/end_to_end_results.json"
    if not os.path.exists(results_file):
        print("Run generate_answers.py first!")
        sys.exit(1)

    with open(results_file) as f:
        data = json.load(f)

    # take first 5 questions with valid answers
    questions = [
        q for q in data["per_question"]
        if q["generated_answer"] != "Generation failed due to API error."
    ][:5]

    print(f"\nAnalyzing {len(questions)} questions...")
    print("-"*55)

    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    variance_results = []

    for i, q in enumerate(questions):
        print(f"\nQ{i+1}: {q['question'][:50]}...")
        print(f"  Answer: {q['generated_answer'][:60]}...")

        # Run 1
        time.sleep(12)  # rate limit: 5 calls/min
        r1 = run_judge(
            q["question"], q["ground_truth"],
            q["generated_answer"], client, 1)

        # Run 2
        time.sleep(12)  # rate limit
        r2 = run_judge(
            q["question"], q["ground_truth"],
            q["generated_answer"], client, 2)

        score1 = r1.get("score", -1)
        score2 = r2.get("score", -1)
        diff = abs(score1 - score2) \
            if score1 > 0 and score2 > 0 else -1
        agreement = "AGREE" if diff == 0 else \
                   "MINOR" if diff == 1 else "DISAGREE"

        print(f"  Difference: {diff} points → {agreement}")

        variance_results.append({
            "question": q["question"][:60],
            "answer": q["generated_answer"][:60],
            "run1_score": score1,
            "run2_score": score2,
            "difference": diff,
            "agreement": agreement
        })

    # summary
    print(f"\n{'='*55}")
    print(f"VARIANCE SUMMARY")
    print(f"{'='*55}")

    valid = [r for r in variance_results if r["difference"] >= 0]
    if valid:
        avg_diff = sum(r["difference"] for r in valid) / len(valid)
        agreements = sum(1 for r in valid if r["agreement"] == "AGREE")
        minor = sum(1 for r in valid if r["agreement"] == "MINOR")
        disagree = sum(1 for r in valid if r["agreement"] == "DISAGREE")

        print(f"Questions analyzed:    {len(valid)}")
        print(f"Perfect agreement:     {agreements}/{len(valid)}")
        print(f"Minor disagreement:    {minor}/{len(valid)} (1 point)")
        print(f"Major disagreement:    {disagree}/{len(valid)} (2+ points)")
        print(f"Average difference:    {avg_diff:.2f} points")
        print(f"\nNoise level: {'LOW' if avg_diff < 0.5 else 'MEDIUM' if avg_diff < 1.0 else 'HIGH'}")
        print(f"\nKEY FINDING:")
        print(f"LLM-as-judge has ~{avg_diff:.1f} point variance")
        print(f"Single-run scores have limited reliability")
        print(f"For production: average 2-3 judge runs")

    # save
    output = {
        "n_questions": len(valid),
        "avg_score_difference": round(avg_diff, 2) if valid else 0,
        "agreement_rate": round(agreements/len(valid)*100, 1) if valid else 0,
        "per_question": variance_results
    }
    with open("results/judge_variance.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: results/judge_variance.json")


if __name__ == "__main__":
    measure_variance()