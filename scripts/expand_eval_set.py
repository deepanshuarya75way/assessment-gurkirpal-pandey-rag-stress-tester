# scripts/expand_eval_set.py
# Expands gold eval set from 25 to 50 questions
# Adds 25 new questions covering more documents
# and more question types

import json
import os

# load existing eval set
with open('data/eval/gold_eval_set.json', 'r') as f:
    data = json.load(f)

existing = data['eval_set']
print(f"Existing questions: {len(existing)}")

# 25 new questions
new_questions = [
    # FACTUAL — Monetary Policy
    {
        "id": "q026",
        "question": "What was the repo rate decision in October 2022?",
        "ground_truth_answer": "The MPC raised the repo rate by 50 basis points to 5.9 per cent in October 2022",
        "source_doc": "monetary_policy_report_2022_oct.pdf",
        "source_pages": [12, 13],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q027",
        "question": "What was India's CPI inflation in October 2022?",
        "ground_truth_answer": "CPI inflation was elevated at around 7 per cent in 2022 driven by food and fuel prices",
        "source_doc": "monetary_policy_report_2022_oct.pdf",
        "source_pages": [20, 21],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q028",
        "question": "What was the gross NPA ratio of banks in 2022?",
        "ground_truth_answer": "The gross NPA ratio of scheduled commercial banks declined to 5.0 per cent in 2022",
        "source_doc": "financial_stability_report_2022_dec.pdf",
        "source_pages": [15, 16],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q029",
        "question": "What was India's GDP growth rate in 2022-23?",
        "ground_truth_answer": "India's real GDP growth was 7.0 per cent in 2022-23",
        "source_doc": "economic_survey_2022_23.pdf",
        "source_pages": [10, 11],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q030",
        "question": "What was the fiscal deficit target for 2022-23?",
        "ground_truth_answer": "The fiscal deficit target for 2022-23 was 6.4 per cent of GDP",
        "source_doc": "economic_survey_2022_23.pdf",
        "source_pages": [85, 86],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    # FACTUAL — Financial Stability
    {
        "id": "q031",
        "question": "What was the capital adequacy ratio of banks in 2024?",
        "ground_truth_answer": "The capital to risk weighted assets ratio of scheduled commercial banks was above 16 per cent in 2024",
        "source_doc": "financial_stability_report_2024_jun.pdf",
        "source_pages": [20, 21],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q032",
        "question": "What was the net NPA ratio of banks in 2023?",
        "ground_truth_answer": "The net NPA ratio of scheduled commercial banks declined to around 1.3 per cent in 2023",
        "source_doc": "financial_stability_report_2023_dec.pdf",
        "source_pages": [18, 19],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q033",
        "question": "What was India's current account deficit in 2023-24?",
        "ground_truth_answer": "India's current account deficit narrowed to 0.7 per cent of GDP in 2023-24",
        "source_doc": "economic_survey_2023_24.pdf",
        "source_pages": [62, 63],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q034",
        "question": "What was the RBI's foreign exchange reserve in 2024?",
        "ground_truth_answer": "India's foreign exchange reserves were around USD 640-650 billion in 2024",
        "source_doc": "financial_stability_report_2024_jun.pdf",
        "source_pages": [28, 29],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q035",
        "question": "What was India's WPI inflation in 2022?",
        "ground_truth_answer": "WPI inflation remained elevated at double digits for most of 2022 before moderating",
        "source_doc": "monetary_policy_report_2022_oct.pdf",
        "source_pages": [25, 26],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    # DOMAIN JARGON
    {
        "id": "q036",
        "question": "What is the Marginal Standing Facility rate?",
        "ground_truth_answer": "The Marginal Standing Facility rate is 25 basis points above the repo rate allowing banks to borrow overnight",
        "source_doc": "monetary_policy_report_2024_apr.pdf",
        "source_pages": [89, 90],
        "question_type": "domain_jargon",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q037",
        "question": "What is the Open Market Operations policy of RBI?",
        "ground_truth_answer": "Open Market Operations are RBI purchases or sales of government securities to manage liquidity in the banking system",
        "source_doc": "monetary_policy_report_2023_apr.pdf",
        "source_pages": [45, 46],
        "question_type": "domain_jargon",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q038",
        "question": "What is the Prompt Corrective Action framework?",
        "ground_truth_answer": "Prompt Corrective Action is a framework to restore financial health of weak banks by imposing restrictions on dividends lending and management",
        "source_doc": "financial_stability_report_2022_dec.pdf",
        "source_pages": [55, 56],
        "question_type": "domain_jargon",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q039",
        "question": "What is the Base Rate in Indian banking?",
        "ground_truth_answer": "Base Rate is the minimum interest rate below which banks cannot lend, replaced by MCLR in 2016",
        "source_doc": "monetary_policy_report_2022_apr.pdf",
        "source_pages": [60, 61],
        "question_type": "domain_jargon",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q040",
        "question": "What is the role of the Monetary Policy Committee?",
        "ground_truth_answer": "The MPC is a six-member committee that decides the policy repo rate to achieve the inflation target of 4 per cent",
        "source_doc": "monetary_policy_report_2024_apr.pdf",
        "source_pages": [12, 13],
        "question_type": "domain_jargon",
        "ground_truth_chunk_ids": []
    },
    # CROSS DOCUMENT
    {
        "id": "q041",
        "question": "How did India's GDP growth change from 2022 to 2024?",
        "ground_truth_answer": "India's GDP growth remained resilient at 7 per cent in 2022-23 and accelerated to 8.2 per cent in 2023-24",
        "source_doc": "economic_survey_2023_24.pdf",
        "source_pages": [10, 11],
        "question_type": "cross_document",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q042",
        "question": "How did inflation trend change between 2022 and 2024?",
        "ground_truth_answer": "CPI inflation declined from elevated levels above 7 per cent in 2022 to below 5 per cent by 2024",
        "source_doc": "monetary_policy_report_2024_apr.pdf",
        "source_pages": [15, 16],
        "question_type": "cross_document",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q043",
        "question": "How did bank asset quality improve from 2022 to 2024?",
        "ground_truth_answer": "Gross NPA ratio declined from 5 per cent in 2022 to below 3 per cent by 2024 showing significant improvement",
        "source_doc": "financial_stability_report_2024_jun.pdf",
        "source_pages": [18, 19],
        "question_type": "cross_document",
        "ground_truth_chunk_ids": []
    },
    # ANNUAL REPORT QUESTIONS
    {
        "id": "q044",
        "question": "What was RBI's total balance sheet size in 2023?",
        "ground_truth_answer": "RBI's balance sheet expanded significantly in 2023 reflecting growth in foreign currency assets",
        "source_doc": "annual_report_2023.pdf",
        "source_pages": [75, 76],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q045",
        "question": "How many bank branches were opened in India in 2023?",
        "ground_truth_answer": "Thousands of new bank branches were opened in 2023 as part of financial inclusion efforts",
        "source_doc": "annual_report_2023.pdf",
        "source_pages": [100, 101],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q046",
        "question": "What was the dividend paid by RBI to government in 2024?",
        "ground_truth_answer": "RBI paid a record surplus transfer to the government in 2024",
        "source_doc": "annual_report_2024.pdf",
        "source_pages": [55, 56],
        "question_type": "factual",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q047",
        "question": "What is UPI and how has it grown?",
        "ground_truth_answer": "UPI is Unified Payments Interface that has grown to process billions of transactions monthly making India a leader in digital payments",
        "source_doc": "annual_report_2023.pdf",
        "source_pages": [120, 121],
        "question_type": "domain_jargon",
        "ground_truth_chunk_ids": []
    },
    # UNANSWERABLE
    {
        "id": "q048",
        "question": "What is the RBI governor's home address?",
        "ground_truth_answer": "This information is not available in the RBI documents",
        "source_doc": "none",
        "source_pages": [],
        "question_type": "unanswerable",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q049",
        "question": "What will India's GDP be in 2050?",
        "ground_truth_answer": "Future GDP projections for 2050 are not available in the provided documents",
        "source_doc": "none",
        "source_pages": [],
        "question_type": "unanswerable",
        "ground_truth_chunk_ids": []
    },
    {
        "id": "q050",
        "question": "What is the personal net worth of RBI governor?",
        "ground_truth_answer": "Personal financial information about RBI officials is not in these documents",
        "source_doc": "none",
        "source_pages": [],
        "question_type": "unanswerable",
        "ground_truth_chunk_ids": []
    },
]

# combine
all_questions = existing + new_questions
data['eval_set'] = all_questions

# save
with open('data/eval/gold_eval_set.json', 'w',
          encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Expanded eval set: {len(all_questions)} questions")
print(f"\nQuestion type breakdown:")
types = {}
for q in all_questions:
    t = q['question_type']
    types[t] = types.get(t, 0) + 1
for t, count in sorted(types.items()):
    print(f"  {t}: {count}")