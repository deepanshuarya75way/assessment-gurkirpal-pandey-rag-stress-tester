readme = """# RAG Pipeline Stress-Tester

A research project to build, systematically break, and understand RAG pipelines on Indian financial documents (RBI Reports + Economic Survey 2022-2024).

## Project Overview

This project stress-tests RAG pipelines by testing 6 chunking strategies, comparing 3 embedding models, evaluating 5 retrieval strategies, running 20 targeted configurations, and identifying 3 systematic failure modes.

Best Result: Hybrid + Fixed256 + Gemini gives Recall@5=0.77 and MRR=0.61

## Quick Start

### 1. Clone and Install
git clone https://github.com/gurkirpalpandey/rag-stress-tester.git
cd rag-stress-tester
py -3.11 -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt

### 2. Add API Keys
Create .env file with:
GEMINI_API_KEY=your-key-here
HUGGINGFACE_TOKEN=hf_your-token-here

Get Gemini key from: https://aistudio.google.com/apikey
Get HuggingFace token from: https://huggingface.co/settings/tokens

### 3. Download Corpus
Download PDFs from rbi.org.in and indiabudget.gov.in
Place in data/raw/rbi/ and data/raw/economic_survey/

RBI Documents needed:
- Monetary Policy Reports: 2022 Apr/Oct, 2023 Apr/Oct, 2024 Apr/Oct
- Financial Stability Reports: 2022 Jun/Dec, 2023 Jun/Dec, 2024 Jun/Dec
- Annual Reports: 2022, 2023, 2024

Economic Survey:
- Economic Survey 2022-23
- Economic Survey 2023-24

### 4. Process PDFs
.venv\\Scripts\\python.exe scripts\\prepare_data.py

### 5. Update Eval Set
.venv\\Scripts\\python.exe scripts\\update_eval_ids.py

### 6. Embed Corpus
.venv\\Scripts\\python.exe scripts\\smart_embed.py

### 7. Run Best Configuration
.venv\\Scripts\\python.exe scripts\\run_eval.py --config config\\exp07_fixed256_gemini_hybrid.yaml

Expected output:
Recall@1: 0.5000
Recall@3: 0.7273
Recall@5: 0.7727
MRR:      0.6098

### 8. View MLflow Dashboard
mlflow ui
Then open http://localhost:5000

### 9. Run Failure Analysis
.venv\\Scripts\\python.exe scripts\\failure_analysis.py
.venv\\Scripts\\python.exe scripts\\deep_dive.py

## Repository Structure

src/interfaces.py - Shared data structures
src/chunking/ - 6 chunking strategies (fixed, sentence, paragraph, recursive, semantic, hierarchical)
src/embeddings/ - 3 embedding models (BGE, Gemini, Instructor)
src/retrieval/ - 5 retrieval strategies (BM25, Dense, Hybrid, HyDE, Reranker)
src/evaluation/ - Metrics and MLflow logging
config/ - 20 experiment configuration files
data/eval/gold_eval_set.json - 25 hand-written Q&A pairs
scripts/ - All runnable scripts
results/ - Experiment outputs (gitignored)

## Key Results

Config 07: Fixed256 + Gemini + Hybrid -> R@5=0.7727, MRR=0.6098 (BEST)
Config 13: Recursive + Gemini + Hybrid -> R@5=0.7727, MRR=0.6098
Config 20: Hierarchical + Gemini + Hybrid -> R@5=0.7727, MRR=0.6098
Config 08: Fixed256 + Gemini + HyDE -> R@5=0.7727, MRR=0.5773
Config 11: Fixed128 + Gemini + Hybrid -> R@5=0.7273, MRR=0.5227
Config 01: Fixed256 + BGE + BM25 -> R@5=0.2273, MRR=0.1098 (BASELINE)

## 3 Failure Modes Identified

1. Wrong-Year Retrieval (60% of failures)
   All MPR docs discuss same topics, system retrieves wrong year.
   Fix: Add year metadata filter to ChromaDB queries.

2. Cross-Document Type Confusion (30% of failures)
   Jargon terms like MCLR appear in multiple doc types.
   Fix: Add doc_type metadata filter.

3. Cross-Document Synthesis (10% of failures)
   Single-hop RAG cannot synthesize across multiple years.
   Fix: Multi-hop retrieval strategy.

## Evaluation Metrics Used

Recall@k: Correct document found in top-k results
MRR: Mean Reciprocal Rank of correct document
Context Precision: Relevant chunks divided by total retrieved
Exact Match: Exact string match between predicted and ground truth
ROUGE-L: Longest Common Subsequence based overlap score
LLM-as-Judge: Gemini scores generated answer from 1 to 5

## Tech Stack

Python 3.11, PyMuPDF, ChromaDB, sentence-transformers,
google-genai, rank-bm25, MLflow, PyYAML, python-dotenv

## What I Would Explore Next

1. Year and doc_type metadata filtering in ChromaDB
2. Multi-hop retrieval for cross-document questions
3. Fine-tuning BGE on RBI domain text
4. Query rewriting for jargon disambiguation
5. Ensemble of multiple retrievers with voting

## Project Timeline

Week 1 (Days 1-5): Foundation, data pipeline, baseline pipeline
Week 2 (Days 6-10): Chunking strategies and retrieval experiments
Week 3 (Days 11-12): Missing components - HyDE, reranker, metrics
Week 4 (Days 13-14): 20-config experiment matrix and failure analysis
Week 5 (Days 15+): MLflow, README, report, presentation

## Author

Gurkirpal Pandey
Summer Internship Project 2026
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("README.md written successfully!")