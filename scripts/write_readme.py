readme = """# RAG Pipeline Stress-Tester

> A research project to build, systematically break, and understand RAG pipelines on Indian financial documents.

**Corpus:** RBI Reports + Economic Survey 2022-2024 | **Experiments:** 20 configurations | **Best Result:** Recall@5=0.77, MRR=0.61

---

## What This Project Does

This project stress-tests RAG pipelines by:

- Testing **6 chunking strategies** (fixed, sentence, paragraph, recursive, semantic, hierarchical)
- Comparing **3 embedding models** (BGE-large, Gemini, Instructor)
- Evaluating **5 retrieval strategies** (BM25, Dense, Hybrid, HyDE, Reranker)
- Running **20 targeted configurations** using one-factor-at-a-time approach
- Identifying **3 systematic failure modes** with root cause analysis

---

## Quick Start

**Step 1 — Clone and install**

    git clone https://github.com/gurkirpalpandey/rag-stress-tester.git
    cd rag-stress-tester
    py -3.11 -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt

**Step 2 — Add API keys** (create .env file)

    GEMINI_API_KEY=your-key-here
    HUGGINGFACE_TOKEN=hf_your-token-here

Get Gemini key: https://aistudio.google.com/apikey

**Step 3 — Download corpus**

Download from rbi.org.in and indiabudget.gov.in, place in data/raw/rbi/ and data/raw/economic_survey/

**Step 4 — Process PDFs**

    .venv\\Scripts\\python.exe scripts\\prepare_data.py
    .venv\\Scripts\\python.exe scripts\\update_eval_ids.py
    .venv\\Scripts\\python.exe scripts\\smart_embed.py

**Step 5 — Run best configuration**

    .venv\\Scripts\\python.exe scripts\\run_eval.py --config config\\exp07_fixed256_gemini_hybrid.yaml

Expected output: Recall@1=0.50, Recall@3=0.73, Recall@5=0.77, MRR=0.61

**Step 6 — View MLflow dashboard**

    mlflow ui
    # Open http://localhost:5000

**Step 7 — Run analysis**

    .venv\\Scripts\\python.exe scripts\\failure_analysis.py
    .venv\\Scripts\\python.exe scripts\\deep_dive.py

---

## Repository Structure

    rag-stress-tester/
    |
    +-- src/
    |   +-- interfaces.py              # Shared data structures (Chunk, EmbeddedChunk, etc.)
    |   +-- chunking/
    |   |   +-- fixed_size.py          # Split every N words with overlap
    |   |   +-- sentence_based.py      # Split at sentence boundaries
    |   |   +-- paragraph_based.py     # Split at paragraph boundaries
    |   |   +-- recursive.py           # LangChain-style recursive splitting
    |   |   +-- semantic.py            # Split where topic changes (cosine similarity)
    |   |   +-- hierarchical.py        # Parent (512w) + child (128w) chunks
    |   +-- embeddings/
    |   |   +-- bge_embedder.py        # BGE-large-en-v1.5 (local, free, 1024 dims)
    |   |   +-- gemini_embedder.py     # Gemini-embedding-001 (API, 3072 dims)
    |   |   +-- instructor_embedder.py # Instructor-large with task instructions
    |   +-- retrieval/
    |   |   +-- bm25_retriever.py      # Keyword search (no embeddings needed)
    |   |   +-- dense_retriever.py     # Vector similarity search via ChromaDB
    |   |   +-- hybrid_retriever.py    # BM25 + Dense blended (alpha=0.6)
    |   |   +-- hyde_retriever.py      # Hypothetical Document Embeddings
    |   |   +-- reranker.py            # Cross-encoder reranker (ms-marco-MiniLM)
    |   +-- evaluation/
    |       +-- metrics.py             # Recall@k, MRR, Context Precision, ROUGE-L, LLM-judge
    |       +-- mlflow_logger.py       # MLflow experiment tracking
    |
    +-- config/                        # 20 YAML experiment configuration files
    |   +-- baseline.yaml
    |   +-- exp01_fixed256_bge_bm25.yaml
    |   +-- exp07_fixed256_gemini_hybrid.yaml  # BEST CONFIG
    |   +-- ... (20 configs total)
    |
    +-- data/
    |   +-- raw/                       # Downloaded PDFs (gitignored)
    |   +-- processed/                 # Cleaned JSONL files (gitignored)
    |   +-- eval/
    |       +-- gold_eval_set.json     # 25 hand-written Q&A pairs
    |
    +-- scripts/
    |   +-- prepare_data.py            # PDF extraction and cleaning
    |   +-- run_eval.py                # Main evaluation pipeline
    |   +-- smart_embed.py             # Efficient Gemini embedding (saves quota)
    |   +-- update_eval_ids.py         # Sync ground truth chunk IDs
    |   +-- failure_analysis.py        # Analyze which questions fail
    |   +-- deep_dive.py               # Root cause investigations
    |   +-- collect_results.py         # Aggregate all experiment results
    |
    +-- results/                       # JSON outputs per experiment (gitignored)
    +-- requirements.txt
    +-- README.md
    +-- .gitignore
    +-- .env                           # API keys (gitignored, never committed)

---

## Key Results

All 20 configurations tested using one-factor-at-a-time (OFAT) approach:

    Rank  Config                              R@1    R@3    R@5    MRR
    ----  ----------------------------------  -----  -----  -----  -----
    1     Fixed256 + Gemini + Hybrid          0.500  0.727  0.773  0.610  BEST
    2     Recursive + Gemini + Hybrid         0.500  0.727  0.773  0.610
    3     Hierarchical + Gemini + Hybrid      0.500  0.727  0.773  0.610
    4     Fixed256 + Gemini + HyDE            0.455  0.636  0.773  0.577
    5     Fixed256 + Gemini + Hybrid (a=0.8)  0.455  0.682  0.773  0.583
    6     Fixed128 + Gemini + Hybrid          0.409  0.636  0.727  0.523
    7     Fixed256 + BGE + Dense              0.318  0.455  0.591  0.403
    8     Fixed128 + BGE + BM25               0.318  0.318  0.409  0.336
    9     Fixed256 + Gemini + Dense           0.273  0.318  0.364  0.299
    10    Fixed256 + BGE + BM25               0.045  0.182  0.227  0.110  BASELINE

---

## 3 Failure Modes Identified

**Failure Mode 1 — Wrong-Year Retrieval (60% of failures)**

All 6 MPR documents discuss identical topics (repo rate, inflation, growth).
System retrieves correct topic but wrong year.
Fix: Add year metadata filter to ChromaDB queries.

**Failure Mode 2 — Cross-Document Type Confusion (30% of failures)**

Jargon terms like MCLR, SLR, LAF appear in both MPR and FSR documents.
System retrieves FSR when answer is in MPR, or vice versa.
Fix: Add doc_type metadata filter to ChromaDB queries.

**Failure Mode 3 — Cross-Document Synthesis (10% of failures)**

Questions needing 2022+2023+2024 data always fail.
Single-hop RAG cannot synthesize across multiple documents.
Fix: Multi-hop retrieval strategy.

---

## Evaluation Metrics

    Metric             Description
    -----------------  -----------------------------------------------
    Recall@k           Correct document found in top-k results (k=1,3,5)
    MRR                Mean Reciprocal Rank of correct document
    Context Precision  Relevant chunks divided by total retrieved
    Exact Match        Exact string match between answer and ground truth
    ROUGE-L            Longest Common Subsequence based overlap score
    LLM-as-Judge       Gemini scores generated answer from 1 to 5

---

## Tech Stack

    Component           Library/Tool
    ------------------  ---------------------------
    Language            Python 3.11
    PDF Extraction      PyMuPDF (fitz)
    Vector Database     ChromaDB
    Local Embeddings    sentence-transformers
    Cloud Embeddings    google-genai (Gemini API)
    Keyword Search      rank-bm25
    Experiment Tracking MLflow
    Config Management   PyYAML
    API Key Security    python-dotenv

---

## What I Would Explore Next

1. Year and doc_type metadata filtering in ChromaDB to fix wrong-year retrieval
2. Multi-hop retrieval for cross-document synthesis questions
3. Fine-tuning BGE on RBI domain text for better jargon understanding
4. Query rewriting to handle ambiguous financial terminology
5. Ensemble of multiple retrievers with score voting

---

## Project Timeline

    Week 1 (Days 1-5)   Foundation, data pipeline, baseline pipeline
    Week 2 (Days 6-10)  Chunking strategies and retrieval experiments
    Week 3 (Days 11-12) HyDE, reranker, new eval metrics implementation
    Week 4 (Days 13-14) 20-config experiment matrix and failure analysis
    Week 5 (Days 15+)   MLflow, README, report, presentation

---

## Author

Gurkirpal Pandey — Summer Internship Project 2026
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("README.md written successfully!")