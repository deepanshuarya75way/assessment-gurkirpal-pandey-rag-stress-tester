# scripts/create_presentation_notes.py
# Creates detailed presentation notes for each slide

notes = """
PRESENTATION NOTES — RAG Pipeline Stress-Tester
Duration: 20 minutes + Q&A
Slides: 12

================================================
SLIDE 1: TITLE
================================================
Title: RAG Pipeline Stress-Tester
Subtitle: Building, Breaking and Understanding
          Retrieval-Augmented Generation on
          Indian Financial Documents

Say:
"Today I'll walk you through a 5-week project
where I built a RAG system, deliberately broke it
in 20 different ways, and figured out exactly
why it fails. The corpus is 17 RBI and Economic
Survey documents from 2022-2024."

================================================
SLIDE 2: PROBLEM STATEMENT
================================================
Title: Why Does RAG Fail?

Key points:
- RAG is used everywhere in enterprise AI
- But fails in subtle, predictable ways
- A confident wrong answer is worse than no answer
- We need to understand failure patterns

Show: Example of wrong-year retrieval
"The system retrieved 2022 data when asked
about 2024 — with full confidence!"

Say:
"The core question is: under what conditions
does RAG fail, and can we explain it?"

================================================
SLIDE 3: CORPUS AND PIPELINE
================================================
Title: What We Built

Corpus:
- 17 PDFs, 3700 pages, 1.33M words
- RBI Monetary Policy Reports (6)
- Financial Stability Reports (6)
- Annual Reports (3)
- Economic Survey (2)

Pipeline diagram:
PDF → Extract → Chunk → Embed → Index → Retrieve → Generate → Evaluate

Say:
"The corpus was deliberately chosen because
all documents discuss the same topics across
different years — this creates realistic
retrieval challenges."

================================================
SLIDE 4: WHAT WE TESTED (20 CONFIGS)
================================================
Title: 20 Configurations — One Factor at a Time

Show table:
Factor          Options Tested
Chunking        Fixed(128/256/512), Sentence,
                Paragraph, Recursive, Semantic,
                Hierarchical
Embeddings      BGE-large, Gemini, Instructor
Retrieval       BM25, Dense, Hybrid, HyDE, Reranker
Alpha           0.4, 0.6, 0.8

Say:
"We used one-factor-at-a-time approach.
Fix everything, vary one thing, measure impact.
This isolates each component's contribution."

================================================
SLIDE 5: CHUNKING RESULTS
================================================
Title: Chunking Module Findings

Key finding:
- For BM25: more chunks = better (128 wins)
- For Hybrid: 256 words = optimal
- Advanced chunkers match fixed-size performance
- Paragraph chunker worst (fewest chunks)

Show chart: chunk size vs R@5 for BM25 and Hybrid

Say:
"The interesting finding here is that
sophisticated chunking strategies like semantic
and hierarchical achieve identical scores to
simple fixed-size chunking. Complexity doesn't
always help!"

================================================
SLIDE 6: EMBEDDING RESULTS
================================================
Title: Embedding Model Findings

Show comparison:
BGE-large (1024 dims, local):  R@5 = 0.59
Gemini (3072 dims, API):       R@5 = 0.77
Instructor (1024 dims, local): R@5 = 0.41

Key finding:
- Gemini significantly outperforms local models
- 3072 dimensions vs 1024 explains some gap
- Better domain pre-training explains rest

Say:
"For financial domain text, Gemini embeddings
are clearly superior. The 18% improvement over
BGE justifies the API cost for production use."

================================================
SLIDE 7: RETRIEVAL RESULTS
================================================
Title: Retrieval Strategy Findings

Show bar chart:
BM25:           R@5 = 0.23
Dense(Gemini):  R@5 = 0.36
Hybrid(Gemini): R@5 = 0.77  ← WINNER
HyDE(Gemini):   R@5 = 0.77
Reranker@10:    R@5 = 0.73

Key finding:
- Hybrid dominates all strategies
- BM25 alone inadequate for financial questions
- HyDE competitive but adds API cost
- Reranker adds latency with marginal gain

Say:
"The 3x improvement from BM25 to Hybrid is
the most important finding in this project.
Keyword search simply cannot handle financial
domain semantics."

================================================
SLIDE 8: FAILURE MODE 1
================================================
Title: Failure Mode 1 — Wrong Year (60%)

Show example:
Question: "What was repo rate in April 2024?"
Expected: monetary_policy_report_2024_apr
Got:      monetary_policy_report_2022_oct

Why:
- All 6 MPR docs discuss identical topics
- Year info weakly encoded in chunk text
- Embeddings capture topic not time

Fix:
collection.query(where={"year": "2024"})

Say:
"This is the most common failure. The system
knows WHAT to look for but not WHEN.
Adding a year metadata filter would fix 60%
of all failures instantly."

================================================
SLIDE 9: FAILURE MODE 2 AND 3
================================================
Title: Failure Modes 2 and 3

Failure Mode 2 — Cross-Doc Type Confusion (30%)
- MCLR, SLR, LAF appear in ALL doc types
- System retrieves FSR when answer is in MPR
- Fix: doc_type metadata filter

Failure Mode 3 — Cross-Document Synthesis (10%)
- "How did stance change 2022-2024?" always fails
- Single-hop RAG can't synthesize multiple years
- Fix: Multi-hop retrieval strategy

Say:
"Failure modes 2 and 3 are architectural
limitations. Mode 2 is fixable with metadata.
Mode 3 requires a fundamentally different
retrieval approach."

================================================
SLIDE 10: END-TO-END QUALITY
================================================
Title: Answer Generation Quality

Show:
Retrieval:  Recall@5 = 0.77
Generation: ROUGE-L  = 0.18
            Judge     = 3.67/5

Key insight:
Good retrieval is NECESSARY but NOT SUFFICIENT
Generation step introduces its own errors

Show example of 5/5 answer:
Q: "What was India GDP growth 2023-24?"
A: "India real GDP growth was 8.2% in 2023-24"
Judge: 5/5 — Correct and complete!

Say:
"Even with 77% retrieval accuracy, answer
quality is 3.67/5. The generation step loses
some information. This shows why measuring
retrieval alone is insufficient."

================================================
SLIDE 11: RECOMMENDATION
================================================
Title: If You Were Shipping Today

Recommended config:
Fixed256 + Gemini + Hybrid (alpha=0.6)

Performance:
Recall@5 = 0.77
MRR      = 0.61
Judge    = 3.67/5

What it handles well:
- Factual questions: 80% accuracy
- Domain jargon: 25% accuracy (needs work)

What it won't handle:
- Year-specific disambiguation
- Cross-document synthesis
- Jargon in multiple doc types

Next steps:
1. Year + doc_type metadata filters
2. Multi-hop retrieval
3. Domain fine-tuning

Say:
"A system with 77% recall and a clear
explanation of the remaining 23% is more
valuable than a black-box claiming 90%."

================================================
SLIDE 12: SUMMARY AND LEARNINGS
================================================
Title: Key Learnings

1. Hybrid >> Dense >> BM25
   (3x improvement, most impactful finding)

2. 256-word chunks optimal for hybrid
   (balance of precision and context)

3. Gemini embeddings >> local models
   (3072 dims, better domain pre-training)

4. RAG fails predictably, not randomly
   (3 clear failure modes, all explainable)

5. Good retrieval ≠ good answers
   (generation step introduces own errors)

6. Advanced chunking ≠ better results
   (semantic/hierarchical = same as fixed)

Say:
"The most surprising finding was that
sophisticated chunking strategies don't
outperform simple fixed-size chunking.
Sometimes simple is best."

Project repo:
https://github.com/gurkirpalpandey/rag-stress-tester
"""

with open('results/presentation_notes.md', 'w',
          encoding='utf-8') as f:
    f.write(notes)

print("Presentation notes saved!")
print("File: results/presentation_notes.md")