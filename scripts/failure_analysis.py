# scripts/failure_analysis.py
import json
import os
import re
import sys
sys.path.append('.')


def extract_doc_name(chunk_id):
    return re.sub(r'_p\d+_[a-z]+\d+$', '', chunk_id)


results_dir = 'results'
result_files = sorted([f for f in os.listdir(results_dir)
                      if 'hybrid' in f and f.endswith('.json')])

if not result_files:
    print("No hybrid results found!")
    sys.exit(1)

latest = os.path.join(results_dir, result_files[-1])
print(f"Loading: {latest}\n")

with open(latest, 'r') as f:
    data = json.load(f)

results = data['per_question_results']

hits = [r for r in results if r['recall@3'] > 0]
misses = [r for r in results if r['recall@3'] == 0]

print(f"{'='*55}")
print(f"FAILURE ANALYSIS REPORT")
print(f"{'='*55}")
print(f"Total questions: {len(results)}")
print(f"Hits:   {len(hits)}")
print(f"Misses: {len(misses)}")
print(f"Success rate: {len(hits)/len(results)*100:.1f}%")

print(f"\n{'='*55}")
print(f"HITS")
print(f"{'='*55}")
for r in hits:
    print(f"Q: {r['question'][:55]}...")
    print(f"   Type:     {r['question_type']}")
    print(f"   Expected: {r['correct_doc']}")
    got = extract_doc_name(r['retrieved_docs'][0]) if r['retrieved_docs'] else 'none'
    print(f"   Got:      {got}")
    print()

print(f"\n{'='*55}")
print(f"MISSES")
print(f"{'='*55}")
for r in misses:
    print(f"Q: {r['question'][:55]}...")
    print(f"   Type:     {r['question_type']}")
    print(f"   Expected: {r['correct_doc']}")
    got = extract_doc_name(r['retrieved_docs'][0]) if r['retrieved_docs'] else 'none'
    print(f"   Got:      {got}")
    print()

print(f"\n{'='*55}")
print(f"BY QUESTION TYPE")
print(f"{'='*55}")
types = {}
for r in results:
    t = r['question_type']
    if t not in types:
        types[t] = {'hits': 0, 'total': 0}
    types[t]['total'] += 1
    if r['recall@3'] > 0:
        types[t]['hits'] += 1

for t, counts in types.items():
    rate = counts['hits'] / counts['total'] * 100
    print(f"{t:20} → {counts['hits']}/{counts['total']} ({rate:.0f}%)")

print(f"\n{'='*55}")
print(f"BY DOCUMENT")
print(f"{'='*55}")
docs = {}
for r in results:
    d = r['correct_doc']
    if d not in docs:
        docs[d] = {'hits': 0, 'total': 0}
    docs[d]['total'] += 1
    if r['recall@3'] > 0:
        docs[d]['hits'] += 1

for d, counts in sorted(docs.items()):
    rate = counts['hits'] / counts['total'] * 100
    status = "OK" if rate > 50 else "FAIL"
    print(f"[{status}] {d[:50]}: {counts['hits']}/{counts['total']} ({rate:.0f}%)")

print(f"\n{'='*55}")
print(f"FAILURE PATTERNS")
print(f"{'='*55}")
mpr_miss = [r for r in misses if 'monetary_policy' in r['correct_doc']]
econ_miss = [r for r in misses if 'economic_survey' in r['correct_doc']]
fsr_miss = [r for r in misses if 'financial_stability' in r['correct_doc']]
cross_miss = [r for r in misses if r['question_type'] == 'cross_document']
jargon_miss = [r for r in misses if r['question_type'] == 'domain_jargon']

print(f"Monetary policy misses:    {len(mpr_miss)}")
print(f"Economic survey misses:    {len(econ_miss)}")
print(f"Financial stability misses:{len(fsr_miss)}")
print(f"Cross-document misses:     {len(cross_miss)}")
print(f"Domain jargon misses:      {len(jargon_miss)}")

print(f"\n{'='*55}")
print(f"3 MAIN FAILURE MODES")
print(f"{'='*55}")
print(f"1. MONETARY POLICY CONFUSION ({len(mpr_miss)} questions)")
print(f"   All MPR docs discuss same topics (repo rate,")
print(f"   inflation) so system retrieves wrong year.")
print()
print(f"2. CROSS-DOCUMENT FAILURE ({len(cross_miss)} questions)")
print(f"   Questions needing multiple docs always fail.")
print(f"   System retrieves from one doc only.")
print()
print(f"3. DOMAIN JARGON CONFUSION ({len(jargon_miss)} questions)")
print(f"   MCLR, SLR, LAF appear in multiple docs.")
print(f"   System retrieves semantically similar wrong doc.")