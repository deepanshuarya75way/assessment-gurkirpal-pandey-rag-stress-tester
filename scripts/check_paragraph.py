import sys
sys.path.append('.')

# test the doc name extraction
chunk_ids = [
    "monetary_policy_report_2023_oct_p27_pc3830",
    "monetary_policy_report_2023_oct_p66_pc3881",
    "financial_stability_report_2024_dec_p81_pc3138",
]

for cid in chunk_ids:
    doc = cid.rsplit('_p', 1)[0]
    print(f"Chunk: {cid}")
    print(f"Doc:   {doc}")
    print()