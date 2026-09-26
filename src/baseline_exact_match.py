#!/usr/bin/env python3
"""
Baseline 1 (Exact Name Match) Submission Generator
Amazon ML Challenge 2026

This script generates a fast, high-precision smoke-test submission by executing
exact normalized name matching partitioned strictly by country.
Execution time: < 60 seconds on 11.7 million test records.

Outputs:
  - output/matching_results.tsv  (Scored on portal)
  - output/candidate_pairs.tsv   (Identical to matching_results in this baseline)
"""

import os
import re
import time
import unicodedata
import polars as pl
from collections import defaultdict


def clean_business_name(text: str) -> str:
    """Fast lexical normalization for baseline."""
    if not text or not isinstance(text, str):
        return ""

    # Unicode NFKD normalization to cleanly strip French diacritics
    s = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    s = s.lower()

    # Strip basic domain extensions commonly found in S3
    s = re.sub(r'([a-z0-9-]+)\.(?:com|org|net|in|co|us|biz|info|io|fr)\b', r'\1', s)

    # Strip punctuation and standard legal entity noise
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r'\b(inc|corp|corporation|llc|ltd|limited|pvt|private|co|company|sarl|sa)\b', ' ', s)

    # Collapse whitespace
    return re.sub(r'\s+', ' ', s).strip()


def run_exact_match_baseline(data_dir: str, output_dir: str):
    t0 = time.time()

    s1_path = os.path.join(data_dir, "test_source1.tsv")
    s2_path = os.path.join(data_dir, "test_source2.tsv")
    s3_path = os.path.join(data_dir, "test_source3.tsv")

    print("[*] Building Hash Index for Target Candidates (S2 & S3)...")
    # Lookup structure: (country, normalized_name) -> list of entity IDs
    lookup = defaultdict(list)

    # Process S2 and S3 using Polars streaming
    for path, prefix in [(s2_path, "S2"), (s3_path, "S3")]:
        print(f"    - Loading {os.path.basename(path)}...")
        df = pl.read_csv(path, separator="\t", columns=["entity_id", "business_name", "country"])
        for row in df.iter_rows(named=True):
            eid = row["entity_id"]
            name = row["business_name"]
            country = row["country"]

            if not country or not name:
                continue

            clean_name = clean_business_name(name)
            # Only index entities with meaningful names after cleaning
            if clean_name and len(clean_name) > 2:
                key = (country, clean_name)
                lookup[key].append(eid)

    print(f"[*] Hash Index Built: {len(lookup):,} unique (country, name) keys indexed.")

    print("\n[*] Processing Source 1 Queries...")
    df_s1 = pl.read_csv(s1_path, separator="\t", columns=["entity_id", "business_name", "country"])
    all_s1_ids = df_s1["entity_id"].to_list()

    predictions = {}
    match_count = 0

    for row in df_s1.iter_rows(named=True):
        eid = row["entity_id"]
        name = row["business_name"]
        country = row["country"]

        matches = []
        if country and name:
            clean_name = clean_business_name(name)
            if clean_name and len(clean_name) > 2:
                key = (country, clean_name)
                if key in lookup:
                    matches = lookup[key]

        # Deduplicate and sort targets
        matches = sorted(list(set(matches)))
        predictions[eid] = ",".join(matches)

        if matches:
            match_count += 1

    print(f"[*] Total S1 Entities Evaluated: {len(df_s1):,}")
    print(f"    Entities mapped to >=1 match: {match_count:,} ({match_count / len(df_s1) * 100:.2f}%)")
    print(f"    Predicted Singletons (Empty): {len(df_s1) - match_count:,} ({(len(df_s1) - match_count) / len(df_s1) * 100:.2f}%)")

    # Write exact formatted validation files
    match_out = os.path.join(output_dir, "matching_results.tsv")
    cand_out  = os.path.join(output_dir, "candidate_pairs.tsv")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[*] Writing outputs to {output_dir}/...")

    # In this exact match baseline, the candidates are exactly the final matches.
    with open(match_out, "w", encoding="utf-8") as fm, open(cand_out, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_eid in all_s1_ids:
            res_str = predictions.get(s1_eid, "")
            fm.write(f"{s1_eid}\t{res_str}\n")
            fc.write(f"{s1_eid}\t{res_str}\n")

    print(f"[+] DONE in {time.time() - t0:.2f} seconds!")
    print("\nNext step -> RUN VALIDATOR:")
    print(f"python validate_submission.py --matching {match_out} --candidate {cand_out} --test-dir {data_dir}")


if __name__ == "__main__":
    run_exact_match_baseline(data_dir="data/raw/test", output_dir="output")
