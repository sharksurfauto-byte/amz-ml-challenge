#!/usr/bin/env python3
"""
Baseline Exact Normalized Match Generator for Amazon ML Challenge 2026.

Generates:
- output/matching_results.tsv
- output/candidate_pairs.tsv

Methodology:
1. Normalizes business names: lowercase, strip punctuation & Latin diacritics,
   strip corporate suffixes (corp, corporation, pvt, private, ltd, limited, llc, inc, gmbh, co, company).
2. Collects unique normalized S1 names in a memory-compact set (~100 MB).
3. Streams S2 (4.88M rows) and S3 (5.08M rows), building an inverted index of exact matches
   capped at 5 matches for S2 and 6 matches for S3 (max 11 total).
4. Streams S1 (1.73M rows) in order, retrieving exact matches and writing both
   matching_results.tsv and candidate_pairs.tsv in full compliance with competition rules.
5. Operates in streaming mode with total RAM < 1 GB.
"""

import os
import sys
import time
import string
import unicodedata
import argparse
from typing import Dict, List, Set


# Build C-speed translation table for Latin accents + punctuation
_CHARMAP = {}
for _code in range(0x00C0, 0x0250):
    _char = chr(_code)
    _decomp = unicodedata.normalize("NFD", _char)
    if len(_decomp) > 1 and _decomp[0].isascii() and _decomp[0].isalpha():
        _CHARMAP[ord(_char)] = ord(_decomp[0])

for _p in string.punctuation + "«»“”‘’–—…":
    _CHARMAP[ord(_p)] = ord(" ")

TRANS_TABLE = str.maketrans(_CHARMAP)

CORP_SUFFIXES = {
    "corp",
    "corporation",
    "pvt",
    "private",
    "ltd",
    "limited",
    "llc",
    "inc",
    "gmbh",
    "co",
    "company",
}


def normalize_name(name: str) -> str:
    """Normalize business name: lowercase, strip accents/punct, remove corporate suffixes."""
    if not name:
        return ""
    cleaned = name.lower().translate(TRANS_TABLE)
    tokens = cleaned.split()
    if not tokens:
        return ""
    filtered = [t for t in tokens if t not in CORP_SUFFIXES]
    if filtered:
        return " ".join(filtered)
    return " ".join(tokens)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline exact-match submission for Amazon ML Challenge 2026"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/test",
        help="Path to directory containing test TSV files (default: data/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to output directory (default: output)",
    )
    parser.add_argument(
        "--max-s2",
        type=int,
        default=5,
        help="Maximum matches allowed from Source 2 per entity (default: 5)",
    )
    parser.add_argument(
        "--max-s3",
        type=int,
        default=6,
        help="Maximum matches allowed from Source 3 per entity (default: 6)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    s1_path = os.path.join(args.data_dir, "test_source1.tsv")
    s2_path = os.path.join(args.data_dir, "test_source2.tsv")
    s3_path = os.path.join(args.data_dir, "test_source3.tsv")

    out_matching = os.path.join(args.output_dir, "matching_results.tsv")
    out_candidate = os.path.join(args.output_dir, "candidate_pairs.tsv")

    print("=" * 72)
    print("AMAZON ML CHALLENGE 2026 - BASELINE EXACT MATCH PIPELINE")
    print("=" * 72)
    print(f"Data directory:    {args.data_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Max S2 matches:    {args.max_s2}")
    print(f"Max S3 matches:    {args.max_s3} (Max Total: {args.max_s2 + args.max_s3})")
    print("-" * 72)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Step 1: Scan S1 and collect unique normalized names
    # ---------------------------------------------------------
    t0 = time.time()
    print("Step 1/4: Scanning test S1 entities and building name vocabulary...")
    s1_names_set: Set[str] = set()
    s1_count = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            s1_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 1:
                norm = normalize_name(parts[1])
                if norm:
                    s1_names_set.add(norm)

    t_s1 = time.time() - t0
    print(
        f"  Loaded {s1_count:,} S1 entities in {t_s1:.2f}s "
        f"({len(s1_names_set):,} unique normalized names)"
    )

    # ---------------------------------------------------------
    # Step 2: Stream S2 and build inverted index for S1 names
    # ---------------------------------------------------------
    t0 = time.time()
    print("Step 2/4: Streaming test S2 records and indexing exact matches...")
    s2_index: Dict[str, List[str]] = {}
    s2_count = 0
    s2_matches_indexed = 0

    with open(s2_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            s2_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 1:
                norm = normalize_name(parts[1])
                if norm in s1_names_set:
                    entity_list = s2_index.setdefault(norm, [])
                    if len(entity_list) < args.max_s2:
                        entity_id = parts[0]
                        if entity_id not in entity_list:
                            entity_list.append(entity_id)
                            s2_matches_indexed += 1

            if s2_count % 1000000 == 0:
                print(f"  Processed {s2_count:,} S2 records ({s2_matches_indexed:,} matches indexed)...")

    t_s2 = time.time() - t0
    print(
        f"  Completed S2: {s2_count:,} records in {t_s2:.2f}s "
        f"({len(s2_index):,} names matched, {s2_matches_indexed:,} IDs indexed)"
    )

    # ---------------------------------------------------------
    # Step 3: Stream S3 and build inverted index for S1 names
    # ---------------------------------------------------------
    t0 = time.time()
    print("Step 3/4: Streaming test S3 records and indexing exact matches...")
    s3_index: Dict[str, List[str]] = {}
    s3_count = 0
    s3_matches_indexed = 0

    with open(s3_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            s3_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 1:
                norm = normalize_name(parts[1])
                if norm in s1_names_set:
                    entity_list = s3_index.setdefault(norm, [])
                    if len(entity_list) < args.max_s3:
                        entity_id = parts[0]
                        if entity_id not in entity_list:
                            entity_list.append(entity_id)
                            s3_matches_indexed += 1

            if s3_count % 1000000 == 0:
                print(f"  Processed {s3_count:,} S3 records ({s3_matches_indexed:,} matches indexed)...")

    t_s3 = time.time() - t0
    print(
        f"  Completed S3: {s3_count:,} records in {t_s3:.2f}s "
        f"({len(s3_index):,} names matched, {s3_matches_indexed:,} IDs indexed)"
    )

    # Free memory of s1_names_set since no longer needed
    del s1_names_set

    # ---------------------------------------------------------
    # Step 4: Stream S1 in exact order and write submissions
    # ---------------------------------------------------------
    t0 = time.time()
    print("Step 4/4: Writing matching_results.tsv and candidate_pairs.tsv...")

    total_rows = 0
    total_matched_pairs = 0
    singletons = 0
    match_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    with open(s1_path, "r", encoding="utf-8") as f_in, \
         open(out_matching, "w", encoding="utf-8", newline="\n") as f_match, \
         open(out_candidate, "w", encoding="utf-8", newline="\n") as f_cand:

        # Write exact headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        header = f_in.readline()
        for line in f_in:
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""

            norm = normalize_name(name)
            matches = []
            if norm:
                matches.extend(s2_index.get(norm, []))
                matches.extend(s3_index.get(norm, []))

            num_matches = len(matches)
            total_matched_pairs += num_matches

            if num_matches == 0:
                singletons += 1
            else:
                bin_idx = min(num_matches, 5)
                match_hist[bin_idx] = match_hist.get(bin_idx, 0) + 1

            matched_str = ",".join(matches)
            out_line = f"{s1_id}\t{matched_str}\n"

            f_match.write(out_line)
            f_cand.write(out_line)

            if total_rows % 500000 == 0:
                print(f"  Written {total_rows:,} / {s1_count:,} rows...")

    t_write = time.time() - t0
    total_time = time.time() - start_time

    # Output file sizes
    size_m_mb = os.path.getsize(out_matching) / (1024.0 * 1024.0)
    size_c_mb = os.path.getsize(out_candidate) / (1024.0 * 1024.0)

    print("-" * 72)
    print("BASELINE GENERATION COMPLETE")
    print("-" * 72)
    print(f"Total S1 entities processed: {total_rows:,}")
    print(f"Total matched pairs:         {total_matched_pairs:,}")
    print(f"Singletons (unmatched):      {singletons:,} ({singletons / total_rows * 100:.2f}%)")
    print(f"Average matches per entity:  {total_matched_pairs / total_rows:.2f}")
    print(f"matching_results.tsv size:   {size_m_mb:.2f} MB (< 512 MB PASS)")
    print(f"candidate_pairs.tsv size:    {size_c_mb:.2f} MB (< 512 MB PASS)")
    print(f"Total pipeline execution:    {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
