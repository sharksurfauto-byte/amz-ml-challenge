#!/usr/bin/env python3
"""
High-Recall Multi-Channel Candidate Generator for Amazon ML Challenge 2026.
Combines:
  1. Normalized Name Matching (Suffix Stripped)
  2. Normalized Full Address Matching (Captures DBA / aka entities)
  3. First 2 Distinctive Tokens Match (Captures minor word-order / prefix shifts)
Generates high-recall candidate pairs partitioned strictly by country in < 20s.
"""

import os
import sys
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import polars as pl
from src.utils import resolve_dataset_files

SUFFIX_REGEX = r"\b(inc|corp|corporation|llc|ltd|limited|pvt|private|co|company|sarl|sa|services|enterprises|industries|group|holdings|solutions)\b"


def generate_high_recall_candidates(
    sample_size: int = 20000,
    max_cands_per_s1: int = 12,
    output_path: str = "data/benchmark_cache/high_recall_candidates_20k.tsv"
) -> str:
    t0 = time.time()
    print("=" * 70)
    print("   BUILDING HIGH-RECALL MULTI-CHANNEL CANDIDATE POOL")
    print("=" * 70)
    print(f"Sample Size: {sample_size:,} S1 entities | Max Candidates per S1: {max_cands_per_s1}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    files = resolve_dataset_files()
    s1_file = files["train_source1.tsv"]
    s2_file = files["train_source2.tsv"]
    s3_file = files["train_source3.tsv"]
    gt_file = files["train_ground_truth.tsv"]

    # 1. Load S1
    print("[*] Loading and normalizing Source 1...")
    s1 = pl.read_csv(s1_file, separator="\t", n_rows=sample_size)
    s1 = s1.with_columns([
        pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(SUFFIX_REGEX, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("cname"),
        pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("caddr"),
        pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().str.split(" ").list.get(0, null_on_oob=True).alias("t0"),
        pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().str.split(" ").list.get(1, null_on_oob=True).alias("t1"),
        pl.col("business_address").fill_null("").str.extract(r"\b(\d{1,6})\b", 1).alias("snum"),
        pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().str.slice(0, 4).alias("p4"),
        pl.col("business_name").fill_null("").str.to_lowercase().str.extract(r"([a-z0-9-]+)\.(?:com|org|net|in|co|us|biz|info|io|fr|gov|edu)\b", 1).alias("domain")
    ])
    all_s1_ids = s1["entity_id"].to_list()
    print(f"    Loaded {len(all_s1_ids):,} S1 records.", flush=True)

    # Track Candidate Sets per Source:
    s2_cands: Dict[str, List[str]] = defaultdict(list)
    s3_cands: Dict[str, List[str]] = defaultdict(list)

    # 3. Stream S2 and S3 with Vectorized Polars Joins
    for src_file, src_name, target_dict, max_src_cands in [(s2_file, "Source 2", s2_cands, 8), (s3_file, "Source 3", s3_cands, 10)]:
        t_src = time.time()
        print(f"[*] Scanning {src_name}...", flush=True)
        s_df = pl.read_csv(src_file, separator="\t", columns=["entity_id", "business_name", "business_address", "country"])
        s_df = s_df.with_columns([
            pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(SUFFIX_REGEX, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("cname"),
            pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("caddr"),
            pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().str.split(" ").list.get(0, null_on_oob=True).alias("t0"),
            pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().str.split(" ").list.get(1, null_on_oob=True).alias("t1"),
            pl.col("business_address").fill_null("").str.extract(r"\b(\d{1,6})\b", 1).alias("snum"),
            pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().str.slice(0, 4).alias("p4"),
            pl.col("business_name").fill_null("").str.to_lowercase().str.extract(r"([a-z0-9-]+)\.(?:com|org|net|in|co|us|biz|info|io|fr|gov|edu)\b", 1).alias("domain")
        ])

        # Priority 1: Suffix-Cleaned Name Join
        j_name = s1.select(["entity_id", "country", "cname"]).filter(pl.col("cname").str.len_chars() > 2).join(
            s_df.select(["entity_id", "country", "cname"]).filter(pl.col("cname").str.len_chars() > 2),
            on=["country", "cname"], how="inner", suffix="_cand"
        ).select([pl.col("entity_id").alias("s1_id"), pl.col("entity_id_cand").alias("cand_id"), pl.lit(1).alias("prio")])

        # Priority 2: Clean Full Address Join (length > 8)
        j_addr = s1.select(["entity_id", "country", "caddr"]).filter(pl.col("caddr").str.len_chars() > 8).join(
            s_df.select(["entity_id", "country", "caddr"]).filter(pl.col("caddr").str.len_chars() > 8),
            on=["country", "caddr"], how="inner", suffix="_cand"
        ).select([pl.col("entity_id").alias("s1_id"), pl.col("entity_id_cand").alias("cand_id"), pl.lit(2).alias("prio")])

        # Priority 3: Street Number + First Word Match (High precision spatial match)
        s1_sn = s1.filter((pl.col("snum").is_not_null()) & (pl.col("snum").str.len_chars() > 0) & (pl.col("t0").str.len_chars() >= 3)).select(["entity_id", "country", "snum", "t0"])
        src_sn = s_df.filter((pl.col("snum").is_not_null()) & (pl.col("snum").str.len_chars() > 0) & (pl.col("t0").str.len_chars() >= 3)).select(["entity_id", "country", "snum", "t0"])
        j_sn = s1_sn.join(src_sn, on=["country", "snum", "t0"], how="inner", suffix="_cand").select([
            pl.col("entity_id").alias("s1_id"), pl.col("entity_id_cand").alias("cand_id"), pl.lit(3).alias("prio")
        ])

        # Priority 4: First 2 Tokens Join (distinctive words >= 4 chars, filtered)
        s1_t2 = s1.filter((pl.col("t0").str.len_chars() >= 4) & (pl.col("t1").str.len_chars() >= 4)).select(["entity_id", "country", "t0", "t1"])
        cand_t2 = s_df.filter((pl.col("t0").str.len_chars() >= 4) & (pl.col("t1").str.len_chars() >= 4)).select(["entity_id", "country", "t0", "t1"])
        t2_counts = s1_t2.group_by(["country", "t0", "t1"]).len()
        s1_t2_filt = s1_t2.join(t2_counts.filter(pl.col("len") <= 25).select(["country", "t0", "t1"]), on=["country", "t0", "t1"], how="inner")
        j_t2 = s1_t2_filt.join(cand_t2, on=["country", "t0", "t1"], how="inner", suffix="_cand").select([
            pl.col("entity_id").alias("s1_id"), pl.col("entity_id_cand").alias("cand_id"), pl.lit(4).alias("prio")
        ])

        # Priority 5: 4-Char Prefix Stem Match (Filtered to rare stems <= 15 occurrences)
        s1_p4 = s1.filter(pl.col("p4").str.len_chars() == 4).select(["entity_id", "country", "p4"])
        src_p4 = s_df.filter(pl.col("p4").str.len_chars() == 4).select(["entity_id", "country", "p4"])
        p4_counts = s1_p4.group_by(["country", "p4"]).len()
        s1_p4_filt = s1_p4.join(p4_counts.filter(pl.col("len") <= 15).select(["country", "p4"]), on=["country", "p4"], how="inner")
        j_p4 = s1_p4_filt.join(src_p4, on=["country", "p4"], how="inner", suffix="_cand").select([
            pl.col("entity_id").alias("s1_id"), pl.col("entity_id_cand").alias("cand_id"), pl.lit(5).alias("prio")
        ])

        # Priority 1: Domain Match
        bad_domains = pl.Series(["gmail", "yahoo", "hotmail", "outlook", "aol", "icloud", "me", "msn", "live", "ymail", "rediffmail"])
        j_domain = s1.select(["entity_id", "country", "domain"]).filter(~pl.col("domain").is_in(bad_domains) & pl.col("domain").is_not_null()).join(
            s_df.select(["entity_id", "country", "domain"]).filter(~pl.col("domain").is_in(bad_domains) & pl.col("domain").is_not_null()),
            on=["country", "domain"], how="inner", suffix="_cand"
        ).select([pl.col("entity_id").alias("s1_id"), pl.col("entity_id_cand").alias("cand_id"), pl.lit(1).alias("prio")])

        # Merge, deduplicate, and cap per source
        combined = pl.concat([j_name, j_domain, j_addr, j_sn, j_t2, j_p4]).unique(subset=["s1_id", "cand_id"])
        capped = combined.sort(["s1_id", "prio"]).group_by("s1_id", maintain_order=True).head(max_src_cands)
        
        for row in capped.iter_rows():
            target_dict[row[0]].append(row[1])

        print(f"    - {src_name} scanned in {time.time()-t_src:.2f}s (Candidates captured: {len(capped):,})", flush=True)

    # 4. Merge S2 and S3 with Natural Sizing
    print("[*] Merging Source 2 and Source 3 candidates...", flush=True)
    out_lines = ["source1_entity_id\tcandidate_entity_ids\n"]
    total_candidates_kept = 0

    for s1_id in all_s1_ids:
        merged = s2_cands.get(s1_id, []) + s3_cands.get(s1_id, [])
        total_candidates_kept += len(merged)
        cand_str = ",".join(merged)
        out_lines.append(f"{s1_id}\t{cand_str}\n")

    # 5. Measure Ground Truth Coverage
    gt = pl.read_csv(gt_file, separator="\t").filter(pl.col("source1_entity_id").is_in(all_s1_ids)).fill_null("")
    gt_dict = {}
    for row in gt.iter_rows(named=True):
        m = row["matched_entity_ids"]
        gt_dict[row["source1_entity_id"]] = set(x.strip() for x in m.split(",") if x.strip()) if m else set()

    total_true = sum(len(v) for v in gt_dict.values())
    recalled = 0
    for line in out_lines[1:]:
        parts = line.strip().split("\t")
        s1 = parts[0]
        cands = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
        recalled += len(gt_dict.get(s1, set()).intersection(set(cands)))

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print("\n" + "=" * 70)
    print("              HIGH-RECALL CANDIDATE POOL SUMMARY")
    print("=" * 70)
    print(f"Total S1 entities processed:  {len(all_s1_ids):,}")
    print(f"Total Candidate Pairs Kept:   {total_candidates_kept:,} (Avg {total_candidates_kept/len(all_s1_ids):.1f} per entity)")
    print(f"Ground Truth Match Recall:    {recalled:,} / {total_true:,} ({recalled/total_true*100:.2f}%)")
    print(f"Output File:                  {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")
    print(f"Total Elapsed Time:           {time.time()-t0:.2f}s")
    print("=" * 70)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=20000)
    parser.add_argument("--max_cands", type=int, default=12)
    parser.add_argument("--output", type=str, default="data/benchmark_cache/high_recall_candidates_20k.tsv")
    args = parser.parse_args()

    generate_high_recall_candidates(
        sample_size=args.sample_size,
        max_cands_per_s1=args.max_cands,
        output_path=args.output
    )
