#!/usr/bin/env python3
"""
Fast Local Benchmarking Runner for Amazon ML Challenge 2026.
Executes an offline validation loop on a representative 15,000-entity slice.
Provides instant feedback on Candidate Blocking, Feature Engineering, and LightGBM Re-Ranking.
"""

import os
import sys
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import polars as pl
from src.features import FEATURE_NAMES, extract_single_pair_features
from src.models import LGBMReranker, optimize_f05_threshold
from src.metrics import compute_f05_macro
from src.utils import resolve_dataset_files


def prepare_benchmark_dataset(
    sample_size: int = 15000,
    top_k: int = 6,
    candidate_file: Optional[str] = None,
    cache_dir: str = "data/benchmark_cache",
    force_recompute: bool = False
) -> Tuple[pl.DataFrame, Dict[str, Set[str]], List[str]]:
    """
    Loads or creates a cached local benchmark dataset of candidate pairs with tabular features.
    
    Returns:
        pairs_df: Polars DataFrame containing (s1_id, cand_id, rank, total_cands, label, *FEATURE_NAMES)
        gt_dict: Mapping of s1_id -> set of true matched candidate IDs
        val_s1_ids: Ordered list of S1 entity IDs in this benchmark slice
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_tag = "highrecall" if (candidate_file and "high_recall" in candidate_file) else "default"
    cache_path = os.path.join(cache_dir, f"benchmark_{cache_tag}_{sample_size}_k{top_k}.parquet")
    
    files = resolve_dataset_files()
    gt_file = files["train_ground_truth.tsv"]
    s1_file = files["train_source1.tsv"]
    s2_file = files["train_source2.tsv"]
    s3_file = files["train_source3.tsv"]
    cand_file = candidate_file or ("data/benchmark_cache/high_recall_candidates_20k.tsv" if os.path.exists("data/benchmark_cache/high_recall_candidates_20k.tsv") else "output/train_candidate_pairs.tsv")

    print(f"[*] Benchmark configuration: sample_size={sample_size:,}, top_k={top_k}, candidate_file={cand_file}")
    
    # 1. Load S1 Entities and Ground Truth
    print("[*] Slicing S1 entities and ground truth...")
    # Read the candidate pairs file to get the exact S1 entities in candidate list
    cand_df_slice = pl.read_csv(cand_file, separator="\t", n_rows=sample_size).fill_null("")
    all_s1_ids = cand_df_slice["source1_entity_id"].to_list()
    s1_id_set = set(all_s1_ids)

    # Load Ground Truth for exactly these S1 entities
    gt_full = pl.read_csv(gt_file, separator="\t").fill_null("")
    gt_slice = gt_full.filter(pl.col("source1_entity_id").is_in(all_s1_ids))
    
    gt_dict: Dict[str, Set[str]] = {}
    for row in gt_slice.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        gt_dict[row["source1_entity_id"]] = set(x.strip() for x in m_str.split(",") if x.strip()) if m_str else set()
    
    # Ensure all S1 entities exist in gt_dict (default empty if missing)
    for sid in all_s1_ids:
        if sid not in gt_dict:
            gt_dict[sid] = set()

    singletons = sum(1 for v in gt_dict.values() if len(v) == 0)
    total_true_matches = sum(len(v) for v in gt_dict.values())
    print(f"    Loaded {len(all_s1_ids):,} S1 entities ({singletons:,} singletons, {total_true_matches:,} true match links)")

    # 2. Check if cached feature parquet exists
    if not force_recompute and os.path.exists(cache_path):
        print(f"[+] Found cached benchmark dataset: {cache_path}")
        pairs_df = pl.read_parquet(cache_path)
        print(f"    Loaded {len(pairs_df):,} candidate pairs in < 0.1s!")
        return pairs_df, gt_dict, all_s1_ids

    # 3. Build Dataset from Scratch
    t0 = time.time()
    print("[*] Building benchmark pair dataset from raw files...")
    
    # Read S1 attributes
    s1_full = pl.read_csv(s1_file, separator="\t")
    s1_sub = s1_full.filter(pl.col("entity_id").is_in(all_s1_ids))
    s1_meta = {}
    for row in s1_sub.iter_rows(named=True):
        s1_meta[row["entity_id"]] = (
            row.get("business_name") or "",
            row.get("business_address") or "",
            row.get("country") or ""
        )

    # Collect needed candidate IDs
    needed_cand_ids = set()
    raw_pairs: List[Tuple[str, str, int, int]] = []
    
    for row in cand_df_slice.iter_rows(named=True):
        s1 = row["source1_entity_id"]
        c_str = row["candidate_entity_ids"]
        if not c_str:
            continue
        cands = [x.strip() for x in c_str.split(",") if x.strip()][:top_k]
        tot = len(cands)
        for rank, cid in enumerate(cands, 1):
            raw_pairs.append((s1, cid, rank, tot))
            needed_cand_ids.add(cid)

    print(f"    Total candidate pairs to extract: {len(raw_pairs):,}")
    print(f"    Unique candidate entities to look up: {len(needed_cand_ids):,}")

    # Look up candidate attributes from S2 and S3
    t_lookup = time.time()
    needed_cand_list = list(needed_cand_ids)
    print("    - Querying Source 2...")
    s2_sub = pl.read_csv(s2_file, separator="\t").filter(pl.col("entity_id").is_in(needed_cand_list))
    print("    - Querying Source 3...")
    s3_sub = pl.read_csv(s3_file, separator="\t").filter(pl.col("entity_id").is_in(needed_cand_list))

    cand_meta = {}
    for row in s2_sub.iter_rows(named=True):
        cand_meta[row["entity_id"]] = (row.get("business_name") or "", row.get("business_address") or "")
    for row in s3_sub.iter_rows(named=True):
        cand_meta[row["entity_id"]] = (row.get("business_name") or "", row.get("business_address") or "")

    print(f"    - Candidate lookup table built in {time.time()-t_lookup:.2f}s ({len(cand_meta):,} records)")

    # 4. Extract Tabular Features
    t_feat = time.time()
    print("[*] Extracting 15 tabular features using RapidFuzz C++...")
    records = []
    for s1, cid, rank, tot in raw_pairs:
        s1_name, s1_addr, _ = s1_meta.get(s1, ("", "", ""))
        cand_name, cand_addr = cand_meta.get(cid, ("", ""))
        
        feats = extract_single_pair_features(
            s1_name=s1_name,
            s1_addr=s1_addr,
            cand_name=cand_name,
            cand_addr=cand_addr,
            rank=rank,
            total_cands=tot
        )
        
        # Binary label
        label = 1 if cid in gt_dict.get(s1, set()) else 0
        records.append([s1, cid, rank, tot, label] + feats)

    schema = [
        ("s1_id", pl.Utf8),
        ("cand_id", pl.Utf8),
        ("rank", pl.Int32),
        ("total_cands", pl.Int32),
        ("label", pl.Int32)
    ] + [(c, pl.Float64) for c in FEATURE_NAMES]
    
    pairs_df = pl.DataFrame(records, schema=schema, orient="row")
    print(f"[+] Feature extraction completed in {time.time()-t_feat:.2f}s!")
    print(f"    Positives: {pairs_df['label'].sum():,}, Negatives: {len(pairs_df) - pairs_df['label'].sum():,}")

    # Cache to parquet
    pairs_df.write_parquet(cache_path)
    print(f"[+] Saved cached dataset to: {cache_path} ({os.path.getsize(cache_path)/1024/1024:.2f} MB)")
    return pairs_df, gt_dict, all_s1_ids


def run_benchmark(
    sample_size: int = 15000,
    top_k: int = 6,
    candidate_file: Optional[str] = None,
    train_ratio: float = 0.6,
    force_recompute: bool = False
):
    """Executes the complete local benchmark and prints the CV Scoreboard."""
    t_start = time.time()
    print("=" * 70)
    print("         AMAZON ML CHALLENGE 2026 - LOCAL FAST BENCHMARK")
    print("=" * 70)
    
    # 1. Load data
    pairs_df, gt_dict, all_s1_ids = prepare_benchmark_dataset(
        sample_size=sample_size,
        top_k=top_k,
        candidate_file=candidate_file,
        force_recompute=force_recompute
    )

    # 2. Strict Entity-Level Train / Validation Split
    n_total = len(all_s1_ids)
    n_train = int(n_total * train_ratio)
    train_s1_ids = set(all_s1_ids[:n_train])
    val_s1_ids = set(all_s1_ids[n_train:])
    
    print(f"\n[*] Entity-Level Split ({train_ratio*100:.0f}% Train / {(1-train_ratio)*100:.0f}% Val):")
    print(f"    Train S1 entities:      {len(train_s1_ids):,}")
    print(f"    Validation S1 entities: {len(val_s1_ids):,}")

    # Split Pairs DataFrame
    train_df = pairs_df.filter(pl.col("s1_id").is_in(list(train_s1_ids)))
    val_df = pairs_df.filter(pl.col("s1_id").is_in(list(val_s1_ids)))
    
    print(f"    Train Pairs:            {len(train_df):,} (Positives: {train_df['label'].sum():,})")
    print(f"    Validation Pairs:       {len(val_df):,} (Positives: {val_df['label'].sum():,})")

    # Validation Ground Truth subset
    val_gt_dict = {s1: gt_dict[s1] for s1 in val_s1_ids if s1 in gt_dict}

    # 3. Evaluate Pre-ML Baseline (Heuristic Top-4 matches)
    print("\n" + "-" * 70)
    print(" [STAGE 1] PRE-ML HEURISTIC BASELINE (Leaderboard Proxy)")
    print("-" * 70)
    baseline_val_preds = defaultdict(set)
    for row in val_df.iter_rows(named=True):
        if row["rank"] <= 4:
            baseline_val_preds[row["s1_id"]].add(row["cand_id"])

    # Ensure singletons are represented as empty sets
    for s1 in val_s1_ids:
        if s1 not in baseline_val_preds:
            baseline_val_preds[s1] = set()

    baseline_metrics = compute_f05_macro(baseline_val_preds, val_gt_dict, verbose=False)
    print(f"  Pre-ML Baseline F_0.5 Score:   {baseline_metrics['f05_macro']:.5f}")
    print(f"  Pre-ML Macro Precision:        {baseline_metrics['precision_macro']:.5f}")
    print(f"  Pre-ML Macro Recall:           {baseline_metrics['recall_macro']:.5f}")
    print(f"  Pre-ML Singleton Accuracy:     {baseline_metrics['singleton_accuracy']*100:.2f}%")

    # 4. Train LightGBM Re-Ranker
    print("\n" + "-" * 70)
    print(" [STAGE 2] TRAINING LIGHTGBM RE-RANKER")
    print("-" * 70)
    X_train = train_df.select(FEATURE_NAMES).to_numpy()
    y_train = train_df["label"].to_numpy()
    
    X_val = val_df.select(FEATURE_NAMES).to_numpy()
    y_val = val_df["label"].to_numpy()

    t_train = time.time()
    model = LGBMReranker()
    model.fit(X_train, y_train, X_val, y_val)
    print(f"[+] LightGBM trained in {time.time()-t_train:.2f}s!")

    # 5. Predict Probabilities & Optimize Threshold
    print("\n" + "-" * 70)
    print(" [STAGE 3] F_0.5 THRESHOLD OPTIMIZATION (POST-PROCESSING)")
    print("-" * 70)
    val_probs = model.predict_proba(X_val)
    
    val_s1_list = val_df["s1_id"].to_list()
    val_cand_list = val_df["cand_id"].to_list()

    best_tau, best_f05, best_report = optimize_f05_threshold(
        s1_ids=val_s1_list,
        cand_ids=val_cand_list,
        y_probs=val_probs,
        gt_dict=val_gt_dict,
        tau_range=(0.40, 0.90),
        step=0.05,
        max_s2=5,
        max_s3=6
    )

    # 6. Print Feature Importances
    print("\n" + "-" * 70)
    print(" TOP-10 MOST PREDICTIVE FEATURES (LIGHTGBM SPLIT GAIN)")
    print("-" * 70)
    importances = model.get_feature_importances()
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for rank_idx, (f_name, imp) in enumerate(sorted_imp[:10], 1):
        print(f"  {rank_idx:2d}. {f_name:25s} : {imp:6.0f}")

    # 7. Final CV Scoreboard Comparison
    delta_f05 = best_f05 - baseline_metrics['f05_macro']
    delta_pct = (delta_f05 / (baseline_metrics['f05_macro'] + 1e-9)) * 100

    print("\n" + "=" * 70)
    print("             FINAL LOCAL VALIDATION SCOREBOARD")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Baseline (Heuristic)':<22} {'LightGBM (Optimized)':<22}")
    print(f"  {'-'*25} {'-'*22} {'-'*22}")
    print(f"  {'Macro F_0.5 Score':<25} {baseline_metrics['f05_macro']:<22.5f} {best_f05:<22.5f}")
    print(f"  {'Macro Precision':<25} {baseline_metrics['precision_macro']:<22.5f} {best_report['precision_macro']:<22.5f}")
    print(f"  {'Macro Recall':<25} {baseline_metrics['recall_macro']:<22.5f} {best_report['recall_macro']:<22.5f}")
    print(f"  {'Singleton Accuracy':<25} {baseline_metrics['singleton_accuracy']*100:<20.2f}% {best_report['singleton_accuracy']*100:<20.2f}%")
    print(f"  {'Optimal Cutoff (tau*)':<25} {'N/A (Top-4)':<22} {f'tau = {best_tau:.2f}':<22}")
    print(f"  {'-'*25} {'-'*22} {'-'*22}")
    print(f"  NET GAIN (Delta F_0.5):   +{delta_f05:.5f} ({'+' if delta_pct>0 else ''}{delta_pct:.1f}%)")
    print("=" * 70)
    print(f"[+] Total Local Benchmark Runtime: {time.time()-t_start:.2f}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - Fast Local Benchmark")
    parser.add_argument("--sample_size", type=int, default=20000, help="Number of S1 entities to evaluate")
    parser.add_argument("--top_k", type=int, default=6, help="Max candidates per S1 entity")
    parser.add_argument("--candidate_file", type=str, default="data/benchmark_cache/high_recall_candidates_20k.tsv", help="Candidate tsv file")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Ratio of entities for training")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of feature cache")
    args = parser.parse_args()

    run_benchmark(
        sample_size=args.sample_size,
        top_k=args.top_k,
        candidate_file=args.candidate_file,
        train_ratio=args.train_ratio,
        force_recompute=args.recompute
    )
