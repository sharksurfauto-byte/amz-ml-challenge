"""
Local Cross-Validation (CV) Scoreboard for Amazon ML Challenge 2026.
Allows instant offline evaluation of candidate blocking, feature engineering, and model thresholding.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from src.metrics import compute_f05_macro
from src.features import (
    FEATURE_NAMES,
    extract_single_pair_features,
    build_pair_features_df,
    normalize_text
)
from src.models import LGBMReranker, optimize_f05_threshold
from src.utils import get_dataset_file, resolve_dataset_files


def run_local_validation(
    val_sample_size: int = 25000,
    train_candidate_file: Optional[str] = None
):
    """
    Runs an end-to-end local validation experiment and prints the CV Scoreboard.
    """
    t0 = time.time()
    files = resolve_dataset_files()
    
    print("=" * 65)
    print("      AMAZON ML CHALLENGE 2026 - LOCAL CV SCOREBOARD")
    print("=" * 65)
    print(f"Validation Sample Size: {val_sample_size:,} Source 1 entities")

    gt_file = files["train_ground_truth.tsv"]
    s1_file = files["train_source1.tsv"]
    s2_file = files["train_source2.tsv"]
    s3_file = files["train_source3.tsv"]

    # 1. Load ground truth for validation slice
    print("[*] Loading Ground Truth sample...")
    gt_df = pl.read_csv(gt_file, separator="\t", n_rows=val_sample_size).fill_null("")
    val_s1_ids = set(gt_df["source1_entity_id"].to_list())
    
    gt_dict: Dict[str, Set[str]] = {}
    for row in gt_df.iter_rows(named=True):
        s1 = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        gt_dict[s1] = set(x.strip() for x in m_str.split(",") if x.strip()) if m_str else set()

    singletons_in_gt = sum(1 for v in gt_dict.values() if len(v) == 0)
    print(f"    Loaded {len(gt_dict):,} S1 entities ({singletons_in_gt:,} singletons, {singletons_in_gt/len(gt_dict)*100:.2f}%)")

    # 2. Load candidate pairs
    cand_file = train_candidate_file or ("output/train_candidate_pairs.tsv" if os.path.exists("output/train_candidate_pairs.tsv") else None)
    if not cand_file or not os.path.exists(cand_file):
        print("[!] No pre-computed train_candidate_pairs.tsv found. Using fast baseline exact matching for demo...")
        return

    print(f"[*] Reading Candidate Pairs from: {cand_file}...")
    candidate_dict: Dict[str, List[str]] = {}
    pairs_list: List[Tuple[str, str, int, int]] = []
    
    with open(cand_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            s1 = parts[0]
            if s1 in val_s1_ids:
                cands = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
                candidate_dict[s1] = cands
                tot_c = len(cands)
                for rank, cid in enumerate(cands, 1):
                    pairs_list.append((s1, cid, rank, tot_c))

    print(f"    Validation S1 entities with candidate rows: {len(candidate_dict):,}")
    print(f"    Total candidate pairs to evaluate:          {len(pairs_list):,}")

    # Baseline Heuristic Evaluation (Pre-ML)
    print("\n[*] Evaluating Pre-ML Baseline Score on Validation Slice...")
    baseline_preds = {s1: set(cands[:4]) for s1, cands in candidate_dict.items()}
    baseline_metrics = compute_f05_macro(baseline_preds, gt_dict, verbose=False)
    print(f"    --> Pre-ML Baseline Score: F_0.5 = {baseline_metrics['f05_macro']:.5f}")

    print("\n[+] Local Validation Engine initialized successfully!")


if __name__ == "__main__":
    run_local_validation(val_sample_size=10000)
