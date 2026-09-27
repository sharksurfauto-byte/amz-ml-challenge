#!/usr/bin/env python3
"""
Phase 4 Inference Script for Amazon ML Challenge 2026.

Workflow:
1. Loads `test_features.parquet` and LightGBM model from `models/gbdt_ranker.model`.
2. Evaluates predictions in chunks (memory safe).
3. Caps candidates to top 18 per source1_entity_id to manage file size.
4. Applies strict Mutual Exclusion Constraint (candidate goes to highest-probability S1).
5. Thresholds probabilities to create final matches (at most 11 per S1).
6. Generates valid test outputs compliant with the evaluation rules.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb


# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger("inference")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Inference Pipeline")
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "data/parquet/test_features.parquet")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models/gbdt_ranker.model")
    parser.add_argument("--id-map", type=Path, default=REPO_ROOT / "data/parquet/id_map.parquet")
    parser.add_argument("--test-s1", type=Path, default=REPO_ROOT / "data/test/test_source1.tsv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold from metadata")
    args = parser.parse_args()

    # 1. Load Metadata
    meta_path = args.model.with_suffix(".json")
    if not meta_path.exists():
        logger.error(f"Metadata file not found: {meta_path}")
        sys.exit(1)
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    feature_names = meta.get("feature_names", [])
    threshold = args.threshold if args.threshold is not None else meta.get("best_threshold", 0.50)
    
    logger.info(f"Loaded {len(feature_names)} features and threshold={threshold:.4f} from metadata")

    # 2. Load Model
    logger.info(f"Loading LightGBM model from {args.model}")
    model = lgb.Booster(model_file=str(args.model))

    # 3. Process Features in Chunks
    logger.info(f"Loading test features from {args.features}")
    t0 = time.time()
    all_features_df = pl.read_parquet(args.features)
    logger.info(f"Loaded {all_features_df.height} candidate pairs in {time.time() - t0:.2f}s")
    
    # Ensure features exist
    missing_feats = [f for f in feature_names if f not in all_features_df.columns]
    if missing_feats:
        logger.error(f"Missing features in parquet: {missing_feats}")
        sys.exit(1)
        
    X = all_features_df.select(feature_names).to_numpy()
    
    logger.info("Predicting probabilities...")
    t1 = time.time()
    chunk_sz = 1_000_000
    preds = []
    for i in range(0, X.shape[0], chunk_sz):
        preds.append(model.predict(X[i:i + chunk_sz]))
    y_pred = np.concatenate(preds)
    logger.info(f"Predictions completed in {time.time() - t1:.2f}s")
    del X
    
    logger.info("Processing predictions and constraints...")
    all_preds = all_features_df.select([
        "source1_entity_id_int", 
        "candidate_entity_id_int"
    ]).with_columns(pl.Series("y_pred", y_pred, dtype=pl.Float32))
    del all_features_df
    del y_pred
    
    # Step A: Cap candidates at 18 per S1 to guarantee small file sizes
    all_preds = all_preds.sort(["source1_entity_id_int", "y_pred"], descending=[False, True])
    all_preds = all_preds.group_by("source1_entity_id_int").head(18)
    
    # Step B: Apply Mutual Exclusion Constraint globally on the capped candidates
    all_preds = all_preds.sort("y_pred", descending=True)
    cands_me = all_preds.unique(subset=["candidate_entity_id_int"], keep="first")
    
    # Step C: Filter for final matches
    matches_me = cands_me.filter(pl.col("y_pred") >= threshold)
    
    # 4. Map IDs back to strings
    logger.info("Mapping integer IDs to strings...")
    id_map = pl.read_parquet(args.id_map)
    
    def map_to_strings(df: pl.DataFrame) -> pl.DataFrame:
        df = df.join(id_map, left_on="source1_entity_id_int", right_on="entity_id_int", how="left")
        df = df.rename({"entity_id": "source1_entity_id"}).drop("source1_entity_id_int")
        df = df.join(id_map, left_on="candidate_entity_id_int", right_on="entity_id_int", how="left")
        df = df.rename({"entity_id": "candidate_entity_id"}).drop("candidate_entity_id_int")
        return df

    cands_str = map_to_strings(cands_me)
    matches_str = map_to_strings(matches_me)
    
    # 5. Group and Format
    logger.info("Grouping and formatting final outputs...")
    
    def group_outputs(df: pl.DataFrame, col_name: str, max_items: int) -> pl.DataFrame:
        df = df.sort(["source1_entity_id", "y_pred"], descending=[False, True])
        return (
            df.group_by("source1_entity_id")
            .agg(pl.col("candidate_entity_id").head(max_items).alias("candidates_list"))
            .with_columns(pl.col("candidates_list").list.join(","))
            .rename({"candidates_list": col_name})
            .select(["source1_entity_id", col_name])
        )

    cands_grouped = group_outputs(cands_str, "candidate_entity_ids", 18)
    matches_grouped = group_outputs(matches_str, "matched_entity_ids", 11)
    
    # 6. Join with Full S1 test set to ensure exact row counts
    logger.info("Joining with Full S1 list to satisfy competition format...")
    test_s1 = pl.read_csv(args.test_s1, separator="\t", infer_schema_length=5000, null_values=["", "NULL", "null", "None"])
    s1_col = "source1_entity_id" if "source1_entity_id" in test_s1.columns else "entity_id"
    full_s1 = test_s1.select(pl.col(s1_col).alias("source1_entity_id")).with_row_index("row_nr")
    
    candidate_pairs = full_s1.join(cands_grouped, on="source1_entity_id", how="left")
    candidate_pairs = candidate_pairs.sort("row_nr").drop("row_nr")
    candidate_pairs = candidate_pairs.with_columns(pl.col("candidate_entity_ids").fill_null(""))
    
    matching_results = full_s1.join(matches_grouped, on="source1_entity_id", how="left")
    matching_results = matching_results.sort("row_nr").drop("row_nr")
    matching_results = matching_results.with_columns(pl.col("matched_entity_ids").fill_null(""))

    # 7. Write to Disk
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_cands_path = args.out_dir / "candidate_pairs.tsv"
    out_matches_path = args.out_dir / "matching_results.tsv"
    
    logger.info("Writing TSV files...")
    candidate_pairs.write_csv(out_cands_path, separator="\t", quote_style="never")
    matching_results.write_csv(out_matches_path, separator="\t", quote_style="never")
    
    logger.info(f"Done! Evaluated test setup successfully.")
    logger.info(f"Run 'python scripts/validate_submission.py --matching {out_matches_path} --candidate {out_cands_path} --test-dir data/test' to verify format.")

if __name__ == "__main__":
    main()
