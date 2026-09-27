#!/usr/bin/env python3
"""
Phase 3 GBDT Re-Ranker Training Pipeline for Amazon ML Challenge 2026.

Workflow:
  1. Loads candidate pair features from `data/parquet/train_features.parquet`.
  2. Uses deterministic MD5 hash splitting (`src.eval.harness.assign_entity_split`)
     to split S1 reference entities into 'train' (fitting) and 'dev' (evaluation/early stopping).
  3. Trains LightGBM / XGBoost GBDT re-ranker on hard negatives.
  4. Scans decision thresholds to optimize competition Macro F_0.5.
  5. Evaluates and prints pair-level Classification Report and entity-level Macro F_0.5.
  6. Saves trained model to `models/gbdt_ranker.model`.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import classification_report

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.harness import assign_entity_split, evaluate_predictions
from src.features import FEATURE_COLUMNS

logger = logging.getLogger("train_ranker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def resolve_file(base_dir: Path, *candidates: str) -> Optional[Path]:
    """Resolves an existing file path from relative candidate paths."""
    for rel in candidates:
        p = base_dir / rel
        if p.exists():
            return p
        root_p = REPO_ROOT / rel
        if root_p.exists():
            return root_p
    return None


def add_deterministic_split(df: pl.DataFrame, id_col: str = "source1_entity_id_int") -> pl.DataFrame:
    """
    Adds a deterministic 'split' column ('train', 'dev', 'val') to DataFrame
    by hashing unique S1 entity IDs using `assign_entity_split`.
    """
    logger.info("Computing deterministic entity splits using MD5 hash partitioning...")
    t0 = time.time()
    unique_s1 = df.select(id_col).unique()
    s1_ids = unique_s1[id_col].to_list()

    split_tags = [assign_entity_split(str(eid)) for eid in s1_ids]
    split_lookup = pl.DataFrame({
        id_col: unique_s1[id_col],
        "split": pl.Series(split_tags, dtype=pl.String),
    })

    result_df = df.join(split_lookup, on=id_col, how="left")
    elapsed = time.time() - t0
    logger.info("Assigned entity splits across %d unique S1 entities in %.2fs", len(s1_ids), elapsed)
    return result_df


def find_optimal_threshold(
    dev_df: pl.DataFrame,
    dev_probs: np.ndarray,
    s1_id_col: str = "source1_entity_id_int",
    cand_id_col: str = "candidate_entity_id_int",
    target_col: str = "target",
    thresholds: Optional[np.ndarray] = None,
    ground_truth_map: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Vectorized grid search to find probability threshold maximizing Macro F_0.5 on dev split.
    """
    if thresholds is None:
        thresholds = np.linspace(0.20, 0.80, 25)

    s1_list = dev_df[s1_id_col].cast(pl.String).to_list()
    cand_list = dev_df[cand_id_col].cast(pl.String).to_list()
    target_list = dev_df[target_col].to_list()

    # Ground truth mapping: s1_id -> set of true candidate IDs
    if ground_truth_map is None:
        gt_map: Dict[str, Set[str]] = {}
        for s1, c, tgt in zip(s1_list, cand_list, target_list):
            if s1 not in gt_map:
                gt_map[s1] = set()
            if tgt == 1:
                gt_map[s1].add(c)
    else:
        gt_map = ground_truth_map

    best_thresh = 0.50
    best_f05 = -1.0
    best_metrics: Dict[str, float] = {}

    logger.info("Scanning %d thresholds from %.2f to %.2f for Macro F0.5 optimization...", len(thresholds), thresholds[0], thresholds[-1])

    for thresh in thresholds:
        # Build predictions dict for current threshold
        pred_map: Dict[str, Set[str]] = {s1: set() for s1 in gt_map.keys()}
        for s1, c, p in zip(s1_list, cand_list, dev_probs):
            if p >= thresh and s1 in pred_map:
                pred_map[s1].add(c)

        metrics = evaluate_predictions(gt_map, pred_map)
        f05 = metrics.get("macro_F05", 0.0)

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(thresh)
            best_metrics = metrics

    return best_thresh, best_f05, best_metrics


def train_gbdt_ranker(
    train_features_path: Path,
    output_model_path: Path,
    gt_path: Optional[Path] = None,
    model_type: str = "lgbm",
    learning_rate: float = 0.05,
    n_estimators: int = 500,
    num_leaves: int = 31,
    max_depth: int = 6,
    early_stopping_rounds: int = 30,
    fixed_threshold: Optional[float] = None,
    sample_size: Optional[int] = None,
    n_jobs: int = -1,
) -> None:
    """
    Main training execution function.
    """
    total_start = time.time()
    logger.info("=" * 70)
    logger.info("AMAZON ML CHALLENGE 2026 - PHASE 3 GBDT RANKER TRAINING")
    logger.info("=" * 70)
    logger.info("Features File : %s", train_features_path.resolve())
    logger.info("Model Target  : %s", output_model_path.resolve())
    logger.info("Model Type    : %s", model_type.upper())

    if not train_features_path.exists():
        raise FileNotFoundError(
            f"Train features file not found: {train_features_path}. "
            f"Please run scripts/04_build_features.py first."
        )

    # 1. Load train features
    logger.info("Loading feature matrix from Parquet...")
    df = pl.read_parquet(train_features_path)
    if sample_size and sample_size < df.height:
        logger.info("Subsetting to first %d rows for fast demonstration...", sample_size)
        df = df.head(sample_size)

    logger.info("Loaded dataset with %d rows, %d columns.", df.height, len(df.columns))

    if "target" not in df.columns:
        raise ValueError("Missing 'target' column in train_features.parquet.")

    # 2. Assign deterministic splits
    s1_id_col = "source1_entity_id_int" if "source1_entity_id_int" in df.columns else "source1_entity_id"
    cand_id_col = "candidate_entity_id_int" if "candidate_entity_id_int" in df.columns else "candidate_entity_id"

    df_split = add_deterministic_split(df, id_col=s1_id_col)

    train_df = df_split.filter(pl.col("split") == "train")
    dev_df = df_split.filter(pl.col("split") == "dev")

    logger.info(
        "Train split : %d pairs (Positives: %d, Negatives: %d)",
        train_df.height,
        train_df.filter(pl.col("target") == 1).height,
        train_df.filter(pl.col("target") == 0).height,
    )
    logger.info(
        "Dev split   : %d pairs (Positives: %d, Negatives: %d)",
        dev_df.height,
        dev_df.filter(pl.col("target") == 1).height,
        dev_df.filter(pl.col("target") == 0).height,
    )

    if train_df.height == 0 or dev_df.height == 0:
        raise RuntimeError("Train or Dev split is empty. Check split configuration and sample size.")

    # 3. Resolve active feature columns
    active_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not active_features:
        excluded = {s1_id_col, cand_id_col, "target", "split"}
        active_features = [c for c in df.columns if c not in excluded]

    logger.info("Training features (%d): %s", len(active_features), ", ".join(active_features))

    X_train = train_df.select(active_features).to_numpy()
    y_train = train_df["target"].to_numpy().astype(np.int32)

    X_dev = dev_df.select(active_features).to_numpy()
    y_dev = dev_df["target"].to_numpy().astype(np.int32)

    # 4. Train Model
    t_train = time.time()
    num_threads = os.cpu_count() or 4 if n_jobs <= 0 else n_jobs

    if model_type == "lgbm":
        import lightgbm as lgb

        logger.info("Initializing LightGBM Binary Classifier (logloss / GBDT)...")
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            boosting_type="gbdt",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=num_threads,
            verbose=-1,
        )

        callbacks = []
        if early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True))

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_dev, y_dev)],
            eval_names=["dev"],
            callbacks=callbacks,
        )

        dev_probs = model.predict_proba(X_dev)[:, 1]

    elif model_type == "xgb":
        import xgboost as xgb

        logger.info("Initializing XGBoost Binary Classifier...")
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=num_threads,
            early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_dev, y_dev)],
            verbose=True,
        )
        dev_probs = model.predict_proba(X_dev)[:, 1]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info("Model fitting completed in %.2fs", time.time() - t_train)

    # 5. Full Ground Truth Evaluation (if GT file available)
    gt_map = None
    if gt_path and gt_path.exists():
        logger.info("Loading ground truth file for full dev evaluation: %s", gt_path.name)
        try:
            if gt_path.suffix == ".parquet":
                full_gt = pl.read_parquet(gt_path)
            else:
                full_gt = pl.read_csv(gt_path, separator="\t", infer_schema_length=5000)

            s1_raw_col = "source1_entity_id" if "source1_entity_id" in full_gt.columns else "entity_id"
            if "matched_entity_ids" in full_gt.columns:
                dev_gt = add_deterministic_split(full_gt, id_col=s1_raw_col).filter(pl.col("split") == "dev")

                gt_map = {}
                for row in dev_gt.iter_rows(named=True):
                    s1_id_str = str(row[s1_raw_col])
                    matches_str = row["matched_entity_ids"] or ""
                    gt_map[s1_id_str] = {x.strip() for x in matches_str.split(",") if x.strip()}
        except Exception as e:
            logger.warning("Failed to load or parse full dev ground truth: %s", e)
            gt_map = None

    # 6. Optimize Probability Threshold for Macro F_0.5
    if fixed_threshold is not None:
        best_threshold = fixed_threshold
        logger.info("Using fixed probability threshold: %.4f", best_threshold)
        pred_map = {s1: set() for s1 in (gt_map.keys() if gt_map else dev_df[s1_id_col].cast(pl.String).unique().to_list())}
        for s1, c, p in zip(dev_df[s1_id_col].cast(pl.String).to_list(), dev_df[cand_id_col].cast(pl.String).to_list(), dev_probs):
            if p >= best_threshold and s1 in pred_map:
                pred_map[s1].add(c)
        if gt_map:
            best_metrics = evaluate_predictions(gt_map, pred_map)
        else:
            _, _, best_metrics = find_optimal_threshold(
                dev_df, dev_probs, s1_id_col, cand_id_col, thresholds=np.array([best_threshold])
            )
        best_f05 = best_metrics.get("macro_F05", 0.0)
    else:
        best_threshold, best_f05, best_metrics = find_optimal_threshold(
            dev_df=dev_df,
            dev_probs=dev_probs,
            s1_id_col=s1_id_col,
            cand_id_col=cand_id_col,
            target_col="target",
            ground_truth_map=gt_map,
        )

    # 7. Print Classification Report & Metrics
    y_dev_binary = (dev_probs >= best_threshold).astype(int)
    print("\n" + "=" * 70)
    print(f"DEV SET PAIR-LEVEL CLASSIFICATION REPORT (Threshold = {best_threshold:.4f}):")
    print("=" * 70)
    print(classification_report(y_dev, y_dev_binary, target_names=["Hard Negative (0)", "GT Match (1)"], digits=4))

    print("=" * 70)
    print("DEV SET COMPETITION MACRO F_0.5 EVALUATION SCOREBOARD:")
    print("=" * 70)
    print(f"  Macro F0.5 Score           : {best_f05:.5f}")
    print(f"  Ceiling F0.5 Score         : {best_metrics.get('ceiling_F05', 0.0):.5f}")
    print(f"  Precision (Macro)          : {best_metrics.get('precision_macro', 0.0):.5f}")
    print(f"  Recall (Macro)             : {best_metrics.get('recall_macro', 0.0):.5f}")
    print(f"  Singleton Accuracy         : {best_metrics.get('singleton_accuracy', 0.0):.5f}")
    print(f"  Candidate Recall Ceiling   : {best_metrics.get('candidate_recall_ceiling', 0.0):.5f}")
    print(f"  Total Evaluated Entities   : {best_metrics.get('total_entities', dev_df[s1_id_col].n_unique()):,}")
    print("=" * 70)

    # 8. Print Feature Importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("\nFEATURE IMPORTANCE RANKING:")
        for rank, idx in enumerate(sorted_idx, start=1):
            feat = active_features[idx]
            imp = importances[idx]
            print(f"  {rank:2d}. {feat:<28} : {imp:,.1f}")
        print("=" * 70)

    # 9. Save Model and Metadata
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving trained model to: %s", output_model_path.resolve())

    if model_type == "lgbm":
        model.booster_.save_model(str(output_model_path))
    elif model_type == "xgb":
        model.save_model(str(output_model_path))

    # Also save metadata JSON
    meta_path = output_model_path.with_suffix(".json")
    metadata = {
        "model_type": model_type,
        "feature_names": active_features,
        "best_threshold": float(best_threshold),
        "macro_f05": float(best_f05),
        "metrics": best_metrics,
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_rows": train_df.height,
        "dev_rows": dev_df.height,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata configuration to: %s", meta_path.name)
    logger.info("Entire pipeline completed successfully in %.2fs.", time.time() - total_start)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 GBDT Re-Ranker Training Pipeline for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=REPO_ROOT / "data/parquet/train_features.parquet",
        help="Path to input train_features.parquet (default: data/parquet/train_features.parquet)",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=REPO_ROOT / "models/gbdt_ranker.model",
        help="Path to save output GBDT model (default: models/gbdt_ranker.model)",
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=REPO_ROOT / "data/parquet/train/train_ground_truth.parquet",
        help="Optional path to ground truth table for comprehensive dev evaluation (includes singletons)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lgbm",
        choices=["lgbm", "xgb"],
        help="GBDT model architecture (default: lgbm)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for boosting (default: 0.05)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Maximum boosting iterations (default: 500)",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="Maximum tree leaves for LightGBM (default: 31)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: 6)",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Early stopping rounds on dev split (default: 30, set 0 to disable)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed probability threshold (default: auto-tune for max Macro F0.5)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row limit on train features for fast testing",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of threads (-1 for all cores, default: -1)",
    )
    args = parser.parse_args()

    train_gbdt_ranker(
        train_features_path=args.features_path,
        output_model_path=args.output_model,
        gt_path=args.gt_path,
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        early_stopping_rounds=args.early_stopping_rounds,
        fixed_threshold=args.threshold,
        sample_size=args.sample_size,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
