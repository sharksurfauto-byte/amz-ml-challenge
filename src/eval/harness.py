#!/usr/bin/env python3
"""
Evaluation Harness and Deterministic Splitter for Amazon ML Challenge 2026.

Features:
- Fast computation of Macro-Averaged F_0.5 score per official competition guidelines.
- Computes ceiling_F05: the theoretical maximum F_0.5 achievable by an oracle selector
  from the candidate pool (or predictions).
- Deterministic train/dev/val splitting based on stable hashing of entity_id.
- Supports Polars, Pandas, dictionaries of sets, or file paths as inputs.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# =============================================================================
# 1. Metric Calculations
# =============================================================================

def compute_f05_single(
    pred_set: Union[Set[str], Iterable[str], str, None],
    gt_set: Union[Set[str], Iterable[str], str, None],
) -> Tuple[float, float, float]:
    """
    Computes (f05, precision, recall) for a single Source 1 entity.

    Rules:
    - If |G| == 0 and |P| == 0: F_0.5 = 1.0 (True Negative singleton)
    - If |G| == 0 and |P| > 0: F_0.5 = 0.0 (False Positive merge on singleton)
    - If |G| > 0 and |P| == 0: F_0.5 = 0.0 (Missed entity)
    - If |P ∩ G| == 0: F_0.5 = 0.0
    - Otherwise:
        P = |P ∩ G| / |P|
        R = |P ∩ G| / |G|
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    # Normalize prediction set
    if isinstance(pred_set, str):
        p = {x.strip() for x in pred_set.split(",") if x.strip()} if pred_set.strip() else set()
    elif isinstance(pred_set, (set, frozenset)):
        p = pred_set
    elif pred_set is None:
        p = set()
    else:
        p = {str(x).strip() for x in pred_set if str(x).strip()}

    # Normalize ground truth set
    if isinstance(gt_set, str):
        g = {x.strip() for x in gt_set.split(",") if x.strip()} if gt_set.strip() else set()
    elif isinstance(gt_set, (set, frozenset)):
        g = gt_set
    elif gt_set is None:
        g = set()
    else:
        g = {str(x).strip() for x in gt_set if str(x).strip()}

    len_p = len(p)
    len_g = len(g)

    # Singleton case
    if len_g == 0:
        return (1.0, 1.0, 1.0) if len_p == 0 else (0.0, 0.0, 1.0)

    # Empty prediction on non-singleton
    if len_p == 0:
        return 0.0, 1.0, 0.0

    # Overlap
    tp = len(p.intersection(g))
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len_p
    recall = tp / len_g

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0, precision, recall

    f05 = (1.25 * precision * recall) / denom
    return f05, precision, recall


def compute_ceiling_f05_single(
    cand_set: Union[Set[str], Iterable[str], str, None],
    gt_set: Union[Set[str], Iterable[str], str, None],
) -> float:
    """
    Computes the ceiling F_0.5 achievable by an oracle selector given candidate set C.
    Oracle selects P* = C ∩ G (all captured true positives, zero false positives):
    - If |G| == 0: Oracle selects empty set P* = Ø -> F_0.5 = 1.0.
    - If |G| > 0:
        - If |C ∩ G| == 0: F_0.5 = 0.0
        - If |C ∩ G| > 0: Precision is 1.0, Recall = |C ∩ G| / |G|
          F_0.5 = (1.25 * 1.0 * Recall) / (0.25 * 1.0 + Recall)
    """
    if isinstance(cand_set, str):
        c = {x.strip() for x in cand_set.split(",") if x.strip()} if cand_set.strip() else set()
    elif isinstance(cand_set, (set, frozenset)):
        c = cand_set
    elif cand_set is None:
        c = set()
    else:
        c = {str(x).strip() for x in cand_set if str(x).strip()}

    if isinstance(gt_set, str):
        g = {x.strip() for x in gt_set.split(",") if x.strip()} if gt_set.strip() else set()
    elif isinstance(gt_set, (set, frozenset)):
        g = gt_set
    elif gt_set is None:
        g = set()
    else:
        g = {str(x).strip() for x in gt_set if str(x).strip()}

    len_g = len(g)

    # Oracle can always achieve 1.0 on singletons by predicting empty
    if len_g == 0:
        return 1.0

    if not c:
        return 0.0

    tp = len(c.intersection(g))
    if tp == 0:
        return 0.0

    # With zero false positives, precision is 1.0
    recall = tp / len_g
    denom = 0.25 + recall
    return (1.25 * recall) / denom if denom > 0 else 0.0


def _parse_entity_map(
    data: Any,
    default_id_col: str,
    default_val_col: str,
) -> Dict[str, Set[str]]:
    """
    Converts DataFrame (Polars/Pandas), TSV/Parquet file path, or dict into Dict[str, Set[str]].
    """
    if isinstance(data, dict):
        return {
            str(k): (
                v if isinstance(v, (set, frozenset))
                else {x.strip() for x in v.split(",") if x.strip()} if isinstance(v, str)
                else {str(x).strip() for x in v if str(x).strip()}
            )
            for k, v in data.items()
        }

    # If file path
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if path.suffix == ".parquet" and HAS_POLARS:
            df = pl.read_parquet(path)
            return _parse_entity_map(df, default_id_col, default_val_col)
        elif HAS_POLARS:
            df = pl.read_csv(path, separator="\t").fill_null("")
            return _parse_entity_map(df, default_id_col, default_val_col)
        else:
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            return _parse_entity_map(df, default_id_col, default_val_col)

    # Polars DataFrame
    if HAS_POLARS and isinstance(data, pl.DataFrame):
        cols = data.columns
        id_col = default_id_col if default_id_col in cols else ("source1_entity_id" if "source1_entity_id" in cols else cols[0])
        val_col = default_val_col if default_val_col in cols else ("matched_entity_ids" if "matched_entity_ids" in cols else ("candidate_entity_ids" if "candidate_entity_ids" in cols else cols[1]))

        id_series = data.get_column(id_col).cast(pl.String).to_list()
        val_series = data.get_column(val_col).fill_null("").cast(pl.String).to_list()

        result = {}
        for s1, m_str in zip(id_series, val_series):
            result[s1] = {x.strip() for x in m_str.split(",") if x.strip()} if m_str else set()
        return result

    # Pandas DataFrame
    if isinstance(data, pd.DataFrame):
        cols = list(data.columns)
        id_col = default_id_col if default_id_col in cols else ("source1_entity_id" if "source1_entity_id" in cols else cols[0])
        val_col = default_val_col if default_val_col in cols else ("matched_entity_ids" if "matched_entity_ids" in cols else ("candidate_entity_ids" if "candidate_entity_ids" in cols else cols[1]))

        id_arr = data[id_col].astype(str).values
        val_arr = data[val_col].fillna("").astype(str).values

        result = {}
        for s1, m_str in zip(id_arr, val_arr):
            result[s1] = {x.strip() for x in m_str.split(",") if x.strip()} if m_str else set()
        return result

    raise TypeError(f"Unsupported data type for entity mapping: {type(data)}")


def compute_macro_and_ceiling_f05(
    ground_truth: Any,
    predictions: Any,
    s1_col: str = "source1_entity_id",
    match_col: str = "matched_entity_ids",
) -> Tuple[float, float]:
    """
    Computes both macro F_0.5 and ceiling F_0.5.

    Args:
        ground_truth: Polars/Pandas DataFrame, dict, or file path to ground truth
        predictions: Polars/Pandas DataFrame, dict, or file path to candidate/matching predictions
        s1_col: ID column name for source 1 (default: 'source1_entity_id')
        match_col: Matches column name (default: 'matched_entity_ids')

    Returns:
        (macro_F05, ceiling_F05) as a tuple of floats
    """
    gt_map = _parse_entity_map(ground_truth, default_id_col=s1_col, default_val_col=match_col)
    pred_map = _parse_entity_map(predictions, default_id_col=s1_col, default_val_col=match_col)

    if not gt_map:
        return 0.0, 0.0

    empty_set = set()
    f05_scores = []
    ceiling_scores = []

    for s1_id, g_set in gt_map.items():
        p_set = pred_map.get(s1_id, empty_set)
        f05, _, _ = compute_f05_single(p_set, g_set)
        ceil_f05 = compute_ceiling_f05_single(p_set, g_set)

        f05_scores.append(f05)
        ceiling_scores.append(ceil_f05)

    macro_f05 = float(np.mean(f05_scores))
    ceiling_f05 = float(np.mean(ceiling_scores))

    return round(macro_f05, 5), round(ceiling_f05, 5)


def evaluate_predictions(
    ground_truth: Any,
    predictions: Any,
    s1_col: str = "source1_entity_id",
    match_col: str = "matched_entity_ids",
) -> Dict[str, float]:
    """
    Comprehensive evaluation returning macro F_0.5, ceiling F_0.5, precision, recall, and singleton stats.
    """
    gt_map = _parse_entity_map(ground_truth, default_id_col=s1_col, default_val_col=match_col)
    pred_map = _parse_entity_map(predictions, default_id_col=s1_col, default_val_col=match_col)

    total_entities = len(gt_map)
    if total_entities == 0:
        return {"macro_F05": 0.0, "ceiling_F05": 0.0}

    empty_set = set()
    f05_list = []
    ceil_list = []
    prec_list = []
    rec_list = []

    num_singletons = 0
    singleton_correct = 0
    captured_gt_matches = 0
    total_gt_matches = 0

    for s1_id, g_set in gt_map.items():
        p_set = pred_map.get(s1_id, empty_set)
        f05, prec, rec = compute_f05_single(p_set, g_set)
        ceil_f05 = compute_ceiling_f05_single(p_set, g_set)

        f05_list.append(f05)
        ceil_list.append(ceil_f05)

        len_g = len(g_set)
        total_gt_matches += len_g

        if len_g == 0:
            num_singletons += 1
            if len(p_set) == 0:
                singleton_correct += 1
        else:
            prec_list.append(prec)
            rec_list.append(rec)
            captured_gt_matches += len(p_set.intersection(g_set))

    macro_f05 = float(np.mean(f05_list))
    ceiling_f05 = float(np.mean(ceil_list))
    prec_macro = float(np.mean(prec_list)) if prec_list else 1.0
    rec_macro = float(np.mean(rec_list)) if rec_list else 1.0
    singleton_acc = float(singleton_correct / num_singletons) if num_singletons > 0 else 1.0
    cand_recall_ceiling = (captured_gt_matches / total_gt_matches) if total_gt_matches > 0 else 1.0

    return {
        "macro_F05": round(macro_f05, 5),
        "ceiling_F05": round(ceiling_f05, 5),
        "precision_macro": round(prec_macro, 5),
        "recall_macro": round(rec_macro, 5),
        "singleton_accuracy": round(singleton_acc, 5),
        "candidate_recall_ceiling": round(cand_recall_ceiling, 5),
        "total_entities": total_entities,
        "num_singletons": num_singletons,
        "num_non_singletons": total_entities - num_singletons,
    }


# =============================================================================
# 2. Deterministic Hash-Based Splitting
# =============================================================================

def assign_entity_split(
    entity_id: str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    val_ratio: float = 0.1,
    salt: str = "amazon_er_2026",
) -> str:
    """
    Deterministically maps an entity_id to 'train', 'dev', or 'val' via MD5 hashing.
    Runs in O(1) time without requiring a global dataset shuffle or seed synchronization.
    """
    total = train_ratio + dev_ratio + val_ratio
    t_bound = int((train_ratio / total) * 10000)
    d_bound = t_bound + int((dev_ratio / total) * 10000)

    # Compute deterministic hash bucket [0, 9999]
    key = f"{salt}:{entity_id}".encode("utf-8")
    hash_val = int(hashlib.md5(key).hexdigest()[:8], 16) % 10000

    if hash_val < t_bound:
        return "train"
    elif hash_val < d_bound:
        return "dev"
    else:
        return "val"


def add_split_column(
    df: Any,
    id_col: str = "source1_entity_id",
    split_col: str = "split",
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    val_ratio: float = 0.1,
    salt: str = "amazon_er_2026",
) -> Any:
    """
    Adds a deterministic 'split' column to a Polars or Pandas DataFrame.
    """
    total = train_ratio + dev_ratio + val_ratio
    t_bound = int((train_ratio / total) * 10000)
    d_bound = t_bound + int((dev_ratio / total) * 10000)

    def _get_split(eid: str) -> str:
        key = f"{salt}:{eid}".encode("utf-8")
        h = int(hashlib.md5(key).hexdigest()[:8], 16) % 10000
        if h < t_bound:
            return "train"
        elif h < d_bound:
            return "dev"
        return "val"

    if HAS_POLARS and isinstance(df, pl.DataFrame):
        return df.with_columns(
            pl.col(id_col)
            .cast(pl.String)
            .map_elements(_get_split, return_dtype=pl.String)
            .alias(split_col)
        )
    elif isinstance(df, pd.DataFrame):
        out = df.copy()
        out[split_col] = out[id_col].astype(str).map(_get_split)
        return out
    else:
        raise TypeError("df must be a Polars or Pandas DataFrame")


def create_splits(
    s1_ids: Iterable[str],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    val_ratio: float = 0.1,
    salt: str = "amazon_er_2026",
) -> Dict[str, List[str]]:
    """
    Splits an iterable of S1 entity IDs into 'train', 'dev', and 'val' lists.
    """
    splits = {"train": [], "dev": [], "val": []}
    for s1_id in s1_ids:
        s = assign_entity_split(
            s1_id,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            val_ratio=val_ratio,
            salt=salt,
        )
        splits[s].append(str(s1_id))
    return splits


# =============================================================================
# 3. CLI Demonstration & Self-Test
# =============================================================================

if __name__ == "__main__":
    # Quick self-test demonstration
    toy_gt = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),  # Singleton
        "S1-3": {"S2-30", "S2-31", "S3-30"},
        "S1-4": {"S2-40"},
    }

    toy_preds = {
        "S1-1": {"S2-10", "S3-20", "S2-99"},  # 2 TP, 1 FP
        "S1-2": set(),                         # 1 True Singleton
        "S1-3": {"S2-30", "S2-31"},            # 2 TP, 0 FP (Missed 1)
        "S1-4": set(),                         # Missed all
    }

    macro_f, ceil_f = compute_macro_and_ceiling_f05(toy_gt, toy_preds)
    full_eval = evaluate_predictions(toy_gt, toy_preds)

    print("Toy Evaluation Results:")
    print(f"  Macro F0.5:   {macro_f}")
    print(f"  Ceiling F0.5: {ceil_f}")
    print("Full Metrics:", full_eval)

    splits = create_splits(toy_gt.keys())
    print("\nDeterministic Entity Splits:")
    for k, v in splits.items():
        print(f"  {k}: {v}")
