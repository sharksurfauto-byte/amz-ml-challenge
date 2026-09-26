"""
Official Competition Metric Implementation for Amazon ML Challenge 2026.
Evaluation Metric: Macro-Averaged F_0.5 Score across Source 1 entities.

Formula:
    Precision = |P ∩ G| / |P|
    Recall    = |P ∩ G| / |G|
    F_0.5     = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Special Singleton Rules:
    - If G is empty (|G| == 0) and P is empty (|P| == 0): F_0.5 = 1.0 (True Negative singleton)
    - If G is empty (|G| == 0) and P is non-empty (|P| > 0): F_0.5 = 0.0 (False Positive merge)
    - If G is non-empty (|G| > 0) and P is empty (|P| == 0): F_0.5 = 0.0 (False Negative miss)
    - If |P ∩ G| == 0: F_0.5 = 0.0
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


def compute_f05_single(
    pred_set: Union[Set[str], Iterable[str], str],
    gt_set: Union[Set[str], Iterable[str], str]
) -> Tuple[float, float, float]:
    """
    Computes (f05, precision, recall) for a single Source 1 entity.

    Args:
        pred_set: Predicted matched entity IDs (set, list, or comma-separated string)
        gt_set: Ground truth matched entity IDs (set, list, or comma-separated string)

    Returns:
        (f05, precision, recall) as floats
    """
    # Normalize inputs to sets of non-empty stripped strings
    if isinstance(pred_set, str):
        p = {x.strip() for x in pred_set.split(",") if x.strip()} if pred_set.strip() else set()
    elif isinstance(pred_set, (set, frozenset)):
        p = pred_set
    elif pred_set is None:
        p = set()
    else:
        p = {str(x).strip() for x in pred_set if str(x).strip()}

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

    # Singleton case: ground truth has no matches
    if len_g == 0:
        if len_p == 0:
            return 1.0, 1.0, 1.0  # Correctly predicted singleton
        else:
            return 0.0, 0.0, 1.0  # False merge on a singleton

    # Ground truth is non-empty, but prediction is empty
    if len_p == 0:
        return 0.0, 1.0, 0.0

    # Overlap
    tp = len(p.intersection(g))
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len_p
    recall = tp / len_g

    # F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0, precision, recall

    f05 = (1.25 * precision * recall) / denominator
    return f05, precision, recall


def compute_f05_macro(
    predictions: Union[pd.DataFrame, Dict[str, Set[str]], str],
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]], str],
    s1_col: str = "source1_entity_id",
    match_col: str = "matched_entity_ids",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Computes macro-averaged F_0.5 score across all Source 1 entities in ground truth.

    Args:
        predictions: DataFrame, dict, or file path to matching_results.tsv
        ground_truth: DataFrame, dict, or file path to train_ground_truth.tsv
        s1_col: Column name for Source 1 ID (default: 'source1_entity_id')
        match_col: Column name for matched IDs (default: 'matched_entity_ids')
        verbose: Whether to print diagnostic summary

    Returns:
        dict containing:
            - 'f05_macro': Overall competition score
            - 'precision_macro': Mean precision across non-singleton entities
            - 'recall_macro': Mean recall across non-singleton entities
            - 'singleton_accuracy': Accuracy on singletons
            - 'total_entities': Number of Source 1 entities evaluated
            - 'num_singletons': Number of ground truth singletons
            - 'num_non_singletons': Number of entities with at least 1 match
    """
    # 1. Parse ground truth into dict
    if isinstance(ground_truth, str):
        gt_df = pd.read_csv(ground_truth, sep="\t", dtype=str).fillna("")
        s1_arr = gt_df[s1_col].values
        m_arr = gt_df[match_col].values
        gt_dict = {
            s1: {x.strip() for x in m.split(",") if x.strip()} if m else set()
            for s1, m in zip(s1_arr, m_arr)
        }
    elif isinstance(ground_truth, pd.DataFrame):
        gt_df = ground_truth.fillna("")
        s1_arr = gt_df[s1_col].astype(str).values
        m_arr = gt_df[match_col].astype(str).values
        gt_dict = {
            s1: {x.strip() for x in m.split(",") if x.strip()} if m else set()
            for s1, m in zip(s1_arr, m_arr)
        }
    elif isinstance(ground_truth, dict):
        gt_dict = ground_truth
    else:
        raise TypeError("ground_truth must be DataFrame, dict, or TSV filepath")

    # 2. Parse predictions into dict
    if isinstance(predictions, str):
        pred_df = pd.read_csv(predictions, sep="\t", dtype=str).fillna("")
        s1_arr = pred_df[s1_col].values
        m_arr = pred_df[match_col].values
        pred_dict = {
            s1: {x.strip() for x in m.split(",") if x.strip()} if m else set()
            for s1, m in zip(s1_arr, m_arr)
        }
    elif isinstance(predictions, pd.DataFrame):
        pred_df = predictions.fillna("")
        s1_arr = pred_df[s1_col].astype(str).values
        m_arr = pred_df[match_col].astype(str).values
        pred_dict = {
            s1: {x.strip() for x in m.split(",") if x.strip()} if m else set()
            for s1, m in zip(s1_arr, m_arr)
        }
    elif isinstance(predictions, dict):
        pred_dict = predictions
    else:
        raise TypeError("predictions must be DataFrame, dict, or TSV filepath")

    total_entities = len(gt_dict)
    if total_entities == 0:
        return {"f05_macro": 0.0}

    f05_list = []
    prec_list = []
    rec_list = []

    singleton_correct = 0
    num_singletons = 0
    non_singleton_f05 = []

    empty_set = set()

    for s1_id, g_set in gt_dict.items():
        p_set = pred_dict.get(s1_id, empty_set)
        f05, prec, rec = compute_f05_single(p_set, g_set)

        f05_list.append(f05)

        if len(g_set) == 0:
            num_singletons += 1
            if len(p_set) == 0:
                singleton_correct += 1
        else:
            prec_list.append(prec)
            rec_list.append(rec)
            non_singleton_f05.append(f05)

    f05_macro = float(np.mean(f05_list))
    singleton_acc = float(singleton_correct / num_singletons) if num_singletons > 0 else 1.0
    prec_macro = float(np.mean(prec_list)) if prec_list else 1.0
    rec_macro = float(np.mean(rec_list)) if rec_list else 1.0
    non_singleton_f05_macro = float(np.mean(non_singleton_f05)) if non_singleton_f05 else 0.0

    metrics = {
        "f05_macro": round(f05_macro, 5),
        "precision_macro": round(prec_macro, 5),
        "recall_macro": round(rec_macro, 5),
        "singleton_accuracy": round(singleton_acc, 5),
        "non_singleton_f05": round(non_singleton_f05_macro, 5),
        "total_entities": total_entities,
        "num_singletons": num_singletons,
        "num_non_singletons": total_entities - num_singletons,
    }

    if verbose:
        print("=" * 60)
        print(" [EVALUATION REPORT: F_0.5 Macro]")
        print("=" * 60)
        print(f"Overall F_0.5 Macro:         {metrics['f05_macro']:.5f}")
        print(f"Non-Singleton F_0.5:         {metrics['non_singleton_f05']:.5f}")
        print(f"Precision (Non-Singletons):  {metrics['precision_macro']:.5f}")
        print(f"Recall    (Non-Singletons):  {metrics['recall_macro']:.5f}")
        print(f"Singleton Accuracy:          {metrics['singleton_accuracy'] * 100:.2f}% ({singleton_correct:,}/{num_singletons:,})")
        print(f"Total Evaluated Entities:    {metrics['total_entities']:,}")
        print("=" * 60)

    return metrics


if __name__ == "__main__":
    # Test 1: Official Problem Statement PDF Example
    # S1-00001: Pred=[S2-00047, S2-00193, S3-00812], GT=[S2-00047, S3-00812]
    # Expected: Precision = 2/3, Recall = 1.0, F_0.5 = 0.714
    f05, p, r = compute_f05_single(
        {"S2-00047", "S2-00193", "S3-00812"},
        {"S2-00047", "S3-00812"}
    )
    print(f"Test 1 (PDF Example): F_0.5={f05:.4f} (Expected: ~0.7143), Prec={p:.4f}, Rec={r:.4f}")
    assert abs(f05 - 0.7142857) < 1e-4, f"Mismatch: {f05}"

    # Test 2: Singleton correct
    f05, _, _ = compute_f05_single(set(), set())
    assert f05 == 1.0, f"Expected 1.0 for true singleton, got {f05}"

    # Test 3: Singleton false merge
    f05, _, _ = compute_f05_single({"S2-12345"}, set())
    assert f05 == 0.0, f"Expected 0.0 for false merge on singleton, got {f05}"

    # Test 4: Macro evaluation test
    test_gt = {
        "S1-01": {"S2-01", "S3-01"},
        "S1-02": {"S3-02"},
        "S1-03": set() # singleton
    }
    test_pred = {
        "S1-01": {"S2-01", "S3-01"}, # perfect (1.0)
        "S1-02": set(),              # missed (0.0)
        "S1-03": set()               # correct singleton (1.0)
    }
    res = compute_f05_macro(test_pred, test_gt, verbose=True)
    # Expected macro: (1.0 + 0.0 + 1.0) / 3 = 0.66667
    assert abs(res["f05_macro"] - 0.66667) < 1e-4, f"Mismatch: {res}"
    print("\n[SUCCESS] All unit tests PASSED successfully!")
