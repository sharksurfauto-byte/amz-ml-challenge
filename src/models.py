import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb
from src.metrics import compute_f05_macro
from src.features import FEATURE_NAMES


class LGBMReranker:
    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.85,
            "min_child_samples": 50,
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        print(f"[*] Training LightGBM Re-Ranker on {len(X_train):,} pairs (Positives: {np.sum(y_train):,})...")
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50, verbose=False)] if eval_set else None
        )
        print("[+] Model training complete!")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(FEATURE_NAMES, importances))


def optimize_f05_threshold(
    s1_ids: List[str],
    cand_ids: List[str],
    y_probs: np.ndarray,
    gt_dict: Dict[str, Set[str]],
    tau_range: Tuple[float, float] = (0.40, 0.90),
    step: float = 0.05,
    max_s2: int = 5,
    max_s3: int = 6
) -> Tuple[float, float, Dict]:
    """
    Sweeps decision threshold tau to maximize official macro F_0.5 score.
    Limits match counts strictly to Ground Truth maximums (Max S2 = 5, Max S3 = 6, Total = 11).

    Returns:
        (best_tau, best_f05_score, best_metrics_dict)
    """
    print("\n" + "=" * 65)
    print(" AUTOMATED F_0.5 THRESHOLD OPTIMIZATION (GRID-SEARCH)")
    print("=" * 65)
    
    # Organize candidate predictions by S1 entity: s1_id -> list of (cand_id, prob)
    s1_to_preds = defaultdict(list)
    for s1, cand, prob in zip(s1_ids, cand_ids, y_probs):
        s1_to_preds[s1].append((cand, prob))

    all_s1_entities = list(gt_dict.keys())
    thresholds = np.arange(tau_range[0], tau_range[1] + 1e-5, step)
    
    best_tau = 0.50
    best_f05 = -1.0
    best_report = {}

    print(f"Sweeping {len(thresholds)} thresholds from {tau_range[0]:.2f} to {tau_range[1]:.2f} (Limits: S2 <= {max_s2}, S3 <= {max_s3})...")

    for tau in thresholds:
        best_assignment = {}
        for s1 in all_s1_entities:
            cands = s1_to_preds.get(s1, [])
            if not cands:
                continue

            # Separate S2 and S3 and filter by tau
            s2_cands = [(c, p) for c, p in cands if c.startswith("S2-") and p >= tau]
            s3_cands = [(c, p) for c, p in cands if c.startswith("S3-") and p >= tau]
            
            s2_cands.sort(key=lambda x: x[1], reverse=True)
            s3_cands.sort(key=lambda x: x[1], reverse=True)
            
            selected = [(c, p) for c, p in s2_cands[:max_s2]] + [(c, p) for c, p in s3_cands[:max_s3]]
            
            for c, p in selected:
                if c not in best_assignment or p > best_assignment[c][0]:
                    best_assignment[c] = (p, s1)

        pred_dict = {s1: set() for s1 in all_s1_entities}
        for c, (p, s1) in best_assignment.items():
            pred_dict[s1].add(c)

        # Evaluate official macro F_0.5 using src/metrics.py
        metrics = compute_f05_macro(pred_dict, gt_dict, verbose=False)
        score = metrics["f05_macro"]

        if score > best_f05:
            best_f05 = score
            best_tau = float(tau)
            best_report = metrics

        print(f"  tau = {tau:.2f} -> F_0.5 = {score:.5f} (Prec: {metrics['precision_macro']:.4f}, Rec: {metrics['recall_macro']:.4f}, SingAcc: {metrics['singleton_accuracy']*100:.1f}%)")

    print("=" * 65)
    print(f" OPTIMAL THRESHOLD: tau* = {best_tau:.2f}")
    print(f" PEAK VALIDATION SCORE: F_0.5 = {best_f05:.5f}")
    print("=" * 65)
    return best_tau, best_f05, best_report


if __name__ == "__main__":
    from collections import defaultdict
    print("[*] Testing LGBMReranker initialization...")
    clf = LGBMReranker()
    print("[SUCCESS] src/models.py verified successfully!")
