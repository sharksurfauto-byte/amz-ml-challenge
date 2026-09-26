import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import polars as pl
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from src.metrics import compute_f05_macro
from src.features import FEATURE_NAMES

df = pl.read_parquet('data/benchmark_cache/benchmark_highrecall_20000_k18.parquet')
all_s1 = df['s1_id'].unique().to_list()
n_train = int(len(all_s1) * 0.6)
train_s1 = set(all_s1[:n_train])
val_s1 = set(all_s1[n_train:])

train_df = df.filter(pl.col('s1_id').is_in(list(train_s1)))
val_df = df.filter(pl.col('s1_id').is_in(list(val_s1)))

X_train = train_df.select(FEATURE_NAMES).to_numpy()
y_train = train_df['label'].to_numpy()
X_val = val_df.select(FEATURE_NAMES).to_numpy()

clf = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=-1, verbose=-1)
clf.fit(X_train, y_train)

val_probs = clf.predict_proba(X_val)[:, 1]
val_s1_list = val_df['s1_id'].to_list()
val_cand_list = val_df['cand_id'].to_list()

gt = pl.read_csv('data/raw/train/train_ground_truth.tsv', separator='\t').filter(pl.col('source1_entity_id').is_in(list(val_s1))).fill_null('')
gt_dict = {}
for row in gt.iter_rows():
    gt_dict[row[0]] = set(x.strip() for x in row[1].split(',') if x.strip()) if row[1] else set()
for s in val_s1:
    if s not in gt_dict: gt_dict[s] = set()

s1_to_preds = defaultdict(list)
for s, c, p in zip(val_s1_list, val_cand_list, val_probs):
    s1_to_preds[s].append((c, p))

best_score = 0
best_combo = None

print("Grid Searching (anchor_tau, match_tau)...")
for anchor_tau in [0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85]:
    for tau in [0.30, 0.35, 0.40, 0.45, 0.50]:
        pred_dict = {}
        for s in val_s1:
            cands = s1_to_preds.get(s, [])
            if not cands:
                pred_dict[s] = set()
                continue
            max_p = max(p for _, p in cands)
            if max_p < anchor_tau:
                pred_dict[s] = set()
                continue
            s2_cands = [(c, p) for c, p in cands if c.startswith('S2-') and p >= tau]
            s3_cands = [(c, p) for c, p in cands if c.startswith('S3-') and p >= tau]
            s2_cands.sort(key=lambda x: x[1], reverse=True)
            s3_cands.sort(key=lambda x: x[1], reverse=True)
            pred_dict[s] = set([c for c, _ in s2_cands[:5]] + [c for c, _ in s3_cands[:6]])
        m = compute_f05_macro(pred_dict, gt_dict, verbose=False)
        score = m['f05_macro']
        if score > best_score:
            best_score = score
            best_combo = (anchor_tau, tau, m)
        if anchor_tau in [0.0, 0.70, 0.75, 0.80] and tau in [0.35, 0.40, 0.45]:
            p = m['precision_macro']
            r = m['recall_macro']
            sa = m['singleton_accuracy'] * 100
            print(f"  anchor={anchor_tau:.2f}, tau={tau:.2f} -> F_0.5 = {score:.5f} (Prec: {p:.4f}, Rec: {r:.4f}, SingAcc: {sa:.1f}%)")

print("=" * 65)
print(f"PEAK SCORE: F_0.5 = {best_score:.5f} with anchor_tau={best_combo[0]:.2f}, tau={best_combo[1]:.2f}")
p = best_combo[2]['precision_macro']
r = best_combo[2]['recall_macro']
sa = best_combo[2]['singleton_accuracy'] * 100
print(f"Precision: {p:.4f}, Recall: {r:.4f}, Singleton Acc: {sa:.1f}%")
print("=" * 65)
