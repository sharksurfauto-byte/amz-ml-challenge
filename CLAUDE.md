# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview & Persona

- **Competition:** Amazon ML Challenge 2026 — Business Entity Resolution (ER).
- **Core Objective:** Resolve noisy, multi-source business records (Source 2 and Source 3) against deduplicated reference entities (Source 1).
- **Evaluation Metric:** Macro-averaged $F_{0.5}$ across all test S1 entities:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  - Precision is weighted $2\times$ higher than Recall. False merges severely penalize the score.
  - Singletons (~5.58% in GT) score 1.0 if predicted empty, 0.0 if any match is predicted.
- **Role Directive:** Lead ML Architect & Strategist.
  - **Do NOT code directly in the main agent turn** — delegate implementation to specialized subagents (`Agent` tool).
  - Design mathematically sound architectures, enforce rigorous hard-negative validation, verify outputs, and manage parallel workstreams for the 4-person team.

---

## Dataset & Topology Specs

- **Data Path:** `data/train/` and `data/test/`
  - `data/train/train_source1.tsv`: 2,206,821 reference entities (`entity_id`, `business_name`, `business_address`, `country`)
  - `data/train/train_source2.tsv`: 4,887,273 records (noisy business entities)
  - `data/train/train_source3.tsv`: 5,082,316 records (web/domain-heavy records)
  - `data/train/train_ground_truth.tsv`: 2,206,821 rows (`source1_entity_id`, `matched_entity_ids`)
  - `data/test/test_source1.tsv`: 1,732,544 test reference entities
  - `data/test/test_source2.tsv`: 4,887,273 test records
  - `data/test/test_source3.tsv`: 5,082,316 test records
- **Ground Truth Cluster Rules (Empirically Verified):**
  - Average matches per entity: 3.46 (median 3.0)
  - Singletons: Exactly 5.58% (123,086 out of 2.2M)
  - Max S2 matches per entity: **5**
  - Max S3 matches per entity: **6**
  - Max Total matches (S2 + S3): **11**
  - *Never artificially clamp predictions to $\le 2$ or $\le 4$ matches.*

---

## Output Format & Submission Rules

Two tab-separated (`.tsv`) files in `output/`:
1. `output/matching_results.tsv` (Portal scored):
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Exactly 1,732,544 rows (matches test S1 IDs in order).
   - Comma-separated list with no spaces, quotes, or brackets (e.g. `S2-00047,S3-00812`).
   - Singletons must have an empty string in `matched_entity_ids`.
   - Only test S2 and S3 IDs allowed (no S1 IDs, no duplicates).
2. `output/candidate_pairs.tsv` (Blocking verification):
   - Header: `source1_entity_id\tcandidate_entity_ids`
   - Same format rules as `matching_results.tsv`.
   - **Strict Subset Constraint:** Every ID in `matching_results.tsv` must exist in `candidate_pairs.tsv` ($P \subseteq C$).
3. **File Size Limit:** Both files must be strictly **$< 512\text{ MB}$**. (Keep candidate pairs $\le 18$ per entity to stay safely around 250–280 MB).

---

## Architecture Pipeline

1. **Stage 1: Multi-Channel Candidate Blocking (Streaming & Inverted Index)**
   - Exact & Cleaned Name Equality.
   - Token & Prefix (2-token / 4-char prefix) Inverted Index.
   - Physical Address & Street-Number matching (catches DBA / subsidiary co-location).
   - Domain Root matching for S3.
   - Memory-safe streaming in chunks of 250k entities to keep RAM $< 2\text{ GB}$.
2. **Stage 2: Tabular Re-Ranking (LightGBM / CatBoost / XGBoost)**
   - **Critical Rule:** MUST train on **Hard Negatives** (candidates retrieved by blocking that are not in GT), NEVER purely random negatives (which causes precision collapse).
   - 15+ RapidFuzz C++ similarity features (token sort ratio, token set ratio, partial ratio, Jaro-Winkler, address Jaccard, domain root match, rank).
   - Calibrated probability thresholding ($\tau \approx 0.48 - 0.52$) tuned specifically for Macro $F_{0.5}$.
3. **Stage 3: SOTA Ensembling & Semantic Search (GPU Workstreams)**
   - Sentence-Transformers Bi-Encoder embeddings (`all-MiniLM-L6-v2`) + FAISS indexing for semantic candidate retrieval.
   - Transitive Graph Closure (Connected Components) to validate cross-source consistency ($S2 \leftrightarrow S3$).
   - Rank-averaging ensemble across GBDT variants (LightGBM + CatBoost + XGBoost).

---

## Common Development Commands

### Environment Setup
```powershell
pip install -r requirements.txt
# Key packages: rapidfuzz, lightgbm, xgboost, catboost, polars, pandas, scikit-learn, torch
```

### Local Validation & Scoring
```powershell
# Validate submission compliance (format, subset rule, file size, row counts)
python scripts/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/test

# Run fast local 5-fold CV or holdout validation (< 1 min on 10k entities)
python scripts/run_local_benchmark.py --sample-size 20000 --threshold 0.50
```

### End-to-End Pipeline Execution
```powershell
# Run baseline exact-match generator (immediate sanity check submission)
python scripts/baseline_exact_match.py

# Run full ML candidate generation & inference
python scripts/run_full_pipeline.py --mode test
```
