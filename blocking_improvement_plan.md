# Blocking Engine: Gap Analysis & Complete Implementation Plan

> Comparing [`blocking.py`](file:///C:/Users/pujit/.gemini/antigravity-ide/scratch/amz-ml-challenge/src/blocking.py) and [`blocking_lsa.py`](file:///C:/Users/pujit/.gemini/antigravity-ide/scratch/amz-ml-challenge/src/blocking_lsa.py) against the Executive Summary

---

## 1. Current State Snapshot

### What Exists (3-Channel in blocking.py, 5-Channel in blocking_lsa.py)

| Channel | Key | Method | File |
|---------|-----|---------|------|
| C1 | `name_token_sorted_key` | Exact hash join | Both |
| C2 | `name_no_legal` | Exact hash join | Both |
| C3 | `name_no_legal` | TF-IDF char 3-4 n-gram cosine (sparse_dot_topn) | `blocking.py` |
| C3 | `name_no_legal` | TF-IDF → TruncatedSVD(128d) → KNN | `blocking_lsa.py` |
| C4 | `address_street_number` | Exact hash join | `blocking_lsa.py` only |
| C5 | `domain_root` | Exact hash join | `blocking_lsa.py` only |

### Key Parameters (Current)

| Parameter | blocking.py | blocking_lsa.py |
|-----------|-------------|-----------------|
| `top_k` | 20 | 20 |
| `min_similarity` | 0.20 | 0.20 |
| `max_exact_per_key` | 200 | 200 |
| `max_candidates_per_s1` | 20 | 20 |
| `ngram_range` | (3, 4) | (3, 4) |
| `batch_size` | 25,000 | 50,000 |
| `TF-IDF max_features` | 100,000 | 100,000 |
| `TF-IDF min_df` | 10 | 10 |

---

## 2. Gap Analysis: Executive Summary vs. Current Implementation

### 2.1 Missing Blocking Methods

| ES Recommendation | Current Status | Gap Severity |
|-------------------|---------------|--------------|
| **MinHash / LSH** (token Jaccard) | ❌ Not implemented | 🔴 High |
| **Phonetic keys** (Soundex / Metaphone) | ❌ Not implemented | 🟡 Medium |
| **Sorted Neighborhood** | ❌ Not implemented | 🟡 Medium |
| **Canopy Clustering** | ❌ Not implemented | 🟢 Low (niche) |
| **ANN with FAISS / Annoy** | ❌ FAISS/Annoy not used (LSA uses sklearn KNN) | 🔴 High |
| **Learned/Hybrid blocking** | ❌ Not implemented | 🟡 Medium |
| **Multi-field composite keys** | ⚠️ Partial (2 exact keys) | 🟡 Medium |
| **Token blocking** (word-level) | ⚠️ Partial (only char n-gram) | 🟡 Medium |

### 2.2 Architectural Weaknesses in `blocking.py`

```
❌  No channel-level recall/reduction metrics logged at runtime
❌  No pairwise recall evaluation harness
❌  TF-IDF fallback (no sparse_dot_topn) is GIL-bound ThreadPoolExecutor 
    — not truly parallel on CPU-bound scipy.sparse.dot
❌  max_candidates_per_s1=20 uses simple row-rank, not similarity-ranked priority
❌  No score/similarity passed through to the re-ranker as a feature
❌  S1 entities already having ≥ max_candidates skip TF-IDF entirely,
    even though TF-IDF candidates may be higher quality (higher similarity)
❌  No deduplication across S2 vs S3 source awareness in candidate cap
❌  Single-threaded vectorizer.fit_transform (scikit-learn GIL-bound)
❌  No handling of very short strings (<3 chars) in TF-IDF — produces empty vocab
❌  char_wb analyzer pads word boundaries; pure char may be better for entity names
❌  LSA (blocking_lsa.py) redundantly duplicates block_channel_exact_key logic
```

### 2.3 Critical Domain-Specific Gaps (from REPORT.md findings)

```
🔴  Max 11 matches per entity, but max_candidates_per_s1=20 is already fine
    HOWEVER — the cap should be per-source (S2 cap=5, S3 cap=6) not global
🔴  Recall bottleneck: 52% recall at best; ES recommends ≥95% recall as target
🔴  No 2-token prefix blocking (documented as the key recall channel in REPORT.md)
🔴  No 4-character prefix blocking (also in REPORT.md architecture diagram)
🔴  Similarity score not returned — re-ranker misses the `candidate_rank` signal
    from the TF-IDF channel (only exact channels have implicit rank)
```

---

## 3. Implementation Plan

### Phase 1 — Quick Wins (Estimated +3–5% Recall)

#### 1.1 Fix Candidate Cap to Be Per-Source
Current global cap of 20 is oblivious to which source a candidate comes from.
The competition allows S2 ≤ 5 and S3 ≤ 6 — set separate caps.

```python
# In MultiChannelBlocker.__init__, replace:
max_candidates_per_s1: int = 20
# With:
max_candidates_s2: int = 8   # headroom above true max of 5
max_candidates_s3: int = 10  # headroom above true max of 6
```

#### 1.2 Expose Similarity Score as a Column
Pass similarity scores through so the re-ranker can use them.

```python
# block_channel_tfidf_ngram should return 3 columns:
# [s1_id_col, pool_id_col, "tfidf_sim"]
return pl.DataFrame({
    s1_id_col: pl.Series(matched_s1, dtype=pl.UInt32),
    pool_id_col: pl.Series(matched_pool, dtype=pl.UInt32),
    "tfidf_sim": pl.Series(matched_sims, dtype=pl.Float32),  # NEW
})
```

#### 1.3 Add 2-Token Prefix Blocking Channel
Per REPORT.md, this is the key recall driver (boosted recall from 7.5% → 39.46%).

```python
def block_channel_token_prefix(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    text_col: str = "name_clean",
    n_tokens: int = 2,          # use first N sorted tokens as key
    prefix_len: int = 0,        # 0 = full token, >0 = truncate
    max_candidates_per_key: int = 100,
) -> pl.DataFrame:
    """
    Block on the sorted first-N tokens of a name field.
    Handles word-order variance (e.g. 'Tata Steel' vs 'Steel Tata').
    """
    def make_token_key(s: pl.Series) -> pl.Series:
        return (
            s.str.to_lowercase()
             .str.replace_all(r"[^\w\s]", "")
             .str.split(" ")
             .list.sort()
             .list.head(n_tokens)
             .list.join(" ")
        )
    # ... join on key
```

#### 1.4 Add 4-Character Prefix Blocking Channel
```python
def block_channel_prefix(
    s1_df, pool_df,
    text_col="name_no_legal",
    prefix_len: int = 4,
    max_candidates_per_key: int = 150,
) -> pl.DataFrame:
    key_expr = pl.col(text_col).str.slice(0, prefix_len).alias("_prefix_key")
    # ... join on _prefix_key
```

---

### Phase 2 — Medium Impact (Estimated +5–10% Recall)

#### 2.1 Add MinHash / LSH Channel (Token Jaccard)

```python
# New file: src/blocking_minhash.py
from datasketch import MinHash, MinHashLSH

def block_channel_minhash_lsh(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    text_col: str = "name_no_legal",
    num_perm: int = 128,        # MinHash permutations
    threshold: float = 0.3,    # Jaccard similarity threshold
    n_gram: int = 3,            # char n-gram size for shingle set
) -> pl.DataFrame:
    """
    Uses datasketch MinHash LSH to find candidates with Jaccard(shingles) >= threshold.
    Handles near-duplicates that differ in stopwords, punctuation, or word order.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    pool_minhashes = {}
    for row in pool_df.iter_rows(named=True):
        text = row[text_col] or ""
        shingles = {text[i:i+n_gram] for i in range(len(text) - n_gram + 1)}
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        pool_minhashes[row[pool_id_col]] = m
        lsh.insert(row[pool_id_col], m)
    # Query with S1 entities...
```

> **Dependency:** `pip install datasketch`
> **Trade-off:** ~3–5× more candidates than TF-IDF cosine but higher recall on token-order-variant names.

#### 2.2 Add Phonetic Key Channel (Soundex / Double Metaphone)

Captures name variants like "Srivastava" / "Shrivastava" / "Srivastav".

```python
import jellyfish  # or phonetics library

def block_channel_phonetic(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    text_col: str = "name_no_legal",
    method: str = "metaphone",  # "soundex" | "metaphone" | "nysiis"
    max_candidates_per_key: int = 50,
) -> pl.DataFrame:
    encode_fn = {
        "soundex": jellyfish.soundex,
        "metaphone": jellyfish.metaphone,
        "nysiis": jellyfish.nysiis,
    }[method]

    # Create phonetic key: encode first word of name
    def phonetic_key(s: str) -> str:
        first_word = s.strip().split()[0] if s.strip() else ""
        return encode_fn(first_word) if first_word else ""

    # Apply via map_elements, then join
```

> **Dependency:** `pip install jellyfish`

#### 2.3 Replace sklearn KNN with FAISS in `blocking_lsa.py`

Replace slow sklearn `NearestNeighbors` with FAISS `IndexFlatIP` or `IndexIVFFlat`.

```python
import faiss

def build_faiss_index(pool_dense: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    d = pool_dense.shape[1]
    # Normalize for cosine similarity (use inner product)
    faiss.normalize_L2(pool_dense)
    index = faiss.IndexFlatIP(d)  # exact; swap to IndexIVFFlat for >1M
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(pool_dense)
    return index

# For large datasets (>500K records):
def build_faiss_ivf_index(pool_dense, nlist=256, nprobe=32):
    d = pool_dense.shape[1]
    faiss.normalize_L2(pool_dense)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(pool_dense)
    index.add(pool_dense)
    index.nprobe = nprobe
    return index
```

---

### Phase 3 — Advanced / High-Impact (Estimated +8–15% Recall)

#### 3.1 FAISS + Dense Embeddings Channel (Semantic Blocking)

Use a lightweight pre-trained model (e.g., `all-MiniLM-L6-v2` from sentence-transformers)
to embed business names and find semantically similar candidates.

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

def block_channel_dense_embed(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    text_col: str = "name_no_legal",
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 15,
    min_sim: float = 0.60,
    batch_size: int = 4096,
) -> pl.DataFrame:
    model = SentenceTransformer(model_name)
    s1_texts = s1_df[text_col].fill_null("").to_list()
    pool_texts = pool_df[text_col].fill_null("").to_list()

    s1_emb = model.encode(s1_texts, batch_size=batch_size, normalize_embeddings=True)
    pool_emb = model.encode(pool_texts, batch_size=batch_size, normalize_embeddings=True)

    index = faiss.IndexFlatIP(pool_emb.shape[1])
    index.add(pool_emb.astype(np.float32))

    sims, indices = index.search(s1_emb.astype(np.float32), top_k)
    # Filter by min_sim and return pairs...
```

> **Note:** ~384-dim embeddings. On Kaggle (16GB RAM), can handle ~500K–1M records.
> For all 5M+, use `IndexIVFFlat` with `nlist=1024, nprobe=64`.

#### 3.2 Unified Scoring & Candidate Priority Queue

Instead of simple row-rank, merge all channels with a unified priority score.

```python
# Priority weights per channel
CHANNEL_PRIORITY = {
    "exact_token_sorted":  1.0,   # highest
    "exact_no_legal":      0.98,
    "token_prefix_2":      0.90,
    "char_prefix_4":       0.85,
    "address_street":      0.80,
    "domain_root":         0.78,
    "tfidf_ngram":         "tfidf_sim",   # use actual sim score
    "minhash_lsh":         "jaccard_sim",
    "phonetic":            0.50,
    "dense_embed":         "cosine_sim",
}

# After union of all channels, sort by priority + sim score
# and take top max_candidates_per_s1
combined = (
    combined
    .with_columns([
        pl.col("channel_priority").alias("_priority"),
        pl.col("sim_score").fill_null(0.0).alias("_sim"),
    ])
    .sort(["source1_entity_id_int", "_priority", "_sim"], descending=[False, True, True])
    .with_columns(pl.int_range(0, pl.len()).over("source1_entity_id_int").alias("_rank"))
    .filter(pl.col("_rank") < max_candidates_per_s1)
    .drop(["_priority", "_sim", "_rank"])
)
```

#### 3.3 Evaluation Harness (Pair Completeness & Reduction Ratio)

Add a metrics module to evaluate blocking quality against ground truth.

```python
# src/eval/blocking_metrics.py
def evaluate_blocking(
    candidate_pairs: pl.DataFrame,      # (s1_id, pool_id)
    ground_truth: pl.DataFrame,         # (s1_id, pool_id) from train GT
    s1_id_col: str = "source1_entity_id_int",
    pool_id_col: str = "candidate_entity_id_int",
    total_s1: int = None,
    total_pool: int = None,
) -> dict:
    """
    Computes:
      - Pair Completeness (Recall): fraction of GT pairs retained in candidates
      - Reduction Ratio: 1 - (n_candidate_pairs / n_total_possible_pairs)
      - Precision: fraction of candidate pairs that are in GT
    """
    gt_set = set(zip(ground_truth[s1_id_col].to_list(),
                     ground_truth[pool_id_col].to_list()))
    cand_set = set(zip(candidate_pairs[s1_id_col].to_list(),
                       candidate_pairs[pool_id_col].to_list()))

    tp = len(gt_set & cand_set)
    pair_completeness = tp / len(gt_set) if gt_set else 0.0
    precision = tp / len(cand_set) if cand_set else 0.0

    reduction_ratio = None
    if total_s1 and total_pool:
        total_possible = total_s1 * total_pool
        reduction_ratio = 1.0 - (len(cand_set) / total_possible)

    return {
        "pair_completeness": pair_completeness,
        "precision": precision,
        "n_gt_pairs": len(gt_set),
        "n_candidate_pairs": len(cand_set),
        "n_true_positives": tp,
        "reduction_ratio": reduction_ratio,
    }
```

---

## 4. Refactored Architecture (Proposed)

```mermaid
flowchart TD
    subgraph Input
        S1[S1 DataFrame] 
        Pool[S2 / S3 Pool DataFrame]
    end

    subgraph Exact Channels - O(n)
        C1[name_token_sorted_key\nexact hash join]
        C2[name_no_legal\nexact hash join]
        C3[2-token prefix\nexact hash join]
        C4[4-char prefix\nexact hash join]
        C5[address_street_number\nexact hash join]
        C6[domain_root\nexact hash join]
    end

    subgraph Fuzzy Channels - O(n log n)
        C7[TF-IDF char 3-4 ngram\nsparse_dot_topn cosine]
        C8[MinHash LSH\nJaccard token shingles]
        C9[Phonetic keys\nSoundex / Metaphone]
    end

    subgraph Semantic Channel - O(n)
        C10[Dense Embeddings\nFAISS ANN cosine]
    end

    S1 --> C1 & C2 & C3 & C4 & C5 & C6
    Pool --> C1 & C2 & C3 & C4 & C5 & C6
    S1 --> C7 & C8 & C9
    Pool --> C7 & C8 & C9
    S1 --> C10
    Pool --> C10

    C1 & C2 & C3 & C4 & C5 & C6 --> UNION[Priority Union + Dedup]
    C7 & C8 & C9 --> UNION
    C10 --> UNION

    UNION --> CAP[Per-Source Cap\nS2 ≤ 8, S3 ≤ 10]
    CAP --> OUT[candidate_pairs with sim_score column]

    OUT --> EVAL[blocking_metrics.py\nPair Completeness, Reduction Ratio]
    OUT --> RERANK[LightGBM Re-Ranker]
```

---

## 5. Prioritized Roadmap

| Priority | Task | Estimated Recall Gain | Complexity | File |
|----------|------|-----------------------|------------|------|
| 🔴 P0 | Add 2-token prefix channel (`block_channel_token_prefix`) | +5–8% | Low | `blocking.py` |
| 🔴 P0 | Add 4-char prefix channel (`block_channel_prefix`) | +3–5% | Low | `blocking.py` |
| 🔴 P0 | Fix per-source candidate cap (S2 ≤ 8, S3 ≤ 10 separately) | +2–3% | Low | `blocking.py` |
| 🟡 P1 | Expose TF-IDF sim score as column for re-ranker feature | quality | Low | `blocking.py` |
| 🟡 P1 | Replace sklearn KNN with FAISS in `blocking_lsa.py` | +10–20× speed | Medium | `blocking_lsa.py` |
| 🟡 P1 | Add MinHash LSH channel | +3–6% | Medium | `blocking_minhash.py` (new) |
| 🟡 P1 | Add phonetic key channel | +2–4% | Low | `blocking.py` |
| 🟢 P2 | Add dense embedding + FAISS ANN channel | +5–10% | High | `blocking_embed.py` (new) |
| 🟢 P2 | Implement `evaluate_blocking()` metrics harness | visibility | Medium | `src/eval/blocking_metrics.py` |
| 🟢 P2 | Unified priority scoring across all channels | quality | Medium | `blocking.py` |
| 🟢 P3 | Hyperparameter ablation study (min_df, ngram_range, top_k) | optimization | Medium | Notebook |

---

## 6. Quick Configuration Fixes (No New Code Needed)

These can be tuned immediately in the existing `MultiChannelBlocker`:

```python
# Current (suboptimal)
MultiChannelBlocker(
    ngram_range=(3, 4),          # ✅ Keep — char 3-4 grams
    tfidf_top_k=20,              # ⚠️  Raise to 30 — more recall, manageable cost
    min_tfidf_sim=0.20,          # ⚠️  Lower to 0.15 — catch more borderline matches
    max_exact_per_key=200,       # ✅ Fine
    max_candidates_per_s1=20,    # ⚠️  Split into per-source caps
    batch_size=25000,            # ✅ Fine for memory
)

# Recommended
MultiChannelBlocker(
    ngram_range=(2, 4),          # Expand to bigrams for short names
    tfidf_top_k=30,              # More fuzzy candidates
    min_tfidf_sim=0.15,          # Lower threshold → higher recall
    max_exact_per_key=200,
    max_candidates_s2=8,
    max_candidates_s3=10,
    batch_size=25000,
)
```

> **Expected impact:** ~2–4% immediate recall improvement with no new dependencies.

---

## 7. Assumptions & Open Questions

| Assumption | Risk | Mitigation |
|------------|------|------------|
| Ground truth labels are available for blocking recall eval | Low (train_ground_truth.tsv exists) | Use 10% of train as held-out |
| Kaggle environment has ~16 GB RAM | Medium | Use chunked streaming per country partition |
| `sparse_dot_topn` will be installed | Medium | Fallback path exists; warn loudly |
| Dense embeddings fit in memory | High for 5M records | Use IVF index + per-country partitioning |
| `datasketch` (MinHash) available | Low | Standard pip package |

---

## 8. Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| [`src/blocking.py`](file:///C:/Users/pujit/.gemini/antigravity-ide/scratch/amz-ml-challenge/src/blocking.py) | Modify | Add C3/C4 prefix channels, per-source caps, sim score output |
| [`src/blocking_lsa.py`](file:///C:/Users/pujit/.gemini/antigravity-ide/scratch/amz-ml-challenge/src/blocking_lsa.py) | Modify | Replace sklearn KNN with FAISS |
| `src/blocking_minhash.py` | Create | MinHash LSH channel |
| `src/blocking_embed.py` | Create | Dense embedding + FAISS ANN channel |
| `src/blocking_phonetic.py` | Create | Soundex / Metaphone channel |
| `src/eval/blocking_metrics.py` | Create | Pair completeness & reduction ratio harness |
