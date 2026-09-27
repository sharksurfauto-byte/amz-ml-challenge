#!/usr/bin/env python3
"""
8-Channel High-Recall Candidate Blocking Engine for Amazon ML Challenge 2026.

Channels:
  1. Exact matching on `name_token_sorted_key` (word-order invariant token match)
  2. Exact matching on `name_no_legal` (clean name stripped of legal suffixes)
  3. Clean full address join (`address_clean`, min length 8 chars)
  4. Street number + first distinctive name token (`street_number` + `first_token`)
  5. S3 Domain root matching S1 clean name tokens / domain
  6. First two distinctive tokens match (`t0` + `t1`, length >= 4, frequency <= 25)
  7. Character 3-4 n-gram TF-IDF cosine similarity top-10 via multithreaded sparse_dot_topn
  8. Semantic candidates from FAISS retrieval (optional semantic candidate join)

All channels operate on country-partitioned datasets, track channel provenance,
and enforce candidate budgeting (<= 8 from S2, <= 10 from S3, total <= 18 per S1 entity).
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

logger = logging.getLogger("blocking")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# sparse_dot_topn dynamic detection
# ---------------------------------------------------------------------------

_SPARSE_DOT_TOPN_MODE: Optional[str] = None

try:
    import sparse_dot_topn

    if hasattr(sparse_dot_topn, "sp_matmul_topn"):
        _SPARSE_DOT_TOPN_MODE = "sp_matmul_topn"  # sparse_dot_topn >= 1.0.0
    elif hasattr(sparse_dot_topn, "awesome_cossim_topn"):
        _SPARSE_DOT_TOPN_MODE = "awesome_cossim_topn"  # sparse_dot_topn < 1.0.0
except ImportError:
    _SPARSE_DOT_TOPN_MODE = None


def is_sparse_dot_topn_available() -> bool:
    """Returns True if sparse_dot_topn library is installed and available."""
    return _SPARSE_DOT_TOPN_MODE is not None


# ---------------------------------------------------------------------------
# Core CPU Sparse Dot Product Top-K Operations
# ---------------------------------------------------------------------------
def _topk_from_csr(
    csr_mat: sparse.csr_matrix,
    top_k: int = 15,
    min_similarity: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts top-K column indices per row from a CSR matrix where similarity >= min_similarity.
    Uses numpy.argpartition for O(N) top-K selection per row.
    """
    indptr = csr_mat.indptr
    indices = csr_mat.indices
    data = csr_mat.data
    n_rows = csr_mat.shape[0]

    out_rows: List[np.ndarray] = []
    out_cols: List[np.ndarray] = []

    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i + 1]
        if start == end:
            continue

        row_cols = indices[start:end]
        row_vals = data[start:end]

        valid_mask = row_vals >= min_similarity
        if not np.any(valid_mask):
            continue

        row_cols = row_cols[valid_mask]
        row_vals = row_vals[valid_mask]

        if len(row_vals) > top_k:
            top_part = np.argpartition(row_vals, -top_k)[-top_k:]
            sorted_order = top_part[np.argsort(-row_vals[top_part])]
            out_rows.append(np.full(top_k, i, dtype=np.int64))
            out_cols.append(row_cols[sorted_order].astype(np.int64))
        else:
            sorted_order = np.argsort(-row_vals)
            out_rows.append(np.full(len(row_vals), i, dtype=np.int64))
            out_cols.append(row_cols[sorted_order].astype(np.int64))

    if not out_rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(out_rows), np.concatenate(out_cols)


def _chunked_sparse_dot_topn(
    A: sparse.csr_matrix,
    B_T: sparse.csr_matrix,
    top_k: int = 15,
    min_similarity: float = 0.20,
    batch_size: int = 25000,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes top-K dot products between A and B_T in memory-safe row batches.
    """
    n_rows = A.shape[0]
    all_rows: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []

    actual_threads = os.cpu_count() if n_jobs <= 0 else n_jobs
    if actual_threads is None:
        actual_threads = 4

    for start_idx in range(0, n_rows, batch_size):
        end_idx = min(start_idx + batch_size, n_rows)
        A_chunk = A[start_idx:end_idx]

        if _SPARSE_DOT_TOPN_MODE == "sp_matmul_topn":
            try:
                res = sparse_dot_topn.sp_matmul_topn(
                    A_chunk,
                    B_T,
                    top_n=top_k,
                    threshold=min_similarity,
                    n_threads=actual_threads,
                    sort=True,
                )
                r, c = res.nonzero()
                all_rows.append(r.astype(np.int64) + start_idx)
                all_cols.append(c.astype(np.int64))
                continue
            except Exception as e:
                logger.warning("sp_matmul_topn failed on chunk %d-%d: %s. Using fallback.", start_idx, end_idx, e)

        elif _SPARSE_DOT_TOPN_MODE == "awesome_cossim_topn":
            try:
                res = sparse_dot_topn.awesome_cossim_topn(
                    A_chunk,
                    B_T,
                    ntop=top_k,
                    lower_bound=min_similarity,
                    use_threads=True,
                    n_jobs=actual_threads,
                )
                r, c = res.nonzero()
                all_rows.append(r.astype(np.int64) + start_idx)
                all_cols.append(c.astype(np.int64))
                continue
            except Exception as e:
                logger.warning("awesome_cossim_topn failed on chunk %d-%d: %s. Using fallback.", start_idx, end_idx, e)

        # SciPy CSR Fallback
        C_chunk = A_chunk.dot(B_T)
        r, c = _topk_from_csr(C_chunk, top_k=top_k, min_similarity=min_similarity)
        if len(r) > 0:
            all_rows.append(r + start_idx)
            all_cols.append(c)

    if not all_rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(all_rows), np.concatenate(all_cols)


# ---------------------------------------------------------------------------
# Individual Blocking Channel Implementations
# ---------------------------------------------------------------------------

def block_channel_exact_key(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    key_col: str,
    s1_id_col: str = "source1_entity_id_int",
    pool_id_col: str = "candidate_entity_id_int",
    max_candidates_per_key: int = 200,
    priority: int = 1,
    channel_bit: int = 1,
) -> pl.DataFrame:
    """
    Exact equality blocking on a specified column (e.g. name_token_sorted_key, name_no_legal, address_clean).
    """
    empty_res = pl.DataFrame(
        schema={
            s1_id_col: pl.UInt32,
            pool_id_col: pl.UInt32,
            "priority": pl.UInt8,
            "channel_mask": pl.UInt8,
        }
    )

    if key_col not in s1_df.columns or key_col not in pool_df.columns:
        return empty_res

    s1_sub = s1_df.select([s1_id_col, key_col]).filter(
        pl.col(key_col).is_not_null() & (pl.col(key_col).str.strip_chars() != "")
    )
    pool_sub = pool_df.select([pool_id_col, key_col]).filter(
        pl.col(key_col).is_not_null() & (pl.col(key_col).str.strip_chars() != "")
    )

    if s1_sub.height == 0 or pool_sub.height == 0:
        return empty_res

    if max_candidates_per_key > 0:
        s1_sub = (
            s1_sub
            .with_columns(pl.len().over(key_col).alias("_s1_freq"))
            .filter(pl.col("_s1_freq") <= max_candidates_per_key)
            .drop("_s1_freq")
        )
        pool_sub = (
            pool_sub
            .with_columns(
                pl.int_range(0, pl.len()).over(key_col).alias("_rank_in_key")
            )
            .filter(pl.col("_rank_in_key") < max_candidates_per_key)
            .drop("_rank_in_key")
        )

    joined = (
        s1_sub.join(pool_sub, on=key_col, how="inner")
        .select([
            pl.col(s1_id_col).cast(pl.UInt32),
            pl.col(pool_id_col).cast(pl.UInt32),
            pl.lit(priority, dtype=pl.UInt8).alias("priority"),
            pl.lit(channel_bit, dtype=pl.UInt8).alias("channel_mask"),
        ])
    )
    return joined


def block_channel_multi_key(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    key_cols: List[str],
    s1_id_col: str = "source1_entity_id_int",
    pool_id_col: str = "candidate_entity_id_int",
    max_candidates_per_key: int = 100,
    priority: int = 2,
    channel_bit: int = 8,
) -> pl.DataFrame:
    """
    Exact equality blocking on multiple composite columns (e.g. ['street_number', 'first_token'] or ['t0', 't1']).
    """
    empty_res = pl.DataFrame(
        schema={
            s1_id_col: pl.UInt32,
            pool_id_col: pl.UInt32,
            "priority": pl.UInt8,
            "channel_mask": pl.UInt8,
        }
    )

    for k in key_cols:
        if k not in s1_df.columns or k not in pool_df.columns:
            return empty_res

    filter_expr = pl.all_horizontal([
        pl.col(k).is_not_null() & (pl.col(k).str.strip_chars() != "")
        for k in key_cols
    ])

    s1_sub = s1_df.select([s1_id_col] + key_cols).filter(filter_expr)
    pool_sub = pool_df.select([pool_id_col] + key_cols).filter(filter_expr)

    if s1_sub.height == 0 or pool_sub.height == 0:
        return empty_res

    if max_candidates_per_key > 0:
        s1_sub = (
            s1_sub
            .with_columns(pl.len().over(key_cols).alias("_freq"))
            .filter(pl.col("_freq") <= max_candidates_per_key)
            .drop("_freq")
        )
        pool_sub = (
            pool_sub
            .with_columns(
                pl.int_range(0, pl.len()).over(key_cols).alias("_rank")
            )
            .filter(pl.col("_rank") < max_candidates_per_key)
            .drop("_rank")
        )

    joined = (
        s1_sub.join(pool_sub, on=key_cols, how="inner")
        .select([
            pl.col(s1_id_col).cast(pl.UInt32),
            pl.col(pool_id_col).cast(pl.UInt32),
            pl.lit(priority, dtype=pl.UInt8).alias("priority"),
            pl.lit(channel_bit, dtype=pl.UInt8).alias("channel_mask"),
        ])
    )
    return joined


def block_channel_tfidf_ngram(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    text_col: str = "name_no_legal",
    s1_id_col: str = "source1_entity_id_int",
    pool_id_col: str = "candidate_entity_id_int",
    top_k: int = 15,
    min_similarity: float = 0.20,
    ngram_range: Tuple[int, int] = (3, 4),
    batch_size: int = 25000,
    n_jobs: int = -1,
    priority: int = 5,
    channel_bit: int = 64,
) -> pl.DataFrame:
    """
    Character N-gram TF-IDF cosine similarity top-K candidate retrieval via sparse_dot_topn.
    """
    empty_res = pl.DataFrame(
        schema={
            s1_id_col: pl.UInt32,
            pool_id_col: pl.UInt32,
            "priority": pl.UInt8,
            "channel_mask": pl.UInt8,
        }
    )

    if s1_df.height == 0 or pool_df.height == 0:
        return empty_res

    s1_cols = s1_df.columns
    s1_exprs = [pl.col(c) for c in [text_col, "name_clean", "business_name"] if c in s1_cols]
    if not s1_exprs:
        return empty_res

    pool_cols = pool_df.columns
    pool_exprs = [pl.col(c) for c in [text_col, "name_clean", "business_name"] if c in pool_cols]
    if not pool_exprs:
        return empty_res

    s1_text_series = (
        s1_df.select(pl.coalesce(s1_exprs).fill_null("").str.strip_chars().alias("text"))
        .get_column("text")
        .to_list()
    )
    pool_text_series = (
        pool_df.select(pl.coalesce(pool_exprs).fill_null("").str.strip_chars().alias("text"))
        .get_column("text")
        .to_list()
    )

    if not any(s1_text_series) or not any(pool_text_series):
        return empty_res

    min_df = 3 if len(pool_text_series) > 100 else 1
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2",
    )

    try:
        M_pool = vectorizer.fit_transform(pool_text_series)
        M_s1 = vectorizer.transform(s1_text_series)
    except Exception as e:
        logger.warning("TF-IDF Vectorizer failed: %s. Returning empty result.", e)
        return empty_res

    if M_pool.shape[1] == 0 or M_s1.shape[1] == 0:
        return empty_res

    M_pool_T = M_pool.T.tocsr()

    s1_local_idx, pool_local_idx = _chunked_sparse_dot_topn(
        A=M_s1,
        B_T=M_pool_T,
        top_k=top_k,
        min_similarity=min_similarity,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    if len(s1_local_idx) == 0:
        return empty_res

    s1_ids_arr = s1_df.get_column(s1_id_col).to_numpy()
    pool_ids_arr = pool_df.get_column(pool_id_col).to_numpy()

    matched_s1 = s1_ids_arr[s1_local_idx]
    matched_pool = pool_ids_arr[pool_local_idx]

    return pl.DataFrame({
        s1_id_col: pl.Series(matched_s1, dtype=pl.UInt32),
        pool_id_col: pl.Series(matched_pool, dtype=pl.UInt32),
        "priority": pl.lit(priority, dtype=pl.UInt8),
        "channel_mask": pl.lit(channel_bit, dtype=pl.UInt8),
    })


# ---------------------------------------------------------------------------
# Multi-Channel Blocker Engine (8 Channels)
# ---------------------------------------------------------------------------
class MultiChannelBlocker:
    """
    Unified 8-Channel High-Recall Candidate Blocking Engine.

    Channels:
      Ch 1 (bit 1, prio 1): Exact name_token_sorted_key
      Ch 2 (bit 2, prio 1): Exact name_no_legal
      Ch 3 (bit 4, prio 2): Clean full address join (address_clean, length > 8)
      Ch 4 (bit 8, prio 2): Street number + first distinctive name token
      Ch 5 (bit 16, prio 3): S3 Domain root matching S1 domain/name
      Ch 6 (bit 32, prio 4): First two distinctive tokens match (t0, t1)
      Ch 7 (bit 64, prio 5): Character 3-4 n-gram TF-IDF cosine similarity top-15
      Ch 8 (bit 128, prio 6): Semantic candidate retrieval (optional FAISS candidates)

    Enforces candidate budgeting:
      <= max_s2 candidates from Source 2 (default: 8)
      <= max_s3 candidates from Source 3 (default: 10)
      <= max_candidates_per_s1 total (default: 18)
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        tfidf_top_k: int = 15,
        min_tfidf_sim: float = 0.22,
        max_exact_per_key: int = 200,
        max_s2_candidates: int = 8,
        max_s3_candidates: int = 10,
        max_candidates_per_s1: int = 18,
        batch_size: int = 25000,
        n_jobs: int = -1,
    ):
        self.ngram_range = ngram_range
        self.tfidf_top_k = tfidf_top_k
        self.min_tfidf_sim = min_tfidf_sim
        self.max_exact_per_key = max_exact_per_key
        self.max_s2_candidates = max_s2_candidates
        self.max_s3_candidates = max_s3_candidates
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def block_country(
        self,
        s1_df: pl.DataFrame,
        pool_df: pl.DataFrame,
        semantic_cands_df: Optional[pl.DataFrame] = None,
        s1_id_col: str = "source1_entity_id_int",
        pool_id_col: str = "candidate_entity_id_int",
    ) -> pl.DataFrame:
        """
        Executes the high-recall channels on a country partition and returns
        budgeted, deduplicated candidate pairs with provenance.
        """
        empty_res = pl.DataFrame(
            schema={
                s1_id_col: pl.UInt32,
                pool_id_col: pl.UInt32,
                "priority_rank": pl.UInt8,
                "channel_mask": pl.UInt8,
            }
        )

        if s1_df.height == 0 or pool_df.height == 0:
            return empty_res

        channel_dfs: List[pl.DataFrame] = []

        # Channel 1: Exact match on name_token_sorted_key (priority 1, bit 1)
        if "name_token_sorted_key" in s1_df.columns and "name_token_sorted_key" in pool_df.columns:
            p1 = block_channel_exact_key(
                s1_df, pool_df, "name_token_sorted_key",
                s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                max_candidates_per_key=self.max_exact_per_key,
                priority=1, channel_bit=1
            )
            if p1.height > 0:
                channel_dfs.append(p1)

        # Channel 2: Exact match on name_no_legal (priority 1, bit 2)
        if "name_no_legal" in s1_df.columns and "name_no_legal" in pool_df.columns:
            p2 = block_channel_exact_key(
                s1_df, pool_df, "name_no_legal",
                s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                max_candidates_per_key=self.max_exact_per_key,
                priority=1, channel_bit=2
            )
            if p2.height > 0:
                channel_dfs.append(p2)

        # Channel 3: Normalized full address join (priority 2, bit 4)
        if "address_clean" in s1_df.columns and "address_clean" in pool_df.columns:
            s1_addr = s1_df.filter(pl.col("address_clean").str.len_chars() >= 8)
            pool_addr = pool_df.filter(pl.col("address_clean").str.len_chars() >= 8)
            if s1_addr.height > 0 and pool_addr.height > 0:
                p3 = block_channel_exact_key(
                    s1_addr, pool_addr, "address_clean",
                    s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                    max_candidates_per_key=100,
                    priority=2, channel_bit=4
                )
                if p3.height > 0:
                    channel_dfs.append(p3)

        # Channel 4: Street number + first distinctive token (priority 2, bit 8)
        if "street_number" in s1_df.columns and "first_token" in s1_df.columns:
            if "street_number" in pool_df.columns and "first_token" in pool_df.columns:
                p4 = block_channel_multi_key(
                    s1_df, pool_df, ["street_number", "first_token"],
                    s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                    max_candidates_per_key=50,
                    priority=2, channel_bit=8
                )
                if p4.height > 0:
                    channel_dfs.append(p4)

        # Channel 5: S3 Domain root matching S1 domain (priority 3, bit 16)
        if "domain_root" in s1_df.columns and "domain_root" in pool_df.columns:
            s1_dom = s1_df.filter(pl.col("domain_root").is_not_null() & (pl.col("domain_root").str.len_chars() >= 3))
            pool_dom = pool_df.filter(pl.col("domain_root").is_not_null() & (pl.col("domain_root").str.len_chars() >= 3))
            if s1_dom.height > 0 and pool_dom.height > 0:
                p5 = block_channel_exact_key(
                    s1_dom, pool_dom, "domain_root",
                    s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                    max_candidates_per_key=100,
                    priority=3, channel_bit=16
                )
                if p5.height > 0:
                    channel_dfs.append(p5)

        # Channel 6: First two distinctive tokens (priority 4, bit 32)
        if "first_token" in s1_df.columns and "second_token" in s1_df.columns:
            if "first_token" in pool_df.columns and "second_token" in pool_df.columns:
                p6 = block_channel_multi_key(
                    s1_df, pool_df, ["first_token", "second_token"],
                    s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                    max_candidates_per_key=25,
                    priority=4, channel_bit=32
                )
                if p6.height > 0:
                    channel_dfs.append(p6)

        # Channel 7: TF-IDF char 3-4 n-grams cosine top-15 (priority 5, bit 64)
        p7 = block_channel_tfidf_ngram(
            s1_df=s1_df,
            pool_df=pool_df,
            text_col="name_no_legal",
            s1_id_col=s1_id_col,
            pool_id_col=pool_id_col,
            top_k=self.tfidf_top_k,
            min_similarity=self.min_tfidf_sim,
            ngram_range=self.ngram_range,
            batch_size=self.batch_size,
            n_jobs=self.n_jobs,
            priority=5,
            channel_bit=64
        )
        if p7.height > 0:
            channel_dfs.append(p7)

        # Channel 8: Optional semantic candidates from FAISS retrieval (priority 6, bit 128)
        if semantic_cands_df is not None and semantic_cands_df.height > 0:
            # Filter semantic candidates belonging to current S1 partition
            s1_ids_set = s1_df.select(s1_id_col)
            sem_filtered = (
                semantic_cands_df
                .join(s1_ids_set, on=s1_id_col, how="inner")
                .select([
                    pl.col(s1_id_col).cast(pl.UInt32),
                    pl.col(pool_id_col).cast(pl.UInt32),
                    pl.lit(6, dtype=pl.UInt8).alias("priority"),
                    pl.lit(128, dtype=pl.UInt8).alias("channel_mask"),
                ])
            )
            if sem_filtered.height > 0:
                channel_dfs.append(sem_filtered)

        if not channel_dfs:
            return empty_res

        # Union all channels
        combined = pl.concat(channel_dfs)

        # Aggregate channel mask and best priority rank per (s1, cand)
        deduped = (
            combined
            .group_by([s1_id_col, pool_id_col])
            .agg([
                pl.col("priority").min().alias("priority_rank"),
                # Bitwise OR across all channel masks
                pl.col("channel_mask").cast(pl.UInt32).sum().cast(pl.UInt8).alias("channel_mask"),
            ])
        )

        # Budgeting by candidate source: determine source from pool_df source_type if present
        if "source_type" in pool_df.columns:
            deduped = deduped.join(
                pool_df.select([pool_id_col, "source_type"]),
                on=pool_id_col,
                how="left"
            )
            # Separate S2 and S3 candidates
            s2_cands = (
                deduped
                .filter(pl.col("source_type") == 2)
                .sort([s1_id_col, "priority_rank"])
                .with_columns(pl.int_range(0, pl.len()).over(s1_id_col).alias("_rank_s2"))
                .filter(pl.col("_rank_s2") < self.max_s2_candidates)
                .drop(["source_type", "_rank_s2"])
            )
            s3_cands = (
                deduped
                .filter(pl.col("source_type") == 3)
                .sort([s1_id_col, "priority_rank"])
                .with_columns(pl.int_range(0, pl.len()).over(s1_id_col).alias("_rank_s3"))
                .filter(pl.col("_rank_s3") < self.max_s3_candidates)
                .drop(["source_type", "_rank_s3"])
            )
            other_cands = (
                deduped
                .filter(~pl.col("source_type").is_in([2, 3]))
                .sort([s1_id_col, "priority_rank"])
                .with_columns(pl.int_range(0, pl.len()).over(s1_id_col).alias("_rank_oth"))
                .filter(pl.col("_rank_oth") < 5)
                .drop(["source_type", "_rank_oth"])
            )
            budgeted = pl.concat([s2_cands, s3_cands, other_cands])
        else:
            budgeted = (
                deduped
                .sort([s1_id_col, "priority_rank"])
                .with_columns(pl.int_range(0, pl.len()).over(s1_id_col).alias("_rank_all"))
                .filter(pl.col("_rank_all") < self.max_candidates_per_s1)
                .drop("_rank_all")
            )

        # Final safety clamp to max_candidates_per_s1
        if self.max_candidates_per_s1 > 0:
            budgeted = (
                budgeted
                .sort([s1_id_col, "priority_rank"])
                .with_columns(pl.int_range(0, pl.len()).over(s1_id_col).alias("_rank_tot"))
                .filter(pl.col("_rank_tot") < self.max_candidates_per_s1)
                .drop("_rank_tot")
            )

        return budgeted.select([s1_id_col, pool_id_col, "priority_rank", "channel_mask"])
