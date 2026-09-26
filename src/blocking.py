#!/usr/bin/env python3
"""
Multi-Channel Candidate Blocking Engine for Amazon ML Challenge 2026.

Channels:
  1. Exact matching on `name_token_sorted_key` (word-order invariant token match).
  2. Exact matching on `name_no_legal` (clean name stripped of legal suffixes).
  3. TF-IDF character 3-4 n-grams cosine similarity top-20 on `name_no_legal`:
     - Ultra-fast multithreaded C++ sparse_dot_topn library across all CPU cores.
     - Robust fallback chunked CSR dot-product if sparse_dot_topn is missing.

All channels operate on country-partitioned datasets and return pairs of
(source1_entity_id_int, candidate_entity_id_int) as uint32.
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
    is_avail = _SPARSE_DOT_TOPN_MODE is not None
    if not is_avail:
        logger.warning(
            "\033[91;1m"
            "[CRITICAL WARNING] 'sparse_dot_topn' is not active or failed to load. "
            "Pipeline will fall back to slower execution."
            "\033[0m"
        )
    return is_avail


# ---------------------------------------------------------------------------
# Core CPU Sparse Dot Product Top-K Operations
# ---------------------------------------------------------------------------
def _topk_from_csr(
    csr_mat: sparse.csr_matrix,
    top_k: int = 20,
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

        if min_similarity > 0.0:
            mask = row_vals >= min_similarity
            if not np.any(mask):
                continue
            row_cols = row_cols[mask]
            row_vals = row_vals[mask]

        k = min(top_k, len(row_vals))
        if len(row_vals) > k:
            part_idx = np.argpartition(-row_vals, k)[:k]
            # Sort the top-k in descending similarity
            sorted_part = part_idx[np.argsort(-row_vals[part_idx])]
            selected_cols = row_cols[sorted_part]
        else:
            sorted_idx = np.argsort(-row_vals)
            selected_cols = row_cols[sorted_idx]

        out_rows.append(np.full(len(selected_cols), i, dtype=np.int64))
        out_cols.append(selected_cols.astype(np.int64))

    if out_rows:
        return np.concatenate(out_rows), np.concatenate(out_cols)
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def sparse_dot_topk(
    A: sparse.csr_matrix,
    B_T: sparse.csr_matrix,
    top_k: int = 20,
    min_similarity: float = 0.20,
    batch_size: int = 25000,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes top-K values from sparse matrix multiplication A x B_T on CPU.

    Leverages high-performance multithreaded C++ sparse_dot_topn (OpenMP across all cores)
    when available. If missing, falls back to chunked CSR matrix multiplication with a
    bright red warning to install sparse_dot_topn.

    Parameters:
      A: CSR matrix of queries (shape: N x D).
      B_T: CSR matrix of transposed candidates (shape: D x M).
      top_k: Maximum candidate indices to keep per row.
      min_similarity: Minimum cosine similarity threshold.
      batch_size: Batch size of rows of A to process per step if using fallback.
      n_jobs: Number of threads to use (-1 for all available CPU cores).

    Returns:
      Tuple of (row_indices, col_indices) as numpy int64 arrays.
    """
    n_queries = A.shape[0]
    n_candidates = B_T.shape[1]

    if n_queries == 0 or n_candidates == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    effective_threads = max(1, os.cpu_count() or 1) if n_jobs <= 0 else n_jobs

    # 1. Native multithreaded C++ sparse_dot_topn path
    if _SPARSE_DOT_TOPN_MODE == "sp_matmul_topn":
        try:
            logger.info("[+] Executing C++ sparse_dot_topn OpenMP engine...")
            import sparse_dot_topn

            res_csr = sparse_dot_topn.sp_matmul_topn(
                A,
                B_T,
                top_n=top_k,
                threshold=min_similarity,
                sort=True,
                n_threads=effective_threads,
            )
            res_csr = res_csr.tocsr()
            row_idx = np.repeat(np.arange(res_csr.shape[0], dtype=np.int64), np.diff(res_csr.indptr))
            col_idx = res_csr.indices.astype(np.int64)
            return row_idx, col_idx
        except Exception as e:
            logger.warning("sparse_dot_topn sp_matmul_topn failed with %s; falling back to CSR chunking.", e)

    elif _SPARSE_DOT_TOPN_MODE == "awesome_cossim_topn":
        try:
            logger.info("[+] Executing C++ sparse_dot_topn OpenMP engine...")
            import sparse_dot_topn

            res_csr = sparse_dot_topn.awesome_cossim_topn(
                A,
                B_T,
                ntop=top_k,
                lower_bound=min_similarity,
                use_threads=True,
                n_jobs=effective_threads,
                return_best_ntop=True,
            )
            res_csr = res_csr.tocsr()
            row_idx = np.repeat(np.arange(res_csr.shape[0], dtype=np.int64), np.diff(res_csr.indptr))
            col_idx = res_csr.indices.astype(np.int64)
            return row_idx, col_idx
        except Exception as e:
            logger.warning("sparse_dot_topn awesome_cossim_topn failed with %s; falling back to CSR chunking.", e)

    # 2. Fallback chunked CSR multiplication if sparse_dot_topn is missing or failed
    logger.warning(
        "\033[91;1m"
        "[CRITICAL WARNING] 'sparse_dot_topn' is not active! Falling back to slow chunked CSR dot-product.\n"
        "Run 'pip install sparse_dot_topn' immediately to enable 10x-50x faster OpenMP C++ top-K search!"
        "\033[0m"
    )

    all_row_parts: List[np.ndarray] = []
    all_col_parts: List[np.ndarray] = []

    def _process_batch(start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sub_A = A[start_idx:end_idx]
        sim_mat = sub_A.dot(B_T)
        b_rows, b_cols = _topk_from_csr(sim_mat, top_k=top_k, min_similarity=min_similarity)
        if len(b_rows) > 0:
            return b_rows + start_idx, b_cols
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    batch_ranges = [
        (i, min(i + batch_size, n_queries))
        for i in range(0, n_queries, batch_size)
    ]

    # For multi-batch workloads, execute batches concurrently via thread pool
    if len(batch_ranges) > 1 and effective_threads > 1:
        with ThreadPoolExecutor(max_workers=min(effective_threads, len(batch_ranges))) as executor:
            futures = [executor.submit(_process_batch, s, e) for s, e in batch_ranges]
            for f in futures:
                r, c = f.result()
                if len(r) > 0:
                    all_row_parts.append(r)
                    all_col_parts.append(c)
    else:
        for s, e in batch_ranges:
            r, c = _process_batch(s, e)
            if len(r) > 0:
                all_row_parts.append(r)
                all_col_parts.append(c)

    if all_row_parts:
        return np.concatenate(all_row_parts), np.concatenate(all_col_parts)
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Individual Blocking Channels
# ---------------------------------------------------------------------------
def block_channel_exact_key(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    key_col: str,
    s1_id_col: str = "source1_entity_id_int",
    pool_id_col: str = "candidate_entity_id_int",
    max_candidates_per_key: int = 200,
) -> pl.DataFrame:
    """
    Performs exact equality blocking on a specified key column (e.g. name_token_sorted_key or name_no_legal).
    Caps candidate pool entries per key to prevent combinatorial explosion on high-frequency noise.
    """
    empty_res = pl.DataFrame(
        schema={s1_id_col: pl.UInt32, pool_id_col: pl.UInt32}
    )

    if key_col not in s1_df.columns or key_col not in pool_df.columns:
        return empty_res

    # Filter non-null, non-empty keys
    s1_sub = s1_df.select([s1_id_col, key_col]).filter(
        pl.col(key_col).is_not_null() & (pl.col(key_col).str.strip_chars() != "")
    )
    pool_sub = pool_df.select([pool_id_col, key_col]).filter(
        pl.col(key_col).is_not_null() & (pl.col(key_col).str.strip_chars() != "")
    )

    if s1_sub.height == 0 or pool_sub.height == 0:
        return empty_res

    # Cap high-frequency keys in S1 to prevent S1 combinatorial explosion
    if max_candidates_per_key > 0:
        s1_sub = (
            s1_sub
            .with_columns(pl.len().over(key_col).alias("_s1_freq"))
            .filter(pl.col("_s1_freq") <= max_candidates_per_key)
            .drop("_s1_freq")
        )

    # Cap high-frequency keys in pool to max_candidates_per_key
    if max_candidates_per_key > 0:
        pool_sub = (
            pool_sub
            .with_columns(
                pl.int_range(0, pl.len()).over(key_col).alias("_rank_in_key")
            )
            .filter(pl.col("_rank_in_key") < max_candidates_per_key)
            .drop("_rank_in_key")
        )

    # Fast inner join on exact key, ensuring uint32 integer ID output
    joined = (
        s1_sub.join(pool_sub, on=key_col, how="inner")
        .select([
            pl.col(s1_id_col).cast(pl.UInt32),
            pl.col(pool_id_col).cast(pl.UInt32),
        ])
    )
    return joined


def block_channel_tfidf_ngram(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    text_col: str = "name_no_legal",
    s1_id_col: str = "source1_entity_id_int",
    pool_id_col: str = "candidate_entity_id_int",
    top_k: int = 20,
    min_similarity: float = 0.20,
    ngram_range: Tuple[int, int] = (3, 4),
    batch_size: int = 25000,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """
    Channel 3: Character N-gram TF-IDF cosine similarity top-K candidate retrieval.

    Strictly uses scikit-learn TfidfVectorizer(analyzer="char", ngram_range=(3,4), norm="l2", sublinear_tf=True)
    and multithreaded OpenMP sparse_dot_topn for top-K candidate retrieval across all CPU threads.

    Parameters:
      s1_df: Reference S1 entity DataFrame.
      pool_df: Candidate pool DataFrame (S2/S3).
      text_col: Column name containing the text representation.
      s1_id_col: Column name for S1 integer ID.
      pool_id_col: Column name for candidate pool integer ID.
      top_k: Maximum candidate pool matches to retrieve per S1 entity.
      min_similarity: Minimum cosine similarity threshold.
      ngram_range: Tuple of (min_n, max_n) character n-gram lengths.
      batch_size: Number of queries per processing chunk.
      n_jobs: Number of CPU worker threads (-1 for all available cores).

    Returns:
      Polars DataFrame with schema [s1_id_col: UInt32, pool_id_col: UInt32].
    """
    empty_res = pl.DataFrame(schema={s1_id_col: pl.UInt32, pool_id_col: pl.UInt32})

    if s1_df.height == 0 or pool_df.height == 0:
        return empty_res

    # Coalesce text fields to ensure non-empty strings (fallback: name_clean -> business_name)
    s1_cols = s1_df.columns
    s1_exprs = [pl.col(c) for c in [text_col, "name_clean", "business_name"] if c in s1_cols]
    if not s1_exprs:
        return empty_res

    pool_cols = pool_df.columns
    pool_exprs = [pl.col(c) for c in [text_col, "name_clean", "business_name"] if c in pool_cols]
    if not pool_exprs:
        return empty_res

    # Extract text as Python lists for scikit-learn TfidfVectorizer
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

    # Check if there are non-empty texts to vectorize
    if not any(s1_text_series) or not any(pool_text_series):
        return empty_res

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=10,
        max_df=0.3,
        max_features=100000,
        norm="l2",
        sublinear_tf=True,
        dtype=np.float32,
    )

    try:
        M_pool = vectorizer.fit_transform(pool_text_series)
        M_s1 = vectorizer.transform(s1_text_series)
    except ValueError:
        # Handles empty vocabulary edge cases
        return empty_res

    # Transpose candidate pool for sparse dot-product: (N_s1, D) x (D, N_pool) -> (N_s1, N_pool)
    M_pool_T = M_pool.T.tocsr()

    # Query sparse top-k on CPU via multithreaded sparse_dot_topn
    s1_local_idx, pool_local_idx = sparse_dot_topk(
        A=M_s1,
        B_T=M_pool_T,
        top_k=top_k,
        min_similarity=min_similarity,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    if len(s1_local_idx) == 0:
        return empty_res

    # Map local matrix indices back to global uint32 integer IDs
    s1_ids_arr = s1_df.get_column(s1_id_col).to_numpy()
    pool_ids_arr = pool_df.get_column(pool_id_col).to_numpy()

    matched_s1 = s1_ids_arr[s1_local_idx]
    matched_pool = pool_ids_arr[pool_local_idx]

    return pl.DataFrame({
        s1_id_col: pl.Series(matched_s1, dtype=pl.UInt32),
        pool_id_col: pl.Series(matched_pool, dtype=pl.UInt32),
    })


# ---------------------------------------------------------------------------
# Multi-Channel Blocker Engine
# ---------------------------------------------------------------------------
class MultiChannelBlocker:
    """
    Unified Multi-Channel Blocking Engine.

    Channels:
      1. Exact matching on `name_token_sorted_key`
      2. Exact matching on `name_no_legal`
      3. TF-IDF char 3-4 n-gram cosine similarity top-20 on `name_no_legal`
         (Ultra-fast C++ sparse_dot_topn across CPU cores)

    Outputs deduplicated pairs of (s1_id_int, pool_id_int) capped to max_candidates_per_s1.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        tfidf_top_k: int = 20,
        min_tfidf_sim: float = 0.20,
        max_exact_per_key: int = 200,
        max_candidates_per_s1: int = 20,
        batch_size: int = 25000,
        n_jobs: int = -1,
    ):
        self.ngram_range = ngram_range
        self.tfidf_top_k = tfidf_top_k
        self.min_tfidf_sim = min_tfidf_sim
        self.max_exact_per_key = max_exact_per_key
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def block_country(
        self,
        s1_df: pl.DataFrame,
        pool_df: pl.DataFrame,
        s1_id_col: str = "source1_entity_id_int",
        pool_id_col: str = "candidate_entity_id_int",
    ) -> pl.DataFrame:
        """
        Executes all 3 blocking channels on a country subset and returns
        deduplicated candidate pairs capped to max_candidates_per_s1 per S1 entity.
        """
        empty_res = pl.DataFrame(schema={s1_id_col: pl.UInt32, pool_id_col: pl.UInt32})

        if s1_df.height == 0 or pool_df.height == 0:
            return empty_res

        # Channel 1: Exact match on name_token_sorted_key
        pairs_c1 = block_channel_exact_key(
            s1_df=s1_df,
            pool_df=pool_df,
            key_col="name_token_sorted_key",
            s1_id_col=s1_id_col,
            pool_id_col=pool_id_col,
            max_candidates_per_key=self.max_exact_per_key,
        )

        # Channel 2: Exact match on name_no_legal
        pairs_c2 = block_channel_exact_key(
            s1_df=s1_df,
            pool_df=pool_df,
            key_col="name_no_legal",
            s1_id_col=s1_id_col,
            pool_id_col=pool_id_col,
            max_candidates_per_key=self.max_exact_per_key,
        )

        # Combine exact match pairs to find how many candidates each S1 entity has
        exact_pairs = pl.concat([pairs_c1, pairs_c2]).unique(subset=[s1_id_col, pool_id_col])

        # Filter s1_df to only process S1 IDs that have fewer than max_candidates_per_s1 candidates
        if exact_pairs.height > 0 and self.max_candidates_per_s1 > 0:
            s1_cand_counts = exact_pairs.group_by(s1_id_col).agg(pl.len().alias("_cand_count"))
            # S1 entities that already have enough candidates
            satisfied_s1 = s1_cand_counts.filter(pl.col("_cand_count") >= self.max_candidates_per_s1)
            if satisfied_s1.height > 0:
                s1_df_tfidf = s1_df.join(satisfied_s1.select(s1_id_col), on=s1_id_col, how="anti")
            else:
                s1_df_tfidf = s1_df
        else:
            s1_df_tfidf = s1_df

        # Channel 3: TF-IDF char n-grams cosine similarity top-20 (multithreaded CPU)
        pairs_c3 = block_channel_tfidf_ngram(
            s1_df=s1_df_tfidf,
            pool_df=pool_df,
            text_col="name_no_legal",
            s1_id_col=s1_id_col,
            pool_id_col=pool_id_col,
            top_k=self.tfidf_top_k,
            min_similarity=self.min_tfidf_sim,
            ngram_range=self.ngram_range,
            batch_size=self.batch_size,
            n_jobs=self.n_jobs,
        )

        # Union pairs in priority order: Channel 1 -> Channel 2 -> Channel 3
        combined = pl.concat([pairs_c1, pairs_c2, pairs_c3])
        if combined.height == 0:
            return empty_res

        # Deduplicate preserving first occurrence (exact matches have priority over fuzzy)
        deduped = combined.unique(
            subset=[s1_id_col, pool_id_col],
            keep="first",
            maintain_order=True,
        )

        # Restrict to max candidates per S1 entity
        if self.max_candidates_per_s1 > 0:
            deduped = (
                deduped
                .with_columns(
                    pl.int_range(0, pl.len(), dtype=pl.UInt32)
                    .over(s1_id_col)
                    .alias("_cand_rank")
                )
                .filter(pl.col("_cand_rank") < self.max_candidates_per_s1)
                .drop("_cand_rank")
            )

        return deduped

    def block_to_pairs(
        self,
        s1_df: pl.DataFrame,
        pool_df: pl.DataFrame,
        s1_id_col: str = "source1_entity_id_int",
        pool_id_col: str = "candidate_entity_id_int",
    ) -> List[Tuple[int, int]]:
        """
        Executes multi-channel blocking and returns a list of integer pairs (s1_id_int, pool_id_int).
        """
        df_pairs = self.block_country(
            s1_df=s1_df,
            pool_df=pool_df,
            s1_id_col=s1_id_col,
            pool_id_col=pool_id_col,
        )
        return df_pairs.select([s1_id_col, pool_id_col]).rows()


def run_multi_channel_blocking(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    s1_id_col: str = "source1_entity_id_int",
    pool_id_col: str = "candidate_entity_id_int",
    max_candidates: int = 20,
    tfidf_top_k: int = 20,
    min_tfidf_sim: float = 0.20,
    max_exact_per_key: int = 200,
    batch_size: int = 25000,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """
    Functional interface to run multi-channel candidate blocking on a country partition.
    Returns a Polars DataFrame with columns [s1_id_col, pool_id_col] as UInt32.
    """
    blocker = MultiChannelBlocker(
        tfidf_top_k=tfidf_top_k,
        min_tfidf_sim=min_tfidf_sim,
        max_exact_per_key=max_exact_per_key,
        max_candidates_per_s1=max_candidates,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )
    return blocker.block_country(
        s1_df=s1_df,
        pool_df=pool_df,
        s1_id_col=s1_id_col,
        pool_id_col=pool_id_col,
    )
