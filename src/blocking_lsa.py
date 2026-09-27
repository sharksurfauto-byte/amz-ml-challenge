#!/usr/bin/env python3
"""
LSA and Multi-Channel Candidate Blocking Engine for Amazon ML Challenge 2026.
Uses TruncatedSVD + NearestNeighbors (cuML on GPU if available, sklearn fallback)
along with high-recall symbolic blocking channels.

Channels:
  1. Exact matching on `name_token_sorted_key` (priority 1, bit 1)
  2. Exact matching on `name_no_legal` (priority 1, bit 2)
  3. Clean full address join (`address_clean`, length >= 8) (priority 2, bit 4)
  4. Street number + first distinctive token (`street_number` + `first_token`) (priority 2, bit 8)
  5. S3 Domain root matching S1 domain/name (priority 3, bit 16)
  6. First two distinctive tokens match (`first_token` + `second_token`) (priority 4, bit 32)
  7. LSA/TF-IDF char n-gram TruncatedSVD + NearestNeighbors top-K (priority 5, bit 64)
  8. Optional integration of dense semantic candidates from FAISS retrieval (priority 6, bit 128)

Enforces candidate budgeting:
  <= max_s2 candidates from Source 2 (default: 15-20)
  <= max_s3 candidates from Source 3 (default: 15-25)
  <= max_candidates_per_s1 total (default: 35-50)
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    from cuml.neighbors import NearestNeighbors
    _HAS_CUML = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    _HAS_CUML = False

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

logger = logging.getLogger("blocking_lsa")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def block_channel_exact_key(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    key_col: str,
    s1_id_col: str = "source1_entity_id_int",
    pool_id_col: str = "candidate_entity_id_int",
    max_candidates_per_key: int = 250,
    priority: int = 1,
    channel_bit: int = 1,
) -> pl.DataFrame:
    """Exact equality blocking on key_col."""
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
    max_candidates_per_key: int = 150,
    priority: int = 2,
    channel_bit: int = 8,
) -> pl.DataFrame:
    """Exact equality blocking on multiple composite columns."""
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


def block_channel_tfidf_lsa(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    text_col: str = "name_no_legal",
    s1_id_col: str = "source1_entity_id_int",
    pool_id_col: str = "candidate_entity_id_int",
    top_k: int = 25,
    min_similarity: float = 0.20,
    ngram_range: Tuple[int, int] = (3, 4),
    batch_size: int = 50000,
    n_jobs: int = -1,
    priority: int = 5,
    channel_bit: int = 64,
) -> pl.DataFrame:
    """TF-IDF + TruncatedSVD (LSA) + NearestNeighbors based blocking."""
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

    min_df = 5 if len(pool_text_series) > 1000 else 1

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.35,
        max_features=120000,
        dtype=np.float32,
    )

    try:
        M_pool = vectorizer.fit_transform(pool_text_series)
        M_s1 = vectorizer.transform(s1_text_series)
    except Exception as e:
        logger.warning(f"TF-IDF failed: {e}")
        return empty_res

    if M_pool.shape[1] < 2:
        return empty_res

    n_comp = min(128, M_pool.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    pool_dense = svd.fit_transform(M_pool).astype(np.float32)
    s1_dense = svd.transform(M_s1).astype(np.float32)

    all_r: List[np.ndarray] = []
    all_c: List[np.ndarray] = []
    n_queries = s1_dense.shape[0]
    n_neighbors = min(top_k, pool_dense.shape[0])

    if _HAS_FAISS:
        norm_pool = pool_dense.copy()
        norm_s1 = s1_dense.copy()
        faiss.normalize_L2(norm_pool)
        faiss.normalize_L2(norm_s1)

        index = faiss.IndexFlatIP(norm_pool.shape[1])
        index.add(norm_pool)

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            sub_s1 = norm_s1[start_idx:end_idx]
            sims, indices = index.search(sub_s1, n_neighbors)

            for i in range(len(sims)):
                valid_mask = sims[i] >= min_similarity
                valid_indices = indices[i][valid_mask]
                if len(valid_indices) > 0:
                    all_r.append(np.full(len(valid_indices), start_idx + i, dtype=np.int64))
                    all_c.append(valid_indices.astype(np.int64))
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(pool_dense)

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            sub_s1 = s1_dense[start_idx:end_idx]

            distances, indices = nn.kneighbors(sub_s1)

            if _HAS_CUML and not isinstance(distances, np.ndarray):
                distances = distances.get()
                indices = indices.get()

            for i in range(len(distances)):
                valid_mask = (1.0 - distances[i]) >= min_similarity
                valid_indices = indices[i][valid_mask]
                if len(valid_indices) > 0:
                    all_r.append(np.full(len(valid_indices), start_idx + i, dtype=np.int64))
                    all_c.append(valid_indices.astype(np.int64))

    if not all_r:
        return empty_res

    s1_local_idx = np.concatenate(all_r)
    pool_local_idx = np.concatenate(all_c)

    s1_ids_arr = s1_df.get_column(s1_id_col).to_numpy()
    pool_ids_arr = pool_df.get_column(pool_id_col).to_numpy()

    matched_s1 = s1_ids_arr[s1_local_idx]
    matched_pool = pool_ids_arr[pool_local_idx]

    logger.info(f"LSA KNN found {len(matched_s1):,} fuzzy matches.")

    return pl.DataFrame({
        s1_id_col: pl.Series(matched_s1, dtype=pl.UInt32),
        pool_id_col: pl.Series(matched_pool, dtype=pl.UInt32),
        "priority": pl.lit(priority, dtype=pl.UInt8),
        "channel_mask": pl.lit(channel_bit, dtype=pl.UInt8),
    })


class MultiChannelBlocker:
    """
    Unified 8-Channel High-Recall Candidate Blocking Engine using LSA + Symbolic Keys.
    Supports candidate budgeting up to 30-50 candidates per S1 entity.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        tfidf_top_k: int = 25,
        min_tfidf_sim: float = 0.20,
        max_exact_per_key: int = 250,
        max_s2_candidates: int = 20,
        max_s3_candidates: int = 25,
        max_candidates_per_s1: int = 40,
        batch_size: int = 50000,
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
        """Executes all high-recall channels on country partition."""
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

        # Channel 1: Exact name_token_sorted_key (prio 1, bit 1)
        if "name_token_sorted_key" in s1_df.columns and "name_token_sorted_key" in pool_df.columns:
            p1 = block_channel_exact_key(
                s1_df, pool_df, "name_token_sorted_key",
                s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                max_candidates_per_key=self.max_exact_per_key,
                priority=1, channel_bit=1
            )
            if p1.height > 0:
                channel_dfs.append(p1)

        # Channel 2: Exact name_no_legal (prio 1, bit 2)
        if "name_no_legal" in s1_df.columns and "name_no_legal" in pool_df.columns:
            p2 = block_channel_exact_key(
                s1_df, pool_df, "name_no_legal",
                s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                max_candidates_per_key=self.max_exact_per_key,
                priority=1, channel_bit=2
            )
            if p2.height > 0:
                channel_dfs.append(p2)

        # Channel 3: Normalized full address join (prio 2, bit 4)
        addr_col = "address_clean" if "address_clean" in s1_df.columns else "business_address"
        pool_addr_col = "address_clean" if "address_clean" in pool_df.columns else "business_address"
        if addr_col in s1_df.columns and pool_addr_col in pool_df.columns:
            s1_addr = s1_df.filter(pl.col(addr_col).str.len_chars() >= 8)
            pool_addr = pool_df.filter(pl.col(pool_addr_col).str.len_chars() >= 8)
            if s1_addr.height > 0 and pool_addr.height > 0:
                p3 = block_channel_exact_key(
                    s1_addr, pool_addr, addr_col,
                    s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                    max_candidates_per_key=150,
                    priority=2, channel_bit=4
                )
                if p3.height > 0:
                    channel_dfs.append(p3)

        # Channel 4: Street number + first token (prio 2, bit 8)
        s_num_col = "street_number" if "street_number" in s1_df.columns else "address_street_number"
        p_s_num_col = "street_number" if "street_number" in pool_df.columns else "address_street_number"
        if s_num_col in s1_df.columns and "first_token" in s1_df.columns:
            if p_s_num_col in pool_df.columns and "first_token" in pool_df.columns:
                p4 = block_channel_multi_key(
                    s1_df, pool_df, [s_num_col, "first_token"],
                    s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                    max_candidates_per_key=80,
                    priority=2, channel_bit=8
                )
                if p4.height > 0:
                    channel_dfs.append(p4)

        # Channel 5: Domain root join (prio 3, bit 16)
        if "domain_root" in s1_df.columns and "domain_root" in pool_df.columns:
            s1_dom = s1_df.filter(pl.col("domain_root").is_not_null() & (pl.col("domain_root").str.len_chars() >= 3))
            pool_dom = pool_df.filter(pl.col("domain_root").is_not_null() & (pl.col("domain_root").str.len_chars() >= 3))
            if s1_dom.height > 0 and pool_dom.height > 0:
                p5 = block_channel_exact_key(
                    s1_dom, pool_dom, "domain_root",
                    s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                    max_candidates_per_key=150,
                    priority=3, channel_bit=16
                )
                if p5.height > 0:
                    channel_dfs.append(p5)

        # Channel 6: First two distinctive tokens (prio 4, bit 32)
        if "first_token" in s1_df.columns and "second_token" in s1_df.columns:
            if "first_token" in pool_df.columns and "second_token" in pool_df.columns:
                p6 = block_channel_multi_key(
                    s1_df, pool_df, ["first_token", "second_token"],
                    s1_id_col=s1_id_col, pool_id_col=pool_id_col,
                    max_candidates_per_key=40,
                    priority=4, channel_bit=32
                )
                if p6.height > 0:
                    channel_dfs.append(p6)

        # Channel 7: LSA TF-IDF + SVD + KNN (prio 5, bit 64)
        p7 = block_channel_tfidf_lsa(
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

        # Channel 8: Optional semantic candidates from FAISS retrieval (prio 6, bit 128)
        if semantic_cands_df is not None and semantic_cands_df.height > 0:
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

        combined = pl.concat(channel_dfs)

        # Aggregate channel mask and best priority rank
        deduped = (
            combined
            .group_by([s1_id_col, pool_id_col])
            .agg([
                pl.col("priority").min().alias("priority_rank"),
                pl.col("channel_mask").cast(pl.UInt32).sum().cast(pl.UInt8).alias("channel_mask"),
            ])
        )

        # Budgeting by candidate source (S2 vs S3)
        if "source_type" in pool_df.columns:
            deduped = deduped.join(
                pool_df.select([pool_id_col, "source_type"]),
                on=pool_id_col,
                how="left"
            )
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
                .filter(pl.col("_rank_oth") < 10)
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

        if self.max_candidates_per_s1 > 0:
            budgeted = (
                budgeted
                .sort([s1_id_col, "priority_rank"])
                .with_columns(pl.int_range(0, pl.len()).over(s1_id_col).alias("_rank_tot"))
                .filter(pl.col("_rank_tot") < self.max_candidates_per_s1)
                .drop("_rank_tot")
            )

        return budgeted.select([s1_id_col, pool_id_col, "priority_rank", "channel_mask"])

    def block_to_pairs(
        self,
        s1_df: pl.DataFrame,
        pool_df: pl.DataFrame,
        s1_id_col: str = "source1_entity_id_int",
        pool_id_col: str = "candidate_entity_id_int",
    ) -> List[Tuple[int, int]]:
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
    max_candidates: int = 40,
) -> pl.DataFrame:
    blocker = MultiChannelBlocker(max_candidates_per_s1=max_candidates)
    return blocker.block_country(
        s1_df=s1_df,
        pool_df=pool_df,
        s1_id_col=s1_id_col,
        pool_id_col=pool_id_col,
    )
