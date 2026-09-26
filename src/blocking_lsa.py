#!/usr/bin/env python3
"""
LSA-based candidate blocking engine for Amazon ML Challenge 2026.
Uses TruncatedSVD + NearestNeighbors instead of exact sparse matrix multiplication.
"""

import logging
import os
import sys
from typing import List, Tuple

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
    max_candidates_per_key: int = 200,
) -> pl.DataFrame:
    """Exact equality blocking."""
    empty_res = pl.DataFrame(schema={s1_id_col: pl.UInt32, pool_id_col: pl.UInt32})

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
    batch_size: int = 50000,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """TF-IDF + LSA + KNN based blocking."""
    empty_res = pl.DataFrame(schema={s1_id_col: pl.UInt32, pool_id_col: pl.UInt32})

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

    min_df = 10

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.3,
        max_features=100000,
        dtype=np.float32,
    )

    try:
        M_pool = vectorizer.fit_transform(pool_text_series)
        M_s1 = vectorizer.transform(s1_text_series)
    except Exception as e:
        logger.warning(f"TF-IDF failed: {e}")
        return empty_res

    svd = TruncatedSVD(n_components=128)
    pool_dense = svd.fit_transform(M_pool)
    s1_dense = svd.transform(M_s1)

    nn = NearestNeighbors(n_neighbors=min(top_k, pool_dense.shape[0]), metric='cosine')
    nn.fit(pool_dense)

    all_r: List[np.ndarray] = []
    all_c: List[np.ndarray] = []

    n_queries = s1_dense.shape[0]
    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        sub_s1 = s1_dense[start_idx:end_idx]

        distances, indices = nn.kneighbors(sub_s1)

        if _HAS_CUML and not isinstance(distances, np.ndarray):
            distances = distances.get()
            indices = indices.get()

        # KNN with cosine metric returns distances. similarity = 1 - distance
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

    logger.info(f"LSA KNN found {len(matched_s1)} fuzzy matches.")

    return pl.DataFrame({
        s1_id_col: pl.Series(matched_s1, dtype=pl.UInt32),
        pool_id_col: pl.Series(matched_pool, dtype=pl.UInt32),
    })


class MultiChannelBlocker:
    """LSA Multi-Channel Blocking Engine."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 4),
        tfidf_top_k: int = 20,
        min_tfidf_sim: float = 0.20,
        max_exact_per_key: int = 200,
        max_candidates_per_s1: int = 20,
        batch_size: int = 50000,
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
        empty_res = pl.DataFrame(schema={s1_id_col: pl.UInt32, pool_id_col: pl.UInt32})

        if s1_df.height == 0 or pool_df.height == 0:
            return empty_res

        pairs_c1 = block_channel_exact_key(
            s1_df=s1_df,
            pool_df=pool_df,
            key_col="name_token_sorted_key",
            s1_id_col=s1_id_col,
            pool_id_col=pool_id_col,
            max_candidates_per_key=self.max_exact_per_key,
        )

        pairs_c2 = block_channel_exact_key(
            s1_df=s1_df,
            pool_df=pool_df,
            key_col="name_no_legal",
            s1_id_col=s1_id_col,
            pool_id_col=pool_id_col,
            max_candidates_per_key=self.max_exact_per_key,
        )

        pairs_c3 = block_channel_tfidf_ngram(
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
        )

        combined = pl.concat([pairs_c1, pairs_c2, pairs_c3])
        if combined.height == 0:
            return empty_res

        deduped = combined.unique(
            subset=[s1_id_col, pool_id_col],
            keep="first",
            maintain_order=True,
        )

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
    batch_size: int = 50000,
    n_jobs: int = -1,
) -> pl.DataFrame:
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
