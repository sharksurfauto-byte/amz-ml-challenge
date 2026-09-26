#!/usr/bin/env python3
"""
Multi-Channel Candidate Blocking Engine for Amazon ML Challenge 2026.

Channels:
  1. Exact matching on `name_token_sorted_key` (word-order invariant token match).
  2. Exact matching on `name_no_legal` (clean name stripped of legal suffixes).
  3. TF-IDF character 3-4 n-grams cosine similarity top-20 on `name_no_legal`:
     - GPU Accelerated Path: cuML TfidfVectorizer + TruncatedSVD (LSA) + cuML NearestNeighbors (cosine).
     - CPU Optimized Path: sparse_dot_topn when available, with chunked CSR fallback.

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
# GPU (cuML + CuPy) dynamic detection and configuration
# ---------------------------------------------------------------------------
_GPU_AVAILABLE: bool = False
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
    from cuml.decomposition import TruncatedSVD as cuTruncatedSVD
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

    # Verify that a CUDA device is present and accessible
    if cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0:
        _GPU_AVAILABLE = True
except (ImportError, Exception):
    _GPU_AVAILABLE = False

try:
    import cudf
    _CUDF_AVAILABLE = True
except (ImportError, Exception):
    _CUDF_AVAILABLE = False

HAS_GPU_RAPIDS_CUML: bool = _GPU_AVAILABLE and _CUDF_AVAILABLE

try:
    import faiss
    _FAISS_AVAILABLE = True
except (ImportError, Exception):
    _FAISS_AVAILABLE = False


def is_gpu_blocking_available() -> bool:
    """Returns True if cuML and CuPy with an active CUDA device are available."""
    return HAS_GPU_RAPIDS_CUML


# ---------------------------------------------------------------------------
# sparse_dot_topn dynamic detection and configuration (CPU path)
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
# GPU Sparse Dot Product & LSA NearestNeighbors Top-K Operations (CuPy / cuML)
# ---------------------------------------------------------------------------
def gpu_knn_lsa_topk(
    M_s1,
    M_pool,
    top_k: int = 20,
    min_similarity: float = 0.20,
    n_components: int = 128,
    batch_size: int = 25000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes top-K candidate matches per S1 query row using GPU Latent Semantic Analysis (LSA)
    with cuML TruncatedSVD and cuML NearestNeighbors (cosine metric).

    Steps:
      1. Apply cuML TruncatedSVD(n_components=128) on M_pool and M_s1 if M_pool.shape[1] > 128.
      2. Normalize the resulting dense matrices to unit L2 norm using CuPy.
      3. Fit cuML NearestNeighbors(n_neighbors=top_k, metric='cosine') on M_pool_dense.
      4. Query M_s1_dense in memory-safe batches to obtain indices and distances.
      5. Filter by 1 - distances >= min_similarity.
      6. Return row_idx, col_idx numpy arrays.

    Parameters:
      M_s1: CuPy sparse CSR matrix of S1 query vectors (shape: N x D).
      M_pool: CuPy sparse CSR matrix of candidate pool vectors (shape: M x D).
      top_k: Maximum candidate indices to retrieve per S1 query row.
      min_similarity: Minimum cosine similarity threshold.
      n_components: Target dimension for TruncatedSVD dimensionality reduction (default: 128).
      batch_size: Batch size of S1 queries to evaluate per KNN inference step.

    Returns:
      Tuple of (matched_s1_local_indices, matched_pool_local_indices) as numpy int64 arrays.
    """
    n_s1 = M_s1.shape[0]
    n_candidates = M_pool.shape[0]

    if n_s1 == 0 or n_candidates == 0 or top_k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    n_features = M_pool.shape[1]

    # Step 1 & 2: Apply TruncatedSVD if n_features > n_components
    if n_features > n_components and min(n_s1, n_candidates) > n_components:
        logger.info(
            "Applying cuML TruncatedSVD (n_components=%d) on %d sparse features (pool: %d, s1: %d)...",
            n_components,
            n_features,
            n_candidates,
            n_s1,
        )
        try:
            from cuml.decomposition import TruncatedSVD as cuTruncatedSVD
            try:
                svd = cuTruncatedSVD(n_components=n_components, output_type="cupy")
            except TypeError:
                svd = cuTruncatedSVD(n_components=n_components)

            M_pool_dense = svd.fit_transform(M_pool)
            M_s1_dense = svd.transform(M_s1)
        except Exception as e_svd:
            logger.warning(
                "cuML TruncatedSVD failed with %s; falling back to direct dense conversion.",
                e_svd,
            )
            if hasattr(M_pool, "toarray"):
                M_pool_dense = M_pool.toarray()
            else:
                M_pool_dense = cp.asarray(M_pool.todense())
            if hasattr(M_s1, "toarray"):
                M_s1_dense = M_s1.toarray()
            else:
                M_s1_dense = cp.asarray(M_s1.todense())
    else:
        # Small feature space or sample size, convert directly to dense
        if hasattr(M_pool, "toarray"):
            M_pool_dense = M_pool.toarray()
        else:
            M_pool_dense = cp.asarray(M_pool.todense())
        if hasattr(M_s1, "toarray"):
            M_s1_dense = M_s1.toarray()
        else:
            M_s1_dense = cp.asarray(M_s1.todense())

    # Ensure CuPy ndarray with float32 precision
    if hasattr(M_pool_dense, "to_cupy"):
        M_pool_dense = M_pool_dense.to_cupy()
    elif hasattr(M_pool_dense, "values"):
        M_pool_dense = cp.asarray(M_pool_dense.values)
    elif not isinstance(M_pool_dense, cp.ndarray):
        M_pool_dense = cp.asarray(M_pool_dense)

    if hasattr(M_s1_dense, "to_cupy"):
        M_s1_dense = M_s1_dense.to_cupy()
    elif hasattr(M_s1_dense, "values"):
        M_s1_dense = cp.asarray(M_s1_dense.values)
    elif not isinstance(M_s1_dense, cp.ndarray):
        M_s1_dense = cp.asarray(M_s1_dense)

    M_pool_dense = M_pool_dense.astype(cp.float32)
    M_s1_dense = M_s1_dense.astype(cp.float32)

    # Step 3: Normalize dense vectors to unit L2 norm using CuPy
    pool_norms = cp.linalg.norm(M_pool_dense, axis=1, keepdims=True)
    pool_norms = cp.maximum(pool_norms, 1e-12)
    M_pool_dense = M_pool_dense / pool_norms

    s1_norms = cp.linalg.norm(M_s1_dense, axis=1, keepdims=True)
    s1_norms = cp.maximum(s1_norms, 1e-12)
    M_s1_dense = M_s1_dense / s1_norms

    actual_k = min(top_k, n_candidates)
    matched_s1_list: List[np.ndarray] = []
    matched_pool_list: List[np.ndarray] = []

    # Step 4: Fit cuML NearestNeighbors (metric='cosine')
    use_cuml_nn = True
    nn = None
    try:
        from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
        try:
            nn = cuNearestNeighbors(n_neighbors=actual_k, metric="cosine", output_type="cupy")
        except TypeError:
            nn = cuNearestNeighbors(n_neighbors=actual_k, metric="cosine")
        nn.fit(M_pool_dense)
    except Exception as e_nn:
        logger.warning(
            "cuML NearestNeighbors initialization/fit failed (%s); trying Faiss fallback.",
            e_nn,
        )
        use_cuml_nn = False

    # Step 5 & 6: Query S1 against candidate pool and filter by 1 - distances >= min_similarity
    if use_cuml_nn and nn is not None:
        try:
            query_batch = max(1000, min(batch_size, 50000))
            for q_start in tqdm(
                range(0, n_s1, query_batch),
                desc="cuML KNN Cosine Batches",
                leave=False,
            ):
                q_end = min(q_start + query_batch, n_s1)
                q_chunk = M_s1_dense[q_start:q_end]

                distances, indices = nn.kneighbors(q_chunk)

                if hasattr(distances, "to_cupy"):
                    distances = distances.to_cupy()
                elif hasattr(distances, "values"):
                    distances = cp.asarray(distances.values)
                elif not isinstance(distances, cp.ndarray):
                    distances = cp.asarray(distances)

                if hasattr(indices, "to_cupy"):
                    indices = indices.to_cupy()
                elif hasattr(indices, "values"):
                    indices = cp.asarray(indices.values)
                elif not isinstance(indices, cp.ndarray):
                    indices = cp.asarray(indices)

                # For cosine metric: similarity = 1 - distance
                sims = 1.0 - distances
                valid_mask = (sims >= min_similarity) & (indices >= 0)

                if not bool(cp.any(valid_mask)):
                    continue

                chunk_rows = cp.broadcast_to(
                    cp.arange(q_start, q_end, dtype=cp.int64)[:, None],
                    indices.shape,
                )

                matched_s1_list.append(cp.asnumpy(chunk_rows[valid_mask]))
                matched_pool_list.append(cp.asnumpy(indices[valid_mask]))
        except Exception as e_query:
            logger.warning(
                "cuML kneighbors query encountered an issue (%s); falling back to Faiss.",
                e_query,
            )
            use_cuml_nn = False
            matched_s1_list.clear()
            matched_pool_list.clear()

    # Fallback to Faiss if cuML NearestNeighbors failed or was bypassed
    if not use_cuml_nn or not matched_s1_list:
        try:
            import faiss
            logger.info("Running Faiss IndexFlatIP on CPU for 128-dim dense matching...")
            M_pool_np = cp.asnumpy(M_pool_dense)
            M_s1_np = cp.asnumpy(M_s1_dense)
            dim = M_pool_np.shape[1]

            index = faiss.IndexFlatIP(dim)
            index.add(M_pool_np)

            query_batch = max(5000, min(batch_size, 100000))
            for q_start in tqdm(
                range(0, n_s1, query_batch),
                desc="Faiss KNN IP Batches",
                leave=False,
            ):
                q_end = min(q_start + query_batch, n_s1)
                q_chunk = M_s1_np[q_start:q_end]

                sims, indices = index.search(q_chunk, actual_k)
                valid_mask = (sims >= min_similarity) & (indices >= 0)

                if not np.any(valid_mask):
                    continue

                chunk_rows = np.broadcast_to(
                    np.arange(q_start, q_end, dtype=np.int64)[:, None],
                    indices.shape,
                )
                matched_s1_list.append(chunk_rows[valid_mask])
                matched_pool_list.append(indices[valid_mask])
        except Exception as e_faiss:
            if not matched_s1_list:
                logger.error("Faiss KNN fallback failed with %s", e_faiss)
                raise

    # Cleanup VRAM allocations
    del M_pool_dense
    del M_s1_dense
    cp.get_default_memory_pool().free_all_blocks()

    if matched_s1_list:
        return np.concatenate(matched_s1_list), np.concatenate(matched_pool_list)
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def gpu_sparse_dot_topk(
    M_s1,
    M_pool_T,
    top_k: int = 20,
    min_similarity: float = 0.20,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes top-K candidate matches per S1 query row using GPU cuSPARSE and CuPy.

    Iterates through batches of M_s1, computes the sparse dot product against M_pool_T,
    prunes elements below min_similarity, and performs GPU-accelerated argpartition/argsort
    to extract the top-K column indices per row, returning host numpy arrays.

    Parameters:
      M_s1: CuPy sparse CSR matrix of S1 query vectors (shape: N x D).
      M_pool_T: CuPy sparse CSR matrix of transposed candidate pool vectors (shape: D x M).
      top_k: Maximum candidate indices to keep per S1 query row.
      min_similarity: Minimum cosine similarity threshold.
      batch_size: Number of S1 queries to process concurrently on GPU.
                  Dynamically scaled to bound dense memory allocation to ~1 GB VRAM.

    Returns:
      Tuple of (matched_s1_local_indices, matched_pool_local_indices) as numpy int64 arrays.
    """
    n_s1 = M_s1.shape[0]
    n_candidates = M_pool_T.shape[1]

    if n_s1 == 0 or n_candidates == 0 or top_k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Calculate optimal GPU batch size to bound dense memory to ~2 GB to prevent cuSPARSE workspace explosions
    max_dense_elements = 500_000_000
    safe_gpu_batch = max(50, min(5000, max_dense_elements // max(1, n_candidates)))

    if batch_size is None or batch_size > safe_gpu_batch:
        effective_batch_size = safe_gpu_batch
    else:
        effective_batch_size = max(1, batch_size)

    actual_k = min(top_k, n_candidates)
    matched_s1_list: List[np.ndarray] = []
    matched_pool_list: List[np.ndarray] = []

    for i in tqdm(
        range(0, n_s1, effective_batch_size),
        desc="GPU TF-IDF Batches",
        leave=False,
    ):
        end_idx = min(i + effective_batch_size, n_s1)
        chunk = M_s1[i:end_idx]

        # 1. Sparse dot product on GPU: (B, D) x (D, M) -> (B, M)
        sim_chunk = chunk.dot(M_pool_T)

        # 2. Convert to dense array on GPU
        if hasattr(sim_chunk, "toarray"):
            sim_dense = sim_chunk.toarray()
        else:
            sim_dense = cp.asarray(sim_chunk.todense())

        B = sim_dense.shape[0]

        # 3. Filter entries below similarity threshold
        if min_similarity > 0.0:
            sim_dense[sim_dense < min_similarity] = 0.0

        # 4. Top-K extraction using CuPy argpartition and argsort
        if n_candidates > actual_k:
            topk_idx = cp.argpartition(-sim_dense, actual_k - 1, axis=1)[:, :actual_k]
            topk_vals = sim_dense[cp.arange(B)[:, None], topk_idx]

            # Sort the top-k in descending order
            order = cp.argsort(-topk_vals, axis=1)
            sorted_cols = topk_idx[cp.arange(B)[:, None], order]
            sorted_vals = topk_vals[cp.arange(B)[:, None], order]
        else:
            # Full sort if total candidates <= actual_k
            sorted_cols = cp.argsort(-sim_dense, axis=1)
            sorted_vals = sim_dense[cp.arange(B)[:, None], sorted_cols]

        # 5. Extract only valid pairs >= min_similarity (and > 0.0)
        if min_similarity > 0.0:
            valid_mask = sorted_vals >= min_similarity
        else:
            valid_mask = sorted_vals > 0.0

        if not bool(cp.any(valid_mask)):
            continue

        # Row indices for the batch: shape (B, 1) broadcasted to (B, actual_k)
        batch_rows = cp.broadcast_to(
            cp.arange(i, end_idx, dtype=cp.int64)[:, None],
            sorted_cols.shape,
        )

        matched_rows_cp = batch_rows[valid_mask]
        matched_cols_cp = sorted_cols[valid_mask]

        matched_s1_list.append(cp.asnumpy(matched_rows_cp))
        matched_pool_list.append(cp.asnumpy(matched_cols_cp))

    # Free memory pool blocks allocated during batch processing
    cp.get_default_memory_pool().free_all_blocks()

    if matched_s1_list:
        return np.concatenate(matched_s1_list), np.concatenate(matched_pool_list)
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def _block_channel_tfidf_ngram_gpu(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    s1_exprs: List[pl.Expr],
    pool_exprs: List[pl.Expr],
    s1_id_col: str,
    pool_id_col: str,
    top_k: int,
    min_similarity: float,
    ngram_range: Tuple[int, int],
    batch_size: int,
) -> pl.DataFrame:
    """
    GPU-accelerated Channel 3 TF-IDF n-gram candidate retrieval using cuML, cuDF, and CuPy.
    """
    import cudf

    empty_res = pl.DataFrame(
        schema={s1_id_col: pl.UInt32, pool_id_col: pl.UInt32}
    )

    # For GPU path: directly use Arrow to bypass Python lists
    pool_sel = pool_df.select(pl.coalesce(pool_exprs).fill_null("").str.strip_chars())
    try:
        pool_arrow = pool_sel.get_column(0).to_arrow()
    except (TypeError, Exception):
        pool_arrow = pool_sel.to_series(0).to_arrow()
    pool_cudf = cudf.Series(pool_arrow)
    del pool_arrow

    s1_sel = s1_df.select(pl.coalesce(s1_exprs).fill_null("").str.strip_chars())
    try:
        s1_arrow = s1_sel.get_column(0).to_arrow()
    except (TypeError, Exception):
        s1_arrow = s1_sel.to_series(0).to_arrow()
    s1_cudf = cudf.Series(s1_arrow)
    del s1_arrow

    if len(pool_cudf) == 0 or len(s1_cudf) == 0:
        del pool_cudf
        del s1_cudf
        return empty_res

    min_df = 2 if len(pool_cudf) > 100 else 1

    # Instantiate cuML TfidfVectorizer with robust argument fallback
    vec_kwargs = {
        "analyzer": "char",
        "ngram_range": ngram_range,
        "norm": "l2",
        "sublinear_tf": True,
        "min_df": min_df,
    }

    vectorizer = None
    for attempt_args in [
        vec_kwargs,
        {k: v for k, v in vec_kwargs.items() if k != "min_df"},
        {"analyzer": "char", "ngram_range": ngram_range, "norm": "l2", "sublinear_tf": True},
        {"analyzer": "char", "ngram_range": ngram_range, "norm": "l2"},
        {"analyzer": "char", "ngram_range": ngram_range},
    ]:
        try:
            vectorizer = cuTfidfVectorizer(**attempt_args)
            break
        except TypeError:
            continue

    if vectorizer is None:
        del pool_cudf
        del s1_cudf
        raise RuntimeError("Failed to initialize cuML TfidfVectorizer.")

    # Now fit transform the cudf Series
    try:
        M_pool = vectorizer.fit_transform(pool_cudf)
        M_s1 = vectorizer.transform(s1_cudf)
    except ValueError:
        # Handles empty vocabulary
        del pool_cudf
        del s1_cudf
        return empty_res

    # Free cudf Series and vectorizer to clean VRAM before heavy dot product loop
    del pool_cudf
    del s1_cudf
    del vectorizer
    cp.get_default_memory_pool().free_all_blocks()

    # Convert to CuPy CSR format if needed
    if not isinstance(M_pool, cp_sparse.csr_matrix):
        if hasattr(M_pool, "tocsr"):
            M_pool = M_pool.tocsr()
        else:
            M_pool = cp_sparse.csr_matrix(M_pool)

    if not isinstance(M_s1, cp_sparse.csr_matrix):
        if hasattr(M_s1, "tocsr"):
            M_s1 = M_s1.tocsr()
        else:
            M_s1 = cp_sparse.csr_matrix(M_s1)

    # Query GPU top-K using TruncatedSVD and NearestNeighbors (LSA cosine matching)
    s1_local_idx, pool_local_idx = gpu_knn_lsa_topk(
        M_s1=M_s1,
        M_pool=M_pool,
        top_k=top_k,
        min_similarity=min_similarity,
        n_components=128,
        batch_size=batch_size,
    )

    del M_s1
    del M_pool
    cp.get_default_memory_pool().free_all_blocks()

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

    for i in tqdm(range(n_rows), desc="CPU Sparse Dot Top-K", leave=False, miniters=10000):
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

    If sparse_dot_topn is installed, it leverages native C++ sp_matmul_topn /
    awesome_cossim_topn. Otherwise, falls back to chunked CSR matrix multiplication
    with vectorized argpartition to prevent high memory usage.

    Parameters:
      A: CSR matrix of queries (shape: N x D).
      B_T: CSR matrix of transposed candidates (shape: D x M).
      top_k: Maximum candidate indices to keep per row.
      min_similarity: Minimum cosine similarity threshold.
      batch_size: Batch size of rows of A to process per step.
      n_jobs: Number of threads to use (-1 for all CPU cores).

    Returns:
      Tuple of (row_indices, col_indices) as numpy int64 arrays.
    """
    n_queries = A.shape[0]
    n_candidates = B_T.shape[1]

    if n_queries == 0 or n_candidates == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    effective_threads = max(1, os.cpu_count() or 1) if n_jobs <= 0 else n_jobs

    # 1. Native sparse_dot_topn path if available
    if _SPARSE_DOT_TOPN_MODE == "sp_matmul_topn":
        try:
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

    # 2. Optimized chunked CSR multiplication fallback
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
    use_gpu: Optional[bool] = None,
) -> pl.DataFrame:
    """
    Channel 3: Character N-gram TF-IDF cosine similarity top-K candidate retrieval.

    If cuML and CuPy are available (and not disabled via use_gpu=False), vectors are
    computed on GPU using cuML's TfidfVectorizer and queried via CuPy sparse matrix
    multiplication. Otherwise, runs the optimized CPU path.

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
      n_jobs: Number of CPU worker threads for CPU fallback.
      use_gpu: Optional boolean flag to force GPU (True) or CPU (False). Defaults to auto-detect.

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

    # 1. GPU RAPIDS accelerated path if available
    gpu_active = HAS_GPU_RAPIDS_CUML if use_gpu is None else (use_gpu and HAS_GPU_RAPIDS_CUML)
    if gpu_active:
        try:
            return _block_channel_tfidf_ngram_gpu(
                s1_df=s1_df,
                pool_df=pool_df,
                s1_exprs=s1_exprs,
                pool_exprs=pool_exprs,
                s1_id_col=s1_id_col,
                pool_id_col=pool_id_col,
                top_k=top_k,
                min_similarity=min_similarity,
                ngram_range=ngram_range,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.warning(
                "GPU TF-IDF blocking failed with error: %s; falling back to CPU implementation.",
                e,
            )

    # 2. CPU fallback path: only extract Python lists when running on CPU
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
    min_df = 2 if len(pool_text_series) > 100 else 1
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        norm="l2",
        sublinear_tf=True,
    )

    try:
        M_pool = vectorizer.fit_transform(pool_text_series)
        M_s1 = vectorizer.transform(s1_text_series)
    except ValueError:
        # Handles empty vocabulary edge cases
        return empty_res

    M_pool_T = M_pool.T.tocsr()

    # Query sparse top-k on CPU
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
         (GPU RAPIDS cuML/CuPy accelerated with CPU fallback)

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
        use_gpu: Optional[bool] = None,
    ):
        self.ngram_range = ngram_range
        self.tfidf_top_k = tfidf_top_k
        self.min_tfidf_sim = min_tfidf_sim
        self.max_exact_per_key = max_exact_per_key
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu

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

        # Channel 3: TF-IDF char n-grams cosine similarity top-20 (GPU or CPU)
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
            use_gpu=self.use_gpu,
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
    use_gpu: Optional[bool] = None,
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
        use_gpu=use_gpu,
    )
    return blocker.block_country(
        s1_df=s1_df,
        pool_df=pool_df,
        s1_id_col=s1_id_col,
        pool_id_col=pool_id_col,
    )
