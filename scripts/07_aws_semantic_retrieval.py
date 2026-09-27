"""
AWS Multi-GPU Semantic Candidate Generation using SentenceTransformers and FAISH.

This script performs semantic similarity-based candidate retrieval for entity resolution:
- Encodes S1 queries and S2+S3 pool using multilingual SentenceTransformers
- Leverages multi-GPU parallelism via start_multi_process_pool()
- Builds FAISS IndexFlatIP for cosine similarity search
- Outputs top-K candidates per S1 entity for downstream re-ranking

Usage:
    python scripts/07_aws_semantic_retrieval.py --split test --top-k 5
    python scripts/07_aws_semantic_retrieval.py --split train --top-k 10 --batch-size 512

Dependencies:
    pip install sentence-transformers faiss-gpu polars pyarrow

Requirements:
    - AWS g6.12xlarge or g6e.4xlarge (multi-GPU)
    - sentence-transformers, faiss-gpu, polars
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import polars as pl
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def format_entity_text(df: pl.DataFrame) -> List[str]:
    """
    Format entity records as text strings for encoding.

    Format: "{name} | {address} | {country}"
    Example: "Zephay Labs Inc | 2621 Cotten Road, Tyler, TX | US"

    Args:
        df: Polars DataFrame with name_no_legal, address_clean, country columns

    Returns:
        List of formatted text strings
    """
    logger.info(f"Formatting {len(df):,} entity records as text...")

    # Handle missing columns with fallbacks
    name_col = "name_no_legal" if "name_no_legal" in df.columns else (
        "business_name" if "business_name" in df.columns else None
    )
    address_col = "address_clean" if "address_clean" in df.columns else (
        "address" if "address" in df.columns else None
    )
    country_col = "country" if "country" in df.columns else None

    # Build expressions with fallbacks
    name_expr = pl.col(name_col) if name_col else pl.lit("")
    address_expr = pl.col(address_col) if address_col else pl.lit("")
    country_expr = pl.col(country_col) if country_col else pl.lit("")

    # Use coalesce to handle nulls, fallback to empty string
    texts = (
        df.select([
            pl.coalesce(name_expr, pl.lit("")).alias("name"),
            pl.coalesce(address_expr, pl.lit("")).alias("address"),
            pl.coalesce(country_expr, pl.lit("")).alias("country")
        ])
        .select(
            (pl.col("name") + pl.lit(" | ") + pl.col("address") + pl.lit(" | ") + pl.col("country"))
            .alias("text")
        )
        .get_column("text")
        .to_list()
    )

    if texts:
        logger.info(f"Sample formatted text: {texts[0]}")
    else:
        logger.warning("No texts generated - check column names")

    return texts


def resolve_split_path(data_dir: Path, split: str, filename: str) -> Path:
    """Resolves file path checking split subdirectory first, then data_dir."""
    candidates = [
        data_dir / split / filename,
        data_dir / filename,
        REPO_ROOT / "data" / "parquet" / split / filename,
        REPO_ROOT / "data" / "parquet" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find '{filename}' for split '{split}' in {[str(c) for c in candidates]}")


def load_and_prepare_data(split: str, data_dir: Path) -> Tuple[pl.DataFrame, pl.DataFrame, List[str], List[str]]:
    """
    Load S1 queries and S2+S3 pool, format as text.

    Args:
        split: "train" or "test"
        data_dir: Path to data/parquet directory

    Returns:
        (s1_df, pool_df, s1_texts, pool_texts)
    """
    logger.info(f"Loading {split} data from {data_dir}...")

    # Load S1 (queries)
    s1_path = resolve_split_path(data_dir, split, f"{split}_source1.parquet")
    s1_df = pl.read_parquet(s1_path)
    logger.info(f"Loaded {len(s1_df):,} S1 entities from {s1_path}")

    # Load S2
    s2_path = resolve_split_path(data_dir, split, f"{split}_source2.parquet")
    s2_df = pl.read_parquet(s2_path)
    logger.info(f"Loaded {len(s2_df):,} S2 records from {s2_path}")

    # Load S3
    s3_path = resolve_split_path(data_dir, split, f"{split}_source3.parquet")
    s3_df = pl.read_parquet(s3_path)
    logger.info(f"Loaded {len(s3_df):,} S3 records from {s3_path}")

    # Resolve id_map if available
    id_map = None
    for p in [data_dir / "id_map.parquet", REPO_ROOT / "data" / "parquet" / "id_map.parquet"]:
        if p.exists():
            id_map = pl.read_parquet(p)
            break

    # Ensure source1_entity_id_int in s1_df
    if "source1_entity_id_int" not in s1_df.columns:
        if "entity_id_int" in s1_df.columns:
            s1_df = s1_df.with_columns(pl.col("entity_id_int").alias("source1_entity_id_int"))
        elif id_map is not None and "entity_id" in s1_df.columns:
            s1_df = s1_df.join(id_map.rename({"entity_id_int": "source1_entity_id_int"}), on="entity_id", how="left")
        else:
            s1_df = s1_df.with_columns(pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("source1_entity_id_int"))

    s1_df = s1_df.with_columns(pl.col("source1_entity_id_int").cast(pl.UInt32))

    # Ensure candidate entity_id_int in s2_df and s3_df
    for df_name, df_ref in [("s2", s2_df), ("s3", s3_df)]:
        cand_id_col = "candidate_entity_id_int" if "candidate_entity_id_int" in df_ref.columns else "entity_id_int"
        if cand_id_col not in df_ref.columns:
            if id_map is not None and "entity_id" in df_ref.columns:
                df_ref = df_ref.join(id_map, on="entity_id", how="left")
            else:
                df_ref = df_ref.with_columns(pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("entity_id_int"))
        if df_name == "s2":
            s2_df = df_ref.with_columns(pl.col("entity_id_int" if "entity_id_int" in df_ref.columns else cand_id_col).cast(pl.UInt32).alias("entity_id_int"))
        else:
            s3_df = df_ref.with_columns(pl.col("entity_id_int" if "entity_id_int" in df_ref.columns else cand_id_col).cast(pl.UInt32).alias("entity_id_int"))

    # Common schema projection before vertical concat
    common_cols = [c for c in ["entity_id_int", "name_no_legal", "business_name", "address_clean", "address", "business_address", "country"] if c in s2_df.columns and c in s3_df.columns]
    pool_df = pl.concat([s2_df.select(common_cols), s3_df.select(common_cols)], how="vertical")
    logger.info(f"Combined pool size: {len(pool_df):,} records")

    # Format as text
    s1_texts = format_entity_text(s1_df)
    pool_texts = format_entity_text(pool_df)

    return s1_df, pool_df, s1_texts, pool_texts


def encode_in_batches(model, texts: List[str], batch_size: int, description: str = "Encoding") -> np.ndarray:
    """
    Encode texts in pre-allocated array with GPU acceleration and progress tracking.

    Args:
        model: SentenceTransformer model
        texts: List of text strings
        batch_size: Batch size for encoding
        description: Description for progress logging

    Returns:
        Normalized embeddings as np.ndarray (N, embedding_dim)
    """
    dim = model.get_sentence_embedding_dimension()
    n_texts = len(texts)
    logger.info(f"{description} {n_texts:,} texts (embedding dim={dim}) with batch_size={batch_size}...")

    embeddings = np.empty((n_texts, dim), dtype=np.float32)
    t0 = time.time()

    for i in range(0, n_texts, batch_size):
        end_idx = min(i + batch_size, n_texts)
        batch_texts = texts[i:end_idx]
        batch_embs = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings[i:end_idx] = batch_embs

        if (i // batch_size) % 50 == 0 or end_idx >= n_texts:
            elapsed = time.time() - t0
            rate = end_idx / elapsed if elapsed > 0 else 0
            logger.info(f"  [{description}] {end_idx:,}/{n_texts:,} ({end_idx/n_texts*100:.1f}%) | {rate:.0f} texts/sec")

    return embeddings


class NativeTransformerEmbedder:
    """
    Native PyTorch + HuggingFace Transformers embedding engine.
    Completely bypasses sentence-transformers and datasets packages to avoid
    conda metadata incompatibilities in Python 3.12 (e.g. TF_VERSION NoneType bug).
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        hf_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
        logger.info(f"Loading native HuggingFace model: {hf_id} on {device}...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).to(self.device)
        self.model.eval()
        self.dim = self.model.config.hidden_size
        logger.info(f"Native model loaded on {self.device}. Hidden dim: {self.dim}")

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 512,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda"), dtype=torch.bfloat16):
                    out = self.model(**encoded)
                    tok_embs = out[0]
                    mask = encoded["attention_mask"].unsqueeze(-1).expand(tok_embs.size()).float()
                    sum_embs = torch.sum(tok_embs * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    sent_embs = sum_embs / sum_mask

                    if normalize_embeddings:
                        sent_embs = F.normalize(sent_embs, p=2, dim=1)

            all_embs.append(sent_embs.cpu().to(torch.float32).numpy())

        return np.vstack(all_embs)


def encode_with_multi_gpu(model_or_name, texts: List[str], batch_size: int, model_name_str: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """
    Encode texts using all available GPUs in parallel (e.g. 4x L40S on ml.g6e.16xlarge).
    Dispatches 1 independent model replica per GPU across parallel threads.
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        logger.info(f"Single GPU/CPU detected (gpu_count={gpu_count}). Using batched encode()...")
        return encode_in_batches(model_or_name, texts, batch_size, "Single GPU/CPU encoding")

    logger.info(f"Detected {gpu_count} GPUs. Distributing encoding across all {gpu_count} GPUs in parallel...")
    n_texts = len(texts)
    chunk_size = (n_texts + gpu_count - 1) // gpu_count

    resolved_name = getattr(model_or_name, "model_name_or_path", None) or model_name_str

    def _worker(gpu_id: int) -> Tuple[int, int, np.ndarray]:
        st = gpu_id * chunk_size
        en = min(st + chunk_size, n_texts)
        if st >= n_texts:
            return st, en, np.empty((0, 384), dtype=np.float32)

        sub_texts = texts[st:en]
        logger.info(f"  [GPU {gpu_id}] Initializing model replica on cuda:{gpu_id} ({len(sub_texts):,} texts)...")
        embedder = NativeTransformerEmbedder(resolved_name, device=f"cuda:{gpu_id}")
        sub_embs = encode_in_batches(embedder, sub_texts, batch_size, f"GPU {gpu_id}")
        return st, en, sub_embs

    with ThreadPoolExecutor(max_workers=gpu_count) as executor:
        results = list(executor.map(_worker, range(gpu_count)))

    dim = 384
    for _, _, embs in results:
        if embs.shape[0] > 0:
            dim = embs.shape[1]
            break

    all_embeddings = np.empty((n_texts, dim), dtype=np.float32)
    for st, en, embs in results:
        if embs.shape[0] > 0:
            all_embeddings[st:en] = embs

    logger.info(f"Multi-GPU encoding across all {gpu_count} GPUs completed. Shape: {all_embeddings.shape}")
    return all_embeddings


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> "faiss.Index":
    """
    Build FAISS IndexFlatIP (Inner Product for cosine similarity).
    Distributes across all GPUs on multi-GPU instances (e.g. 4x L40S on ml.g6e.16xlarge).

    Args:
        embeddings: Normalized embeddings (N, D)
        use_gpu: Whether to attempt GPU index

    Returns:
        FAISS index
    """
    import faiss

    dimension = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP with dimension {dimension}...")

    index = faiss.IndexFlatIP(dimension)

    # Move to GPU(s)
    if use_gpu:
        try:
            gpu_count = faiss.get_num_gpus()
            if gpu_count > 1:
                logger.info(f"Detected {gpu_count} GPUs in FAISS. Sharding index across all GPUs...")
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                index = faiss.index_cpu_to_all_gpus(index, co, ngpu=gpu_count)
                logger.info(f"FAISS index successfully sharded across {gpu_count} GPUs")
            elif gpu_count == 1:
                res = faiss.StandardGpuResources()
                res.setTempMemory(1536 * 1024 * 1024)
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index successfully moved to single GPU")
        except Exception as e:
            logger.warning(f"Could not initialize GPU FAISS: {e}. Using multithreaded CPU index.")

    # Add vectors
    logger.info(f"Adding {len(embeddings):,} vectors to index...")
    index.add(embeddings.astype(np.float32))
    logger.info(f"Index built. Total vectors: {index.ntotal:,}")

    return index

    # Add vectors
    logger.info(f"Adding {len(embeddings):,} vectors to index...")
    index.add(embeddings.astype(np.float32))
    logger.info(f"Index built. Total vectors: {index.ntotal:,}")

    return index


def search_candidates(
    index: "faiss.Index",
    query_embeddings: np.ndarray,
    top_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for top-K nearest neighbors.

    Args:
        index: FAISS index
        query_embeddings: Query vectors (N_queries, D)
        top_k: Number of neighbors to retrieve per query

    Returns:
        (distances, indices) - both shape (N_queries, top_k)
    """
    logger.info(f"Searching index for top-{top_k} candidates per query...")
    start_time = time.time()

    distances, indices = index.search(query_embeddings.astype(np.float32), top_k)

    elapsed = time.time() - start_time
    logger.info(f"Search complete in {elapsed:.2f}s. Throughput: {len(query_embeddings) / elapsed:.0f} queries/sec")

    return distances, indices


def build_candidate_dataframe(
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    distances: np.ndarray,
    indices: np.ndarray
) -> pl.DataFrame:
    """
    Build output DataFrame with S1 entity ID, candidate entity ID, and semantic score.

    Args:
        s1_df: S1 DataFrame with source1_entity_id_int
        pool_df: Pool DataFrame (S2 + S3) with entity_id_int
        distances: FAISS distances (cosine similarities)
        indices: FAISS indices into pool

    Returns:
        Polars DataFrame with (source1_entity_id_int, candidate_entity_id_int, semantic_score)
    """
    logger.info("Building candidate pairs DataFrame...")

    n_queries, top_k = distances.shape

    # Flatten arrays
    s1_ids = np.repeat(s1_df.get_column("source1_entity_id_int").to_numpy(), top_k)
    pool_indices_flat = indices.flatten()
    scores_flat = distances.flatten()

    # Map pool indices to entity_id_int
    pool_id_array = pool_df.get_column("entity_id_int").to_numpy()
    candidate_ids = pool_id_array[pool_indices_flat]

    # Build DataFrame
    candidates_df = pl.DataFrame({
        "source1_entity_id_int": s1_ids,
        "candidate_entity_id_int": candidate_ids,
        "semantic_score": scores_flat
    })

    # Filter out invalid candidates (score <= 0 indicates padding or no match)
    candidates_df = candidates_df.filter(pl.col("semantic_score") > 0.0)

    logger.info(f"Built {len(candidates_df):,} candidate pairs")
    logger.info(f"Average candidates per S1 entity: {len(candidates_df) / n_queries:.2f}")

    return candidates_df


def main():
    parser = argparse.ArgumentParser(description="AWS Multi-GPU Semantic Candidate Generation")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Data split to process (train or test)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of semantic candidates to retrieve per S1 entity (default: 15, optimal for high recall)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size per GPU"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name (crucial for French/Hindi)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/parquet"),
        help="Path to parquet data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/parquet"),
        help="Path to output directory"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration for FAISS (default: True)"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("AWS Multi-GPU Semantic Candidate Generation")
    logger.info("=" * 80)
    logger.info(f"Split: {args.split}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    try:
        import faiss
        logger.info(f"FAISS version: {faiss.__version__}")
    except ImportError:
        logger.error("FAISS not installed. Run: pip install faiss-gpu (or faiss-cpu)")
        sys.exit(1)

    # Load Model (try sentence-transformers first, fall back to native transformers on conda/datasets conflict)
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer model: {args.model_name}...")
        model = SentenceTransformer(args.model_name)
    except Exception as e:
        logger.warning(
            f"SentenceTransformer import/init failed ({e}). "
            f"Using native HuggingFace Transformers fallback..."
        )
        model = NativeTransformerEmbedder(args.model_name)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Load and prepare data
    s1_df, pool_df, s1_texts, pool_texts = load_and_prepare_data(args.split, args.data_dir)

    # Encode pool (S2 + S3)
    logger.info("=" * 80)
    logger.info("Encoding pool (S2 + S3)...")
    logger.info("=" * 80)
    pool_embeddings = encode_with_multi_gpu(model, pool_texts, args.batch_size, args.model_name)

    # Encode queries (S1)
    logger.info("=" * 80)
    logger.info("Encoding queries (S1)...")
    logger.info("=" * 80)
    s1_embeddings = encode_with_multi_gpu(model, s1_texts, args.batch_size, args.model_name)

    # Build FAISS index
    logger.info("=" * 80)
    logger.info("Building FAISS index...")
    logger.info("=" * 80)
    index = build_faiss_index(pool_embeddings, use_gpu=args.use_gpu)

    # Search for candidates
    logger.info("=" * 80)
    logger.info("Searching for semantic candidates...")
    logger.info("=" * 80)
    distances, indices = search_candidates(index, s1_embeddings, args.top_k)

    # Build output DataFrame
    candidates_df = build_candidate_dataframe(s1_df, pool_df, distances, indices)

    # Save output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.split}_semantic_candidates.parquet"

    logger.info(f"Saving candidates to {output_path}...")
    candidates_df.write_parquet(output_path)

    # Summary statistics
    logger.info("=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)
    logger.info(f"Total S1 entities: {len(s1_df):,}")
    logger.info(f"Total candidate pairs: {len(candidates_df):,}")
    logger.info(f"Avg candidates per entity: {len(candidates_df) / len(s1_df):.2f}")
    logger.info(f"Min semantic score: {candidates_df.get_column('semantic_score').min():.4f}")
    logger.info(f"Max semantic score: {candidates_df.get_column('semantic_score').max():.4f}")
    logger.info(f"Mean semantic score: {candidates_df.get_column('semantic_score').mean():.4f}")
    logger.info(f"Output saved: {output_path}")
    logger.info("=" * 80)
    logger.info("Semantic candidate generation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()