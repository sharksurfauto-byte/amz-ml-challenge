#!/usr/bin/env python3
"""
Phase 2 Multi-Channel Candidate Generation Pipeline for Amazon ML Challenge 2026.

Steps:
  1. Load S1, S2, and S3 Parquet tables for train and test splits.
  2. Merge S2 and S3 into a single searchable pool per split, tracking entity_id_int.
  3. Partition by country:
     - Subset S1 and searchable pool to country.
     - Channel 1: Exact matching on `name_token_sorted_key`.
     - Channel 2: Exact matching on `name_no_legal`.
     - Channel 3: TF-IDF char 3-4 n-gram cosine similarity top-20 on `name_no_legal`.
     - Union, deduplicate (preserving priority), and cap to top-20 candidates per S1 entity.
  4. Save results:
     - `data/parquet/train_candidates.parquet`
     - `data/parquet/test_candidates.parquet`
     Schema: `source1_entity_id_int` (uint32), `candidate_entity_id_int` (uint32).
  5. If ground truth is available on train, evaluate Candidate Recall Ceiling.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.blocking_lsa import MultiChannelBlocker
from src.normalize import (
    address_keys,
    clean_text,
    extract_and_remove_legal_forms,
    kanan_transliterate_devanagari,
    remove_junk_tokens,
    token_sort_key,
)

logger = logging.getLogger("candidate_gen")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def resolve_parquet_file(base_dir: Path, *candidates: str) -> Optional[Path]:
    """Resolves an existing file path from candidate relative paths."""
    for rel in candidates:
        p = base_dir / rel
        if p.exists():
            return p
    return None


def ensure_id_int_and_keys(
    df: pl.DataFrame,
    id_map: Optional[pl.DataFrame] = None,
    id_col_name: str = "entity_id_int",
) -> pl.DataFrame:
    """
    Ensures the DataFrame contains an integer ID column (uint32) and normalized keys.
    Computes normalized keys on the fly if not already present.
    """
    # 1. Resolve entity_id_int
    if id_col_name not in df.columns and "entity_id_int" not in df.columns:
        if id_map is not None and "entity_id" in df.columns:
            logger.info("    Joining entity_id_int from id_map...")
            df = df.join(id_map, on="entity_id", how="left")
        else:
            logger.info("    Generating contiguous uint32 IDs on the fly...")
            df = df.with_columns(
                pl.int_range(0, pl.len(), dtype=pl.UInt32).alias(id_col_name)
            )

    if id_col_name not in df.columns and "entity_id_int" in df.columns:
        df = df.rename({"entity_id_int": id_col_name})

    df = df.with_columns(pl.col(id_col_name).cast(pl.UInt32))

    # 2. Check and compute normalized keys if missing
    needed_keys = ["name_token_sorted_key", "name_no_legal"]
    missing_keys = [k for k in needed_keys if k not in df.columns]

    if missing_keys:
        logger.info("    Computing missing normalization keys %s on the fly...", missing_keys)
        bname = pl.col("business_name") if "business_name" in df.columns else pl.lit("")
        name_prep = kanan_transliterate_devanagari(remove_junk_tokens(bname))
        cleaned_name = clean_text(name_prep)
        no_legal, legal_form = extract_and_remove_legal_forms(cleaned_name)

        sorted_key = (
            pl.when(no_legal.is_not_null() & (no_legal != ""))
            .then(token_sort_key(no_legal))
            .otherwise(token_sort_key(cleaned_name))
        )

        df = df.with_columns([
            no_legal.alias("name_no_legal"),
            sorted_key.alias("name_token_sorted_key"),
        ])

    return df


def load_dataset_split(
    parquet_dir: Path,
    split: str,
    id_map: Optional[pl.DataFrame] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Loads Source 1 and merges Source 2 and Source 3 into a single searchable pool.

    Returns:
      (s1_df, pool_df)
      s1_df schema: source1_entity_id_int, country, name_token_sorted_key, name_no_legal, business_name
      pool_df schema: candidate_entity_id_int, country, name_token_sorted_key, name_no_legal, business_name
    """
    t0 = time.time()
    logger.info("Loading %s datasets from %s...", split.upper(), parquet_dir.resolve())

    # Resolve paths for S1, S2, S3
    s1_path = resolve_parquet_file(
        parquet_dir,
        f"{split}/{split}_source1.parquet",
        f"{split}_source1.parquet",
    )
    s2_path = resolve_parquet_file(
        parquet_dir,
        f"{split}/{split}_source2.parquet",
        f"{split}_source2.parquet",
    )
    s3_path = resolve_parquet_file(
        parquet_dir,
        f"{split}/{split}_source3.parquet",
        f"{split}_source3.parquet",
    )

    if not s1_path or not s2_path or not s3_path:
        raise FileNotFoundError(
            f"Could not locate all Parquet files for {split} split in {parquet_dir}.\n"
            f"  S1: {s1_path}\n  S2: {s2_path}\n  S3: {s3_path}"
        )

    logger.info("  - Reading S1: %s", s1_path.name)
    df_s1 = pl.read_parquet(s1_path)
    df_s1 = ensure_id_int_and_keys(df_s1, id_map=id_map, id_col_name="source1_entity_id_int")

    logger.info("  - Reading S2: %s", s2_path.name)
    df_s2 = pl.read_parquet(s2_path)
    df_s2 = ensure_id_int_and_keys(df_s2, id_map=id_map, id_col_name="candidate_entity_id_int")

    logger.info("  - Reading S3: %s", s3_path.name)
    df_s3 = pl.read_parquet(s3_path)
    df_s3 = ensure_id_int_and_keys(df_s3, id_map=id_map, id_col_name="candidate_entity_id_int")

    # Select common schema columns for searchable pool
    pool_cols = ["candidate_entity_id_int", "country", "name_token_sorted_key", "name_no_legal"]
    for opt_col in ["business_name", "name_clean"]:
        if opt_col in df_s2.columns and opt_col in df_s3.columns:
            pool_cols.append(opt_col)

    s1_cols = ["source1_entity_id_int", "country", "name_token_sorted_key", "name_no_legal"]
    for opt_col in ["business_name", "name_clean"]:
        if opt_col in df_s1.columns:
            s1_cols.append(opt_col)

    s1_clean = df_s1.select(s1_cols)
    pool = pl.concat([
        df_s2.select(pool_cols),
        df_s3.select(pool_cols),
    ])

    elapsed = time.time() - t0
    logger.info(
        "Successfully loaded %s split: S1=%d rows, Pool (S2+S3)=%d rows in %.2fs",
        split.upper(),
        s1_clean.height,
        pool.height,
        elapsed,
    )
    return s1_clean, pool


def evaluate_ground_truth_recall(
    candidates_df: pl.DataFrame,
    gt_path: Path,
    id_map: Optional[pl.DataFrame] = None,
) -> None:
    """
    Evaluates the Candidate Recall Ceiling against ground truth.
    Candidate Recall Ceiling = (Captured GT Matches in Candidates) / (Total GT Matches) * 100
    """
    logger.info("Evaluating Candidate Recall Ceiling against %s...", gt_path.name)
    t0 = time.time()

    if gt_path.suffix == ".parquet":
        gt_df = pl.read_parquet(gt_path)
    else:
        gt_df = pl.read_csv(
            gt_path,
            separator="\t",
            infer_schema_length=5000,
        )

    # Resolve source1_entity_id column name
    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else "entity_id"

    # Filter non-empty matches and explode comma-separated IDs
    gt_pairs = (
        gt_df.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .select([s1_col, "matched_entity_ids"])
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .filter(pl.col("matched_entity_ids") != "")
        .rename({s1_col: "source1_entity_id", "matched_entity_ids": "candidate_entity_id"})
    )

    # Join integer IDs if needed
    if id_map is not None:
        id_map_s1 = id_map.rename({"entity_id": "source1_entity_id", "entity_id_int": "source1_entity_id_int"})
        id_map_cand = id_map.rename({"entity_id": "candidate_entity_id", "entity_id_int": "candidate_entity_id_int"})

        gt_pairs = (
            gt_pairs
            .join(id_map_s1, on="source1_entity_id", how="inner")
            .join(id_map_cand, on="candidate_entity_id", how="inner")
            .select(["source1_entity_id_int", "candidate_entity_id_int"])
        )

    total_gt = gt_pairs.height

    # Calculate intersection
    captured_df = gt_pairs.join(
        candidates_df,
        on=["source1_entity_id_int", "candidate_entity_id_int"],
        how="inner",
    )
    captured_count = captured_df.height
    recall_ceiling = (captured_count / total_gt * 100.0) if total_gt > 0 else 0.0

    # Calculate average candidates per S1
    cand_counts = candidates_df.group_by("source1_entity_id_int").len()
    avg_cands = cand_counts.select(pl.mean("len")).item() if cand_counts.height > 0 else 0.0

    elapsed = time.time() - t0
    logger.info("=" * 65)
    logger.info(" [CANDIDATE BLOCKING RECALL CEILING EVALUATION]")
    logger.info("=" * 65)
    logger.info(" Total Ground Truth Matches    : %d", total_gt)
    logger.info(" Captured True Matches in Pool : %d", captured_count)
    logger.info(" Candidate Recall Ceiling      : %.2f%%", recall_ceiling)
    logger.info(" Average Candidates per Entity : %.2f", avg_cands)
    logger.info(" Evaluation Time               : %.2fs", elapsed)
    logger.info("=" * 65)


def generate_candidates_for_split(
    parquet_dir: Path,
    output_dir: Path,
    split: str,
    max_candidates: int = 20,
    tfidf_top_k: int = 20,
    min_tfidf_sim: float = 0.20,
    max_exact_per_key: int = 200,
    batch_size: int = 25000,
    n_jobs: int = -1,
    compression: str = "zstd",
    id_map: Optional[pl.DataFrame] = None,
    eval_gt: bool = True,
) -> Path:
    """
    Executes multi-channel blocking partitioned strictly by country for a dataset split.
    Saves the resulting pairs to data/parquet/{split}_candidates.parquet.
    """
    split_start = time.time()
    logger.info("=" * 70)
    logger.info("STARTING CANDIDATE GENERATION FOR: %s", split.upper())
    logger.info("=" * 70)

    # 1. Load S1 and searchable pool (S2 + S3)
    s1_df, pool_df = load_dataset_split(parquet_dir=parquet_dir, split=split, id_map=id_map)

    # 2. Extract unique countries present in S1
    countries = (
        s1_df.select("country")
        .unique()
        .drop_nulls()
        .sort("country")
        .get_column("country")
        .to_list()
    )

    logger.info(
        "Identified %d unique countries in %s S1 reference set: %s",
        len(countries),
        split.upper(),
        ", ".join(str(c) for c in countries),
    )

    # 3. Instantiate MultiChannelBlocker
    blocker = MultiChannelBlocker(
        ngram_range=(3, 4),
        tfidf_top_k=tfidf_top_k,
        min_tfidf_sim=min_tfidf_sim,
        max_exact_per_key=max_exact_per_key,
        max_candidates_per_s1=max_candidates,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    country_pair_frames: List[pl.DataFrame] = []
    total_s1_processed = 0

    # 4. Iterate strictly by country partition
    for idx, country in enumerate(tqdm(countries, desc=f"Blocking {split.upper()} by Country"), start=1):
        c_start = time.time()
        s1_c = s1_df.filter(pl.col("country") == country)
        pool_c = pool_df.filter(pl.col("country") == country)

        total_s1_processed += s1_c.height

        if pool_c.height == 0:
            logger.debug(
                "[%d/%d] Country '%s': 0 candidate records in pool for %d S1 entities. Skipping.",
                idx,
                len(countries),
                country,
                s1_c.height,
            )
            continue

        logger.debug(
            "[%d/%d] Blocking country '%s' | S1: %d | Pool: %d...",
            idx,
            len(countries),
            country,
            s1_c.height,
            pool_c.height,
        )

        # Run multi-channel blocking on country subset
        pairs_c = blocker.block_country(
            s1_df=s1_c,
            pool_df=pool_c,
            s1_id_col="source1_entity_id_int",
            pool_id_col="candidate_entity_id_int",
        )

        c_elapsed = time.time() - c_start
        s1_with_cands = pairs_c.select("source1_entity_id_int").unique().height
        coverage_pct = (s1_with_cands / s1_c.height * 100.0) if s1_c.height > 0 else 0.0

        logger.debug(
            "  -> Country '%s' finished in %.2fs: %d candidate pairs | Coverage: %d/%d (%.2f%%)",
            country,
            c_elapsed,
            pairs_c.height,
            s1_with_cands,
            s1_c.height,
            coverage_pct,
        )

        if pairs_c.height > 0:
            country_pair_frames.append(pairs_c)

    # 5. Union all country candidate pairs
    if country_pair_frames:
        all_candidates = pl.concat(country_pair_frames)
    else:
        all_candidates = pl.DataFrame(
            schema={
                "source1_entity_id_int": pl.UInt32,
                "candidate_entity_id_int": pl.UInt32,
            }
        )

    # Ensure correct schema types and ordering
    all_candidates = (
        all_candidates
        .select([
            pl.col("source1_entity_id_int").cast(pl.UInt32),
            pl.col("candidate_entity_id_int").cast(pl.UInt32),
        ])
        .sort(["source1_entity_id_int", "candidate_entity_id_int"])
    )

    # 6. Save candidate pairs to Parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}_candidates.parquet"
    all_candidates.write_parquet(out_path, compression=compression)

    total_time = time.time() - split_start
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 70)
    logger.info("SAVED %s CANDIDATES TO: %s", split.upper(), out_path.resolve())
    logger.info("  Total Candidate Pairs : %d", all_candidates.height)
    logger.info("  File Size             : %.2f MB (Limit: 512 MB)", file_size_mb)
    logger.info("  Total Processing Time : %.2fs", total_time)
    logger.info("=" * 70)

    # 7. Ground truth recall evaluation (Train only)
    if split == "train" and eval_gt:
        gt_path = resolve_parquet_file(
            parquet_dir,
            "train/train_ground_truth.parquet",
            "train_ground_truth.parquet",
        )
        if not gt_path:
            raw_gt = REPO_ROOT / "data/train/train_ground_truth.tsv"
            if raw_gt.exists():
                gt_path = raw_gt

        if gt_path and gt_path.exists():
            evaluate_ground_truth_recall(
                candidates_df=all_candidates,
                gt_path=gt_path,
                id_map=id_map,
            )

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Multi-Channel Candidate Generation Engine for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=REPO_ROOT / "data/parquet",
        help="Root directory containing input Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/parquet",
        help="Output directory to save candidate Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Dataset split to generate candidates for: 'train', 'test', or 'both' (default: both)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Maximum candidates per S1 entity to stay under 512 MB limit (default: 20)",
    )
    parser.add_argument(
        "--tfidf-top-k",
        type=int,
        default=20,
        help="Top-K candidates to retrieve from TF-IDF channel (default: 20)",
    )
    parser.add_argument(
        "--min-tfidf-sim",
        type=float,
        default=0.20,
        help="Minimum cosine similarity for TF-IDF char n-gram matching (default: 0.20)",
    )
    parser.add_argument(
        "--max-exact-per-key",
        type=int,
        default=200,
        help="Frequency cap per exact key in pool to prune generic noise (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25000,
        help="Batch size for sparse matrix multiplication chunking (default: 25000)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of threads for parallel sparse computation (-1 for all cores, default: -1)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "lz4", "uncompressed"],
        help="Compression codec for output Parquet files (default: zstd)",
    )
    parser.add_argument(
        "--no-eval-gt",
        action="store_true",
        help="Disable recall ceiling evaluation against ground truth on train split.",
    )
    args = parser.parse_args()

    pipeline_start = time.time()
    logger.info("Amazon ML Challenge 2026 - Phase 2 Multi-Channel Blocking")
    logger.info("  Parquet Dir          : %s", args.parquet_dir.resolve())
    logger.info("  Output Dir           : %s", args.output_dir.resolve())
    logger.info("  Split Target         : %s", args.split)
    logger.info("  Max Cands / Entity   : %d", args.max_candidates)
    logger.info("  TF-IDF Top-K         : %d", args.tfidf_top_k)
    logger.info("  Min TF-IDF Sim       : %.2f", args.min_tfidf_sim)
    logger.info("  Max Pool Key Cands   : %d", args.max_exact_per_key)
    logger.info("  sparse_dot_topn      : %s", "Installed" if is_sparse_dot_topn_available() else "Not Installed (Using CSR chunked fallback)")

    # Load global id_map if present
    id_map_path = args.parquet_dir / "id_map.parquet"
    id_map: Optional[pl.DataFrame] = None
    if id_map_path.exists():
        logger.info("Found id_map at %s. Loading...", id_map_path.name)
        id_map = pl.read_parquet(id_map_path)

    splits_to_run = ["train", "test"] if args.split == "both" else [args.split]

    for s in splits_to_run:
        generate_candidates_for_split(
            parquet_dir=args.parquet_dir,
            output_dir=args.output_dir,
            split=s,
            max_candidates=args.max_candidates,
            tfidf_top_k=args.tfidf_top_k,
            min_tfidf_sim=args.min_tfidf_sim,
            max_exact_per_key=args.max_exact_per_key,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            compression=args.compression,
            id_map=id_map,
            eval_gt=not args.no_eval_gt,
        )

    logger.info("All requested splits completed in %.2fs.", time.time() - pipeline_start)


if __name__ == "__main__":
    main()
