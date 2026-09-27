#!/usr/bin/env python3
"""
Phase 3 Feature Engineering Pipeline for Amazon ML Challenge 2026.

Builds tabular candidate features for GBDT Re-Ranking:
  1. Loads candidate pairs:
     - data/parquet/train_candidates.parquet
     - data/parquet/test_candidates.parquet
  2. For train split:
     - Resolves ground truth from train_ground_truth.parquet (or .tsv).
     - Joins true positive matches to assign target = 1 for GT pairs and target = 0
       for candidate blocking hard negatives.
  3. Extracts 10 high-signal lexical, physical address, legal, domain, and structural features
     via `src.features.build_pairwise_features`.
  4. Saves output to:
     - data/parquet/train_features.parquet
     - data/parquet/test_features.parquet
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import FEATURE_COLUMNS, build_pairwise_features
from src.normalize import (
    address_keys,
    clean_text,
    extract_and_remove_legal_forms,
    extract_domain_root,
    kanan_transliterate_devanagari,
    remove_junk_tokens,
    token_sort_key,
)

logger = logging.getLogger("build_features")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def resolve_file(base_dir: Path, *candidates: str) -> Optional[Path]:
    """Resolves an existing file path from relative candidate paths."""
    for rel in candidates:
        p = base_dir / rel
        if p.exists():
            return p
        root_p = REPO_ROOT / rel
        if root_p.exists():
            return root_p
    return None


def ensure_metadata_columns(df: pl.DataFrame, is_source3: bool = False) -> pl.DataFrame:
    """
    Ensures required feature extraction columns are present in entity table:
    name_clean, address_clean, legal_form, domain_root, address_street_number.
    Computes them on the fly if missing.
    """
    needed = ["name_clean", "address_clean", "legal_form", "domain_root", "address_street_number", "pincode"]
    missing = [c for c in needed if c not in df.columns]

    if not missing:
        return df

    logger.info("    Computing missing normalization columns %s on the fly...", missing)
    bname = pl.col("business_name") if "business_name" in df.columns else pl.lit("")
    baddr = pl.col("business_address") if "business_address" in df.columns else pl.lit("")

    name_prep = kanan_transliterate_devanagari(remove_junk_tokens(bname))
    cleaned_name = clean_text(name_prep)
    no_legal, legal_form = extract_and_remove_legal_forms(cleaned_name)
    clean_addr, street_num, pcode = address_keys(baddr)
    domain_expr = extract_domain_root(bname) if is_source3 else pl.lit(None, dtype=pl.String)

    exprs = []
    if "name_clean" in missing:
        exprs.append(cleaned_name.alias("name_clean"))
    if "legal_form" in missing:
        exprs.append(legal_form.alias("legal_form"))
    if "address_clean" in missing:
        exprs.append(clean_addr.alias("address_clean"))
    if "address_street_number" in missing:
        exprs.append(street_num.alias("address_street_number"))
    if "pincode" in missing:
        exprs.append(pcode.alias("pincode"))
    if "domain_root" in missing:
        exprs.append(domain_expr.alias("domain_root"))

    return df.with_columns(exprs)


def load_entity_tables(
    parquet_dir: Path,
    split: str,
    id_map: Optional[pl.DataFrame] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Loads S1 reference table and merges S2 and S3 into a single candidate pool table.
    Retains only essential metadata columns to minimize memory consumption.
    """
    s1_path = resolve_file(
        parquet_dir,
        f"{split}/{split}_source1.parquet",
        f"{split}_source1.parquet",
        f"data/{split}/{split}_source1.tsv",
    )
    s2_path = resolve_file(
        parquet_dir,
        f"{split}/{split}_source2.parquet",
        f"{split}_source2.parquet",
        f"data/{split}/{split}_source2.tsv",
    )
    s3_path = resolve_file(
        parquet_dir,
        f"{split}/{split}_source3.parquet",
        f"{split}_source3.parquet",
        f"data/{split}/{split}_source3.tsv",
    )

    if not s1_path or not s2_path or not s3_path:
        raise FileNotFoundError(
            f"Could not locate all entity files for {split} split in {parquet_dir}.\n"
            f"  S1: {s1_path}\n  S2: {s2_path}\n  S3: {s3_path}"
        )

    logger.info("  Loading S1 entity table: %s", s1_path.name)
    df_s1 = pl.read_parquet(s1_path) if s1_path.suffix == ".parquet" else pl.read_csv(s1_path, separator="\t")
    if "source1_entity_id_int" not in df_s1.columns and "entity_id_int" not in df_s1.columns and id_map is not None:
        df_s1 = df_s1.join(id_map.rename({"entity_id_int": "source1_entity_id_int"}), on="entity_id", how="left")
    df_s1 = ensure_metadata_columns(df_s1, is_source3=False)

    logger.info("  Loading S2 entity table: %s", s2_path.name)
    df_s2 = pl.read_parquet(s2_path) if s2_path.suffix == ".parquet" else pl.read_csv(s2_path, separator="\t")
    if "candidate_entity_id_int" not in df_s2.columns and "entity_id_int" not in df_s2.columns and id_map is not None:
        df_s2 = df_s2.join(id_map.rename({"entity_id_int": "candidate_entity_id_int"}), on="entity_id", how="left")
    df_s2 = ensure_metadata_columns(df_s2, is_source3=False)

    logger.info("  Loading S3 entity table: %s", s3_path.name)
    df_s3 = pl.read_parquet(s3_path) if s3_path.suffix == ".parquet" else pl.read_csv(s3_path, separator="\t")
    if "candidate_entity_id_int" not in df_s3.columns and "entity_id_int" not in df_s3.columns and id_map is not None:
        df_s3 = df_s3.join(id_map.rename({"entity_id_int": "candidate_entity_id_int"}), on="entity_id", how="left")
    df_s3 = ensure_metadata_columns(df_s3, is_source3=True)

    # Standardize column selections
    metadata_cols = ["name_clean", "address_clean", "legal_form", "domain_root", "address_street_number", "pincode"]

    s1_id_col = "source1_entity_id_int" if "source1_entity_id_int" in df_s1.columns else "entity_id_int"
    s1_clean = df_s1.select([pl.col(s1_id_col).cast(pl.UInt32).alias("source1_entity_id_int")] + [
        pl.col(c) for c in metadata_cols
    ])

    s2_id_col = "candidate_entity_id_int" if "candidate_entity_id_int" in df_s2.columns else "entity_id_int"
    s3_id_col = "candidate_entity_id_int" if "candidate_entity_id_int" in df_s3.columns else "entity_id_int"

    s2_sub = df_s2.select([pl.col(s2_id_col).cast(pl.UInt32).alias("candidate_entity_id_int")] + [
        pl.col(c) for c in metadata_cols
    ])
    s3_sub = df_s3.select([pl.col(s3_id_col).cast(pl.UInt32).alias("candidate_entity_id_int")] + [
        pl.col(c) for c in metadata_cols
    ])

    cand_pool = pl.concat([s2_sub, s3_sub])
    logger.info("  S1: %d rows | Candidate Pool: %d rows loaded.", s1_clean.height, cand_pool.height)
    return s1_clean, cand_pool


def load_ground_truth_pairs(
    gt_path: Path,
    id_map: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Parses train ground truth table into exploded pairs of (source1_entity_id_int, candidate_entity_id_int).
    """
    logger.info("Loading ground truth from: %s", gt_path.name)
    if gt_path.suffix == ".parquet":
        gt_df = pl.read_parquet(gt_path)
    else:
        gt_df = pl.read_csv(gt_path, separator="\t", infer_schema_length=5000)

    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else (
        "entity_id" if "entity_id" in gt_df.columns else "source1_entity_id_int"
    )

    if "matched_entity_ids" in gt_df.columns:
        gt_exploded = (
            gt_df.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
            .select([s1_col, "matched_entity_ids"])
            .with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .with_columns(pl.col("matched_entity_ids").str.strip_chars())
            .filter(pl.col("matched_entity_ids") != "")
            .rename({s1_col: "source1_entity_id", "matched_entity_ids": "candidate_entity_id"})
        )

        if id_map is not None:
            id_map_s1 = id_map.rename({"entity_id": "source1_entity_id", "entity_id_int": "source1_entity_id_int"})
            id_map_cand = id_map.rename({"entity_id": "candidate_entity_id", "entity_id_int": "candidate_entity_id_int"})
            gt_exploded = (
                gt_exploded
                .join(id_map_s1, on="source1_entity_id", how="inner")
                .join(id_map_cand, on="candidate_entity_id", how="inner")
                .select(["source1_entity_id_int", "candidate_entity_id_int"])
            )
        return gt_exploded.unique()

    # Already tabular pairs
    return gt_df.select(["source1_entity_id_int", "candidate_entity_id_int"]).unique()


def process_features_for_split(
    parquet_dir: Path,
    output_dir: Path,
    split: str,
    id_map: Optional[pl.DataFrame] = None,
    sample_size: Optional[int] = None,
    n_jobs: int = -1,
    compression: str = "zstd",
) -> Path:
    """
    Builds feature matrix for a specific split ('train' or 'test').
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("STARTING FEATURE EXTRACTION FOR SPLIT: %s", split.upper())
    logger.info("=" * 70)

    # 1. Resolve candidates file
    cand_path = resolve_file(
        parquet_dir,
        f"{split}_candidates.parquet",
        f"{split}/{split}_candidates.parquet",
    )
    if not cand_path:
        raise FileNotFoundError(
            f"Candidate file not found for split '{split}' in {parquet_dir}. "
            f"Please run scripts/03_candidate_generation.py first."
        )

    logger.info("Loading %s candidate pairs from: %s", split.upper(), cand_path.name)
    pairs_df = pl.read_parquet(cand_path)

    if sample_size and sample_size < pairs_df.height:
        logger.info("Sampling %d pairs for fast benchmarking...", sample_size)
        pairs_df = pairs_df.head(sample_size)

    logger.info("Candidate pairs to process: %d rows", pairs_df.height)

    # 2. For train split: assign binary target labels (1 = Ground Truth Match, 0 = Hard Negative)
    if split == "train":
        gt_path = resolve_file(
            parquet_dir,
            "train/train_ground_truth.parquet",
            "train_ground_truth.parquet",
            "data/train/train_ground_truth.tsv",
        )
        if not gt_path:
            raise FileNotFoundError(f"Train ground truth not found in {parquet_dir} or data/train/")

        gt_pairs = load_ground_truth_pairs(gt_path, id_map=id_map)
        logger.info("Total unique Ground Truth match pairs: %d", gt_pairs.height)

        # Left-join GT pairs onto candidates to create target label
        pairs_df = (
            pairs_df
            .join(
                gt_pairs.with_columns(pl.lit(1).cast(pl.Int8).alias("target")),
                on=["source1_entity_id_int", "candidate_entity_id_int"],
                how="left",
            )
            .with_columns(pl.col("target").fill_null(0).cast(pl.Int8))
        )

        n_pos = pairs_df.filter(pl.col("target") == 1).height
        n_neg = pairs_df.filter(pl.col("target") == 0).height
        pos_ratio = (n_pos / pairs_df.height * 100.0) if pairs_df.height > 0 else 0.0
        logger.info(
            "Target distribution: Positives (target=1): %d (%.2f%%) | Hard Negatives (target=0): %d (%.2f%%)",
            n_pos,
            pos_ratio,
            n_neg,
            100.0 - pos_ratio,
        )

    # 3. Load entity tables (S1 reference and Candidate Pool)
    s1_table, cand_table = load_entity_tables(parquet_dir=parquet_dir, split=split, id_map=id_map)

    # 4. Extract tabular features using src.features
    logger.info("Computing pairwise features across %d rows...", pairs_df.height)
    features_df = build_pairwise_features(
        s1_df=s1_table,
        cand_df=cand_table,
        pairs_df=pairs_df,
        n_jobs=n_jobs,
    )

    # 5. Save output Parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}_features.parquet"
    temp_path = out_path.with_suffix(".tmp.parquet")

    logger.info("Writing output to temporary file: %s", temp_path.name)
    features_df.write_parquet(temp_path, compression=compression)

    if temp_path.exists():
        if out_path.exists():
            out_path.unlink()
        temp_path.rename(out_path)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 70)
    logger.info("SUCCESSFULLY SAVED: %s", out_path.resolve())
    logger.info("  Total Rows      : %d", features_df.height)
    logger.info("  Total Columns   : %d (%s)", len(features_df.columns), ", ".join(features_df.columns))
    logger.info("  File Size       : %.2f MB", size_mb)
    logger.info("  Total Time      : %.2fs", elapsed)
    logger.info("=" * 70)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 Pairwise Feature Extraction Engine for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=REPO_ROOT / "data/parquet",
        help="Directory containing candidate and entity Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/parquet",
        help="Directory to save output feature Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Dataset split to build features for: 'train', 'test', or 'both' (default: both)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row limit on candidate pairs for fast benchmarking / debugging",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel worker threads (-1 for all cores, default: -1)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "lz4", "uncompressed"],
        help="Parquet compression codec (default: zstd)",
    )
    args = parser.parse_args()

    total_start = time.time()
    logger.info("Amazon ML Challenge 2026 - Phase 3 Feature Extraction Engine")
    logger.info("  Parquet Dir : %s", args.parquet_dir.resolve())
    logger.info("  Output Dir  : %s", args.output_dir.resolve())
    logger.info("  Split       : %s", args.split)

    # Load global id_map if present
    id_map_path = resolve_file(args.parquet_dir, "id_map.parquet")
    id_map = None
    if id_map_path and id_map_path.exists():
        logger.info("Found id_map at %s. Loading...", id_map_path.name)
        id_map = pl.read_parquet(id_map_path)

    splits_to_run = ["train", "test"] if args.split == "both" else [args.split]

    for s in splits_to_run:
        process_features_for_split(
            parquet_dir=args.parquet_dir,
            output_dir=args.output_dir,
            split=s,
            id_map=id_map,
            sample_size=args.sample_size,
            n_jobs=args.n_jobs,
            compression=args.compression,
        )

    logger.info("All requested feature extraction tasks completed in %.2fs.", time.time() - total_start)


if __name__ == "__main__":
    main()
