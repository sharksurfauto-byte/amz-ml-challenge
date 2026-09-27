#!/usr/bin/env python3
"""
Merge candidate sets from LSA blocking and semantic retrieval, then re-extract features.

This script:
1. Loads LSA candidates (test_candidates.parquet) and semantic candidates (test_semantic_candidates.parquet)
2. Unions and deduplicates them by (source1_entity_id_int, candidate_entity_id_int)
3. Loads normalized source entity tables (test_source1/2/3.parquet)
4. Re-extracts the 10 RapidFuzz features using src.features.build_pairwise_features
5. Saves merged features to test_features_merged.parquet
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import build_pairwise_features
from src.normalize import (
    address_keys,
    clean_text,
    extract_and_remove_legal_forms,
    kanan_transliterate_devanagari,
    remove_junk_tokens,
    token_sort_key,
)

logger = logging.getLogger("merge_and_refeature")
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
    needed = ["name_clean", "address_clean", "legal_form", "domain_root", "address_street_number"]
    missing = [c for c in needed if c not in df.columns]

    if not missing:
        return df

    logger.info("    Computing missing normalization columns %s on the fly...", missing)
    bname = pl.col("business_name") if "business_name" in df.columns else pl.lit("")
    baddr = pl.col("business_address") if "business_address" in df.columns else pl.lit("")

    name_prep = kanan_transliterate_devanagari(remove_junk_tokens(bname))
    cleaned_name = clean_text(name_prep)
    no_legal, legal_form = extract_and_remove_legal_forms(cleaned_name)
    clean_addr, street_num = address_keys(baddr)
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
    if "domain_root" in missing:
        exprs.append(domain_expr.alias("domain_root"))

    return df.with_columns(exprs)


def load_entity_tables(
    parquet_dir: Path,
    split: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Loads S1 reference table and merges S2 and S3 into a single candidate pool table.
    Retains only essential metadata columns to minimize memory consumption.
    """
    s1_path = resolve_file(
        parquet_dir,
        f"{split}/{split}_source1.parquet",
        f"{split}_source1.parquet",
    )
    s2_path = resolve_file(
        parquet_dir,
        f"{split}/{split}_source2.parquet",
        f"{split}_source2.parquet",
    )
    s3_path = resolve_file(
        parquet_dir,
        f"{split}/{split}_source3.parquet",
        f"{split}_source3.parquet",
    )

    if not s1_path or not s2_path or not s3_path:
        raise FileNotFoundError(
            f"Could not locate all entity files for {split} split in {parquet_dir}.\n"
            f"  S1: {s1_path}\n  S2: {s2_path}\n  S3: {s3_path}"
        )

    logger.info("  Loading S1 entity table: %s", s1_path.name)
    df_s1 = pl.read_parquet(s1_path)
    df_s1 = ensure_metadata_columns(df_s1, is_source3=False)

    logger.info("  Loading S2 entity table: %s", s2_path.name)
    df_s2 = pl.read_parquet(s2_path)
    df_s2 = ensure_metadata_columns(df_s2, is_source3=False)

    logger.info("  Loading S3 entity table: %s", s3_path.name)
    df_s3 = pl.read_parquet(s3_path)
    df_s3 = ensure_metadata_columns(df_s3, is_source3=True)

    # Standardize column selections
    metadata_cols = ["name_clean", "address_clean", "legal_form", "domain_root", "address_street_number"]

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


def load_and_merge_candidates(
    primary_path: Path,
    secondary_path: Path,
) -> pl.DataFrame:
    """
    Load primary and secondary candidate sets, union them, and deduplicate.
    """
    logger.info("Loading primary candidates from: %s", primary_path.name)
    primary_df = pl.read_parquet(primary_path)
    logger.info("Primary candidates: %d rows", primary_df.height)

    logger.info("Loading secondary candidates from: %s", secondary_path.name)
    secondary_df = pl.read_parquet(secondary_path)
    logger.info("Secondary candidates: %d rows", secondary_df.height)

    # Union both datasets
    merged_df = pl.concat([primary_df, secondary_df])
    logger.info("After union: %d rows", merged_df.height)

    # Deduplicate by (source1_entity_id_int, candidate_entity_id_int)
    deduplicated_df = merged_df.unique(subset=["source1_entity_id_int", "candidate_entity_id_int"])
    logger.info("After deduplication: %d rows", deduplicated_df.height)

    return deduplicated_df


def process_merged_features(
    primary_candidates_path: Path,
    secondary_candidates_path: Path,
    parquet_dir: Path,
    output_path: Path,
) -> Path:
    """
    Process merged candidate set and extract features.
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("STARTING MERGED FEATURE EXTRACTION")
    logger.info("=" * 70)

    # 1. Load and merge candidate sets
    pairs_df = load_and_merge_candidates(primary_candidates_path, secondary_candidates_path)
    logger.info("Merged candidate pairs to process: %d rows", pairs_df.height)

    # 2. Load entity tables (S1 reference and Candidate Pool)
    s1_table, cand_table = load_entity_tables(parquet_dir=parquet_dir, split="test")

    # 3. Extract tabular features using src.features
    logger.info("Computing pairwise features across %d rows...", pairs_df.height)
    features_df = build_pairwise_features(
        s1_df=s1_table,
        cand_df=cand_table,
        pairs_df=pairs_df,
        n_jobs=-1,
    )

    # 4. Save output Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.parquet")

    logger.info("Writing output to temporary file: %s", temp_path.name)
    features_df.write_parquet(temp_path, compression="zstd")

    if temp_path.exists():
        if output_path.exists():
            output_path.unlink()
        temp_path.rename(output_path)

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 70)
    logger.info("SUCCESSFULLY SAVED: %s", output_path.resolve())
    logger.info("  Total Rows      : %d", features_df.height)
    logger.info("  Total Columns   : %d (%s)", len(features_df.columns), ", ".join(features_df.columns))
    logger.info("  File Size       : %.2f MB", size_mb)
    logger.info("  Total Time      : %.2fs", elapsed)
    logger.info("=" * 70)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge LSA and semantic candidates, then re-extract features for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default=REPO_ROOT / "data/parquet/test_candidates.parquet",
        help="Path to primary candidates (LSA blocking) (default: data/parquet/test_candidates.parquet)",
    )
    parser.add_argument(
        "--secondary",
        type=Path,
        default=REPO_ROOT / "data/parquet/test_semantic_candidates.parquet",
        help="Path to secondary candidates (semantic retrieval) (default: data/parquet/test_semantic_candidates.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/parquet/test_features_merged.parquet",
        help="Path to output features (default: data/parquet/test_features_merged.parquet)",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=REPO_ROOT / "data/parquet",
        help="Directory containing source parquet files (default: data/parquet)",
    )
    args = parser.parse_args()

    total_start = time.time()
    logger.info("Amazon ML Challenge 2026 - Merge and Re-feature Extraction")
    logger.info("  Primary Candidates : %s", args.primary.resolve())
    logger.info("  Secondary Candidates: %s", args.secondary.resolve())
    logger.info("  Parquet Dir        : %s", args.parquet_dir.resolve())
    logger.info("  Output             : %s", args.output.resolve())

    process_merged_features(
        primary_candidates_path=args.primary,
        secondary_candidates_path=args.secondary,
        parquet_dir=args.parquet_dir,
        output_path=args.output,
    )

    logger.info("All tasks completed in %.2fs.", time.time() - total_start)


if __name__ == "__main__":
    main()