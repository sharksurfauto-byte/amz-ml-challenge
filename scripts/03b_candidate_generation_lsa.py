#!/usr/bin/env python3
"""
Phase 2 Multi-Channel LSA Candidate Generation Pipeline for Amazon ML Challenge 2026.

Implements the 8-Channel High-Recall Candidate Blocking Engine using LSA:
  1. Exact matching on `name_token_sorted_key`
  2. Exact matching on `name_no_legal`
  3. Clean full address join (`address_clean`, min length 8 chars)
  4. Street number + first distinctive name token (`street_number` + `first_token`)
  5. S3 Domain root matching S1 clean name tokens / domain
  6. First two distinctive tokens match (`first_token` + `second_token`)
  7. LSA (TruncatedSVD + NearestNeighbors) char n-gram cosine similarity top-25
  8. Optional integration of dense semantic candidates from FAISS retrieval

Enforces candidate budgeting:
  Supports 30-50 candidates per S1 entity (default max 40: <=20 from S2, <=25 from S3)
  to dramatically raise the candidate recall ceiling towards >=95%.
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
    clean_address,
    clean_text,
    extract_and_remove_legal_forms,
    extract_domain_root,
    extract_first_two_tokens,
    extract_pincode,
    extract_street_number,
    kanan_transliterate_devanagari,
    phonetic_first_token,
    remove_junk_tokens,
    token_sort_key,
)

logger = logging.getLogger("candidate_gen_lsa")
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
        root_p = REPO_ROOT / rel
        if root_p.exists():
            return root_p
    return None


def ensure_id_int_and_keys(
    df: pl.DataFrame,
    id_map: Optional[pl.DataFrame] = None,
    id_col_name: str = "entity_id_int",
    is_source3: bool = False,
    source_type: int = 2,
) -> pl.DataFrame:
    """
    Ensures the DataFrame contains uint32 integer ID and all 8-channel normalized blocking keys.
    Computes missing keys on the fly.
    """
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

    needed_keys = [
        "name_clean", "name_no_legal", "name_token_sorted_key",
        "address_clean", "street_number", "pincode", "domain_root",
        "first_token", "second_token"
    ]
    missing_keys = [k for k in needed_keys if k not in df.columns]

    if missing_keys:
        logger.info("    Computing missing normalization keys on the fly...")
        bname = pl.col("business_name") if "business_name" in df.columns else pl.lit("")
        baddr = pl.col("business_address") if "business_address" in df.columns else pl.lit("")

        name_prep = kanan_transliterate_devanagari(remove_junk_tokens(bname))
        cleaned_name = clean_text(name_prep)
        no_legal, legal_form = extract_and_remove_legal_forms(cleaned_name)

        sorted_key = (
            pl.when(no_legal.is_not_null() & (no_legal != ""))
            .then(token_sort_key(no_legal))
            .otherwise(token_sort_key(cleaned_name))
        )

        c_addr, s_num, pcode = address_keys(baddr)
        t0, t1 = extract_first_two_tokens(cleaned_name)
        d_root = extract_domain_root(bname)

        exprs = []
        if "name_clean" in missing_keys:
            exprs.append(cleaned_name.alias("name_clean"))
        if "name_no_legal" in missing_keys:
            exprs.append(no_legal.alias("name_no_legal"))
        if "name_token_sorted_key" in missing_keys:
            exprs.append(sorted_key.alias("name_token_sorted_key"))
        if "address_clean" in missing_keys:
            exprs.append(c_addr.alias("address_clean"))
        if "street_number" in missing_keys:
            exprs.append(s_num.alias("street_number"))
        if "pincode" in missing_keys:
            exprs.append(pcode.alias("pincode"))
        if "domain_root" in missing_keys:
            exprs.append(d_root.alias("domain_root"))
        if "first_token" in missing_keys:
            exprs.append(t0.alias("first_token"))
        if "second_token" in missing_keys:
            exprs.append(t1.alias("second_token"))

        df = df.with_columns(exprs)

    if "source_type" not in df.columns:
        df = df.with_columns(pl.lit(source_type, dtype=pl.UInt8).alias("source_type"))

    return df


def load_dataset_split(
    parquet_dir: Path,
    split: str,
    id_map: Optional[pl.DataFrame] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Loads Source 1 and merges Source 2 and Source 3 into a single searchable pool."""
    t0 = time.time()
    logger.info("Loading %s datasets from %s...", split.upper(), parquet_dir.resolve())

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
    df_s1 = ensure_id_int_and_keys(
        df_s1, id_map=id_map, id_col_name="source1_entity_id_int", source_type=1
    )

    logger.info("  - Reading S2: %s", s2_path.name)
    df_s2 = pl.read_parquet(s2_path)
    df_s2 = ensure_id_int_and_keys(
        df_s2, id_map=id_map, id_col_name="candidate_entity_id_int", is_source3=False, source_type=2
    )

    logger.info("  - Reading S3: %s", s3_path.name)
    df_s3 = pl.read_parquet(s3_path)
    df_s3 = ensure_id_int_and_keys(
        df_s3, id_map=id_map, id_col_name="candidate_entity_id_int", is_source3=True, source_type=3
    )

    common_cols = [
        "country", "name_token_sorted_key", "name_no_legal", "name_clean",
        "address_clean", "street_number", "pincode", "domain_root",
        "first_token", "second_token"
    ]

    s1_cols = ["source1_entity_id_int"] + [c for c in common_cols if c in df_s1.columns]
    pool_cols = ["candidate_entity_id_int", "source_type"] + [c for c in common_cols if c in df_s2.columns and c in df_s3.columns]

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
) -> float:
    """Evaluates the Candidate Recall Ceiling against ground truth."""
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

    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else "entity_id"

    gt_pairs = (
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

        gt_pairs = (
            gt_pairs
            .join(id_map_s1, on="source1_entity_id", how="inner")
            .join(id_map_cand, on="candidate_entity_id", how="inner")
            .select(["source1_entity_id_int", "candidate_entity_id_int"])
        )

    total_gt = gt_pairs.height

    captured_df = gt_pairs.join(
        candidates_df,
        on=["source1_entity_id_int", "candidate_entity_id_int"],
        how="inner",
    )
    captured_count = captured_df.height
    recall_ceiling = (captured_count / total_gt * 100.0) if total_gt > 0 else 0.0

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

    return recall_ceiling


def generate_candidate_pairs_tsv(
    candidates_df: pl.DataFrame,
    id_map: pl.DataFrame,
    test_s1_path: Path,
    output_tsv_path: Path,
) -> None:
    """Generates portal-compliant output/candidate_pairs.tsv matching test S1 ordering."""
    logger.info("Generating portal-compliant candidate_pairs.tsv at %s...", output_tsv_path)
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    if test_s1_path.suffix == ".parquet":
        s1_test_df = pl.read_parquet(test_s1_path)
    else:
        s1_test_df = pl.read_csv(test_s1_path, separator="\t", columns=["entity_id"])

    s1_order = s1_test_df.select(pl.col("entity_id").alias("source1_entity_id"))
    id_map_lookup = id_map.select(["entity_id_int", "entity_id"])

    cands_mapped = (
        candidates_df
        .join(id_map_lookup, left_on="source1_entity_id_int", right_on="entity_id_int", how="left")
        .rename({"entity_id": "source1_entity_id"})
        .drop("source1_entity_id_int")
        .join(id_map_lookup, left_on="candidate_entity_id_int", right_on="entity_id_int", how="left")
        .rename({"entity_id": "candidate_entity_id"})
        .drop("candidate_entity_id_int")
    )

    grouped = (
        cands_mapped
        .sort(["source1_entity_id", "priority_rank"])
        .group_by("source1_entity_id", maintain_order=True)
        .agg(pl.col("candidate_entity_id").str.concat(","))
        .rename({"candidate_entity_id": "candidate_entity_ids"})
    )

    final_tsv_df = (
        s1_order
        .join(grouped, on="source1_entity_id", how="left")
        .with_columns(pl.col("candidate_entity_ids").fill_null(""))
    )

    final_tsv_df.write_csv(output_tsv_path, separator="\t")
    size_mb = output_tsv_path.stat().st_size / (1024 * 1024)
    logger.info("Successfully generated %s (%d rows, %.2f MB)", output_tsv_path.name, final_tsv_df.height, size_mb)


def generate_candidates_for_split(
    parquet_dir: Path,
    output_dir: Path,
    split: str,
    max_candidates: int = 40,
    max_s2_candidates: int = 20,
    max_s3_candidates: int = 25,
    tfidf_top_k: int = 25,
    min_tfidf_sim: float = 0.20,
    max_exact_per_key: int = 250,
    batch_size: int = 50000,
    n_jobs: int = -1,
    compression: str = "zstd",
    id_map: Optional[pl.DataFrame] = None,
    eval_gt: bool = True,
    generate_tsv: bool = True,
) -> Path:
    """Executes the 8-channel candidate blocking partitioned strictly by country."""
    split_start = time.time()
    logger.info("=" * 70)
    logger.info("STARTING LSA 8-CHANNEL CANDIDATE GENERATION FOR: %s", split.upper())
    logger.info(f"Budget: Max Total={max_candidates}, Max S2={max_s2_candidates}, Max S3={max_s3_candidates}")
    logger.info("=" * 70)

    s1_df, pool_df = load_dataset_split(parquet_dir=parquet_dir, split=split, id_map=id_map)

    sem_path = resolve_parquet_file(
        parquet_dir,
        f"{split}/{split}_semantic_candidates.parquet",
        f"{split}_semantic_candidates.parquet",
    )
    semantic_df = None
    if sem_path and sem_path.exists():
        logger.info("Found semantic candidate file at: %s", sem_path.resolve())
        semantic_df = pl.read_parquet(sem_path)

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

    blocker = MultiChannelBlocker(
        ngram_range=(3, 4),
        tfidf_top_k=tfidf_top_k,
        min_tfidf_sim=min_tfidf_sim,
        max_exact_per_key=max_exact_per_key,
        max_s2_candidates=max_s2_candidates,
        max_s3_candidates=max_s3_candidates,
        max_candidates_per_s1=max_candidates,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    country_pair_frames: List[pl.DataFrame] = []

    for idx, country in enumerate(tqdm(countries, desc=f"Blocking {split.upper()} by Country"), start=1):
        c_start = time.time()
        s1_c = s1_df.filter(pl.col("country") == country)
        pool_c = pool_df.filter(pl.col("country") == country)

        if pool_c.height == 0:
            continue

        pairs_c = blocker.block_country(
            s1_df=s1_c,
            pool_df=pool_c,
            semantic_cands_df=semantic_df,
            s1_id_col="source1_entity_id_int",
            pool_id_col="candidate_entity_id_int",
        )

        c_elapsed = time.time() - c_start
        s1_with_cands = pairs_c.select("source1_entity_id_int").unique().height
        coverage_pct = (s1_with_cands / s1_c.height * 100.0) if s1_c.height > 0 else 0.0

        logger.info(
            "  [%d/%d] Country '%s' in %.2fs: %d pairs | S1 covered: %d/%d (%.2f%%)",
            idx,
            len(countries),
            country,
            c_elapsed,
            pairs_c.height,
            s1_with_cands,
            s1_c.height,
            coverage_pct,
        )

        if pairs_c.height > 0:
            country_pair_frames.append(pairs_c)

    if country_pair_frames:
        all_candidates = pl.concat(country_pair_frames)
    else:
        all_candidates = pl.DataFrame(
            schema={
                "source1_entity_id_int": pl.UInt32,
                "candidate_entity_id_int": pl.UInt32,
                "priority_rank": pl.UInt8,
                "channel_mask": pl.UInt8,
            }
        )

    all_candidates = (
        all_candidates
        .select([
            pl.col("source1_entity_id_int").cast(pl.UInt32),
            pl.col("candidate_entity_id_int").cast(pl.UInt32),
            pl.col("priority_rank").cast(pl.UInt8),
            pl.col("channel_mask").cast(pl.UInt8),
        ])
        .sort(["source1_entity_id_int", "priority_rank"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}_candidates.parquet"
    all_candidates.write_parquet(out_path, compression=compression)

    total_time = time.time() - split_start
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 70)
    logger.info("SAVED %s CANDIDATES TO: %s", split.upper(), out_path.resolve())
    logger.info("  Total Candidate Pairs : %d", all_candidates.height)
    logger.info("  File Size             : %.2f MB", file_size_mb)
    logger.info("  Total Processing Time : %.2fs", total_time)
    logger.info("=" * 70)

    if split == "test" and generate_tsv and id_map is not None:
        test_s1_path = resolve_parquet_file(
            parquet_dir,
            "test/test_source1.parquet",
            "test_source1.parquet",
            "../data/test/test_source1.tsv",
        )
        if test_s1_path:
            tsv_path = REPO_ROOT / "output/candidate_pairs.tsv"
            generate_candidate_pairs_tsv(all_candidates, id_map, test_s1_path, tsv_path)

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
        description="Phase 2 8-Channel LSA Candidate Generation Engine for Amazon ML Challenge 2026."
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
        help="Dataset split: 'train', 'test', or 'both' (default: both)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=40,
        help="Maximum candidates per S1 entity (default: 40, user requested 30-50)",
    )
    parser.add_argument(
        "--max-s2",
        type=int,
        default=20,
        help="Maximum candidates from Source 2 per S1 entity (default: 20)",
    )
    parser.add_argument(
        "--max-s3",
        type=int,
        default=25,
        help="Maximum candidates from Source 3 per S1 entity (default: 25)",
    )
    parser.add_argument(
        "--tfidf-top-k",
        type=int,
        default=25,
        help="Top-K candidates to retrieve from LSA TF-IDF channel (default: 25)",
    )
    parser.add_argument(
        "--min-tfidf-sim",
        type=float,
        default=0.20,
        help="Minimum cosine similarity for LSA matching (default: 0.20)",
    )
    parser.add_argument(
        "--max-exact-per-key",
        type=int,
        default=250,
        help="Frequency cap per exact key in pool (default: 250)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for KNN chunking (default: 50000)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of threads for parallel computation (-1 for all cores, default: -1)",
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

    id_map_path = resolve_parquet_file(args.parquet_dir, "id_map.parquet")
    id_map = None
    if id_map_path and id_map_path.exists():
        logger.info("Found ID mapping table at: %s", id_map_path.resolve())
        id_map = pl.read_parquet(id_map_path)

    splits_to_run = ["train", "test"] if args.split == "both" else [args.split]

    for s in splits_to_run:
        generate_candidates_for_split(
            parquet_dir=args.parquet_dir,
            output_dir=args.output_dir,
            split=s,
            max_candidates=args.max_candidates,
            max_s2_candidates=args.max_s2,
            max_s3_candidates=args.max_s3,
            tfidf_top_k=args.tfidf_top_k,
            min_tfidf_sim=args.min_tfidf_sim,
            max_exact_per_key=args.max_exact_per_key,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            compression=args.compression,
            id_map=id_map,
            eval_gt=not args.no_eval_gt,
        )


if __name__ == "__main__":
    main()
