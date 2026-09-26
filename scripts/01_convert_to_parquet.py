#!/usr/bin/env python3
"""
Convert Amazon ML Challenge 2026 TSV files to compressed Parquet format.

Features:
- Uses Polars streaming sink_parquet with zstd compression for memory efficiency.
- Converts all train and test TSV files:
    - data/train/train_source1.tsv
    - data/train/train_source2.tsv
    - data/train/train_source3.tsv
    - data/train/train_ground_truth.tsv
    - data/test/test_source1.tsv
    - data/test/test_source2.tsv
    - data/test/test_source3.tsv
- Generates data/parquet/id_map.parquet mapping every unique entity_id to a uint32 integer.
- Supports applying entity_id_int across converted files or saving mapping separately.
- Handles missing or alternative column names gracefully (e.g. source1_entity_id in GT).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import polars as pl


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUT_DIR = Path("data/parquet")

CONVERSION_TARGETS = [
    # (relative_tsv_path, relative_parquet_path, id_column_candidates)
    (
        Path("train/train_source1.tsv"),
        Path("train/train_source1.parquet"),
        ["entity_id"],
    ),
    (
        Path("train/train_source2.tsv"),
        Path("train/train_source2.parquet"),
        ["entity_id"],
    ),
    (
        Path("train/train_source3.tsv"),
        Path("train/train_source3.parquet"),
        ["entity_id"],
    ),
    (
        Path("train/train_ground_truth.tsv"),
        Path("train/train_ground_truth.parquet"),
        ["source1_entity_id", "entity_id"],
    ),
    (
        Path("test/test_source1.tsv"),
        Path("test/test_source1.parquet"),
        ["entity_id"],
    ),
    (
        Path("test/test_source2.tsv"),
        Path("test/test_source2.parquet"),
        ["entity_id"],
    ),
    (
        Path("test/test_source3.tsv"),
        Path("test/test_source3.parquet"),
        ["entity_id"],
    ),
]


def convert_tsv_to_parquet(
    tsv_path: Path,
    parquet_path: Path,
    compression: str = "zstd",
    id_map: Optional[pl.DataFrame] = None,
    id_col_candidates: Optional[List[str]] = None,
) -> int:
    """
    Streams a TSV file and sinks it to Parquet.
    If id_map is provided and a matching id column exists, joins entity_id_int.
    """
    if not tsv_path.exists():
        print(f"[!] Warning: Input file not found: {tsv_path}")
        return 0

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lf = pl.scan_csv(
        source=str(tsv_path),
        separator="\t",
        quote_char=None,  # TSVs can have raw quotes inside business names/addresses
        infer_schema_length=10000,
        null_values=["", "NULL", "null", "None"],
    )

    schema_names = lf.collect_schema().names()

    # If id_map is provided and we want to attach entity_id_int
    if id_map is not None:
        target_id_col = None
        for candidate in id_col_candidates or ["entity_id", "source1_entity_id"]:
            if candidate in schema_names:
                target_id_col = candidate
                break

        if target_id_col:
            id_lf = id_map.lazy().rename({"entity_id": target_id_col})
            lf = lf.join(id_lf, on=target_id_col, how="left")

    # Streaming write to Parquet
    lf.sink_parquet(
        path=str(parquet_path),
        compression=compression,
    )

    row_count = pl.scan_parquet(str(parquet_path)).select(pl.len()).collect().item()
    elapsed = time.time() - t0
    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(
        f"[*] Converted {tsv_path.name} -> {parquet_path.relative_to(parquet_path.parents[2]) if len(parquet_path.parents) > 2 else parquet_path} "
        f"({row_count:,} rows, {size_mb:.2f} MB, {elapsed:.1f}s)"
    )
    return row_count


def extract_unique_entity_ids(tsv_path: Path, id_col_candidates: List[str]) -> pl.Series:
    """
    Extracts unique entity IDs from a single TSV file.
    Also extracts IDs from comma-delimited matched_entity_ids if present.
    """
    if not tsv_path.exists():
        return pl.Series("entity_id", [], dtype=pl.String)

    lf = pl.scan_csv(
        source=str(tsv_path),
        separator="\t",
        quote_char=None,
        infer_schema_length=5000,
    )
    schema_names = lf.collect_schema().names()

    collected_series = []

    # Check primary ID column
    for candidate in id_col_candidates:
        if candidate in schema_names:
            ids = (
                lf.select(pl.col(candidate).drop_nulls().cast(pl.String))
                .unique()
                .collect()
                .get_column(candidate)
                .alias("entity_id")
            )
            collected_series.append(ids)
            break

    # If matched_entity_ids column exists (e.g. ground truth), split and collect those IDs
    if "matched_entity_ids" in schema_names:
        matched_ids = (
            lf.select(pl.col("matched_entity_ids").drop_nulls().cast(pl.String))
            .filter(pl.col("matched_entity_ids") != "")
            .with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .with_columns(pl.col("matched_entity_ids").str.strip_chars())
            .filter(pl.col("matched_entity_ids") != "")
            .rename({"matched_entity_ids": "entity_id"})
            .select("entity_id")
            .unique()
            .collect()
            .get_column("entity_id")
        )
        collected_series.append(matched_ids)

    if not collected_series:
        return pl.Series("entity_id", [], dtype=pl.String)

    return pl.concat(collected_series).unique()


def build_id_map(
    data_dir: Path,
    output_path: Path,
    compression: str = "zstd",
) -> pl.DataFrame:
    """
    Collects all unique entity IDs across train and test datasets,
    assigns an uint32 integer ID, and saves to id_map.parquet.
    """
    print("[*] Collecting unique entity IDs across all source files...")
    t0 = time.time()
    id_series_list: List[pl.Series] = []

    for rel_tsv, _, id_cols in CONVERSION_TARGETS:
        full_tsv = data_dir / rel_tsv
        if full_tsv.exists():
            print(f"    Scanning IDs from {full_tsv.name}...")
            series = extract_unique_entity_ids(full_tsv, id_cols)
            id_series_list.append(series)

    if not id_series_list:
        raise RuntimeError(f"No valid TSV files found in {data_dir}")

    all_ids = pl.concat(id_series_list).unique().sort()
    total_unique = len(all_ids)

    print(f"[*] Total unique entity IDs found: {total_unique:,}")

    id_map = pl.DataFrame({
        "entity_id": all_ids,
    }).with_columns(
        pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("entity_id_int")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    id_map.write_parquet(output_path, compression=compression)

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[*] Saved id_map to {output_path} ({size_mb:.2f} MB, {elapsed:.1f}s)")
    return id_map


def main():
    parser = argparse.ArgumentParser(
        description="Convert Amazon ML Challenge TSV files to compressed Parquet."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory containing train/ and test/ folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "lz4", "uncompressed"],
        help="Parquet compression algorithm (default: zstd)",
    )
    parser.add_argument(
        "--apply-id-map",
        action="store_true",
        help="If set, also joins entity_id_int directly into each converted Parquet file.",
    )
    parser.add_argument(
        "--id-map-only",
        action="store_true",
        help="Only build and save id_map.parquet without converting dataset files.",
    )
    args = parser.parse_args()

    total_start = time.time()
    id_map_path = args.output_dir / "id_map.parquet"

    # Step 1: Build global entity_id -> entity_id_int mapping
    id_map = build_id_map(
        data_dir=args.data_dir,
        output_path=id_map_path,
        compression=args.compression,
    )

    if args.id_map_only:
        print("[*] Completed ID map generation.")
        return

    # Step 2: Convert TSV files to Parquet
    print("\n[*] Converting TSV files to Parquet...")
    join_map = id_map if args.apply_id_map else None

    for rel_tsv, rel_parquet, id_cols in CONVERSION_TARGETS:
        tsv_path = args.data_dir / rel_tsv
        parquet_path = args.output_dir / rel_parquet
        convert_tsv_to_parquet(
            tsv_path=tsv_path,
            parquet_path=parquet_path,
            compression=args.compression,
            id_map=join_map,
            id_col_candidates=id_cols,
        )

    total_time = time.time() - total_start
    print(f"\n[+] All conversions successfully finished in {total_time:.1f}s.")
    print(f"[+] Parquet files stored in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
