#!/usr/bin/env python3
"""
Build normalized text keys and blocking features across Amazon ML Challenge Parquet datasets.

Generates the following normalized columns:
  - name_clean: lowercased, diacritics stripped, non-alphanumeric punctuation removed.
  - name_no_legal: name with corporate legal forms (inc, llc, ltd, pvt, corp, etc.) stripped.
  - legal_form: extracted canonical legal form label (or null).
  - name_token_sorted_key: tokens sorted alphabetically for word-order invariant matching.
  - address_clean: normalized address string.
  - address_street_number: first occurring digits from address (or null).
  - domain_root: extracted domain root for Source 3 records (e.g. 'wilfordhancock' from 'wilfordhancock.com'), else null.

Supports reading from existing Parquet files or falling back to raw TSVs if Parquets have
not been generated yet.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl

# Add repository root to sys.path so src imports work cleanly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.normalize import (
    clean_text,
    remove_junk_tokens,
    kanan_transliterate_devanagari,
    extract_and_remove_legal_forms,
    extract_domain_root,
    token_sort_key,
    address_keys,
)


TARGET_DATASETS = [
    # (relative_path, is_source3, fallback_tsv)
    (Path("train/train_source1.parquet"), False, Path("data/train/train_source1.tsv")),
    (Path("train/train_source2.parquet"), False, Path("data/train/train_source2.tsv")),
    (Path("train/train_source3.parquet"), True, Path("data/train/train_source3.tsv")),
    (Path("test/test_source1.parquet"), False, Path("data/test/test_source1.tsv")),
    (Path("test/test_source2.parquet"), False, Path("data/test/test_source2.tsv")),
    (Path("test/test_source3.parquet"), True, Path("data/test/test_source3.tsv")),
]


def build_key_expressions(is_source3: bool = False) -> List[pl.Expr]:
    """
    Returns the list of Polars expressions to compute the normalized key columns.
    """
    bname = pl.col("business_name")
    baddr = pl.col("business_address")

    # Name normalization pipeline
    name_prep = kanan_transliterate_devanagari(remove_junk_tokens(bname))
    cleaned_name = clean_text(name_prep)
    no_legal, legal_form = extract_and_remove_legal_forms(cleaned_name)

    # Word-order invariant token-sorted key
    sorted_key = (
        pl.when(no_legal.is_not_null() & (no_legal != ""))
        .then(token_sort_key(no_legal))
        .otherwise(token_sort_key(cleaned_name))
    )

    from src.normalize import extract_first_two_tokens
    t0, t1 = extract_first_two_tokens(cleaned_name)

    # Address normalization pipeline
    clean_addr, street_num, pincode = address_keys(baddr)

    # Domain root extraction (Source 3 web records only)
    domain_expr = (
        extract_domain_root(bname)
        if is_source3
        else pl.lit(None, dtype=pl.String)
    )

    return [
        cleaned_name.alias("name_clean"),
        no_legal.alias("name_no_legal"),
        legal_form.alias("legal_form"),
        sorted_key.alias("name_token_sorted_key"),
        t0.alias("first_token"),
        t1.alias("second_token"),
        clean_addr.alias("address_clean"),
        street_num.alias("address_street_number"),
        pincode.alias("pincode"),
        domain_expr.alias("domain_root"),
    ]


def process_dataset(
    input_path: Path,
    output_path: Path,
    is_source3: bool,
    fallback_tsv: Optional[Path] = None,
    compression: str = "zstd",
    sample_size: Optional[int] = None,
) -> int:
    """
    Loads a dataset (Parquet or fallback TSV), applies key building expressions,
    and saves the mutated Parquet.
    """
    t0 = time.time()
    source_desc = ""

    if input_path.exists():
        lf = pl.scan_parquet(str(input_path))
        source_desc = f"Parquet: {input_path.name}"
    elif fallback_tsv and fallback_tsv.exists():
        print(f"[*] Input Parquet {input_path.name} not found. Falling back to TSV: {fallback_tsv.name}")
        lf = pl.scan_csv(
            source=str(fallback_tsv),
            separator="\t",
            quote_char=None,
            infer_schema_length=10000,
            null_values=["", "NULL", "null", "None"],
        )
        source_desc = f"TSV fallback: {fallback_tsv.name}"
    else:
        print(f"[!] Warning: Neither {input_path} nor fallback {fallback_tsv} exists. Skipping.")
        return 0

    if sample_size:
        lf = lf.limit(sample_size)

    # Check that required columns exist
    schema_names = lf.collect_schema().names()
    if "business_name" not in schema_names or "business_address" not in schema_names:
        print(f"[!] Skipping {input_path.name}: missing 'business_name' or 'business_address' column.")
        return 0

    key_exprs = build_key_expressions(is_source3=is_source3)
    mutated_lf = lf.with_columns(key_exprs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(".tmp.parquet")

    # Write to temporary file first for atomic replacement
    mutated_lf.sink_parquet(
        path=str(temp_output_path),
        compression=compression,
    )

    if temp_output_path.exists():
        if output_path.exists():
            output_path.unlink()
        temp_output_path.rename(output_path)

    row_count = pl.scan_parquet(str(output_path)).select(pl.len()).collect().item()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0

    print(
        f"[+] Built keys for {source_desc} -> {output_path.name} "
        f"({row_count:,} rows, {size_mb:.2f} MB, {elapsed:.1f}s)"
    )
    return row_count


def main():
    parser = argparse.ArgumentParser(
        description="Build normalized keys and blocking features across Parquet datasets."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("data/parquet"),
        help="Root directory containing input Parquet files (default: data/parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output Parquet files (default: same as --parquet-dir)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix for output files (e.g. '_keys' to create 'train_source1_keys.parquet'). Default: '' replaces in-place.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "lz4", "uncompressed"],
        help="Parquet compression algorithm (default: zstd)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size limit (for fast testing / benchmarking).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not fallback to reading TSVs if Parquet files are missing.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.parquet_dir
    total_start = time.time()
    total_rows = 0

    print("=" * 70)
    print("Amazon ML Challenge 2026 - Key Building & Normalization")
    print(f"Input Parquet Dir : {args.parquet_dir.resolve()}")
    print(f"Output Dir        : {output_dir.resolve()}")
    print(f"Filename Suffix   : '{args.suffix}'")
    print(f"Compression       : {args.compression}")
    if args.sample_size:
        print(f"Sample Limit      : {args.sample_size:,} rows")
    print("=" * 70)

    for rel_path, is_s3, fallback_tsv in TARGET_DATASETS:
        in_path = args.parquet_dir / rel_path

        # Determine output filename
        if args.suffix:
            out_stem = f"{rel_path.stem}{args.suffix}"
            out_path = output_dir / rel_path.parent / f"{out_stem}.parquet"
        else:
            out_path = output_dir / rel_path

        tsv_path = None if args.no_fallback else REPO_ROOT / fallback_tsv

        rows = process_dataset(
            input_path=in_path,
            output_path=out_path,
            is_source3=is_s3,
            fallback_tsv=tsv_path,
            compression=args.compression,
            sample_size=args.sample_size,
        )
        total_rows += rows

    elapsed_total = time.time() - total_start
    print("=" * 70)
    print(f"[+] All key building completed: {total_rows:,} total rows processed in {elapsed_total:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
