#!/usr/bin/env python3
"""
Official Submission Validation Script for Amazon ML Challenge 2026.

Verifies submission compliance against official competition rules:
1. Both files exist and file size strictly < 512 MB.
2. Headers are exact:
   - matching_results.tsv: "source1_entity_id\\tmatched_entity_ids"
   - candidate_pairs.tsv:  "source1_entity_id\\tcandidate_entity_ids"
3. Exactly 1,732,544 rows in each file, matching test S1 entity IDs in exact order.
4. No S1 entity IDs in predictions/candidates (only S2- or S3- prefixes).
5. Comma-separated lists with no spaces, quotes, brackets, or trailing delimiters.
6. No duplicate IDs within any row's list.
7. Strict Subset Rule: Every predicted ID in matching_results.tsv MUST exist
   in candidate_pairs.tsv for that same S1 entity (P subset of C).
8. Exit 0 on PASS, Exit 1 with numbered failure list on violation.
"""

import os
import sys
import argparse
from typing import List, Set, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate submission files for Amazon ML Challenge 2026"
    )
    parser.add_argument(
        "--matching",
        type=str,
        default="output/matching_results.tsv",
        help="Path to matching_results.tsv (default: output/matching_results.tsv)",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="output/candidate_pairs.tsv",
        help="Path to candidate_pairs.tsv (default: output/candidate_pairs.tsv)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/test",
        help="Path to directory containing test data (default: data/test)",
    )
    parser.add_argument(
        "--test-s1",
        type=str,
        default=None,
        help="Path to test_source1.tsv (defaults to <test-dir>/test_source1.tsv)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=512.0,
        help="Maximum allowed file size in MB (default: 512.0)",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=1732544,
        help="Expected number of test S1 rows (default: 1,732,544)",
    )
    return parser.parse_args()


def validate_file_existence_and_size(
    file_path: str, label: str, max_size_mb: float
) -> Tuple[bool, str, float]:
    """Check if file exists and is strictly under the maximum allowed size."""
    if not os.path.exists(file_path):
        return False, f"{label} not found at '{file_path}'.", 0.0

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024.0 * 1024.0)

    if size_mb >= max_size_mb:
        return (
            False,
            f"{label} size is {size_mb:.2f} MB ({size_bytes} bytes), which exceeds or equals the {max_size_mb} MB limit.",
            size_mb,
        )

    return True, "", size_mb


def parse_id_list(
    raw_str: str, row_num: int, label: str, errors: List[str]
) -> Tuple[List[str], Set[str]]:
    """
    Parse and validate comma-separated entity IDs.
    Returns (list_of_ids, set_of_ids).
    """
    if not raw_str:
        return [], set()

    # Check for forbidden characters: spaces, quotes, brackets
    forbidden_chars = {" ", "\t", '"', "'", "[", "]", "{", "}", "(", ")"}
    found_forbidden = [c for c in raw_str if c in forbidden_chars]
    if found_forbidden:
        if len(errors) < 20:
            errors.append(
                f"Row {row_num} in {label} contains forbidden characters: {set(found_forbidden)!r} in '{raw_str[:50]}'"
            )

    tokens = raw_str.split(",")
    valid_ids = []
    seen = set()
    has_dups = False

    for t in tokens:
        if not t:
            if len(errors) < 20:
                errors.append(
                    f"Row {row_num} in {label} contains empty token in comma-separated list: '{raw_str[:50]}'"
                )
            continue

        if t.startswith("S1-"):
            if len(errors) < 20:
                errors.append(
                    f"Row {row_num} in {label} contains S1 entity ID '{t}', only S2- or S3- prefixes allowed."
                )

        if not (t.startswith("S2-") or t.startswith("S3-")):
            if len(errors) < 20:
                errors.append(
                    f"Row {row_num} in {label} contains invalid prefix in ID '{t}', must start with S2- or S3-."
                )

        if t in seen:
            has_dups = True
        else:
            seen.add(t)

        valid_ids.append(t)

    if has_dups:
        if len(errors) < 20:
            errors.append(
                f"Row {row_num} in {label} contains duplicate IDs in list: '{raw_str[:60]}'"
            )

    return valid_ids, seen


def main():
    args = parse_args()

    test_s1_path = args.test_s1
    if test_s1_path is None:
        test_s1_path = os.path.join(args.test_dir, "test_source1.tsv")

    failures = []

    print("=" * 72)
    print("AMAZON ML CHALLENGE 2026 - SUBMISSION VALIDATION")
    print("=" * 72)
    print(f"Matching file:  {args.matching}")
    print(f"Candidate file: {args.candidate}")
    print(f"Reference S1:   {test_s1_path}")
    print(f"Expected rows:  {args.expected_rows:,}")
    print(f"Size limit:     < {args.max_size_mb} MB")
    print("-" * 72)

    # 1. Existence and size verification
    m_ok, m_msg, m_size = validate_file_existence_and_size(
        args.matching, "matching_results.tsv", args.max_size_mb
    )
    if not m_ok:
        failures.append(m_msg)

    c_ok, c_msg, c_size = validate_file_existence_and_size(
        args.candidate, "candidate_pairs.tsv", args.max_size_mb
    )
    if not c_ok:
        failures.append(c_msg)

    if not os.path.exists(test_s1_path):
        failures.append(f"Reference test file not found at '{test_s1_path}'.")

    if failures:
        print("\nFATAL ERRORS DETECTED:")
        for idx, fail in enumerate(failures, 1):
            print(f"  {idx}. {fail}")
        print("\nValidation aborted. Result: FAIL")
        sys.exit(1)

    print(f"Matching file size:  {m_size:.2f} MB (PASS < {args.max_size_mb} MB)")
    print(f"Candidate file size: {c_size:.2f} MB (PASS < {args.max_size_mb} MB)")

    # 2. Header and row-by-row streaming verification
    expected_m_header = "source1_entity_id\tmatched_entity_ids"
    expected_c_header = "source1_entity_id\tcandidate_entity_ids"

    row_count = 0
    s1_mismatches = 0
    subset_violations = 0
    matching_format_errors = []
    candidate_format_errors = []
    sample_subset_violations = []

    total_matched_ids = 0
    total_candidate_ids = 0
    matched_singletons = 0
    candidate_singletons = 0

    with open(test_s1_path, "r", encoding="utf-8") as f_ref, \
         open(args.matching, "r", encoding="utf-8") as f_match, \
         open(args.candidate, "r", encoding="utf-8") as f_cand:

        # Header check
        ref_header = f_ref.readline().rstrip("\r\n")
        m_header = f_match.readline().rstrip("\r\n")
        c_header = f_cand.readline().rstrip("\r\n")

        if m_header != expected_m_header:
            failures.append(
                f"matching_results.tsv header is invalid.\n"
                f"    Expected: {expected_m_header!r}\n"
                f"    Got:      {m_header!r}"
            )

        if c_header != expected_c_header:
            failures.append(
                f"candidate_pairs.tsv header is invalid.\n"
                f"    Expected: {expected_c_header!r}\n"
                f"    Got:      {c_header!r}"
            )

        if not ref_header.startswith("entity_id\t"):
            failures.append(
                f"test_source1.tsv header unexpected: {ref_header!r}"
            )

        # Stream lines simultaneously
        while True:
            ref_line = f_ref.readline()
            m_line = f_match.readline()
            c_line = f_cand.readline()

            # Check EOF
            if not ref_line and not m_line and not c_line:
                break

            row_count += 1

            if not ref_line:
                failures.append(
                    f"Submission files have more rows than reference test file (> {row_count - 1:,} rows)."
                )
                break
            if not m_line:
                failures.append(
                    f"matching_results.tsv truncated early at row {row_count - 1:,}."
                )
                break
            if not c_line:
                failures.append(
                    f"candidate_pairs.tsv truncated early at row {row_count - 1:,}."
                )
                break

            # Parse ref S1 ID
            ref_parts = ref_line.rstrip("\r\n").split("\t")
            expected_s1_id = ref_parts[0]

            # Parse matching line
            m_parts = m_line.rstrip("\r\n").split("\t")
            if len(m_parts) > 2:
                if len(matching_format_errors) < 20:
                    matching_format_errors.append(
                        f"Row {row_count} in matching_results.tsv has {len(m_parts)} columns (expected 2)."
                    )
            m_s1_id = m_parts[0]
            m_val = m_parts[1] if len(m_parts) > 1 else ""

            # Parse candidate line
            c_parts = c_line.rstrip("\r\n").split("\t")
            if len(c_parts) > 2:
                if len(candidate_format_errors) < 20:
                    candidate_format_errors.append(
                        f"Row {row_count} in candidate_pairs.tsv has {len(c_parts)} columns (expected 2)."
                    )
            c_s1_id = c_parts[0]
            c_val = c_parts[1] if len(c_parts) > 1 else ""

            # Check S1 entity ID alignment
            if m_s1_id != expected_s1_id or c_s1_id != expected_s1_id:
                s1_mismatches += 1
                if s1_mismatches <= 5:
                    failures.append(
                        f"Row {row_count} S1 ID mismatch: expected '{expected_s1_id}', "
                        f"matching got '{m_s1_id}', candidate got '{c_s1_id}'."
                    )

            # Validate IDs formatting, prefixes, duplicates
            m_ids, m_set = parse_id_list(
                m_val, row_count, "matching_results.tsv", matching_format_errors
            )
            c_ids, c_set = parse_id_list(
                c_val, row_count, "candidate_pairs.tsv", candidate_format_errors
            )

            total_matched_ids += len(m_ids)
            total_candidate_ids += len(c_ids)

            if len(m_ids) == 0:
                matched_singletons += 1
            if len(c_ids) == 0:
                candidate_singletons += 1

            # Strict Subset Rule: P <= C
            if not m_set.issubset(c_set):
                subset_violations += 1
                if len(sample_subset_violations) < 5:
                    missing_ids = m_set - c_set
                    sample_subset_violations.append(
                        f"Row {row_count} ({expected_s1_id}): Predicted IDs {missing_ids} "
                        f"not found in candidate set."
                    )

            # Periodically print progress
            if row_count % 500000 == 0:
                print(f"Validated {row_count:,} / {args.expected_rows:,} rows...")

    # Validate row count
    if row_count != args.expected_rows:
        failures.append(
            f"Row count mismatch: processed {row_count:,} rows, expected exactly {args.expected_rows:,}."
        )

    # Collect format errors
    if matching_format_errors:
        failures.append(
            f"matching_results.tsv format errors ({len(matching_format_errors)} instances, first few: "
            + "; ".join(matching_format_errors[:3])
            + ")"
        )
    if candidate_format_errors:
        failures.append(
            f"candidate_pairs.tsv format errors ({len(candidate_format_errors)} instances, first few: "
            + "; ".join(candidate_format_errors[:3])
            + ")"
        )

    # Collect subset violations
    if subset_violations > 0:
        failures.append(
            f"Strict Subset Rule violated in {subset_violations:,} rows! Every predicted match must be in candidate pairs.\n"
            f"    Sample violations:\n    "
            + "\n    ".join(sample_subset_violations)
        )

    print("-" * 72)
    print("VALIDATION SUMMARY REPORT:")
    print(f"  Total Rows Checked:      {row_count:,}")
    print(f"  Total Matched IDs:       {total_matched_ids:,}")
    print(f"  Matched Singletons:      {matched_singletons:,} ({matched_singletons / max(row_count, 1) * 100:.2f}%)")
    print(f"  Total Candidate IDs:     {total_candidate_ids:,}")
    print(f"  Candidate Singletons:    {candidate_singletons:,} ({candidate_singletons / max(row_count, 1) * 100:.2f}%)")
    print(f"  S1 Entity Alignments:    {'PASS (0 mismatches)' if s1_mismatches == 0 else f'FAIL ({s1_mismatches:,} mismatches)'}")
    print(f"  Strict Subset Rule:      {'PASS (0 violations)' if subset_violations == 0 else f'FAIL ({subset_violations:,} violations)'}")
    print(f"  Formatting & Prefixes:   {'PASS' if not matching_format_errors and not candidate_format_errors else 'FAIL'}")
    print("-" * 72)

    if failures:
        print("\n======================================================================")
        print(f"SUBMISSION VALIDATION FAILED - {len(failures)} VIOLATIONS FOUND")
        print("======================================================================")
        for idx, failure in enumerate(failures, 1):
            print(f"{idx}. {failure}")
        print("\nPlease fix the above violations before submitting to the portal.")
        sys.exit(1)

    print("\n======================================================================")
    print("SUBMISSION VALIDATION PASSED - 100% COMPLIANT WITH OFFICIAL RULES")
    print("======================================================================")
    print("Files are valid, strictly under 512 MB, and ready for portal upload.")
    sys.exit(0)


if __name__ == "__main__":
    main()
