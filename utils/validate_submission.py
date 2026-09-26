#!/usr/bin/env python3
"""
Submission Validator for Amazon ML Challenge 2026.
Standard library only (no external dependencies).

Checks matching_results.tsv and candidate_pairs.tsv against official rules:
  1. Header format (tab-separated, exact column names)
  2. Exactly one row per test Source 1 entity (no missing, no duplicates)
  3. Valid entity IDs (S2- and S3- only; must exist in test_source2 / test_source3)
  4. No self-matches (S1- IDs strictly prohibited in match lists)
  5. No intra-list duplicates
  6. Subset integrity: matching_results IDs must be a subset of candidate_pairs IDs
  7. Correct formatting for singletons (empty matched_entity_ids)

Exit Code:
  0: PASS (All checks cleared, safe to submit)
  1: Issues found (numbered list printed)
"""

import argparse
import os
import sys
from typing import Dict, List, Set


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Amazon ML Challenge 2026 submission files.")
    parser.add_argument("--matching", type=str, required=True,
                        help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=str, required=True,
                        help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")
    return parser.parse_args()


def load_test_ids(test_dir: str):
    """Loads all valid entity IDs from test source files using streaming reading."""
    def find_file(fname):
        for candidate in [os.path.join(test_dir, fname), os.path.join(test_dir, "test", fname)]:
            if os.path.exists(candidate):
                return candidate
        return os.path.join(test_dir, fname)

    s1_file = find_file("test_source1.tsv")
    s2_file = find_file("test_source2.tsv")
    s3_file = find_file("test_source3.tsv")

    for fpath in [s1_file, s2_file, s3_file]:
        if not os.path.exists(fpath):
            print(f"ERROR: Missing test file: {fpath}", file=sys.stderr)
            sys.exit(1)

    print(f"[*] Loading valid Test IDs from {test_dir}...")
    
    # Load S1 IDs
    test_s1_ids = set()
    with open(s1_file, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[id_idx]:
                test_s1_ids.add(parts[id_idx])

    # Load S2 and S3 IDs into candidate pool
    valid_candidate_ids = set()
    for fpath, prefix in [(s2_file, "S2-"), (s3_file, "S3-")]:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split("\t")
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            for line in f:
                parts = line.strip().split("\t")
                if parts and parts[id_idx]:
                    valid_candidate_ids.add(parts[id_idx])

    print(f"    - Test S1 Entities: {len(test_s1_ids):,}")
    print(f"    - Test S2/S3 Target Pool: {len(valid_candidate_ids):,}")
    return test_s1_ids, valid_candidate_ids


def validate_file(
    file_path: str,
    expected_header: List[str],
    test_s1_ids: Set[str],
    valid_candidate_ids: Set[str],
    file_type: str,
    issues: List[str]
) -> Dict[str, Set[str]]:
    """Validates a single TSV file and returns dict mapping s1_id -> set of candidate/matched IDs."""
    if not os.path.exists(file_path):
        issues.append(f"File not found: {file_path}")
        return {}

    parsed_data: Dict[str, Set[str]] = {}
    seen_s1_ids: Set[str] = set()
    duplicate_rows = 0
    invalid_prefix_cnt = 0
    unseen_candidate_cnt = 0
    duplicate_ids_in_list_cnt = 0

    print(f"[*] Validating {file_type} ({file_path})...")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        # Check header
        first_line = f.readline()
        if not first_line:
            issues.append(f"{file_type}: File is completely empty.")
            return {}

        header = first_line.rstrip("\r\n").split("\t")
        if header != expected_header:
            issues.append(
                f"{file_type}: Incorrect header. Expected {expected_header}, got {header}."
            )

        line_num = 1
        for line in f:
            line_num += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 1:
                # In TSV, a singleton line has "S1-XXXX\t" or "S1-XXXX"
                s1_id = parts[0]
                ids_str = ""
            elif len(parts) == 2:
                s1_id = parts[0]
                ids_str = parts[1]
            else:
                issues.append(
                    f"{file_type} (Line {line_num}): Expected 2 tab-separated columns, found {len(parts)}."
                )
                continue

            # Check duplicate S1 row
            if s1_id in seen_s1_ids:
                duplicate_rows += 1
            seen_s1_ids.add(s1_id)

            # Parse matched / candidate ID list
            if ids_str.strip():
                raw_ids = [x.strip() for x in ids_str.split(",")]
                id_set = set(raw_ids)
                if len(raw_ids) != len(id_set):
                    duplicate_ids_in_list_cnt += 1

                for eid in id_set:
                    if not (eid.startswith("S2-") or eid.startswith("S3-")):
                        invalid_prefix_cnt += 1
                    elif eid not in valid_candidate_ids:
                        unseen_candidate_cnt += 1

                parsed_data[s1_id] = id_set
            else:
                parsed_data[s1_id] = set()

    # Rule checks
    if duplicate_rows > 0:
        issues.append(f"{file_type}: Found {duplicate_rows:,} duplicate source1_entity_id rows.")

    missing_s1 = test_s1_ids - seen_s1_ids
    if missing_s1:
        issues.append(
            f"{file_type}: Missing {len(missing_s1):,} Source 1 test entities (e.g., {list(missing_s1)[:3]}). Every test entity must be present."
        )

    extra_s1 = seen_s1_ids - test_s1_ids
    if extra_s1:
        issues.append(
            f"{file_type}: Found {len(extra_s1):,} unrecognized source1_entity_ids not in test_source1.tsv."
        )

    if invalid_prefix_cnt > 0:
        issues.append(
            f"{file_type}: Found {invalid_prefix_cnt:,} entity IDs not starting with 'S2-' or 'S3-' (Self-matches to S1 are forbidden)."
        )

    if unseen_candidate_cnt > 0:
        issues.append(
            f"{file_type}: Found {unseen_candidate_cnt:,} entity IDs that do not exist in test_source2.tsv or test_source3.tsv."
        )

    if duplicate_ids_in_list_cnt > 0:
        issues.append(
            f"{file_type}: Found {duplicate_ids_in_list_cnt:,} rows with duplicate IDs inside the comma-separated list."
        )

    return parsed_data


def check_subset_integrity(
    matching_dict: Dict[str, Set[str]],
    candidate_dict: Dict[str, Set[str]],
    issues: List[str]
):
    """Verifies that every matched ID is present in candidate_pairs.tsv."""
    print("[*] Checking subset integrity (matching_results subset of candidate_pairs)...")
    not_in_candidates_count = 0
    sample_violations = []

    for s1_id, matched_ids in matching_dict.items():
        cand_ids = candidate_dict.get(s1_id, set())
        diff = matched_ids - cand_ids
        if diff:
            not_in_candidates_count += len(diff)
            if len(sample_violations) < 3:
                sample_violations.append((s1_id, list(diff)[:2]))

    if not_in_candidates_count > 0:
        issues.append(
            f"Subset Violation: {not_in_candidates_count:,} matched IDs in matching_results.tsv were NEVER present in candidate_pairs.tsv! (Samples: {sample_violations})"
        )


def main():
    args = parse_args()
    print("=" * 65)
    print("       AMAZON ML CHALLENGE 2026 - SUBMISSION VALIDATOR")
    print("=" * 65)

    test_s1_ids, valid_candidate_ids = load_test_ids(args.test_dir)
    issues: List[str] = []

    # 1. Validate matching_results.tsv
    matching_data = validate_file(
        file_path=args.matching,
        expected_header=["source1_entity_id", "matched_entity_ids"],
        test_s1_ids=test_s1_ids,
        valid_candidate_ids=valid_candidate_ids,
        file_type="matching_results.tsv",
        issues=issues
    )

    # 2. Validate candidate_pairs.tsv
    candidate_data = validate_file(
        file_path=args.candidate,
        expected_header=["source1_entity_id", "candidate_entity_ids"],
        test_s1_ids=test_s1_ids,
        valid_candidate_ids=valid_candidate_ids,
        file_type="candidate_pairs.tsv",
        issues=issues
    )

    # 3. Check subset integrity if both loaded
    if matching_data and candidate_data:
        check_subset_integrity(matching_data, candidate_data, issues)

    print("=" * 65)
    if not issues:
        singletons_cnt = sum(1 for m in matching_data.values() if len(m) == 0)
        total_matches = sum(len(m) for m in matching_data.values())
        print(" PASS: All validation checks cleared successfully!")
        print(f" Summary:")
        print(f"   - Total S1 Entities:     {len(matching_data):,}")
        print(f"   - Total Matches Found:   {total_matches:,}")
        print(f"   - Singletons (no match): {singletons_cnt:,} ({singletons_cnt/len(matching_data)*100:.2f}%)")
        print(" Files are 100% compliant with competition rules and safe to submit.")
        print("=" * 65)
        sys.exit(0)
    else:
        print(f" FAIL: Found {len(issues)} issue(s) that will cause portal rejection:")
        for idx, issue in enumerate(issues, 1):
            print(f"   {idx}. {issue}")
        print("=" * 65)
        sys.exit(1)


if __name__ == "__main__":
    main()
