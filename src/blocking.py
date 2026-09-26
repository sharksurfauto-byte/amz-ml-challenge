#!/usr/bin/env python3
"""
Fast, scalable candidate pair generation partitioned strictly by country.

Techniques:
  1. Strict Country Partitioning (Zero cross-country matches in data).
  2. Unicode NFKD Diacritic Normalization (Critical for France in Test set).
  3. URL/Domain Stripping for Source 3 Web Records (e.g., 'xyzcompany.com' -> 'xyzcompany').
  4. Legal Suffix & Entity Noise Removal.
  5. Multi-Token Inverted Indexing with Frequency Capping (Prunes generic stopwords like 'store', 'restaurant').
  6. 4-Character Name Prefix Matching (Recovers misspelled stems and brand variations).
  7. Exact First-Token Priority Boost.

Outputs:
  - candidate_pairs.tsv (source1_entity_id \t candidate_entity_ids)
"""

import os
import re
import sys
import time
import argparse
import unicodedata
import itertools
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

import polars as pl


COMMON_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "co", "company", "sarl", "sa", "services", "enterprises",
    "industries", "group", "holdings", "solutions", "international", "tech",
    "technologies", "trading", "agency", "associates", "consulting", "gmbh"
}

DOMAIN_REGEX = re.compile(
    r"([a-z0-9-]+)\.(?:com|org|net|in|co|us|biz|info|io|fr|gov|edu)\b",
    flags=re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Normalizes text by removing domain extensions, standardizing unicode, and lowercasing."""
    if not text or not isinstance(text, str):
        return ""
    # Strip basic web domain extensions commonly found in S3
    text = DOMAIN_REGEX.sub(r"\1", text)
    # Unicode NFKD normalization to cleanly strip French diacritics
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    # Strip punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def extract_tokens(text: str) -> List[str]:
    """Extracts distinctive tokens for inverted indexing, filtering out legal suffixes and noise."""
    norm = normalize_text(text)
    if not norm:
        return []
    tokens = [
        t for t in norm.split()
        if len(t) >= 3 and t not in COMMON_SUFFIXES and not t.isdigit()
    ]
    return tokens


def extract_prefix(text: str, prefix_len: int = 4) -> str:
    """Extracts the first N characters of the normalized name as a blocking key."""
    norm = normalize_text(text)
    if not norm:
        return ""
    # First token stripped of non-alpha
    tokens = [t for t in norm.split() if t not in COMMON_SUFFIXES]
    if tokens and len(tokens[0]) >= prefix_len:
        return tokens[0][:prefix_len]
    return ""


class MultiKeyCandidateBlocker:
    def __init__(self, top_k: int = 20, max_token_freq: int = 50000):
        self.top_k = top_k
        self.max_token_freq = max_token_freq

        # Use integer IDs for 5x faster Counter hashing
        self.id_to_str: Dict[int, str] = {}
        self.str_to_id: Dict[str, int] = {}
        self._next_id = 0

        # (country, token) -> list of candidate integer IDs
        self.token_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.prefix_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.first_token_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    def _get_id(self, entity_str: str) -> int:
        if entity_str not in self.str_to_id:
            self.str_to_id[entity_str] = self._next_id
            self.id_to_str[self._next_id] = entity_str
            self._next_id += 1
        return self.str_to_id[entity_str]

    def fit(self, s2_path: str, s3_path: str):
        """Builds multi-rule inverted index over S2 and S3 candidate records."""
        t0 = time.time()
        print("[*] Building Multi-Key Inverted Candidate Index (S2 & S3)...")

        total_candidates = 0
        for path, src_name in [(s2_path, "Source 2"), (s3_path, "Source 3")]:
            print(f"    - Loading {src_name} ({os.path.basename(path)})...")
            df = pl.read_csv(
                path,
                separator="\t",
                columns=["entity_id", "business_name", "country"],
                infer_schema_length=10000
            )

            for row in df.iter_rows(named=True):
                eid = row["entity_id"]
                name = row["business_name"]
                country = row["country"]

                if not country or not name:
                    continue

                total_candidates += 1
                eid_int = self._get_id(eid)
                tokens = extract_tokens(name)

                # Index all distinctive tokens
                for t in tokens:
                    self.token_index[(country, t)].append(eid_int)

                # Index first token (exact match anchor)
                if tokens:
                    self.first_token_index[(country, tokens[0])].append(eid_int)

                # Index 4-char prefix for fuzzy / misspelling capture
                prefix = extract_prefix(name, prefix_len=4)
                if prefix:
                    self.prefix_index[(country, prefix)].append(eid_int)

        print(f"[*] Raw Index Built across {total_candidates:,} candidate entities in {time.time()-t0:.2f}s.")
        print(f"    - Unique (country, token) keys:  {len(self.token_index):,}")
        print(f"    - Unique (country, prefix) keys: {len(self.prefix_index):,}")

        # Prune high-frequency tokens (generic stopwords like 'store', 'market', 'hotel', 'traders')
        print(f"[*] Pruning stop-tokens exceeding frequency threshold ({self.max_token_freq:,})...")
        pruned_cnt = 0
        for key in list(self.token_index.keys()):
            if len(self.token_index[key]) > self.max_token_freq:
                del self.token_index[key]
                pruned_cnt += 1

        print(f"    - Pruned {pruned_cnt:,} generic stop-tokens. Active token keys: {len(self.token_index):,}")

    def query(self, s1_path: str) -> Dict[str, List[str]]:
        """Queries S1 records against index to retrieve Top-K candidates per entity."""
        t0 = time.time()
        print(f"\n[*] Querying S1 Entities from {os.path.basename(s1_path)}...")

        df_s1 = pl.read_csv(
            s1_path,
            separator="\t",
            columns=["entity_id", "business_name", "country"],
            infer_schema_length=10000
        )

        candidate_map: Dict[str, List[str]] = {}
        zero_candidate_cnt = 0
        total_s1 = len(df_s1)

        for idx, row in enumerate(df_s1.iter_rows(named=True)):
            s1_id = row["entity_id"]
            name = row["business_name"]
            country = row["country"]

            if not country or not name:
                candidate_map[s1_id] = []
                zero_candidate_cnt += 1
                continue

            tokens = extract_tokens(name)
            prefix = extract_prefix(name, prefix_len=4)

            # Lists of candidate IDs to feed to itertools.chain
            # Repeating a list N times is equivalent to giving it weight N in Counter
            cids_lists = []

            # Rule 1: First-token exact match (High confidence boost) -> Weight 4
            if tokens and (country, tokens[0]) in self.first_token_index:
                # 80,000 cap to prevent outright explosion, but high enough to capture massive clusters
                exact_cands = self.first_token_index[(country, tokens[0])][:80000]
                if exact_cands:
                    cids_lists.extend([exact_cands] * 4)

            # Rule 2: Multi-token inverted index overlap -> Weight 2
            for t in tokens:
                if (country, t) in self.token_index:
                    tok_cands = self.token_index[(country, t)]
                    if tok_cands:
                        cids_lists.extend([tok_cands] * 2)

            # Rule 3: 4-character prefix match (Fuzzy typo fallback) -> Weight 1
            if prefix and (country, prefix) in self.prefix_index:
                # 30,000 cap
                pref_cands = self.prefix_index[(country, prefix)][:30000]
                if pref_cands:
                    cids_lists.append(pref_cands)

            if not cids_lists:
                candidate_map[s1_id] = []
                zero_candidate_cnt += 1
            else:
                # Chain all lists and count frequencies (this runs entirely in C and is extremely fast)
                flat_cids = itertools.chain.from_iterable(cids_lists)
                counts = Counter(flat_cids)
                candidate_map[s1_id] = [self.id_to_str[cid] for cid, _ in counts.most_common(self.top_k)]

            if (idx + 1) % 250000 == 0:
                print(f"    - Processed {idx + 1:,} / {total_s1:,} queries ({(idx + 1)/total_s1*100:.1f}%)...")

        elapsed = time.time() - t0
        print(f"[*] Completed {total_s1:,} queries in {elapsed:.2f}s ({total_s1/elapsed:,.0f} queries/sec).")
        print(f"    - Entities with >=1 candidate: {total_s1 - zero_candidate_cnt:,} ({(total_s1 - zero_candidate_cnt)/total_s1*100:.2f}%)")
        print(f"    - Singletons (0 candidates):    {zero_candidate_cnt:,} ({zero_candidate_cnt/total_s1*100:.2f}%)")

        return candidate_map


def evaluate_recall_ceiling(candidate_map: Dict[str, List[str]], gt_path: str):
    """Calculates the recall ceiling of the candidate generator against ground truth."""
    print(f"\n[*] Evaluating Candidate Recall Ceiling against {os.path.basename(gt_path)}...")
    t0 = time.time()

    total_gt_matches = 0
    captured_gt_matches = 0
    num_gt_evaluated = 0
    candidate_lengths = []

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            if s1_id not in candidate_map:
                continue

            num_gt_evaluated += 1
            cand_set = set(candidate_map[s1_id])
            candidate_lengths.append(len(cand_set))

            if len(parts) > 1 and parts[1].strip():
                gt_ids = [x.strip() for x in parts[1].split(",") if x.strip()]
                total_gt_matches += len(gt_ids)
                captured = sum(1 for gid in gt_ids if gid in cand_set)
                captured_gt_matches += captured

    recall_ceiling = (captured_gt_matches / total_gt_matches * 100) if total_gt_matches else 0.0
    avg_candidates = sum(candidate_lengths) / len(candidate_lengths) if candidate_lengths else 0.0

    print("=" * 65)
    print(" [CANDIDATE BLOCKING RECALL CEILING EVALUATION]")
    print("=" * 65)
    print(f" Evaluated S1 Entities:          {num_gt_evaluated:,}")
    print(f" Total True Matches in GT:       {total_gt_matches:,}")
    print(f" True Matches in Candidate Set:  {captured_gt_matches:,}")
    print(f" Candidate Recall Ceiling:       {recall_ceiling:.2f}%")
    print(f" Average Candidates per S1:      {avg_candidates:.1f}")
    print("=" * 65)


def write_candidate_pairs_tsv(candidate_map: Dict[str, List[str]], output_path: str):
    """Writes the candidate pairs in the exact competition submission format."""
    print(f"\n[*] Writing candidate pairs to {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in candidate_map.items():
            cand_str = ",".join(cands)
            f.write(f"{s1_id}\t{cand_str}\n")

    print(f"[+] Successfully wrote {len(candidate_map):,} rows to {output_path}!")


def main():
    parser = argparse.ArgumentParser(description="Multi-Key Candidate Blocker for Amazon ML Challenge 2026")
    parser.add_argument("--s1", type=str, default="data/raw/test/test_source1.tsv", help="Path to source1 TSV")
    parser.add_argument("--s2", type=str, default="data/raw/test/test_source2.tsv", help="Path to source2 TSV")
    parser.add_argument("--s3", type=str, default="data/raw/test/test_source3.tsv", help="Path to source3 TSV")
    parser.add_argument("--out", type=str, default="output/candidate_pairs.tsv", help="Path to output candidate_pairs.tsv")
    parser.add_argument("--top-k", type=int, default=20, help="Max candidates per S1 entity (default: 20)")
    parser.add_argument("--max-freq", type=int, default=60000, help="Max token frequency threshold (default: 60000)")
    parser.add_argument("--eval-gt", type=str, default=None, help="Optional path to ground truth TSV to compute recall ceiling")
    args = parser.parse_args()

    blocker = MultiKeyCandidateBlocker(top_k=args.top_k, max_token_freq=args.max_freq)
    blocker.fit(s2_path=args.s2, s3_path=args.s3)
    candidate_map = blocker.query(s1_path=args.s1)

    if args.eval_gt and os.path.exists(args.eval_gt):
        evaluate_recall_ceiling(candidate_map, args.eval_gt)

    write_candidate_pairs_tsv(candidate_map, args.out)


if __name__ == "__main__":
    main()
