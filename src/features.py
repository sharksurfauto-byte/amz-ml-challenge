"""
High-Speed Feature Extraction Engine for Candidate Pairs (S1, S_cand).
Uses C++ RapidFuzz and Polars to extract pairwise lexical, address, legal, domain,
and structural ambiguity features.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz

logger = logging.getLogger("features")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Core Feature Names for Phase 3 GBDT Re-Ranker
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "name_len_diff",
    "name_jaro",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "exact_legal_match",
    "address_jaro",
    "exact_street_num_match",
    "domain_root_match",
    "cands_per_s1",
    "claims_per_cand",
]

# Legacy feature names maintained for backward compatibility
FEATURE_NAMES: List[str] = [
    "fuzz_token_sort_ratio",
    "fuzz_token_set_ratio",
    "fuzz_partial_ratio",
    "jaro_winkler_sim",
    "exact_name_match",
    "clean_name_match",
    "name_len_diff",
    "name_len_ratio",
    "has_s3_domain",
    "domain_root_match",
    "both_have_address",
    "street_number_match",
    "address_jaccard",
    "candidate_rank",
    "total_s1_candidates",
]

COMMON_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "co", "company", "sarl", "sa", "sas", "societe", "snc", "scs", "sca",
    "services", "enterprises", "industries", "group", "holdings", "solutions", "international", "tech",
    "technologies", "trading", "agency", "associates", "consulting", "gmbh", "llp",
}

DOMAIN_REGEX = re.compile(
    r"([a-z0-9-]+)\.(?:com|org|net|in|co|us|biz|info|io|fr|gov|edu)\b",
    flags=re.IGNORECASE,
)
STREET_NUM_REGEX = re.compile(r"\b(\d{1,5})\b")


# ---------------------------------------------------------------------------
# Helper Text Normalization (Legacy & Fallback)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalizes text: removes domain extensions, strips accents, lowercases, cleans punctuation."""
    if not text or not isinstance(text, str):
        return ""
    text = DOMAIN_REGEX.sub(r"\1", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_clean_tokens(text: str) -> List[str]:
    """Extracts non-suffix words."""
    norm = normalize_text(text)
    if not norm:
        return []
    return [t for t in norm.split() if len(t) >= 3 and t not in COMMON_SUFFIXES and not t.isdigit()]


def extract_domain_root(text: str) -> Optional[str]:
    """Extracts domain root from text if present (e.g. 'xyzcompany.com' -> 'xyzcompany')."""
    if not text or not isinstance(text, str):
        return None
    matches = DOMAIN_REGEX.findall(text)
    return matches[0].lower() if matches else None


def extract_street_number(address: str) -> Optional[str]:
    """Extracts leading street number from address."""
    if not address or not isinstance(address, str):
        return None
    nums = STREET_NUM_REGEX.findall(address)
    return nums[0] if nums else None


# ---------------------------------------------------------------------------
# Parallel Worker for String Distances
# ---------------------------------------------------------------------------
def _compute_string_features_chunk(
    s1_names: List[str],
    cand_names: List[str],
    s1_addrs: List[str],
    cand_addrs: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes rapidfuzz string distances for a chunk of paired strings.
    Pre-allocates float32 numpy arrays for minimal memory footprint and maximum speed.
    """
    n_rows = len(s1_names)
    name_jaro = np.empty(n_rows, dtype=np.float32)
    name_token_set = np.empty(n_rows, dtype=np.float32)
    name_token_sort = np.empty(n_rows, dtype=np.float32)
    address_jaro = np.empty(n_rows, dtype=np.float32)

    for i in range(n_rows):
        s1_n = s1_names[i]
        c_n = cand_names[i]
        s1_a = s1_addrs[i]
        c_a = cand_addrs[i]

        if s1_n and c_n:
            name_jaro[i] = distance.Jaro.similarity(s1_n, c_n)
            name_token_set[i] = fuzz.token_set_ratio(s1_n, c_n) / 100.0
            name_token_sort[i] = fuzz.token_sort_ratio(s1_n, c_n) / 100.0
        else:
            name_jaro[i] = 0.0
            name_token_set[i] = 0.0
            name_token_sort[i] = 0.0

        if s1_a and c_a:
            address_jaro[i] = distance.Jaro.similarity(s1_a, c_a)
        else:
            address_jaro[i] = 0.0

    return name_jaro, name_token_set, name_token_sort, address_jaro


# ---------------------------------------------------------------------------
# Core Fast Pairwise Feature Extraction Function
# ---------------------------------------------------------------------------
def build_pairwise_features(
    s1_df: Union[pl.DataFrame, pl.LazyFrame],
    cand_df: Union[pl.DataFrame, pl.LazyFrame],
    pairs_df: Union[pl.DataFrame, pl.LazyFrame],
    n_jobs: int = -1,
    batch_size: int = 250000,
) -> pl.DataFrame:
    """
    Fast parallel extraction of pairwise features using Polars native expressions
    and C++ RapidFuzz multithreading.

    Features computed:
      - name_len_diff: Absolute difference in length between cleaned names.
      - name_jaro: Jaro similarity between cleaned business names.
      - name_token_set_ratio: RapidFuzz token set ratio (0.0 to 1.0).
      - name_token_sort_ratio: RapidFuzz token sort ratio (0.0 to 1.0).
      - exact_legal_match: 1.0 if identical non-empty legal forms, 0.0 if different, -1.0 if missing.
      - address_jaro: Jaro similarity between cleaned addresses.
      - exact_street_num_match: 1.0 if identical non-empty street numbers, 0.0 if different, -1.0 if missing.
      - domain_root_match: 1.0 if candidate domain matches S1 domain or is contained in S1 name, 0.0 otherwise.
      - cands_per_s1: Frequency of source1_entity_id_int in pairs_df (structural ambiguity).
      - claims_per_cand: Frequency of candidate_entity_id_int in pairs_df (candidate popularity).

    Args:
        s1_df: S1 reference entities with columns (name_clean, address_clean, legal_form, domain_root, address_street_number).
        cand_df: Candidate pool entities with the same schema.
        pairs_df: Candidate pairs DataFrame containing (source1_entity_id_int, candidate_entity_id_int).
        n_jobs: Number of parallel worker threads (-1 for all available cores).
        batch_size: Chunk size for thread distribution.

    Returns:
        pl.DataFrame: pairs_df augmented with all 10 feature columns.
    """
    if isinstance(s1_df, pl.LazyFrame):
        s1_df = s1_df.collect()
    if isinstance(cand_df, pl.LazyFrame):
        cand_df = cand_df.collect()
    if isinstance(pairs_df, pl.LazyFrame):
        pairs_df = pairs_df.collect()

    if pairs_df.height == 0:
        schema = dict(pairs_df.schema)
        for col in FEATURE_COLUMNS:
            schema[col] = pl.UInt32 if col in ("cands_per_s1", "claims_per_cand") else pl.Float32
        return pl.DataFrame(schema=schema)

    # 1. Identify and resolve ID column names in pairs_df
    s1_id_col = "source1_entity_id_int" if "source1_entity_id_int" in pairs_df.columns else (
        "source1_entity_id" if "source1_entity_id" in pairs_df.columns else "s1_id"
    )
    cand_id_col = "candidate_entity_id_int" if "candidate_entity_id_int" in pairs_df.columns else (
        "candidate_entity_id" if "candidate_entity_id" in pairs_df.columns else "cand_id"
    )

    # 2. Identify ID column names in s1_df and cand_df
    s1_id_in_s1 = "source1_entity_id_int" if "source1_entity_id_int" in s1_df.columns else (
        "entity_id_int" if "entity_id_int" in s1_df.columns else (
            "source1_entity_id" if "source1_entity_id" in s1_df.columns else (
                "entity_id" if "entity_id" in s1_df.columns else s1_id_col
            )
        )
    )
    cand_id_in_cand = "candidate_entity_id_int" if "candidate_entity_id_int" in cand_df.columns else (
        "entity_id_int" if "entity_id_int" in cand_df.columns else (
            "candidate_entity_id" if "candidate_entity_id" in cand_df.columns else (
                "entity_id" if "entity_id" in cand_df.columns else cand_id_col
            )
        )
    )

    # Helper to resolve/fallback metadata columns
    def _prepare_metadata_table(df: pl.DataFrame, id_in: str, id_out: str, prefix: str) -> pl.DataFrame:
        exprs = [pl.col(id_in).alias(id_out)]

        # name_clean
        if "name_clean" in df.columns:
            exprs.append(pl.col("name_clean").fill_null("").cast(pl.String).alias(f"{prefix}_name_clean"))
        elif "business_name" in df.columns:
            exprs.append(pl.col("business_name").fill_null("").cast(pl.String).str.to_lowercase().alias(f"{prefix}_name_clean"))
        else:
            exprs.append(pl.lit("").alias(f"{prefix}_name_clean"))

        # address_clean
        if "address_clean" in df.columns:
            exprs.append(pl.col("address_clean").fill_null("").cast(pl.String).alias(f"{prefix}_address_clean"))
        elif "business_address" in df.columns:
            exprs.append(pl.col("business_address").fill_null("").cast(pl.String).str.to_lowercase().alias(f"{prefix}_address_clean"))
        else:
            exprs.append(pl.lit("").alias(f"{prefix}_address_clean"))

        # legal_form
        if "legal_form" in df.columns:
            exprs.append(pl.col("legal_form").fill_null("").cast(pl.String).alias(f"{prefix}_legal_form"))
        else:
            exprs.append(pl.lit("").alias(f"{prefix}_legal_form"))

        # domain_root
        if "domain_root" in df.columns:
            exprs.append(pl.col("domain_root").fill_null("").cast(pl.String).alias(f"{prefix}_domain_root"))
        else:
            exprs.append(pl.lit("").alias(f"{prefix}_domain_root"))

        # address_street_number
        if "address_street_number" in df.columns:
            exprs.append(pl.col("address_street_number").fill_null("").cast(pl.String).alias(f"{prefix}_address_street_number"))
        else:
            exprs.append(pl.lit("").alias(f"{prefix}_address_street_number"))

        return df.select(exprs)

    s1_sub = _prepare_metadata_table(s1_df, s1_id_in_s1, s1_id_col, "s1")
    cand_sub = _prepare_metadata_table(cand_df, cand_id_in_cand, cand_id_col, "cand")

    # 3. Join metadata columns onto pairs_df
    joined = (
        pairs_df
        .join(s1_sub, on=s1_id_col, how="left")
        .join(cand_sub, on=cand_id_col, how="left")
    )

    # Fill null strings produced by any unmatched IDs
    str_cols_to_fill = [
        "s1_name_clean", "cand_name_clean",
        "s1_address_clean", "cand_address_clean",
        "s1_legal_form", "cand_legal_form",
        "s1_domain_root", "cand_domain_root",
        "s1_address_street_number", "cand_address_street_number",
    ]
    joined = joined.with_columns([pl.col(c).fill_null("") for c in str_cols_to_fill])

    # 4. Polars native expressions: structural, length diff, exact matches, domain root match
    polars_features = [
        pl.len().over(s1_id_col).cast(pl.UInt32).alias("cands_per_s1"),
        pl.len().over(cand_id_col).cast(pl.UInt32).alias("claims_per_cand"),
        (
            pl.col("s1_name_clean").str.len_bytes().cast(pl.Int32)
            - pl.col("cand_name_clean").str.len_bytes().cast(pl.Int32)
        ).abs().cast(pl.Float32).alias("name_len_diff"),
        pl.when(
            (pl.col("s1_legal_form") != "")
            & (pl.col("cand_legal_form") != "")
            & (pl.col("s1_legal_form") == pl.col("cand_legal_form"))
        )
        .then(1.0)
        .when((pl.col("s1_legal_form") != "") & (pl.col("cand_legal_form") != ""))
        .then(0.0)
        .otherwise(-1.0)
        .cast(pl.Float32)
        .alias("exact_legal_match"),
        pl.when(
            (pl.col("s1_address_street_number") != "")
            & (pl.col("cand_address_street_number") != "")
            & (pl.col("s1_address_street_number") == pl.col("cand_address_street_number"))
        )
        .then(1.0)
        .when((pl.col("s1_address_street_number") != "") & (pl.col("cand_address_street_number") != ""))
        .then(0.0)
        .otherwise(-1.0)
        .cast(pl.Float32)
        .alias("exact_street_num_match"),
        pl.when(
            (pl.col("cand_domain_root") != "")
            & (
                ((pl.col("s1_domain_root") != "") & (pl.col("s1_domain_root") == pl.col("cand_domain_root")))
                | (pl.col("s1_name_clean").str.contains(pl.col("cand_domain_root"), literal=True))
            )
        )
        .then(1.0)
        .when(pl.col("cand_domain_root") != "")
        .then(0.0)
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("domain_root_match"),
    ]

    joined = joined.with_columns(polars_features)

    # 5. Multithreaded RapidFuzz string distances (name_jaro, name_token_set_ratio, name_token_sort_ratio, address_jaro)
    num_workers = os.cpu_count() or 4 if n_jobs <= 0 else n_jobs
    total_rows = joined.height

    s1_names = joined["s1_name_clean"].to_list()
    cand_names = joined["cand_name_clean"].to_list()
    s1_addrs = joined["s1_address_clean"].to_list()
    cand_addrs = joined["cand_address_clean"].to_list()

    chunk_size = max(1000, (total_rows + num_workers - 1) // num_workers)
    chunk_slices = []
    for st in range(0, total_rows, chunk_size):
        en = min(total_rows, st + chunk_size)
        chunk_slices.append((
            s1_names[st:en],
            cand_names[st:en],
            s1_addrs[st:en],
            cand_addrs[st:en],
        ))

    if num_workers > 1 and len(chunk_slices) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chunk_results = list(executor.map(lambda c: _compute_string_features_chunk(*c), chunk_slices))
    else:
        chunk_results = [_compute_string_features_chunk(*c) for c in chunk_slices]

    name_jaro_arr = np.concatenate([r[0] for r in chunk_results])
    name_token_set_arr = np.concatenate([r[1] for r in chunk_results])
    name_token_sort_arr = np.concatenate([r[2] for r in chunk_results])
    address_jaro_arr = np.concatenate([r[3] for r in chunk_results])

    # 6. Attach computed string distance features as Polars Series
    joined = joined.with_columns([
        pl.Series("name_jaro", name_jaro_arr, dtype=pl.Float32),
        pl.Series("name_token_set_ratio", name_token_set_arr, dtype=pl.Float32),
        pl.Series("name_token_sort_ratio", name_token_sort_arr, dtype=pl.Float32),
        pl.Series("address_jaro", address_jaro_arr, dtype=pl.Float32),
    ])

    # 7. Drop intermediate text columns to conserve memory
    keep_cols = [c for c in pairs_df.columns]
    for feat in FEATURE_COLUMNS:
        if feat not in keep_cols:
            keep_cols.append(feat)

    result_df = joined.select(keep_cols)
    return result_df


# ---------------------------------------------------------------------------
# Legacy Functions for Backward Compatibility
# ---------------------------------------------------------------------------
def extract_single_pair_features(
    s1_name: str,
    s1_addr: str,
    cand_name: str,
    cand_addr: str,
    rank: int,
    total_cands: int,
) -> List[float]:
    """Computes the 15 tabular features for a single candidate pair (legacy)."""
    n1 = normalize_text(s1_name)
    n2 = normalize_text(cand_name)

    token_sort = fuzz.token_sort_ratio(n1, n2)
    token_set = fuzz.token_set_ratio(n1, n2)
    partial_r = fuzz.partial_ratio(n1, n2)
    jw = distance.JaroWinkler.similarity(n1, n2) * 100.0

    exact_name = 1.0 if n1 and n1 == n2 else 0.0

    t1 = set(extract_clean_tokens(s1_name))
    t2 = set(extract_clean_tokens(cand_name))
    clean_name = 1.0 if t1 and t1 == t2 else 0.0

    l1, l2 = len(n1), len(n2)
    len_diff = float(abs(l1 - l2))
    len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

    dom2 = extract_domain_root(cand_name)
    has_dom = 1.0 if dom2 else 0.0
    dom_match = 1.0 if dom2 and dom2 in t1 else 0.0

    both_addr = 1.0 if s1_addr and cand_addr else 0.0

    s1_num = extract_street_number(s1_addr)
    cand_num = extract_street_number(cand_addr)
    if s1_num and cand_num:
        street_match = 1.0 if s1_num == cand_num else 0.0
    else:
        street_match = -1.0

    addr_jaccard = 0.0
    if s1_addr and cand_addr:
        a1_toks = set(normalize_text(s1_addr).split())
        a2_toks = set(normalize_text(cand_addr).split())
        if a1_toks and a2_toks:
            inter = len(a1_toks.intersection(a2_toks))
            addr_jaccard = inter / len(a1_toks.union(a2_toks))

    c_rank = float(rank)
    tot_cands = float(total_cands)

    return [
        token_sort, token_set, partial_r, jw,
        exact_name, clean_name, len_diff, len_ratio,
        has_dom, dom_match, both_addr, street_match,
        addr_jaccard, c_rank, tot_cands,
    ]


def build_pair_features_df(
    pairs_list: List[Tuple[str, str, int, int]],
    s1_meta: Dict[str, Tuple[str, str]],
    cand_meta: Dict[str, Tuple[str, str]],
    labels: Optional[Dict[Tuple[str, str], int]] = None,
) -> pl.DataFrame:
    """Vectorized construction of tabular features dataframe (legacy)."""
    feature_matrix = []
    y_labels = [] if labels is not None else None
    s1_out = []
    cand_out = []

    for s1_id, cand_id, rank, total_c in pairs_list:
        s1_n, s1_a = s1_meta.get(s1_id, ("", ""))
        c_n, c_a = cand_meta.get(cand_id, ("", ""))

        feats = extract_single_pair_features(s1_n, s1_a, c_n, c_a, rank, total_c)
        feature_matrix.append(feats)
        s1_out.append(s1_id)
        cand_out.append(cand_id)

        if labels is not None:
            y_labels.append(labels.get((s1_id, cand_id), 0))

    feature_arr = np.array(feature_matrix, dtype=np.float32)
    data_dict = {
        "s1_id": s1_out,
        "cand_id": cand_out,
    }
    for i, col_name in enumerate(FEATURE_NAMES):
        data_dict[col_name] = feature_arr[:, i]

    if y_labels is not None:
        data_dict["label"] = y_labels

    return pl.DataFrame(data_dict)


# ---------------------------------------------------------------------------
# Self-Test Demonstration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[*] Testing src/features.py build_pairwise_features...")
    sample_s1 = pl.DataFrame({
        "source1_entity_id_int": [1, 2],
        "name_clean": ["acme corporation", "global logistics services"],
        "address_clean": ["100 main st suite 200 new york ny", "50 harbour road singapore"],
        "legal_form": ["corp", "llc"],
        "domain_root": ["acmecorp", None],
        "address_street_number": ["100", "50"],
    })

    sample_cand = pl.DataFrame({
        "candidate_entity_id_int": [10, 11, 20],
        "name_clean": ["acme corp", "acme supplies inc", "global logistics co"],
        "address_clean": ["100 main street new york", "999 broadway", "50 harbour road"],
        "legal_form": ["corp", "inc", "co"],
        "domain_root": ["acmecorp", "acmesupplies", "globallogistics"],
        "address_street_number": ["100", "999", "50"],
    })

    sample_pairs = pl.DataFrame({
        "source1_entity_id_int": [1, 1, 2],
        "candidate_entity_id_int": [10, 11, 20],
        "target": [1, 0, 1],
    })

    feats = build_pairwise_features(sample_s1, sample_cand, sample_pairs, n_jobs=2)
    print("\n[+] Feature DataFrame Result:")
    print(feats)
    print("\nFeature Columns verification:", all(c in feats.columns for c in FEATURE_COLUMNS))
    print("[SUCCESS] src/features.py verified successfully!")
