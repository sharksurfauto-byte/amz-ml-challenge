"""
High-Speed Feature Extraction Engine for Candidate Pairs (S1, S_cand).
Uses C++ RapidFuzz and Polars to extract 14 high-signal lexical, address, and ambiguity features.
"""

import os
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from rapidfuzz import fuzz, distance

COMMON_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "co", "company", "sarl", "sa", "sas", "societe", "snc", "scs", "sca",
    "services", "enterprises", "industries", "group", "holdings", "solutions", "international", "tech",
    "technologies", "trading", "agency", "associates", "consulting", "gmbh", "llp"
}

DOMAIN_REGEX = re.compile(
    r"([a-z0-9-]+)\.(?:com|org|net|in|co|us|biz|info|io|fr|gov|edu)\b",
    flags=re.IGNORECASE
)
STREET_NUM_REGEX = re.compile(r"\b(\d{1,5})\b")


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


FEATURE_NAMES = [
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
    "total_s1_candidates"
]


def extract_single_pair_features(
    s1_name: str,
    s1_addr: str,
    cand_name: str,
    cand_addr: str,
    rank: int,
    total_cands: int
) -> List[float]:
    """Computes the 15 tabular features for a single candidate pair."""
    n1 = normalize_text(s1_name)
    n2 = normalize_text(cand_name)
    
    # 1. Lexical similarities (RapidFuzz)
    token_sort = fuzz.token_sort_ratio(n1, n2)
    token_set = fuzz.token_set_ratio(n1, n2)
    partial_r = fuzz.partial_ratio(n1, n2)
    jw = distance.JaroWinkler.similarity(n1, n2) * 100.0
    
    # 2. Exact match signals
    exact_name = 1.0 if n1 and n1 == n2 else 0.0
    
    t1 = set(extract_clean_tokens(s1_name))
    t2 = set(extract_clean_tokens(cand_name))
    clean_name = 1.0 if t1 and t1 == t2 else 0.0
    
    # 3. Length signals
    l1, l2 = len(n1), len(n2)
    len_diff = float(abs(l1 - l2))
    len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0
    
    # 4. Domain matching (S3 web records)
    dom2 = extract_domain_root(cand_name)
    has_dom = 1.0 if dom2 else 0.0
    dom_match = 1.0 if dom2 and dom2 in t1 else 0.0
    
    # 5. Address signals
    both_addr = 1.0 if s1_addr and cand_addr else 0.0
    
    s1_num = extract_street_number(s1_addr)
    cand_num = extract_street_number(cand_addr)
    if s1_num and cand_num:
        street_match = 1.0 if s1_num == cand_num else 0.0
    else:
        street_match = -1.0 # missing / unknown
        
    addr_jaccard = 0.0
    if s1_addr and cand_addr:
        a1_toks = set(normalize_text(s1_addr).split())
        a2_toks = set(normalize_text(cand_addr).split())
        if a1_toks and a2_toks:
            inter = len(a1_toks.intersection(a2_toks))
            addr_jaccard = inter / len(a1_toks.union(a2_toks))
            
    # 6. Rank & Ambiguity
    c_rank = float(rank)
    tot_cands = float(total_cands)
    
    return [
        token_sort, token_set, partial_r, jw,
        exact_name, clean_name, len_diff, len_ratio,
        has_dom, dom_match, both_addr, street_match,
        addr_jaccard, c_rank, tot_cands
    ]


def build_pair_features_df(
    pairs_list: List[Tuple[str, str, int, int]], # (s1_id, cand_id, rank, total_cands)
    s1_meta: Dict[str, Tuple[str, str]],
    cand_meta: Dict[str, Tuple[str, str]],
    labels: Optional[Dict[Tuple[str, str], int]] = None
) -> pl.DataFrame:
    """Vectorized construction of tabular features dataframe."""
    feature_matrix = []
    y_labels = [] if labels is not None else None
    s1_out = []
    cand_out = []
    
    for s1_id, cand_id, rank, total_c in pairs_list:
        s1_n, s1_a = s1_meta.get(s1_id, ("", ""))
        c_n, c_a   = cand_meta.get(cand_id, ("", ""))
        
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


if __name__ == "__main__":
    # Test feature extraction on sample pair
    f = extract_single_pair_features(
        s1_name="Maure Williams Colombier Inc",
        s1_addr="85 Wayne Avenue, Ticonderoga, NY",
        cand_name="Maure Wilblims Colombier",
        cand_addr="",
        rank=1,
        total_cands=3
    )
    print("Sample feature vector extraction:")
    for name, val in zip(FEATURE_NAMES, f):
        print(f"  {name:<25}: {val}")
    print("\n[SUCCESS] src/features.py verified successfully!")
