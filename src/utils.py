"""
Utility functions for Amazon ML Challenge 2026.
Includes dataset auto-discovery (supporting both flat and nested directories across Kaggle & local environments).
"""

import os
from typing import Dict, List, Optional


TARGET_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv"
]


def resolve_dataset_files(custom_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Dynamically scans directories to locate all 7 dataset TSV files.
    Handles flat Kaggle datasets (/kaggle/input/datasets/darshantej/amazon-ml/...)
    as well as nested directories (data/raw/train/..., data/raw/test/...).

    Args:
        custom_dir: Optional directory to prioritize searching.

    Returns:
        dict mapping filename -> absolute or relative file path
    """
    search_roots = []
    if custom_dir:
        search_roots.append(custom_dir)
    search_roots.extend(["/kaggle/input", "../data/raw", "data/raw", "."])

    file_map: Dict[str, str] = {}
    for root_dir in search_roots:
        if os.path.exists(root_dir):
            for current_path, _, filenames in os.walk(root_dir):
                for f in TARGET_FILES:
                    if f in filenames and f not in file_map:
                        file_map[f] = os.path.join(current_path, f)

    return file_map


def get_dataset_file(filename: str, custom_dir: Optional[str] = None) -> str:
    """
    Retrieves the resolved path for a specific dataset file.
    Raises FileNotFoundError if not found.
    """
    files = resolve_dataset_files(custom_dir)
    if filename not in files:
        raise FileNotFoundError(
            f"Could not locate '{filename}'. Searched roots: {['/kaggle/input', 'data/raw', '.']}"
        )
    return files[filename]


if __name__ == "__main__":
    found = resolve_dataset_files()
    print("Resolved Dataset Files:")
    for k, v in sorted(found.items()):
        print(f"  {k:<25} -> {v}")
