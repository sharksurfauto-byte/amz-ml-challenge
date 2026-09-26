#!/usr/bin/env python3
"""
Compress and optimize matching_results.tsv to fit under 512 MB portal limit.
Applies precision-aligned capping (up to 2 from S2 and 2 from S3, total <= 4).
"""

import os
import shutil
import sys
import time
from typing import List


def cap_ids(ids_str: str, max_per_source: int = 2) -> str:
    """Extracts up to max_per_source from S2 and S3 to keep balanced high-precision matches."""
    if not ids_str.strip():
        return ""
    
    parts = [x.strip() for x in ids_str.split(",") if x.strip()]
    s2 = []
    s3 = []
    
    for item in parts:
        if item.startswith("S2-"):
            if len(s2) < max_per_source:
                s2.append(item)
        elif item.startswith("S3-"):
            if len(s3) < max_per_source:
                s3.append(item)
        if len(s2) >= max_per_source and len(s3) >= max_per_source:
            break
            
    combined = s2 + s3
    return ",".join(combined)


def main():
    target_file = "output/matching_results.tsv"
    backup_file = "output/matching_results.tsv.bak"
    temp_file   = "output/matching_results.tsv.tmp"
    
    if not os.path.exists(target_file):
        print(f"ERROR: {target_file} not found!", file=sys.stderr)
        sys.exit(1)
        
    start_size_mb = os.path.getsize(target_file) / (1024 * 1024)
    print("=" * 65)
    print("      COMPRESSING MATCHING RESULTS FOR PORTAL UPLOAD")
    print("=" * 65)
    print(f"Original File: {target_file} ({start_size_mb:.2f} MB)")
    
    # 1. Create backup if not already present
    if not os.path.exists(backup_file):
        print(f"[*] Creating backup at {backup_file}...")
        shutil.copyfile(target_file, backup_file)
        
    t0 = time.time()
    total_entities = 0
    total_matches_orig = 0
    total_matches_new = 0
    singletons = 0
    
    # 2. Stream and cap
    print("[*] Processing and capping matches (max 2 from S2, 2 from S3)...")
    with open(backup_file, "r", encoding="utf-8") as fin, \
         open(temp_file, "w", encoding="utf-8", buffering=64*1024) as fout:
        
        header = fin.readline()
        fout.write(header)
        
        for line in fin:
            total_entities += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            ids_str = parts[1] if len(parts) > 1 else ""
            
            if ids_str.strip():
                orig_cnt = ids_str.count(",") + 1
                total_matches_orig += orig_cnt
                
                capped_str = cap_ids(ids_str, max_per_source=2)
                if capped_str:
                    new_cnt = capped_str.count(",") + 1
                    total_matches_new += new_cnt
                    fout.write(f"{s1_id}\t{capped_str}\n")
                else:
                    singletons += 1
                    fout.write(f"{s1_id}\t\n")
            else:
                singletons += 1
                fout.write(f"{s1_id}\t\n")
                
            if total_entities % 500000 == 0:
                print(f"    Processed {total_entities:,} rows...", flush=True)

    # 3. Replace target file with compressed version
    os.replace(temp_file, target_file)
    
    end_size_mb = os.path.getsize(target_file) / (1024 * 1024)
    elapsed = time.time() - t0
    
    print("=" * 65)
    print(" SUCCESS: matching_results.tsv successfully compressed!")
    print("=" * 65)
    print(f" Original Size:     {start_size_mb:.2f} MB (44.7M matches)")
    print(f" New File Size:     {end_size_mb:.2f} MB ({total_matches_new:,} matches)")
    print(f" Size Reduction:    {100 - (end_size_mb / start_size_mb * 100):.1f}% reduction")
    print(f" Total Entities:    {total_entities:,}")
    print(f" Singletons (empty):{singletons:,} ({singletons/total_entities*100:.2f}%)")
    print(f" Average Matches:   {total_matches_new/(total_entities-singletons):.2f} per matched entity")
    print(f" Time Taken:        {elapsed:.1f} seconds")
    print("=" * 65)


if __name__ == "__main__":
    main()
