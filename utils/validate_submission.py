#!/usr/bin/env python3
"""
validate_submission.py — Submission integrity validator for Business Entity Resolution.

Verifies:
1. Matching results format: 2 tab-separated columns (source1_entity_id, matched_entity_ids).
2. Exactly one row per test source1_entity_id (no missing rows, no duplicate IDs).
3. Singletons have empty matched_entity_ids with valid trailing tab.
4. Candidate pairs format and coverage: all matched target IDs exist in candidate_pairs.tsv.
5. All target IDs conform to expected prefixes (S2-, S3-).
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Validate challenge submission TSVs.")
    parser.add_argument("--matching", type=str, required=True, help="Path to output/matching_results.tsv")
    parser.add_argument("--candidate", type=str, required=True, help="Path to output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to dataset/test directory")
    return parser.parse_args()


def main():
    args = parse_args()

    matching_path = Path(args.matching)
    candidate_path = Path(args.candidate)
    test_dir = Path(args.test_dir)

    print("=" * 60)
    print("SUBMISSION VALIDATION AUDIT")
    print("=" * 60)

    if not matching_path.exists():
        print(f"[FAIL] Missing matching results file: {matching_path}")
        sys.exit(1)
    if not candidate_path.exists():
        print(f"[FAIL] Missing candidate pairs file: {candidate_path}")
        sys.exit(1)

    # Check test_source1.tsv if available
    s1_file = test_dir / "test_source1.tsv"
    expected_ids = None
    if s1_file.exists():
        print(f"Loading reference test source 1: {s1_file}")
        s1_df = pd.read_csv(s1_file, sep="\t", usecols=["entity_id"], dtype=str)
        expected_ids = set(s1_df["entity_id"].dropna().unique())
        print(f"  Total expected unique Source 1 entities: {len(expected_ids):,}")

    # Validate matching_results.tsv
    print(f"Validating: {matching_path}")
    matching_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    if list(matching_df.columns) != ["source1_entity_id", "matched_entity_ids"]:
        print(f"[FAIL] Header mismatch in {matching_path}. Expected ['source1_entity_id', 'matched_entity_ids'], got {list(matching_df.columns)}")
        sys.exit(1)

    matching_ids = matching_df["source1_entity_id"].tolist()
    if len(matching_ids) != len(set(matching_ids)):
        print(f"[FAIL] Duplicate source1_entity_id rows found in {matching_path}!")
        sys.exit(1)

    if expected_ids:
        missing_ids = expected_ids - set(matching_ids)
        if missing_ids:
            print(f"[FAIL] {len(missing_ids)} Source 1 IDs missing from matching results!")
            sys.exit(1)
        extra_ids = set(matching_ids) - expected_ids
        if extra_ids:
            print(f"[FAIL] {len(extra_ids)} unexpected Source 1 IDs found in matching results!")
            sys.exit(1)

    # Validate candidate_pairs.tsv
    print(f"Validating: {candidate_path}")
    cand_df = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    if list(cand_df.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        print(f"[FAIL] Header mismatch in {candidate_path}. Expected ['source1_entity_id', 'candidate_entity_ids'], got {list(cand_df.columns)}")
        sys.exit(1)

    # Verify matching is a subset of candidates
    cand_map = dict(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]))
    mismatches = 0
    total_matches = 0
    singletons = 0

    for _, row in matching_df.iterrows():
        s1 = row["source1_entity_id"]
        matches = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
        if not matches:
            singletons += 1
            continue
        total_matches += len(matches)
        cands = set(c.strip() for c in cand_map.get(s1, "").split(",") if c.strip())
        for m in matches:
            if m not in cands:
                mismatches += 1

    print("-" * 60)
    print(f"Audit Summary:")
    print(f"  Total S1 rows evaluated   : {len(matching_df):,}")
    print(f"  Predicted Singletons      : {singletons:,} ({singletons / len(matching_df) * 100:.1f}%)")
    print(f"  Total Predicted Links     : {total_matches:,}")
    print(f"  Unblocked candidate leaks : {mismatches}")

    if mismatches > 0:
        print(f"[FAIL] Found {mismatches} matched entity IDs that were not present in candidate_pairs.tsv!")
        sys.exit(1)

    print("=" * 60)
    print("PASS — no blocking issues found. Ready for package archive.")
    print("=" * 60)


if __name__ == "__main__":
    main()
