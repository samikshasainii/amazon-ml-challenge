#!/usr/bin/env python3
"""
fast_baseline.py — Day 1 baseline for Business Entity Resolution.

Approach
--------
Pure streaming, exact-match blocking on (normalized country, normalized
business_name). No TF-IDF, no full-corpus in-memory matrix, no sklearn.
Source 2 and Source 3 are streamed in fixed-size chunks into a single
Python dict:

    (country, clean_name) -> [target_entity_id, ...]   (capped length)

Source 1 is then streamed in the same chunk size, each row's key is looked
up in that dict, and results are appended to the two output TSVs
incrementally, so peak memory stays roughly flat regardless of whether the
target side has 50 rows or 10,000,000.

Honest limitation (by design, for Day 1)
-----------------------------------------
This script does NOT run a Stage 2 precision model. `matched_entity_ids`
is currently set equal to `candidate_entity_ids` for every row. Exact
match on (country, cleaned name) is a reasonably high-precision signal,
but it is not calibrated against Macro F_0.5, and it will produce false
positives wherever two distinct businesses share a normalized name in the
same country (holding companies, common franchise names, stripped-down
generic names, etc.). That is expected and acceptable for a Day 1
submission whose purpose is to get a correctly-formatted, non-trivial
baseline on the board — it is the first thing to replace with a real
Stage 2 scorer (RapidFuzz / Levenshtein / Jaccard on tokens, a trained
gradient-boosted classifier, etc.) before you rely on this for score.

Usage
-----
    python code/business_entity_resolution/src/fast_baseline.py

Run from the repository root so the default relative paths
(dataset/test, output/, utils/validate_submission.py) resolve correctly.
All paths are also overridable via CLI flags — see --help.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Config defaults
# --------------------------------------------------------------------------

DEFAULT_CHUNKSIZE = 250_000
DEFAULT_MAX_CANDIDATES_PER_KEY = 50

# Exact suffix set requested for Day 1. Matched as whole tokens (after
# punctuation is stripped), case-insensitively, anywhere in the name -
# not just at the end - so "pvt ltd", "ltd", "co", etc. are all removed
# regardless of position. This intentionally does NOT try to catch every
# spelled-out or punctuated variant (e.g. "S.A.S", "Corporation",
# "Limited") - that is a Day 2 normalization improvement, not this list.
LEGAL_SUFFIXES = {"corp", "ltd", "inc", "pvt", "co", "llc", "sa", "sarl"}

# ASCII punctuation to strip, collapsed to a single space. Deliberately
# NOT implemented as "drop anything that isn't \w": Python's \w only
# counts Unicode categories L/N/_ as word characters and excludes
# combining marks (category M) - so on Devanagari/other Indic-script
# names, \w-stripping silently deletes vowel signs (matras) and corrupts
# the name (e.g. "मॉडर्न" -> "म डर न"). Stripping a fixed, known set of
# ASCII punctuation instead leaves every script's letters (and any
# combining marks) untouched, which is what we actually want here since
# the punctuation in this dataset (commas, periods, parens, quotes,
# angle brackets, ampersands, slashes) is all ASCII.
_PUNCT_RE = re.compile(r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]+""")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def normalize_name(value) -> str:
    """Lowercase, strip punctuation, drop legal-suffix tokens.

    Returns "" for missing/blank/unusable names. An empty result is never
    used as a blocking key (see build_index / lookup_candidates) - keying
    on ("US", "") would create one gigantic, useless bucket.
    """
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    s = _PUNCT_RE.sub(" ", s)
    tokens = [t for t in _WS_RE.split(s) if t and t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_country(value) -> str:
    """Uppercase + strip. No hardcoded country list/allowlist."""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


# --------------------------------------------------------------------------
# Stage 1a: build the blocking index from Source 2 + Source 3
# --------------------------------------------------------------------------

def build_index(
    paths: list[Path],
    chunksize: int,
    max_per_key: int,
) -> dict[tuple[str, str], list[str]]:
    """Stream target files, return {(country, clean_name): [entity_id, ...]}.

    Buckets are capped at `max_per_key` entries: once a key has that many
    IDs, further rows with the same key are skipped (not counted, not
    appended). This bounds the size of any single candidate list even if
    a normalized name is extremely common (e.g. a generic name that
    collapses to almost nothing after suffix-stripping).
    """
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    capped_keys: set[tuple[str, str]] = set()
    total_rows = 0
    skipped_blank_name = 0

    for path in paths:
        if not path.exists():
            print(f"  [warn] target file not found, skipping: {path}")
            continue

        t0 = time.time()
        rows_this_file = 0
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize):
            names = chunk["business_name"].map(normalize_name)
            countries = chunk["country"].map(normalize_country)
            ids = chunk["entity_id"]

            for entity_id, country, name in zip(ids, countries, names):
                if not name:
                    skipped_blank_name += 1
                    continue
                key = (country, name)
                if key in capped_keys:
                    continue
                bucket = index[key]
                bucket.append(entity_id)
                if len(bucket) >= max_per_key:
                    capped_keys.add(key)

            rows_this_file += len(chunk)
            total_rows += len(chunk)
            print(f"  [index] {path.name}: {rows_this_file:,} rows processed...")

        print(f"  [index] finished {path.name} ({rows_this_file:,} rows, "
              f"{time.time() - t0:.1f}s)")

    print(f"  [index] total target rows indexed: {total_rows:,} "
          f"(skipped {skipped_blank_name:,} with blank normalized name)")
    print(f"  [index] distinct (country, name) keys: {len(index):,} "
          f"({len(capped_keys):,} capped at {max_per_key})")
    return index


# --------------------------------------------------------------------------
# Stage 1b + 2 (Day-1 stand-in): look up candidates for each Source 1 row
# --------------------------------------------------------------------------

def process_source1(
    source1_path: Path,
    index: dict[tuple[str, str], list[str]],
    chunksize: int,
    candidate_path: Path,
    matching_path: Path,
) -> tuple[int, int]:
    """Stream Source 1, look up exact-match candidates, write both TSVs.

    Row order in the outputs matches the row order of `source1_path`
    because chunks are read and written in file order with no reordering
    or grouping in between.
    """
    n_total = 0
    n_with_matches = 0
    first_chunk = True

    for chunk in pd.read_csv(source1_path, sep="\t", dtype=str, chunksize=chunksize):
        names = chunk["business_name"].map(normalize_name)
        countries = chunk["country"].map(normalize_country)

        candidate_strs = []
        for country, name in zip(countries, names):
            if name:
                candidates = index.get((country, name), ())
            else:
                candidates = ()
            candidate_strs.append(",".join(candidates))
            if candidates:
                n_with_matches += 1

        # Day-1 baseline: matched_entity_ids == candidate_entity_ids.
        # This trivially satisfies the subset invariant
        # (matched ⊆ candidate) required by rules.md.
        out_candidates = pd.DataFrame({
            "source1_entity_id": chunk["entity_id"],
            "candidate_entity_ids": candidate_strs,
        })
        out_matching = pd.DataFrame({
            "source1_entity_id": chunk["entity_id"],
            "matched_entity_ids": candidate_strs,
        })

        mode = "w" if first_chunk else "a"
        header = first_chunk
        out_candidates.to_csv(candidate_path, sep="\t", index=False,
                               mode=mode, header=header, encoding="utf-8")
        out_matching.to_csv(matching_path, sep="\t", index=False,
                             mode=mode, header=header, encoding="utf-8")
        first_chunk = False

        n_total += len(chunk)
        print(f"  [match] {n_total:,} Source-1 entities processed "
              f"({n_with_matches:,} with >=1 candidate so far)...")

    if first_chunk:
        # source1_path was empty (header only, or truly empty) - still
        # need valid, correctly-headered output files.
        pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(
            candidate_path, sep="\t", index=False)
        pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
            matching_path, sep="\t", index=False)

    return n_total, n_with_matches


# --------------------------------------------------------------------------
# CLI / orchestration
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 1 fast baseline: streaming exact-match blocking "
                    "for Business Entity Resolution.",
    )
    p.add_argument("--dataset-dir", type=Path, default=Path("dataset"),
                    help="Root dataset dir (default: %(default)s)")
    p.add_argument("--test-dir", type=Path, default=None,
                    help="Dir with test_source{1,2,3}.tsv "
                         "(default: <dataset-dir>/test)")
    p.add_argument("--output-dir", type=Path, default=Path("output"),
                    help="Where to write the two output TSVs "
                         "(default: %(default)s)")
    p.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
                    help="Rows per streaming chunk (default: %(default)s)")
    p.add_argument("--max-candidates-per-key", type=int,
                    default=DEFAULT_MAX_CANDIDATES_PER_KEY,
                    help="Cap on candidates per (country, name) blocking "
                         "key (default: %(default)s)")
    p.add_argument("--validate-script", type=Path,
                    default=Path("utils/validate_submission.py"),
                    help="Path to the validator to auto-run afterward "
                         "(default: %(default)s)")
    p.add_argument("--skip-validate", action="store_true",
                    help="Don't auto-run the validation script")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    test_dir = args.test_dir or (args.dataset_dir / "test")
    source1_path = test_dir / "test_source1.tsv"
    source2_path = test_dir / "test_source2.tsv"
    source3_path = test_dir / "test_source3.tsv"

    for required in (source1_path,):
        if not required.exists():
            print(f"ERROR: required file not found: {required}", file=sys.stderr)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    matching_path = args.output_dir / "matching_results.tsv"

    t_start = time.time()

    print("=== Stage 1a: building target index (Source 2 + Source 3) ===")
    index = build_index(
        [source2_path, source3_path],
        chunksize=args.chunksize,
        max_per_key=args.max_candidates_per_key,
    )

    print()
    print("=== Stage 1b/2 (Day-1 stand-in): matching Source 1 ===")
    n_total, n_with_matches = process_source1(
        source1_path, index, args.chunksize, candidate_path, matching_path,
    )

    elapsed = time.time() - t_start
    print()
    print("=== Done ===")
    print(f"  Source-1 entities processed: {n_total:,}")
    if n_total:
        print(f"  With >=1 candidate:          {n_with_matches:,} "
              f"({100.0 * n_with_matches / n_total:.1f}%)")
        print(f"  Singletons (empty list):     {n_total - n_with_matches:,} "
              f"({100.0 * (n_total - n_with_matches) / n_total:.1f}%)")
    print(f"  Wrote: {candidate_path}")
    print(f"  Wrote: {matching_path}")
    print(f"  Elapsed: {elapsed:.1f}s")

    if args.skip_validate:
        return 0

    if not args.validate_script.exists():
        print()
        print(f"  [warn] validator not found at {args.validate_script}, "
              f"skipping auto-validation. Run it manually once available.")
        return 0

    print()
    print("=== Auto-running validate_submission.py --check-ids ===")
    cmd = [
        sys.executable, str(args.validate_script),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
        "--check-ids",
    ]
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())