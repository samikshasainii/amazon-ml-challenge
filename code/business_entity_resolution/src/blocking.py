#!/usr/bin/env python3
"""
Stage 1: Sparse TF-IDF blocking (vectorized).

Per country partition:
  * Fit a word (1-2) + char_wb (3-4) TF-IDF vectorizer on S2+S3
    (sublinear_tf=True, min_df=2, no max_df / stop-word cap).
  * Score S1 against S2+S3 with sparse matrix multiplication (cosine =
    L2-normalized dot product) in BLAS/scipy, then keep top-K.

This replaces sequential inverted-index Python loops.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
RE_MULTI_SPACE = re.compile(r"\s+")

BUSINESS_ABBREVS = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "llc": "llc", "llp": "llp", "co": "company",
    "assoc": "associates", "intl": "international", "natl": "national",
    "govt": "government", "dept": "department", "svcs": "services",
    "svc": "service", "mfg": "manufacturing", "grp": "group",
    "tech": "technology", "mgmt": "management", "soln": "solutions",
    "solns": "solutions", "engr": "engineering", "engg": "engineering",
    "hosp": "hospital", "univ": "university", "acad": "academy",
    "assn": "association", "fdn": "foundation", "dev": "development",
    "sys": "systems", "ind": "industries", "enter": "enterprises",
    "ent": "enterprises", "prop": "properties", "consult": "consultants",
}

STREET_ABBREVS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "cir": "circle", "hwy": "highway", "pkwy": "parkway",
    "ste": "suite", "apt": "apartment", "bldg": "building",
}

COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "united states": "us",
    "united states of america": "us",
    "india": "india", "in": "india", "ind": "india",
    "france": "france", "fr": "france",
}


def normalize_text(text) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = RE_NON_ALNUM.sub(" ", text)
    return RE_MULTI_SPACE.sub(" ", text).strip()


def _expand_tokens(text: str, mapping: dict) -> str:
    if not text:
        return ""
    parts = text.split()
    return " ".join(mapping.get(t, t) for t in parts)


def record_text(name, addr) -> str:
    """Normalized name + address; abbreviations expanded, stop-words kept.

    IDF down-weights frequent tokens; we do not drop brand prefixes.
    """
    name_n = _expand_tokens(normalize_text(name), BUSINESS_ABBREVS)
    addr_n = _expand_tokens(normalize_text(addr), STREET_ABBREVS)
    if name_n and addr_n:
        return name_n + " " + addr_n
    return name_n or addr_n


def normalize_country(country) -> str:
    if country is None or (isinstance(country, float) and np.isnan(country)):
        return "unknown"
    c = str(country).lower().strip()
    if not c:
        return "unknown"
    return COUNTRY_ALIASES.get(c, c)


def make_vectorizer(n_docs: int) -> FeatureUnion:
    min_df = 2 if n_docs >= 40 else 1
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        stop_words=None,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=min_df,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
        lowercase=False,
        stop_words=None,
    )
    return FeatureUnion(
        [("word", word), ("char", char)],
        transformer_weights={"word": 1.0, "char": 1.15},
    )


def _as_csr_f32(mat) -> sparse.csr_matrix:
    if not sparse.issparse(mat):
        mat = sparse.csr_matrix(mat, dtype=np.float32)
    else:
        mat = mat.tocsr()
        if mat.dtype != np.float32:
            mat = mat.astype(np.float32)
    mat.sort_indices()
    mat.eliminate_zeros()
    return mat


def batched_topk(
    Q: sparse.csr_matrix,
    X: sparse.csr_matrix,
    k: int,
    q_batch: int = 2048,
    x_chunk: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine top-K via chunked sparse-sparse dots. Returns (indices, scores)."""
    n_q = Q.shape[0]
    n_x = X.shape[0]
    k = min(k, n_x) if n_x else 0
    if k == 0 or n_q == 0:
        return (
            np.full((n_q, 0), -1, dtype=np.int32),
            np.full((n_q, 0), -np.inf, dtype=np.float32),
        )

    all_idx = np.full((n_q, k), -1, dtype=np.int32)
    all_sc = np.full((n_q, k), -np.inf, dtype=np.float32)

    for q0 in range(0, n_q, q_batch):
        q1 = min(q0 + q_batch, n_q)
        Qb = Q[q0:q1]
        bsz = q1 - q0
        best_s = np.full((bsz, k), -np.inf, dtype=np.float32)
        best_i = np.full((bsz, k), -1, dtype=np.int32)

        for x0 in range(0, n_x, x_chunk):
            x1 = min(x0 + x_chunk, n_x)
            S = Qb @ X[x0:x1].T
            if sparse.issparse(S):
                S = S.toarray()
            S = np.asarray(S, dtype=np.float32)

            chunk_i = np.arange(x0, x1, dtype=np.int32)
            cand_s = np.concatenate([best_s, S], axis=1)
            cand_i = np.concatenate(
                [best_i, np.broadcast_to(chunk_i, S.shape)], axis=1
            )
            part = np.argpartition(cand_s, -k, axis=1)[:, -k:]
            best_s = np.take_along_axis(cand_s, part, axis=1)
            best_i = np.take_along_axis(cand_i, part, axis=1)

        order = np.argsort(-best_s, axis=1)
        all_idx[q0:q1] = np.take_along_axis(best_i, order, axis=1)
        all_sc[q0:q1] = np.take_along_axis(best_s, order, axis=1)

    return all_idx, all_sc


class CountryBlock:
    __slots__ = ("country", "ids", "X", "vectorizer")

    def __init__(self, country: str, ids: np.ndarray, X: sparse.csr_matrix, vectorizer):
        self.country = country
        self.ids = ids
        self.X = X
        self.vectorizer = vectorizer

    def query_texts(self, texts: list[str], k: int, min_score: float,
                    q_batch: int, x_chunk: int):
        Q = _as_csr_f32(self.vectorizer.transform(texts))
        idx, sc = batched_topk(Q, self.X, k, q_batch=q_batch, x_chunk=x_chunk)
        out_ids: list[list[str]] = []
        out_sc: list[list[float]] = []
        for i in range(len(texts)):
            row_ids: list[str] = []
            row_sc: list[float] = []
            for j in range(idx.shape[1]):
                score = float(sc[i, j])
                if idx[i, j] < 0 or score < min_score:
                    continue
                row_ids.append(str(self.ids[idx[i, j]]))
                row_sc.append(score)
            out_ids.append(row_ids)
            out_sc.append(row_sc)
        return out_ids, out_sc


def load_source(path: str) -> pd.DataFrame:
    print(f"  Loading {os.path.basename(path)}")
    t0 = time.time()
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str,
        keep_default_na=False,
    )
    print(f"    {len(df):,} rows in {time.time() - t0:.1f}s")
    return df


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["country_n"] = [normalize_country(c) for c in df["country"].tolist()]
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    df["text"] = [record_text(n, a) for n, a in zip(names, addrs)]
    df.drop(columns=["business_name", "business_address", "country"], inplace=True, errors="ignore")
    return df


def load_prepared_sources(paths: list[str]) -> pd.DataFrame:
    """Load S2/S3 in chunks, keep only id + country + blocking text."""
    parts = []
    total = 0
    t0 = time.time()
    for path in paths:
        print(f"  Loading+normalizing {os.path.basename(path)}")
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype=str,
            keep_default_na=False,
            chunksize=250_000,
        ):
            parts.append(prepare_frame(chunk))
            total += len(chunk)
            if total % 1_000_000 < 250_000:
                print(f"    {total:,} records ({total / max(time.time() - t0, 0.01):,.0f}/s)")
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"  Prepared {len(df):,} S2+S3 records in {time.time() - t0:.1f}s")
    return df


def fit_country_block(country: str, g: pd.DataFrame) -> CountryBlock:
    n = len(g)
    print(f"  Fitting TF-IDF [{country}] n={n:,} ...")
    t1 = time.time()
    vec = make_vectorizer(n)
    X = _as_csr_f32(vec.fit_transform(g["text"].tolist()))
    ids = g["entity_id"].to_numpy()
    nnz = int(X.nnz)
    print(
        f"    {country}: docs={n:,} feats={X.shape[1]:,} nnz={nnz:,} "
        f"avg_nnz={nnz / max(n, 1):.1f} ({time.time() - t1:.1f}s)"
    )
    return CountryBlock(country, ids, X, vec)


def merge_by_score(
    ids_a: list[str], sc_a: list[float],
    ids_b: list[str], sc_b: list[float],
    k: int,
) -> tuple[list[str], list[float]]:
    best: dict[str, float] = {}
    for i, s in zip(ids_a, sc_a):
        if s > best.get(i, -1.0):
            best[i] = s
    for i, s in zip(ids_b, sc_b):
        if s > best.get(i, -1.0):
            best[i] = s
    ranked = sorted(best.items(), key=lambda x: -x[1])[:k]
    return [p[0] for p in ranked], [p[1] for p in ranked]


def query_group(
    indexes: dict[str, CountryBlock],
    country: str,
    texts: list[str],
    k: int,
    min_score: float,
    q_batch: int,
    x_chunk: int,
) -> tuple[list[list[str]], list[list[float]]]:
    n = len(texts)
    ids_out = [[] for _ in range(n)]
    sc_out = [[] for _ in range(n)]

    block = indexes.get(country)
    if block is not None:
        ids_out, sc_out = block.query_texts(
            texts, k, min_score, q_batch, x_chunk
        )

    if country != "unknown" and "unknown" in indexes:
        extra_k = max(k // 4, 8)
        e_ids, e_sc = indexes["unknown"].query_texts(
            texts, extra_k, min_score, q_batch, x_chunk
        )
        for i in range(n):
            ids_out[i], sc_out[i] = merge_by_score(
                ids_out[i], sc_out[i], e_ids[i], e_sc[i], k
            )
    return ids_out, sc_out


def generate_for_s1(
    s1: pd.DataFrame,
    gallery: pd.DataFrame,
    k: int,
    min_score: float,
    q_batch: int,
    x_chunk: int,
) -> tuple[list[list[str]], list[list[float]]]:
    """Fit one country TF-IDF index at a time, query matching S1 rows, free RAM."""
    n = len(s1)
    all_ids: list[Optional[list[str]]] = [None] * n
    all_sc: list[Optional[list[float]]] = [None] * n

    s1_countries = s1["country_n"].to_numpy()
    s1_texts = s1["text"].tolist()
    order = np.arange(n)
    t0 = time.time()
    done = 0

    gallery_countries = list(pd.unique(gallery["country_n"]))
    s1_only = [c for c in pd.unique(s1_countries) if c not in set(gallery_countries)]
    countries = list(gallery_countries) + s1_only
    # Fit "unknown" first so other partitions can blend those candidates in.
    countries = sorted(countries, key=lambda c: 0 if c == "unknown" else 1)

    unknown_block = None
    for country in countries:
        g = gallery.loc[gallery["country_n"] == country]
        block = None
        if len(g):
            block = fit_country_block(country, g)
            if country == "unknown":
                unknown_block = block
        del g
        # Drop this partition from the gallery so peak RAM falls after each country.
        gallery = gallery.loc[gallery["country_n"] != country]
        gc.collect()

        mask = s1_countries == country
        idx = order[mask]
        if len(idx) == 0:
            if country != "unknown":
                del block
                gc.collect()
            continue

        sub_texts = [s1_texts[i] for i in idx]
        print(f"    Query [{country}] n={len(idx):,}")
        t1 = time.time()
        indexes = {}
        if block is not None:
            indexes[country] = block
        if unknown_block is not None and country != "unknown":
            indexes["unknown"] = unknown_block
        c_ids, c_sc = query_group(
            indexes, country, sub_texts, k, min_score, q_batch, x_chunk
        )
        rate = len(idx) / max(time.time() - t1, 1e-6)
        print(f"      {len(idx):,} in {time.time() - t1:.1f}s ({rate:,.0f}/s)")
        for local, global_i in enumerate(idx):
            all_ids[int(global_i)] = c_ids[local]
            all_sc[int(global_i)] = c_sc[local]
        done += len(idx)
        print(f"    Progress {done:,}/{n:,}  elapsed={time.time() - t0:.1f}s")

        if country != "unknown":
            del block, indexes
            gc.collect()

    return [x or [] for x in all_ids], [x or [] for x in all_sc]


def evaluate_blocking(s1_ids, cand_ids, gt_path):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt_map: dict[str, set] = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        val = row["matched_entity_ids"]
        gt_map[s1_id] = set(val.split(",")) if val.strip() else set()

    total = singletons = singleton_empty = 0
    total_true = total_found = perfect = zero = 0
    for s1_id, cands in zip(s1_ids, cand_ids):
        true_set = gt_map.get(s1_id)
        if true_set is None:
            continue
        total += 1
        cand_set = set(cands)
        if not true_set:
            singletons += 1
            if not cand_set:
                singleton_empty += 1
            continue
        found = true_set & cand_set
        total_true += len(true_set)
        total_found += len(found)
        if found == true_set:
            perfect += 1
        if not found:
            zero += 1

    recall = total_found / max(total_true, 1)
    ns = total - singletons
    avg_c = sum(len(v) for v in cand_ids) / max(len(cand_ids), 1)

    print(f"\n  +----------------------------------------------+")
    print(f"  |        BLOCKING EVALUATION RESULTS           |")
    print(f"  +----------------------------------------------+")
    print(f"  |  Entities evaluated     : {total:>10,}         |")
    print(f"  |  Singletons             : {singletons:>10,} ({100*singletons/max(total,1):4.1f}%)  |")
    print(f"  |  Non-singletons         : {ns:>10,}         |")
    print(f"  +----------------------------------------------+")
    print(f"  |  Candidate Recall       : {recall:>10.4f} ({100*recall:.1f}%)  |")
    print(f"  |  True matches           : {total_true:>10,}         |")
    print(f"  |  Found in candidates    : {total_found:>10,}         |")
    print(f"  |  Missed                 : {total_true - total_found:>10,}         |")
    print(f"  +----------------------------------------------+")
    print(f"  |  Perfect recall entities: {perfect:>10,} ({100*perfect/max(ns,1):4.1f}%)  |")
    print(f"  |  Zero recall entities   : {zero:>10,} ({100*zero/max(ns,1):4.1f}%)  |")
    print(f"  +----------------------------------------------+")
    print(f"  |  Avg candidates/entity  : {avg_c:>10.1f}         |")
    print(f"  +----------------------------------------------+")
    return {"recall": recall, "avg_candidates": avg_c}


def write_outputs(
    s1_ids,
    cand_ids,
    cand_scores,
    cand_path,
    match_path,
    match_top_k: int,
    match_min_cosine: float,
):
    os.makedirs(os.path.dirname(os.path.abspath(cand_path)), exist_ok=True)
    n_nonempty = 0
    n_cands = 0
    n_matches = 0
    with open(cand_path, "w", encoding="utf-8") as fc, \
         open(match_path, "w", encoding="utf-8") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, cands, scores in zip(s1_ids, cand_ids, cand_scores):
            fc.write(f"{sid}\t{','.join(cands)}\n")
            matched = [
                c for c, s in zip(cands, scores)
                if s >= match_min_cosine
            ][:match_top_k]
            fm.write(f"{sid}\t{','.join(matched)}\n")
            if cands:
                n_nonempty += 1
            n_cands += len(cands)
            n_matches += len(matched)
    n = len(s1_ids)
    print(f"  Wrote {cand_path}")
    print(f"  Wrote {match_path}")
    print(f"  Non-empty candidate rows: {n_nonempty:,}/{n:,} "
          f"({100 * n_nonempty / max(n, 1):.1f}%)")
    print(f"  Avg candidates: {n_cands / max(n, 1):.1f}  "
          f"avg matches: {n_matches / max(n, 1):.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 sparse TF-IDF blocking")
    p.add_argument("--data-dir", default="dataset/sample")
    p.add_argument("--prefix", default="test")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--holdout-size", type=int, default=5_000)
    p.add_argument("--gt-path", default=None)
    p.add_argument("--max-candidates", type=int, default=40)
    p.add_argument("--min-score", type=float, default=0.08)
    p.add_argument("--match-top-k", type=int, default=5)
    p.add_argument("--match-min-cosine", type=float, default=0.42)
    p.add_argument("--q-batch", type=int, default=2048)
    p.add_argument("--x-chunk", type=int, default=65536)
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    data_dir = args.data_dir
    prefix = args.prefix
    s1_path = os.path.join(data_dir, f"{prefix}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{prefix}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{prefix}_source3.tsv")

    print("=" * 60)
    print("[Stage 1] Sparse TF-IDF blocking (word 1-2 + char_wb 3-4)")
    print(f"  K={args.max_candidates}  min_score={args.min_score}  "
          f"match_min_cosine={args.match_min_cosine}")
    print("=" * 60)

    print(f"\n  Loading S1: {os.path.basename(s1_path)}")
    s1 = prepare_frame(load_source(s1_path))

    if args.evaluate:
        gt_path = args.gt_path or os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
        print(f"\n{'=' * 60}")
        print(f"[Stage 1] Evaluation holdout {args.holdout_size:,}")
        print(f"{'=' * 60}")
        gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        holdout = gt.sample(n=min(args.holdout_size, len(gt)), random_state=42)
        holdout_ids = set(holdout["source1_entity_id"].tolist())
        s1 = s1[s1["entity_id"].isin(holdout_ids)].reset_index(drop=True)
        print(f"  Holdout S1 rows in file: {len(s1):,}")
        del gt, holdout
        gc.collect()

    gallery = load_prepared_sources([s2_path, s3_path])

    print(f"\n{'=' * 60}")
    print(f"[Stage 1] Querying {len(s1):,} S1 records")
    print(f"{'=' * 60}")
    cand_ids, cand_sc = generate_for_s1(
        s1, gallery,
        k=args.max_candidates,
        min_score=args.min_score,
        q_batch=args.q_batch,
        x_chunk=args.x_chunk,
    )
    del gallery
    gc.collect()
    s1_ids = s1["entity_id"].tolist()

    if args.evaluate:
        gt_path = args.gt_path or os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
        evaluate_blocking(s1_ids, cand_ids, gt_path)

    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    write_outputs(
        s1_ids, cand_ids, cand_sc, cand_path, match_path,
        match_top_k=args.match_top_k,
        match_min_cosine=args.match_min_cosine,
    )

    print(f"\n[Stage 1] Total elapsed: {time.time() - t_start:.1f}s "
          f"({(time.time() - t_start) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
