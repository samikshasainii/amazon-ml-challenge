# Amazon ML Challenge 2026: Business Entity Resolution
# Persistent Agent Operational Memory & State

## 1. Project Overview & Constraints
- Goal: Link Source 1 (S1) entities against targets in Source 2 (S2) and Source 3 (S3).
- Scale: 1.73M S1 test entities; ~10M targets in test S2/S3.
- Metric: Macro F_0.5 (Precision-heavy; false merges penalized 2x).
- Singletons: If S1 has no target matches, output an empty prediction list (yields 1.0 F_0.5).
- Constraints: 100% offline; strictly NO hosted LLM APIs, NO external gazetteers/postal databases.
- Open-Label Partitioning: Test set includes France (train had US, India). NEVER drop or ignore France.

## 2. Directory Layout & Key Artifacts
- `dataset/test/`: Contains test_source1.tsv, test_source2.tsv, test_source3.tsv.
- `output/candidate_pairs.tsv`: Candidate blocking output (S1 -> target pool).
- `output/matching_results.tsv`: Final predictions submitted to portal. Format: `source1_entity_id\tmatched_entity_ids\n`.
  * CRITICAL FORMATTING: Singletons MUST have a trailing tab (`S1-ID\t\n`).
- `utils/validate_submission.py`: Standard local validator. MUST return PASS before submitting.
- `code/business_entity_resolution/src/`: Core Python pipeline scripts.

## 3. Team Responsibilities
- Samiksha (Teammate 1): Candidate Generation & Blocking (`output/candidate_pairs.tsv`, target >= 88% recall).
- Verniya (Teammate 2): Feature Engineering & Tabular GBDT Matcher (`output/matching_results.tsv`, threshold tuning).
- Suhina (Teammate 3): Semantic Embeddings (Apache 2.0 / MIT models) & `Documentation_template.md`.

## 4. Current State & Immediate Progress
- [x] Initialized architecture, .gitignore, and local validator.
- [x] Resolved memory/runtime bottleneck (pivoted from 49h full matrix to out-of-core chunk streaming).
- [x] Generated first full test baseline via `fast_baseline.py`.
- [x] Pruned generic runaway matches down to 67MB.
- [x] Enforced strict 2-column TSV delimiter for all 1,732,544 rows (fixed singleton missing tabs).
- [x] Local validator passed cleanly (`PASS — no blocking issues found`).
- [ ] Submission 1 uploaded to portal; awaiting public score logging.

## 5. Next Planned Action (Stage 1.5 Blocking)
1. Log baseline score to `README.md`.
2. Upgrade `code/business_entity_resolution/src/` to multi-pass streaming candidate blocking:
   - Dynamic country partitioning.
   - Chunked inverted token indexing (cap candidates at <= 40 per entity).
   - In-block character 3-gram similarity.
3. Validate candidate recall on a 5k-record train holdout slice (aim >= 88%).
4. Export `output/candidate_pairs.tsv` for Teammate 2 handoff and generate Day 1 Submission 2.
'@ | Out-File -Encoding utf8 AGENTS.md
