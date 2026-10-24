# Problem Statement: Business Entity Resolution Challenge

## Overview
Business identity data arrives from multiple independent sources with noisy, inconsistent fields and no common identifiers. Build a machine learning solution that identifies which records across sources refer to the same real-world business entity.

- **Source 1 (`S1-`):** Deduplicated reference source.
- **Source 2 (`S2-`) & Source 3 (`S3-`):** Match candidate sources.
- **Task:** For each Source 1 entity, predict all matching records from Source 2 and Source 3. A Source 1 entity may match zero (singleton), one, or many target records.

---

## Evaluation Metric: Macro $F_{0.5}$
Submissions are evaluated on **Macro-averaged $F_{0.5}$** across all Source 1 entities in the evaluation set.

$$\beta = 0.5 \implies F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

### Critical Metric Mechanics:
1. **Precision-Heavy Penalty:** Merging two distinct businesses (false positive) is penalized 2× more heavily than missing a true link (false negative).
2. **Singletons:**
   - A Source 1 entity with no true matches scores **1.0** if an empty list `""` is predicted.
   - Predicting *any* false candidate for a true singleton immediately drops that entity's score to **0.0**.
   - If an entity's candidate scores fall below high-confidence thresholds, predict an empty match list.

---

## Dataset Structure & Path Conventions
All data resides in `dataset/` and must be parsed with tab-separation (`sep="\t"`).

- **Training:**
  - `dataset/train/train_source1.tsv`
  - `dataset/train/train_source2.tsv`
  - `dataset/train/train_source3.tsv`
  - `dataset/train/train_ground_truth.tsv` (columns: `source1_entity_id`, `matched_entity_ids`)
- **Test:**
  - `dataset/test/test_source1.tsv` (must generate matches for all IDs here)
  - `dataset/test/test_source2.tsv`
  - `dataset/test/test_source3.tsv`
  - No test ground truth is provided. Maintain an internal 20% Source 1 holdout split for validation.

---

## Schema & Domain Noise
Each source record contains:
- `entity_id`: `S1-xxxxx`, `S2-xxxxx`, or `S3-xxxxx` (determines source origin).
- `business_name`: Contains abbreviations (Corp/Corporation, Pvt/Private, Ltd), DBA names, typos, and transliterations.
- `business_address`: Contains landmark indicators (e.g., "Near SBI ATM"), missing postal codes, street abbreviations (Rd/St), and local permutations.
- `country`: Training data contains **US** and **India**. Test data introduces **France**. Country must remain an open categorical string variable (never hardcode filters to only US and India).