# Execution Rules & Constraints

## Model & License Restrictions
- **Architecture Limit:** Open-source models up to **8 Billion parameters** maximum.
- **Permitted Licenses:** Strictly **MIT** or **Apache 2.0**.
- **Fair Play & Anti-Cheating (Strictly Prohibited):**
  - No external lookup APIs or commercial ER systems.
  - No web scraping, government registration queries, or geocoding APIs.
  - Training and feature extraction must rely strictly on the provided dataset and self-contained models.

---

## Technical Pipeline Constraints
1. **Two-Stage Architecture:**
   - **Stage 1: Blocking / Candidate Generation:** Maximize candidate recall while keeping candidate size tractable. Outputs `output/candidate_pairs.tsv`.
   - **Stage 2: High-Precision Scoring / Matching:** Cross-encoders, TF-IDF cosine, Levenshtein/Jaccard string distances, and gradient-boosted trees. Narrow candidates to high-confidence matches. Outputs `output/matching_results.tsv`.
2. **Subset Invariant:**
   - $\text{matched\_entity\_ids} \subseteq \text{candidate\_entity\_ids}$
   - Every entity ID appearing in `matching_results.tsv` must exist in `candidate_pairs.tsv`.
3. **Target Constraints:**
   - `matched_entity_ids` and `candidate_entity_ids` must **only** contain Source 2 (`S2-`) or Source 3 (`S3-`) IDs.
   - Self-matches (`S1-` IDs) will trigger validation failure.

---

## Agent Code & Credit-Efficiency Constraints
- **Test on Samples First:** Never run full iterations on full test or train TSVs initially. Run functional smoke tests on `data/sample/` (10–50 rows).
- **Silent Standard Output:** Never print raw dataframes, full candidate lists, or entire file contents to stdout. Print only stage progress and validation metric summaries.
- **Explicit TSV Parsing:** Always specify `pd.read_csv(path, sep="\t")`.

## Strict Library & Data Guardrails
- **PROHIBITED:** Packages bundling external geographical, postal, or gazetteer databases (e.g., `libpostal`, `geopy`, `geocoders`, postal code tables), external lookup APIs, and web scraping[cite: 4].
- **ALLOWED:** Pure-algorithm libraries (`RapidFuzz`, `jellyfish`, `scikit-learn`, `LightGBM`, `XGBoost`, `pandas`), offline MIT/Apache 2.0 pretrained language/embedding models ($\le 8$B params), and custom hand-written normalization dictionaries[cite: 4, 8].

