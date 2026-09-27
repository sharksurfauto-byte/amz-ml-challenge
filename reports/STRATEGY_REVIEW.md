# Strategy Review for Amazon ML Challenge 2026

An analysis of the current pipeline (`src/blocking.py`, `src/features.py`, `src/normalize.py`, and `scripts/05_train_ranker.py`) highlights several critical areas for boosting the Macro F0.5 score.

## 1. Missing Phonetic Blocking
Currently, `src/normalize.py` effectively transliterates Devanagari text into Latin script and cleans punctuation/stop-words. Still, there is **no phonetic blocking mechanism**.
*   **Assessment:** Neither `Double Metaphone`, `Soundex`, nor Indian-specific phonetic algorithms (like `Indic NLP` tokenization) are used when building indexing keys across chunks in `src/blocking.py`.
*   **Recommendation:** Adding a Double Metaphone (or fuzzy phonetic encoding) to the generated inverted index would capture transcription errors that simple spelling token sorts (as seen in `token_sort_key` and TF-IDF blocking) miss.

## 2. Advanced String Distances and Semantic Embeddings
Upon reviewing `src/features.py`, the `FEATURE_COLUMNS` strictly consist of fast edit-distance metrics. 
*   **Assessment:** The features utilized are: `name_len_diff`, `name_jaro`, `name_token_set_ratio`, `name_token_sort_ratio`, `address_jaro`, plus boolean exact matches (`exact_street_num_match`, `domain_root_match`, `exact_legal_match`). 
*   **Missing Features:**
    *   **LLM / Semantic Embeddings:** Although sentence-transformers embeddings are supposedly available per `CLAUDE.md`, they are not materialized inside `features.py`. There are no embedding cosine similarities computed between Names/Addresses in the existing tabular feature extractor.
    *   **Advanced Distances:** `Smith-Waterman` (local alignment) or similar advanced text alignments are missing. Time-permitting, RapidFuzz allows `fuzz.partial_ratio_alignment` which behaves somewhat like local alignment but isn't explicitly used as a core feature.

## 3. Discovered Constraint: Bipartite Greedy Mutual Exclusion
*   **Context:** By definition, `S1` entities are deduplicated references. The ground truth demands each `S2/S3` noisy record must map to **at most one** `S1` entity.
*   **Assessment:** Graph-level constraint satisfaction (e.g., connected components or greedy bipartite matching) is entirely absent from the prediction codebase.
*   **Impact:** Without this, the model severely punishes *Precision* because two different `S1` references predicting the exact same `S2` candidate will create false positives. Since Precision is weighted heavily ($2 \times$ Recall in Macro F0.5), enforcing mutual exclusion mathematically eliminates duplicate assignments and strictly boosts precision scores.

## 4. Does `05_train_ranker.py` Enforce Mutual Exclusion?
*   **Assessment:** **No.** `scripts/05_train_ranker.py` operates on a completely independent row-wise evaluation.
*   **Evidence:** In the `find_optimal_threshold()` function from `scripts/05_train_ranker.py`, the final assignment loop simply aggregates all candidate predictions exceeding the scalar `thresh`:
    ```python
    for s1, c, p in zip(s1_list, cand_list, dev_probs):
        if p >= thresh and s1 in pred_map:
            pred_map[s1].add(c)
    ```
    This means if `S2-X` is a candidate for `S1-A` and `S1-B` (and both scores pass the threshold), `S2-X` is added to both clusters.
*   **Corrective Action:** The pipeline must globally sort predictions by probability and use a visited set for candidates (`S2`, `S3`) to ensure they are strictly assigned once to the absolute highest-scoring `S1` entity (`greedy mutual exclusion`). Alternatively, a Maximum Weight Bipartite Matching algorithm can be evaluated if greedy sorting is sub-optimal.
