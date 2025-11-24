# Level 3: Quality Metric or RAG-Lite ⭐⭐⭐

*Tier 3 advanced level; pick **one** path and probe how retrieval quality impacts downstream answers.*

Choose **Option A** (design a retrieval quality metric) **or** **Option B** (build a tiny RAG-lite loop). Document which path you took in your submission.

---

## Option A: Retrieval Quality Metric

**Goal:** Quantify how “good” your Level 2 retrieval results are for different query types.

**Task:**

1. Define what “good retrieval” means for your corpus (e.g., topic alignment, important keyword overlap, labels you generate manually).
2. Implement a metric that accepts a query plus its top-`k` docs and returns a score from 0–100.
3. Mix heuristics such as:
   - Overlap in important content words (after stopword removal).
   - Penalty if retrieved docs all belong to an off-topic cluster.
   - Bonus if query terms appear in titles/headings.
4. Collect at least **10** queries. For each:
   - Run your Level 2 search.
   - Compute the metric score.
   - Manually label the retrieval as **Good / OK / Bad**.
5. Compare how well your metric aligns with manual labels and write 1–2 paragraphs reflecting on:
   - When it succeeds.
   - When it fails or misses nuanced behaviour.

**Submit:**

- `week4/level3/quality_metric.py` containing the metric + evaluation loop.
- `week4/level3/quality_results.txt` showing each query, retrieved docs, metric score, manual judgment, and your reflection.

---

## Option B: RAG-Lite (Retrieval + Generation)

**Goal:** Show how retrieval conditioning changes answer quality relative to naked generation.

**Task:**

1. Reuse your Level 2 search to fetch top-`k` docs per query.
2. Load a lightweight generator (e.g., `distilgpt2`, `google/flan-t5-small`).
3. For each query, build a prompt of the form:

   ```text
   Context:
   [DOC 1]
   [DOC 2]
   ...
   Question: [QUERY]
   Answer:
   ```

4. Generate two answers per query:
   - **Without context:** feed only the question.
   - **With context (RAG-lite):** feed the prompt above.
5. Evaluate at least **10** queries. For each, note which answer is better and why (hallucinations reduced, specificity increased, etc.).

**Submit:**

- `week4/level3/rag_lite.py` containing the retrieval + generation workflow.
- `week4/level3/rag_comparison.txt` with the query, both answers, and your qualitative judgment.

**If Level 3 works → you’ve met Tier 3 expectations and can tackle Level 4 freestyle builds.**
