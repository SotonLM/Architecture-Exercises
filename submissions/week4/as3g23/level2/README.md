# Level 2: Tiny Semantic Search ⭐⭐

*Tier 2 core project; demonstrates turning embeddings into a useful retrieval system.*

**Goal:** Build a miniature semantic search engine over a purposely diverse corpus, measure latency, and sanity-check retrieval quality.

**Task:**

1. Create a corpus of at least 20–30 short paragraphs that span 3–4 distinct topics (hard-code a Python list or load from a local `.json` / `.txt`).
2. Compute and store embeddings for every document. In-memory caching is fine; bonus points for saving `.npy` or `.json` files for reuse.
3. Accept a query string via `input()`.
4. Embed the query, compute cosine similarity against your corpus, and return the top-`k` (default `k=3`) results along with document IDs/indices, snippets, and similarity scores.
5. Log timings for:
   - Embedding the full corpus once.
   - Answering a single query.
6. Run at least **five** diverse queries (factual, vague, long, short, multi-topic) and note whether the retrieved snippets make sense.

**Submit:**

- `week4/level2/semantic_search.py` containing your search system.
- `week4/level2/search_examples.txt` with:
  - The 5 queries you tried.
  - Top-3 results (ID + snippet + score) for each query.
  - A 3–5 sentence reflection describing when retrievals shine vs fail.

**Implementation Notes:**

- Consider normalising embeddings once so cosine similarity is just a dot product.
- Use `time.perf_counter()` (or similar) for timing logs.
- If your corpus is static, precompute embeddings when the script starts and reuse them inside the query loop.
- Optional extras: add CLI arguments for `k`, dump embeddings to disk, or show a bar chart of similarity scores.

**If Level 2 works → you’re ready for Tier 2 submissions and Level 3 experiments.**
