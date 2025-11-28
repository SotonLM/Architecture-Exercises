# Level 1: Get Embeddings Working ⭐

*Tier 1 baseline; verifies you can generate and reason about vector similarity.*

**Start here:** run `python week4/diagnostic_task.py` without changing it. That script shows the exact behaviour you need (load model → encode sentences → cosine similarity → top neighbours). Once you’ve seen it work, create your own file and customise it.

**Goal:** Produce sentence embeddings, measure cosine similarity, and observe semantic vs lexical behaviour.

**Task:**

1. Copy the diagnostic logic into a new file named `week4/level1/embeddings_basic.py`.
2. Swap in your own 8–10 short sentences that mix clearly related and clearly unrelated ideas.
3. Use `SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")` (already installed once you run `uv pip install sentence-transformers`).
4. Encode the sentences with `normalize_embeddings=True`, compute cosine via `emb @ emb.T`, and list the top-3 neighbours per sentence (skip itself).
5. Highlight at least two cases in your output/logs:
   - Lexically different but semantically close.
   - Lexically similar but semantically far apart.

**Submit:**

- `week4/level1/embeddings_basic.py` (or notebook) containing your customised code.
- `week4/level1/nearest_neighbours.txt` with:
  - The source sentences.
  - Top-3 neighbours + cosine similarity per sentence.
  - A short note identifying the two contrasting cases above.

**Hints:**

- Stick to CPU-friendly models; this level should run in <5 seconds.
- Log timings if you can—you will reuse them later.
- Feel free to hand-code cosine (`emb @ emb.T`) instead of pulling in extra libraries.

**If Level 1 works → you’re cleared for Tier 1 submissions.**
