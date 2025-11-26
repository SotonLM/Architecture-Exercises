"""
Week 4 Diagnostic Task — Embeddings & Retrieval

LEVEL 1  → Turn sentences into vectors, measure cosine similarity, print nearest neighbours.
LEVEL 2  → Build `semantic_search.py` (20-30 docs + interactive queries).
LEVEL 3  → Pick ONE: quality metric (`quality_results.txt`) OR tiny RAG loop (`rag_comparison.txt`).
LEVEL 4  → Freestyle retrieval-based tool with sample outputs + "what I'd improve next".

Most students should stop after Level 2. Levels 3-4 are optional Tier 3 stretch goals.
"""

import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "sentence-transformers is not installed.\n"
        "Run: uv pip install sentence-transformers"
    )

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

def clean_text(unclean_text):
    clean_text = " ".join(line.strip() for line in unclean_text)
    clean_text = clean_text.split("  ")
    # print(clean_text)
    return clean_text

def main() -> None:

    file = open("semantics.txt", "r", encoding="utf-8")
    text_lines = file.readlines()
    file.close()


    cleaned = clean_text(text_lines)

    # print("=== LEVEL 1: EMBEDDINGS + NEAREST NEIGHBOURS ===")
    # print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    start = time.perf_counter()
    corpus_embeddings = model.encode(cleaned, normalize_embeddings=True)

    print(f"Encoded {len(cleaned)} paragraphs in {time.perf_counter() - start:.2f}s")

    user_query = str(input("Enter query: "))
    query_embedding = model.encode(user_query, normalize_embeddings=True)

    similarity = corpus_embeddings @ query_embedding
    sim_scores =[]
    for idx in range(len(cleaned)):
        sim_scores.append((idx, similarity[idx]))

    # get the topk results with their correspondiong paragraph
    top_matches = sorted(sim_scores, key=lambda item: item[1], reverse=True)[:TOP_K]
    print(f"\n{user_query} ")
    for rank, (indx, score) in enumerate(top_matches):
        print(f" #{rank+1}  cosine={score:.3f}  →  [{indx}] {cleaned[indx][:70]}")    

# === SPOTLIGHT EXAMPLES ===
# Semantically similar but different words -> sentences 0 & 1
#   [0] Neural networks learn by adjusting billions of parameters.
#   [1] Backpropagation updates weights to minimise loss in a model.
# Lexically similar but different meaning -> sentences 2 & 3
#   [2] My grandma's apple pie relies on butter, cinnamon, and patience.
#   [3] Apple just announced a new chip for thin-and-light laptops.

# === NEXT STEPS ===
# Level 2 → Build semantic_search.py (corpus embeddings + input() queries + timing logs).
# Level 3 → Pick ONE: retrieval metric or RAG-lite (context + question → generator).
# Level 4 → Freestyle retrieval-based tool with sample outputs + improvement notes.
# See week4/level*/README.md for exact deliverables.


if __name__ == "__main__":
    main()
