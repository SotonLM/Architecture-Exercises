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

sentences = [
    "The cat slept peacefully on the sunny windowsill.",
    "The kitten napped in the warm sunlight.",
    "I brewed a cup of coffee before starting work.",
    "The Eiffel Tower lights up beautifully at night.",
    "Baking bread fills the house with a warm aroma.",
    "Astronauts train for years to go to space.",
    "He swung the baseball bat with all his strength.",
    "The children played near the river bank.",
    "A bat flew out of the cave at dusk.",
    "I deposited my paycheck at the bank.",
]

def main() -> None:
    output_file = "/Users/yannis/Desktop/Ai proj/Architecture-Exercises/week4/level1/nearest_neighbours.txt"
    with open(output_file, "w") as f:

        f.write("Source Sentences: \n")
        for i in sentences:
            f.write(f"{i}\n")

        f.write("\n")

        f.write("=== LEVEL 1: EMBEDDINGS + NEAREST NEIGHBOURS ===\n")
        f.write(f"Loading model: {MODEL_NAME}\n")
        model = SentenceTransformer(MODEL_NAME)

        start = time.perf_counter()
        embeddings = model.encode(sentences, normalize_embeddings=True)
        print(f"Encoded {len(sentences)} sentences in {time.perf_counter() - start:.2f}s")

        # Cosine similarity matrix (vectors are L2-normalised, so dot product == cosine)
        similarity = embeddings @ embeddings.T

        for idx, sentence in enumerate(sentences):
            row = similarity[idx]
            others = [
                (other_idx, score)
                for other_idx, score in enumerate(row)
                if other_idx != idx
            ]
            top_matches = sorted(others, key=lambda item: item[1], reverse=True)[:TOP_K]

            f.write(f"\nSentence [{idx}]: {sentence}\n")
            for rank, (match_idx, score) in enumerate(top_matches, start=1):
                f.write(f"  #{rank}  cosine={score:.3f}  →  [{match_idx}] {sentences[match_idx]}\n")

        f.write("\n")

        f.write("# === HIGHLIGHT===\n")
        f.write("# Semantically similar but different words -> sentences 0 & 1\n")
        f.write("#   [0] The cat slept peacefully on the sunny windowsill.\n")
        f.write("#   [1] The kitten napped in the warm sunlight.\n")
        f.write("# Lexically similar but different meaning -> sentences 8 & 10\n")
        f.write("#   [8] The children played near the river bank.\n")
        f.write("#   [10] I deposited my paycheck at the bank.\n")

# === NEXT STEPS ===
# Level 2 → Build semantic_search.py (corpus embeddings + input() queries + timing logs).
# Level 3 → Pick ONE: retrieval metric or RAG-lite (context + question → generator).
# Level 4 → Freestyle retrieval-based tool with sample outputs + improvement notes.
# See week4/level*/README.md for exact deliverables.

if __name__ == "__main__":
    main()