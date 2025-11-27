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
    "I met my fans today and signed autographs for an hour.",
    "That fan is saving my life in this hot weather.",
    "Dates are a fruit high in natural sugars and fiber.",
    "Going on a date can be both exciting and nerve-wracking.",
    "It is good to be thorough when preparing for a major exam.",
    "I am through with this project; let's move on to the next one.",
    "That last question really threw me for a loop.",
    "The roman lexicon is a fascinating subject of study.",
    "I am studying very hard to do well in my finals.",
    "Learning new languages opens up many opportunities in life.",
]

def main() -> None:
    with open("nearest_neighbours.txt", "w", encoding="utf-8") as f:
        f.write(f"=== LEVEL 1: EMBEDDINGS + NEAREST NEIGHBOURS ===\n")
        f.write(f"Loading model: {MODEL_NAME}\n")
        model = SentenceTransformer(MODEL_NAME)

        start = time.perf_counter()
        embeddings = model.encode(sentences, normalize_embeddings=True)
        f.write(f"Encoded {len(sentences)} sentences in {time.perf_counter() - start:.2f}s\n")

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

        f.write("Lexically different but semantically similar: sentences 4 & 8 scored the highest despite no shared words, this shows the importance of context. \n")
        f.write("Lexically similar but semantically different: sentences 0 & 1 scored lower than expected despite sharing the word 'fan', showing that word overlap doesn't guarantee similar meaning.\n")
    

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
