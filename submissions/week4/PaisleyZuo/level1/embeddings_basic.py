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
    "I love walking on the beach at sunset.",
    "Neural networks learn patterns from data.",
    "My laptop loses its charge within just a few hours.",
    "Sunsets over the ocean make me calm.",
    "Machine learning models require a lot of computation.",
    "The lawyer decided to file a charge against the company.",
    "Cooking pasta always reminds me of home.",
    "Deep learning relies on gradient-based optimisation.",
]

def main() -> None:
    print("=== LEVEL 1: EMBEDDINGS + NEAREST NEIGHBOURS ===")
    print(f"Loading model: {MODEL_NAME}")
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

        print(f"\nSentence [{idx}]: {sentence}")
        for rank, (match_idx, score) in enumerate(top_matches, start=1):
            print(f"  #{rank}  cosine={score:.3f}  →  [{match_idx}] {sentences[match_idx]}")

    out_path = "nearest_neighbours.txt"
    with open(out_path, "w", encoding="utf-8") as f:

        f.write("=== Source Sentences ===\n")
        for i, s in enumerate(sentences):
            f.write(f"[{i}] {s}\n")

        f.write("\n=== Nearest Neighbours ===")
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
        
        # Required notes
        f.write("\n=== Notes ===\n")        
        # Semantically similar but lexically different
        f.write(
            "- Semantically similar but different words -> sentences 0 & 3\n"
            f"    [{0}] {sentences[0]}\n"
            f"    [{3}] {sentences[3]}\n"
        )

        # Lexically similar but semantically far apart
        f.write(
            "- Lexically similar but different meaning -> sentences 2 & 5\n"
            f"    [{2}] {sentences[2]}\n"
            f"    [{5}] {sentences[5]}\n"
        )

    print("Done! Check nearest_neighbours.txt")
# === SPOTLIGHT EXAMPLES ===
#   - The source sentences.
#   - Top-3 neighbours + cosine similarity per sentence.
#   - A short note identifying the two contrasting cases above.
# Semantically similar but different words -> sentences 0 & 1
#   [0] Neural networks learn by adjusting billions of parameters.
#   [1] Backpropagation updates weights to minimise loss in a model.
# Lexically similar but different meaning -> sentences 2 & 3
#   [2] My grandma's apple pie relies on butter, cinnamon, and patience.
#   [3] Apple just announced a new chip for thin-and-light laptops.

if __name__ == "__main__":
    main()