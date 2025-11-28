
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

sentences = [ #changed sentences
    "The cat sat on the comfortable mat near the window.",
    "Felines often enjoy resting on soft surfaces in sunny spots.",
    "Python programming requires understanding basic syntax and data structures.",
    "Large snakes like pythons can grow over twenty feet in length.",
    "Baking bread requires flour, water, yeast, and patience.",
    "Cooking involves combining ingredients and applying heat to create food.",
    "Electric vehicles are becoming more popular due to environmental concerns.",
    "Tesla announced new battery technology for longer driving range.",
    "Reading books helps expand your knowledge and vocabulary.",
    "Libraries provide free access to books and digital resources for communities.",
]

def main() -> None:
    print("=== LEVEL 1: EMBEDDINGS + NEAREST NEIGHBOURS ===")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME) #Use SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    start = time.perf_counter()
    embeddings = model.encode(sentences, normalize_embeddings=True) #normalize_embeddings=True
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
        top_matches = sorted(others, key=lambda item: item[1], reverse=True)[:TOP_K] #list the top-3 neighbours per sentence (skip itself)

        print(f"\nSentence [{idx}]: {sentence}")
        for rank, (match_idx, score) in enumerate(top_matches, start=1):
            print(f"  #{rank}  cosine={score:.3f}  →  [{match_idx}] {sentences[match_idx]}")

if __name__ == "__main__":
    main()
