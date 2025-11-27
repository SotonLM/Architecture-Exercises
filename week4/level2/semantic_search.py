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
    "The capital of France",
    "Paris has a very large population",
    "Teriyaki sauce is a popular Japanese condiment",
    "Sushi often includes raw fish and rice",
    "A common caliber for handguns is 9mm",
    "Man landed on the moon in 1969",
    "The Great Wall of China is visible from space",
    "Mount Everest is the highest mountain on Earth",
    "The theory of relativity was developed by Einstein",
    "apple apple apple apple apple apple apple apple",
]

search_queries = [
    "Where is the Eiffel Tower located?",
    "What ingredients are used in sushi?",
    "Who was the first person to walk on the moon?",
    "Explain Einstein's contributions to physics.",
    "What is a common type of handgun ammunition?",
]

with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus_sentences = f.read().split(".")
    corpus_sentences = [s.strip() for s in corpus_sentences if s.strip()]
    

def main() -> None:
    print("=== LEVEL 2: Semantic seaching ===")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    start = time.perf_counter()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    print(f"Encoded {len(sentences)} sentences in {time.perf_counter() - start:.2f}s")

    # Cosine similarity matrix (vectors are L2-normalised, so dot product == cosine)
    similarity = embeddings @ embeddings.T

    with open("nearest_neighbours.txt", "w", encoding="utf-8") as f:
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



if __name__ == "__main__":
    main()
