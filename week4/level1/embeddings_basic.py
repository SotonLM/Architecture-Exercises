import time
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

sentences = [
    "Apple grows on trees in orchards around the world.",
    "Apple pies are desert made with apples and cinnamon.",
    "Ice cream is cpld so it's good for the summer.",
    "Summer is hot so people like to eat ice cream.",
    "Marathon runners eat lots of pasta.",
    "I go for a jog every Wednesday.",
    "Museums display ancient or cool things.",
    "Budget airlines trade uncomfortable for cheap tickets.",
]

def main() -> None:
    print("=== LEVEL 1: EMBEDDINGS + NEAREST NEIGHBOURS ===")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    start = time.perf_counter()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    print(f"Encoded {len(sentences)} sentences in {time.perf_counter() - start:.2f}s")

    # Cosine similarity matrix (dot product works because vectors are normalised)
    similarity = embeddings @ embeddings.T

    for idx, sentence in enumerate(sentences):
        row = similarity[idx]

        others = [
            (j, score)
            for j, score in enumerate(row)
            if j != idx
        ]

        top_matches = sorted(
            others,
            key=lambda item: item[1],
            reverse=True
        )[:TOP_K]

        print(f"\nSentence [{idx}]: {sentence}")
        for rank, (match_idx, score) in enumerate(top_matches, start=1):
            print(
                f"  #{rank}  cosine={score:.3f}  →  "
                f"[{match_idx}] {sentences[match_idx]}"
            )

if __name__ == "__main__":
    main()
