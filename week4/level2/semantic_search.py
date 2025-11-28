import time
import numpy as np

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

corpus_embeddings = None


def main() -> None:
    print("=== LEVEL 2: Semantic seaching ===")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    start = time.perf_counter()
    embeddings = model.encode(search_queries, normalize_embeddings=True)
    print(f"Encoded {len(search_queries)} sentences in {time.perf_counter() - start:.2f}s")

    try:
        corpus_embeddings = np.load("corpus_embeddings.npy")
        print(f"Loaded corpus embeddings from file.")
    except FileNotFoundError:
        start = time.perf_counter()
        corpus_embeddings = model.encode(corpus_sentences, normalize_embeddings=True)
        print(f"Encoded {len(corpus_sentences)} sentences in {time.perf_counter() - start:.2f}s")

        np.save("corpus_embeddings.npy", corpus_embeddings)

    # Cosine similarity matrix (vectors are L2-normalised, so dot product == cosine)
    similarity = embeddings @ corpus_embeddings.T#change this to be queries x corpus embeddings

    with open("search_examples.txt", "w", encoding="utf-8") as f:
        for idx, query in enumerate(search_queries):
            row = similarity[idx]
            results = [(c_idx, score) for c_idx, score in enumerate(row)]
            top_matches = sorted(results, key=lambda item: item[1], reverse=True)[:TOP_K]

            f.write(f"\nQuery [{idx}]: {query}\n")
            for rank, (match_idx, score) in enumerate(top_matches, start=1):
                f.write(f"  #{rank}  cosine={score:.3f}  →  [{match_idx}] {corpus_sentences[match_idx]}\n")



if __name__ == "__main__":
    main()
