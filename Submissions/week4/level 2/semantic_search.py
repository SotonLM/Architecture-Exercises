import time
import json
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
NUMBER_OF_QUERIES = 5



def main() -> None:
    print("=== LEVEL 2: Tiny Semantic Search ===")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    corpus = {}
    # Load corpus from JSON file
    with open("corpus.json", "r") as file:
        corpus = json.load(file)
    paragraphs = corpus["paragraphs"]
    start = time.perf_counter()
    embeddings = model.encode(paragraphs, normalize_embeddings=True)
    print(f"Encoded corpus in {time.perf_counter() - start:.2f}s")
    for i in range(NUMBER_OF_QUERIES):
        # Get user input
        query = []
        user_input = input("\nEnter your search query: ")
        query.append(user_input)
        query_time_start = time.perf_counter()
        query_embedding = model.encode([user_input], normalize_embeddings=True)

        # Cosine similarity between query and all paragraph embeddings
        similarity = embeddings @ query_embedding.T
        # Transform into a 1D array
        similarity = similarity.T
        similarity = similarity.flatten()
        # Get top K matches
        indexed_scores = [(index, score) for index, score in enumerate(similarity)]
        top_matches = sorted(indexed_scores, key=lambda item: item[1], reverse=True)[:TOP_K]
        # Display results
        print(f"\nTop matches for query: '{user_input}'")
        for rank, (index, score) in enumerate(top_matches, start=1):
            print(f"  #{rank}  cosine={score:.3f}  →  [{index}] {paragraphs[index]}")
        print(f"\nQuery processed in {time.perf_counter() - query_time_start:.2f}s")

if __name__ == "__main__":
    main()