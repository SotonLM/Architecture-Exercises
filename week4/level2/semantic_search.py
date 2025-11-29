
# Import libraries
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# Attempting to import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "sentence-transformers is not installed.\nRun: uv pip install sentence-transformers"
    )

# Model settings
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3


# Read the corpus
def read_corpus(file_path: str) -> dict:
    corpus = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            formatted = line.strip().split(".")[0]
            if formatted:
                corpus.append(formatted)
    return corpus


# Save computed embeddings to disk for reuse
def store_embeddings(embeddings: list, file_path: str) -> None:
    np.save(file_path, embeddings)


# Visualize similarity distribution as a bar chart
def plot_similarities(similarities, query):
    x = np.arange(len(similarities))
    plt.figure(figsize=(12, 4))
    plt.bar(x, similarities)
    plt.xlabel("Sentence Index")
    plt.ylabel("Similarity")
    plt.title(f"Similarity Scores for Query: {query}")
    filename = f"plot_{query}_similarity.png"
    filename = filename.replace(" ", "_").replace("?", "")
    plt.savefig(filename)


# Main function
def main() -> None:
    print("=== LEVEL 2: SEMANTIC SEARCH ===")
    print(f"Loading model: {MODEL_NAME}")
    
    # Load model and corpus
    model = SentenceTransformer(MODEL_NAME)
    corpus = read_corpus("level2/corpus.txt")

    # Compute or load corpus embeddings
    if os.path.exists("level2/embeddings.npy"):
        print("Embeddings already stored.")
        corpus_embeddings = np.load("level2/embeddings.npy")
    else:
        print("Computing embeddings for the corpus.")
        start = time.perf_counter()
        corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
        print(f"Encoded {len(corpus)} sentences in {time.perf_counter() - start:.2f}s")
        print("Storing embeddings.")
        store_embeddings(corpus_embeddings, "level2/embeddings.npy")

    # Get user query
    query = input("Enter your search query: ")

    # Perform semantic search
    start = time.perf_counter()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = corpus_embeddings @ query_embedding
    top_matches = np.argsort(-similarities)[:TOP_K]
    print(f"Search completed in {time.perf_counter() - start:.2f}s")

    # Display results
    print(f"\nTop {TOP_K} results for query: '{query}'")
    for idx in range(TOP_K):
        rank = idx + 1
        corpus_index = top_matches[idx]
        similarity = similarities[corpus_index]
        print(
            f"#{rank}   Cosine: {similarity:.3f}  Index: {corpus_index}  Sentence: {corpus[corpus_index]}"
        )

    plot_similarities(similarities, query)


if __name__ == "__main__":
    main()
