# week4/level2/semantic_search.py

import json
import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3
OUT_PATH = "week4/level2/search_examples.txt"



def load_corpus(path: str = "week4/level2/corpus.json") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_corpus(model, corpus):
    print("\nEncoding corpus...")
    start = time.perf_counter()

    embeddings = model.encode(
        corpus,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    elapsed = time.perf_counter() - start
    print(f"Corpus embedded in {elapsed:.3f}s")

    return embeddings, elapsed



def write_header(corpus_size, model_time, embed_time):
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(" SEMANTIC SEARCH EXAMPLES \n\n")
        f.write(f"Corpus size: {corpus_size}\n")
        f.write(f"Model load time: {model_time:.3f}s\n")
        f.write(f"Corpus embedding time: {embed_time:.3f}s\n")
        f.write("\n YOUR QUERIES: \n\n")

    print(f"Logging to: {OUT_PATH}")



def search_query(model, corpus, corpus_emb, query):
    q_start = time.perf_counter()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_time = time.perf_counter() - q_start

    sims = q_emb @ corpus_emb.T
    sims = sims.flatten()

    top_idx = sims.argsort()[::-1][:TOP_K]

    results = [(idx, corpus[idx], float(sims[idx])) for idx in top_idx]

    return results, q_time


def save_query(query, results, q_time):
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Embedding time: {q_time:.4f}s\n")

        for rank, (idx, text, score) in enumerate(results, start=1):
            f.write(f"  #{rank} — Doc {idx} — score={score:.4f}\n")
            f.write(f"      {text}\n")

        f.write("\n")


def main():
    print(" Tiny Semantic Search ")

    # Load corpus
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} documents.")

    # Load model
    print("\nLoading model...")
    m_start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    model_time = time.perf_counter() - m_start
    print(f"Model loaded in {model_time:.3f}s")

    # Embed corpus
    corpus_emb, embed_time = embed_corpus(model, corpus)

    # Create header file
    write_header(len(corpus), model_time, embed_time)

    # Query loop
    print("\nEnter your search queries (type 'quit' to exit):\n")

    while True:
        query = input("> ")

        if query.lower().strip() in {"quit", "exit"}:
            print("\nDone. results are saved in search_examples.txt")
            break

        # Run search
        results, q_time = search_query(model, corpus, corpus_emb, query)

        # Display results
        print("\nTop results:")
        for rank, (idx, text, score) in enumerate(results, start=1):
            print(f"#{rank} — Doc {idx} — score={score:.4f}")
            print(f"   {text}\n")

        # Save to file
        save_query(query, results, q_time)
        print(f"(Saved to {OUT_PATH})\n")


if __name__ == "__main__":
    main()
