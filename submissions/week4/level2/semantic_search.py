import time
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "sentence-transformers is not installed.\n"
        "Run: uv pip install sentence-transformers"
    )

import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

# --- 1) CORPUS: 20–30 short paragraphs across 3–4 topics ---

# I did just use AI to generate these paragraphs
corpus: List[str] = [
    # --- Finance / Economics ---
    "Central banks raise interest rates to cool down high inflation and reduce spending.",
    "Index funds allow investors to buy a broad slice of the stock market at low cost.",
    "During a recession, governments often increase spending to support demand.",
    "High-frequency traders use algorithms to exploit tiny, short-lived price changes.",
    "Diversification reduces risk by spreading investments across many assets.",
    "Bond prices usually fall when interest rates go up, and rise when rates go down.",

    # --- Machine learning / AI ---
    "Neural networks learn patterns by adjusting weights to minimise a loss function.",
    "Overfitting happens when a model memorises the training data instead of generalising.",
    "Gradient descent iteratively updates parameters in the direction of steepest descent.",
    "Transformers use attention mechanisms to focus on the most relevant parts of the input.",
    "A validation set is used to tune hyperparameters without touching the test set.",
    "Regularisation techniques like dropout help prevent overfitting in deep models.",

    # --- History / Culture ---
    "The Roman Empire built an extensive road network that supported trade and armies.",
    "The printing press drastically reduced the cost of copying books in medieval Europe.",
    "Ancient Egyptian civilisation depended on the predictable flooding of the Nile River.",
    "The Industrial Revolution transformed economies from agrarian to manufacturing-based.",
    "The Great Wall of China was built over centuries to defend against invasions.",
    "Greek philosophers like Plato and Aristotle laid foundations for Western thought.",

    # --- Everyday life / Study / Wellbeing ---
    "Taking short breaks while studying can improve focus and long-term memory.",
    "A consistent sleep schedule helps regulate energy levels throughout the day.",
    "Cooking at home lets you control ingredients and often saves money.",
    "Going for a walk outside can reduce stress and clear your mind.",
    "Group study sessions can be helpful if everyone stays on task.",
    "Time-blocking your day can make big projects feel more manageable.",
]


def createEmbeddings(model: SentenceTransformer, corpus: List[str]) -> np.ndarray:
    start = time.perf_counter()
    embeddings = model.encode(corpus, normalize_embeddings=True)
    print(f"Corpus embedded in {time.perf_counter() - start:.2f} seconds.\n")
    return embeddings

def semantic_search(query: str, model: SentenceTransformer, corpus: List[str], corpusEmbeddings: np.ndarray, topK: int = TOP_K) -> None:
    if not query.strip():
        print("Empty query, please type again.")
        return

    print(f"\n=== QUERY: {query!r} ===")
    start = time.perf_counter()
    queryEmbedding = model.encode([query], normalize_embeddings=True)[0]
    print(f"Encoded query in {time.perf_counter() - start:.2f}s")

    similarities = corpusEmbeddings @ queryEmbedding

    k = min(topK, len(corpus))

    top_indices = np.argsort(-similarities)[:k]

    print(f"Top {k} results:\n")

    for rank, idx in enumerate(top_indices, start=1):
        queryStart = time.perf_counter()
        score = float(similarities[idx])
        document = corpus[idx]
        print(f"#{rank}\t[doc_id={idx}]\tcosine={score:.3f}\n{document}\n")
        print(f"Answered query in {time.perf_counter() - queryStart:.4f}s\n")

def main():
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    corpus_embeddings = createEmbeddings(model, corpus)

    print("Type a query to search the corpus (or just press ENTER to quit).")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            break
        semantic_search(query, model, corpus, corpus_embeddings, topK=TOP_K)

if __name__ == "__main__":
    main()
