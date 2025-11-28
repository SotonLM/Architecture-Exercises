import time
import numpy as np
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

# A static corpus of 20 documents (our "Knowledge Base")
CORPUS = [
    "The Mars Rover has sent back new images of the red planet.",
    "NASA plans to send humans to the moon by 2025.",
    "SpaceX focuses on reusable rocket technology.",
    "Saturn's rings are made mostly of ice particles.",
    "Jupiter is the largest planet in our solar system.",
    "A black hole has a gravitational pull so strong light cannot escape.",
    "The International Space Station orbits Earth every 90 minutes.",
    "Pizza dough requires yeast, flour, water, and time to rise.",
    "Sushi is a traditional Japanese dish with vinegared rice.",
    "Tacos are best served with fresh lime and cilantro.",
    "Italian cuisine often features pasta, tomatoes, and olive oil.",
    "Dark chocolate is rich in antioxidants.",
    "Spicy food can temporarily boost metabolism.",
    "Python functions are defined using the def keyword.",
    "Java is a statically typed object-oriented language.",
    "React is a popular JavaScript library for building UIs.",
    "Docker containers ensure software runs the same everywhere.",
    "Machine learning models require training data to generalize.",
    "Recursion is a function calling itself to solve a problem.",
    "Variables in programming store data values for later use."
]

def main():
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Indexing {len(CORPUS)} documents...")
    start_time = time.perf_counter()
    
    corpus_embeddings = model.encode(CORPUS, normalize_embeddings=True)
    
    duration = time.perf_counter() - start_time
    print(f"Indexing complete in {duration:.4f}s.")

    print("\n=== TINY SEMANTIC SEARCH ENGINE (Type 'exit' to quit) ===")

    while True:
        query = input("\nQuery: ")
        if query.strip().lower() in ["exit", "quit"]:
            break
        
        if not query.strip():
            continue

        t0 = time.perf_counter()
        
        query_embedding = model.encode(query, normalize_embeddings=True)
        scores = corpus_embeddings @ query_embedding
        top_indices = np.argsort(scores)[-TOP_K:][::-1]
        
        t1 = time.perf_counter()

        # 3. Display Results
        print(f"Results found in {(t1-t0)*1000:.2f}ms:")
        for rank, idx in enumerate(top_indices, start=1):
            score = scores[idx]
            print(f"  #{rank} [Score: {score:.3f}] {CORPUS[idx]}")

if __name__ == "__main__":
    main()