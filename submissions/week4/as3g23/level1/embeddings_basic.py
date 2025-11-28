import time
from sentence_transformers import SentenceTransformer

# 1. Setup the model
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)


sentences = [
    # Case A: Semantically similar, Lexically different (The "Cat" test)
    "The feline was sleeping on the rug.",       # Index 0
    "A cat rested on the carpet.",               # Index 1

    # Case B: Lexically similar, Semantically different (The "Bank" test)
    "I deposited money at the river bank.",      # Index 2
    "I deposited money at the savings bank.",    # Index 3

    # Random Distractors (to fill out the list)
    "SpaceX launched a new Starship rocket.",    # Index 4
    "The chef cooked a spicy curry.",            # Index 5
    "Python is a great language for data.",      # Index 6
    "It is raining heavily outside today."       # Index 7
]

def main():
    print("=== LEVEL 1: EMBEDDINGS BASIC ===")
    
    start = time.perf_counter()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    print(f"Encoded {len(sentences)} sentences in {time.perf_counter() - start:.4f}s")
    similarity = embeddings @ embeddings.T

    for idx, sentence in enumerate(sentences):
        print(f"\nSource [{idx}]: {sentence}")
        scores = []
        for i, score in enumerate(similarity[idx]):
            if i != idx:
                scores.append((i, score))
        top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        
        for rank, (match_idx, score) in enumerate(top_matches, start=1):
            print(f"  #{rank}  Score: {score:.3f}  ->  [{match_idx}] {sentences[match_idx]}")

if __name__ == "__main__":
    main()