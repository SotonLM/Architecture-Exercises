
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

# 1. Create corpus (3–4 topics)
corpus = [
    # AI
    "Neural networks learn by adjusting millions of parameters.",
    "Backpropagation is the algorithm used to train deep learning models.",
    "Transformers revolutionized NLP by introducing attention mechanisms.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "AI ethics focuses on fairness, transparency, and accountability.",
    # Cooking
    "My grandma's apple pie relies on butter, cinnamon, and patience.",
    "Boiling pasta requires plenty of salted water for best results.",
    "Grilling vegetables adds a smoky flavor and preserves nutrients.",
    "Sourdough bread uses natural fermentation for a tangy taste.",
    "Chocolate ganache is made by mixing cream and melted chocolate.",
    # Travel
    "Paris is famous for the Eiffel Tower and its rich culture.",
    "Tokyo blends modern skyscrapers with ancient temples.",
    "Backpacking through Europe offers diverse experiences and cuisines.",
    "The Grand Canyon is a breathtaking natural wonder in Arizona.",
    "Safari trips in Kenya allow you to see lions and elephants.",
    # Health
    "Regular exercise improves cardiovascular health and mood.",
    "A balanced diet includes proteins, carbs, and healthy fats.",
    "Meditation can reduce stress and improve mental clarity.",
    "Sleep is essential for memory consolidation and recovery.",
    "Hydration is key to maintaining energy and focus."
]


print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

start_corpus = time.perf_counter()
embeddings = model.encode(corpus, normalize_embeddings=True)
end_corpus = time.perf_counter()
print(f"Corpus embedding time: {end_corpus - start_corpus:.2f}s")


np.save("corpus_embeddings.npy", embeddings)
with open("corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=4)
# corpus being saved in multiple formats

while True:
    query = input("\nEnter your query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    start_query = time.perf_counter()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_embedding  
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    end_query = time.perf_counter()

    print(f"\nQuery processing time: {end_query - start_query:.2f}s")
    print(f"Top {TOP_K} results for: '{query}'")
    for rank, idx in enumerate(top_indices, start=1):
        print(f"#{rank} (score={scores[idx]:.3f}) → [{idx}] {corpus[idx]}")

