"""
Week 4 Diagnostic Task — Embeddings & Retrieval

LEVEL 1  → Turn sentences into vectors, measure cosine similarity, print nearest neighbours.
LEVEL 2  → Build `semantic_search.py` (20-30 docs + interactive queries).
LEVEL 3  → Pick ONE: quality metric (`quality_results.txt`) OR tiny RAG loop (`rag_comparison.txt`).
LEVEL 4  → Freestyle retrieval-based tool with sample outputs + "what I'd improve next".

Most students should stop after Level 2. Levels 3-4 are optional Tier 3 stretch goals.
"""

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

corpus = [
    # --- Technology ---
    "Modern smartphones contain billions of transistors, allowing them to perform complex tasks that once required entire rooms of equipment.",
    "Artificial intelligence has quickly become a tool for everyday life, assisting in everything from writing emails to diagnosing medical images.",
    "Cloud computing enables companies to scale their services rapidly without having to maintain their own hardware infrastructure.",
    "Quantum computers operate on qubits, which can exist in multiple states at once, making them powerful for certain computations.",
    "Electric vehicles continue to grow in popularity, driven by improvements in battery technology and charging infrastructure.",
    "Cybersecurity threats evolve every year, requiring constant updates to encryption methods and network defenses.",

    # --- Nature ---
    "A single oak tree can support hundreds of species, from insects living under its bark to birds nesting among its branches.",
    "Coral reefs, often called underwater rainforests, are home to some of the most diverse ecosystems on the planet.",
    "Mountain ranges influence weather patterns, causing rain to fall on one side while leaving the opposite side unusually dry.",
    "Many migratory birds navigate thousands of miles using Earth’s magnetic field as a natural compass.",
    "Desert landscapes, though seemingly barren, host a remarkable variety of plants adapted to survive extreme temperatures.",
    "Rivers carve deep canyons over millions of years, revealing layers of geological history.",

    # --- History ---
    "The ancient city of Babylon was known for its impressive walls, which some historical accounts claim were wide enough for chariots.",
    "The invention of the printing press in the 15th century dramatically accelerated the spread of knowledge across Europe.",
    "The Silk Road facilitated trade between distant cultures, exchanging goods, technologies, and ideas.",
    "Early astronomers in many civilizations used simple tools to track celestial movements and create surprisingly accurate calendars.",
    "During the Renaissance, artists and scientists often overlapped in their pursuits, blending creativity with empirical study.",
    "Exploration of the Pacific by seafaring cultures demonstrates remarkable navigation skills long before modern instruments existed.",

    # --- Personal Anecdotes ---
    "When I first tried to bake bread, I accidentally added twice as much salt, creating a loaf that tasted like the ocean.",
    "I once got lost in a museum after following a group tour that wasn't mine, only realizing minutes later that they were speaking Italian.",
    "During a camping trip, I woke up to find a curious squirrel rummaging through my backpack in search of trail mix.",
    "The first time I rode a bicycle without training wheels, I was so excited that I forgot how to stop and crashed into a bush.",
    "I tried to learn guitar once, but my dog barked every time I practiced, which was both discouraging and oddly motivating.",
    "Last winter, I attempted to build a snowman, but the snow was so powdery that it fell apart faster than I could shape it."
]


def main() -> None:
    print("=== LEVEL 2")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    start = time.perf_counter()
    embeddings = model.encode(corpus, normalize_embeddings=True)
    print(f"Encoded {len(corpus)} sentences in {time.perf_counter() - start:.2f}s")

    # Cosine similarity matrix (vectors are L2-normalised, so dot product == cosine)
    similarity = embeddings @ embeddings.T

    sentence = input("What string would you like to find nearest neighbours for: ")
    embedding = model.encode(sentence)
    np.append(embeddings,[embedding])
    similarity = embeddings @ embeddings.T
    
    row = similarity[-1]
    item = row[-1]
    
    top_k = sorted(row,reverse=True,key=lambda x: x!=item)[:TOP_K]
    print(top_k)
    

# === SPOTLIGHT EXAMPLES ===
# Semantically similar but different words -> sentences 0 & 1
#   [0] Neural networks learn by adjusting billions of parameters.
#   [1] Backpropagation updates weights to minimise loss in a model.
# Lexically similar but different meaning -> sentences 2 & 3
#   [2] My grandma's apple pie relies on butter, cinnamon, and patience.
#   [3] Apple just announced a new chip for thin-and-light laptops.

# === NEXT STEPS ===
# Level 2 → Build semantic_search.py (corpus embeddings + input() queries + timing logs).
# Level 3 → Pick ONE: retrieval metric or RAG-lite (context + question → generator).
# Level 4 → Freestyle retrieval-based tool with sample outputs + improvement notes.
# See week4/level*/README.md for exact deliverables.


if __name__ == "__main__":
    main()
