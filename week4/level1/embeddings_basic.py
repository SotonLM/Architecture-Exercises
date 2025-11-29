"""
Week 4 Diagnostic Task — Embeddings & Retrieval

LEVEL 1  → Turn sentences into vectors, measure cosine similarity, print nearest neighbours.
LEVEL 2  → Build `semantic_search.py` (20-30 docs + interactive queries).
LEVEL 3  → Pick ONE: quality metric (`quality_results.txt`) OR tiny RAG loop (`rag_comparison.txt`).
LEVEL 4  → Freestyle retrieval-based tool with sample outputs + "what I'd improve next".

Most students should stop after Level 2. Levels 3-4 are optional Tier 3 stretch goals.
"""

import time

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
    "The bank raised interest rates to control inflation.",          
    "She sat by the river bank watching the sunset.",               
    "A financial institution approved my mortgage application.",    
    "The chef prepared a spicy curry for the evening guests.",      
    "A hot and flavorful dish was served at the dinner table.",     
    "The jaguar prowled silently through the dense rainforest.",    
    "A luxury car brand unveiled its latest sports model.",         
    "Astronomers discovered a new planet orbiting a distant star.", 
    "Scientists found an exoplanet in a faraway solar system.",     
    "The artist painted a vivid portrait of a young woman.",        
]






def main() -> None:
    print("=== LEVEL 1: EMBEDDINGS + NEAREST NEIGHBOURS ===")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    start = time.perf_counter()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    print(f"Encoded {len(sentences)} sentences in {time.perf_counter() - start:.2f}s")

    # Cosine similarity matrix (vectors are L2-normalised, so dot product == cosine)
    similarity = embeddings @ embeddings.T

    for idx, sentence in enumerate(sentences):
        row = similarity[idx]
        others = [
            (other_idx, score)
            for other_idx, score in enumerate(row)
            if other_idx != idx
        ]
        top_matches = sorted(others, key=lambda item: item[1], reverse=True)[:TOP_K]

        print(f"\nSentence [{idx}]: {sentence}")
        for rank, (match_idx, score) in enumerate(top_matches, start=1):
            print(f"  #{rank}  cosine={score:.3f}  →  [{match_idx}] {sentences[match_idx]}")

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
