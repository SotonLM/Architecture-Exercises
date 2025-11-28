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


def loadcorpus():
    with open("corpus.txt","r") as file:
        sentences = file.read().splitlines()
    return sentences
    
def create_embedding(model):
    corpus = loadcorpus()
    start = time.perf_counter()
    embeddings = model.encode(corpus, normalize_embeddings=True)
    print(f"Encoded {len(corpus)} sentences in {time.perf_counter() - start:.2f}s")


    np.save("embedding.npy",embeddings)
    print("Saved embeddings → embeddings.npy")
    return embeddings

def load_embeddings():
    return np.load("embedding.npy")    
 
def search(model,query,embeddings):
    corpus = loadcorpus()
    start = time.perf_counter()
    queryembeddings = model.encode(query, normalize_embeddings=True)
    print(f"Encoded {len(corpus)} sentences in {time.perf_counter() - start:.2f}s")
    
    similarity = embeddings @ queryembeddings
    
    result = []
    top_indices = similarity.argsort()[::-1][:TOP_K]
    for idx in top_indices:
        result.append((idx, corpus[idx],float(similarity[idx])))
        
    return result


      
def main() -> None:
    print("=== LEVEL 1: EMBEDDINGS + NEAREST NEIGHBOURS ===")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    try:
        embeddings = load_embeddings()
    except:
        embeddings = create_embedding(model)
        
    query = input("\nEnter your search query: ").strip()
    result = search(model,query,embeddings)
    
    
    # Cosine similarity matrix (vectors are L2-normalised, so dot product == cosine)
    
    print("\n=== Top Results ===")
    for idx, text, score in result:
            print(f"[{idx}] (cosine={score:.3f})  {text}")


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
