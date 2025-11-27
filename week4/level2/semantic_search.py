import time
import json
from pathlib import Path
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "sentence-transformers is not installed.\n"
        "Run: uv pip install sentence-transformers"
    )

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

def load_corpus(filename: str = "corpus.json") -> list[dict]:
    """Loads the corpus data from a JSON file relative to this script."""
    path = Path(__file__).parent / filename
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found at: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def search(query: str, model: SentenceTransformer, corpus: list[dict], corpus_embeddings: list):
    query_embedding = model.encode([query], normalize_embeddings=True)[0] # Encode query
    similarity = query_embedding @ corpus_embeddings.T # Calculate cosine similarity
    results = list(enumerate(similarity)) # Get list of (index, score)
    results = sorted(results, key=lambda item: item[1], reverse=True)[:TOP_K] # Top K results
    top_results = results[:TOP_K] # Get top K results
    return top_results


def main():
    with open("search_examples.txt", "w", encoding="utf-8") as f:
        f.write("=== LEVEL 2: TINY SEMANTIC SEARCH ===\n")
        model = SentenceTransformer(MODEL_NAME)
        corpus_path = "corpus.json" 
        f.write(f"Loading corpus from {corpus_path}...\n")
        
        corpus = load_corpus(corpus_path)
        corpus_texts = [doc["text"] for doc in corpus]
        
        f.write(f"Loaded {len(corpus)} documents.\n")
        start_time = time.perf_counter()
        corpus_embeddings = model.encode(corpus_texts, normalize_embeddings=True) # Encode corpus
        end_time = time.perf_counter()
        f.write(f"Encoded corpus in {end_time - start_time:.2f} seconds.\n")
        queries = ["What is being used to solve QEC problems?", "How do mRNA vaccines work?", "How long after the finding of the rosetta stone did we land on the moon?", "How to temper chocolate?", "How do quantum computers differ from the computers in apollos guidance system?"]
        for query in queries:
            f.write(f"\n=== Query: '{query}' ===\n")
            
            start = time.perf_counter()
            top_results = search(query, model, corpus, corpus_embeddings)
            end = time.perf_counter() # Stop timer immediately after search
            
            for rank, (idx, score) in enumerate(top_results, start=1):
                doc_text = corpus[idx]["text"]
                preview = doc_text[:100] + "..."
                f.write(f" #{rank} cosine={score:.3f} -> [{idx}] {preview}\n") 
            f.write(f"Search time: {end - start:.4f} seconds.\n")

if __name__ == "__main__":
    main()