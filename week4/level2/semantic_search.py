"""
Level 2
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("sentence-transformers not installed. Run: uv pip install sentence-transformers")

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3


def build_corpus() -> List[str]:
    # 24 short paragraphs covering multiple topics 
    return [
        # Technology / Computing
        "Researchers released a new open-source machine learning library for image processing.",
        "Quantum computing research is accelerating with new error-correction techniques.",
        "A major tech company announced an affordable smartphone with a long-lasting battery.",
        "Advances in natural language models enable faster document summarization.",
        "Edge computing moves inference closer to IoT devices to reduce latency.",
        "A startup launched a privacy-focused messaging app that uses end-to-end encryption.",

        # Travel / Airlines / Transport
        "Budget airlines have begun charging separately for checked bags and seat selection.",
        "The city expanded its light-rail network to connect the airport with downtown.",
        "Travelers experienced long security queues at the international terminal last weekend.",
        "High-speed rail promises to reduce commute time between the two major cities.",
        "A new rideshare service offers contactless pickup and bicycle delivery options.",
        "Port authorities announced upgrades to the cruise terminal and passenger facilities.",

        # Food / Cooking
        "The bakery is famous for its sourdough loaves and seasonal fruit tarts.",
        "She follows a Mediterranean diet, cooking with olive oil, fish, and fresh vegetables.",
        "Slow-cooked stews develop deeper flavors when simmered for several hours.",
        "A local chef demonstrated how to make flaky pastry for apple galettes.",
        "Farmers markets sell heirloom tomatoes, artisanal cheeses, and fresh herbs.",
        "Home fermentation projects include kombucha, kimchi, and sourdough starters.",

        # Astronomy / Science / Museums
        "Amateur astronomers gathered to observe the annual meteor shower from a dark site.",
        "The museum reopened an exhibit showcasing Renaissance paintings and sculptures.",
        "A science magazine published an explainer on how gravitational waves are detected.",
        "Conservationists study coral reef recovery after protective measures were introduced.",
        "Satellite data helped researchers track seasonal changes in the polar ice caps.",
        "An observatory installed a new wide-field camera for deep-sky surveys.",
    ]


def embed_corpus(model: SentenceTransformer, corpus: List[str]): # convert each paragraph into a vector of numbers
    t0 = time.perf_counter()
    embeddings = model.encode(corpus, normalize_embeddings=True) # make vectors unit length
    t1 = time.perf_counter()
    return embeddings, t1 - t0


def search_embeddings(embeddings, query_emb, k=TOP_K) -> List[Tuple[int, float]]:
    # embeddings
    scores = embeddings @ query_emb
    # get top k indices
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(int(idx), float(score)) for idx, score in ranked]


def run_queries(model: SentenceTransformer, corpus: List[str], embeddings, queries: List[str], out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Queries and top-3 results:\n\n")

        for q_idx, q in enumerate(queries, start=1):
            t0 = time.perf_counter()
            q_emb = model.encode([q], normalize_embeddings=True)[0]
            t1 = time.perf_counter()
            query_time = t1 - t0

            results = search_embeddings(embeddings, q_emb, k=TOP_K)

            f.write(f"Query {q_idx}: {q}\n")
            f.write(f"Embedding time: {query_time:.4f}s\n")
            for rank, (doc_idx, score) in enumerate(results, start=1):
                snippet = corpus[doc_idx][:200].replace("\n", " ")
                f.write(f"  #{rank}  id={doc_idx}  score={score:.4f}  snippet={snippet}\n")
            f.write("\n")

        # Reflection and timings
        f.write("Reflection:\n")
        f.write("- Retrieval works best when queries match content in the corpus and phrasing.\n")
        f.write("- It may fail when queries are ambiguous or require deep world knowledge not captured by short paragraphs.\n")
        f.write("- Normalising embeddings makes similarity a fast dot product and keeps ranking stable.\n\n")

    print(f"Wrote search examples to: {out_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run demo queries and write examples file")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of results to return")
    args = parser.parse_args()

    corpus = build_corpus()
    print(f"Loaded corpus with {len(corpus)} documents")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    emb, elapsed = embed_corpus(model, corpus)
    print(f"Embedded corpus in {elapsed:.4f}s")

    if args.demo: # demo mode
        demo_queries = [ # predefined queries
            "best ways to preserve sourdough starter",                    
            "how to reduce airplane baggage fees",                        
            "recent breakthroughs in quantum computing",              
            "when is the next meteor shower visible",                   
            "museum opening hours for European paintings exhibit",  
        ]
        out_file = Path(__file__).resolve().parent / "search_examples.txt"
        run_queries(model, corpus, emb, demo_queries, out_file)
        return

    # Interactive mode
    print("Enter a query (or 'exit' to quit):")
    while True:
        q = input("query> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        t0 = time.perf_counter()
        q_emb = model.encode([q], normalize_embeddings=True)[0] # query convert into vector
        t1 = time.perf_counter()
        results = search_embeddings(emb, q_emb, k=args.k)
        print(f"Query embedded in {t1-t0:.4f}s — top {args.k} results:")
        for rank, (doc_idx, score) in enumerate(results, start=1):
            print(f"  #{rank}  id={doc_idx}  score={score:.4f}  snippet={corpus[doc_idx][:140]}")


if __name__ == "__main__":
    main()
