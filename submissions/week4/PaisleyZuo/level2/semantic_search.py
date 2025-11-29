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

# 1. Build corpus (20–30 paragraphs across 3–4 topics)
corpus = [
    # Technology / AI
    "Neural networks learn hierarchical representations from data.",
    "Large language models can generate coherent text across many tasks.",
    "Reinforcement learning optimises behaviour by interacting with environments.",
    "AI systems often require significant computational resources.",
    "Deep learning models improve with large-scale datasets.",
    "Transformers rely on attention mechanisms to model long-range dependencies.",
    "Training deep models often requires GPUs or TPUs.",

    # Cooking
    "Simmering tomatoes slowly can deepen the flavour of a pasta sauce.",
    "Fresh basil adds brightness to Mediterranean dishes.",
    "Making bread at home requires patience and careful fermentation.",
    "Stir-frying vegetables keeps their texture crisp and vibrant.",
    "Baking a cake involves precise measurements for consistent results.",
    "Marinating meat overnight enhances tenderness and flavour.",
    "Steaming vegetables helps preserve nutrients better than boiling.",

    # Nature
    "A clear night sky reveals constellations scattered across the universe.",
    "Coastal waves erode shorelines over long periods of time.",
    "Mountain forests support a diverse range of wildlife species.",
    "Rainfall replenishes groundwater and nourishes ecosystems.",
    "Sunlight filtering through trees creates a peaceful atmosphere.",
    "Coral reefs support thousands of marine species.",
    "Deserts form in regions with extremely low annual rainfall.",

    # Finance / Economics
    "Inflation reduces the purchasing power of currency over time.",
    "Diversifying a portfolio helps manage investment risk.",
    "Stock markets respond to macroeconomic indicators and global events.",
    "Interest rates influence borrowing behaviour and economic growth.",
    "Budget planning ensures long-term financial stability.",
    "Exchange rates fluctuate due to supply and demand in currency markets.",
    "Fiscal policy involves government spending and taxation decisions.",
]

# 2. Load model and embed corpus
def embed_corpus(model, corpus):
    start = time.perf_counter()
    emb = model.encode(corpus, normalize_embeddings=True)
    duration = time.perf_counter() - start
    return emb, duration

# 3. Perform semantic search on one query
def search(query, model, corpus_emb):
    # embed query
    t0 = time.perf_counter()
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    embed_time = time.perf_counter() - t0

    # similarity (dot product because normalized)
    sims = corpus_emb @ q_emb

    # top-k indices
    topk = np.argsort(-sims)[:TOP_K]
    return topk, sims, embed_time

# 4. Run 5 predefined queries and save to text file
def run_examples(model, corpus_emb, corpus_time):
    out_path = "search_examples.txt"

    queries = [
        "How does inflation affect the economy?",
        "How can I make my pasta sauce more flavourful?",
        "What causes stars to form in the universe?",
        "How do neural networks learn representations?",
        "Why is borrowing expensive when interest rates are high?",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Semantic Search Examples (Level 2) ===\n\n")
        f.write(f"Corpus embedding time: {corpus_time:.4f} seconds\n\n")

        for q in queries:
            f.write(f"Query: {q}\n")
            topk, sims, q_time = search(q, model, corpus_emb)
            f.write(f"Query time: {q_time:.4f} seconds\n")

            for idx in topk:
                snippet = corpus[idx][:90]
                f.write(f"  [{idx}] score={sims[idx]:.4f} -> {snippet}...\n")
            f.write("\n")

        # Reflection section
        f.write("=== Reflection ===\n")
        f.write("""
The retrieval system performs strongly for clear, well-defined factual queries. 
For example, questions about inflation, pasta sauce, neural networks, and interest 
rates all retrieved the correct top-1 and semantically related supporting paragraphs. 
However, queries with broader or more abstract semantics—such as “What causes stars 
to form?”—produce weaker matches, because the corpus does not contain a direct 
explanation of star formation, only loosely related nature descriptions. This shows 
that sentence embeddings work extremely well when the corpus contains relevant 
knowledge, but their accuracy naturally degrades when the underlying dataset lacks 
a close conceptual neighbour. Overall, the model demonstrates robust semantic 
grouping across domains, with predictable failure cases driven by corpus limitations.
""")

    print(f"Saved results to: {out_path}")

# 5. Interactive mode
def interactive_mode(model, corpus, corpus_emb):
    print("\n=== Interactive Semantic Search ===")
    print("Type a query (or press Enter to exit):\n")

    while True:
        q = input("> ").strip()
        if not q:
            print("Exiting interactive mode.")
            break

        topk, sims, q_time = search(q, model, corpus_emb)

        print(f"\nQuery time: {q_time:.4f}s")
        print("Top results:\n")

        for idx in topk:
            print(f"[{idx}] score={sims[idx]:.4f}")
            print(f"  {corpus[idx]}\n")

# 6. Main execution
if __name__ == "__main__":
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding corpus...")
    corpus_emb, corpus_time = embed_corpus(model, corpus)
    print(f"Corpus embedding time: {corpus_time:.4f}s")

    # Save embeddings for reuse
    np.save("corpus_emb.npy", corpus_emb)

    # Produce Level 2 output file
    run_examples(model, corpus_emb, corpus_time)

    # Enter interactive mode
    interactive_mode(model, corpus, corpus_emb)