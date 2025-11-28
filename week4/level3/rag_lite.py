"""
Level 3, Option B
"""

import time
from pathlib import Path
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("sentence-transformers not installed. Run: uv pip install sentence-transformers")

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    raise SystemExit("transformers not installed. Run: uv pip install transformers")

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"
TOP_K = 3


def build_corpus() -> List[str]:
    # Reuse a compact 24-item corpus
    # similar to Level 2
    return [
        "Researchers released a new open-source machine learning library for image processing.",
        "Quantum computing research is accelerating with new error-correction techniques.",
        "A major tech company announced an affordable smartphone with a long-lasting battery.",
        "Advances in natural language models enable faster document summarization.",
        "Edge computing moves inference closer to IoT devices to reduce latency.",
        "A startup launched a privacy-focused messaging app that uses end-to-end encryption.",
        "Budget airlines have begun charging separately for checked bags and seat selection.",
        "The city expanded its light-rail network to connect the airport with downtown.",
        "Travelers experienced long security queues at the international terminal last weekend.",
        "High-speed rail promises to reduce commute time between the two major cities.",
        "A new rideshare service offers contactless pickup and bicycle delivery options.",
        "Port authorities announced upgrades to the cruise terminal and passenger facilities.",
        "The bakery is famous for its sourdough loaves and seasonal fruit tarts.",
        "She follows a Mediterranean diet, cooking with olive oil, fish, and fresh vegetables.",
        "Slow-cooked stews develop deeper flavors when simmered for several hours.",
        "A local chef demonstrated how to make flaky pastry for apple galettes.",
        "Farmers markets sell heirloom tomatoes, artisanal cheeses, and fresh herbs.",
        "Home fermentation projects include kombucha, kimchi, and sourdough starters.",
        "Amateur astronomers gathered to observe the annual meteor shower from a dark site.",
        "The museum reopened an exhibit showcasing Renaissance paintings and sculptures.",
        "A science magazine published an explainer on how gravitational waves are detected.",
        "Conservationists study coral reef recovery after protective measures were introduced.",
        "Satellite data helped researchers track seasonal changes in the polar ice caps.",
        "An observatory installed a new wide-field camera for deep-sky surveys.",
    ]


def embed_corpus(model, corpus):
    t0 = time.perf_counter()
    embeddings = model.encode(corpus, normalize_embeddings=True)
    return embeddings, time.perf_counter() - t0


def search_topk(embeddings, q_emb, k=TOP_K):
    scores = embeddings @ q_emb
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(int(i), float(s)) for i, s in ranked]


def generate_answer(tokenizer, gen_model, prompt: str, max_length=128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    out = gen_model.generate(**inputs, max_new_tokens=max_length, num_beams=3)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def main():
    corpus = build_corpus()
    print(f"Loaded corpus with {len(corpus)} documents")

    # Load encoder for retrieval
    print(f"Loading encoder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME)
    emb, enc_time = embed_corpus(encoder, corpus)
    print(f"Embedded corpus in {enc_time:.4f}s")

    # Load generator
    print(f"Loading generator: {GEN_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

    # Ten diverse queries for evaluation
    queries = [
        "How can I keep a sourdough starter alive for months?",
        "Ways to avoid high baggage fees when flying on budget airlines",
        "What are recent developments in quantum computing error correction?",
        "When is the next meteor shower and where to watch it?",
        "How do I make flaky pastry for apple galettes?",
        "What precautions help coral reef recovery?",
        "How does edge computing reduce latency for IoT devices?",
        "Museum hours for Renaissance painting exhibits in the city?",
        "How to detect gravitational waves?",
        "What are typical symptoms of long security queues at airports and remedies?",
    ]

    out_lines = []
    out_lines.append("RAG-lite comparisons\n")

    for qi, q in enumerate(queries, start=1):
        print(f"Running query {qi}/{len(queries)}")
        # Embed query
        t0 = time.perf_counter()
        q_emb = encoder.encode([q], normalize_embeddings=True)[0]
        q_enc = time.perf_counter() - t0

        # Retrieve top-k
        topk = search_topk(emb, q_emb, k=TOP_K)

        # Build context strings
        context_parts_log = []
        context_parts_gen = []
        for doc_idx, score in topk:
            context_parts_log.append(f"[{doc_idx}] {corpus[doc_idx]}")
            context_parts_gen.append(corpus[doc_idx])
        context_text_log = "\n".join(context_parts_log)
        context_text = " ".join(context_parts_gen)

        # Generate answers
        prompt_no_ctx = f"Question: {q}\nAnswer concisely."
        ans_no_ctx = generate_answer(tokenizer, gen_model, prompt_no_ctx)

        prompt_with_ctx = (
            f"Use the following information to answer the question. "
            f"Write a natural, concise answer in complete sentences.\n\n"
            f"{context_text}\n\nQuestion: {q}\nAnswer:"
        )
        
        ans_with_ctx = generate_answer(tokenizer, gen_model, prompt_with_ctx)
        ans_with_ctx = ans_with_ctx.replace("(iii)", "").replace("(II)", "").replace("(B).", "").strip()


        # Logging
        out_lines.append(f"Query {qi}: {q}")
        out_lines.append(f"Embedding time: {q_enc:.4f}s\n")
        out_lines.append("Retrieved context:")
        for doc_idx, score in topk:
            out_lines.append(f"  id={doc_idx} score={score:.4f} snippet={corpus[doc_idx]}")
        out_lines.append("\nAnswer — Without context:")
        out_lines.append(ans_no_ctx)
        out_lines.append("\nAnswer — With context:")
        out_lines.append(ans_with_ctx)
        out_lines.append("\n" + ("-"*50) + "\n")

    out_path = Path(__file__).resolve().parent / "rag_comparison.txt"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote RAG comparisons to: {out_path}")


if __name__ == "__main__":
    main()
