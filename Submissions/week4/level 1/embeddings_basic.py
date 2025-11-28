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
"The singer practiced a short song.",
"A pianist played the melody.",
"I mixed the dough in a large bowl.",
"He cut the bread into small pieces.",
"She recorded the lecture for later.",
"I recorded the drum track for the band.",
"The cake cooled on the counter.",
"The drummer tapped a steady beat.",
"I studied the math problems carefully.",
"She studied the cookie recipe quickly.",
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
if __name__ == "__main__":
    main()
# === EXAMPLES ===
""" (semantically close, lexically different) -> sentences 0 & 1
        [0] "The singer practiced a short song."
        [1] "The pianist played the melody."
    (lexically similar, semantically different) -> sentences 9 & 10
        [9] "I studied the math problems carefully."
        [10] "I studied the cookie recipe quickly." """