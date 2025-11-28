"""
Week 4 Diagnostic Task — Level 1
"""

import time
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

sentences = [
    "The cat chased the mouse across the yard.",
    "A kitten pursued a small rodent in the garden.",
    "Apple released a new smartphone this week.",
    "She baked an apple pie for dessert.",
    "I enjoy jogging early in the morning.",
    "Running at dawn clears my mind before work.",
    "Quantum computers could break current cryptography.",
    "Stargazers watched the meteor shower from the beach.",
    "Budget airlines often charge high fees for luggage.",
    "The museum displays centuries-old European paintings.",
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

    # Collect top-k neighbours for each sentence so we can both print and log to file
    results = []
    for idx, sentence in enumerate(sentences):
        row = similarity[idx]
        others = [
            (other_idx, float(score))
            for other_idx, score in enumerate(row)
            if other_idx != idx
        ]
        top_matches = sorted(others, key=lambda item: item[1], reverse=True)[:TOP_K]

        print(f"\nSentence [{idx}]: {sentence}")
        for rank, (match_idx, score) in enumerate(top_matches, start=1):
            print(f"  #{rank}  cosine={score:.3f}  →  [{match_idx}] {sentences[match_idx]}")

        results.append({
            "idx": idx,
            "sentence": sentence,
            "top_matches": top_matches,
        })

    # Write the neighbours and notes to a log file in the same directory as this script
    out_path = Path(__file__).resolve().parent / "nearest_neighbours.txt"
    lines = []
    lines.append("Source sentences:")
    for r in results:
        lines.append(f"[{r['idx']}] {r['sentence']}")

    lines.append("")
    lines.append("Top-3 neighbours (cosine similarity):")
    lines.append("")
    for r in results:
        lines.append(f"Sentence [{r['idx']}]: {r['sentence']}")
        for rank, (match_idx, score) in enumerate(r['top_matches'], start=1):
            lines.append(f"  #{rank}  cosine={score:.3f}  →  [{match_idx}] {sentences[match_idx]}")
        lines.append("")

    # Add short note identifying contrasting cases (semantic vs lexical)
    lines.append("Notes — contrasting cases:")
    lines.append("- Lexically different but semantically close:")
    lines.append("  Sentences [4] and [5] are phrased differently (jogging vs running, morning vs dawn) ")
    lines.append("  but are semantically near (exercise/time of day).")
    lines.append("- Lexically similar but semantically far apart:")
    lines.append("  Sentences [2] and [3] both contain the token 'apple' but refer to different senses (company vs food).")

    out_text = "\n".join(lines)
    out_path.write_text(out_text, encoding="utf-8")
    print(f"\nWrote nearest neighbours log to: {out_path}")

if __name__ == "__main__":
    main()