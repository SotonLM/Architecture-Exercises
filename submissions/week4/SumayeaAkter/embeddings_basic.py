import time
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 3

# --- Your custom sentences ---
sentences = [
    "Jogging at sunrise helps me clear my mind.",
    "Running in the morning boosts my energy throughout the day.",
    "My laptop refused to turn on before my presentation.",
    "A dead battery can ruin someone's entire work schedule.",
    "Museums preserve important historical and cultural artifacts.",
    "Ancient Egyptian statues reveal details about past civilisations.",
    "Chocolate cake always improves my mood after a stressful day.",
    "Too much sugar can make it difficult to focus during meetings.",
    "Airplanes rely on wings and lift to stay in the air.",
    "Birds use flapping and gliding to travel long distances.",
]

def main() -> None:
    print("=== LEVEL 1: EMBEDDINGS + NEAREST NEIGHBOURS ===")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Encode
    start = time.perf_counter()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    print(f"Encoded {len(sentences)} sentences in {time.perf_counter() - start:.2f}s")

    # Cosine similarity matrix
    similarity = embeddings @ embeddings.T

    lines_to_write = []
    for idx, sentence in enumerate(sentences):
        row = similarity[idx]

        # Exclude itself
        others = [
            (other_idx, score)
            for other_idx, score in enumerate(row)
            if other_idx != idx
        ]

        # Top K neighbours
        top_matches = sorted(others, key=lambda item: item[1], reverse=True)[:TOP_K]

        header = f"\nSentence [{idx}]: {sentence}"
        print(header)
        lines_to_write.append(header + "\n")

        for rank, (match_idx, score) in enumerate(top_matches, start=1):
            result_line = f"  #{rank} cosine={score:.3f} → [{match_idx}] {sentences[match_idx]}"
            print(result_line)
            lines_to_write.append(result_line + "\n")

    # Save results
    with open("nearest_neighbours.txt", "w") as f:
        f.write("=== LEVEL 1 — Nearest Neighbours ===\n")
        f.write("\n--- Sentences ---\n")
        for s in sentences:
            f.write(f"- {s}\n")
        f.write("\n--- Nearest Neighbours Results ---\n")
        f.writelines(lines_to_write)

        # Required cases
        f.write("\n\n=== REQUIRED CASES ===\n")
        f.write("1. Semantically similar but lexically different:\n")
        f.write("   Example: Sentence about jogging vs running.\n\n")
        f.write("2. Lexically similar but semantically far apart:\n")
        f.write("   Example: Sugar improving mood vs sugar harming focus.\n")

if __name__ == "__main__":
    main()
