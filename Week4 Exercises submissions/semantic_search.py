import time
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")

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
    # Space 
    "The Moon's surface is covered in fine dust left by billions of years of meteor impacts.",
    "Astronauts aboard the ISS experience microgravity, causing their muscles to weaken over time.",
    "Telescopes allow scientists to observe distant galaxies and study their formation.",
    "A supernova occurs when a massive star collapses and releases an enormous burst of energy.",
    "Mars has polar ice caps and seasons similar to Earth, but its thin atmosphere makes it colder.",
    "The James Webb Space Telescope captures infrared light from early galaxies.",

    # Cooking 
    "Simmering tomatoes slowly brings out their natural sweetness and deepens the flavour of sauces.",
    "Proper knife skills make cooking faster, safer, and more enjoyable.",
    "Fresh herbs like basil and parsley brighten dishes when added at the end of cooking.",
    "Bread dough rises because yeast produces carbon dioxide when it feeds on sugars.",
    "Grilling vegetables adds smoky flavour while preserving nutrients.",
    "Marinating chicken overnight helps tenderise the meat and enhance flavour.",

    # Fitness & Health
    "Strength training increases muscle mass and boosts metabolism over time.",
    "Running long distances requires both cardiovascular endurance and efficient breathing.",
    "High-intensity interval training alternates short bursts of effort with recovery periods.",
    "A balanced diet rich in protein helps muscles recover after exercise.",
    "Proper stretching improves flexibility and reduces the risk of injury.",
    "Cycling is a low-impact exercise that strengthens the legs and improves heart health.",

    # Technology
    "Machine learning algorithms identify patterns in data to make predictions.",
    "Smartphones rely on energy-efficient chips to maximise battery life.",
    "Cybersecurity experts monitor networks to detect potential intrusions.",
    "Cloud computing allows companies to scale their services without buying extra hardware.",
    "Quantum computers use qubits to perform computations beyond the capabilities of classical machines.",
    "Virtual reality headsets immerse users in 3-D digital environments."
]

def main():
    output_file = "/Users/yannis/Desktop/Ai proj/Architecture-Exercises/week4/level2/search_examples.txt"
    with open(output_file, "w") as f:
        f.write("Corpus: \n")
        for i in corpus:
            f.write(f"{i}\n")

        f.write("\n")

        f.write("=== LEVEL 2: MINI SEMANTIC SEARCH ENGINE ===\n")
        f.write(f"Loading model: {MODEL_NAME}\n")
        model = SentenceTransformer(MODEL_NAME)

        start = time.perf_counter()
        embeddings = model.encode(corpus, normalize_embeddings=True, device='cpu')
        print(f"Encoded {len(corpus)} corpus in {time.perf_counter() - start:.2f}s")

        while True:
            query = input("Enter your search query(or exit): ").strip()
            if query.lower() == "exit":
                break
            query_start = time.perf_counter()
            query_embeddings = model.encode([query], normalize_embeddings=True, device='cpu')[0]
            query_time = time.perf_counter() - query_start

            scores = embeddings @ query_embeddings

            top_matches = np.argsort(scores)[::-1][:TOP_K]
            print(f"Query embedding time: {query_time:.2f}s\n")

            for rank, idx in enumerate(top_matches, start=1):
                snippet = corpus[idx][:120]
                score = scores[idx]
                print(f"#{rank} (doc {idx}) | score = {score:.3f}")
                print(f"{snippet}\n")

                f.write(f"#{rank} (doc {idx}) | score = {score:.3f}\n")
                f.write(f"{snippet}\n")

            f.write("\n")

            f.write("Reflections:\n")
            f.write("It took me a while to understands what the code does and which variables represent what.")
            f.write("I found building the snippet hardest.")
            f.write("I wasn't able to run this piece of code on my mac due to a bus error, I tried to fix it but it doesn't work.")
            f.write("Hopefully, it will work as normal when this piece of code is ran on windows")

if __name__ == "__main__":
    main()