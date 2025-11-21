from transformers import pipeline
import time

# Load model
generator = pipeline("text-generation", model="distilgpt2")

# 5 different prompts
prompts = [
    "The future of AI is",
    "In the next 10 years, humans will",
    "The most important skill to learn is",
    "If I could travel anywhere, I would go to",
    "Creativity means"
]

# Values to test
max_lengths = [20, 50, 100]
temperatures = [0.5, 1.0, 1.5]
top_ks = [10, 50, 100]

# File to save outputs
output_path = "results.txt"

with open(output_path, "w") as f:
    f.write("=== LEVEL 2 RESULTS ===\n")

    for prompt in prompts:
        f.write(f"\n\n### PROMPT: {prompt}\n")

        for ml in max_lengths:
            for temp in temperatures:
                for k in top_ks:

                    # Start timer
                    start_t = time.time()

                    # Generate text
                    out = generator(
                        prompt,
                        max_length=ml,
                        temperature=temp,
                        top_k=k,
                        num_return_sequences=1
                    )

                    text = out[0]["generated_text"]
                    elapsed = time.time() - start_t

                    # Count tokens
                    token_count = len(generator.tokenizer.encode(text))

                    # Print to screen
                    print(f"\nPrompt: {prompt}")
                    print(f"Settings → max_length={ml}, temp={temp}, top_k={k}")
                    print("Generated:", text)
                    print("Tokens:", token_count)
                    print("Time:", round(elapsed, 4), "seconds")

                    # Save to file
                    f.write(f"\n--- SETTINGS: max_length={ml}, temp={temp}, top_k={k} ---\n")
                    f.write(f"Generated: {text}\n")
                    f.write(f"Token count: {token_count}\n")
                    f.write(f"Time: {elapsed:.4f} seconds\n")

print("\nAll results saved to results.txt!")
