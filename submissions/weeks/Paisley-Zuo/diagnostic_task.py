"""
DIAGNOSTIC TASK - Complete as many levels as you can

LEVEL 1: Get this working (required)
LEVEL 2: Expand it (tests Python skills)
LEVEL 3: Pick a challenge (tests thinking)
LEVEL 4: Build something new (tests creativity)

DUE: Wednesday 11:59 PM
Submit via: GitHub PR (preferred) or Teams #architecture channel.
See submission_format.txt for details.
"""

from transformers import pipeline
import time

# LEVEL 1: Basic generation
print("=== LEVEL 1: BASIC GENERATION ===")
generator = pipeline('text-generation', model='distilgpt2')

prompts = [
    "The future of AI is",
    "In the year 2030",
    "The secret to happiness is"
]

for prompt in prompts:
    output = generator(prompt, max_length=30)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}")
    print("-" * 50)

# LEVEL 2: Your code here
# TODO: Save to file
# TODO: Try different parameters
# TODO: Time generation
# TODO: Count tokens
print("LEVEL 2 IS RUNNING")
# Try different prompts
level2_prompts = [
    "AI will change the world because",
    "One important life lesson is",
    "If humans lived on Mars,",
    "My favourite childhood memory is",
    "In 100 years, the world will"
]
max_lengths = [20, 50, 100]
temperatures = [0.5, 1.0, 1.5]
top_ks = [10, 50, 100]
with open("results.txt", "w", encoding="utf-8") as f:

    for prompt in level2_prompts:
        f.write(f"\n==============================\n")
        f.write(f"PROMPT: {prompt}\n")

        for ml in max_lengths:
            for temp in temperatures:
                for topk in top_ks:

                    f.write(f"\n--- Params: max_length={ml}, temperature={temp}, top_k={topk} ---\n")

                    start_time = time.time()

                    # generate text
                    output = generator(
                        prompt,
                        max_length=ml,
                        temperature=temp,
                        top_k=topk
                    )

                    generated_text = output[0]["generated_text"]
                    elapsed = time.time() - start_time

                    # Count tokens
                    token_count = len(generator.tokenizer.encode(generated_text))

                    # Save results into file
                    f.write(f"Generated text:\n{generated_text}\n")
                    f.write(f"Time taken: {elapsed:.2f} seconds\n")
                    f.write(f"Token count: {token_count}\n")


# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them? Pick B
# TODO: Implement your challenge

# LEVEL 4: Your code here
# TODO: Build something new