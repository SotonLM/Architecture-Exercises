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

# LEVEL 2
print("=== LEVEL 2 ===")

from transformers import pipeline
import time

generator = pipeline("text-generation", model="distilgpt2")

#5 different prompts
prompts = [
    "Computer science students often",
    "The secret to immortality is",
    "The secret to productivity is",
    "The future of computer sience is",
    "In the year 2050"
]

# open file 
with open("results.txt", "w") as f:
    for prompt in prompts:
        f.write("-" * 50 + "\n")
        start = time.time()
        output = generator(
            prompt,
            max_length = 50,
            temperature = 1.0,
            top_k = 50
        )

        end = time.time()

        text = output[0]["generated_text"]
        token_count = len(text.split())


        #save to file
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated: {text}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Time taken: {end - start:.2f}s | Tokens: {token_count}\n")
    f.write("-" * 50 + "\n")
        

print("\nDone")
