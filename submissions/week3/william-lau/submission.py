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

from transformers import pipeline, GPT2TokenizerFast
import time

# LEVEL 1: Basic generation
""" print("=== LEVEL 1: BASIC GENERATION ===")
generator = pipeline('text-generation', model='distilgpt2')

prompts = [
    "The future of AI is",
    "In the year 2030",
    "The secret to happiness is"
]

for prompt in prompts:
    output = generator(prompt, max_length=30, num_return_sequences=1)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}\n")
    print("-" * 50) """

# LEVEL 2: Your code here
# TODO: Save to file
# TODO: Try different parameters
# TODO: Time generation
# TODO: Count tokens

import time

generator = pipeline('text-generation', model='distilgpt2')
prompts = [
    "The future of Lebron is",
    "If Bronny James can",
    "But only if Bryce James is able to",
    "The NBA will become",
    "In 2030, the Lakers will",
]

""" - Try `max_length` = 20, 50, 100
    - Try `temperature` = 0.5, 1.0, 1.5
    - Try `top_k` = 10, 50, 100 """

paramsList: list[tuple[int,int,int]] = [(20, 0.5, 10), (50, 1.0, 50), (100, 1.5, 100)] 
tokeniser = GPT2TokenizerFast.from_pretrained("distilgpt2")

with open("result.txt", "w") as f:
    count = 1
    absoluteStart = time.time()
    for prompt in prompts:
        for params in paramsList:
            length, temp, top_k = params
            start = time.time()
            output = generator(prompt, max_length=length, temperature=temp, top_k=top_k, num_return_sequences=1)
            end = time.time()
            
            tokens = tokeniser.encode(output[0]["generated_text"])
            token_count = len(tokens)

            f.write(f"Output {count}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Parameters: max_length={length}, temperature={temp}, top_k={top_k}\n")
            f.write(f"Time taken: {end - start} seconds\n")
            f.write(f"Result: {output}\n\n")
            count += 1

    print(f"Time taken for generation: {time.time() - absoluteStart} seconds")

# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them?
# TODO: Implement your challenge

# LEVEL 4: Your code here
# TODO: Build something new