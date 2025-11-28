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
print("\n=== LEVEL 2: EXPANDED GENERATION ===")
file = "results2.txt"

def experiment(prompt, max_length=50, temperature=1.0, top_k=50):
    start_time = time.time()
    output = generator(prompt, max_length=max_length, temperature=temperature, top_k=top_k)
    stop_time = time.time()-start_time

    token_count = len(generator.tokenizer.encode(output[0]['generated_text']))

    with open(file, "a") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Max Length: {max_length}, Temperature: {temperature}, Top K: {top_k}\n")
        f.write(f"Generated: {output[0]['generated_text']}\n")
        f.write(f"Time taken: {stop_time:.4f} seconds\n")
        f.write("-" * 50 + "\n")

my_prompts = [
    "If cats could talk, they would say",
    "In a cyberpunk city, the rain tastes like",
    "The secret ingredient to the best coffee is",
    "Python is popular because"
    "The most important skill in coding is",
]

for p in my_prompts:
    experiment(p)


test_prompt = "Artificial Intelligence will change the world by"

experiment(test_prompt, max_length=20)
experiment(test_prompt, max_length=100)
experiment(test_prompt, temperature=0.5)
experiment(test_prompt, temperature=1.5)

# LEVEL 3: Your code here
print("\n=== LEVEL 3A: CHALLENGE GENERATION ===")

def experiment(failure_name, prompt, max_length=50, temperature=1.0, top_k=50, do_sample=True):
    start_time = time.time()
    
    output = generator(
        prompt, 
        max_length=max_length, 
        temperature=temperature, 
        top_k=top_k, 
        do_sample=do_sample,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    stop_time = time.time() - start_time

    generated_text = output[0]['generated_text']


    print(f"Running Test: {failure_name}")
    print(f"Output: {generated_text[:100]}...\n")


experiment(
    "Infinite Repetition",
    "The definition of insanity is",
    max_length=100,
    do_sample=False
)

experiment(
    "Coherence Collapse",
    "My favorite color is",
    max_length=50,
    temperature=3.5,
    do_sample=True
)

logic_prompt = "John is in the kitchen. Mary is in the garden. John goes to the garden. Mary goes to the kitchen. Where is John? Answer:"
experiment(
    "Logic Failure",
    logic_prompt,
    max_length=60,
    temperature=0.7
)

experiment(
    "Sentence Fragmentation",
    "The 5 steps to success are: 1.",
    max_length=20
)

experiment(
    "Punctuation Confusion",
    "!!!!!!!????????........",
    max_length=50
)

# LEVEL 4: Your code here
# TODO: Build something new