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
start_time = time.time()
generator = pipeline('text-generation', model='distilgpt2')

prompts = [
    "When I was little",
    "1995 is amazing because",
    "why would he know if "
    "when I say hey you say"
    "merry christmas for the"
]

#for prompt in prompts:
    #output = generator(prompt, max_length=30)
    #print(f"\nPrompt: {prompt}")
    #print(f"Generated: {output[0]['generated_text']}")
    #print("-" * 50)

# LEVEL 2: Your code here

with open('results1.txt','w', encoding="utf-8")as f:
    for prompt in prompts:
        output = generator(prompt, max_length=30)
        generated_text = output[0]['generated_text']

        
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated: {generated_text}\n")
        f.write("-" * 50 + "\n")

end_time = time.time()
print(f"time took for generation is {start_time-end_time:.4f} seconds")
#,temperature = 0.5,top_k = 100


# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them?
# TODO: Implement your challenge

# LEVEL 4: Your code here
# TODO: Build something new