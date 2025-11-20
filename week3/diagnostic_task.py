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

def Repitionscore(text):
    # find % along text till repition
    return 0

# LEVEL 1: Basic generation
print("=== LEVEL 1: BASIC GENERATION ===")
models = ['distilgpt2', 'gpt2', 'gpt2-medium']
generators = [pipeline('text-generation', model='distilgpt2'),pipeline('text-generation', model='gpt2'),pipeline('text-generation', model='gpt2-medium')]

prompts = [
    "apple apple"
]

filesavedata = {}
for prompt in prompts:
    filesavedata[prompt] = {}

for m in models:
    for prompt in prompts:
        generator = pipeline('text-generation', model = m)
        starttime = time.time()
        output = generator(prompt, max_length=30, temperature = 0.3, top_k = 50)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output[0]['generated_text'].strip()}")
        print("-" * 50)
        tokens = len(generator.tokenizer.encode(output[0]['generated_text']))
        filesavedata[prompt][m] =  (output[0]['generated_text'], time.time() - starttime, tokens)

with open("results.txt", "w") as file:
    for k,v in filesavedata.items():
        for model, data in v.items():
            file.write(f"Prompt:{k}\nGenerated with {model}:\n{data[0].strip()}\nTime to generate: {data[1]}\nNumber of tokens{data[2]}\n\n\n")



# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them?
# TODO: Implement your challenge

# TODO: check each model for repetition and store results in txt

# LEVEL 4: Your code here
# TODO: Build something new