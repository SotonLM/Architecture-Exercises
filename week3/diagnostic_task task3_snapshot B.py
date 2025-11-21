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


#measures how long it takes until the generation tails with repetition
def Repitionscore(text):
    score = 100
    c = 1
    rep = 1
    maxrep = 1
    while (rep < 2 and c < len(text)):
        test = text[-1:-1-c:-1][::-1] #substring to test against
        conflict = False
        p = 1 #period of repitition
        rep = 1 #number of repititions
        while(not conflict):
            if text[-p*c - 1: - p*c - 1 - c:-1][::-1] == test:
                p += 1
                rep += 1
            else:
                maxrep = max(maxrep, rep)
                conflict = True
        c += 1
    if (maxrep > 1):
        s = "".join(text.split(test))
        score = len(s)/len(text)
    return score


# LEVEL 1: Basic generation
print("=== LEVEL 1: BASIC GENERATION ===")
models = ['distilgpt2', 'gpt2', 'gpt2-medium']
generators = [pipeline('text-generation', model='distilgpt2'),pipeline('text-generation', model='gpt2'),pipeline('text-generation', model='gpt2-medium')]

prompts = [
    "apple",
    "apple apple",
    "apple apple apple",
    "My favoutite food is",
    "1 + 1 = 2",
    "asjfjasfopja",
    "1234567890",
    "",
    "                               ",
    "User: What is the capital of France?\nAgent: The capital of France is " 
]


filesavedata = {}
for model in models:
    filesavedata[model] = {}

for m in models:
    for prompt in prompts:
        generator = pipeline('text-generation', model = m)
        starttime = time.time()
        output = generator(prompt, max_length=10, temperature = 0.6, top_k = 50)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output[0]['generated_text'].strip()}")
        print("-" * 50)
        tokens = len(generator.tokenizer.encode(output[0]['generated_text']))
        s = Repitionscore(output[0]['generated_text'])
        print("Score:", s)
        filesavedata[m][prompt] =  (output[0]['generated_text'], time.time() - starttime, tokens, s)

with open("results.txt", "w", encoding="utf-8") as file:
    for model,v in filesavedata.items():
        n = 0
        s = 0
        for prompt, data in v.items():
            n += 1
            s += data[3]
            file.write(f"Prompt:{prompt}\nGenerated with {model}:\n{data[0].strip()}\nTime to generate: {data[1]}\nNumber of tokens: {data[2]}\nThe repetition score: {data[3]}\n\n\n")
        file.write(f"\n=================\nAverage score for {model} = {s/n}\n============\n")


# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them?
# TODO: Implement your challenge

# TODO: check each model for repetition and store results in txt

# LEVEL 4: Your code here
# TODO: Build something new