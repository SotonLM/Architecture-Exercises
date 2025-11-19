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

def level1Task():
    print("=== LEVEL 1: BASIC GENERATION ===")
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
        print("-" * 50)

#level1Task()

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

def level2Task():
    paramsList: list[tuple[int,int,int]] = [(20, 0.5, 10), (50, 1.0, 50), (100, 1.5, 100)] 
    tokeniser = GPT2TokenizerFast.from_pretrained("distilgpt2")

    with open("result.txt", "w", encoding="utf-8") as f:
        count = 1
        absoluteStart = time.time()
        for prompt in prompts:
            for params in paramsList:
                length, temp, top_k = params
                start = time.time()
                output = generator(prompt, max_new_tokens=length, temperature=temp, top_k=top_k, num_return_sequences=1)
                text = output[0]["generated_text"]
                cleanedText = "\n".join([line for line in text.splitlines() if line.strip() != ""])
                end = time.time()
                
                tokens = tokeniser.encode(cleanedText)
                tokenCount = len(tokens)

                f.write(f"Output {count}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Parameters: max_length={length}, temperature={temp}, top_k={top_k}\n")
                f.write(f"Time taken: {end - start} seconds\n")
                f.write(f"Result: {cleanedText}\n")
                f.write(f"Token Count: {tokenCount}\n\n")
                count += 1

        print(f"Time taken for generation: {time.time() - absoluteStart} seconds")

#level2Task()

# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them?
# TODO: Implement your challenge

# Option B - Measure Quality

def quality_score(prompt, text):
    words = text.strip().split()
    numWords = len(words)

    lengthScore = 0
    repetitionScore = 0
    punctuationScore = 0
    relevanceScore = 0

    if numWords == 0:
        return 0.0
    
    if numWords < 5:
        lengthScore = 0.2  
    elif numWords > 70:
        lengthScore = 0.3
    elif 8 <= numWords <= 40:
        lengthScore = 1.0
    else:
        lengthScore = 0.5

    uniqueWords = set(word.lower().strip(".,!?;:") for word in words)
    uniquenessRatio = len(uniqueWords) / numWords 

    if uniquenessRatio < 0.4:
        repetitionScore = 0.2 
    elif uniquenessRatio < 0.7:
        repetitionScore = 0.6
    elif uniquenessRatio < 1:
        repetitionScore = 0.8
    else:
        repetitionScore = 1.0

    end_char = text.strip()[-1]
    if end_char in [".", "!", "?"]:
        punctuationScore = 1.0
    else:
        punctuationScore = 0.3

    promptWords = set(word.lower().strip(".,!?;:") for word in prompt.split())
    overlap = promptWords.intersection(uniqueWords)

    if not promptWords:
        relevanceScore = 0.5 
    else:
        overlapRatio = len(overlap) / len(promptWords)
        if overlapRatio == 0:
            relevanceScore = 0.1
        elif overlapRatio < 0.5:
            relevanceScore = 0.6
        elif overlapRatio < 1:
            relevanceScore = 0.8
        else:
            relevanceScore = 1.0


    total = 0.2*lengthScore + 0.3*repetitionScore + 0.1*punctuationScore + 0.4*relevanceScore

    return (0.2*lengthScore*100, 0.3*repetitionScore*100, 0.1*punctuationScore*100, 0.4*relevanceScore*100, float(round(total * 100, 1)))

""" - Score text from 0-100
- Test on at least 10 generated texts
- Show examples of high-scoring vs low-scoring
- Explain: does your metric actually correlate with good text? """

def level3Task():
    paramsList: list[tuple[int,int,int]] = [(20, 0.5, 10), (50, 1.0, 50), (100, 1.5, 100)] 

    count = 1
    with open("level3-results.txt", "w", encoding="utf-8") as f:
        for prompt in prompts:
            for params in paramsList:
                length, temp, top_k = params
                output = generator(prompt, max_new_tokens=length, temperature=temp, top_k=top_k, num_return_sequences=1)
                text = output[0]["generated_text"]
                cleanedText = "\n".join([line for line in text.splitlines() if line.strip() != ""])

                lengthScore, repititionScore, punctuationScore, relevanceScore, totalScore = quality_score(prompt, cleanedText)

                f.write(f"Output {count}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Parameters: max_length={length}, temperature={temp}, top_k={top_k}\n")
                f.write(f"Result: {cleanedText}\n")
                f.write(f"Length Score: {lengthScore}\n")
                f.write(f"Repition Score: {repititionScore}\n")
                f.write(f"Punctuation Score: {punctuationScore}\n")
                f.write(f"Relevance Score: {relevanceScore}\n")
                f.write(f"Score: {totalScore}\n\n")

                count += 1

level3Task()

# LEVEL 4: Your code here
# TODO: Build something new