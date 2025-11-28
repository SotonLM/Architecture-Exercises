'''
For Option B, I used perplexity,
relevance, repitition and length.
These are all used to create a weighted average for the quality of the generated text.

I've stored the quality score of each output inside a DataFrame as well,
making it easier to compare the score betweeen all 10 prompts. 
'''

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from evaluate import load
import pandas as pd

perplexity = load("perplexity")
embed = SentenceTransformer("all-mpnet-base-v2")

# Counts the number of times a word is repeated, then returns
# a score' 'for repetition
def repetition(text, n=2):
    words = text.split()
    if len(words) < n: return 0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return 1 - len(set(ngrams))/ len(ngrams)

def length_score(text, ideal=25):
    return min(len(text.split()) / ideal, 1.0)

def relevance(prompt, text):
    p = embed.encode(prompt, convert_to_tensor=True)
    g = embed.encode(text, convert_to_tensor=True)
    return float((p @ g) / (p.norm() * g.norm()))

def quality_score(prompt, text):
    ppl = perplexity.compute(model_id="distilgpt2", predictions=[prompt])['perplexities'][0]
    f = 1 / ppl
    r = relevance(prompt, text)
    d = 1 - repetition(text)
    l  =length_score(text)

    score = (.45*r + .35*f + .15*d + .05*l) * 100
    return round(score, 2), {'relevance': r, "fluency": f, "diversity": d, "length": l}

# Load a small model
generator = pipeline('text-generation', model='distilgpt2')

# Generate text
prompts = [
    "The future of AI is",
    "In the year 2030",
    "The secret to happiness is",
    "My name is Bob",
    "The world will end because",
    "To fix a car, check its",
    "How to",
    "The meaning of humanity",
    "The quadratic formula is",
    "If I run really fast",
]

df = pd.DataFrame(index=prompts,columns=['score', 'relevance', 'fluency', 'diversity', 'length'])

for prompt in prompts:
    output = generator(prompt, max_length=30, num_return_sequences=1)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}\n")
    print("-" * 50)
    qscore, qualities = quality_score(prompt, output[0]['generated_text'])
    df.loc[prompt, ['score', 'relevance', 'fluency', 'diversity','length']] = [qscore,qualities['relevance'], qualities['fluency'], qualities['diversity'], qualities['length']]
    
print(df)