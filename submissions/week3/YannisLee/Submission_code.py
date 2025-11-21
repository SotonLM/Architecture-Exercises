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

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ENABLE"] = "0"

from transformers import pipeline
import time

# LEVEL 1: Basic generation
print("=== LEVEL 1: BASIC GENERATION ===")

generator = pipeline("text-generation", model="distilgpt2", device="cpu",
    pad_token_id=50256)

prompts = [
    "The future of AI is",
    "In the year 2030",
    "The secret to happiness is"
]

for prompt in prompts:
    output = generator(prompt, max_new_tokens=30, truncation=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}")
    print("-" * 50)

# LEVEL 2: Your code here
# TODO: Save to file
# TODO: Try different parameters
# TODO: Time generation
# TODO: Count tokens
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ENABLE"] = "0"

from transformers import pipeline, AutoTokenizer
import time

# LEVEL 1: Basic generation
with open("results.txt", "w") as file:
    file.write("=== LEVEL 1: BASIC GENERATION === \n\n")
    generator = pipeline('text-generation', model='distilgpt2', device="cpu",
        pad_token_id=50256)
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    prompts = [
        "The best place to travel to in summer is",
        "Happiness comes from",
        "The most delicious dessert I've ever had was",
        "Running a Marathon needs a lot of practise because",
        "The best music band in the world is"
    ]

    start_time = time.time()

    for prompt in prompts:
        output = generator(prompt, max_new_tokens=50, temperature = 1.5,
                            top_k = 50, truncation=True)
        file.write(f"\nPrompt: {prompt}")
        file.write(f"Generated: {output[0]['generated_text']}")
        file.write("-" * 100 + "\n")
        generated_text = output[0]['generated_text']

    end_time = time.time()

    total_token = len(tokenizer.encode(generated_text))
    prompt_token = len(tokenizer.encode(prompt))
    generized_token = total_token - prompt_token 
    file.write(f"Total tokens generated {generized_token}\n")

    total_time = end_time - start_time
    file.write(f"The generation took {total_time}")

# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them?
# TODO: Implement your challenge

#Task A 

#Repeat itself endlessly
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ENABLE"] = "0"

from transformers import pipeline, AutoTokenizer
import time

# LEVEL 1: Basic generation
with open("results.txt", "w") as file:
    file.write("=== LEVEL 1: BASIC GENERATION === \n\n")
    generator = pipeline('text-generation', model='distilgpt2', device="cpu",
        pad_token_id=50256)
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    prompts = [
        "The best place to travel to in summer is",
        "Happiness comes from",
        "The most delicious dessert I've ever had was",
        "Running a Marathon needs a lot of practise because",
        "The best music band in the world is"
    ]

    start_time = time.time()

    while True:
        for prompt in prompts:
            output = generator(prompt, max_new_tokens=50, temperature = 1.5,
                            top_k = 50, truncation=True)
            file.write(f"\nPrompt: {prompt}")
            file.write(f"Generated: {output[0]['generated_text']}")
            file.write("-" * 100 + "\n")
        generated_text = output[0]['generated_text']

    end_time = time.time()

    total_token = len(tokenizer.encode(generated_text))
    prompt_token = len(tokenizer.encode(prompts[i]))
    generized_token = total_token - prompt_token 
    file.write(f"Total tokens generated {generized_token}\n")

    total_time = end_time - start_time
    file.write(f"The generation took {total_time}")

#Generate Nonsense/ gibberish
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ENABLE"] = "0"

from transformers import pipeline, AutoTokenizer
import time

# LEVEL 1: Basic generation
with open("results.txt", "w") as file:
    file.write("=== LEVEL 1: BASIC GENERATION === \n\n")
    generator = pipeline('text-generation', model='distilgpt2', device="cpu",
        pad_token_id=50256)
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    prompts = [
        "Hi Blue Royal Left",
        "What logical hey",
        "Fine art basketball",
        "Phone Table wires because",
        "Cream Yellow Red in the world is"
    ]

    start_time = time.time()

    for prompt in prompts:
        output = generator(prompt, max_new_tokens=50, temperature = 2.5,
                            top_k = 200, repetition_penalty = 1.0, truncation=True)
        file.write(f"\nPrompt: {prompt}")
        file.write(f"Generated: {output[0]['generated_text']}")
        file.write("-" * 100 + "\n")
        generated_text = output[0]['generated_text']

    end_time = time.time()

    total_token = len(tokenizer.encode(generated_text))
    prompt_token = len(tokenizer.encode(prompt))
    generized_token = total_token - prompt_token 
    file.write(f"Total tokens generated {generized_token}\n")

    total_time = end_time - start_time
    file.write(f"The generation took {total_time}")

# LEVEL 4: Your code here
# TODO: Build something new