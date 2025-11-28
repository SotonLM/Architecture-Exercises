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
    "The secret to happiness is",
    "meaning  of life is",
    "how emotions adds up"
]
with open("result.txt","w",  encoding="utf-8") as file:
    
    parameter_sets = [
        {"max_length": 50, "temperature": 1.0, "top_k": 50},
        {"max_length": 20, "temperature": 0.5, "top_k": 10},
        {"max_length": 100, "temperature": 1.5, "top_k": 100},

    ]
    
    for param_set in parameter_sets:
        file.write(f"PARAMETERS: max_length={param_set['max_length']}, "
                  f"temperature={param_set['temperature']}, top_k={param_set['top_k']}\n")
    
        for prompt in prompts:
            start_time = time.time()
            
            output = generator(prompt, max_length=param_set['max_length'],
                temperature=param_set['temperature'],
                top_k=param_set['top_k'])

            end_time = time.time()
            generatetime = end_time - start_time
            token = len(output[0]['generated_text'].split())
            
            file.write(f"\nPrompt: {prompt}")
            file.write(f"Generated: {output[0]['generated_text']}")
            file.write(f"Generation Time:{generatetime:.2f} seconds\n")
            file.write("-" * 50)

# LEVEL 2: Your code here
    
# TODO: Save to file
# TODO: Try different parameters
# TODO: Time generation
# TODO: Count tokens

# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them?
# TODO: Implement your challenge

# LEVEL 4: Your code here
# TODO: Build something new