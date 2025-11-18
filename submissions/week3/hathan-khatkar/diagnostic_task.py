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
    "Knock knock. Who's there?",
    "To make a sandwich, you need to"
]

for prompt in prompts:
    output = generator(prompt, max_length=30)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}")
    print("-" * 50)

# params
max_lengths = [20, 50, 100]
temperatures = [0.5, 1.0, 1.5]
top_ks = [10, 50, 100]

def generate(prompt, max_len, temp, k):
    start_time = time.time()

    output = generator(
        prompt,
        max_new_tokens=max_len, # changed from max_length to max_new_tokens as model overrides this parameter
        temperature=temp,
        top_k=k,
        do_sample=True
    )

    end_time = time.time()
    text = output[0]["generated_text"]

    token_count = len(generator.tokenizer.encode(text))
    elapsed = end_time - start_time

    return text, token_count, elapsed


# LEVEL 2: Your code here
# TODO: Save to file
# TODO: Try different parameters
# TODO: Time generation
# TODO: Count tokens

def run_and_save(prompt, max_len, temp, k, run_id, outfile="result.txt"):
    """
    Runs a single generation using generate())
    and appends the result to a file instead of overwriting.
    """
    text, tokens, elapsed = generate(prompt, max_len, temp, k)

    with open(outfile, "a", encoding="utf-8") as f:
        f.write("\n============================\n")
        f.write(f"=== RUN RESULT {run_id} ===\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"max_length={max_len}, temperature={temp}, top_k={k}\n\n")
        f.write("Generated Text:\n")
        f.write(text + "\n\n")
        f.write(f"Token Count: {tokens}\n")
        f.write(f"Time: {elapsed:.4f}s\n")
        f.write("============================\n")

    print(f"Run appended to {outfile}")


# adjusting max tokens
run_and_save(prompts[4], 20, 1.0, 50, 1)
run_and_save(prompts[4], 50, 1.0, 50, 2)
run_and_save(prompts[4], 100, 1.0, 50, 3)

# adjusting temperature
run_and_save(prompts[4], 50, 0.5, 50, 4)
run_and_save(prompts[4], 50, 1.0, 50, 5)
run_and_save(prompts[4], 50, 1.5, 50, 6)

# adjusting top_k
run_and_save(prompts[4], 50, 1.0, 10, 7)
run_and_save(prompts[4], 50, 1.0, 50, 8)
run_and_save(prompts[4], 50, 1.0, 100, 9)

# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them?
# TODO: Implement your challenge



# LEVEL 4: Your code here
# TODO: Build something new