# Level 2
from transformers import pipeline
import time

# Load a small model
generator = pipeline('text-generation', model='distilgpt2')

# Access the tokenizer from the pipeline
tokenizer = generator.tokenizer

# Generate text
prompts = [
    "The most boring topic in computer science is",
    "Artificial Intelligence will lead to",
    "The most pivotal technology from the past 50 years is",
    "Reinforcement learning is best suited",
    "Newton's third law is defined as"
]

for prompt in prompts:
  # Time how long generation takes
  start_time = time.time()
  output = generator(prompt, max_length=30, num_return_sequences=1)
  time_taken = time.time() - start_time

  generated_text = output[0]['generated_text']

  # Count tokens in the generated text
  tokens = tokenizer.encode(generated_text)
  token_count = len(tokens)

  print(f"\nPrompt: {prompt}")
  print(f"Generated: {generated_text}\n")
  print(f"Token count: {token_count}")
  print(f"Time taken: {time_taken}")
  print("-" * 50)

  # Write to text file
  with open("results.txt", "a") as f:
    f.write(f"Prompt: {prompt}\tGenerated: {generated_text}\n")
