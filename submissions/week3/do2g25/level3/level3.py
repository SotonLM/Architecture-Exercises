# Level 2 (save to text file)
from transformers import pipeline
import time
import pandas as pd

# Load different models
distilgpt2 = pipeline('text-generation', model='distilgpt2')
gpt2 = pipeline('text-generation', model='gpt2')
gpt2_medium = pipeline('text-generation', model='gpt2-medium')

models = [
    distilgpt2,
    gpt2,
    gpt2_medium
]

# Generate text
prompts = [
    "The most boring topic in computer science is",
    "Artificial Intelligence will lead to",
    "The most pivotal technology from the past 50 years is",
    "Reinforcement learning is best suited",
    "Newton's third law is defined as"
]


def generate_output(prompt, model):
    start_time = time.time()
    output = model(prompt) # Use default settings
    return output[0]["generated_text"], time.time() - start_time


results = []

for prompt in prompts:
    print(f"Prompt: {prompt}")
    prompt_results = {"prompt": prompt}
    for model in models:
        generated_text, time_taken = generate_output(prompt, model)
        model_name = model.model.config._name_or_path

        print(f"Model: {model_name}")
        print(f"Generated: {generated_text}")
        print(f"Time taken: {time_taken}\n")
        print("-" * 50)

        prompt_results[model_name] = time_taken
    print("+" * 100)
    results.append(prompt_results)

df = pd.DataFrame(results)
print(df.to_string(index=False))

print("\nAverage time each model took across all prompts:")
for col in df.columns:
    if col != 'prompt':
        avg_time = df[col].mean()
        print(f"{col}: {avg_time:.4f}s")
