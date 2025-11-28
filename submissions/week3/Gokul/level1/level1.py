from transformers import pipeline


# Load a small model
generator = pipeline('text-generation', model='distilgpt2')

# Generate text
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
