from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

prompts = ["The future of AI is", "In the year 2030", "The secret to happiness is"]

for prompt in prompts:
    output = generator(prompt, max_length=30, num_return_sequences=1)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}\n")
    print("-" * 50)
# This script uses the Hugging Face Transformers library to generate text based on given prompts. 
# It initializes a text generation pipeline with the 'distilgpt2' model and generates text for each prompt in the list. The generated text is printed to the console.