from transformers import pipeline

import time


generator = pipeline('text-generation', model='distilgpt2')

prompts = ["The future of Quantum Computing", "In the year 3000", "The secret to success is","Once upon a time in a land far away", "The key to success in life"]

with open('results.txt', 'w', encoding='utf-8') as f:
    f.write("Model generation results:\n")

start_time = time.time()
for prompt in prompts:
    output = generator(prompt, max_new_tokens=1000, num_return_sequences=1, temperature=0.5, top_k=50)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}\n")
    text = output[0]['generated_text']
    token_count = len(generator.tokenizer(text)['input_ids'])
    print(f"Token number: {token_count}")
    print("-" * 20)
    with open('results.txt', 'a', encoding='utf-8') as f:
        f.write(f"\nPrompt: {prompt}\n")
        f.write(f"Generated: {text}\n")
        f.write(f"Token number: {token_count}\n")
        f.write("-" * 20 + "\n")
end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds. Results saved to results.txt")
with open('results.txt', 'a', encoding='utf-8') as f:
    f.write(f"Total time taken: {end_time - start_time} seconds.\n")
# This script uses the Hugging Face Transformers library to generate text based on given prompts. 
# It initializes a text generation pipeline with the 'distilgpt2' model and generates text for each prompt in the list. The generated text is printed to the console.