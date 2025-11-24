from transformers import pipeline
import time

# Load a small model
generator1 = pipeline('text-generation', model='distilgpt2')
generator2 = pipeline('text-generation', model='gpt2')
generator3 = pipeline('text-generation', model='gpt2-medium')

# Generate text
prompts = [
    "The secret to happiness is"
]

with open('results.txt', 'w', encoding='utf-8') as f:
    for prompt in prompts:
        start = time.time()
        output1 = generator1(prompt, max_length=100, num_return_sequences=1)
        elapsed = time.time() - start
        model_name = 'distilgpt2'
        print(f"\nPrompt: {prompt}")
        print(f"Model: {model_name}")
        print(f"Generated: {output1[0]['generated_text']}\n")
        print(f"Time taken: {elapsed:.4f} seconds")
        print("-" * 50)
        f.write(f"\nPrompt: {prompt}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Generated: {output1[0]['generated_text']}\n")
        f.write(f"Time taken: {elapsed:.4f} seconds\n\n")
        f.write("-" * 50 + "\n")


    for prompt in prompts:
        start = time.time()
        output2 = generator2(prompt, max_length=100, num_return_sequences=1)
        elapsed = time.time() - start
        model_name = 'gpt2'
        print(f"\nPrompt: {prompt}")
        print(f"Model: {model_name}")
        print(f"Generated: {output2[0]['generated_text']}\n")
        print(f"Time taken: {elapsed:.4f} seconds")
        print("-" * 50)
        f.write(f"\nPrompt: {prompt}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Generated: {output2[0]['generated_text']}\n")
        f.write(f"Time taken: {elapsed:.4f} seconds\n\n")
        f.write("-" * 50 + "\n")

    for prompt in prompts:
        start = time.time()
        output3 = generator3(prompt, max_length=100, num_return_sequences=1)
        elapsed = time.time() - start
        model_name = 'gpt2-medium'
        print(f"\nPrompt: {prompt}")
        print(f"Model: {model_name}")
        print(f"Generated: {output3[0]['generated_text']}\n")
        print(f"Time taken: {elapsed:.4f} seconds")
        print("-" * 50)
        f.write(f"\nPrompt: {prompt}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Generated: {output3[0]['generated_text']}\n")
        f.write(f"Time taken: {elapsed:.4f} seconds\n\n")
        f.write("-" * 50 + "\n")