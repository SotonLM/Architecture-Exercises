from transformers import pipeline
import time

# Load a small model
generator = pipeline('text-generation', model='distilgpt2')

# Open file for writing results
with open('results.txt', 'w', encoding='utf-8') as f:
    # Generate text
    prompts = [
        "The future of AI is",
        "In the year 2030",
        "The secret to happiness is",
        "The best soccer player of all time is",
        "The best way to get good at clash royale is"
    ]
   
    # Parameters
    max_lengths = [20, 50, 100]
    temperatures = [0.5, 1.0, 1.5]
    top_ks = [10, 50, 100]
   
    # Test different parameter combinations
    for max_len in max_lengths:
        for temp in temperatures:
            for top_k in top_ks:
                f.write(f"\n{'='*70}\n")
                f.write(f"Testing with max_length={max_len}, temperature={temp}, top_k={top_k}\n")
                f.write(f"{'='*70}\n\n")
                print(f"\nTesting with max_length={max_len}, temperature={temp}, top_k={top_k}")
               
                for prompt in prompts:
                    start_time = time.time()
                    output = generator(
                        prompt,
                        max_length=max_len,
                        temperature=temp,
                        top_k=top_k,
                        num_return_sequences=1
                    )
                    generation_time = time.time() - start_time
                   
                    generated_text = output[0]['generated_text']
                   
                    token_count = len(generator.tokenizer.encode(generated_text))
                   
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Generated: {generated_text}\n")
                    f.write(f"Time taken: {generation_time:.4f} seconds\n")
                    f.write(f"Token count: {token_count}\n")
                    f.write("-" * 50 + "\n\n")
                   
                    print(f"Prompt: {prompt}")
                    print(f"Generated: {generated_text}")
                    print(f"Time: {generation_time:.4f}s, Tokens: {token_count}")
                    print("-" * 50)