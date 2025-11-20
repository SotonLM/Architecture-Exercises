from transformers import pipeline, AutoTokenizer
import time
import itertools
import os

# Load a small model and tokenizer
generator = pipeline('text-generation', model='distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Prompts to try
prompts = [
    "The difference between a Demo and a Prototype is",
    "Top 5 team sports are",
    "The secret to happiness is",
    "A simple recipe for pankakes:",
    "Top tips for learning MATLAB quickly:" 
]

# Parameter grids
max_lengths = [20, 50, 100]
temperatures = [0.5, 1.0, 1.5]
top_ks = [10, 50, 100]

# Output file (same directory as this code file)
output_file = os.path.join(os.path.dirname(__file__), 'results.txt')

def count_tokens(text: str) -> int:
    ids = tokenizer(text)['input_ids'] # Tokenize the text and get input IDs
    return len(ids)

def run_all_and_save():
    total_runs = 0
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write('Generation results\n')
        out.write('=' * 80 + '\n') # Header line
        for max_len, temp, topk in itertools.product(max_lengths, temperatures, top_ks): # Use all combinations
            out.write(f"Parameters: max_length={max_len}, temperature={temp}, top_k={topk}\n")
            out.write('-' * 80 + '\n') # Sub-header line
            for prompt in prompts:
                total_runs += 1
                start = time.time()
                output = generator(prompt, max_length=max_len, temperature=temp, top_k=topk, num_return_sequences=1) # Run generation
                elapsed = time.time() - start
                generated = output[0]['generated_text']
                tokens = count_tokens(generated)

                # Write results
                out.write(f"Prompt: {prompt}\n")
                out.write(f"Generated: {generated}\n")
                out.write(f"Tokens: {tokens}\n")
                out.write(f"Time (s): {elapsed:.4f}\n")
                out.write('-' * 40 + '\n') # Separator line
                # Console log
                print(f"Done: prompt='{prompt[:30]}...' len={tokens} tokens time={elapsed:.3f}s")
        out.write(f"Total generations: {total_runs}\n")

if __name__ == '__main__':
    print(f"Writing generation outputs to results.txt")
    print("")
    run_all_and_save()
    print("Done! See results.txt.")