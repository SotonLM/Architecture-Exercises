from transformers import pipeline
import time
# Load a small model
generator = pipeline('text-generation', model='gpt2-medium')

# Generate text
prompts = [
    "Space is not infinite",
    "Pets are not happy",
    "Computers are not the future",
    "Only happy people are",
    "Only sad people are"
]
with open('week3/level3/results.txt', 'w') as f:
    for prompt in prompts:
        start_time = time.time()
        output = generator(prompt, max_length=100, num_return_sequences=1, temperature=1, top_k =50, top_p = 0.95)
        f.write(f'\ntime taken to generate response: {time.time()-start_time}\n')
        f.write(f"Prompt: {prompt}")
        f.write(f"Generated: {output[0]['generated_text']}\n")
        f.write(f'Number of tokens in reponse: {len(generator.tokenizer.encode(output[0]['generated_text']))}\n')
        f.write("-" * 50)