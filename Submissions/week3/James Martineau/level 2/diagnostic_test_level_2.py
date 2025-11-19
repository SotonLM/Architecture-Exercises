from transformers import pipeline
import time

generator = pipeline('text-generation', model='distilgpt2')

prompts = [
    "The future of AI is",
    "In the year 2030",
    "The secret to happiness is",
    "Once upon a time in a land far away,",
    "The way to become a billionaire is",

]
with open("generation_results.txt", "w",encoding='utf-8') as f: # automatically closes the file after writing
    for prompt in prompts:
        start_time = time.time() # Start the timer to measure the generation time
        output = generator(prompt, min_new_tokens=100,max_new_tokens=100, num_return_sequences=1, do_sample=True, num_beams=1, temperature = 0.5, top_k = 50,top_p = 0.92, repetition_penalty = 1.075) # Generate text with specified parameters
        end_time = time.time() # stop the timer once generation is complete
        generation_time = end_time - start_time #calculate generation time
        print(output)
        token_count = len(generator.tokenizer.encode(output[0]['generated_text'])) # Count the number of tokens in the generated text
        # Save results to file
        f.write(f"\nPrompt: {prompt}\n")
        f.write(f"Generated: {output[0]['generated_text']}\n")
        f.write(f"Generation Time: {generation_time:.4f} seconds\n")
        f.write(f"Token Count: {token_count}\n")
        f.write("-" * 50 + "\n")