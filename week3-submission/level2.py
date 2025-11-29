from transformers import pipeline, AutoTokenizer
import time

# LEVEL 1: Basic generation
print("=== LEVEL 1: BASIC GENERATION ===")
start_time = time.time()
generator = pipeline('text-generation', model='distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

prompts = [
    "When I was little",
    "1995 is amazing because",
    "why would he know if "
    "when I say hey you say"
    "merry christmas for the"
]

#for prompt in prompts:
    #output = generator(prompt, max_length=30)
    #print(f"\nPrompt: {prompt}")
    #print(f"Generated: {output[0]['generated_text']}")
    #print("-" * 50)

# LEVEL 2: Your code here

with open('results2.txt','w', encoding="utf-8")as f:
    for prompt in prompts:
        output = generator(prompt, max_length=50,temperature = 0.5,top_k = 100)
        generated_text = output[0]['generated_text']

        
        total_tokens = len(tokenizer.encode(generated_text))
        prompt_tokens = len(tokenizer.encode(prompt))
        generated_tokens = total_tokens - prompt_tokens


        
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated: {generated_text}\n")
        f.write(f"Total tokens: {total_tokens}\n")
        f.write(f"Generated tokens only: {generated_tokens}\n")
        f.write("-" * 50 + "\n")

end_time = time.time()
print(f"time took for generation is {end_time-start_time:.4f} seconds")
#,temperature = 0.5,top_k = 100
