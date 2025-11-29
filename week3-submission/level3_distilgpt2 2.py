from transformers import pipeline, AutoTokenizer
import time
 

print("distilgpt2 model used")
start_time = time.time()
generator = pipeline('text-generation', model='distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

prompts = [
    "how many days are in a month?",
    "sum up how Tom holland looks?",
    "i wanna be famous"
]

with open('results_distilgpt2.txt','w', encoding="utf-8")as f:
    for prompt in prompts:
        output = generator(prompt, max_length=50)
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
