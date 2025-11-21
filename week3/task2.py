from transformers import pipeline
import time, unicodedata, re

# Load a small model
generator = pipeline('text-generation', model='distilgpt2')

# Generate text
prompts = [
    # "The future of AI is",
    # "In the year 2030",
    # "The secret to happiness is"
    "My name is Bob",
    "The world will end because",
    "To fix a car, check its",
    "How to",
    "The meaning of humanity"

]

max_lengths = [20, 50, 100]
temperatures = [0.5, 1, 1.5]
top_ks = [10, 50, 100]

def clean_text(text):
    # remove unicode control chars
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

    # normalize fancy quotes to plain ones
    replacements = {
        '“': '"', '”': '"', '„': '"', '‟': '"',
        '‘': "'", '’': "'",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # collapse more than 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


f = open('myresults.txt', 'w', encoding="utf-8")
for prompt in prompts:
    for length in max_lengths:
        for temps in temperatures:
            for k in top_ks:
                start = time.time()
                output = generator(prompt, max_length=length, num_return_sequences=1, temperature=temps, top_k=k)
                end = time.time()
                print(f"Took {end-start} seconds")
                print(f"Used {len(generator.tokenizer.encode(text=(output[0]['generated_text'])))} tokens")
                print(f"\nPrompt: {prompt}")
                print(f"Generated: {output[0]['generated_text']}\n")
                print("-" * 50)
                f.write(f"Original prompt: {prompt}\n")
                cleaned = clean_text(output[0]['generated_text'])
                f.write(f"Generated: {cleaned}\n")
                f.write(f"Parameters: max_length={length}, temperature={temps}, top_k={k}\n")

f.close()
