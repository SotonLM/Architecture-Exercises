from transformers import pipeline
import time 
import os

generator = pipeline('text-generation', model='distilgpt2')
time_start = time.time()

prompts2 = [
    "Haskell best book is",
    "Artificial Intelligence impact is",
    "Quantittative finance is",
    "To sleep properly you should",
    "The meaning of life is"
]
if "results.txt" in os.listdir():
    os.remove("results.txt")

output_file = open("results.txt", "x")

for prompt in prompts2:
    output = generator(prompt, max_length=20,temperature = 0.5,num_return_sequences=1)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}\n")
    print("-" * 50)
    with open(output_file, "w") as f:
        f.write(output[0]['generated_text'])

print(time.time() - time_start)
print("Generated texts have been saved to files.")