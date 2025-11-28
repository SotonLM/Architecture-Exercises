import time
import os 
from transformers import pipeline, AutoTokenizer

generator = pipeline('text-generation', model='distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

PROMPTS = [
    "The Big Bang theory is",
    "To build a own startup",
    "The future of Artificial Intelligence is going to be",
    "The flow of money in international exchange is",
    "The benefits of physical fitness are" 
]

GENERATION_CONFIG = {"max_new_tokens": 20,"num_return_sequences": 1,"top_k": 50,"temperature": 0.7,"do_sample": True,"pad_token_id": tokenizer.eos_token_id}
generated_texts = []
output_dir =os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True) 
start_time = time.time()
output_batch = generator(PROMPTS, **GENERATION_CONFIG)

for prompt, output in zip(PROMPTS, output_batch):
    generated_text_string = output[0]['generated_text'] 
    clean_generated_text = generated_text_string.strip()
    token_count = len(tokenizer.encode(clean_generated_text))
    generated_texts.append(clean_generated_text)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {clean_generated_text}")
    print(f"Token count: {token_count}")
    print("-" * 50)

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal time taken to generate: {total_time:.2f} seconds")

output_file_path = os.path.join(output_dir, "all_text.txt")
with open(output_file_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(generated_texts))

print(f"Output saved successfully to: {os.path.abspath(output_file_path)}")
# ISSUES
# coding runing but showing the errors at last 
# the generate text some times repeating the set of sequenc of word than the generatd text
# it showing lot of space in between each promts  
# counting of token is not working properly 
# time takes is not show random numerical instead of seconds
# the text file is not saving in the same directory


# To add 
# to save as a text file 
# clear all the errors


# Findings 
# using the max_new_tokens to less reduces the extra spaces which generatng between the each prompts 
# used zip to combine the single output
# used .strip() to remove the extra space after each generation 
# when the temperature value is less it has some error but if we use in the average value it not having any issues 