# from transformers import pipeline

# generator = pipeline('text-generation', model='distilgpt2')

# prompts = [
#     "The future of AI is",
#     "In the year 2030",
#     "The secret to happiness is"
# ]

# for prompt in prompts:
#     output = generator(prompt, max_length=30, num_return_sequences=1)
#     print(f"\nPrompt: {prompt}")
#     print(f"Generated: {output[0]['generated_text']}\n")
#     print("-" * 50)


# from transformers import pipeline
# import time

# # LEVEL 1: Basic generation
# print("=== LEVEL 1: BASIC GENERATION ===")
# generator = pipeline('text-generation', model='distilgpt2')

# prompts = [
#     "The future of AI is",
#     "In the year 2030",
#     "The secret to happiness is"
# ]

# for prompt in prompts:
#     output = generator(prompt, max_length=30)
#     print(f"\nPrompt: {prompt}")
#     print(f"Generated: {output[0]['generated_text']}")
#     print("-" * 50)

# LEVEL 2: Your code here
# TODO: Save to file (results.txt)
# 5 diff prompts

# TODO: Try different parameters
    # Try max_length = 20, 50, 100
    # Try temperature = 0.5, 1.0, 1.5
    # Try top_k = 10, 50, 100
# TODO: Time generation (use import time)
# TODO: Count tokens

from transformers import pipeline
import time


generator = pipeline('text-generation', model='distilgpt2')

#5 diff prompts
prompts = [ 
    'The best way to learn programming is',
    'The sky is blue because',
    "The most beautiful place I've visited is",
    'Artificial intelligence is',
    'Sufficient sleep is important because'
    ]
#diff paramemeters
parameters = [
    {'max_length': 20, 'temperature': 0.5, 'top_k': 10},
    {'max_length': 50, 'temperature': 1.0, 'top_k': 50},
    {'max_length': 100, 'temperature': 1.5, 'top_k': 100},
]

#save file
with open('results.txt', 'w') as f:

    for prompt in prompts:
        for param in parameters:
            start_time = time.time()

            output = generator(prompt, **param) #changed to use given parameters
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {output[0]['generated_text']}")
            print("-" * 50)

            end_time = time.time()

            #time generation
            generated_time = end_time - start_time

            #result from prompt
            generated_text = output[0]['generated_text']

            #count tokens of gen text
            tokens = generator.tokenizer.encode(generated_text)
            token_count = len(tokens)

            results = (
                f"Prompt: {prompt}\n"
                f"Generated: {generated_text}\n"
                f"Time: {generated_time: .2f}s\n"
                f"Tokens: {token_count}"
            )

            print(results)
            f.write(results)

