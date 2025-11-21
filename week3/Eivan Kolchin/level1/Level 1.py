from transformers import pipeline

generator = pipeline('text-generation', model = "distilgpt2")

prompts = [
    'the future of AI is',
    'In the year 2030',
    'the secrete to happiness is'
]

for prompt in prompts:
    output = generator(prompt, max_length= 30, num_return_sequences=1)
    print(
        f'\nPrompt: {prompt}',
        f'Generated: {output[0]["generated_text"]}\n',
        '-'*50,
    )