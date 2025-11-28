from transformers import pipeline
import time
# Load a small model



def write_comments(source):
    generator = pipeline('text-generation', model='gpt2-medium')
    prompts = []
    with open(source,'r+') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line)
        for prompt in prompts:
            input_prompt = (
                f"{prompt}\n\nPlease write a single concise one-line comment describing what the above code does. "
                "Start the line with '## ' and return only that comment."
            )

            output = generator(
                input_prompt,
                max_new_tokens=30,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.2,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,)
            f.write(f"##{output[0]['generated_text']}\n")



source = input("Please enter the directory of your code:")

write_comments(source)
