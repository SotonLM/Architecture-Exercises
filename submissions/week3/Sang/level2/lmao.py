from transformers import pipeline
import time


def write_in_txt(prompt, duration, token_num, max_length_num, temperature_num, top_k_num, generated_text):
    

    with open("results.txt", "a", encoding="utf-8") as file:
        file.write(f"Prompt: {prompt}\n")
        file.write(f"Duration: {duration}\n")
        file.write(f"Token_num: {token_num}\n")
        file.write(f"Max_length: {max_length_num}\n")
        file.write(f"temperature_num: {temperature_num}\n")
        file.write(f"top_k_num: {top_k_num}\n")

        file.write(f"Output: {generated_text.rstrip()}\n")
        file.write("===========================================================================================\n")


def main():

    # Load a small model
    generator = pipeline('text-generation', model='distilgpt2')

    amongus = {
        "prompts": ["AI is not real", "how to be sus", "The secret to success is amongus", "how to be an AI engineer", "live is fun without girls"],
        "var_max_length": [20, 50, 100],
        "var_temperature": [0.5, 1.0, 1.5],
        "var_top_k": [10, 50, 100],
    }

    for prompt in amongus["prompts"]:
        for max_length_i in amongus["var_max_length"]:
            for temperature_i in amongus["var_temperature"]:
                for top_k_i in amongus["var_top_k"]:
                    start = time.time()
                    output = generator(prompt, max_length=max_length_i, temperature=temperature_i, top_k=top_k_i)
                    duration = time.time() - start

                    generated_text = output[0]['generated_text']

                    token_num = len(generator.tokenizer.encode(generated_text))


                    write_in_txt(prompt, duration, token_num, max_length_i, temperature_i, top_k_i, generated_text)

main()