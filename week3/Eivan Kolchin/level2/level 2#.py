from transformers import pipeline
import time

generator = pipeline('text-generation', model= "distilgpt2")
tokenizer = generator.tokenizer

prompts = [
    'the future of AI is',
    'In the year 2030',
    'the secrete to happiness is',

    #Extra 5 prompts:
    "My man Donald Trump is", #1
    "Trump & Bill clinton", #1
    "snakes reproduce by", #2
    "data is",#3
    "my dog",#4
    "the dangers of AI are",#5
]

prompt_for_iteration= 'karl marx once said'

max_lengths = [20, 50, 100]
temperatures = [0.5, 1.0, 1.5]
top_ks = [10, 50, 100]

params = {
    "max_length":  max_lengths,
    "temperature": temperatures,
    "top_k":       top_ks,
}

base_cfg = {name: values[0] for name, values in params.items()}

with open("results.txt", "w", encoding= 'UTF-8') as file:
    for prompt in prompts:
        output_1 = generator(
            prompt,
            top_p= 0.96,
            num_return_sequences= 1,
            repetition_penalty= 3.6,
            temperature= 0.69,
            max_new_tokens=69,
        )
        generated = output_1[0]["generated_text"]

        total_tokens = len(tokenizer(generated)["input_ids"])
        prompt_tokens = len(tokenizer(prompt)["input_ids"])
        new_tokens = total_tokens - prompt_tokens

        answer = (
            f"\n\nPrompt: {prompt}\n\n"
            f"Generated: {generated}\n\n"
            f"Output tokens: {new_tokens}\n\n"
            + "-"*69
        )

        print('\n'*2+answer+'\n'*2+'-'*69); file.write(answer)
    file.write('\n\nPart 2:\n\n\n')


    for name, values in params.items():
        for val in values:
            cfg = base_cfg.copy()
            cfg[name] = val

            start = time.perf_counter()
            output_2 = generator(
                prompt_for_iteration,
                do_sample=True,
                top_p=0.96,
                num_return_sequences=1,
                repetition_penalty=3.6,
                **cfg,
            )
            time_taken= time.perf_counter() - start
            info = ", ".join(
                f"{k}={cfg[k]}" for k in ("max_length", "temperature", "top_k")
            )
            total_tokens = len(tokenizer(generated)["input_ids"])
            prompt_tokens = len(tokenizer(prompt)["input_ids"])
            new_toks = total_tokens - prompt_tokens

            txt = (
                f"\nPrompt: {prompt}\n{info}\n"
                f'Generated: {output_2[0]["generated_text"]}\n\n'
                f'Time taken: {time_taken:.2f} s\n\n'
                f'Tokens: {new_toks}\n\n'
                + "-"*69
            )
            print(txt); file.write(txt)
file.close()


