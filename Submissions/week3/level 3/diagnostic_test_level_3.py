"""Level 3 Diagnostic Task: OPTION C: Compare models"""
from transformers import pipeline
import time

generator1 = pipeline('text-generation', model='distilgpt2')
generator2 = pipeline('text-generation', model='gpt2')
generator3 = pipeline('text-generation', model='gpt2-medium')


prompts = [
    "The future of AI is",
    "In the year 2030",
    "Once upon a time in a land far away",
]
# Define common generation parameters
_max_new_tokens=50
_num_return_sequences=1
_do_sample=True
_num_beams=1
_temperature = 0.5
_top_k = 50
_top_p = 0.92
_repetition_penalty = 1.075

with open("time_results.txt", "w",encoding='utf-8') as f: # automatically closes the file after writing
    for _max_new_tokens in [50,100,200]:
        average_times = []
        #================================================distilgpt2=========================================================
        generatortimes = [] #list to store generation times for calculating average later
        for prompt in prompts:
            start_time = time.time() # Start the timer to measure the generation time
            output = generator1(prompt, min_new_tokens = _max_new_tokens,
                                max_new_tokens=_max_new_tokens,
                                num_return_sequences=_num_return_sequences,
                                do_sample=_do_sample,
                                num_beams = _num_beams,
                                temperature = _temperature,
                                top_k = _top_k,
                                top_p = _top_p, 
                                repetition_penalty = _repetition_penalty) # Generate text with specified parameters using distilgpt2
            end_time = time.time() # stop the timer once generation is complete
            generation_time = end_time - start_time #calculate generation time
            generatortimes.append(generation_time)
            token_count = len(generator1.tokenizer.encode(output[0]['generated_text'])) # Count the number of tokens in the generated text
        avg_time = sum(generatortimes) / len(generatortimes)
        average_times.append(avg_time)
        f.write(f"\nAverage Generation Time for distilgpt2 with {_max_new_tokens} tokens: {avg_time:.4f} seconds\n")
        #======================================================gpt2=========================================================
        generatortimes = [] #reset generation times list for next model
        for prompt in prompts:
            start_time = time.time() # Start the timer to measure the generation time
            output = generator2(prompt, min_new_tokens = _max_new_tokens,
                        max_new_tokens=_max_new_tokens,
                        num_return_sequences=_num_return_sequences,
                        do_sample=_do_sample,
                        num_beams = _num_beams,
                        temperature = _temperature,
                        top_k = _top_k,
                        top_p = _top_p, 
                        repetition_penalty = _repetition_penalty) # Generate text with specified parameters using gpt2
            end_time = time.time() # stop the timer once generation is complete
            generation_time = end_time - start_time #calculate generation time
            token_count = len(generator2.tokenizer.encode(output[0]['generated_text'])) # Count the number of tokens in the generated text
            generatortimes.append(generation_time)
        avg_time = sum(generatortimes) / len(generatortimes)
        average_times.append(avg_time)
        f.write(f"\nAverage Generation Time for gpt2 with {_max_new_tokens} tokens: {avg_time:.4f} seconds\n")
        #================================================gpt2-medium=================================================
        generatortimes = [] #reset generation times list for next model
        for prompt in prompts:
            start_time = time.time() # Start the timer to measure the generation time
            output = generator3(prompt, min_new_tokens = _max_new_tokens,
                                max_new_tokens=_max_new_tokens,
                                num_return_sequences=_num_return_sequences,
                                do_sample=_do_sample,
                                num_beams = _num_beams,
                                temperature = _temperature,
                                top_k = _top_k,
                                top_p = _top_p, 
                                repetition_penalty = _repetition_penalty) # Generate text with specified parameters using gpt2_medium
            end_time = time.time() # stop the timer once generation is complete
            generation_time = end_time - start_time #calculate generation time
            generatortimes.append(generation_time)
            token_count = len(generator3.tokenizer.encode(output[0]['generated_text'])) # Count the number of tokens in the generated text
        avg_time = sum(generatortimes) / len(generatortimes)
        average_times.append(avg_time)
        #================================================Summary=================================================
        f.write(f"\nAverage Generation Time for gpt2-medium with {_max_new_tokens} tokens: {avg_time:.4f} seconds\n")
        f.write(f"\nComparison of Average Generation Times with {_max_new_tokens} tokens using distilgpt2 as benchmark:\n")
        f.write(f"distilgpt2: {average_times[0]:.4f} seconds\n")
        f.write(f"gpt2: {average_times[1]:.4f} seconds, Ratio: {((average_times[1]-average_times[0])/average_times[0])*100:.2f}%\n")
        f.write(f"gpt2-medium: {average_times[2]:.4f} seconds, Ratio: {((average_times[2]-average_times[0])/average_times[0])*100:.2f}%\n")

# Generation Comparison
with open("generation_comparison.txt", "w",encoding='utf-8') as f:
    for prompt in prompts:
        f.write(f"\nPrompt: {prompt}\n")
        output1 = generator1(prompt, min_new_tokens = _max_new_tokens,
                                max_new_tokens=_max_new_tokens,
                                num_return_sequences=_num_return_sequences,
                                do_sample=_do_sample,
                                num_beams = _num_beams,
                                temperature = _temperature,
                                top_k = _top_k,
                                top_p = _top_p, 
                                repetition_penalty = _repetition_penalty)
        f.write(f"\ndistilgpt2 Output:\n{output1[0]['generated_text']}\n")
        
        output2 = generator2(prompt, min_new_tokens = _max_new_tokens,
                                max_new_tokens=_max_new_tokens,
                                num_return_sequences=_num_return_sequences,
                                do_sample=_do_sample,
                                num_beams = _num_beams,
                                temperature = _temperature,
                                top_k = _top_k,
                                top_p = _top_p, 
                                repetition_penalty = _repetition_penalty)
        f.write(f"\ngpt2 Output:\n{output2[0]['generated_text']}\n")
        
        output3 = generator3(prompt, min_new_tokens = _max_new_tokens,
                                max_new_tokens=_max_new_tokens,
                                num_return_sequences=_num_return_sequences,
                                do_sample=_do_sample,
                                num_beams = _num_beams,
                                temperature = _temperature,
                                top_k = _top_k,
                                top_p = _top_p, 
                                repetition_penalty = _repetition_penalty)
        f.write(f"\ngpt2-medium Output:\n{output3[0]['generated_text']}\n")


