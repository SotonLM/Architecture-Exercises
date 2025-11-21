import time
from transformers import pipeline, set_seed
from transformers import AutoModel
import csv


def save_as_csv(results_list, filename="generation_results.csv"):

    fieldnames = results_list[0].keys()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write all the data rows
            for data_row in results_list:
                writer.writerow(data_row)
                
    print(f"\nSuccessfully saved results to {filename}")


def main():
    prompt = "Naruto vs sasuke? Who gonna win?"
    models = ["distilgpt2", "gpt2", "gpt2-medium"]

    all_results = []

    for model in models:
        generator = pipeline('text-generation', model=model)

        naruto = AutoModel.from_pretrained(model)

        model_size = naruto.num_parameters()


        start_time = time.time()
        output = generator(prompt, max_length=100, num_return_sequences=1)
        duration = time.time() - start_time
        
        token_length = len(generator.tokenizer.encode(output[0]['generated_text']))


        result_entry = {
            "Model": model,
            "Model Size": model_size,
            "Prompt": prompt,
            "Generated Text": output[0]['generated_text'],
            "Speed (s)": round(duration, 4),
            "Token Length": token_length
        }

        all_results.append(result_entry)

        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output[0]['generated_text']}\n")
        print("-" * 50)
    save_as_csv(all_results, "model_generation_report.csv")
main()