#!/usr/bin/env python3
"""
Run the same prompts across multiple HuggingFace models and produce a comparison table and per-model output files.
>> python compare_models.py --models distilgpt2 gpt2 gpt2-medium
"""
import os
import time
import argparse
import statistics

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


DEFAULT_MODELS = ["distilgpt2", "gpt2", "gpt2-medium"] # suggested in readme
PROMPTS = [
    "Top 5 team sports are",
    "The secret to happiness is",
    "A simple recipe for pankakes:",
]

# Generation settings (fixed)
MAX_LENGTH = 50
TEMPERATURE = 1.0
TOP_K = 50

def human_size(n):
    # Human readable parameter count
    # convert numbers to K, M, B format
    if isinstance(n, (int, float)):
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        if n >= 1e6:
            return f"{n/1e6:.1f}M"
        if n >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(n)
    return str(n)


def repetition_score(token_ids):
    # Repetition metric: 1 - unique_ratio
    if not token_ids:
        return 0.0
    uniq = len(set(token_ids))
    return 1.0 - (uniq / len(token_ids))


def compare_models(models, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    outputs_dir = os.path.join(out_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "model_comparison.md")

    table_rows = []

    for model_name in models:
        print(f"\n=== Testing model: {model_name} ===")
        model_ok = True
        try:
            print("Loading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            # Build a pipeline
            gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            # Parameter count
            num_params = sum(p.numel() for p in model.parameters())

            times = []
            tokens_counts = []
            reps = []

            out_file = os.path.join(outputs_dir, f"{model_name.replace('/','_')}_outputs.txt") # safe filename
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"Model: {model_name}\nParameters: {human_size(num_params)}\n")
                f.write("-" * 60 + "\n\n")
                for prompt in PROMPTS:
                    print(f"Generating for prompt: {prompt[:40]}...")
                    t0 = time.time()
                    generated = gen(prompt, max_length=MAX_LENGTH, temperature=TEMPERATURE, top_k=TOP_K, num_return_sequences=1)
                    elapsed = time.time() - t0
                    text = generated[0]["generated_text"]
                    token_ids = tokenizer(text)["input_ids"]

                    times.append(elapsed)
                    tokens_counts.append(len(token_ids))
                    reps.append(repetition_score(token_ids))

                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Time(s): {elapsed:.3f}\n")
                    f.write(f"Tokens: {len(token_ids)}\n")
                    f.write(f"RepetitionScore: {reps[-1]:.3f}\n\n")
                    f.write(text + "\n\n" + ("=" * 40) + "\n\n")

            avg_time = statistics.mean(times) if times else None
            avg_tokens = statistics.mean(tokens_counts) if tokens_counts else None
            avg_rep = statistics.mean(reps) if reps else None

            table_rows.append({
                "model": model_name,
                "params": human_size(num_params),
                "avg_time_s": avg_time,
                "avg_tokens": avg_tokens,
                "rep_score": avg_rep,
                "status": "ok",
                "output_file": os.path.relpath(out_file, out_dir)
            })

        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            table_rows.append({
                "model": model_name,
                "params": "-",
                "avg_time_s": None,
                "avg_tokens": None,
                "rep_score": None,
                "status": f"failed: {e}",
                "output_file": ""
            })

    # Write Markdown summary
    with open(summary_path, "w", encoding="utf-8") as s:
        s.write("# Model Comparison\n\n")
        s.write("**Prompts used:**\n")
        for p in PROMPTS:
            s.write(f"- {p}\n")
        s.write("\n")
        s.write("| Model | Params | Avg time (s) | Avg tokens | Repetition score (0-1) | Outputs |\n")
        s.write("|---|---:|---:|---:|---:|---:|\n")
        for r in table_rows:
            at = f"{r['avg_time_s']:.3f}" if r['avg_time_s'] else "-"
            atok = f"{r['avg_tokens']:.1f}" if r['avg_tokens'] else "-"
            rep = f"{r['rep_score']:.3f}" if r['rep_score'] is not None else "-"
            outlink = f"[{r['output_file']}]({r['output_file']})" if r['output_file'] else "-"
            s.write(f"| {r['model']} | {r['params']} | {at} | {atok} | {rep} | {outlink} |\n")

    print(f"\nComparison written to: {summary_path}")
    print(f"Per-model outputs in: {outputs_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model names to compare")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "model_comparison"), help="Output directory")
    args = parser.parse_args()

    print("Models to test:", args.models)
    print("Prompts:")
    for p in PROMPTS:
        print(" -", p)

    compare_models(args.models, args.out)


if __name__ == '__main__':
    main()
