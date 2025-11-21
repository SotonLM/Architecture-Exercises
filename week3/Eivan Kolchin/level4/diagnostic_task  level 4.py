"""
DIAGNOSTIC TASK - Complete as many levels as you can

LEVEL 1: Get this working (required)
LEVEL 2: Expand it (tests Python skills)
LEVEL 3: Pick a challenge (tests thinking)
LEVEL 4: Build something new (tests creativity)

DUE: Wednesday 11:59 PM
Submit via: GitHub PR (preferred) or Teams #architecture channel.
See submission_format.txt for details.
"""

'''from transformers import pipeline
import time

# LEVEL 1: Basic generation
print("=== LEVEL 1: BASIC GENERATION ===")
generator = pipeline('text-generation', model='distilgpt2')

prompts = [
    "The future of AI is",
    "In the year 2030",
    "The secret to happiness is"
]

for prompt in prompts:
    output = generator(prompt, max_length=30)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}")
    print("-" * 50)

# LEVEL 2: Your code here
# TODO: Save to file
# TODO: Try different parameters
# TODO: Time generation
# TODO: Count tokens

# LEVEL 3: Your code here
# TODO: Pick Option A, B, C, or D. maybe all of them?
# TODO: Implement your challenge

# LEVEL 4: Your code here
# TODO: Build something new'''

"""
transformer_dissector_multi.py

CLI "x-ray" for GPT-2 family transformers:

- Choose model: distilgpt2, gpt2, gpt2-medium
- Inspect attention:
    For a given layer & head, show which tokens each token attends to most.
- Show next-token predictions:
    Top-k tokens the model thinks might come next.
- Remove one token:
    Remove a chosen token from the input and see how the next-token distribution changes.

Demonstrates the same internal analysis across multiple transformer architectures.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

available_models = ["distilgpt2", "gpt2", "gpt2-medium"]

current_model = None
tokenizer = None
model = None


def load_model(model_name):
    global current_model, tokenizer, model
    if model_name == current_model and model is not None and tokenizer is not None:
        return

    print(f"\n[Init] Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    current_model = model_name

    n_layer = getattr(model.config, "n_layer", "?")
    n_head = getattr(model.config, "n_head", "?")
    hidden_size = getattr(model.config, "n_embd", "?")
    print(f"[Info] {model_name} | layers: {n_layer}, heads: {n_head}, hidden: {hidden_size}\n")


def choose_model_interactively():
    print("\nModels:")
    for i, name in enumerate(available_models):
        print(f"  [{i}] {name}")
    while True:
        try:
            idx = int(input("Choose model index: ").strip())
            if 0 <= idx < len(available_models):
                return available_models[idx]
        except ValueError:
            pass
        print("Invalid index, try again.")


def tokenize_text(text):
    return tokenizer(text, return_tensors="pt")


def get_tokens(input_ids):
    return tokenizer.convert_ids_to_tokens(input_ids.tolist())


def attention(text, layer_idx, head_idx, top_k=3):
    inputs = tokenize_text(text)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    if not (0 <= layer_idx < len(attentions)):
        print(f"Invalid layer {layer_idx}.")
        return

    layer_attn = attentions[layer_idx]
    if not (0 <= head_idx < layer_attn.shape[1]):
        print(f"Invalid head {head_idx}.")
        return

    attn = layer_attn[0, head_idx]
    input_ids = inputs["input_ids"][0]
    tokens = get_tokens(input_ids)

    print(f"\n[Attention] Model: {current_model}, layer {layer_idx}, head {head_idx}")
    print("Tokens:")
    print("  " + " ".join(f"[{i}]{tok}" for i, tok in enumerate(tokens)))

    seq_len = attn.shape[0]
    top_k = min(top_k, seq_len)

    for i in range(seq_len):
        values, indices = torch.topk(attn[i], k=top_k)
        focus = [f"{tokens[int(j)]} ({v.item():.2f})" for v, j in zip(values, indices)]
        print(f"Token [{i}] {tokens[i]:>10} attends to " + ", ".join(focus))


def get_top_k_predictions_from_ids(input_ids, top_k=10):
    input_ids = input_ids.unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=top_k)
    return [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(values, indices)]


def show_next_token_predictions(text, top_k=10):
    inputs = tokenize_text(text)
    input_ids = inputs["input_ids"][0]
    tokens = get_tokens(input_ids)

    print(f"\n[Next-token predictions] Model: {current_model}")
    print("Input tokens:")
    print("  " + " ".join(f"[{i}]{tok}" for i, tok in enumerate(tokens)))

    results = get_top_k_predictions_from_ids(input_ids, top_k=top_k)
    print(f"\nTop {top_k} next-token predictions:")
    for tok, prob in results:
        print(f"  {tok!r:>10}  ->  {prob:.4f}")


def remove_token_and_compare(text, top_k=10):
    inputs = tokenize_text(text)
    input_ids = inputs["input_ids"][0]
    tokens = get_tokens(input_ids)

    print(f"\n[Remove token] Model: {current_model}")
    print("Current tokens:")
    print("  " + " ".join(f"[{i}]{tok}" for i, tok in enumerate(tokens)))

    while True:
        idx_str = input("Index of token to remove (or 'c' to cancel): ").strip().lower()
        if idx_str in ("c", "cancel"):
            print("Cancelled.")
            return
        try:
            idx = int(idx_str)
            if 0 <= idx < len(tokens):
                break
        except ValueError:
            pass
        print("Invalid index, try again.")

    orig_preds = get_top_k_predictions_from_ids(input_ids, top_k=top_k)

    new_ids = torch.cat([input_ids[:idx], input_ids[idx + 1 :]], dim=0)
    new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    new_preds = get_top_k_predictions_from_ids(new_ids, top_k=top_k)

    print(f"\nOriginal text: {text!r}")
    print(f"New text after removing token [{idx}] {tokens[idx]!r}: {new_text!r}")

    print(f"\nTop {top_k} next-token predictions before:")
    for tok, prob in orig_preds:
        print(f"  {tok!r:>10}  ->  {prob:.4f}")

    print(f"\nTop {top_k} next-token predictions after:")
    for tok, prob in new_preds:
        print(f"  {tok!r:>10}  ->  {prob:.4f}")


def main():
    print("=" * 69)
    print("Transformer Dissector (multi model, GPT-2 fam)")
    print("=" * 69)

    print("Choose an initial model:")
    load_model(choose_model_interactively())

    while True:
        print(f"\nCurrent model: {current_model}")
        text = input("Enter a sentence (or 'm' to change model, 'q' to quit): ").strip()

        if text.lower() == "q":
            print("bye bye babes.")
            break
        if text.lower() == "m":
            load_model(choose_model_interactively())
            continue
        if not text:
            continue

        while True:
            print("\nChoose an action:")
            print("  [1] Inspect attention")
            print("  [2] Show next token predictions")
            print("  [3] Remove a token and compare predictions")
            print("  [m] Change model")
            print("  [n] Enter a new sentence")
            print("  [q] Quit")
            choice = input("> ").strip().lower()

            if choice == "1":
                n_layer = getattr(model.config, "n_layer", None)
                n_head = getattr(model.config, "n_head", None)
                print(f"Model layers: {n_layer}, heads per layer: {n_head}")
                try:
                    layer_idx = int(input("Layer index (0 based, < n_layer): ").strip())
                    head_idx = int(input("Head index (0 based, < n_head): ").strip())
                except ValueError:
                    print("Invalid indices.")
                    continue
                attention(text, layer_idx, head_idx, top_k=3)

            elif choice == "2":
                try:
                    k = int(input("How many top tokens to show? [default 10]: ") or "10")
                except ValueError:
                    k = 10
                show_next_token_predictions(text, top_k=k)

            elif choice == "3":
                try:
                    k = int(input("How many top tokens to show? [default 10]: ") or "10")
                except ValueError:
                    k = 10
                remove_token_and_compare(text, top_k=k)

            elif choice == "m":
                load_model(choose_model_interactively())
                break

            elif choice == "n":
                break

            elif choice == "q":
                print("Bye bye babes.")
                return

            else:
                print("Unknown choice, try again.")
                #
main()







