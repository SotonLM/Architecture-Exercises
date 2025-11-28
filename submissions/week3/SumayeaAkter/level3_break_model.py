from transformers import pipeline

# Use the same model as before
generator = pipeline("text-generation", model="distilgpt2")

def show_failure(title, prompt, **gen_kwargs):
    print("\n" + "=" * 60)
    print(f"FAILURE MODE: {title}")
    print(f"PROMPT: {prompt}\n")

    output = generator(prompt, num_return_sequences=1, **gen_kwargs)
    text = output[0]["generated_text"]
    print("OUTPUT:\n")
    print(text)
    print("\n" + "=" * 60)
    return text

def main():
    # 1) Repetition / looping
    rep_prompt = "Repeat the word 'cat' forever: cat"
    rep_text = show_failure(
        "Repetition / Looping",
        rep_prompt,
        max_length=120,
        temperature=0.7,
        top_k=5,
    )

    # 2) Gibberish / nonsense
    gib_prompt = "Describe quantum happiness using made-up technical jargon:"
    gib_text = show_failure(
        "Gibberish / Nonsense",
        gib_prompt,
        max_length=120,
        temperature=1.6,
        top_k=100,
    )

    # 3) Contradiction
    con_prompt = "Write two sentences about the weather that contradict each other:"
    con_text = show_failure(
        "Contradiction",
        con_prompt,
        max_length=60,
        temperature=1.0,
        top_k=50,
    )

    # 4) Instruction-following failure
    inst_prompt = "Answer with ONLY 'YES' or 'NO': Is ice hot?"
    inst_text = show_failure(
        "Instruction following failure",
        inst_prompt,
        max_length=60,
        temperature=1.0,
        top_k=50,
    )

    # 5) Hallucinated facts
    hal_prompt = "Explain the life and major achievements of Queen Zorblax of Jupiter:"
    hal_text = show_failure(
        "Hallucinated facts",
        hal_prompt,
        max_length=120,
        temperature=1.0,
        top_k=50,
    )

    # Save everything to a file
    with open("level3_failures.txt", "w") as f:
        f.write("=== LEVEL 3: BREAK THE MODEL ===\n\n")
        f.write("1) Repetition / Looping\n")
        f.write(f"Prompt: {rep_prompt}\nOutput:\n{rep_text}\n\n")
        f.write("2) Gibberish / Nonsense\n")
        f.write(f"Prompt: {gib_prompt}\nOutput:\n{gib_text}\n\n")
        f.write("3) Contradiction\n")
        f.write(f"Prompt: {con_prompt}\nOutput:\n{con_text}\n\n")
        f.write("4) Instruction following failure\n")
        f.write(f"Prompt: {inst_prompt}\nOutput:\n{inst_text}\n\n")
        f.write("5) Hallucinated facts\n")
        f.write(f"Prompt: {hal_prompt}\nOutput:\n{hal_text}\n\n")

if __name__ == "__main__":
    main()
