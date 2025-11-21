from transformers import pipeline
import time

#Reset file for new prompts
def clear_file():
    with open("./submissions/week3/Nick Gurney/results.txt","w") as f:
        f.write("")
# Save output
def save_output(prompt, output):
    with open("./submissions/week3/Nick Gurney/results.txt","a") as f:
        output = output.encode('cp850','replace').decode('cp850')
        f.writelines([f"Prompt: {prompt}\n"])
        f.writelines([f"  Generation: {output}\n"])



# Load a small model
generator = pipeline('text-generation', model='distilgpt2')

# Generate text
prompts = [
    "Describe a world where technology no longer exists",
    "The moment everything changed was when",
    "If I could travel anywhere in the universe",
    "The biggest challenge humanity will face is",
    "A mystery nobody has solved yet is",
    "If time could flow in reverse for a single day",
    "Describe a city where memories can be bought and sold",
    "The moment the world realized it wasn’t alone",
    "If I could speak to my future self, I would ask",
    "The greatest invention that will reshape our lives is",
    "A secret hidden beneath the ocean's deepest trench is",
    "Describe a society where emotions are traded like currency",
    "If I had the ability to pause the world for one hour",
    "The most important lesson humanity has forgotten is",
    "A discovery that could change the course of history is",
    "If dreams were portals to real places",
    "The last message left by an ancient civilization was",
    "If I could redesign reality from scratch",
    "The moment humanity takes its next big evolutionary step",
    "A world where every lie instantly becomes true",
    "Describe the consequences of waking up with no memories",
    "If every human had a visible aura reflecting their intentions",
    "The most dangerous idea ever conceived is",
    "A phenomenon scientists are terrified to study is",
    "If humanity suddenly gained the ability to read minds"
]

clear_file()
times = []
token_lengths = []
for prompt in prompts:
    start = time.time()
    output = generator(prompt, max_length=100, temperature = 1.5, top_k = 100)
    text = output[0]['generated_text']
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {text}\n")
    print("-" * 50)
    save_output(prompt, text)
    times.append(time.time() - start)
    token_lengths.append(len(generator.tokenizer.encode(text)))

print(times)
print(token_lengths)