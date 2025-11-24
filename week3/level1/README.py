### **LEVEL 1: Install & Run** ⭐
*Tests basic setup. *

**Goal:** Get transformers working on your machine

**Task:**
```
"""
Install and run this code. Submit a screenshot.
"""

# Install (run in terminal first):
# uv venv
# uv pip install transformers torch

from transformers import pipeline

# Load a small model
generator = pipeline('text-generation', model='distilgpt2')

# Generate text
prompts = [
    "The future of AI is",
    "In the year 2030",
    "The secret to happiness is"
]

for prompt in prompts:
    output = generator(prompt, max_length=30, num_return_sequences=1)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}\n")
    print("-" * 50)
```

**Submit:**
- Screenshot showing it works
- Your 3 generated outputs

**If Level 1 works → You're ready for Tier 1 tasks**

## Level 1 Help
- Transformers docs: https://huggingface.co/docs/transformers
- Pipeline guide: https://huggingface.co/docs/transformers/main_classes/pipelines

