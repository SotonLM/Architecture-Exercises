from transformers import AutoModelForCausalLM

models = ["gpt2","gpt2-medium","distilgpt2"]
for name in models:

    model = AutoModelForCausalLM.from_pretrained(name)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"{name} has {num_params:,} parameters")