"""
DIAGNOSTIC TASK - Complete as many levels as you can

LEVEL 1: Get this working (required)
LEVEL 2: Expand it (tests Python skills)
LEVEL 3: Pick a challenge (tests thinking)
LEVEL 4: Build something new (tests creativity)

DUE: Wednesday 11:59 PM
Submit via: GitHub PR (preferred) or Teams #architecture channel.
See submission_format.md for details.
"""

from transformers import pipeline
import time
import json
from datetime import datetime

# LEVEL 1: Basic generation
print("=== LEVEL 1: BASIC GENERATION ===")
generator = pipeline('text-generation', model='distilgpt2')

prompts = [
    "The future of AI is",
    "In the year 2030",
    "The secret to happiness is"
]
level1_results = []
for prompt in prompts:
    output = generator(prompt, max_length=30)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output[0]['generated_text']}")
    print("-" * 70)
    level1_results.append({"prompt": prompt, "generated": output[0]['generated_text']})

# LEVEL 2: Experiment & Document
print("=== LEVEL 2: EXPERIMENT & DOCUMENT ===")

# Open file for saving results
with open("results.txt", "w", encoding="utf-8") as f:
    f.write("DIAGNOSTIC TASK RESULTS\n")
    for result in level1_results:
        f.write(f"Prompt: {result['prompt']}\n")
        f.write(f"Generated: {result['generated']}\n\n")
    
    level2_prompts = [
        "The most important invention of the 21st century is",
        "Climate change will affect",
        "The best way to learn programming is",
        "Artificial intelligence can help humanity by",
        "In a world without technology"
    ]
    parameters_to_test = [
        {"max_length": 20, "temperature": 0.7, "top_k": 50, "name": "Short & Focused"},
        {"max_length": 50, "temperature": 1.0, "top_k": 50, "name": "Medium & Balanced"},
        {"max_length": 100, "temperature": 1.5, "top_k": 100, "name": "Long & Creative"},
        {"max_length": 50, "temperature": 0.5, "top_k": 10, "name": "Conservative"},
        {"max_length": 50, "temperature": 1.5, "top_k": 100, "name": "Experimental"}
    ]
    
    print("\nTesting different parameter combinations...\n")
    experiment_results = []
    for i, prompt in enumerate(level2_prompts, 1):
        params = parameters_to_test[i-1]
        
        print(f"\nPrompt {i}: {prompt}")
        print(f"Parameters: {params['name']}")
        f.write(f"Experiment {i}: {params['name']}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Parameters: max_length={params['max_length']}, temperature={params['temperature']}, top_k={params['top_k']}\n")
        
        # Time the generation
        start_time = time.time()
        output = generator(
            prompt, 
            max_length=params['max_length'],
            temperature=params['temperature'],
            top_k=params['top_k'],
            do_sample=True
        )
        end_time = time.time()
        generated_text = output[0]['generated_text']
        generation_time = end_time - start_time
        
        # Count tokens
        token_count = len(generator.tokenizer.encode(generated_text))
        
        print(f"Generated: {generated_text}")
        print(f"Time: {generation_time:.3f}s | Tokens: {token_count}")
        
        f.write(f"Generated: {generated_text}\n")
        f.write(f"Generation Time: {generation_time:.3f} seconds\n")
        f.write(f"Token Count: {token_count}\n")
        f.write("-" * 70 + "\n\n")
        
        experiment_results.append({
            "prompt": prompt,
            "params": params,
            "generated": generated_text,
            "time": generation_time,
            "tokens": token_count
        })

    analysis = """
FINDINGS:

1. MAX_LENGTH Effect:
   - Shorter (20): Quick, concise outputs but often incomplete thoughts
   - Medium (50): Good balance of coherence and completeness
   - Longer (100): More detailed but can become repetitive or lose focus

2. TEMPERATURE Effect:
   - Low (0.5): More predictable, conservative, grammatically safer
   - Medium (1.0): Balanced creativity and coherence
   - High (1.5): More creative/random but can produce nonsensical text

3. TOP_K Effect:
   - Low (10): Limited vocabulary, more repetitive
   - Medium (50): Good variety while maintaining quality
   - High (100): Maximum diversity but higher risk of odd word choices

4. GENERATION SPEED:
   - Average time: {:.3f} seconds
   - Longer max_length increases generation time proportionally
   - Temperature and top_k have minimal impact on speed

5. BEST SETTINGS:
   - For coherent text: max_length=50, temperature=0.7-1.0, top_k=50
   - For creative text: max_length=50-100, temperature=1.2-1.5, top_k=100
   - For quick generation: max_length=20-30, temperature=0.7, top_k=50
""".format(sum(r['time'] for r in experiment_results) / len(experiment_results))
    
    f.write(analysis)
    print("\n" + analysis)

# LEVEL 3: Open-Ended Challenges
with open("results.txt", "a", encoding="utf-8") as f:
    f.write("LEVEL 3 CHALLENGES:\n")
    print("\n--- OPTION A: Breaking the Model ---")
    f.write("OPTION A: BREAKING THE MODEL\n")    
    failure_modes = [
        {
            "name": "Repetition Loop",
            "prompt": "he he he he he ",
            "params": {"max_length": 50, "temperature": 0.1, "top_k": 1},
            "explanation": "Low temperature and top_k=1 make model deterministic, repeating patterns"
        },
        {
            "name": "Gibberish Generation",
            "prompt": "xqz zyx qwp",
            "params": {"max_length": 50, "temperature": 2.0, "top_k": 100},
            "explanation": "High temperature with nonsense input produces incoherent outputs"
        },
        {
            "name": "Contradiction",
            "prompt": "AI is good and AI is bad because",
            "params": {"max_length": 60, "temperature": 1.0, "top_k": 50},
            "explanation": "Conflicting premises in prompt lead to contradictory continuations"
        },
        {
            "name": "Endless Punctuation",
            "prompt": "...",
            "params": {"max_length": 40, "temperature": 0.5, "top_k": 10},
            "explanation": "Minimal context causes model to continue pattern indefinitely"
        },
        {
            "name": "Topic Drift",
            "prompt": "The recipe for chocolate cake",
            "params": {"max_length": 100, "temperature": 1.8, "top_k": 100},
            "explanation": "Very high temperature causes model to drift away from original topic"
        }
    ]
    
    for i, failure in enumerate(failure_modes, 1):
        print(f"\nFailure Mode {i}: {failure['name']}")
        f.write(f"Failure Mode {i}: {failure['name']}\n")
        f.write(f"Prompt: {failure['prompt']}\n")
        f.write(f"Parameters: {failure['params']}\n")
        
        try:
            output = generator(
                failure['prompt'],
                max_length=failure['params']['max_length'],
                temperature=failure['params']['temperature'],
                top_k=failure['params']['top_k'],
                do_sample=True
            )
            result = output[0]['generated_text']
        except Exception as e:
            result = f"ERROR: {str(e)}"
        
        print(f"Output: {result}")
        print(f"Why it fails: {failure['explanation']}")
        
        f.write(f"Output: {result}\n")
        f.write(f"Explanation: {failure['explanation']}\n")
        f.write(f"Fix: Adjust temperature/top_k, provide better context, add repetition penalty\n")
    
 
