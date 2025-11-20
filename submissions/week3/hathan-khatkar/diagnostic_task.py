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
#NOTE: For level 4 I used sentence transformers library, install using: uv pip install sentence-transformers

from transformers import pipeline, AutoModel, AutoTokenizer
import time

# Configuration: Control which levels to run
levels = [True, True, True, True]

# Initialize generator if needed by any level
generator = None
prompts = [
    "The future of AI is",
    "In the year 2030",
    "The secret to happiness is",
    "Knock knock. Who's there?",
    "To make a sandwich, you need to"
]

# params (used in Level 2)
max_lengths = [20, 50, 100]
temperatures = [0.5, 1.0, 1.5]
top_ks = [10, 50, 100]

if levels[0] or levels[1] or levels[2]:
    generator = pipeline('text-generation', model='distilgpt2')

# LEVEL 1: Basic generation
if levels[0]:
    print("=== LEVEL 1: BASIC GENERATION ===")
    
    for prompt in prompts:
        output = generator(prompt, max_length=30)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output[0]['generated_text']}")
        print("-" * 50)

def generate(prompt, max_len, temp, k):
    start_time = time.time()

    output = generator(
        prompt,
        max_new_tokens=max_len, # changed from max_length to max_new_tokens as model overrides this parameter
        temperature=temp,
        top_k=k,
        do_sample=True
    )

    end_time = time.time()
    text = output[0]["generated_text"]

    token_count = len(generator.tokenizer.encode(text))
    elapsed = end_time - start_time

    return text, token_count, elapsed


# LEVEL 2: Your code here
if levels[1]:
    print("\n=== LEVEL 2: EXPERIMENT & DOCUMENT ===")
    
    def run_and_save(prompt, max_len, temp, k, run_id, outfile="result.txt"):
        """
        Runs a single generation using generate())
        and appends the result to a file instead of overwriting.
        """
        text, tokens, elapsed = generate(prompt, max_len, temp, k)

        with open(outfile, "a", encoding="utf-8") as f:
            f.write("\n============================\n")
            f.write(f"=== RUN RESULT {run_id} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"max_length={max_len}, temperature={temp}, top_k={k}\n\n")
            f.write("Generated Text:\n")
            f.write(text + "\n\n")
            f.write(f"Token Count: {tokens}\n")
            f.write(f"Time: {elapsed:.4f}s\n")
            f.write("============================\n")

        print(f"Run appended to {outfile}")

    # adjusting max tokens
    run_and_save(prompts[4], 20, 1.0, 50, 1)
    run_and_save(prompts[4], 50, 1.0, 50, 2)
    run_and_save(prompts[4], 100, 1.0, 50, 3)

    # adjusting temperature
    run_and_save(prompts[4], 50, 0.5, 50, 4)
    run_and_save(prompts[4], 50, 1.0, 50, 5)
    run_and_save(prompts[4], 50, 1.5, 50, 6)

    # adjusting top_k
    run_and_save(prompts[4], 50, 1.0, 10, 7)
    run_and_save(prompts[4], 50, 1.0, 50, 8)
    run_and_save(prompts[4], 50, 1.0, 100, 9)


# LEVEL 3: Option C - Compare Models
if levels[2]:
    print("\n=== LEVEL 3: MODEL COMPARISON (OPTION C) ===\n")
    
    # Models to compare
    models_to_test = [
        'distilgpt2',
        'gpt2',
        'gpt2-medium'
    ]
    
    # Test prompts (same for all models)
    test_prompts = [
        "The future of AI is",
        "To make a sandwich, you need to",
        "In the year 2030"
    ]

    def get_model_size(model_name):
        """Get approximate model size in parameters"""
        try:
            model = AutoModel.from_pretrained(model_name)
            num_params = sum(p.numel() for p in model.parameters())
            return num_params
        except Exception as e:
            print(f"Warning: Could not get size for {model_name}: {e}")
            # Approximate sizes if we can't load
            sizes = {
                'distilgpt2': 82_000_000,  # ~82M
                'gpt2': 124_000_000,       # ~124M
                'gpt2-medium': 355_000_000  # ~355M
            }
            return sizes.get(model_name, 0)

    def test_model(model_name, prompts, max_new_tokens=50):
        """Test a single model and return results"""
        print(f"\nTesting {model_name}...")
        
        # Load model
        load_start = time.time()
        model_generator = pipeline('text-generation', model=model_name)
        load_time = time.time() - load_start
        
        # Get model size
        model_size = get_model_size(model_name)
        
        results = {
            'model': model_name,
            'load_time': load_time,
            'model_size': model_size,
            'generations': [],
            'avg_time': 0,
            'total_time': 0
        }
        
        # Test each prompt
        times = []
        for prompt in prompts:
            start_time = time.time()
            output = model_generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=50,
                do_sample=True,
                num_return_sequences=1
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            generated_text = output[0]['generated_text']
            token_count = len(model_generator.tokenizer.encode(generated_text))
            
            results['generations'].append({
                'prompt': prompt,
                'output': generated_text,
                'time': elapsed,
                'tokens': token_count
            })
            
            print(f"  Prompt: {prompt[:30]}...")
            print(f"  Time: {elapsed:.4f}s, Tokens: {token_count}")
        
        results['avg_time'] = sum(times) / len(times) if times else 0
        results['total_time'] = sum(times)
        
        return results

    def print_comparison_table(all_results):
        """Print a formatted comparison table"""
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        
        # Header
        print(f"{'Model':<20} {'Size (M)':<12} {'Avg Time (s)':<15} {'Load Time (s)':<15}")
        print("-"*80)
        
        # Data rows
        for result in all_results:
            size_m = result['model_size'] / 1_000_000
            print(f"{result['model']:<20} {size_m:<12.1f} {result['avg_time']:<15.4f} {result['load_time']:<15.4f}")
        
        print("="*80)

    def print_detailed_comparison(all_results):
        """Print detailed comparison results to console"""
        print("\n" + "="*80)
        print("DETAILED MODEL COMPARISON")
        print("="*80)
        
        # Detailed results for each model
        for result in all_results:
            print("\n" + "="*80)
            print(f"MODEL: {result['model']}")
            print(f"Size: {result['model_size']/1_000_000:.1f}M parameters")
            print(f"Load Time: {result['load_time']:.4f}s")
            print(f"Average Generation Time: {result['avg_time']:.4f}s")
            print("-"*80)
            
            for gen in result['generations']:
                print(f"\nPrompt: {gen['prompt']}")
                print(f"Generated: {gen['output']}")
                print(f"Time: {gen['time']:.4f}s | Tokens: {gen['tokens']}")

    # Run comparison
    all_results = []
    for model_name in models_to_test:
        try:
            result = test_model(model_name, test_prompts)
            all_results.append(result)
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue

    # Print all comparisons
    if all_results:
        print_comparison_table(all_results)
        print_detailed_comparison(all_results)
        
        # Print recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        fastest = min(all_results, key=lambda x: x['avg_time'])
        smallest = min(all_results, key=lambda x: x['model_size'])
        largest = max(all_results, key=lambda x: x['model_size'])
        
        print(f"\nFastest Model: {fastest['model']} ({fastest['avg_time']:.4f}s avg)")
        print(f"\nSmallest Model: {smallest['model']} ({smallest['model_size']/1_000_000:.1f}M params)")
        
        print(f"\nLargest Model: {largest['model']} ({largest['model_size']/1_000_000:.1f}M params)")
        
    else:
        print("No models were successfully tested.")






# LEVEL 4: See level4_hybrid_qa.py for the standalone Level 4 implementation
if levels[3]:
    print("\n=== LEVEL 4: HYBRID PROPERTY Q&A SYSTEM ===")
    print("\nLevel 4 has been moved to a separate script: level4_hybrid_qa.py")
    print("To run Level 4, execute: python level4_hybrid_qa.py")

