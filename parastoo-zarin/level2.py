from transformers import pipeline
import time

# load the model
gen=pipeline('text-generation',model='distilgpt2')

f=open('results.txt','w')
f.write("LEVEL 2 RESULTS\n"+"="*60+"\n\n")

# testing different prompts
p=["The future of AI is","In the year 2030","The secret to happiness is","Space exploration will","The most important skill to learn is"]

print("="*60)
print("EXPERIMENT 1: max_lengths")
print("="*60)
f.write("EXPERIMENT 1: max_lengths\n\n")
for l in [20,50,100]:
    print(f"\n--- max_length={l} ---")
    f.write(f"--- max_length={l} ---\n")
    s=time.time()
    o=gen(p[0],max_length=l,num_return_sequences=1)
    e=time.time()-s
    t=o[0]['generated_text']
    tok=len(gen.tokenizer.encode(t))
    print(f"Generated: {t}")
    print(f"Time: {e:.3f}s | Tokens: {tok}")
    f.write(f"Generated: {t}\n")
    f.write(f"Time: {e:.3f}s | Tokens: {tok}\n\n")

print("\n"+"="*60)
print("EXPERIMENT 2: temperatures")
print("="*60)
f.write("\nEXPERIMENT 2: temperatures\n\n")
for temp in [0.5,1.0,1.5]:
    print(f"\n--- temperature={temp} ---")
    f.write(f"--- temperature={temp} ---\n")
    s=time.time()
    o=gen(p[1],max_length=50,temperature=temp,do_sample=True,num_return_sequences=1)
    e=time.time()-s
    t=o[0]['generated_text']
    tok=len(gen.tokenizer.encode(t))
    print(f"Generated: {t}")
    print(f"Time: {e:.3f}s | Tokens: {tok}")
    f.write(f"Generated: {t}\n")
    f.write(f"Time: {e:.3f}s | Tokens: {tok}\n\n")

print("\n"+"="*60)
print("EXPERIMENT 3: top_k")
print("="*60)
f.write("\nEXPERIMENT 3: top_k\n\n")
for k in [10,50,100]:
    print(f"\n--- top_k={k} ---")
    f.write(f"--- top_k={k} ---\n")
    s=time.time()
    o=gen(p[2],max_length=50,top_k=k,do_sample=True,num_return_sequences=1)
    e=time.time()-s
    t=o[0]['generated_text']
    tok=len(gen.tokenizer.encode(t))
    print(f"Generated: {t}")
    print(f"Time: {e:.3f}s | Tokens: {tok}")
    f.write(f"Generated: {t}\n")
    f.write(f"Time: {e:.3f}s | Tokens: {tok}\n\n")

print("\n"+"="*60)
print("ALL 5 PROMPTS")
print("="*60)
f.write("\nALL 5 PROMPTS\n\n")
ts=time.time()
for i,pr in enumerate(p,1):
    print(f"\nPrompt {i}: {pr}")
    f.write(f"Prompt {i}: {pr}\n")
    s=time.time()
    o=gen(pr,max_length=50,temperature=1.0,top_k=50,do_sample=True,num_return_sequences=1)
    e=time.time()-s
    t=o[0]['generated_text']
    tok=len(gen.tokenizer.encode(t))
    print(f"Generated: {t}")
    print(f"Time: {e:.3f}s | Tokens: {tok}")
    f.write(f"Generated: {t}\n")
    f.write(f"Time: {e:.3f}s | Tokens: {tok}\n\n")
tt=time.time()-ts
f.write("\n"+"="*60+"\n")
f.write("SUMMARY\n"+"="*60+"\n")
f.write(f"Total: {tt:.3f}s\n")
f.write(f"Avg: {tt/5:.3f}s\n")
f.close()
print("\n"+"="*60)
print(f"Done! Total: {tt:.3f}s")
print("="*60)
