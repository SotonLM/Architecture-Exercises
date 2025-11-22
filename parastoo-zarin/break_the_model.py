from transformers import pipeline
import time

gen=pipeline('text-generation',model='distilgpt2')

f=open('failure_modes.txt','w')
f.write("BREAKING THE MODEL\n"+"="*60+"\n\n")

# test 1 - repetition
print("TEST 1: repetition loop")
p1="The cat sat on the"
o1=gen(p1,max_length=100,repetition_penalty=1.0,do_sample=True,temperature=1.0)
t1=o1[0]['generated_text']
print(f"Prompt: {p1}")
print(f"Output: {t1}\n")
f.write("FAILURE 1: Repetition Loop\n")
f.write(f"Prompt: {p1}\n")
f.write(f"Output: {t1}\n")
f.write("Why: repetition_penalty=1.0 means no penalty\n")
f.write("Fix: use repetition_penalty=1.2\n\n")

# test 2 - crazy temp
print("="*60)
print("TEST 2: gibberish from high temp")
p2="The future of technology is"
o2=gen(p2,max_length=50,temperature=4.0,do_sample=True,top_k=100)
t2=o2[0]['generated_text']
print("Prompt:",p2)
print("Output:",t2)
print()
f.write("FAILURE 2: Gibberish\n")
f.write(f"Prompt: {p2}\n")
f.write(f"Output: {t2}\n")
f.write("Why: temperature=4.0 way too high\nFix: keep temp between 0.7-1.5\n\n")

# cuts off
print("TEST 3: incomplete")
print("-"*60)
p3="To make chocolate cake you need"
o3=gen(p3,max_length=15,temperature=1.0)
t3=o3[0]['generated_text']
print(f"Prompt: {p3}")
print(f"Output: {t3}")
print()
f.write("FAILURE 3: Cuts off mid-sentence\n")
f.write(f"Prompt: {p3}\nOutput: {t3}\n")
f.write("Why: max_length too short\n")
f.write("Fix: use max_length=50+\n\n")

# contradictions
print("="*60)
print("TEST 4: contradictions")
p4="The sky is blue. Actually, the sky is"
o4=gen(p4,max_length=35,temperature=1.2,do_sample=True)
t4=o4[0]['generated_text']
print("Prompt:",p4)
print("Output:",t4)
f.write("FAILURE 4: Contradictions\n")
f.write(f"Prompt: {p4}\n")
f.write(f"Output: {t4}\n\n")
f.write("Why: model doesnt understand logic\n")
f.write("Fix: use bigger models\n\n")

# no randomness
print("\n"+"="*60)
print("TEST 5: deterministic boring output")
p5="Once upon a time there was"
o5=gen(p5,max_length=40,do_sample=False)
t5=o5[0]['generated_text']
print(f"Prompt: {p5}")
print(f"Output: {t5}")
print()
f.write("FAILURE 5: No creativity\n")
f.write(f"Prompt: {p5}\n")
f.write(f"Output: {t5}\n")
f.write("Why: do_sample=False means always same output\n")
f.write("Fix: use do_sample=True\n\n")

f.write("\n"+"="*60+"\n")
f.write("SUMMARY\n")
f.write("="*60+"\n")
f.write("5 ways to break the model:\n")
f.write("1. repetition penalty too low\n")
f.write("2. temperature too high\n") 
f.write("3. max length too short\n")
f.write("4. model cant do logic\n")
f.write("5. no sampling = boring\n")
f.close()
print("="*60)
print("done! saved to failure_modes.txt")
