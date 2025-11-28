# Diagnostic Task Submission

**Name:** Akhil
**Level Reached:** Level 3

### LEVEL 1: ✅
![alt text](<Level 1.png>)

### LEVEL 2: ✅
- **Code:** [diagnostic_task.py](./diagnostic_task.py)
- **Results:** See findings in results.txt
- **Summary:** Experimented with Temperature and Top_K. Found that Temperature > 1.2 causes hallucinations and < 0.7 causes repetition loops.

### LEVEL 3: ✅ (Option A: Break the Model)
- **Failure Report:** [results.txt](./results.txt)
- **Summary:** Successfully triggered 5 failure modes including Infinite Repetition (using Greedy Search), Logic Failures (Context Loss), and Out-of-Distribution errors (Punctuation spam).

### LEVEL 4: ❌ (Not attempted)

**Time Spent:** ~2 hours
**What I found easy:** Setting up the environment and running basic generation.
**What I found hard:** Understanding why the model enters infinite loops without sampling.