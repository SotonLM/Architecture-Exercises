### **LEVEL 2: Experiment & Document** ⭐⭐
*Tests basic Python skills. *

**Goal:** Modify the code and observe changes

**Task:**
Expand Level 1 code to:

1. **Save outputs to file** (results.txt)
2. **Try 5 different prompts** (your choice of topics)
3. **Experiment with parameters:**
   - Try `max_length` = 20, 50, 100
   - Try `temperature` = 0.5, 1.0, 1.5
   - Try `top_k` = 10, 50, 100
4. **Time how long generation takes** (use `import time`)
5. **Count tokens** in generated text

**Submit:**
- Your modified code
- results.txt file with outputs
- Brief report:
  - What did changing parameters do?
  - Which settings produced best results?
  - How long did generation take?

**If Level 2 works → You're ready for Tier 2 tasks**

## Level 2 Help
- File I/O in Python: https://docs.python.org/3/tutorial/inputoutput.html
- Timing code: `import time; start = time.time(); ... ; print(time.time() - start)`
- Token counting: Use `len(generator.tokenizer.encode(text))`

