```
Subject: Hathan Khatkar - Diagnostic Task Submission

LEVEL REACHED: [1/2/3/4]

LEVEL 1: ✅ 

![UI Screenshot](level1_sc.png)

LEVEL 2: ✅ 
[Code link](diagnostic_task.py)
[Results link](result.txt)
- Brief summary of findings
 - What did changing parameters do?
 max_length - the model overrided this parameter so I switched to use max_new_tokens which limits no. tokens in the generated output 
 temperature - increasing temperature, increased scope of generated text, so text becomes less specific and more exploratative.
 top_k - decreasing top_k, increased likeliness of the generated output being similar to previous outputs controlling variation of responses.
 
  - Which settings produced best results?
  Depends on the prompt and expected response if your expecting more deterministic responses then having a lower top_k will be beneficial
  or if you valued more creative responses higher temperature will compliment this.
  
But there are more other more logical ways to determine this e.g. measuring semantic similarity score with embedding models.
For my example prompt 'to make a sandwich you need to' the most relevant result were from (max_length=50, temperature=1.0, top_k=50)

  - How long did generation take?
 Around 0.3 - 0.8s depending on max tokens.


LEVEL 3: ✅ [Option A/B/C/D]
- [Link to work]
- Brief summary

LEVEL 4: ✅ [If completed]
- Project name: ___
- What it does: ___
- [Demo link]

TIME SPENT: ___ hours

WHAT I FOUND EASY: ___
WHAT I FOUND HARD: ___
QUESTIONS: ___
```