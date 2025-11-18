Subject: James Martineau (jlm1g25) - Diagnostic Task Submission

LEVEL REACHED: [1/2/3/4]

LEVEL 1: ✅ ![alt text](level1.png)

LEVEL 2: ✅ [Code link] ![alt text](image.png)
- Brief summary of findings
    Not setting min_new_tockens parameter gives inconsistent lengths
    The model can generate consecutive sequences of \n (often with a length of 4) which can result in large gaps in the response. Sometimes these are very large as shown in ![alt text](largeSequence.png) so the parameter stop_strings = ["\n\n\n\n\n"] can be used to mitigate this with the trade off that the response will be cut of and may not of been completed. However stop strings doesnt seem to be suported in the distilgpt2 model.
    On the other hand the model can not generate a \n for a while which can result in long single lines of text that are difficult to read when writing to a text file.
    The model often gets stuck in loops which I have learnt is due to greedy decoding, therefore I have changed the model to do sampling instead.
    This on its own doesnt seem to help the looping situation therefore I tried changing the temperature, top_k and top_p to help combat this looping.
    setting temperature to low made the looping worse, setting it too high made it too random
    setting top_p too low made it less creative whereas to high made it loop a lot
    setting top_k too low also made it less creative whereas to high made it talk nonsense
    Best I found: temperature ~ 0.85, top_p ~ 0.9, top_k ~ 40
    The number of tokens also seems to have an effect on the amount of looping
    Generation time is correlated with max new token (provding max and min are the same)
    100 tokens ~ 3.25-3.55 seconds
    200 tokens ~ 6.5-7 seconds
    400 tokens ~ 13.5-14.5 seconds
    800 tokens ~ 28 seconds
    This suggests a linear relationship

    

- What did changing parameters do?
- Which settings produced best results?
- How long did generation take?

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
