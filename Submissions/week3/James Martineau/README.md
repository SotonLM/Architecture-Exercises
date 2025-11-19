Subject: James Martineau (jlm1g25) - Diagnostic Task Submission

LEVEL REACHED: [1/2/3/4]

LEVEL 1: ✅ [text](<level 1>)![alt text](level1.png) 

LEVEL 2: ✅ [\[Code link\]](<level 2>) ![alt text](image.png)
- Brief summary of findings
    Not setting min_new_tockens parameter gives inconsistent lengths
    The model can generate consecutive sequences of \n (often with a length of 4) which can result in large gaps in the response. Sometimes these are very large as shown in ![alt text](largeSequence.png) so the parameter stop_strings = ["\n\n\n\n\n"] can be used to mitigate this with the trade off that the response will be cut of and may not of been completed. However stop strings doesnt seem to be suported in the distilgpt2 model. Another approach is seeting the repetition penalty to a value slightly above 1 which seems to eliminate the long strings of "\n".
    On the other hand the model can not generate a \n for a while which can result in long single lines of text that are difficult to read when writing to a text file.
    The model often gets stuck in loops which I have learnt is due to greedy decoding, therefore I have changed the model to do sampling instead.
    This on its own doesnt seem to help the looping situation therefore I tried changing the temperature, top_k and top_p to help combat this looping.
    setting temperature to low made the looping worse, setting it too high made it too random
    setting top_p too low made it less creative whereas to high made the converstion go off in a tangent
    setting top_k too low also made the looping worse in what it was saying whereas to high made it talk nonsense
    As the number of generated tokens increased the chance of looping eventually occurring also increased.
    Best I found: max_new_tokens = 100, max_temperature ~ 0.5,top_k ~ 50, top_p ~ 0.92, repetition_penalty ~ 1.075
    The model still goes of in tangents as shown in the image above.
    Responses often cut of mid sentence at the end so not sure how to fix this 
    Generation time is correlated with max new token (providing max and min are the same)
    100 tokens ~ 3.25-3.55 seconds
    200 tokens ~ 6.5-7 seconds
    400 tokens ~ 13.5-14.5 seconds
    800 tokens ~ 28 seconds
    This suggests a linear relationship

LEVEL 3: ✅ [Option C]
- [\[Link to work\]](<level 3>)
See time_results.txt for evidence for the generation speed
| Model | Model size | Generation speed | When to use |
|-------|-----------|----------|------|------|
| distilgpt2 |88.2M parameters| time used as a bench mark  | For quick responses to simple questions on systems with limited space |
| gpt2   | 124M parameters 0.55GB  | average of time of +43.5% seconds| For a more detailed response for systems with more space |
| gpt2-medium| 355M parameters 1.52GB | average of time of +241.86% seconds| For longer responses requiring more detail|
The output quality seems to be the same in my opinion


TIME SPENT: 3 hours

WHAT I FOUND EASY: Setting up the environment and programming the python logic surrounding the models
WHAT I FOUND HARD: Fine tuning the model to produce coherent results
QUESTIONS: How do you make the model finish it sentence without being cut of by the token count?
