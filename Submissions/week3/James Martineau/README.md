Subject: James Martineau (jlm1g25) - Diagnostic Task Submission

LEVEL REACHED: [3]

LEVEL 1: ✅ [text](<level 1>)![alt text](level1.png) 
<img width="1126" height="857" alt="image" src="https://github.com/user-attachments/assets/a2ddbc05-cce1-41c3-b6c3-e8427b9759ac" />
LEVEL 2: ✅ [\[Code link\]](<level 2>) ![alt text](image.png)



- Not setting the min_new_tockens parameter and only the max_new_tokens parameter gives inconsistent lengths.
- The model can generate consecutive sequences of \n (often with a length of 4) which can result in large gaps in the response. 
    Sometimes these are very large as shown in the image below:
<img width="1126" height="37" alt="image" src="https://github.com/user-attachments/assets/471d9aee-6916-4206-aa15-ac3c095e8071" />

- The parameter stop_strings = ["\n\n\n\n\n"] can be used to mitigate this with the trade off that the response will be cut of and may not of been completed. However I couldn't get this parameter to work.
- On the other hand the model may not generate a \n for a while which can result in long single lines of text that are difficult to read when writing to a text file.
- The model often gets stuck in loops which I have learnt is due to greedy decoding, therefore I have changed the model to do sampling instead. 
- This on its own doesnt seem to help the looping situation therefore I tried changing the temperature, top_k and top_p to help combat this looping.
    - Setting temperature to low made the looping worse, setting it too high made it too random.
    - Setting top_p too low made it less creative whereas to high made the conversation go off in a tangent
    - Setting top_k too low also made the looping worse in what it was saying whereas to high made it talk nonsense
    - As the number of generated tokens increased the chance of looping eventually occurring also increased.
- Another approach is setting the repetition penalty to a value slightly above 1 which seems to eliminate the long strings of "\n" and looping altoghether.
- Best I found: max_new_tokens = 100, max_temperature ~ 0.5,top_k ~ 50, top_p ~ 0.92, repetition_penalty ~ 1.075
    - The model still goes of in tangents as shown in the image below.
    - Responses often cut of mid sentence at the end so not sure how to fix this
    <img width="1192" height="679" alt="image" src="https://github.com/user-attachments/assets/31f13a57-6f90-4543-bb44-d2e6ac893109" />
- Generation time is correlated with max new token (providing max and min are the same)
    - 100 tokens ~ 3.25-3.55 seconds
    - 200 tokens ~ 6.5-7 seconds
    - 400 tokens ~ 13.5-14.5 seconds
    - 800 tokens ~ 28 seconds
- This suggests a linear relationship

LEVEL 3: ✅ [Option C]
- [\[Link to work\]](<level 3>)
<table>
  <caption><strong>Model comparison</strong></caption>
  <thead>
    <tr>
      <th>Model</th>
      <th>Model size</th>
      <th>Generation speed</th>
      <th>When to use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>distilgpt2</td>
      <td>88.2M parameters</td>
      <td>Time used as a benchmark</td>
      <td>For quick responses to simple questions on systems with limited space</td>
    </tr>
    <tr>
      <td>gpt2</td>
      <td>124M parameters / 0.55 GB</td>
      <td>Average time +43.5% seconds more than distilgpt2</td>
      <td>For a more detailed response on systems with more space</td>
    </tr>
    <tr>
      <td>gpt2-medium</td>
      <td>355M parameters / 1.52 GB</td>
      <td>Average time +241.86% seconds more than distilgpt2</td>
      <td>For longer responses requiring more detail</td>
    </tr>
  </tbody>
</table>


- The output quality seems to be the same in my opinion
- Evidence for how I calculated the average generation time can be found in the level 3 folder in submissions


- TIME SPENT: 3 hours
- WHAT I FOUND EASY: Setting up the environment and programming the python logic surrounding the models
- WHAT I FOUND HARD: Fine tuning the model to produce coherent results
- QUESTIONS: How do you make the model finish its sentence without being cut of by the token count?
