I chose option C (Compare) with the recommended models (`distilgpt2`, `gpt2`, `gpt2-medium`)

# Generation Speed
`distilgpt2` was consistently the quickest model with the shortest generation speed, followed by `gpt2` and then `gpt2-medium`. 
This is shown by this table:

| prompt | distilgpt2 | gpt2 | gpt2-medium |
| ------ | ---------- | ---- | ----------- |
| The most boring topic in computer science is | 6.153108 | 8.364003 | 19.569328 |
| Artificial Intelligence will lead to | 5.879485 | 4.628390 | 19.598192 |
| The most pivotal technology from the past 50 years is | 5.705082 | 8.181596 | 19.876790 |
| Reinforcement learning is best suited | 5.692315 | 8.085861 | 19.184039 |
| Newton's third law is defined as | 6.086588 | 8.500262 | 19.605249 |

Average time each model took across all prompts:
- `distilgpt2`: 5.9033s
- `gpt2`: 7.5520s
- `gpt2-medium`: 19.5667s

