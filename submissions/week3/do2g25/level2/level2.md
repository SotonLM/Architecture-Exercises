# What changing the parameters did

- `max_length`: Making this smaller made the input generally shorter, and vice versa. This is in line with the documentation which states that the `max_length` parameter corresponds to the length of the input prompt + `max_new_tokens`.
- `temperature`: A `temperature` of 0.5 resulted in the model repeating it's output very often. A `temperature` of 1 is the default value that resulted in 'normal' model output. A `temperature` of 1.5 resulted in more coherent sentences overall even though generally higher temperature is supposed to make the model more creative (but also more prone to errors). In general, lower temperature makes the model more predictable.
- `top_k`: This parameter controls how many tokens are considered each time it is time to choose which of the highest probability tokens will be next generated. A higher `top_k` resulted in more creative outputs while a lower `top_k` resulted in more predictable outputs.

 # What settings produced the best results

 For me, a temperature of 1.5 and a top_k of 50 actually appeared to produce the best (most coherent) results. The model in general is pretty rubbish though as many outputs it generates in reponse to prompts are not meaningful.

 # How long generation took

 Generation was very fast for me, a few seconds per prompt, about 20 - 30s overall for five prompts.
