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

# Output Quality
Generally the output quality for `distilgpt2` was the worst, followed by `gpt2` and then `gpt2-medium`.

Examples:
- `distilgpt2`'s output was sometimes incoherent: "The world is changing. Robots are beginning to be able to control the environment in a way that is very different from traditional medicine. This is the end for the robots." and sometimes repetitive: "We need to learn from the world in a constructive and helpful manner.
Learning from the world in a constructive and helpful manner.
We need to learn from the world in a constructive and helpful manner.
Learning from the world in a constructive and helpful manner.
We need to learn from the world in a constructive and helpful manner." Additionally `distilgpt2` sometimes decides to include many newlines in it's generated output (which in this case it did for the prompt "Newton's third law is defined as") and all in all this means `distilgpt2`'s outputs were the lowest quality overall.
- `gpt2`'s output was much better than `distilgpt2`'s output as it did not repeat itself and generally made more insightful comments than `distilgpt2` (e.g. "The most pivotal technology from the past 50 years is the Internet, and it's not just about how many people are using it. It's about how much information is available at once.") however sometimes it still generated an output that did not make much sense / was incomplete: e.g. "Data types that should be trained include:
Type 1 Type 2 Type 3 Type 4 Type 5 Type 6 Type 7 Type 8 Type 9 Type 10 Type 11 Type 12 Type 13 Type 14 Type 15 Type 16 Type 17 Type 18 Type 19 Type 20 Type 21 Type 22 Type 23 Type 24" in this example `gpt2` does not list actual data types and because of this this particular output is not very good.
- `gpt2-medium` had the best overall responses as it did not repeat itself and generated more insightful / less nonsensical outputs compared to `gpt2`. For example, for the prompt "Artificial Intelligence will lead to" part of `gpt2`'s output was:
"Artificial Intelligence will lead to more than just a new computing tool. It will also bring more insight into the future of AI. And even if it doesn't happen, it will provide more understanding of human behavior because we're more likely to make mistakes in our own lives." the last sentence of this extract of `gpt2`'s output does not make sense. Meanwhile `gpt2-medium`'s output makes far more sense (example extract: "This means that the benefits of automation and artificial intelligence will be felt not only by those who create these goods and services, but also by those who consume them. So how do we protect jobs and wages from automation?We must embrace the technological revolution that will eliminate most of the jobs and the loss of wealth that depend on humans. We must also embrace the technological revolution that will replace a large share of our jobs.").

# Model Size
- `distilgpt2` ~ 82 million parameters
- `gpt2` ~ 124 million parameters
- `gpt2-medium` ~ 355 million parameters

# Comparison Table
| Model | Average time taken / s | Relative output quality | Model size / million parameters |
| ------------- | ---------------------------------------------- | ----------------------- | ------------------------------- |
| distilgpt2 | 5.9033 | Worst | 82 |
| gpt2 | 7.5520 | Medium | 124 |
| gpt2-medium | 19.5667 | Best | 355 |

# Use Cases
- `distilgpt2` would be best for use cases where speed is extremely important and subpar output quality is acceptable as it had the fastest output generation (an average time taken of 5.9033s across the five prompts) but it had the worst relative output quality.
- `gpt2` would be best for use cases where speed is still important but better output quality than `distilgpt2` is more important than a slightly quicker generation time as `gpt2` was not that much slower than `distilgpt2` (7.5520s compared to 5.9033s) but it had a notably better output quality.
- `gpt2-medium` would be best for use cases where output quality is far more important than generation time. Coming in with an average time of 19.5667s the generation time of `gpt2-medium` is over double that of `gpt2` and over triple that of `distilgpt2`. However for use cases where the extra time does not matter `gpt2-medium` is the best choice as it has significantly better output quality than `distilgpt2` and notably better output quality than `gpt2`.
