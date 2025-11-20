# Model Comparison

**Prompts used:**
- Top 5 team sports are
- The secret to happiness is
- A simple recipe for pankakes:

| Model | Params | Avg time (s) | Avg tokens | Repetition score (0-1) | Outputs |
|---|---:|---:|---:|---:|---:|
| distilgpt2 | 81.9M | 3.328 | 261.7 | 0.650 | [outputs\distilgpt2_outputs.txt](outputs\distilgpt2_outputs.txt) |
| gpt2 | 124.4M | 3.843 | 204.7 | 0.476 | [outputs\gpt2_outputs.txt](outputs\gpt2_outputs.txt) |
| gpt2-medium | 354.8M | 12.819 | 262.0 | 0.468 | [outputs\gpt2-medium_outputs.txt](outputs\gpt2-medium_outputs.txt) |



## Notes and recommendation
`distilgpt2`: smallest, fastest, OK for short demos
`gpt2`: larger and generally better quality
`gpt2-medium`: noticeably larger and slower, better quality but requires more RAM
**Recommendation:** Use `distilgpt2` for quick demos and low-resource setups; `gpt2` for improved outputs without huge resource demands; `gpt2-medium` when quality matters and you have resources (GPU/memory)