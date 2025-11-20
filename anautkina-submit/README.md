# Ana Utkina - Week 3 Exercise Submission

**LEVEL REACHED: [1/2/3/4]**

## LEVEL 1: ✅ [Level 1 screenshot](level1.png)

## LEVEL 2: ✅ 
- Code: [level2.py](level2.py)  
- Results: [results.txt](results.txt)
### Brief report:
*What did changing parameters do?*
*max_length:* larger values = longer outputs (but often increased repeating text)
*temperature:* higher temperature = increased variability, low temperature = generic output
*top_k:* larger top_k = increased diversity (reduced some repetition) 

*Which settings produced best results?*
temperature ≈ 1.0, top_k = 50–100, max_length = 50
These settings gave the most informative outputs while keeping creativity acceptable

*How long did generation take?*
Per run = ~3.0–3.5 s 
Total runs recorded: 135. Total compute time ≈ 7 minutes


## LEVEL 3: ✅ [C]
- Code: [compare_models.py](compare_models.pyw)
- Comparison summary: [model_comparison/model_comparison.md](model_comparison/model_comparison.md)
- Per-model outputs:
    - [distilgpt2 outputs](model_comparison/outputs/distilgpt2_outputs.txt)
    - [gpt2 outputs](model_comparison/outputs/gpt2_outputs.txt)
    - [gpt2-medium outputs](model_comparison/outputs/gpt2-medium_outputs.txt)


LEVEL 4: ✅ 
- Project name: DND Text Game
- What it does: Text adventure game with choices
- [dndgame.py](dnd-game/dndgame.py)

TIME SPENT: 3 hours

WHAT I FOUND EASY: Levels 1-2, straightforward instructions, no need for imagination
WHAT I FOUND HARD: Level 4, figure out how to make it interact with the player
