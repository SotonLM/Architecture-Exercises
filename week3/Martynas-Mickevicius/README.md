# Subject: Martynas Mickevicius - Diagnostic Task Submission

LEVEL REACHED: [2]

forked repo link: [https://github.com/Phantomnz/Architecture-Exercises](https://github.com/Phantomnz/Architecture-Exercises)

LEVEL 1: ✅ screenshots in folder: week3/Martynas-Mickevicius names: o1t1.png and o2t1.png

LEVEL 2: ✅ code: lvtwo.py output: results.txt  in same folder as the screenshots

Level 2 findings:

generation time varied between 5-30 seconds

changing variable from max-length to max_new_tokens made some auto adjustments the terminal was making no longer necessary when set to higher numbers (100 and above)

increasing temperature a lot seems to make the model no longer able to concentrate on the prompt and go on ramblings, keep this relatively low

but low temperature seems to have repeating text? but when it doesn't repeat text it is making more sense

reducing top k means reducing output diversity

I have these massive gaps in my output, my file isn't broken it's just spread out due to this

the output I'll stick with has temp = 0.5, top_k = 50, max_new_tokens = 100

TIME SPENT: 1.5 hours

WHAT I FOUND EASY: level 1 and downloading libraries lol

WHAT I FOUND HARD: figuring out how to get token number

QUESTIONS: Is there way to get rid of the massive spaces in output? Ali found the same issue so I'm wondering if anyone found a fix

