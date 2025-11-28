# What It is

My project is an app which can detect my face. It will "scan" a video feed in realtime to detect faces, and if a face is found it will decide if that face belongs to me or not. If it gets a certain number of positive matches in a row it will "authorise" the user (basically a Face ID system).

# Why I chose it

Seemed cool

# Code and demo

Unfortunately, due to github file size restrictions I am not able to upload the .pth file that is my trained model here. I am aware there are ways to get around this with github large file storage but I have attached all the code (including the script used to train the model) and a demo so I think that should be good enough. <br> <br>
The code can be found in this folder [here](./) <br>
Demo: [demo](https://github.com/Boggy9888/Architecture-Exercises/blob/main/submissions/week3/do2g25/level4/Screencast%20from%202025-11-21%2018-57-21.webm)

# What I learned

Not much, I've done similar projects in the past

# What I'd improve next

Primarily need more positive samples (images of my face) under more varied conditions. Negative samples of people who look similar to me and can trigger the system (e.g. my brother) would also help. More training time on these new samples (more epochs) would also be good. <br>
Other than that I was quite rushed creating this so the code quality was not great: no type annotations, few to no docstrings, poor quality comments (should have used block comments instead). I was in a bit of a rush to complete this (had another deadline on Friday as well) but with more time I would definitely address the code quality issues as well.
