from transformers import pipeline
import time

generator = pipeline("text-generation", model="gpt2-medium")


context_options = {"Horror":"Horror activated:You are in a dark forest. There is a path to the north and a cave to the east.",
           "Fantasy":"Fantasy activated:You awake in a cell with a goblin in front of you ready to attack,to your left is a knife and to your right is a key to open the cell",
           "Sci-fi":"Sci-fi activated:You are at mobius fighting rogue andriods, you have a lightsaber at the ready but the rogue andriods do not see you yet"}

choice = input("Welcome to the AI Text Adventure!\n Pick your genre:\n-Horror\n-Fantasy\n-Sci-fi")
print(context_options[choice])
context = context_options[choice]
playing = True
with open("history.txt","a", encoding="utf-8")as file:
    file.write(context+"\n")
game_time = time.time()
while playing:
    action = input("\nWhat do you do? ")
    prompt = context + "\nPlayer: " + action + "\nStory:"
    
    output = generator(prompt, max_new_tokens=50, temperature=0.9, top_k=50)[0]['generated_text']# with temp want it not too flat cause it still needs to make sense
    new_story = output[len(prompt):].strip()
    print(new_story)
    context += "\n" + action + "\n" + new_story

    
    keep_playing = input("Do you wish to continue? (yes/no): ").strip().lower()
    if keep_playing != "yes":
        playing = False


    with open("history.txt","w", encoding="utf-8")as file:
        file.write(context)
game_time_end = time.time()
time_m = int((game_time_end-game_time)//60)
time_s = int((game_time_end-game_time)%60)

print(f"you played for {time_m} minutes and {time_s} seconds")