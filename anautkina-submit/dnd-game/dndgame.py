#!/usr/bin/env python3
"""
Simple interactive DnD-style text adventure demo.
"""
import random
import sys
import time
import warnings

# Suppress warnings and transformer verbosity so the player sees only game text
warnings.filterwarnings("ignore")
try:
    import logging
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
except Exception:
    # If transformers unavailable, we'll fall back later
    pass


def roll(dice: int, sides: int = 6):
    """Roll `dice` dice with `sides` sides. Return tuple (total, details)."""
    rolls = [random.randint(1, sides) for _ in range(dice)]
    return sum(rolls), rolls


class ModelWrapper:
    def __init__(self):
        self.model_name = None
        self.generator = None
        # Try preferred models in order
        for name in ("gpt2-medium", "gpt3"):
            try:
                # lazy import: may raise if transformers not installed
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

                self.model_name = name
                self.generator = pipeline(
                    "text-generation", model=name, tokenizer=name, device=-1
                )
                break
            except Exception:
                self.model_name = None
                self.generator = None

    def available(self):
        return self.generator is not None

    def generate(self, prompt: str, max_new_tokens: int = 120):
        if self.generator is None:
            return self._fallback(prompt)
        try:
            out = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.8)
            text = out[0]["generated_text"]
            # The pipeline returns prompt+continuation; strip prompt
            cont = text[len(prompt) :]
            return cont.strip()
        except Exception:
            return self._fallback(prompt)

    def _fallback(self, prompt: str):
        # Very small canned continuation generator when transformers not present
        choices = [
            "A shadow moves between the trees, and you hear breathing close by.",
            "You step cautiously forward, the undergrowth crackling underfoot.",
            "A low growl rolls through the woods; the hairs on your neck rise.",
            "Moonlight filters through the leaves, revealing a narrow path.",
            "You grip your weapon tighter, waiting for whatever comes next."
        ]
        return " " + random.choice(choices)


def choose(prompt: str, options):
    options = list(options)
    while True:
        print(prompt)
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        choice = input("> ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        # allow typing the option itself (case-insensitive)
        for opt in options:
            if choice.lower() == opt.lower():
                return opt
        print("Please choose a valid number or option name.")


def narrative_intro(race, cls):
    return (
        f"You awaken alone in a dark, silent forest. The trees loom tall and cold. "
        f"You are a {race} {cls}, your senses sharpened by training and fate. "
        "A noise to your right—something stepped on a stick and it cracked loudly."
    )


def run_game():
    print("Welcome to the quick DnD demo. (type 'quit' to exit anytime)\n")

    name = input("What is your name, adventurer? ").strip() or "Player"

    race = choose("Choose your race:", ["Human", "Elf", "Dwarf"])
    cls = choose("Choose your class:", ["Warrior", "Rogue", "Mage"])

    # Initialize model
    model = ModelWrapper()
    if model.available():
        print(f"(Using transformer model: {model.model_name})\n")
    else:
        print("(No transformer available—using a lightweight fallback generator.)\n")

    # Initial scene
    context = f"{narrative_intro(race, cls)}\n"
    print(context)

    # Offer a simple ability check example using dice roll
    print("A twig snaps to your right. You decide how to react.\n")

    # We'll run a short interactive loop (approx 6 player turns)
    max_turns = 6
    turn = 0

    while turn < max_turns:
        # Prompt player for action
        action = input("What do you do? ").strip()
        if not action:
            print("You stand motionless, listening.")
            action = "You stand still and listen."
        if action.lower() in ("quit", "exit"):
            print("You decide to leave the woods. The demo ends here. Farewell.")
            return

        # Player-initiated dice example: if they include words like 'attack' or 'sneak', do a check
        lowered = action.lower()
        roll_result = None
        if any(k in lowered for k in ("attack", "strike", "hit")):
            total, detail = roll(1, 20)
            roll_result = f"You roll a d20: {total} (rolls: {detail})"
        elif any(k in lowered for k in ("sneak", "stealth", "hide")):
            total, detail = roll(2, 6)
            roll_result = f"You roll 2d6 for stealth: {total} (rolls: {detail})"
        elif any(k in lowered for k in ("cast", "spell", "magic")):
            total, detail = roll(1, 8)
            roll_result = f"You roll a d8 for magical control: {total} (rolls: {detail})"

        if roll_result:
            print(roll_result)

        # Build prompt for model: include short context, player action, and a short instruction to continue
        prompt = (
            f"Scene: {context}\n"
            f"Player ({name}, {race} {cls}) does: {action}\n"
            "Continue the story with one short narrative paragraph describing what happens next. "
            "Keep it cinematic and direct. Do not ask questions."
        )

        # Generate continuation
        cont = model.generate(prompt, max_new_tokens=120)
        # Trim and ensure it's a paragraph
        cont = cont.strip()
        if not cont:
            cont = "Silence follows; nothing else happens for a moment."

        # Print generated continuation
        print("\n" + cont + "\n")

        # Append to context to keep history manageable; limit context length by trimming older text if needed
        context += f"Player does: {action}\n{cont}\n"
        if len(context) > 3000:
            context = context[-3000:]

        turn += 1

    # Conclude the demo with a short wrap-up
    ending = (
        "As the final echoes settle, the path ahead opens into a moonlit clearing. "
        "Whatever threat you'd found in the trees has retreated — for now. You breathe out, "
        "and the night seems almost calm. This is only the beginning of your legend."
    )
    print(ending)
    print("\nDemo complete. Thank you for playing!")


if __name__ == "__main__":
    try:
        run_game()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
