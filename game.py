from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

has = []

def tastertest():
    a = random.choice(["Judge ", "Chef "])
    length = random.randint(3, 15)
    for i in range(length):
        if i % 2 == 0:
            a += random.choice("aeiou")
        else:
            a += random.choice("bcdfghjklmnpqrstvwxyz")
    return a

# Load model name
with open("./config.txt") as f:
    model_name = f.read().strip()

print("Loading model:", model_name)

model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # puts as much as possible on GPU
    trust_remote_code=True    # required for Qwen models
)

model.eval()
print("Done!")

def run(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def choose_ingredients():
    if not has:
        print("You have no ingredients yet.")
        return []

    print("Inventory:")
    for i, item in enumerate(has, start=1):
        print(f"{i}. {item}")

    chosen = []
    while True:
        to = input("Enter ingredient number to use, or 'done': ").strip().lower()
        if to == "done":
            break
        if not to.isdigit():
            print("Please enter a valid number or 'done'.")
            continue
        idx = int(to)
        if idx < 1 or idx > len(has):
            print("Invalid index.")
            continue
        chosen.append(has[idx - 1])
    return chosen

while True:
    print("\n=== AI Cooking ===")
    print("What would you like to do?")
    print("Buy  -> Get specified ingredients")
    print("Mix  -> Mix specified ingredients")
    print("Cook -> Cook specified ingredients")
    print("Taste -> Let a judge taste your food")
    print("Other -> Specify a custom one-word command")
    print("Quit -> Exit game")
    print("Inventory:", ", ".join(has) if has else "(empty)")

    cmd = input("> ").strip().lower()

    if cmd == "quit":
        print("Goodbye, Chef.")
        break

    if cmd == "buy":
        item = input("What to buy? ").strip()
        if item:
            has.append(item)
            print(f"Bought {item}!")
        continue

    if cmd in ("mix", "cook", "taste", "other"):
        if not has:
            print("You have no ingredients to use.")
            continue

        if cmd == "other":
            action = input("Command (one word, e.g. 'freeze', 'blend'): ").strip()
            if not action:
                print("No command given.")
                continue
        else:
            action = cmd

        tomix = choose_ingredients()
        if not tomix:
            print("No ingredients selected.")
            continue

        if cmd != "taste":
            todo = (
                f"What would happen if you {action} {', '.join(tomix)}? "
                f"Return on the first line what it is called, and on the next lines explain what it is."
            )
            print("Generating...")
            raw = run(todo)
            name = raw.splitlines()[0]

            print("\n--- Result ---")
            print(raw)
            print(f"\nYou now have a(n) {name}!")

            for item in tomix:
                if item in has:
                    has.remove(item)

            has.append(name)
            continue

        else:
            todo = (
                f"If you were a chef and you tried {', '.join(tomix)}, "
                f"give a 1-10 score and some tips/reaction."
            )
            print("Generating...")
            raw = run(todo)

            print("\n--- Result ---")
            print(tastertest(), "says:", raw)

            for item in tomix:
                if item in has:
                    has.remove(item)

            continue

    print("Unknown command. Try: Buy, Mix, Cook, Taste, Other, or Quit.")
