"""H1: Your first HuggingFace model - download, load, generate.

THE BIG IDEA:
Until now you TRAINED models from zero. Today you do what most AI
engineers do every day: download a model someone else trained (from the
HuggingFace Hub - the "GitHub of ML models") and USE it.

TWO NEW OBJECTS:
  - tokenizer:  text  <->  numbers   (models eat numbers, not letters)
  - model:      the trained network. NOTE: it is a torch.nn.Module -
                THE SAME KIND OF OBJECT you built in Lesson 6!
                HuggingFace is "just" PyTorch + sensible defaults.

THE MODEL: SmolLM2-135M-Instruct - a real, modern, tiny open model
(~270 MB download the first time, then cached). Small enough for your
CPU, real enough to be impressive.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"  # tiny but real

# --- 1. Download + load (first run: ~270MB, then cached) -------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# Proof that this is "our world": the HF model IS a PyTorch module
print(f"model type: {type(model).__mro__[0].__name__}")   # PreTrainedModel
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"it is a torch.nn.Module? {isinstance(model, torch.nn.Module)}")

# --- 2. A question from OUR domain: bearing faults --------------------------
question = ("You are a maintenance engineer. In 2-3 sentences, explain why "
            "vibration monitoring can predict a failing bearing in a motor.")

# "Instruct" models expect a CHAT FORMAT (a list of messages), not raw text
messages = [{"role": "user", "content": question}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt")

# --- 3. Generate (this is a forward pass + sampling loop, like our decoder) -
# transformers 5.x returns a dict-like BatchEncoding: unpack it with **
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=120, do_sample=False)

answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
print("\n--- model's answer ---")
print(answer)

# --- 4. Look inside the tokenizer (numbers <=> words) -----------------------
print("\n--- tokenizer demo ---")
ids = tokenizer("bearing vibration temperature", return_tensors="pt")["input_ids"]
print("text -> numbers:", ids[0].tolist())
print("numbers -> text:", tokenizer.decode(ids[0]))
