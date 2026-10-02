"""H2: Fine-tuning - teaching the model OUR domain.

H1 showed a 135M model can talk about vibration generally, but vaguely.
In this lesson we TEACH it our MachineGuard domain with a tiny
hand-made dataset (~24 Q&A pairs about bearing faults, z-score
detectors, autoencoders, and our project's numbers).

KEY CONCEPTS:
  - Fine-tuning = continued training on a small, domain-specific
    dataset, starting FROM the pre-trained weights.
  - We mask the PROMPT tokens (label = -100) so loss is computed only
    on the ANSWER tokens - the model learns to answer, not to repeat.
  - Full fine-tuning of even a tiny model like this is enough to
    visibly change its style and facts. Bigger models use LoRA
    (we keep it simple here - pure PyTorch loop, no magic).

HONESTY NOTE: 12 examples is a TOY dataset for teaching. Real products
use hundreds-thousands of examples. The MECHANISM is identical.

RUNTIME NOTE: full fine-tuning + CPU generation is SLOW (roughly
1 second per generated token; training ~1-2 minutes more). Expect
5-10 minutes total. If Docker Desktop is running in the background,
its virtual machine steals CPU - close it for a faster run.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT_DIR = "../models/hf_smollm2_maintenance"   # relative to lessons/
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# --- 1. Our tiny domain dataset --------------------------------------------
DATA = [
    ("What does MachineGuard do?",
     "MachineGuard reads factory machine sensors - vibration and temperature - and detects developing faults before the machine breaks down."),
    ("Why is predictive maintenance better than scheduled maintenance?",
     "Scheduled maintenance replaces parts on fixed dates, wasting healthy parts, and can still miss early failures. Predictive maintenance repairs exactly when the data says degradation started."),
    ("How does a rolling z-score detector work?",
     "It compares each point to the average and wiggle-size of the last window of points. If a point is more than about 3 standard deviations away, it raises an alert."),
    ("What is the blind spot of a z-score detector?",
     "Slow drift: because the rolling window absorbs gradual changes, slow wear growth never crosses the threshold. Only sudden spikes are caught."),
    ("How does an LSTM autoencoder detect anomalies?",
     "It is trained to compress and rebuild healthy windows. Healthy data rebuilds with small error; a sick window rebuilds badly, and the rebuild error is the alarm."),
    ("What result did MachineGuard's autoencoder achieve?",
     "It flagged 88.7 percent of fault time, versus 0.1 percent for the classical z-score baseline, on simulated machines it had never seen in training."),
    ("Why train an autoencoder only on healthy data?",
     "So it learns the shape of normal operation. Anything that deviates - wear growth or knocks - rebuilds poorly, which makes the error a useful alarm signal."),
    ("What are precision and recall in fault detection?",
     "Precision asks: of all alarms, how many were real? Recall asks: of all faulty time, how much was caught? Raising the threshold trades recall for precision."),
    ("Why is a single flagged point not an alarm?",
     "Noise occasionally crosses any threshold by chance. Production systems alarm only on sustained runs of several flagged windows in a row."),
    ("How does per-machine calibration work?",
     "After installation the system watches the machine for a healthy period and sets the alert threshold from that machine's own error distribution - the learning phase."),
    ("Why did the naive RUL baseline beat the neural network on NASA CMAPSS data?",
     "Because test engines spend most time far from failure where RUL carries no signal, so predicting a constant scored well. This shows you must always compare against a baseline."),
    ("What is the biggest unsolved problem in predictive maintenance?",
     "Organizational: most pilots never reach production because no one owns alerts or proves ROI, not because the models are wrong.")
]

def encode(question, answer):
    """Prompt tokens are masked (-100) so loss only sees the answer."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=False)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer + tokenizer.eos_token,
                      add_special_tokens=False)["input_ids"]
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids

examples = [encode(q, a) for q, a in DATA]
max_len = max(len(x) for x, _ in examples)
input_ids = torch.tensor([x + [tokenizer.pad_token_id or 128004] *
                          (max_len - len(x)) for x, _ in examples])
labels = torch.tensor([y + [-100] * (max_len - len(y)) for _, y in examples])
print(f"dataset: {len(DATA)} examples | max sequence {max_len} tokens")

# --- 2. Ask the test question BEFORE training (saves reloading the base model later)
test_q = ("You are a maintenance assistant. In ONE short sentence: "
          "what is the alert-to-action gap?")

def ask(m, q):
    m.eval()
    msgs = [{"role": "user", "content": q}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        return_tensors="pt")
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=45, do_sample=False)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True)

before = ask(model, test_q)  # model is still untrained here

# --- 3. Full fine-tuning with a plain PyTorch loop ----------------------------
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
EPOCHS = 1                                          # slim: CPU-friendly
for epoch in range(EPOCHS):
    total = 0.0
    for i in range(0, len(input_ids), 4):          # mini-batches of 4
        ids, lab = input_ids[i:i+4], labels[i:i+4]
        out = model(input_ids=ids, labels=lab)
        out.loss.backward()
        optimizer.step(); optimizer.zero_grad()
        total += out.loss.item() * len(ids)
    print(f"epoch {epoch + 1}: loss = {total / len(input_ids):.3f}")

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"\nsaved fine-tuned model to {OUT_DIR}")

# --- 4. BEFORE vs AFTER -------------------------------------------------------
print(f"\n--- BEFORE fine-tuning ---\n{before}")
print(f"\n--- AFTER fine-tuning ---\n{ask(model, test_q)}")
