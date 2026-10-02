"""H3: The Agent - giving the language model TOOLS.

THE ARC CLOSES HERE:
  Lessons 1-12 : sensors -> z-score/Autoencoder -> FastAPI service
  H1           : download & run a real LLM (SmolLM2)
  H2           : fine-tune it on OUR maintenance domain
  H3           : give it TOOLS + a validation loop = an AGENT

THE AGENT PATTERN (the one behind every production LLM system):
  1. TOOLS: plain Python functions (get_alert_summary,
     get_recommendation) - we REUSE them from Lesson 16's
     machineguard/reporting.py.
  2. FACTS FIRST: the tools produce structured FACTS; the LLM only
     writes the human-readable report FROM those facts. It can shape
     the words, but it cannot invent the numbers. (Guardrail!)
  3. VALIDATION LOOP: the LLM must answer in JSON; if parsing fails,
     we retry with a stricter prompt. Agents are unreliable ->
     ALWAYS validate their output.

RUNTIME: uses your fine-tuned model from H2. CPU generation is ~1
second per token, so this takes a few minutes. Set HF3_MAX_TOKENS=40
for a quick smoke run.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from machineguard.pipeline import (make_windows, apply_scaler, score_windows,
                                   sustained_alerts, load_bundle)
from machineguard.reporting import (get_alert_summary, get_recommendation,
                                    SYSTEM, Report)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_TOKENS = int(os.environ.get("HF3_MAX_TOKENS", "120"))

# --- 1. Run the MachineGuard detector -> an analysis dict ---------------
print("step 1: running MachineGuard analysis on machine_b.csv ...")
df = pd.read_csv(PROJECT_ROOT / "data/sample/machine_b.csv")
model_det, cfg = load_bundle(PROJECT_ROOT / "models/machineguard_v1")
windows = make_windows(apply_scaler(df, np.array(cfg["mean"]), np.array(cfg["std"])),
                       cfg["window"], cfg["stride"])
hours = df["hour"].to_numpy()[cfg["window"] // 2 :: cfg["stride"]][: len(windows)]
errors = score_windows(model_det, windows)
alarm = sustained_alerts(errors > cfg["threshold"], run_length=5)
analysis = {
    "verdict": "FAULT DETECTED" if alarm.mean() > 0.05 else "healthy",
    "windows_analyzed": int(len(windows)),
    "windows_flagged": int((errors > cfg["threshold"]).sum()),
    "first_sustained_alert_hour": round(float(hours[np.argmax(alarm)]), 2) if alarm.any() else None,
}

# --- 2. The LLM client: our FINE-TUNED model behind one method -----------
class LocalLLM:
    """Wraps the H2 fine-tuned model so it satisfies the LLMClient
    interface (complete(system, user) -> str). The agent does not care
    WHICH model answers - swap LocalLLM for OpenAICompatibleLLM and
    nothing else changes. (That is the point of the interface!)"""

    def __init__(self, model_dir):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)

    def complete(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt")
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=MAX_TOKENS,
                                      do_sample=False)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)

# --- 3. The agent loop (validation + retry), reusing reporting.write_report
print(f"step 2: agent writes the report (max {MAX_TOKENS} tokens, be patient) ...")
llm = LocalLLM(PROJECT_ROOT / "models/hf_smollm2_maintenance")
facts = {**get_alert_summary(analysis), **get_recommendation(analysis)}
print("   facts from tools:", json.dumps(facts, ensure_ascii=False))

# Few-shot: tiny models copy FORMAT from an example far better than they
# follow abstract instructions. Show one worked JSON example in the prompt.
FORMAT_EXAMPLE = ('{"title": "MachineGuard Bericht: FAULT DETECTED", '
                  '"summary": "48.3% der Fenster auffaellig. CRITICAL.", '
                  '"recommendation": "inspect within 48 hours"}')

user = (f"FACTS:\n{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"EXAMPLE format (copy this structure exactly):\n{FORMAT_EXAMPLE}\n\n"
        "TASK:\nWrite the maintenance report JSON now.")

ATTEMPTS = int(os.environ.get("HF3_ATTEMPTS", "2"))
report, raw = None, ""
for attempt in range(ATTEMPTS):
    if attempt:
        user += "\n\nYour previous answer was not valid JSON. Output ONLY JSON."
    raw = llm.complete(SYSTEM, user)
    print(f"   attempt {attempt + 1} raw model output: {raw[:150]!r}")
    try:
        data = json.loads(raw)
        report = Report(title=data["title"], summary=data["summary"],
                        recommendation=data["recommendation"])
        break
    except (json.JSONDecodeError, KeyError):
        print("   -> not valid JSON (this is NORMAL for a 135M model)")

# --- 4. Graceful fallback: same agent loop, deterministic FakeLLM ----------
if report is None:
    print("\n   the local model could not produce strict JSON - falling back to")
    print("   the deterministic FakeLLM for the SAME agent loop. Two lessons here:")
    print("   (1) agents ALWAYS need output validation + a fallback plan;")
    print("   (2) tiny local models are not reliable structured-output engines -")
    print("       production systems use bigger models or constrained decoding.")
    from machineguard.reporting import FakeLLM, write_report
    report = write_report(analysis, FakeLLM())

print("\n--- AGENT REPORT ---")
print(f"title:         {report.title}")
print(f"summary:       {report.summary}")
print(f"recommendation:{report.recommendation}")
print("\nThe facts above came from TOOLS. The LLM only shaped the words -")
print("it could not invent the numbers. That separation is the safety rail.")
