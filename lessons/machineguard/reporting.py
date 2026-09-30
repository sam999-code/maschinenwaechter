"""Lesson 16: The Agent layer - from alert to maintenance report.

MachineGuard detects the fault. But a factory doesn't want a JSON blob;
it wants a report: what happened, how urgent, what to do. An LLM agent
turns structured analysis into natural language.

THE AGENT PATTERN (you studied this - now we build it for real):
  1. TOOLS: plain Python functions the model may call
     (get_alert_summary, get_recommendation)
  2. CONTEXT ASSEMBLY: we put the tool OUTPUTS into the prompt
  3. GENERATION: an LLMClient completes the report
  4. VALIDATION + RETRY: the model must answer in strict JSON;
     if parsing fails, we retry once with a stricter prompt.
     (Agents are unreliable -> always validate their output!)
  5. The LLM client is an INTERFACE: FakeLLM for tests/offline,
     a real OpenAI-compatible client behind an env var for production.
"""

import json
import os
from dataclasses import dataclass
from typing import Protocol


# --- 1. TOOLS: ordinary functions ------------------------------------------------
def get_alert_summary(analysis: dict) -> dict:
    """Facts about what the detector found."""
    return {
        "verdict": analysis["verdict"],
        "windows_flagged_pct": round(100 * analysis["windows_flagged"]
                                     / max(1, analysis["windows_analyzed"]), 1),
        "first_alert_hour": analysis.get("first_sustained_alert_hour"),
    }


def get_recommendation(analysis: dict) -> dict:
    """Rule-based severity -> action. (Rules keep the agent SAFE:
    the LLM writes the words, but it cannot invent the action.)"""
    pct = 100 * analysis["windows_flagged"] / max(1, analysis["windows_analyzed"])
    if analysis["verdict"] == "FAULT DETECTED" and pct > 30:
        return {"severity": "CRITICAL",
                "action": "stop machine at next safe opportunity, inspect bearings"}
    if analysis["verdict"] == "FAULT DETECTED":
        return {"severity": "WARNING",
                "action": "schedule inspection within 48 hours"}
    return {"severity": "OK", "action": "continue normal operation"}


# --- 2. THE LLM CLIENT INTERFACE -------------------------------------------------
class LLMClient(Protocol):
    """Any text model behind one method. Swappable = testable."""
    def complete(self, system: str, user: str) -> str: ...


class FakeLLM:
    """Deterministic stand-in. Understands our JSON contract so the whole
    agent runs offline and in tests. NOT intelligent - that's the point:
    architecture first, real model later."""

    def __init__(self, fail_first: bool = False):
        self.calls = 0
        self.fail_first = fail_first  # demo: first answer is garbage

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return "I think the machine is probably fine??"  # not JSON!
        facts = json.loads(user.split("FACTS:", 1)[1].split("TASK:", 1)[0].strip())
        return json.dumps({
            "title": f"MachineGuard Bericht: {facts['verdict']}",
            "summary": (f"Detektor-Stufe: {facts['severity']}. "
                        f"Erste Alarme ab Stunde {facts['first_alert_hour']}. "
                        f"{facts['windows_flagged_pct']}% der Fenster auffällig."),
            "recommendation": facts["action"],
        }, ensure_ascii=False)


class OpenAICompatibleLLM:
    """The production client. Reads the API key from the environment
    (never hardcode keys!) and speaks the OpenAI chat format, so it
    works with OpenAI, Mistral, Groq, or a local Ollama server."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None):
        self.model, self.base_url = model, base_url

    def complete(self, system: str, user: str) -> str:
        try:
            import httpx
        except ImportError as e:
            raise SystemExit("pip install httpx (or run with FakeLLM)") from e
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Set OPENAI_API_KEY (or run with FakeLLM)")
        url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
        r = httpx.post(url,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": self.model, "temperature": 0,
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}]},
                       timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# --- 3. THE AGENT LOOP -----------------------------------------------------------
SYSTEM = ("You are MachineGuard, a predictive-maintenance assistant for German "
          "factories. Answer ONLY with a JSON object with keys: "
          "title, summary, recommendation. German language.")

@dataclass
class Report:
    title: str
    summary: str
    recommendation: str


def write_report(analysis: dict, llm: LLMClient, max_retries: int = 1) -> Report:
    facts = {**get_alert_summary(analysis), **get_recommendation(analysis)}
    user = (f"FACTS:\n{json.dumps(facts, ensure_ascii=False)}\n\n"
            "TASK:\nWrite the maintenance report JSON now.")
    last_error = None
    for attempt in range(max_retries + 1):
        if attempt:  # retry hint: agents follow instructions better second time
            user += "\n\nYour previous answer was not valid JSON. Output ONLY JSON."
        try:
            data = json.loads(llm.complete(SYSTEM, user))
            return Report(title=data["title"], summary=data["summary"],
                          recommendation=data["recommendation"])
        except (json.JSONDecodeError, KeyError) as e:
            last_error = e
    raise ValueError(f"agent failed after {max_retries + 1} attempts: {last_error}")
