"""Tests for the Lesson 16 agent layer."""

import json
import pytest

from machineguard.reporting import (FakeLLM, write_report, Report,
                                    get_alert_summary, get_recommendation)

ANALYSIS_FAULT = {"verdict": "FAULT DETECTED", "windows_analyzed": 8634,
                  "windows_flagged": 4169, "first_sustained_alert_hour": 12.62}
ANALYSIS_OK = {"verdict": "healthy", "windows_analyzed": 8634,
               "windows_flagged": 40, "first_sustained_alert_hour": None}


def test_tools_are_deterministic_and_safe():
    s = get_alert_summary(ANALYSIS_FAULT)
    assert s["verdict"] == "FAULT DETECTED"
    assert s["windows_flagged_pct"] == pytest.approx(48.3, abs=0.1)
    r = get_recommendation(ANALYSIS_FAULT)
    assert r["severity"] == "CRITICAL"  # 48% flagged -> critical path
    assert get_recommendation(ANALYSIS_OK)["severity"] == "OK"


def test_agent_produces_valid_report():
    report = write_report(ANALYSIS_FAULT, FakeLLM())
    assert isinstance(report, Report)
    assert "FAULT" in report.title
    assert report.recommendation  # non-empty action from the TOOL, not invented


def test_agent_recovers_from_bad_answer():
    """The validation loop: first answer garbage -> retry -> valid JSON."""
    llm = FakeLLM(fail_first=True)
    report = write_report(ANALYSIS_FAULT, llm)
    assert llm.calls == 2          # one retry happened
    assert report.title


def test_agent_gives_up_after_retries():
    class AlwaysGarbage:
        def complete(self, system, user): return "not json at all"
    with pytest.raises(ValueError):
        write_report(ANALYSIS_FAULT, AlwaysGarbage(), max_retries=1)


def test_report_uses_tool_facts_not_hallucination():
    """Safety property: the action in the report comes from the rule-based
    tool, so a 'creative' LLM cannot advise anything dangerous."""
    report = write_report(ANALYSIS_FAULT, FakeLLM())
    assert report.recommendation == get_recommendation(ANALYSIS_FAULT)["action"]
