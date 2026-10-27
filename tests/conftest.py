"""Shared fixtures: a fake Gemini model so tests never hit the network."""

from __future__ import annotations

import threading
import time

import pytest
from langchain_core.messages import AIMessage

import src.agents as agents

SCANNER_REPLY = """ISSUES_FOUND: YES
ISSUE_SUMMARY:
- **Security**: SQL injection via string concatenation
- **Bug**: connection is never closed

#### Summary
The function builds SQL from user input.

#### Issues
1. **Security**: SQL injection.
"""
REFACTOR_REPLY = "```python\ndef get_user(username):\n    return username\n```"
DOCS_REPLY = "## Module Overview\nFetches a user."


class FakeLLM:
    """Returns a canned reply per agent, chosen from the system prompt."""

    def __init__(self, latency: float = 0.0, fail: Exception | None = None):
        self.latency = latency
        self.fail = fail
        self.calls = 0
        self._lock = threading.Lock()

    def invoke(self, messages):
        with self._lock:
            self.calls += 1
        if self.latency:
            time.sleep(self.latency)
        if self.fail:
            raise self.fail
        system = messages[0].content
        if system == agents.SCANNER_SYSTEM_PROMPT:
            return AIMessage(content=SCANNER_REPLY)
        if system == agents.REFACTOR_SYSTEM_PROMPT:
            return AIMessage(content=REFACTOR_REPLY)
        return AIMessage(content=DOCS_REPLY)


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(agents, "initialize_llm", lambda **_: llm)
    return llm


def make_state(code: str = "def f(x):\n    return eval(x)\n", **extra) -> dict:
    state = {
        "raw_code": code,
        "language": "python",
        "api_key": "test-key",
        "model_name": "gemini-test",
        "temperature": 0.2,
        "audit_report": "",
        "clean_code": "",
        "documentation": "",
        "issues_found": False,
        "issue_summary": [],
        "logs": [],
    }
    state.update(extra)
    return state
