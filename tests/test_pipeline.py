"""Agent and LangGraph pipeline tests using the fake LLM."""

import src.agents as agents
from src.agents import docs_agent, refactor_agent, scanner_agent
from src.graph import app_graph, review_graph
from tests.conftest import FakeLLM, make_state


def test_scanner_parses_issues(fake_llm):
    result = scanner_agent(make_state())
    assert result["issues_found"] is True
    assert result["issue_summary"][0].startswith("**Security**")
    assert "ISSUES_FOUND" not in result["audit_report"]


def test_scanner_skips_empty_code(fake_llm):
    result = scanner_agent(make_state(code="   "))
    assert result["issues_found"] is False
    assert fake_llm.calls == 0


def test_refactor_extracts_code(fake_llm):
    result = refactor_agent(make_state(audit_report="SQL injection"))
    assert result["clean_code"].startswith("def get_user")


def test_refactor_failure_preserves_original_code(monkeypatch):
    monkeypatch.setattr(agents, "initialize_llm", lambda **_: FakeLLM(fail=RuntimeError("quota")))
    state = make_state()
    result = refactor_agent(state)
    assert result["clean_code"] == state["raw_code"].strip()
    assert "quota" in result["logs"][0]


def test_scanner_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(agents, "initialize_llm", lambda **_: FakeLLM(fail=RuntimeError("timeout")))
    result = scanner_agent(make_state())
    assert result["issues_found"] is True
    assert "timeout" in result["audit_report"]


def test_docs_skips_without_clean_code(fake_llm):
    assert "No refactored code" in docs_agent(make_state())["documentation"]


def test_config_used_when_state_missing_values(monkeypatch):
    seen = {}

    def fake_init(**kwargs):
        seen.update(kwargs)
        return FakeLLM()

    monkeypatch.setattr(agents, "initialize_llm", fake_init)
    state = make_state(api_key="", model_name="", temperature=None)
    config = {"configurable": {"api_key": "cfg-key", "model": "cfg-model", "temperature": 0.7}}
    scanner_agent(state, config)
    assert (seen["api_key"], seen["model"], seen["temperature"]) == ("cfg-key", "cfg-model", 0.7)


def test_full_pipeline_runs_in_order(fake_llm):
    final = app_graph.invoke(make_state())
    assert final["issues_found"] is True
    assert final["clean_code"].startswith("def get_user")
    assert final["documentation"].startswith("## Module Overview")
    assert [log.split("]")[0] for log in final["logs"]] == ["[Scanner", "[Refactor", "[Docs"]
    assert fake_llm.calls == 3


def test_review_graph_only_scans(fake_llm):
    final = review_graph.invoke(make_state())
    assert final["clean_code"] == ""
    assert fake_llm.calls == 1
