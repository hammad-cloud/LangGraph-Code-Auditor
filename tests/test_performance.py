"""Performance, load, and stress tests. Run with: pytest -m perf

The LLM is faked with a fixed latency, so these measure the pipeline's own
overhead and concurrency behaviour, not Gemini's speed.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import src.agents as agents
from src.graph import app_graph, review_graph
from src.utils import parse_scanner_output
from tests.conftest import SCANNER_REPLY, FakeLLM, make_state

pytestmark = pytest.mark.perf


def test_pipeline_overhead_is_small(fake_llm):
    runs = 50
    start = time.perf_counter()
    for _ in range(runs):
        app_graph.invoke(make_state())
    per_run = (time.perf_counter() - start) / runs
    assert per_run < 0.05, f"graph overhead {per_run * 1000:.1f} ms/run"


def test_parser_throughput():
    start = time.perf_counter()
    for _ in range(10_000):
        parse_scanner_output(SCANNER_REPLY)
    assert time.perf_counter() - start < 2.0


def test_load_concurrent_reviews(monkeypatch):
    """50 users reviewing at once with 100 ms model latency must run in parallel."""
    llm = FakeLLM(latency=0.1)
    monkeypatch.setattr(agents, "initialize_llm", lambda **_: llm)
    users = 50

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=users) as pool:
        results = list(pool.map(lambda i: review_graph.invoke(make_state(code=f"x = {i}")), range(users)))
    elapsed = time.perf_counter() - start

    assert all(r["issues_found"] for r in results)
    assert llm.calls == users
    assert elapsed < 2.0, f"{users} concurrent reviews took {elapsed:.2f}s (serial would be {users * 0.1:.1f}s)"


def test_load_state_is_isolated_between_concurrent_runs(fake_llm):
    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(lambda i: app_graph.invoke(make_state(code=f"v = {i}")), range(100)))
    assert [r["raw_code"] for r in results] == [f"v = {i}" for i in range(100)]
    assert all(len(r["logs"]) == 3 for r in results)


def test_stress_large_input(fake_llm):
    big = "def f():\n    return 1\n" * 50_000  # ~1.2 MB of source
    start = time.perf_counter()
    final = app_graph.invoke(make_state(code=big))
    assert final["documentation"]
    assert time.perf_counter() - start < 2.0


def test_stress_failing_model_under_load(monkeypatch):
    """Model outages under load must degrade to error reports, never crash."""
    monkeypatch.setattr(agents, "initialize_llm", lambda **_: FakeLLM(fail=RuntimeError("503")))
    with ThreadPoolExecutor(max_workers=25) as pool:
        results = list(pool.map(lambda _: app_graph.invoke(make_state()), range(100)))
    assert all("503" in r["audit_report"] for r in results)
    assert all(r["clean_code"] for r in results)  # original code preserved
