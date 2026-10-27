"""LangGraph sequential workflow: Scanner -> Refactor -> Docs."""

from langgraph.graph import END, START, StateGraph

from src.agents import docs_agent, refactor_agent, scanner_agent
from src.state import AgentState


def build_graph() -> StateGraph:
    """Construct the uncompiled sequential audit pipeline graph.

    Returns:
        A ``StateGraph`` wired as ``START -> scanner -> refactor -> docs -> END``.
    """
    graph = StateGraph(AgentState)

    graph.add_node("scanner", scanner_agent)
    graph.add_node("refactor", refactor_agent)
    graph.add_node("docs", docs_agent)

    graph.add_edge(START, "scanner")
    graph.add_edge("scanner", "refactor")
    graph.add_edge("refactor", "docs")
    graph.add_edge("docs", END)

    return graph


app_graph = build_graph().compile()


def build_review_graph():
    """Fast review graph: scanner only."""
    graph = StateGraph(AgentState)
    graph.add_node("scanner", scanner_agent)
    graph.add_edge(START, "scanner")
    graph.add_edge("scanner", END)
    return graph.compile()


review_graph = build_review_graph()
