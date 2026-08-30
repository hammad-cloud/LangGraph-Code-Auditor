"""Central state definition for the LangGraph Sequential Code Auditor."""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """Shared state passed between sequential audit agents.

    Attributes:
        raw_code: Input source code to audit.
        language: Programming language of the input (e.g. python, javascript).
        api_key: Gemini API key forwarded from the UI (avoids LangGraph config drops).
        model_name: Gemini model name selected in the UI.
        temperature: Sampling temperature for generation.
        audit_report: Scanner output detailing vulnerabilities, bugs, and style errors.
        clean_code: Refactor output containing formatted, bug-fixed code.
        documentation: Docs output with function/class docs and README content.
        issues_found: True when the scanner detected bugs, vulnerabilities, or bad code.
        issue_summary: Short bullet list of top issues for the review panel.
        logs: Append-only log list tracking agent progress for live UI updates.
    """

    raw_code: str
    language: str
    api_key: str
    model_name: str
    temperature: float
    audit_report: str
    clean_code: str
    documentation: str
    issues_found: bool
    issue_summary: list[str]
    logs: Annotated[list[str], operator.add]
