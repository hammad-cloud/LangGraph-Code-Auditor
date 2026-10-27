"""LLM agent node implementations for the sequential code audit pipeline."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.state import AgentState
from src.utils import (
    clean_markdown_response,
    extract_code_block,
    initialize_llm,
    language_label,
    normalize_llm_content,
    parse_scanner_output,
)

SCANNER_SYSTEM_PROMPT = """You are a fast code auditor. Inspect the source code and return a concise audit report.

Language-specific rules apply. Be brief but accurate.

Start with exactly one line:
ISSUES_FOUND: YES
or
ISSUES_FOUND: NO

If YES, add immediately:
ISSUE_SUMMARY:
- **Category**: one-line issue (3-5 bullets max)

Then write a short markdown report:
#### Summary
One short paragraph.

#### Issues
Numbered list with **Category**: description.

Keep the full report under 400 words. Focus on security, bugs, and critical style issues only.
"""

REFACTOR_SYSTEM_PROMPT = """You are an expert multi-language refactoring engineer.
Given original source code, its programming language, and an audit report, produce fully optimized,
secure, idiomatic code in the SAME language as the input.

Rules:
- Fix ALL security vulnerabilities, syntax errors, and bugs identified in the audit
- Apply language-specific formatting and best practices
- Preserve the original functionality unless a bug fix requires a behavior change
- Output ONLY the refactored code inside a single fenced code block tagged with the correct language
- Do not include explanations outside the code block
"""

DOCS_SYSTEM_PROMPT = """You are an expert technical documentation writer.
Given clean, refactored source code and its programming language, generate comprehensive documentation in markdown format.

Your output MUST include:
1. **Module Overview** - Purpose and high-level description
2. **Function/Class Documentation** - Doc comments appropriate for the language (Google/JSDoc/Javadoc style as applicable)
3. **README.md Block** - A complete README section with:
   - Project description
   - Installation instructions
   - Usage examples
   - API reference summary
   - Dependencies

Format everything as clean, professional markdown suitable for a production repository.
"""


def _invoke_llm(
    system_prompt: str,
    user_content: str,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int | None = None,
) -> str:
    """Send a prompt to Gemini and return the text response."""
    llm = initialize_llm(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else response
    return normalize_llm_content(content)


def _get_config_values(state: AgentState, config: RunnableConfig | None) -> tuple[str | None, str | None, float, str]:
    """Read API key, model, temperature, and language from state first, then config."""
    configurable = (config or {}).get("configurable", {}) if config else {}
    api_key = (state.get("api_key") or configurable.get("api_key") or "").strip() or None
    model = (state.get("model_name") or configurable.get("model") or "").strip() or None
    temperature = state.get("temperature")
    if temperature is None:
        temperature = configurable.get("temperature", 0.2)
    language = (state.get("language") or configurable.get("language") or "python").strip().lower()
    return api_key, model, float(temperature), language


def scanner_agent(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Inspect raw code for security flaws, bugs, syntax errors, and style issues."""
    api_key, model, temperature, language = _get_config_values(state, config)

    raw_code = state.get("raw_code", "").strip()
    if not raw_code:
        return {
            "audit_report": "No source code provided for audit.",
            "issues_found": False,
            "issue_summary": [],
            "logs": ["[Scanner] Skipped — no input code."],
        }

    lang_label = language_label(language)
    fence = language if language != "other" else ""

    try:
        report = _invoke_llm(
            SCANNER_SYSTEM_PROMPT,
            (
                f"Programming language: {lang_label}\n\n"
                f"Audit the following source code:\n\n```{fence}\n{raw_code}\n```"
            ),
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=1200,
        )
        issues_found, issue_summary, full_report = parse_scanner_output(clean_markdown_response(report))
        status_msg = (
            "[Scanner] Audit complete — issues found."
            if issues_found
            else "[Scanner] Audit complete — no issues found."
        )
        return {
            "audit_report": full_report,
            "issues_found": issues_found,
            "issue_summary": issue_summary,
            "logs": [status_msg],
        }
    except Exception as exc:
        return {
            "audit_report": f"Scanner agent failed: {exc}",
            "issues_found": True,
            "issue_summary": [f"**Scanner Error**: {exc}"],
            "logs": [f"[Scanner] Error — {exc}"],
        }


def refactor_agent(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Refactor code using audit findings to produce clean, secure source."""
    api_key, model, temperature, language = _get_config_values(state, config)

    raw_code = state.get("raw_code", "").strip()
    audit_report = state.get("audit_report", "")

    if not raw_code:
        return {
            "clean_code": "",
            "logs": ["[Refactor] Skipped — no input code."],
        }

    lang_label = language_label(language)
    fence = language if language != "other" else ""

    try:
        user_prompt = (
            f"Programming language: {lang_label}\n\n"
            f"Original code:\n```{fence}\n{raw_code}\n```\n\n"
            f"Audit report:\n{audit_report}\n\n"
            f"Produce the refactored, secure, idiomatic {lang_label} code."
        )
        response = _invoke_llm(
            REFACTOR_SYSTEM_PROMPT,
            user_prompt,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        clean = extract_code_block(response, language=language if language != "other" else "")
        return {
            "clean_code": clean,
            "logs": [f"[Refactor] Code refactored — fixes applied for {lang_label}."],
        }
    except Exception as exc:
        return {
            "clean_code": raw_code,
            "logs": [f"[Refactor] Error — {exc}. Original code preserved."],
        }


def docs_agent(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Generate documentation for refactored code."""
    api_key, model, temperature, language = _get_config_values(state, config)

    clean_code = state.get("clean_code", "").strip()
    if not clean_code:
        return {
            "documentation": "No refactored code available for documentation.",
            "logs": ["[Docs] Skipped — no clean code to document."],
        }

    lang_label = language_label(language)
    fence = language if language != "other" else ""

    try:
        response = _invoke_llm(
            DOCS_SYSTEM_PROMPT,
            (
                f"Programming language: {lang_label}\n\n"
                f"Generate documentation for the following code:\n\n```{fence}\n{clean_code}\n```"
            ),
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        return {
            "documentation": clean_markdown_response(response),
            "logs": [f"[Docs] Documentation generated for {lang_label}."],
        }
    except Exception as exc:
        return {
            "documentation": f"Documentation agent failed: {exc}",
            "logs": [f"[Docs] Error — {exc}"],
        }
