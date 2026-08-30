"""Streamlit dashboard for the LangGraph Sequential Code Auditor."""

from __future__ import annotations

import os
import sys
from typing import Any

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    APIKeyError,
    SUPPORTED_LANGUAGES,
    UPLOAD_EXTENSIONS,
    detect_language_from_filename,
    get_api_key,
    language_label,
    sanitize_report_markdown,
)
from src.graph import app_graph, review_graph
from src.agents import docs_agent, refactor_agent

load_dotenv()


def _detect_language_from_code(code: str) -> str | None:
    """Guess language from common syntax in pasted source code."""
    snippet = (code or "").lstrip()
    if not snippet:
        return None
    if snippet.startswith("<?php"):
        return "php"
    if snippet.startswith("<!") or snippet.startswith("<html") or snippet.startswith("<div"):
        return "html"
    python_signals = (
        "def ",
        "import ",
        "from ",
        "class ",
        "print(",
        "elif ",
        "self.",
        '"""',
        "async def ",
    )
    js_signals = ("function ", "const ", "let ", "=>", "console.log", "export ", "import {")
    python_hits = sum(1 for token in python_signals if token in snippet)
    js_hits = sum(1 for token in js_signals if token in snippet)
    if python_hits >= 2 and python_hits >= js_hits:
        return "python"
    if js_hits >= 2 and js_hits > python_hits:
        return "javascript"
    return None

DEFAULT_SAMPLE = '''def get_user(username):
    import sqlite3
    conn = sqlite3.connect("users.db")
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    return conn.execute(query).fetchone()
'''

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.75rem;
    }

    .panel-box {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.25rem 1.35rem 1.35rem;
        min-height: 520px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #111827 !important;
        border-color: #1e293b !important;
        border-radius: 14px !important;
        padding: 1rem 1.1rem 1.1rem !important;
    }
    [data-testid="column"] > div[data-testid="stVerticalBlock"] {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1rem 1.1rem 1.25rem;
        min-height: 480px;
    }
    .panel-heading {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 0 0 0.35rem 0;
    }
    .panel-caption {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0 0 0.75rem 0;
    }

    /* Code editor inside left panel */
    div[data-testid="stTextArea"] textarea {
        background-color: #050816 !important;
        color: #4ade80 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        font-family: "Consolas", "Monaco", "Courier New", monospace !important;
        font-size: 0.9rem !important;
        min-height: 340px !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background: #2563eb !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1rem !important;
        margin-top: 0.75rem;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #1d4ed8 !important;
    }

    .empty-state {
        color: #64748b;
        text-align: center;
        padding: 3rem 1rem 1rem;
        font-size: 0.95rem;
    }
    .reviewing-state {
        color: #94a3b8;
        text-align: center;
        padding: 3rem 1rem 1rem;
        font-size: 0.95rem;
    }

    .issues-heading {
        color: #facc15;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 0 0 0.65rem 0;
    }
    .clean-heading {
        color: #4ade80;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 0 0 0.65rem 0;
    }
    .report-heading {
        color: #4ade80;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 1.25rem 0 0.65rem 0;
    }
    [data-testid="column"]:last-child [data-testid="stMarkdown"] hr {
        display: none;
    }

    .issue-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .issue-list li { margin-bottom: 0.35rem; }

    .report-box {
        background: #0b1220;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.6;
        max-height: 280px;
        overflow-y: auto;
    }
    .report-box h3, .report-box h4 { color: #f1f5f9; margin-top: 0.5rem; }

    .meta-row { margin-bottom: 0.75rem; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""


def _init_session() -> None:
    defaults = {
        "input_code": DEFAULT_SAMPLE,
        "selected_language": "python",
        "audit_report": "",
        "clean_code": "",
        "documentation": "",
        "issues_found": False,
        "issue_summary": [],
        "logs": [],
        "pipeline_done": False,
        "pipeline_running": False,
        "review_context": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_results() -> None:
    st.session_state.audit_report = ""
    st.session_state.clean_code = ""
    st.session_state.documentation = ""
    st.session_state.issues_found = False
    st.session_state.issue_summary = []
    st.session_state.logs = []
    st.session_state.pipeline_done = False


def _build_config(api_key: str, model: str, temperature: float, language: str) -> dict[str, Any]:
    return {
        "configurable": {
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
            "language": language,
        }
    }


def _build_initial_state(
    raw_code: str,
    language: str,
    api_key: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    return {
        "raw_code": raw_code,
        "language": language,
        "api_key": api_key,
        "model_name": model,
        "temperature": temperature,
        "audit_report": "",
        "clean_code": "",
        "documentation": "",
        "issues_found": False,
        "issue_summary": [],
        "logs": [],
    }


def _apply_update(update: dict[str, Any]) -> None:
    if "audit_report" in update:
        st.session_state.audit_report = update["audit_report"]
    if "clean_code" in update:
        st.session_state.clean_code = update["clean_code"]
    if "documentation" in update:
        st.session_state.documentation = update["documentation"]
    if "issues_found" in update:
        st.session_state.issues_found = update["issues_found"]
    if "issue_summary" in update:
        st.session_state.issue_summary = update["issue_summary"]
    if "logs" in update:
        st.session_state.logs.extend(update["logs"])


def _run_scanner_review(
    raw_code: str,
    language: str,
    api_key: str,
    model: str,
    temperature: float,
) -> None:
    """Fast path: scanner only (~1 API call)."""
    _reset_results()
    st.session_state.pipeline_running = True
    st.session_state.review_context = {
        "raw_code": raw_code,
        "language": language,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
    }

    initial_state = _build_initial_state(raw_code, language, api_key, model, temperature)
    config = _build_config(api_key, model, temperature, language)

    try:
        for event in review_graph.stream(initial_state, config=config, stream_mode="updates"):
            for _node_name, update in event.items():
                _apply_update(update)
        st.session_state.pipeline_done = True
    except Exception as exc:
        st.session_state.issues_found = True
        st.session_state.issue_summary = [f"**Pipeline Error**: {exc}"]
        st.session_state.audit_report = f"The review failed: {exc}"
        st.session_state.pipeline_done = True
    finally:
        st.session_state.pipeline_running = False


def _run_refactor_only() -> None:
    ctx = st.session_state.review_context
    state = _build_initial_state(
        ctx["raw_code"],
        ctx["language"],
        ctx["api_key"],
        ctx["model"],
        ctx["temperature"],
    )
    state["audit_report"] = st.session_state.audit_report
    config = _build_config(ctx["api_key"], ctx["model"], ctx["temperature"], ctx["language"])
    update = refactor_agent(state, config)
    _apply_update(update)


def _run_docs_only() -> None:
    ctx = st.session_state.review_context
    state = _build_initial_state(
        ctx["raw_code"],
        ctx["language"],
        ctx["api_key"],
        ctx["model"],
        ctx["temperature"],
    )
    state["clean_code"] = st.session_state.clean_code
    config = _build_config(ctx["api_key"], ctx["model"], ctx["temperature"], ctx["language"])
    update = docs_agent(state, config)
    _apply_update(update)


def _run_full_pipeline(
    raw_code: str,
    language: str,
    api_key: str,
    model: str,
    temperature: float,
) -> None:
    _reset_results()
    st.session_state.pipeline_running = True

    initial_state = _build_initial_state(raw_code, language, api_key, model, temperature)
    config = _build_config(api_key, model, temperature, language)
    st.session_state.review_context = {
        "raw_code": raw_code,
        "language": language,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
    }

    try:
        for event in app_graph.stream(initial_state, config=config, stream_mode="updates"):
            for _node_name, update in event.items():
                _apply_update(update)

        st.session_state.pipeline_done = True
    except Exception as exc:
        st.session_state.issues_found = True
        st.session_state.issue_summary = [f"**Pipeline Error**: {exc}"]
        st.session_state.audit_report = f"### Code Review Report\n\nThe review pipeline failed: {exc}"
        st.session_state.pipeline_done = True
    finally:
        st.session_state.pipeline_running = False


def _render_review_results() -> None:
    if st.session_state.pipeline_running:
        st.markdown(
            '<p class="reviewing-state">Reviewing your code...</p>',
            unsafe_allow_html=True,
        )
        return

    if not st.session_state.pipeline_done:
        st.markdown(
            '<p class="empty-state">Submit code to see the review</p>',
            unsafe_allow_html=True,
        )
        return

    if st.session_state.issues_found:
        st.markdown('<p class="issues-heading">Issues Found</p>', unsafe_allow_html=True)
        if st.session_state.issue_summary:
            for item in st.session_state.issue_summary:
                st.markdown(f"- {item}")
        else:
            st.markdown("- Issues were detected — see the full report below.")
    else:
        st.markdown('<p class="clean-heading">No Issues Found</p>', unsafe_allow_html=True)
        st.markdown("- Your code looks clean. See the full report below.")

    st.markdown('<p class="report-heading">Full Report</p>', unsafe_allow_html=True)
    st.markdown(sanitize_report_markdown(st.session_state.audit_report or "No report generated."))


def main() -> None:
    st.set_page_config(
        page_title="LangGraph code auditor",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _init_session()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    language_keys = list(SUPPORTED_LANGUAGES.keys())

    with st.sidebar:
        st.header("Settings")
        env_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        if "google_api_key_input" not in st.session_state:
            st.session_state.google_api_key_input = (
                "" if env_key == "your_google_api_key_here" else env_key
            )
        api_key_input = st.text_input(
            "Google Gemini API Key",
            type="password",
            key="google_api_key_input",
            help="Get a free key at aistudio.google.com/apikey",
        )
        st.caption("Required: [Google AI Studio](https://aistudio.google.com/apikey) Gemini key.")
        if st.button("Test API Key", use_container_width=True):
            try:
                test_key = get_api_key(api_key_input or None)
                from src.utils import initialize_llm

                llm = initialize_llm(api_key=test_key, model="gemini-3.6-flash", temperature=0.0)
                llm.invoke("Reply with exactly: OK")
                st.success("Google Gemini API key works.")
            except Exception as exc:
                st.error(f"API test failed: {exc}")
        st.session_state.selected_language = st.selectbox(
            "Language",
            options=language_keys,
            format_func=language_label,
            index=language_keys.index(st.session_state.selected_language),
        )
        model = st.selectbox(
            "Model",
            options=[
                "gemini-3.5-flash-lite",
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-3.5-flash",
            ],
            index=0,
            help="Flash Lite is fastest. Full pipeline uses the selected model.",
        )
        run_full_pipeline = st.checkbox("Full pipeline (scan + refactor + docs)", value=False)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        uploaded = st.file_uploader("Upload file", type=UPLOAD_EXTENSIONS)
        if uploaded is not None:
            st.session_state.input_code = uploaded.read().decode("utf-8")
            detected = detect_language_from_filename(uploaded.name)
            if detected:
                st.session_state.selected_language = detected

    st.title("LangGraph code auditor")
    st.caption("Paste any code, click Review Code, and get Issues Found plus a full report.")

    col_code, col_results = st.columns(2, gap="medium")

    with col_code:
        st.markdown('<p class="panel-heading">Your Code</p>', unsafe_allow_html=True)
        lang_display = language_label(st.session_state.selected_language)
        st.markdown(
            f'<p class="panel-caption">Language: {lang_display}</p>',
            unsafe_allow_html=True,
        )
        raw_code = st.text_area(
            "code_input",
            value=st.session_state.input_code,
            height=360,
            label_visibility="collapsed",
            placeholder="Paste your source code here...",
        )
        st.session_state.input_code = raw_code
        review_clicked = st.button("Review Code", type="primary", use_container_width=True)

    with col_results:
        st.markdown('<p class="panel-heading">Review Results</p>', unsafe_allow_html=True)
        _render_review_results()

    if review_clicked:
        if not raw_code.strip():
            st.warning("Please paste some code before reviewing.")
        else:
            detected = _detect_language_from_code(raw_code)
            if detected:
                st.session_state.selected_language = detected
            try:
                resolved_key = get_api_key(api_key_input or None)
                with st.spinner("Scanning code..." if not run_full_pipeline else "Running full pipeline..."):
                    if run_full_pipeline:
                        _run_full_pipeline(
                            raw_code,
                            st.session_state.selected_language,
                            resolved_key,
                            model,
                            temperature,
                        )
                    else:
                        _run_scanner_review(
                            raw_code,
                            st.session_state.selected_language,
                            resolved_key,
                            model,
                            temperature,
                        )
                st.rerun()
            except APIKeyError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    if st.session_state.pipeline_done:
        st.divider()
        extra_col1, extra_col2 = st.columns(2)
        with extra_col1:
            with st.expander("Refactored Code", expanded=False):
                if st.session_state.clean_code:
                    code_lang = (
                        st.session_state.selected_language
                        if st.session_state.selected_language != "other"
                        else "text"
                    )
                    st.code(st.session_state.clean_code, language=code_lang)
                elif st.session_state.review_context:
                    if st.button("Generate refactored code", key="gen_refactor"):
                        with st.spinner("Refactoring..."):
                            _run_refactor_only()
                        st.rerun()
                else:
                    st.info("Run a review first.")
        with extra_col2:
            with st.expander("Documentation", expanded=False):
                if st.session_state.documentation:
                    st.markdown(st.session_state.documentation)
                elif st.session_state.clean_code:
                    if st.button("Generate documentation", key="gen_docs"):
                        with st.spinner("Writing docs..."):
                            _run_docs_only()
                        st.rerun()
                elif st.session_state.review_context:
                    st.caption("Generate refactored code first, then documentation.")
                else:
                    st.info("Run a review first.")


if __name__ == "__main__":
    main()
