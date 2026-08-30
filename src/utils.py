"""Utility helpers for LLM initialization and response parsing."""

from __future__ import annotations

import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_TEMPERATURE = 0.2


class APIKeyError(ValueError):
    """Raised when a valid Google API key is not available."""


def get_api_key(override: Optional[str] = None) -> str:
    """Resolve the Google API key from override, environment, or raise.

    Args:
        override: Optional API key from UI or caller that takes precedence.

    Returns:
        A non-empty API key string.

    Raises:
        APIKeyError: If no valid API key can be resolved.
    """
    candidates = (
        override,
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
    )
    for candidate in candidates:
        key = (candidate or "").strip().strip('"').strip("'")
        if key and key not in {"your_google_api_key_here", "your-api-key"}:
            return key
    raise APIKeyError(
        "A valid Google API key is required. "
        "Set GOOGLE_API_KEY in .env or provide it in the sidebar."
    )


def initialize_llm(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: Optional[int] = None,
) -> ChatGoogleGenerativeAI:
    """Initialize the Gemini chat model.

    Args:
        api_key: Optional API key override.
        model: Gemini model name. Defaults to GEMINI_MODEL env or gemini-3.6-flash.
        temperature: Sampling temperature for generation.

    Returns:
        Configured ChatGoogleGenerativeAI instance.

    Raises:
        APIKeyError: If the API key is missing or invalid.
    """
    resolved_key = get_api_key(api_key)
    resolved_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    os.environ["GOOGLE_API_KEY"] = resolved_key

    return ChatGoogleGenerativeAI(
        model=resolved_model,
        api_key=resolved_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def extract_code_block(text: object, language: str = "python") -> str:
    """Extract the first fenced code block from LLM output.

    Args:
        text: Raw LLM response text or structured content.
        language: Expected language tag in the fence (e.g. ``python``).

    Returns:
        Extracted code content, or the original text if no fence is found.
    """
    text = normalize_llm_content(text)
    if not text:
        return ""

    pattern = rf"```(?:{re.escape(language)})?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    generic = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if generic:
        return generic.group(1).strip()

    return text.strip()


SUPPORTED_LANGUAGES: dict[str, str] = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "cpp": "C++",
    "c": "C",
    "csharp": "C#",
    "go": "Go",
    "rust": "Rust",
    "ruby": "Ruby",
    "php": "PHP",
    "sql": "SQL",
    "html": "HTML",
    "css": "CSS",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "shell": "Shell/Bash",
    "other": "Other",
}

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}

UPLOAD_EXTENSIONS = sorted({ext.lstrip(".") for ext in EXTENSION_TO_LANGUAGE})


def language_label(language: str) -> str:
    """Return a human-readable label for a language key."""
    return SUPPORTED_LANGUAGES.get(language, language.title())


def detect_language_from_code(code: str) -> str | None:
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


def detect_language_from_filename(filename: str) -> str | None:
    """Infer language key from a file extension, if recognized."""
    _, ext = os.path.splitext(filename.lower())
    return EXTENSION_TO_LANGUAGE.get(ext)


def parse_issues_found(report: object) -> tuple[bool, str]:
    """Parse scanner marker and return (issues_found, cleaned_report).

    The scanner must emit ``ISSUES_FOUND: YES`` or ``ISSUES_FOUND: NO`` on its own line.
    """
    report = normalize_llm_content(report)
    if not report:
        return False, report

    match = re.search(r"^ISSUES_FOUND:\s*(YES|NO)\s*$", report, re.MULTILINE | re.IGNORECASE)
    if not match:
        lowered = report.lower()
        has_issue_signals = any(
            token in lowered
            for token in (
                "critical",
                "high severity",
                "vulnerability",
                "syntax error",
                "bug",
                "logic error",
                "security flaw",
                "incorrect",
                "invalid",
            )
        )
        return has_issue_signals, report.strip()

    issues_found = match.group(1).upper() == "YES"
    cleaned = re.sub(r"^ISSUES_FOUND:\s*(YES|NO)\s*\n?", "", report, count=1, flags=re.MULTILINE | re.IGNORECASE)
    return issues_found, cleaned.strip()


def parse_issue_summary(report: object) -> tuple[list[str], str]:
    """Extract ISSUE_SUMMARY bullets and return (bullets, full_report)."""
    report = normalize_llm_content(report)
    if not report:
        return [], report

    match = re.search(
        r"^ISSUE_SUMMARY:\s*\n((?:-\s+.+\n?)+)",
        report,
        re.MULTILINE,
    )
    if not match:
        return [], report.strip()

    bullets = [
        line.lstrip("- ").strip()
        for line in match.group(1).splitlines()
        if line.strip().startswith("-")
    ]
    full_report = (report[: match.start()] + report[match.end() :]).strip()
    return bullets, full_report


def parse_scanner_output(report: object) -> tuple[bool, list[str], str]:
    """Parse scanner output into issues flag, summary bullets, and full report."""
    report = normalize_llm_content(report)
    issues_found, after_flag = parse_issues_found(report)
    issue_summary, full_report = parse_issue_summary(after_flag)
    return issues_found, issue_summary, full_report


def sanitize_report_markdown(text: object) -> str:
    """Remove horizontal rules and duplicate headings from audit report markdown."""
    normalized = normalize_llm_content(text)
    if not normalized:
        return ""

    # Remove markdown horizontal rules (---, ***, ___)
    cleaned = re.sub(r"^\s*([-*_])\1{2,}\s*$", "", normalized, flags=re.MULTILINE)
    # Drop duplicate top-level report title; UI already shows "Full Report"
    cleaned = re.sub(
        r"^#{1,3}\s*Code Review Report\s*\n?",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_llm_content(content: object) -> str:
    """Convert Gemini/LangChain response content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
            elif hasattr(block, "content"):
                parts.append(normalize_llm_content(block.content))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    if hasattr(content, "text"):
        return str(content.text)
    return str(content)


def clean_markdown_response(text: object) -> str:
    """Normalize LLM markdown output by trimming whitespace.

    Args:
        text: Raw markdown string or structured LLM content.

    Returns:
        Cleaned markdown string.
    """
    normalized = normalize_llm_content(text)
    return normalized.strip() if normalized else ""
