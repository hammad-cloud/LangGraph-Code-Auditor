"""Unit tests for parsing and helper functions in src.utils."""

import pytest

from src.utils import (
    APIKeyError,
    detect_language_from_code,
    detect_language_from_filename,
    extract_code_block,
    get_api_key,
    language_label,
    normalize_llm_content,
    parse_scanner_output,
    sanitize_report_markdown,
)


def test_get_api_key_prefers_override(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    assert get_api_key(" 'ui-key' ") == "ui-key"


def test_get_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert get_api_key(None) == "gemini-key"


@pytest.mark.parametrize("bad", [None, "", "   ", "your_google_api_key_here", "your-api-key"])
def test_get_api_key_rejects_missing_or_placeholder(bad):
    with pytest.raises(APIKeyError):
        get_api_key(bad)


def test_extract_code_block_with_language():
    assert extract_code_block("text\n```python\nx = 1\n```\nmore") == "x = 1"


def test_extract_code_block_generic_fence():
    assert extract_code_block("```\ny = 2\n```", language="go") == "y = 2"


def test_extract_code_block_without_fence_returns_text():
    assert extract_code_block("  plain  ") == "plain"
    assert extract_code_block(None) == ""


def test_normalize_llm_content_handles_blocks():
    content = ["a", {"text": "b"}, {"type": "image"}, {"content": "c"}]
    assert normalize_llm_content(content) == "a\nb\nc"


def test_parse_scanner_output_yes_with_summary():
    report = "ISSUES_FOUND: YES\nISSUE_SUMMARY:\n- **Bug**: one\n- **Style**: two\n\n#### Summary\nok"
    issues, summary, full = parse_scanner_output(report)
    assert issues is True
    assert summary == ["**Bug**: one", "**Style**: two"]
    assert full == "#### Summary\nok"


def test_parse_scanner_output_no():
    issues, summary, full = parse_scanner_output("ISSUES_FOUND: NO\n#### Summary\nClean.")
    assert (issues, summary, full) == (False, [], "#### Summary\nClean.")


def test_parse_scanner_output_without_marker_uses_keywords():
    assert parse_scanner_output("Found a critical vulnerability.")[0] is True
    assert parse_scanner_output("Looks great.")[0] is False


def test_sanitize_report_markdown_strips_rules_and_title():
    text = "### Code Review Report\n---\nBody\n\n\n\nEnd"
    assert sanitize_report_markdown(text) == "Body\n\nEnd"


@pytest.mark.parametrize(
    "code, expected",
    [
        ("import os\ndef main():\n    pass", "python"),
        ("const a = 1;\nconsole.log(a)", "javascript"),
        ("<?php echo 1;", "php"),
        ("<html></html>", "html"),
        ("SELECT 1;", None),
        ("", None),
    ],
)
def test_detect_language_from_code(code, expected):
    assert detect_language_from_code(code) == expected


def test_detect_language_from_filename():
    assert detect_language_from_filename("Main.TSX") == "typescript"
    assert detect_language_from_filename("notes.txt") is None


def test_language_label():
    assert language_label("cpp") == "C++"
    assert language_label("zig") == "Zig"
