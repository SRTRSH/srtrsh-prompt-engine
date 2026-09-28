"""Tests for RFC 8785 JSON Canonicalization Scheme and PromptVersionHash calculation."""

import re

from srtrsh_prompt_engine.canonical.hasher import (
    build_canonical_payload,
    compute_prompt_version_hash,
    normalize_template_text,
)
from srtrsh_prompt_engine.canonical.jcs import canonicalize


def test_jcs_primitives():
    assert canonicalize(None) == b"null"
    assert canonicalize(True) == b"true"
    assert canonicalize(False) == b"false"
    assert canonicalize(123) == b"123"
    assert canonicalize(0) == b"0"
    assert canonicalize(0.0) == b"0"
    assert canonicalize(42.0) == b"42"
    assert canonicalize(3.14) == b"3.14"


def test_jcs_string_escaping_minimal():
    # Quotes and backslashes must be escaped
    assert canonicalize('hello "world"') == b'"hello \\"world\\""'
    assert canonicalize("a\\b") == b'"a\\\\b"'
    # Control characters
    assert canonicalize("line1\nline2") == b'"line1\\nline2"'
    assert canonicalize("tab\tchar") == b'"tab\\tchar"'
    assert canonicalize("cr\r") == b'"cr\\r"'
    # Non-ASCII unicode characters must NOT be escaped
    assert canonicalize("Привет") == b'"' + "Привет".encode("utf-8") + b'"'
    assert canonicalize("€") == b'"' + "€".encode("utf-8") + b'"'


def test_jcs_object_key_sorting_utf16():
    # RFC 8785: keys sorted by UTF-16 code units
    data = {
        "b": 1,
        "a": 2,
        "\u20ac": 3,
        "\n": 4,
    }
    # \n (0x0A) < a (0x61) < b (0x62) < \u20ac (0x20AC)
    expected = b'{"\\n":4,"a":2,"b":1,"\xe2\x82\xac":3}'
    assert canonicalize(data) == expected


def test_normalize_template_text():
    crlf_text = "Line 1\r\nLine 2\rLine 3\n"
    normalized = normalize_template_text(crlf_text)
    assert normalized == "Line 1\nLine 2\nLine 3\n"
    # Preserves indentation
    indented = "  leading space\n\ttab space"
    assert normalize_template_text(indented) == indented


def test_prompt_version_hash_format_and_determinism():
    template = {
        "format": "STANDARD",
        "systemPrompt": "You are a test assistant.",
        "userPrompt": "Input: {{var}}",
    }
    variables = [
        {"name": "var", "type": "STRING", "required": True, "defaultValue": None}
    ]
    schema = {"schemaType": "JSON_SCHEMA", "definition": {"type": "object"}}

    payload1 = build_canonical_payload(template, variables, schema)
    payload2 = build_canonical_payload(template, variables, schema)

    hash1 = compute_prompt_version_hash(payload1)
    hash2 = compute_prompt_version_hash(payload2)

    assert hash1 == hash2
    assert re.match(r"^[0-9a-f]{64}$", hash1)


def test_prompt_version_hash_changes_with_content():
    template1 = {
        "format": "STANDARD",
        "systemPrompt": "System 1",
        "userPrompt": "User {{x}}",
    }
    template2 = {
        "format": "STANDARD",
        "systemPrompt": "System 2",
        "userPrompt": "User {{x}}",
    }
    vars1 = [{"name": "x", "type": "STRING", "required": True}]
    schema = {"schemaType": "JSON_SCHEMA", "definition": {}}

    hash1 = compute_prompt_version_hash(
        build_canonical_payload(template1, vars1, schema)
    )
    hash2 = compute_prompt_version_hash(
        build_canonical_payload(template2, vars1, schema)
    )
    assert hash1 != hash2


def test_prompt_version_hash_independent_of_line_endings():
    t_crlf = {
        "format": "STANDARD",
        "systemPrompt": "Hello\r\nWorld",
        "userPrompt": "Input: {{x}}\r\n",
    }
    t_lf = {
        "format": "STANDARD",
        "systemPrompt": "Hello\nWorld",
        "userPrompt": "Input: {{x}}\n",
    }
    vars_ = [{"name": "x", "type": "STRING", "required": True}]
    schema = {"schemaType": "JSON_SCHEMA", "definition": {}}

    hash1 = compute_prompt_version_hash(
        build_canonical_payload(t_crlf, vars_, schema)
    )
    hash2 = compute_prompt_version_hash(
        build_canonical_payload(t_lf, vars_, schema)
    )
    assert hash1 == hash2


def test_prompt_version_hash_independent_of_variable_input_order():
    # If variables are provided in different order in the list, canonical payload sorts them by name
    t = {"format": "STANDARD", "systemPrompt": "Sys", "userPrompt": "Usr"}
    vars1 = [
        {"name": "zebra", "type": "STRING", "required": True},
        {"name": "alpha", "type": "STRING", "required": True},
    ]
    vars2 = [
        {"name": "alpha", "type": "STRING", "required": True},
        {"name": "zebra", "type": "STRING", "required": True},
    ]
    schema = {"schemaType": "JSON_SCHEMA", "definition": {}}

    hash1 = compute_prompt_version_hash(build_canonical_payload(t, vars1, schema))
    hash2 = compute_prompt_version_hash(build_canonical_payload(t, vars2, schema))
    assert hash1 == hash2
