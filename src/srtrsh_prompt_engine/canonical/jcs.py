"""RFC 8785 JSON Canonicalization Scheme (JCS) implementation.

Reference: https://datatracker.ietf.org/doc/html/rfc8785
"""

import math
import unicodedata
from typing import Any, Dict, List


def _escape_string(s: str) -> bytes:
    """Serialize string according to RFC 8785 rules.

    Only quotes, reverse solidus, and control characters U+0000 through U+001F
    are escaped. Non-ASCII characters are output as UTF-8 directly without
    escaping, and must be in Unicode Normalization Form C (NFC).
    """
    normalized = unicodedata.normalize("NFC", s)
    out: List[str] = ['"']
    for char in normalized:
        code = ord(char)
        if char == '"':
            out.append('\\"')
        elif char == "\\":
            out.append("\\\\")
        elif char == "\b":
            out.append("\\b")
        elif char == "\f":
            out.append("\\f")
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out).encode("utf-8")


def _format_number(val: int | float) -> bytes:
    """Format numbers according to ECMAScript 6 / RFC 8785 Section 3.2.2.3."""
    if isinstance(val, bool):
        raise TypeError("Boolean value passed as number")
    if isinstance(val, int):
        return str(val).encode("utf-8")
    if math.isnan(val) or math.isinf(val):
        raise ValueError("NaN and Infinity are not permitted in RFC 8785 JCS")
    if val == 0.0:
        return b"0"
    if val.is_integer() and abs(val) < 1e21:
        return str(int(val)).encode("utf-8")
    formatted = str(val)
    # Standardize exponent representation (e.g. e+05 -> e+5)
    if "e" in formatted:
        parts = formatted.split("e")
        exp = int(parts[1])
        formatted = f"{parts[0]}e{'+' if exp >= 0 else ''}{exp}"
    return formatted.encode("utf-8")


def canonicalize(obj: Any) -> bytes:
    """Serialize Python object into canonical JSON bytes according to RFC 8785.

    - UTF-8 encoding
    - No structural whitespace
    - Object keys sorted by UTF-16 code units
    - Numbers formatted per ECMAScript 6+
    - Minimal string escaping (only quotes, backslash, control chars)
    """
    if obj is None:
        return b"null"
    if isinstance(obj, bool):
        return b"true" if obj else b"false"
    if isinstance(obj, (int, float)):
        return _format_number(obj)
    if isinstance(obj, str):
        return _escape_string(obj)
    if isinstance(obj, (list, tuple)):
        return b"[" + b",".join(canonicalize(elem) for elem in obj) + b"]"
    if isinstance(obj, dict):
        # Sort keys by UTF-16 code units
        sorted_keys = sorted(obj.keys(), key=lambda k: str(k).encode("utf-16-be"))
        items = [
            _escape_string(str(k)) + b":" + canonicalize(obj[k])
            for k in sorted_keys
        ]
        return b"{" + b",".join(items) + b"}"
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
