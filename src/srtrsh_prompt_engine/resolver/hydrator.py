"""Prompt template hydrator."""

import json
from typing import Any, Dict


def _serialize_value(val: Any) -> str:
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def hydrate_template(
    template_text: str,
    bound_variables: Dict[str, Any],
    raw_input: str = "",
    runtime_context: Dict[str, Any] | None = None,
) -> str:
    """Hydrate prompt template by substituting variable placeholders.

    Supports both {{variable}} and {variable} formats.
    Also makes {{raw_input}} available if present in template.
    """
    hydrated = template_text
    context = runtime_context or {}

    # Pool of available replacements
    replacements: Dict[str, Any] = dict(context)
    replacements.update(bound_variables)
    if "raw_input" not in replacements:
        replacements["raw_input"] = raw_input

    for key, val in replacements.items():
        str_val = _serialize_value(val)
        hydrated = hydrated.replace(f"{{{{{key}}}}}", str_val)
        hydrated = hydrated.replace(f"{{{key}}}", str_val)

    return hydrated
