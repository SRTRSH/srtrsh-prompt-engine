"""Variable and context binding logic for Data Plane resolution."""

import json
import re
from typing import Any, Dict, List, Tuple

from srtrsh_prompt_engine.domain.models import (
    BoundVariable,
    MissingReason,
    MissingVariable,
    PromptVersion,
    VariableSource,
)
from srtrsh_prompt_engine.domain.runtime import ResolutionRequest


def _try_extract_from_raw_input(raw_input: str, var_name: str) -> Any:
    if not raw_input:
        return None
    # 1. Try parsing raw_input as JSON
    try:
        data = json.loads(raw_input)
        if isinstance(data, dict) and var_name in data:
            return data[var_name]
    except Exception:
        pass

    # 2. Try parsing line matching "var_name: value"
    pattern = rf"(?i)^{re.escape(var_name)}\s*[:=]\s*(.+)$"
    for line in raw_input.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            extracted_str = match.group(1).strip()
            try:
                return json.loads(extracted_str)
            except Exception:
                return extracted_str

    return None


def bind_variables(
    version: PromptVersion, request: ResolutionRequest
) -> Tuple[List[BoundVariable], List[MissingVariable]]:
    """Bind incoming runtime inputs, context, and defaults against the version's variable definitions.

    Returns:
        (bound_variables, missing_variables)
    """
    bound_list: List[BoundVariable] = []
    missing_list: List[MissingVariable] = []

    for var_def in version.definition.variables:
        name = var_def.name
        val: Any = None
        source: VariableSource | None = None

        # 1. Explicit runtime input
        if name in request.input:
            val = request.input[name]
            if val is not None:
                source = VariableSource.RUNTIME_INPUT

        # 2. Auxiliary runtime context
        if val is None and name in request.runtime_context:
            val = request.runtime_context[name]
            if val is not None:
                source = VariableSource.CONTEXT_PAYLOAD

        # 3. Extraction from rawInput if present
        if val is None and request.raw_input:
            val = _try_extract_from_raw_input(request.raw_input, name)
            if val is not None:
                source = VariableSource.RUNTIME_INPUT

        # 4. Default value
        if val is None and var_def.default_value is not None:
            val = var_def.default_value
            source = VariableSource.DEFAULT_VALUE

        if val is not None and source is not None:
            bound_list.append(BoundVariable(name=name, value=val, source=source))
        elif var_def.required:
            reason = (
                MissingReason.NULL_VALUE
                if (name in request.input and request.input[name] is None)
                else MissingReason.NOT_PROVIDED
            )
            missing_list.append(MissingVariable(name=name, reason=reason))

    return bound_list, missing_list
