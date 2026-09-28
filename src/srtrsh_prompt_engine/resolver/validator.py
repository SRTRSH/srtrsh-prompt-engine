"""SchemaContract validation using JSON Schema."""

from typing import Any, Dict, Optional, Tuple

import jsonschema


def validate_schema(
    instance: Dict[str, Any], schema_definition: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Validate structured output dictionary against JSON Schema contract definition.

    Returns (is_valid, error_message).
    """
    if not schema_definition:
        return True, None
    try:
        jsonschema.validate(
            instance=instance,
            schema=schema_definition,
            format_checker=jsonschema.FormatChecker(),
        )
        return True, None
    except jsonschema.ValidationError as err:
        return False, err.message
    except jsonschema.SchemaError as err:
        return False, f"Invalid schema definition: {err.message}"
