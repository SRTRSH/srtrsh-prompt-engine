"""Prompt Version canonical payload builder and SHA-256 PromptVersionHash calculator.

Normative reference:
WikiDocs/14-prompt-engine/02-domain-model/03-identity-and-versioning.md
"""

import hashlib
import unicodedata
from typing import Any, Dict, List, Optional, Set

from srtrsh_prompt_engine.canonical.jcs import canonicalize


def normalize_template_text(text: str) -> str:
    """Normalize template text:

    - Line-ending normalization to LF ('\\n')
    - Unicode Normalization Form C (NFC)
    - Whitespace authored by user is preserved.
    """
    if text is None:
        return ""
    # Normalize CRLF and CR to LF
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Unicode NFC
    return unicodedata.normalize("NFC", normalized)


def build_canonical_payload(
    template: Dict[str, Any],
    variables: List[Dict[str, Any]],
    schema_contract: Dict[str, Any],
    execution_configuration: Optional[Dict[str, Any]] = None,
    override_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct the normative CANONICAL_PAYLOAD for PromptVersion hashing.

    The canonical payload MUST contain exactly:
    - executionConfiguration (with maxOutputTokens, providerModel, temperature, timeoutDuration)
    - overridePolicy (with allowedParameters sorted)
    - schemaContract (with definition, schemaType)
    - template (with format, systemPrompt, userPrompt normalized)
    - variables (sorted lexicographically by name)
    """
    exec_cfg = execution_configuration or {}
    pm = exec_cfg.get("providerModel")
    provider_model_normalized: Optional[Dict[str, str]] = None
    if pm:
        provider_model_normalized = {
            "modelName": str(pm.get("modelName", "")),
            "providerId": str(pm.get("providerId", "")),
        }

    canonical_exec_cfg = {
        "maxOutputTokens": exec_cfg.get("maxOutputTokens", None),
        "providerModel": provider_model_normalized,
        "temperature": exec_cfg.get("temperature", None),
        "timeoutDuration": exec_cfg.get("timeoutDuration", None),
    }

    op = override_policy or {}
    allowed_params = op.get("allowedParameters", [])
    if isinstance(allowed_params, (set, list, tuple)):
        sorted_allowed_params = sorted(str(p) for p in allowed_params)
    else:
        sorted_allowed_params = []

    canonical_override_policy = {
        "allowedParameters": sorted_allowed_params,
    }

    canonical_schema_contract = {
        "definition": schema_contract.get("definition", {}),
        "schemaType": schema_contract.get("schemaType", "JSON_SCHEMA"),
    }

    canonical_template = {
        "format": template.get("format", "STANDARD"),
        "systemPrompt": normalize_template_text(template.get("systemPrompt", "")),
        "userPrompt": normalize_template_text(template.get("userPrompt", "")),
    }

    # Deterministic lexical ordering by name for variables
    normalized_vars: List[Dict[str, Any]] = []
    for v in variables:
        normalized_vars.append(
            {
                "defaultValue": v.get("defaultValue", None),
                "name": str(v.get("name", "")),
                "required": bool(v.get("required", False)),
                "type": str(v.get("type", "STRING")),
            }
        )
    normalized_vars.sort(key=lambda item: item["name"])

    return {
        "executionConfiguration": canonical_exec_cfg,
        "overridePolicy": canonical_override_policy,
        "schemaContract": canonical_schema_contract,
        "template": canonical_template,
        "variables": normalized_vars,
    }


def compute_prompt_version_hash(canonical_payload: Dict[str, Any]) -> str:
    """Compute 64-char lowercase hexadecimal SHA-256 PromptVersionHash from canonical payload."""
    canonical_bytes = canonicalize(canonical_payload)
    return hashlib.sha256(canonical_bytes).hexdigest().lower()
