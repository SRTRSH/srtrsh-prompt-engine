"""Context and Prompt Assembly with strict cache boundary order.

Normative references:
WikiDocs/14-prompt-engine/05-context-and-prompt-assembly/01-assembly-overview.md
WikiDocs/14-prompt-engine/05-context-and-prompt-assembly/02-stable-context.md
WikiDocs/14-prompt-engine/05-context-and-prompt-assembly/04-historical-context.md
WikiDocs/14-prompt-engine/05-context-and-prompt-assembly/05-context-integrity.md
WikiDocs/14-prompt-engine/05-context-and-prompt-assembly/06-prompt-caching.md
"""

import json
from typing import Any, Dict, List, Optional

from srtrsh_prompt_engine.domain.models import SessionMessage


class ContextAssembler:
    """Assembles prompt context respecting the normative cache boundary:

    [Organization System Rules]
      -> [Prompt Case Rules]
      -> [Endpoint Catalog]
      -> [Variables Metadata]
      -> CACHE BOUNDARY
      -> [Persistent Session State]
      -> [Conversation History]
      -> [Current User Message / Interaction Result]
    """

    CACHE_BOUNDARY_MARKER = "--- CACHE BOUNDARY ---"

    @classmethod
    def assemble_system_prompt(
        cls,
        org_system_prompt: str,
        case_instructions: str,
        compiled_endpoint_catalog: str,
        compiled_variable_catalog: str,
    ) -> str:
        """Assemble the stable prefix before the cache boundary."""
        parts = [
            "<organization_system>",
            org_system_prompt.strip(),
            "</organization_system>",
            "",
            "<case_instructions>",
            case_instructions.strip(),
            "</case_instructions>",
            "",
            compiled_endpoint_catalog.strip(),
            "",
            compiled_variable_catalog.strip(),
            "",
            cls.CACHE_BOUNDARY_MARKER,
        ]
        return "\n".join(parts)

    @classmethod
    def assemble_user_prompt(
        cls,
        session_state: Dict[str, Any],
        business_references: Dict[str, Any],
        history_messages: List[SessionMessage],
        current_message: Optional[str] = None,
        interaction_result: Optional[Dict[str, Any]] = None,
        operation_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Assemble the dynamic context after the cache boundary."""
        parts = []

        # 1. Structured Session State & Business References
        effective_state = dict(session_state)
        if business_references:
            effective_state["_business_references"] = business_references

        parts.append("<session_state>")
        parts.append(json.dumps(effective_state, ensure_ascii=False, indent=2))
        parts.append("</session_state>")
        parts.append("")

        # 2. Historical Conversation Context
        if history_messages:
            parts.append("<conversation_history>")
            for msg in history_messages:
                parts.append(f'  <message role="{msg.role}">{msg.content}</message>')
            parts.append("</conversation_history>")
            parts.append("")

        # 3. Intermediate Operation Results (if in multi-cycle loop)
        if operation_results:
            parts.append("<operation_results>")
            for op_res in operation_results:
                parts.append(
                    f'  <result operationId="{op_res.get("operationId")}">'
                    f'{json.dumps(op_res.get("result", {}), ensure_ascii=False)}'
                    f'</result>'
                )
            parts.append("</operation_results>")
            parts.append("")

        # 4. Current Interaction Result (button press)
        if interaction_result:
            parts.append("<interaction_result>")
            parts.append(json.dumps(interaction_result, ensure_ascii=False))
            parts.append("</interaction_result>")
            parts.append("")

        # 5. Current User Message
        if current_message:
            parts.append("<current_user_message>")
            parts.append(current_message.strip())
            parts.append("</current_user_message>")

        return "\n".join(parts)
