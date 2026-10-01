"""Multi-cycle AI Execution Engine and Session Loop.

Normative references:
WikiDocs/14-prompt-engine/06-ai-session-lifecycle/05-processing-attempt.md (C-06)
WikiDocs/14-prompt-engine/07-execution-model/01-execution-overview.md
WikiDocs/14-prompt-engine/07-execution-model/03-dynamic-operation-selection.md (C-07)
WikiDocs/14-prompt-engine/07-execution-model/09-execution-outcomes.md (BLK-04)
WikiDocs/14-prompt-engine/07-execution-model/10-execution-failure-semantics.md (C-16)
WikiDocs/14-prompt-engine/11-limits-and-guardrails/
WikiDocs/14-prompt-engine/15-persistence/04-audit-persistence.md (C-19, C-20)
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional
import uuid

from srtrsh_prompt_engine.compiler.catalog_compiler import CatalogCompiler
from srtrsh_prompt_engine.domain.models import (
    AuditRecord,
    PromptCase,
)
from srtrsh_prompt_engine.execution.context_assembler import ContextAssembler
from srtrsh_prompt_engine.execution.dispatcher import OperationDispatcher
from srtrsh_prompt_engine.execution.outcomes import (
    ExecutionOutcome,
    ExecutionOutcomeType,
    InteractiveOption,
)
from srtrsh_prompt_engine.execution.session_engine import SessionEngine
from srtrsh_prompt_engine.spi.provider import (
    HydratedPrompt,
    ProviderExecutionRequest,
    ProviderSPI,
)
from srtrsh_prompt_engine.storage.repository import (
    AuditRepository,
    EndpointCatalogRepository,
    PromptCaseRepository,
    VariableCatalogRepository,
)


class ExecutionLimits:
    def __init__(
        self,
        max_llm_calls: int = 5,
        timeout_seconds: float = 30.0,
        max_result_items: int = 10,
    ) -> None:
        self.max_llm_calls = max_llm_calls
        self.timeout_seconds = timeout_seconds
        self.max_result_items = max_result_items


class ExecutionEngine:
    """Core runtime engine driving the multi-cycle execution loop for a customer attempt."""

    def __init__(
        self,
        session_engine: SessionEngine,
        prompt_case_repo: PromptCaseRepository,
        endpoint_catalog_repo: EndpointCatalogRepository,
        variable_catalog_repo: VariableCatalogRepository,
        audit_repo: AuditRepository,
        dispatcher: OperationDispatcher,
        provider: ProviderSPI,
        limits: Optional[ExecutionLimits] = None,
    ) -> None:
        self.session_engine = session_engine
        self.prompt_case_repo = prompt_case_repo
        self.endpoint_catalog_repo = endpoint_catalog_repo
        self.variable_catalog_repo = variable_catalog_repo
        self.audit_repo = audit_repo
        self.dispatcher = dispatcher
        self.provider = provider
        self.limits = limits or ExecutionLimits()

    async def execute_attempt(
        self,
        conversation_id: str,
        message: Optional[str] = None,
        interaction_result: Optional[Dict[str, Any]] = None,
        runtime_values: Optional[Dict[str, Any]] = None,
        business_references: Optional[Dict[str, Any]] = None,
        case_key: str = "resolve_user_request",
    ) -> ExecutionOutcome:
        """Execute one customer processing attempt across one or more provider cycles (C-06)."""
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        start_time = time.monotonic()

        # 1. Resolve or create active session
        session = self.session_engine.get_or_create_session(
            conversation_id=conversation_id,
            business_references=business_references,
        )
        observed_version = session.session_version

        # 2. Ingest runtime values and interaction result into session state
        if runtime_values:
            session.session_state.update(runtime_values)
        if interaction_result:
            session.interaction_results.append(interaction_result)
            session.session_state.update(interaction_result)

        # 3. Durably record incoming customer message
        if message:
            self.session_engine.record_message(
                session_id=session.session_id,
                conversation_id=conversation_id,
                role="customer",
                content=message,
            )

        # 4. Load Organization System Prompt Case and Target Prompt Case
        org_case = self.prompt_case_repo.get_by_key("organization_system")
        org_published = org_case.get_published_version() if org_case else None
        org_system_prompt = (
            org_published.definition.template.system_prompt
            if org_published
            else "You are the SRTRSH AI Salon Assistant."
        )

        target_case = self.prompt_case_repo.get_by_key(case_key)
        target_published = target_case.get_published_version() if target_case else None
        case_instructions = (
            target_published.definition.template.user_prompt
            if target_published
            else "Assist the client with booking, scheduling, and inquiries."
        )
        prompt_version_hash = (
            target_published.version_hash
            if target_published and target_published.version_hash
            else "00" * 32
        )
        prompt_version_number = (
            target_published.version_number if target_published else 1
        )

        # 5. Compile Endpoint & Variable Catalogs into canonical XML (C-29)
        endpoint_entries = getattr(
            self.endpoint_catalog_repo, "list_endpoint_entries", self.endpoint_catalog_repo.list_entries
        )()
        endpoint_rev = (
            getattr(self.endpoint_catalog_repo, "get_endpoint_catalog_revision", None)()
            if hasattr(self.endpoint_catalog_repo, "get_endpoint_catalog_revision")
            else self.endpoint_catalog_repo.get_current_revision()
        )
        compiled_endpoints = CatalogCompiler.compile_endpoint_catalog(
            endpoint_entries, endpoint_rev
        )

        variable_entries = getattr(
            self.variable_catalog_repo, "list_variable_entries", self.variable_catalog_repo.list_entries
        )()
        variable_rev = (
            getattr(self.variable_catalog_repo, "get_variable_catalog_revision", None)()
            if hasattr(self.variable_catalog_repo, "get_variable_catalog_revision")
            else self.variable_catalog_repo.get_current_revision()
        )
        compiled_variables = CatalogCompiler.compile_variable_catalog(
            variable_entries, variable_rev
        )

        # Build stable prefix before cache boundary
        system_prompt = ContextAssembler.assemble_system_prompt(
            org_system_prompt=org_system_prompt,
            case_instructions=case_instructions,
            compiled_endpoint_catalog=compiled_endpoints,
            compiled_variable_catalog=compiled_variables,
        )

        # 6. Multi-Cycle Execution Loop
        llm_call_count = 0
        accumulated_op_results: List[Dict[str, Any]] = []
        total_tokens = 0
        total_latency_ms = 0.0

        while llm_call_count < self.limits.max_llm_calls:
            # Check elapsed timeout
            elapsed = time.monotonic() - start_time
            if elapsed > self.limits.timeout_seconds:
                return ExecutionOutcome(
                    execution_id=execution_id,
                    status=ExecutionOutcomeType.LIMIT_EXHAUSTED,
                    error_message=f"Attempt timed out after {elapsed:.1f}s",
                    session_id=session.session_id,
                    conversation_id=conversation_id,
                    session_version=session.session_version,
                    structured_state=session.session_state,
                )

            # Select bounded history (C-25, C-26)
            history_messages = self.session_engine.select_history_context(
                conversation_id=conversation_id,
                current_session_id=session.session_id,
                recent_messages_limit=10,
                previous_sessions_limit=2,
            )

            # Assemble dynamic user prompt
            user_prompt = ContextAssembler.assemble_user_prompt(
                session_state=session.session_state,
                business_references=session.business_references,
                history_messages=history_messages,
                current_message=message,
                interaction_result=interaction_result if llm_call_count == 0 else None,
                operation_results=accumulated_op_results if accumulated_op_results else None,
            )

            # Build provider request
            provider_req = ProviderExecutionRequest(
                execution_id=f"{execution_id}_{llm_call_count}",
                prompt_version_hash=prompt_version_hash,
                provider="default",
                model="default",
                prompt=HydratedPrompt(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ),
                available_operations=[e.operation_id for e in endpoint_entries],
            )

            llm_call_count += 1
            call_start = time.monotonic()
            llm_resp = await self.provider.execute(provider_req)
            call_latency = (time.monotonic() - call_start) * 1000.0

            total_tokens += llm_resp.usage.total_tokens
            total_latency_ms += call_latency

            # Check for provider failure
            if llm_resp.status == "FAILURE":
                return ExecutionOutcome(
                    execution_id=execution_id,
                    status=ExecutionOutcomeType.TECHNICAL_ERROR,
                    error_message=llm_resp.error_message or "LLM Provider call failed.",
                    session_id=session.session_id,
                    conversation_id=conversation_id,
                    session_version=session.session_version,
                    structured_state=session.session_state,
                )

            # Intermediate Cycle Action: CALL_OPERATION (BLK-04)
            if llm_resp.operation_calls:
                for op_call in llm_resp.operation_calls:
                    dispatch_res = self.dispatcher.execute_operation(
                        operation_id=op_call.operation_id,
                        arguments=op_call.arguments,
                        context_state=session.session_state,
                    )
                    accumulated_op_results.append({
                        "operationId": op_call.operation_id,
                        "callId": op_call.call_id,
                        "result": dispatch_res.to_dict(),
                    })
                    # If operation produced structured dict output, accumulate into session state
                    if dispatch_res.success and isinstance(dispatch_res.data, dict):
                        session.session_state.update(dispatch_res.data)

                # Continue next cycle in loop
                continue

            # Terminal Outcome: INTERACTIVE_RESPONSE
            if llm_resp.interaction:
                options = [
                    InteractiveOption(label=opt["label"], result=opt["result"])
                    for opt in llm_resp.interaction.options
                ]
                # Record assistant message
                self.session_engine.record_message(
                    session_id=session.session_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=llm_resp.interaction.text,
                )
                # Atomically update session
                session = self.session_engine.update_session(session, observed_version)

                # Record audit record
                self.audit_repo.record_audit(
                    AuditRecord(
                        execution_id=execution_id,
                        session_id=session.session_id,
                        conversation_id=conversation_id,
                        case_key=case_key,
                        prompt_version_number=prompt_version_number,
                        prompt_version_hash=prompt_version_hash,
                        endpoint_catalog_revision=endpoint_rev,
                        variable_catalog_revision=variable_rev,
                        effective_configuration={},
                        inputs={"message": message, "interaction_result": interaction_result},
                        outputs={"status": "INTERACTIVE_RESPONSE", "text": llm_resp.interaction.text},
                        telemetry={"total_tokens": total_tokens, "latency_ms": total_latency_ms, "llm_calls": llm_call_count},
                    )
                )

                return ExecutionOutcome(
                    execution_id=execution_id,
                    status=ExecutionOutcomeType.INTERACTIVE_RESPONSE,
                    text=llm_resp.interaction.text,
                    options=options,
                    session_id=session.session_id,
                    conversation_id=conversation_id,
                    session_version=session.session_version,
                    structured_state=session.session_state,
                    telemetry={"total_tokens": total_tokens, "latency_ms": total_latency_ms, "llm_calls": llm_call_count},
                )

            # Terminal Outcome: Assistant Text Content
            content_text = llm_resp.content or llm_resp.raw_text or ""

            # Save assistant message
            self.session_engine.record_message(
                session_id=session.session_id,
                conversation_id=conversation_id,
                role="assistant",
                content=content_text,
            )

            # Update session with optimistic locking
            session = self.session_engine.update_session(session, observed_version)

            # Check for Human Fallback vs Needs User Input vs Completed (C-16)
            lower_content = content_text.lower()
            if any(k in lower_content for k in ("человек", "оператор", "администратор", "human fallback", "transfer to human")):
                outcome_status = ExecutionOutcomeType.HUMAN_FALLBACK
            elif any(k in lower_content for k in ("уточните", "какой", "когда", "выберите", "?", "подскажите")):
                outcome_status = ExecutionOutcomeType.NEEDS_USER_INPUT
            else:
                outcome_status = ExecutionOutcomeType.COMPLETED

            # Record audit record
            self.audit_repo.record_audit(
                AuditRecord(
                    execution_id=execution_id,
                    session_id=session.session_id,
                    conversation_id=conversation_id,
                    case_key=case_key,
                    prompt_version_number=prompt_version_number,
                    prompt_version_hash=prompt_version_hash,
                    endpoint_catalog_revision=endpoint_rev,
                    variable_catalog_revision=variable_rev,
                    effective_configuration={},
                    inputs={"message": message, "interaction_result": interaction_result},
                    outputs={"status": outcome_status.value, "text": content_text},
                    telemetry={"total_tokens": total_tokens, "latency_ms": total_latency_ms, "llm_calls": llm_call_count},
                )
            )

            return ExecutionOutcome(
                execution_id=execution_id,
                status=outcome_status,
                text=content_text,
                session_id=session.session_id,
                conversation_id=conversation_id,
                session_version=session.session_version,
                structured_state=session.session_state,
                telemetry={"total_tokens": total_tokens, "latency_ms": total_latency_ms, "llm_calls": llm_call_count},
            )

        # Loop exhausted max_llm_calls
        return ExecutionOutcome(
            execution_id=execution_id,
            status=ExecutionOutcomeType.LIMIT_EXHAUSTED,
            error_message=f"Exhausted maximum permitted LLM calls ({self.limits.max_llm_calls})",
            session_id=session.session_id,
            conversation_id=conversation_id,
            session_version=session.session_version,
            structured_state=session.session_state,
            telemetry={"total_tokens": total_tokens, "latency_ms": total_latency_ms, "llm_calls": llm_call_count},
        )
