"""Operation Dispatcher validating and executing permitted READ operations.

Normative references:
WikiDocs/14-prompt-engine/02-architecture/04-safety-and-business-boundary.md (C-11)
WikiDocs/14-prompt-engine/03-domain-model/04-endpoint-catalog.md (C-04)
WikiDocs/14-prompt-engine/04-identities-versioning-and-invariants/04-invariants.md (C-08)
WikiDocs/14-prompt-engine/05-context-and-prompt-assembly/05-context-integrity.md (C-09, C-10)
"""

from typing import Any, Callable, Dict, List, Optional

from srtrsh_prompt_engine.domain.models import EndpointCatalogEntry
from srtrsh_prompt_engine.storage.repository import EndpointCatalogRepository


class DispatchResult:
    """Result of an operation execution."""

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error_message: Optional[str] = None,
        is_ambiguous: bool = False,
        is_truncated: bool = False,
        was_reused: bool = False,
    ) -> None:
        self.success = success
        self.data = data
        self.error_message = error_message
        self.is_ambiguous = is_ambiguous
        self.is_truncated = is_truncated
        self.was_reused = was_reused

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "success": self.success,
            "data": self.data,
            "wasReused": self.was_reused,
        }
        if self.is_ambiguous:
            res["isAmbiguous"] = True
        if self.is_truncated:
            res["isTruncated"] = True
            res["warning"] = "Слишком много результатов. Уточните поиск."
        if self.error_message:
            res["errorMessage"] = self.error_message
        return res


class OperationDispatcher:
    """Validates and executes operations requested by LLM."""

    def __init__(
        self,
        catalog_repository: EndpointCatalogRepository,
        max_result_items: int = 10,
    ) -> None:
        self.catalog_repo = catalog_repository
        self.max_result_items = max_result_items
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._execution_cache: Dict[str, Any] = {}

    def register_handler(
        self, operation_id: str, handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Register execution callable for an operationId."""
        self._handlers[operation_id] = handler

    def execute_operation(
        self,
        operation_id: str,
        arguments: Dict[str, Any],
        context_state: Optional[Dict[str, Any]] = None,
    ) -> DispatchResult:
        """Validate and dispatch operation."""
        # 1. Validate operation exists in Endpoint Catalog (C-04)
        entry: Optional[EndpointCatalogEntry] = self.catalog_repo.get_entry(operation_id)
        if not entry:
            return DispatchResult(
                success=False,
                error_message=f"Operation '{operation_id}' is not registered in Endpoint Catalog.",
            )

        # 2. Safety boundary: AI write operations unavailable in MVP (C-11)
        if not entry.is_read_only:
            return DispatchResult(
                success=False,
                error_message=f"Write operation '{operation_id}' is forbidden in Prompt Engine MVP.",
            )

        # 3. Existing Context Value Reuse (C-08): Check if result already cached with exact arguments
        cache_key = f"{operation_id}::{sorted(arguments.items())}"
        if cache_key in self._execution_cache:
            return DispatchResult(
                success=True,
                data=self._execution_cache[cache_key],
                was_reused=True,
            )

        # 4. Invoke registered handler or default mock response
        handler = self._handlers.get(operation_id)
        if handler:
            try:
                raw_result = handler(arguments)
            except Exception as e:
                return DispatchResult(success=False, error_message=str(e))
        else:
            # Default empty mock result if no handler is registered
            raw_result = {}

        # 5. Result Ambiguity Check (C-09)
        # If result is a list with multiple options, preserve all items (do not silently pick one)
        is_ambiguous = isinstance(raw_result, list) and len(raw_result) > 1

        # 6. Result Limits (C-10)
        # If result is a list exceeding max_result_items, bound it and signal truncation
        is_truncated = False
        if isinstance(raw_result, list) and len(raw_result) > self.max_result_items:
            raw_result = raw_result[: self.max_result_items]
            is_truncated = True

        # Cache result for reuse (C-08)
        self._execution_cache[cache_key] = raw_result

        return DispatchResult(
            success=True,
            data=raw_result,
            is_ambiguous=is_ambiguous,
            is_truncated=is_truncated,
            was_reused=False,
        )

    def clear_cache(self) -> None:
        self._execution_cache.clear()
