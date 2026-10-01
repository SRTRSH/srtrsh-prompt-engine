"""Canonical XML Catalog Compiler for Endpoint Catalog and Variable Catalog.

Normative references:
WikiDocs/14-prompt-engine/03-domain-model/04-endpoint-catalog.md
WikiDocs/14-prompt-engine/03-domain-model/05-variable-catalog.md
WikiDocs/14-prompt-engine/05-context-and-prompt-assembly/02-stable-context.md
WikiDocs/14-prompt-engine/05-context-and-prompt-assembly/06-prompt-caching.md
"""

from typing import List, Optional
import xml.sax.saxutils as saxutils

from srtrsh_prompt_engine.domain.models import (
    EndpointCatalogEntry,
    VariableCatalogEntry,
)


def _escape_xml_text(text: str) -> str:
    """Escape XML entities in text content."""
    if not text:
        return ""
    return saxutils.escape(text)


class CatalogCompiler:
    """Compiles structured Endpoint Catalog and Variable Catalog entries into

    compact, deterministic XML context for LLM consumption.
    """

    @staticmethod
    def compile_endpoint_catalog(
        entries: List[EndpointCatalogEntry], revision: str
    ) -> str:
        """Compile endpoint catalog into canonical XML sorted ascending by operationId.

        Omit persistence fields, transport URLs, HTTP details, and implementation metadata.
        Preserve mandatory variable marker '!'.
        """
        # Sort operations deterministically ascending by operation_id
        sorted_entries = sorted(entries, key=lambda e: e.operation_id)

        lines = [f'<endpoint_catalog revision="{_escape_xml_text(revision)}">']
        for op in sorted_entries:
            lines.append(f'  <operation id="{_escape_xml_text(op.operation_id)}">')
            lines.append(f'    <description>{_escape_xml_text(op.description)}</description>')
            if op.input_variables:
                input_str = " ".join(op.input_variables)
                lines.append(f'    <input>{_escape_xml_text(input_str)}</input>')
            if op.output_variables:
                output_str = " ".join(op.output_variables)
                lines.append(f'    <output>{_escape_xml_text(output_str)}</output>')
            lines.append('  </operation>')
        lines.append('</endpoint_catalog>')
        return "\n".join(lines)

    @staticmethod
    def compile_variable_catalog(
        entries: List[VariableCatalogEntry], revision: str
    ) -> str:
        """Compile variable catalog into canonical XML sorted ascending by variableId."""
        # Sort variables deterministically ascending by variable_id
        sorted_entries = sorted(entries, key=lambda v: v.variable_id)

        lines = [f'<variables_metadata revision="{_escape_xml_text(revision)}">']
        for var in sorted_entries:
            type_val = var.type.value if hasattr(var.type, "value") else str(var.type).lower()
            lines.append(
                f'  <variable id="{_escape_xml_text(var.variable_id)}" type="{_escape_xml_text(type_val)}">'
                f'{_escape_xml_text(var.description)}</variable>'
            )
        lines.append('</variables_metadata>')
        return "\n".join(lines)

    @classmethod
    def compile_full_catalog(
        cls,
        endpoints: List[EndpointCatalogEntry],
        endpoint_revision: str,
        variables: List[VariableCatalogEntry],
        variable_revision: str,
    ) -> str:
        """Compile both catalogs and join them as the stable context catalog section."""
        ep_xml = cls.compile_endpoint_catalog(endpoints, endpoint_revision)
        var_xml = cls.compile_variable_catalog(variables, variable_revision)
        return f"{ep_xml}\n{var_xml}"


class CompiledCatalogCache:
    """Thread-safe cache for compiled catalog XML, invalidated by catalog revision changes (C-30)."""

    def __init__(self) -> None:
        self._cached_endpoint_xml: Optional[str] = None
        self._cached_variable_xml: Optional[str] = None
        self._cached_endpoint_revision: Optional[str] = None
        self._cached_variable_revision: Optional[str] = None

    def get_compiled_endpoint_catalog(
        self, entries: List[EndpointCatalogEntry], revision: str
    ) -> str:
        if self._cached_endpoint_revision != revision or self._cached_endpoint_xml is None:
            self._cached_endpoint_xml = CatalogCompiler.compile_endpoint_catalog(entries, revision)
            self._cached_endpoint_revision = revision
        return self._cached_endpoint_xml

    def get_compiled_variable_catalog(
        self, entries: List[VariableCatalogEntry], revision: str
    ) -> str:
        if self._cached_variable_revision != revision or self._cached_variable_xml is None:
            self._cached_variable_xml = CatalogCompiler.compile_variable_catalog(entries, revision)
            self._cached_variable_revision = revision
        return self._cached_variable_xml

    def get_compiled_full_catalog(
        self,
        endpoints: List[EndpointCatalogEntry],
        endpoint_revision: str,
        variables: List[VariableCatalogEntry],
        variable_revision: str,
    ) -> str:
        ep_xml = self.get_compiled_endpoint_catalog(endpoints, endpoint_revision)
        var_xml = self.get_compiled_variable_catalog(variables, variable_revision)
        return f"{ep_xml}\n{var_xml}"

    def invalidate(self) -> None:
        """Explicitly clear cached compiled representations."""
        self._cached_endpoint_xml = None
        self._cached_variable_xml = None
        self._cached_endpoint_revision = None
        self._cached_variable_revision = None
