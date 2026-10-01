"""Phase 2: Catalog Compiler & Provider SPI Tests.

Verifies:
- Canonical deterministic XML compilation for Endpoint Catalog (BLK-01, C-29)
- Canonical deterministic XML compilation for Variable Catalog (BLK-01, AMB-02, C-29)
- Alphabetical ASC ordering of operations and variables
- Preservation of mandatory '!' marker without expanding into verbose XML
- Omission of persistence/transport details from compiled context
- Cache invalidation on catalog revision changes (C-30)
- Provider SPI normalization of operation calls (C-14, C-28)
- Provider error normalization into Prompt Engine categories (C-15)
"""

import pytest

from srtrsh_prompt_engine.compiler.catalog_compiler import (
    CatalogCompiler,
    CompiledCatalogCache,
)
from srtrsh_prompt_engine.domain.models import (
    EndpointCatalogEntry,
    VariableCatalogEntry,
    VariableType,
)
from srtrsh_prompt_engine.spi.generic_llm import normalize_http_error
from srtrsh_prompt_engine.spi.mock_provider import MockProvider
from srtrsh_prompt_engine.spi.provider import (
    HydratedPrompt,
    NormalizedError,
    NormalizedInteractionOutput,
    NormalizedOperationCall,
    ProviderErrorCategory,
    ProviderExecutionRequest,
)


# ============================================================================
# 1. Canonical Catalog Compilation & Ordering Tests (C-29, BLK-01)
# ============================================================================


def test_endpoint_catalog_canonical_compilation_and_sorting():
    """Verify operations are compiled to canonical XML sorted ASC by operationId (C-29)."""
    # Insert in reverse alphabetical order
    ops = [
        EndpointCatalogEntry(
            operation_id="findServices",
            description="Search available salon services",
            input_variables=["master_id!", "query"],
            output_variables=["service_id", "service_name"],
            revision="rev_101",
        ),
        EndpointCatalogEntry(
            operation_id="findAvailableSlots",
            description="Find available time slots for master and service",
            input_variables=["master_id!", "service_id!", "requested_datetime"],
            output_variables=["available_slots"],
            revision="rev_101",
        ),
    ]

    xml = CatalogCompiler.compile_endpoint_catalog(ops, revision="rev_101")

    expected = (
        '<endpoint_catalog revision="rev_101">\n'
        '  <operation id="findAvailableSlots">\n'
        '    <description>Find available time slots for master and service</description>\n'
        '    <input>master_id! service_id! requested_datetime</input>\n'
        '    <output>available_slots</output>\n'
        '  </operation>\n'
        '  <operation id="findServices">\n'
        '    <description>Search available salon services</description>\n'
        '    <input>master_id! query</input>\n'
        '    <output>service_id service_name</output>\n'
        '  </operation>\n'
        '</endpoint_catalog>'
    )
    assert xml == expected
    # findAvailableSlots appears before findServices
    assert xml.index("findAvailableSlots") < xml.index("findServices")


def test_variable_catalog_canonical_compilation_and_sorting():
    """Verify variables are compiled to canonical XML sorted ASC by variableId (C-29, AMB-02)."""
    # Insert in unordered fashion
    vars_list = [
        VariableCatalogEntry(
            variable_id="service_id",
            type=VariableType.STRING,
            description="Unique identifier of the service",
            revision="rev_202",
        ),
        VariableCatalogEntry(
            variable_id="master_id",
            type=VariableType.STRING,
            description="Unique identifier of the master",
            revision="rev_202",
        ),
        VariableCatalogEntry(
            variable_id="requested_datetime",
            type=VariableType.DATETIME,
            description="Requested appointment date and time in ISO-8601",
            revision="rev_202",
        ),
        VariableCatalogEntry(
            variable_id="query",
            type=VariableType.STRING,
            description="Search query for salon services",
            revision="rev_202",
        ),
    ]

    xml = CatalogCompiler.compile_variable_catalog(vars_list, revision="rev_202")

    expected = (
        '<variables_metadata revision="rev_202">\n'
        '  <variable id="master_id" type="string">Unique identifier of the master</variable>\n'
        '  <variable id="query" type="string">Search query for salon services</variable>\n'
        '  <variable id="requested_datetime" type="datetime">Requested appointment date and time in ISO-8601</variable>\n'
        '  <variable id="service_id" type="string">Unique identifier of the service</variable>\n'
        '</variables_metadata>'
    )
    assert xml == expected

    # Ordering check: master_id < query < requested_datetime < service_id
    assert (
        xml.index("master_id")
        < xml.index("query")
        < xml.index("requested_datetime")
        < xml.index("service_id")
    )


def test_catalog_compilation_determinism_byte_for_byte():
    """Byte-for-byte determinism regardless of Python list order."""
    ops1 = [
        EndpointCatalogEntry(operation_id="b_op", description="Op B"),
        EndpointCatalogEntry(operation_id="a_op", description="Op A"),
    ]
    ops2 = [
        EndpointCatalogEntry(operation_id="a_op", description="Op A"),
        EndpointCatalogEntry(operation_id="b_op", description="Op B"),
    ]

    xml1 = CatalogCompiler.compile_endpoint_catalog(ops1, "rev_1")
    xml2 = CatalogCompiler.compile_endpoint_catalog(ops2, "rev_1")
    assert xml1 == xml2


# ============================================================================
# 2. Catalog Cache Invalidation on Revision Change (C-30)
# ============================================================================


def test_compiled_catalog_cache_invalidation_on_revision():
    cache = CompiledCatalogCache()
    ops = [EndpointCatalogEntry(operation_id="findServices", description="Find services")]
    vars_list = [VariableCatalogEntry(variable_id="master_id", type=VariableType.STRING, description="Master ID")]

    xml_v1 = cache.get_compiled_full_catalog(ops, "rev_1", vars_list, "rev_1")
    assert 'revision="rev_1"' in xml_v1

    # Second fetch with same revisions hits cache
    xml_v1_cached = cache.get_compiled_full_catalog(ops, "rev_1", vars_list, "rev_1")
    assert xml_v1_cached == xml_v1

    # When revision changes to rev_2, cache must recompile
    xml_v2 = cache.get_compiled_full_catalog(ops, "rev_2", vars_list, "rev_2")
    assert 'revision="rev_2"' in xml_v2
    assert xml_v2 != xml_v1


# ============================================================================
# 3. Provider SPI Normalization Tests (C-14, C-15, C-28)
# ============================================================================


@pytest.mark.asyncio
async def test_provider_spi_normalized_operation_calls():
    """C-28: Operation calls are normalized into NormalizedOperationCall without provider-native types."""
    provider = MockProvider()
    expected_calls = [
        NormalizedOperationCall(
            call_id="call_abc_1",
            operation_id="findAvailableSlots",
            arguments={"master_id": "M123", "service_id": "S456"},
        )
    ]
    provider.set_operation_calls(expected_calls)

    req = ProviderExecutionRequest(
        execution_id="exec_spi_test",
        prompt_version_hash="00" * 32,
        provider="mock",
        model="mock-model",
        prompt=HydratedPrompt(system_prompt="System", user_prompt="User"),
        available_operations=["findAvailableSlots"],
    )

    resp = await provider.execute(req)
    assert resp.status == "SUCCESS"
    assert len(resp.operation_calls) == 1
    call = resp.operation_calls[0]
    assert call.call_id == "call_abc_1"
    assert call.operation_id == "findAvailableSlots"
    assert call.arguments == {"master_id": "M123", "service_id": "S456"}
    assert resp.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_provider_spi_normalized_interaction_output():
    """Normalized interactive response output."""
    provider = MockProvider()
    interaction = NormalizedInteractionOutput(
        text="Choose an appointment slot:",
        options=[
            {"label": "10:00", "result": {"time": "10:00"}},
            {"label": "11:00", "result": {"time": "11:00"}},
        ],
    )
    provider.set_interaction(interaction)

    req = ProviderExecutionRequest(
        execution_id="exec_interact_test",
        prompt_version_hash="00" * 32,
        provider="mock",
        model="mock-model",
        prompt=HydratedPrompt(system_prompt="System", user_prompt="User"),
    )

    resp = await provider.execute(req)
    assert resp.status == "SUCCESS"
    assert resp.interaction is not None
    assert resp.interaction.text == "Choose an appointment slot:"
    assert len(resp.interaction.options) == 2


def test_provider_error_normalization():
    """C-15: Provider errors are normalized into stable Prompt Engine technical error categories."""
    p_id = "test_provider"
    m_name = "test_model"

    # 408 Timeout -> TIMEOUT, retryable=True
    e408 = normalize_http_error(408, "Request timed out", p_id, m_name)
    assert e408.category == ProviderErrorCategory.TIMEOUT
    assert e408.retryable is True

    # 429 Rate limit -> RATE_LIMIT, retryable=True
    e429 = normalize_http_error(429, "Too many requests", p_id, m_name)
    assert e429.category == ProviderErrorCategory.RATE_LIMIT
    assert e429.retryable is True

    # 429 Quota exhausted -> QUOTA_EXHAUSTED, retryable=False
    equota = normalize_http_error(429, "You exceeded your current quota", p_id, m_name)
    assert equota.category == ProviderErrorCategory.QUOTA_EXHAUSTED
    assert equota.retryable is False

    # 401 Authentication -> AUTHENTICATION_FAILURE, retryable=False
    e401 = normalize_http_error(401, "Invalid API key", p_id, m_name)
    assert e401.category == ProviderErrorCategory.AUTHENTICATION_FAILURE
    assert e401.retryable is False

    # 400 Context Length -> CONTEXT_LENGTH_EXCEEDED, retryable=False
    econtext = normalize_http_error(400, "maximum context length is 128000 tokens", p_id, m_name)
    assert econtext.category == ProviderErrorCategory.CONTEXT_LENGTH_EXCEEDED
    assert econtext.retryable is False

    # 400 Safety / Policy -> CONTENT_POLICY_VIOLATION, retryable=False
    epolicy = normalize_http_error(400, "safety content policy violation", p_id, m_name)
    assert epolicy.category == ProviderErrorCategory.CONTENT_POLICY_VIOLATION
    assert epolicy.retryable is False

    # 503 Provider unavailable -> PROVIDER_UNAVAILABLE, retryable=True
    e503 = normalize_http_error(503, "Service unavailable", p_id, m_name)
    assert e503.category == ProviderErrorCategory.PROVIDER_UNAVAILABLE
    assert e503.retryable is True
