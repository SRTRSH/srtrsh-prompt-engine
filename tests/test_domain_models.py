"""Tests for domain models, aggregates, invariants, and lifecycle transitions."""

import pytest

from srtrsh_prompt_engine.domain.models import (
    ExecutionConfiguration,
    HashedVersionDefinition,
    Prompt,
    RuntimeOverridePolicy,
    SchemaContract,
    Template,
    VariableDefinition,
    VersionState,
)


def _make_sample_definition():
    return HashedVersionDefinition(
        template=Template(
            system_prompt="Test system",
            user_prompt="Test user {{var1}}",
        ),
        variables=[
            VariableDefinition(name="var1", type="STRING", required=True),
        ],
        schema_contract=SchemaContract(
            schema_type="JSON_SCHEMA",
            definition={"type": "object"},
        ),
        execution_configuration=ExecutionConfiguration(
            temperature=0.7,
            timeout_duration=5000,
        ),
        override_policy=RuntimeOverridePolicy(
            allowed_parameters={"temperature"},
        ),
    )


def test_prompt_lifecycle_transitions():
    prompt = Prompt(identity="test_prompt")
    assert prompt.active_version_reference is None

    # 1. Create version in DRAFT
    defn = _make_sample_definition()
    v1 = prompt.create_version(defn)
    assert v1.version_number == 1
    assert v1.state == VersionState.DRAFT
    assert v1.version_hash is None

    # Cannot activate a draft version
    with pytest.raises(ValueError, match="Only PUBLISHED versions can be active"):
        prompt.activate_version(1)

    # 2. Publish version
    v1_published = prompt.publish_version(1)
    assert v1_published.state == VersionState.PUBLISHED
    assert v1_published.version_hash is not None
    assert len(v1_published.version_hash) == 64

    # 3. Activate version
    prompt.activate_version(1)
    assert prompt.active_version_reference == 1
    assert prompt.get_active_version().version_number == 1

    # 4. Deprecate version
    prompt.deprecate_version(1)
    assert prompt.get_version(1).state == VersionState.DEPRECATED
    # Active reference cleared upon deprecation of active version
    assert prompt.active_version_reference is None

    # Cannot publish a deprecated version
    with pytest.raises(ValueError, match="Cannot publish a deprecated version"):
        v1_published.publish()


def test_required_variable_must_not_have_default():
    with pytest.raises(ValueError, match="must not have a defaultValue"):
        VariableDefinition(
            name="invalid_var",
            type="STRING",
            required=True,
            default_value="illegal_default",
        )


def test_version_number_monotonically_increases():
    prompt = Prompt(identity="multi_version_prompt")
    defn = _make_sample_definition()

    v1 = prompt.create_version(defn)
    v2 = prompt.create_version(defn)
    v3 = prompt.create_version(defn)

    assert v1.version_number == 1
    assert v2.version_number == 2
    assert v3.version_number == 3
