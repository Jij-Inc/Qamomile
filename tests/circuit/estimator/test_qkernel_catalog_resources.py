"""Validate resource estimates for every qkernel-catalog entry.

The companion case table intentionally contains unfilled expectations.  These
tests execute each route first and then skip only the comparison whose expected
value is still ``None``.  Adding an expectation therefore enables its test
without changing the test logic.
"""

from __future__ import annotations

import dataclasses
import functools
from collections import Counter
from collections.abc import Mapping
from typing import Any, NoReturn

import pytest
import sympy as sp

import qamomile.circuit as qmc
from tests.circuit.estimator._qkernel_catalog_resource_cases import (
    CATALOG_RESOURCE_CASES,
    CATALOG_RESOURCE_ERROR_CASES,
    CatalogEstimationRoute,
    CatalogResourceCase,
    CatalogResourceErrorCase,
    CatalogResourceSpecialization,
    ExpectedResourceEstimate,
)
from tests.circuit.qkernel_catalog import QKERNEL_BY_ID

_CASE_BY_ID = {case.id: case for case in CATALOG_RESOURCE_CASES}
_DEFAULT_CASES = tuple(
    case for case in CATALOG_RESOURCE_CASES if case.variant == "default"
)
_SPECIALIZATION_CASES = tuple(
    (case, specialization)
    for case in CATALOG_RESOURCE_CASES
    for specialization in case.specializations
)


def _estimate_symbolically(case: CatalogResourceCase) -> qmc.ResourceEstimate:
    """Estimate one catalog case while retaining resource parameters."""

    entry = QKERNEL_BY_ID[case.catalog_id]
    return entry.qkernel.estimate_resources(
        inputs=dict(case.symbolic_inputs),
        control_decomposition=case.options.control_decomposition,
        unknown_policy=case.options.unknown_policy,
        strategies=dict(case.options.strategies),
    )


@functools.cache
def _cached_symbolic_estimate(case_id: str) -> qmc.ResourceEstimate:
    """Return one symbolic estimate shared by symbolic and substitute tests."""

    return _estimate_symbolically(_CASE_BY_ID[case_id])


def _estimate_with_inputs(
    case: CatalogResourceCase,
    specialization: CatalogResourceSpecialization,
) -> qmc.ResourceEstimate:
    """Estimate one catalog case with direct input specialization."""

    entry = QKERNEL_BY_ID[case.catalog_id]
    inputs = dict(case.symbolic_inputs)
    inputs.update(specialization.inputs)
    return entry.qkernel.estimate_resources(
        inputs=inputs,
        control_decomposition=case.options.control_decomposition,
        unknown_policy=case.options.unknown_policy,
        strategies=dict(case.options.strategies),
    )


def _substitute_estimate(
    case: CatalogResourceCase,
    specialization: CatalogResourceSpecialization,
) -> qmc.ResourceEstimate:
    """Specialize one previously computed symbolic estimate."""

    return _cached_symbolic_estimate(case.id).substitute(
        **dict(specialization.substitutions)
    )


def _assert_expression_equal(actual: Any, expected: Any, path: str) -> None:
    """Assert structural or algebraic equality for one resource expression."""

    if actual == expected:
        return
    difference = sp.simplify(sp.sympify(actual) - sp.sympify(expected))
    assert difference == 0, f"{path}: expected {expected!r}, got {actual!r}"


def _assert_mapping_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    path: str,
) -> None:
    """Assert key and symbolic-value equality for one resource mapping."""

    assert set(actual) == set(expected), (
        f"{path}: expected keys {set(expected)!r}, got {set(actual)!r}"
    )
    for key, expected_value in expected.items():
        _assert_expression_equal(actual[key], expected_value, f"{path}.{key}")


def _assert_resource_record_equal(actual: Any, expected: Any, path: str) -> None:
    """Assert every public dataclass field of one resource record."""

    assert type(actual) is type(expected), (
        f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}"
    )
    for resource_field in dataclasses.fields(expected):
        field_path = f"{path}.{resource_field.name}"
        actual_value = getattr(actual, resource_field.name)
        expected_value = getattr(expected, resource_field.name)
        if isinstance(expected_value, Mapping):
            assert isinstance(actual_value, Mapping), (
                f"{field_path}: expected a mapping, got {type(actual_value).__name__}"
            )
            _assert_mapping_equal(actual_value, expected_value, field_path)
        else:
            _assert_expression_equal(actual_value, expected_value, field_path)


def _assert_estimate_equal(
    actual: qmc.ResourceEstimate,
    expected: ExpectedResourceEstimate,
) -> None:
    """Assert every user-facing resource result represented by the fixture."""

    _assert_resource_record_equal(actual.width, expected.width, "width")
    _assert_resource_record_equal(actual.gates, expected.gates, "gates")
    _assert_resource_record_equal(actual.depth, expected.depth, "depth")
    _assert_resource_record_equal(actual.calls, expected.calls, "calls")
    _assert_resource_record_equal(
        actual.measurements,
        expected.measurements,
        "measurements",
    )
    _assert_resource_record_equal(actual.resets, expected.resets, "resets")

    _assert_expression_equal(actual.qubits, expected.width.peak_qubits, "qubits")
    _assert_expression_equal(
        actual.circuit_qubits,
        expected.width.circuit_qubits,
        "circuit_qubits",
    )
    _assert_expression_equal(
        actual.width.circuit_qubits,
        expected.width.circuit_qubits,
        "width.circuit_qubits",
    )
    _assert_expression_equal(actual.gates.t_gates, expected.gates.t, "gates.t_gates")
    _assert_expression_equal(
        actual.gates.clifford_gates,
        expected.gates.clifford,
        "gates.clifford_gates",
    )
    _assert_expression_equal(
        actual.gates.rotation_gates,
        expected.gates.rotation,
        "gates.rotation_gates",
    )
    _assert_mapping_equal(
        actual.calls.oracle_calls,
        expected.calls.calls_by_name,
        "calls.oracle_calls",
    )
    _assert_mapping_equal(
        actual.calls.oracle_queries,
        expected.calls.queries_by_name,
        "calls.oracle_queries",
    )
    assert set(actual.parameters) == set(expected.parameters)
    actual_assumption_sources = Counter(
        assumption.source for assumption in actual.assumptions
    )
    expected_assumption_sources = Counter(expected.assumption_sources)
    assert actual_assumption_sources == expected_assumption_sources, (
        "assumption sources: expected "
        f"{expected_assumption_sources!r}, got {actual_assumption_sources!r}"
    )
    assert actual.derivation is expected.derivation
    assert actual.quality is expected.quality
    assert actual.approximation is expected.approximation
    assert actual.control_decomposition is expected.control_decomposition


def _resolve_symbolic_expectation(
    case: CatalogResourceCase,
    estimate: qmc.ResourceEstimate,
) -> ExpectedResourceEstimate | None:
    """Resolve a static or parameter-dependent symbolic expectation."""

    expectation = case.expected_symbolic
    if expectation is None or isinstance(expectation, ExpectedResourceEstimate):
        return expectation
    return expectation(estimate.parameters)


def _missing_expectation(route: str, case_id: str) -> NoReturn:
    """Skip one evaluated route until its expected result is supplied."""

    pytest.skip(f"Expected resources are not filled for {case_id} via {route}.")


def test_catalog_resource_case_identifiers_are_unique() -> None:
    """Every estimator-configuration variant must have one stable ID."""

    assert len(_CASE_BY_ID) == len(CATALOG_RESOURCE_CASES)
    specialization_ids = [
        f"{case.id}:{specialization.id}"
        for case, specialization in _SPECIALIZATION_CASES
    ]
    assert len(set(specialization_ids)) == len(specialization_ids)


def test_expected_fixture_covers_every_public_estimate_field() -> None:
    """Keep the expected-result fixture aligned with public result fields."""

    public_fields = {
        resource_field.name
        for resource_field in dataclasses.fields(qmc.ResourceEstimate)
        if not resource_field.name.startswith("_")
    }
    expected_fields = {
        resource_field.name
        for resource_field in dataclasses.fields(ExpectedResourceEstimate)
    }
    expected_fields.remove("assumption_sources")
    expected_fields.add("assumptions")
    assert public_fields == expected_fields | {"trace"}


def test_expected_fixture_accepts_a_zero_estimate() -> None:
    """Exercise the comparison helper before catalog expectations are filled."""

    _assert_estimate_equal(
        qmc.ResourceEstimate.zero(),
        ExpectedResourceEstimate(),
    )


def test_expected_fixture_compares_assumption_source_multisets() -> None:
    """Compare assumption source values and multiplicities without messages."""

    estimate = qmc.ResourceEstimate(
        assumptions=(
            qmc.ResourceAssumption(message="first loop detail", source="for"),
            qmc.ResourceAssumption(message="Oracle detail", source="oracle"),
            qmc.ResourceAssumption(message="second loop detail", source="for"),
        ),
    )
    _assert_estimate_equal(
        estimate,
        ExpectedResourceEstimate(assumption_sources=("oracle", "for", "for")),
    )

    with pytest.raises(AssertionError, match="assumption sources"):
        _assert_estimate_equal(
            estimate,
            ExpectedResourceEstimate(assumption_sources=("oracle", "for")),
        )
    with pytest.raises(AssertionError, match="assumption sources"):
        _assert_estimate_equal(
            estimate,
            ExpectedResourceEstimate(assumption_sources=("other", "for", "for")),
        )


def test_every_catalog_entry_has_one_default_resource_case() -> None:
    """Every shared catalog entry must have exactly one default resource case."""

    default_counts = Counter(case.catalog_id for case in _DEFAULT_CASES)
    assert set(default_counts) == set(QKERNEL_BY_ID)
    assert all(count == 1 for count in default_counts.values())
    assert {case.catalog_id for case in CATALOG_RESOURCE_CASES} <= set(QKERNEL_BY_ID)


def test_every_catalog_resource_case_exercises_all_estimation_routes() -> None:
    """Every case must cover symbolic, direct-input, and substitution routes."""

    assert all(case.specializations for case in CATALOG_RESOURCE_CASES)


@pytest.mark.parametrize(
    "case",
    CATALOG_RESOURCE_CASES,
    ids=[case.id for case in CATALOG_RESOURCE_CASES],
)
def test_catalog_symbolic_resources(case: CatalogResourceCase) -> None:
    """Evaluate and, once filled, assert every symbolic catalog estimate."""

    estimate = _cached_symbolic_estimate(case.id)
    assert estimate.control_decomposition is case.options.control_decomposition

    expected = _resolve_symbolic_expectation(case, estimate)
    if expected is None:
        _missing_expectation("symbolic estimation", case.id)
    _assert_estimate_equal(estimate, expected)


@pytest.mark.parametrize(
    "case,specialization",
    _SPECIALIZATION_CASES,
    ids=[
        f"{case.id}:{specialization.id}"
        for case, specialization in _SPECIALIZATION_CASES
    ],
)
def test_catalog_resources_with_inputs(
    case: CatalogResourceCase,
    specialization: CatalogResourceSpecialization,
) -> None:
    """Evaluate and, once filled, assert direct ``inputs=`` specialization."""

    estimate = _estimate_with_inputs(case, specialization)
    assert not estimate.parameters, (
        f"{case.id}:{specialization.id} leaves direct-input parameters "
        f"{set(estimate.parameters)!r}."
    )
    expected = specialization.expected_from_inputs
    if expected is None:
        _missing_expectation("inputs", f"{case.id}:{specialization.id}")
    _assert_estimate_equal(estimate, expected)


@pytest.mark.parametrize(
    "case,specialization",
    _SPECIALIZATION_CASES,
    ids=[
        f"{case.id}:{specialization.id}"
        for case, specialization in _SPECIALIZATION_CASES
    ],
)
def test_catalog_resources_after_substitution(
    case: CatalogResourceCase,
    specialization: CatalogResourceSpecialization,
) -> None:
    """Evaluate and, once filled, assert late symbolic substitution."""

    symbolic = _cached_symbolic_estimate(case.id)
    assert set(specialization.substitutions) == set(symbolic.parameters), (
        f"{case.id}:{specialization.id} substitutions must exactly name the "
        f"public parameters {set(symbolic.parameters)!r}."
    )
    estimate = _substitute_estimate(case, specialization)
    assert not estimate.parameters, (
        f"{case.id}:{specialization.id} leaves substituted parameters "
        f"{set(estimate.parameters)!r}."
    )
    expected = specialization.expected_from_substitution
    if expected is None:
        _missing_expectation("substitute", f"{case.id}:{specialization.id}")
    _assert_estimate_equal(estimate, expected)


def _run_error_case(case: CatalogResourceErrorCase) -> None:
    """Execute the route selected by one expected-error case."""

    resource_case = CatalogResourceCase(
        catalog_id=case.catalog_id,
        variant=f"error-{case.id}",
        options=case.options,
        symbolic_inputs=case.symbolic_inputs,
    )
    if case.route is CatalogEstimationRoute.SYMBOLIC:
        _estimate_symbolically(resource_case)
        return
    if case.route is CatalogEstimationRoute.INPUTS:
        specialization = CatalogResourceSpecialization(
            id=case.id,
            inputs=case.inputs,
            substitutions=case.substitutions,
        )
        _estimate_with_inputs(resource_case, specialization)
        return
    symbolic = _estimate_symbolically(resource_case)
    symbolic.substitute(**dict(case.substitutions))


@pytest.mark.parametrize(
    "case",
    CATALOG_RESOURCE_ERROR_CASES,
    ids=[case.id for case in CATALOG_RESOURCE_ERROR_CASES],
)
def test_catalog_resource_errors(case: CatalogResourceErrorCase) -> None:
    """Assert explicitly modeled errors without mixing them into result cases."""

    with pytest.raises(case.exception_type, match=case.match):
        _run_error_case(case)
