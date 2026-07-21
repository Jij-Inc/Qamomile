"""Resource-estimation tests for opaque oracle bindings."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc

_COSTED_ORACLE = qmc.opaque(
    "costed_oracle",
    num_qubits=1,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(total=7),
        calls=qmc.CallResources(queries_by_name={"costed_oracle": 1}),
    ),
)


@qmc.qkernel
def _one_gate_implementation(q: qmc.Qubit) -> qmc.Qubit:
    """Implement the oracle with one logical gate."""
    return qmc.h(q)


@qmc.qkernel
def _static_implementation(
    q: qmc.Qubit,
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Qubit:
    """Use an LCU descriptor through a compile-time static binding."""
    return qmc.rx(q, encoding.normalization)


@qmc.qkernel
def _helper(q: qmc.Qubit) -> qmc.Qubit:
    """Invoke the costed oracle from a nested body."""
    (q,) = _COSTED_ORACLE(q)
    return q


@qmc.qkernel
def _algorithm() -> qmc.Qubit:
    """Invoke the costed oracle once."""
    (q,) = _COSTED_ORACLE(qmc.qubit("q"))
    return q


@qmc.qkernel
def _nested_algorithm() -> qmc.Qubit:
    """Invoke the costed oracle through a helper."""
    return _helper(qmc.qubit("q"))


def test_unbound_estimate_uses_cost_and_bound_estimate_uses_body() -> None:
    """A binding replaces the model cost with implementation resources."""
    unbound = _algorithm.estimate_resources()
    bound = _algorithm.estimate_resources(
        oracle_bindings={"costed_oracle": _one_gate_implementation}
    )
    unbound_again = _algorithm.estimate_resources()

    assert unbound.gates.total == 7
    assert unbound.calls.queries_by_name == {"costed_oracle": 1}
    assert bound.gates.total == 1
    assert bound.calls.queries_by_name == {}
    assert unbound_again.gates.total == 7
    assert unbound_again.calls.queries_by_name == {"costed_oracle": 1}


def test_all_estimator_entrypoints_forward_oracle_bindings() -> None:
    """The free facade and estimator instance share binding semantics."""
    bindings = {"costed_oracle": _one_gate_implementation}

    assert (
        qmc.estimate_resources(
            _algorithm,
            oracle_bindings=bindings,
        ).gates.total
        == 1
    )
    assert (
        qmc.ResourceEstimator()
        .estimate(
            _algorithm,
            oracle_bindings=bindings,
        )
        .gates.total
        == 1
    )


def test_estimator_binding_reaches_nested_definition() -> None:
    """Estimator substitution traverses qkernel helper definitions."""
    estimate = _nested_algorithm.estimate_resources(
        oracle_bindings={"costed_oracle": _one_gate_implementation}
    )

    assert estimate.gates.total == 1


def test_estimator_rejects_bindings_for_raw_operation_sequence() -> None:
    """Raw operations lack callable-definition context for strict binding."""
    with pytest.raises(TypeError, match="raw operation sequences"):
        qmc.ResourceEstimator().estimate(
            _algorithm.block.operations,
            oracle_bindings={"costed_oracle": _one_gate_implementation},
        )


def test_estimator_validates_empty_non_mapping_bindings() -> None:
    """An empty non-mapping is not silently treated as no bindings."""
    with pytest.raises(TypeError, match="must be a mapping"):
        qmc.ResourceEstimator().estimate(
            _algorithm,
            oracle_bindings=[],  # type: ignore[arg-type]
        )


def test_estimator_rejects_unresolved_static_oracle_implementation() -> None:
    """Estimator rejects an implementation requiring static specialization."""
    with pytest.raises(ValueError, match="unresolved static bindings.*encoding"):
        qmc.ResourceEstimator().estimate(
            _algorithm,
            oracle_bindings={"costed_oracle": _static_implementation},
        )


def test_estimator_accepts_empty_bindings_for_raw_operations() -> None:
    """An empty mapping remains a no-op for every accepted input form."""
    estimate = qmc.ResourceEstimator().estimate(
        _algorithm.block.operations,
        oracle_bindings={},
    )

    assert estimate.gates.total == 7
