"""Resource-estimation tests for opaque oracle bindings."""

from __future__ import annotations

import dataclasses

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.errors import ValidationError

_COSTED_ORACLE = qmc.opaque(
    "costed_oracle",
    num_qubits=1,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(total=7),
        calls=qmc.CallResources(queries_by_name={"costed_oracle": 1}),
    ),
)
_SIGNATURED_COSTED_ORACLE = qmc.opaque(
    "signatured_costed_oracle",
    signature=qmc.CallableSignature(
        inputs=[qmc.Vector[qmc.Qubit]],
        outputs=[qmc.Vector[qmc.Qubit]],
    ),
    cost=qmc.ResourceEstimate(gates=qmc.GateResources(total=7)),
)
_QPE_COSTED_ORACLE = qmc.opaque(
    "qpe_costed_oracle",
    num_qubits=1,
    cost=qmc.ResourceEstimate(
        calls=qmc.CallResources(queries_by_name={"qpe_costed_oracle": 1}),
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
def _vector_implementation(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Implement a vector oracle with one logical gate."""
    qubits[0] = qmc.h(qubits[0])
    return qubits


@qmc.qkernel
def _qpe_oracle_implementation(q: qmc.Qubit) -> qmc.Qubit:
    """Implement the resource-test phase oracle with one Z gate."""
    return qmc.z(q)


@qmc.qkernel
def _qpe_algorithm() -> qmc.Float:
    """Run three-bit QPE with a bodyless, costed oracle."""
    counting = qmc.qubit_array(3, "counting")
    target = qmc.x(qmc.qubit("target"))
    phase = qmc.qpe(target, counting, _QPE_COSTED_ORACLE)
    return qmc.measure(phase)


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


@qmc.qkernel
def _signatured_algorithm() -> qmc.Vector[qmc.Qubit]:
    """Invoke an oracle carrying an explicit vector signature."""
    return _SIGNATURED_COSTED_ORACLE(qmc.qubit_array(2, "qubits"))


def _mismatched_signature_block() -> Block:
    """Return a block whose declared oracle result disagrees with its call."""
    source = _signatured_algorithm.block
    declared = qmc.CallableSignature(
        inputs=[qmc.Vector[qmc.Qubit]],
        outputs=[qmc.Vector[qmc.Bit]],
    ).to_ir_signature()
    matched = False
    operations = []
    for operation in source.operations:
        if (
            isinstance(operation, InvokeOperation)
            and operation.target.name == "signatured_costed_oracle"
        ):
            assert operation.definition is not None
            matched = True
            operation = dataclasses.replace(
                operation,
                definition=dataclasses.replace(
                    operation.definition,
                    signature=declared,
                ),
            )
        operations.append(operation)
    assert matched
    return dataclasses.replace(source, operations=operations)


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


def test_estimator_substitutes_explicit_signature_oracle() -> None:
    """Estimator validates and expands an explicitly signed oracle."""
    unbound = _signatured_algorithm.estimate_resources()
    bound = _signatured_algorithm.estimate_resources(
        oracle_bindings={"signatured_costed_oracle": _vector_implementation}
    )

    assert unbound.gates.total == 7
    assert bound.gates.total == 1
    with pytest.raises(
        ValidationError,
        match=r"Input shape mismatch.*source is an array, target is a scalar",
    ):
        _signatured_algorithm.estimate_resources(
            oracle_bindings={"signatured_costed_oracle": _one_gate_implementation}
        )


def test_estimator_rejects_declared_signature_callsite_mismatch() -> None:
    """Estimator rejects an oracle call that violates its declaration."""
    with pytest.raises(
        ValidationError,
        match=r"declared callable signature disagrees.*Return type mismatch",
    ):
        qmc.ResourceEstimator().estimate(
            _mismatched_signature_block(),
            oracle_bindings={"signatured_costed_oracle": _vector_implementation},
        )


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


def test_qpe_oracle_estimation_survives_serialization_and_binding() -> None:
    """QPE powers produce seven queries before and after load or substitution."""
    payload = serialize(_qpe_algorithm)
    restored = deserialize(payload)

    original = _qpe_algorithm.estimate_resources()
    unbound = qmc.estimate_resources(restored)
    bound = qmc.estimate_resources(
        restored,
        oracle_bindings={"qpe_costed_oracle": _qpe_oracle_implementation},
    )

    expected_queries = {"qpe_costed_oracle": 7}
    assert original.calls.queries_by_name == expected_queries
    assert unbound.calls.queries_by_name == expected_queries
    assert bound.calls.queries_by_name == {}
    assert bound.gates.total == unbound.gates.total + 7
    assert bound.gates.two_qubit == unbound.gates.two_qubit + 7
