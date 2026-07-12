"""Tests for the single QKernel-backed composite callable model."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.ir.operation.callable import (
    CallPolicy,
    CallTransform,
    CompositeGateType,
    InvokeOperation,
)


@qmc.composite_gate(name="bell_pair")
def bell_pair(
    left: qmc.Qubit,
    right: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Prepare a Bell pair."""
    left = qmc.h(left)
    return qmc.cx(left, right)


@qmc.qkernel
def use_bell_pair() -> tuple[qmc.Bit, qmc.Bit]:
    """Call the reusable Bell composite."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left, right = bell_pair(left, right)
    return qmc.measure(left), qmc.measure(right)


def _invokes(kernel: QKernel) -> list[InvokeOperation]:
    """Collect top-level named invocations from a kernel.

    Args:
        kernel (QKernel): Kernel to build.

    Returns:
        list[InvokeOperation]: Top-level invocation operations.
    """
    return [op for op in kernel.block.operations if isinstance(op, InvokeOperation)]


def test_composite_decorator_returns_qkernel() -> None:
    """A composite is the same frontend object kind as every qkernel."""
    assert isinstance(bell_pair, QKernel)
    assert bell_pair._callable_kind == "composite"
    assert bell_pair._callable_name == "bell_pair"
    assert bell_pair._callable_policy is CallPolicy.PRESERVE_BOX


def test_composite_accepts_an_existing_qkernel_without_wrapper() -> None:
    """Stacked decorators preserve the original QKernel identity."""

    @qmc.qkernel
    def phase(q: qmc.Qubit) -> qmc.Qubit:
        """Apply one phase gate."""
        return qmc.z(q)

    decorated = qmc.composite_gate(name="phase_box")(phase)

    assert decorated is phase
    assert isinstance(decorated, QKernel)


def test_composite_call_stays_named_with_body() -> None:
    """Calling a composite emits one named invocation with a fallback body."""
    [invoke] = _invokes(use_bell_pair)

    assert invoke.custom_name == "bell_pair"
    assert invoke.target.namespace == "user.composite"
    assert invoke.transform is CallTransform.DIRECT
    assert invoke.effective_body() is bell_pair.block


def test_composite_does_not_have_a_second_controls_call_protocol() -> None:
    """Control is expressed through qmc.control, not a controls keyword."""

    @qmc.composite_gate
    def one_h(target: qmc.Qubit) -> qmc.Qubit:
        """Apply one H gate."""
        return qmc.h(target)

    @qmc.qkernel
    def invalid() -> qmc.Qubit:
        """Attempt the removed legacy call shape."""
        control = qmc.qubit("control")
        target = qmc.qubit("target")
        return one_h(target, controls=(control,))

    with pytest.raises(TypeError, match="unexpected keyword argument 'controls'"):
        _ = invalid.block


def test_qmc_control_accepts_a_composite_qkernel() -> None:
    """A higher-order control consumes a composite through the QKernel API."""
    controlled_bell = qmc.control(bell_pair)

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
        """Control the complete Bell composite."""
        control = qmc.qubit("control")
        left = qmc.qubit("left")
        right = qmc.qubit("right")
        return controlled_bell(control, left, right)

    [controlled] = _invokes(circuit)
    assert controlled.target.name == "bell_pair"
    assert controlled.attrs["kind"] == "composite"
    assert controlled.transform is CallTransform.CONTROLLED
    assert controlled.num_control_qubits == 1
    assert controlled.definition is not None
    assert controlled.definition.body is bell_pair.block


def test_qmc_control_accepts_vector_composite_targets() -> None:
    """Vector-shaped composite ports work through the normal control transform."""
    controlled_qft = qmc.control(qmc.qft)

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
        """Control QFT on a symbolic-width target vector."""
        control = qmc.qubit("control")
        targets = qmc.qubit_array(n, "targets")
        return controlled_qft(control, targets)

    [controlled] = _invokes(circuit)
    assert controlled.attrs["gate_type"] == CompositeGateType.QFT.name
    assert controlled.transform is CallTransform.CONTROLLED
    assert controlled.num_control_qubits == 1


def test_interleaved_composite_operands_classify_by_type() -> None:
    """Quantum targets remain visible after an interleaved parameter."""

    @qmc.composite_gate(name="interleaved_box")
    def interleaved_box(
        first: qmc.Qubit,
        theta: qmc.Float,
        second: qmc.Qubit,
    ) -> tuple[qmc.Qubit, qmc.Qubit]:
        """Rotate the first qubit and flip the second qubit."""
        first = qmc.ry(first, theta)
        second = qmc.x(second)
        return first, second

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> tuple[qmc.Qubit, qmc.Qubit]:
        """Invoke a composite whose Python signature interleaves operand kinds."""
        first = qmc.qubit("first")
        second = qmc.qubit("second")
        return interleaved_box(first, theta, second)

    [invoke] = _invokes(circuit)

    assert [value.name for value in invoke.target_qubits] == ["first", "second"]
    assert [value.name for value in invoke.parameters] == ["theta"]


def test_inverse_accepts_a_composite_qkernel() -> None:
    """Inverse preserves the same callable identity and uses a transform."""
    inverse_bell = qmc.inverse(bell_pair)

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
        """Apply the inverse Bell composite."""
        left = qmc.qubit("left")
        right = qmc.qubit("right")
        return inverse_bell(left, right)

    [inverse] = _invokes(circuit)
    assert inverse.target.name == "bell_pair"
    assert inverse.attrs["kind"] == "composite"
    assert inverse.transform is CallTransform.INVERSE
    assert inverse.definition is not None
    assert inverse.definition.body is bell_pair.block
    assert inverse.implementation_for() is not None


def test_composite_resource_estimate_is_derived_from_its_body() -> None:
    """A body-backed composite has no second resource-definition surface."""

    @qmc.composite_gate
    def two_gates(q: qmc.Qubit) -> qmc.Qubit:
        """Apply one H gate."""
        return qmc.x(qmc.h(q))

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        """Call the body-derived composite."""
        return two_gates(qmc.qubit("q"))

    assert circuit.estimate_resources().gates.total == 2


def test_symbolic_composite_transforms_remain_estimable() -> None:
    """Direct, inverse, and controlled calls share one symbolic definition."""

    @qmc.composite_gate
    def repeated_h(q: qmc.Qubit, rounds: qmc.UInt) -> qmc.Qubit:
        """Apply a symbolic number of Hadamards."""
        for _ in qmc.range(rounds):
            q = qmc.h(q)
        return q

    @qmc.qkernel
    def algorithm(rounds: qmc.UInt) -> tuple[qmc.Qubit, qmc.Qubit]:
        """Compose every supported transform of one composite."""
        control = qmc.qubit("control")
        target = qmc.qubit("target")
        target = repeated_h(target, rounds)
        target = qmc.inverse(repeated_h)(target, rounds)
        control, target = qmc.control(repeated_h)(control, target, rounds)
        return control, target

    estimate = algorithm.estimate_resources()
    concrete = algorithm.estimate_resources(inputs={"rounds": 3})
    invokes = _invokes(algorithm)

    assert str(estimate.gates.total) == "3*rounds"
    assert concrete.gates.total == 9
    assert [operation.transform for operation in invokes] == [
        CallTransform.DIRECT,
        CallTransform.INVERSE,
        CallTransform.CONTROLLED,
    ]
    assert len({operation.target for operation in invokes}) == 1


def test_custom_composite_transpiles_through_its_body() -> None:
    """A backend without a native emitter lowers the embedded qkernel body."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    executable = QiskitTranspiler().transpile(use_bell_pair)

    assert executable.quantum_circuit.num_qubits == 2
    decomposed = executable.quantum_circuit.decompose()
    assert decomposed.count_ops()["h"] == 1
    assert decomposed.count_ops()["cx"] == 1
