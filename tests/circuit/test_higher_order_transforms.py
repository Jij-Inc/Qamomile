"""Tests for direct composition of qkernel control and inverse wrappers."""

from __future__ import annotations

from typing import Any

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.frontend.operation.inverse import InverseGate
from qamomile.circuit.frontend.qkernel_callable import (
    qkernel_callable_attrs,
    qkernel_callable_ref,
)
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.callable import CallableRef, InvokeOperation
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.serialization import deserialize, serialize


def test_control_of_inverse_qkernel_materializes_inverse_target_body() -> None:
    """control(inverse(qkernel)) controls the reversed dagger gate sequence."""

    @qmc.qkernel
    def layer(target: qmc.Qubit) -> qmc.Qubit:
        """Apply a non-self-inverse two-gate layer."""
        target = qmc.h(target)
        return qmc.s(target)

    controlled_inverse = qmc.control(qmc.inverse(layer))

    @qmc.qkernel
    def circuit(
        control: qmc.Qubit,
        target: qmc.Qubit,
    ) -> tuple[qmc.Qubit, qmc.Qubit]:
        """Apply the directly composed controlled inverse."""
        return controlled_inverse(control, target)

    [operation] = [
        operation
        for operation in circuit.block.operations
        if isinstance(operation, ControlledUOperation)
    ]
    gate_names = [
        nested.gate_type.name
        for nested in operation.block.operations
        if isinstance(nested, GateOperation)
    ]

    assert gate_names == ["SDG", "H"]


def test_inverse_of_controlled_qkernel_matches_control_of_inverse() -> None:
    """inverse(control(qkernel)) has the same resources as the opposite order."""

    @qmc.qkernel
    def layer(target: qmc.Qubit) -> qmc.Qubit:
        """Apply one non-self-inverse gate."""
        return qmc.t(target)

    control_then_inverse = qmc.inverse(qmc.control(layer, num_controls=2))
    inverse_then_control = qmc.control(qmc.inverse(layer), num_controls=2)

    @qmc.qkernel
    def left() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
        """Apply inverse(control(layer))."""
        return control_then_inverse(
            qmc.qubit("control_0"),
            qmc.qubit("control_1"),
            qmc.qubit("target"),
        )

    @qmc.qkernel
    def right() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
        """Apply control(inverse(layer))."""
        return inverse_then_control(
            qmc.qubit("control_0"),
            qmc.qubit("control_1"),
            qmc.qubit("target"),
        )

    assert left.estimate_resources().gates == right.estimate_resources().gates
    assert left.estimate_resources().depth == right.estimate_resources().depth


def test_control_of_inverse_rotation_matches_inverse_of_control() -> None:
    """Native rotations compose with control in either transform order."""
    angle = 0.375
    control_of_inverse = qmc.control(qmc.inverse(qmc.rx))
    inverse_of_control = qmc.inverse(qmc.control(qmc.rx))

    @qmc.qkernel
    def left() -> tuple[qmc.Qubit, qmc.Qubit]:
        """Apply a controlled inverse RX directly."""
        return control_of_inverse(
            qmc.qubit("control"),
            qmc.qubit("target"),
            angle,
        )

    @qmc.qkernel
    def right() -> tuple[qmc.Qubit, qmc.Qubit]:
        """Invert an ordinary controlled RX."""
        return inverse_of_control(
            qmc.qubit("control"),
            qmc.qubit("target"),
            angle,
        )

    [operation] = [
        candidate
        for candidate in left.block.operations
        if isinstance(candidate, ControlledUOperation)
    ]
    [rotation] = [
        candidate
        for candidate in operation.block.operations
        if isinstance(candidate, GateOperation)
    ]

    [negation] = [
        candidate
        for candidate in operation.block.operations
        if isinstance(candidate, BinOp)
    ]

    assert rotation.gate_type is GateOperationType.RX
    assert rotation.theta is not None
    assert rotation.theta.uuid == negation.results[0].uuid
    assert negation.kind is BinOpKind.MUL
    assert any(operand.get_const() == -1.0 for operand in negation.operands)
    assert left.estimate_resources().gates == right.estimate_resources().gates
    assert left.estimate_resources().depth == right.estimate_resources().depth


def test_nested_qkernel_controls_prepend_pattern_and_preserve_inverse() -> None:
    """Nested control flattens prefixes without losing target inversion."""

    @qmc.qkernel
    def layer(target: qmc.Qubit) -> qmc.Qubit:
        """Apply one non-self-inverse phase gate."""
        return qmc.s(target)

    inner = qmc.control(layer, num_controls=2, control_value=1)
    transformed = qmc.control(
        qmc.inverse(inner),
        num_controls=1,
        control_value=0,
    )

    @qmc.qkernel
    def circuit() -> tuple[
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Apply the flattened outer-then-inner control prefix."""
        return transformed(
            qmc.qubit("outer"),
            qmc.qubit("inner_0"),
            qmc.qubit("inner_1"),
            qmc.qubit("target"),
        )

    [operation] = [
        operation
        for operation in circuit.block.operations
        if isinstance(operation, ControlledUOperation)
    ]
    gate_names = [
        nested.gate_type.name
        for nested in operation.block.operations
        if isinstance(nested, GateOperation)
    ]

    assert operation.num_controls == 3
    assert operation.control_value == 2
    assert gate_names == ["SDG"]

    restored = deserialize(serialize(circuit))
    [restored_operation] = [
        candidate
        for candidate in restored.block.operations
        if isinstance(candidate, ControlledUOperation)
    ]
    restored_gate_names = [
        nested.gate_type.name
        for nested in restored_operation.block.operations
        if isinstance(nested, GateOperation)
    ]

    assert restored_operation.num_controls == 3
    assert restored_operation.control_value == 2
    assert restored_operation.callable_ref == operation.callable_ref
    assert restored_gate_names == ["SDG"]


def test_nested_symbolic_qkernel_controls_combine_all_ones_width() -> None:
    """Nested symbolic controls flatten to one resolvable control pool."""

    @qmc.qkernel
    def layer(target: qmc.Qubit) -> qmc.Qubit:
        """Apply one phase gate."""
        return qmc.s(target)

    @qmc.qkernel
    def circuit(width: qmc.UInt) -> qmc.Qubit:
        """Compose one concrete control ahead of a symbolic control group."""
        controls = qmc.qubit_array(width + 1, "controls")
        target = qmc.qubit("target")
        transformed = qmc.control(
            qmc.control(layer, num_controls=width),
            num_controls=1,
        )
        controls, target = transformed(controls, target)
        return target

    @qmc.qkernel
    def concrete() -> qmc.Qubit:
        """Apply the equivalent three-control operation directly."""
        controls = qmc.qubit_array(3, "controls")
        target = qmc.qubit("target")
        controls, target = qmc.control(layer, num_controls=3)(controls, target)
        return target

    symbolic = circuit.estimate_resources(inputs={"width": 2})
    expected = concrete.estimate_resources()

    assert symbolic.gates == expected.gates
    assert symbolic.depth == expected.depth
    assert symbolic.width.clean_ancilla_qubits == expected.width.clean_ancilla_qubits


def test_controlled_inverse_composite_materializes_structural_fallback() -> None:
    """A composite without an explicit transformed body controls its inverse."""

    @qmc.composite_gate(name="inverse_box")
    def boxed(target: qmc.Qubit) -> qmc.Qubit:
        """Apply one boxed phase gate."""
        return qmc.s(target)

    transformed = qmc.control(qmc.inverse(boxed), num_controls=2)

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
        """Apply the directly transformed composite."""
        return transformed(
            qmc.qubit("control_0"),
            qmc.qubit("control_1"),
            qmc.qubit("target"),
        )

    controlled_operations = [
        operation
        for operation in circuit.block.operations
        if isinstance(operation, ControlledUOperation)
    ]
    [operation] = controlled_operations
    gate_names = [
        nested.gate_type.name
        for nested in operation.block.operations
        if isinstance(nested, GateOperation)
    ]

    assert gate_names == ["SDG"]


def test_controlled_inverse_composite_round_trips_structurally() -> None:
    """Serialization preserves the directly composed inverse target body."""

    @qmc.composite_gate(name="serialized_inverse_box")
    def boxed(target: qmc.Qubit) -> qmc.Qubit:
        """Apply one boxed T gate."""
        return qmc.t(target)

    transformed = qmc.inverse(qmc.control(boxed))

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
        """Apply the transformed composite before serialization."""
        return transformed(
            qmc.qubit("control"),
            qmc.qubit("target"),
        )

    restored = deserialize(serialize(circuit))
    controlled_operations = [
        operation
        for operation in restored.block.operations
        if isinstance(operation, ControlledUOperation)
    ]
    [operation] = controlled_operations
    gate_names = [
        nested.gate_type.name
        for nested in operation.block.operations
        if isinstance(nested, GateOperation)
    ]

    assert gate_names == ["TDG"]


def test_nested_composite_controls_preserve_callable_metadata() -> None:
    """Nested composite control keeps identity on its structural fallback."""

    @qmc.composite_gate(name="nested_inverse_box")
    def boxed(target: qmc.Qubit) -> qmc.Qubit:
        """Apply one boxed phase gate."""
        return qmc.t(target)

    transformed = qmc.control(
        qmc.inverse(qmc.control(boxed, num_controls=2, control_value=1)),
        num_controls=1,
        control_value=0,
    )

    @qmc.qkernel
    def circuit() -> tuple[
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Apply the nested controlled-inverse composite."""
        return transformed(
            qmc.qubit("outer"),
            qmc.qubit("inner_0"),
            qmc.qubit("inner_1"),
            qmc.qubit("target"),
        )

    [operation] = [
        operation
        for operation in circuit.block.operations
        if isinstance(operation, ControlledUOperation)
    ]
    gate_names = [
        nested.gate_type.name
        for nested in operation.block.operations
        if isinstance(nested, GateOperation)
    ]

    assert operation.num_controls == 3
    assert operation.control_value == 2
    assert operation.callable_ref is not None
    assert operation.callable_ref.name == "nested_inverse_box"
    assert operation.callable_attrs["kind"] == "composite"
    assert gate_names == ["TDG"]


def test_controlled_inverse_composite_executes_phase_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """Every SDK preserves the dagger phase of a controlled inverse composite."""

    @qmc.composite_gate(name="emitted_inverse_box")
    def boxed(target: qmc.Qubit) -> qmc.Qubit:
        """Apply one boxed S gate."""
        return qmc.s(target)

    transformed = qmc.control(qmc.inverse(boxed))

    @qmc.qkernel
    def circuit(observable: qmc.Observable) -> qmc.Float:
        """Measure the control-Y phase induced by controlled S-dagger."""
        qubits = qmc.qubit_array(2, "qubits")
        qubits[0] = qmc.h(qubits[0])
        qubits[1] = qmc.x(qubits[1])
        qubits[0], qubits[1] = transformed(qubits[0], qubits[1])
        return qmc.expval(qubits, observable)

    executable = sdk_transpiler.transpiler.transpile(
        circuit,
        bindings={"observable": qm_o.Y(0)},
    )
    observed = float(executable.run(sdk_transpiler.transpiler.executor()).result())

    assert observed == pytest.approx(-1.0, abs=1e-6)


def test_double_inverse_preserves_effective_callable_metadata() -> None:
    """Cancelling inverse wrappers retains identity and resource contracts."""

    @qmc.qkernel
    def layer(target: qmc.Qubit) -> qmc.Qubit:
        """Apply one phase gate."""
        return qmc.s(target)

    callable_ref = CallableRef(
        namespace="test.inverse.metadata",
        name="contracted_layer",
    )
    callable_attrs = qkernel_callable_attrs(layer)
    callable_attrs["resource_contract"] = {
        "quantum_operand_widths": [
            {"index": 0, "name": "target", "width": 1},
        ],
    }
    restored = qmc.inverse(
        InverseGate(
            layer,
            callable_ref=callable_ref,
            callable_attrs=callable_attrs,
        )
    )

    assert restored is not layer
    assert qkernel_callable_ref(restored) == callable_ref
    assert qkernel_callable_attrs(restored) == callable_attrs

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        """Invoke the restored callable once."""
        return restored(qmc.qubit("target"))

    [operation] = [
        operation
        for operation in circuit.block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert operation.target == callable_ref
    assert operation.attrs == callable_attrs
