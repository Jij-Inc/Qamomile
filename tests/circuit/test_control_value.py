"""Execution and IR coverage for value-activated coherent controls."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.frontend.handle import (
    Bit,
    Float,
    Observable,
    Qubit,
    UInt,
    Vector,
)
from qamomile.circuit.ir.canonical import content_hash
from qamomile.circuit.ir.operation.callable import (
    CallTransform,
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.gate import ConcreteControlledU
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.passes.inline import InlinePass


@qmc.qkernel
def _patterned_x_layer(
    control_0: Qubit,
    control_1: Qubit,
    target: Qubit,
) -> tuple[Qubit, Qubit, Qubit]:
    """Apply X when the two controls hold the LSB-first value two.

    Args:
        control_0 (Qubit): Least-significant control qubit.
        control_1 (Qubit): Most-significant control qubit.
        target (Qubit): Target qubit.

    Returns:
        tuple[Qubit, Qubit, Qubit]: Updated controls and target.
    """
    return qmc.control(qmc.x, num_controls=2, control_value=2)(
        control_0,
        control_1,
        target,
    )


@qmc.qkernel
def _patterned_ry_layer(
    control_0: Qubit,
    control_1: Qubit,
    target: Qubit,
    angle: Float,
) -> tuple[Qubit, Qubit, Qubit]:
    """Apply RY when the two controls hold the LSB-first value two.

    Args:
        control_0 (Qubit): Least-significant control qubit.
        control_1 (Qubit): Most-significant control qubit.
        target (Qubit): Target qubit.
        angle (Float): Rotation angle.

    Returns:
        tuple[Qubit, Qubit, Qubit]: Updated controls and target.
    """
    return qmc.control(qmc.ry, num_controls=2, control_value=2)(
        control_0,
        control_1,
        target,
        angle,
    )


@qmc.composite_gate(name="control_value_test_x")
def _composite_x(target: Qubit) -> Qubit:
    """Apply a boxed X gate.

    Args:
        target (Qubit): Target qubit.

    Returns:
        Qubit: Updated target qubit.
    """
    return qmc.x(target)


@qmc.qkernel
def _patterned_composite_layer(
    control_0: Qubit,
    control_1: Qubit,
    target: Qubit,
) -> tuple[Qubit, Qubit, Qubit]:
    """Control a boxed X when the register value is two.

    Args:
        control_0 (Qubit): Least-significant control qubit.
        control_1 (Qubit): Most-significant control qubit.
        target (Qubit): Target qubit.

    Returns:
        tuple[Qubit, Qubit, Qubit]: Updated controls and target.
    """
    return qmc.control(
        _composite_x,
        num_controls=2,
        control_value=2,
    )(control_0, control_1, target)


@qmc.qkernel
def _inverse_patterned_composite_layer(
    control_0: Qubit,
    control_1: Qubit,
    target: Qubit,
) -> tuple[Qubit, Qubit, Qubit]:
    """Invert a composite activated by control-register value two.

    Args:
        control_0 (Qubit): Least-significant control qubit.
        control_1 (Qubit): Most-significant control qubit.
        target (Qubit): Target qubit.

    Returns:
        tuple[Qubit, Qubit, Qubit]: Updated controls and target.
    """
    return qmc.inverse(_patterned_composite_layer)(
        control_0,
        control_1,
        target,
    )


@qmc.qkernel
def _patterned_qft_layer(
    control_0: Qubit,
    control_1: Qubit,
    targets: Vector[Qubit],
) -> tuple[Qubit, Qubit, Vector[Qubit]]:
    """Control QFT when the register value is two.

    Args:
        control_0 (Qubit): Least-significant control qubit.
        control_1 (Qubit): Most-significant control qubit.
        targets (Vector[Qubit]): QFT target register.

    Returns:
        tuple[Qubit, Qubit, Vector[Qubit]]: Updated controls and targets.
    """
    return qmc.control(
        qmc.qft,
        num_controls=2,
        control_value=2,
    )(control_0, control_1, targets)


@qmc.qkernel
def _identity(target: Qubit) -> Qubit:
    """Return a target unchanged.

    Args:
        target (Qubit): Target qubit.

    Returns:
        Qubit: Unchanged target qubit.
    """
    return target


@qmc.qkernel
def _phased_identity(target: Qubit, angle: Float) -> Qubit:
    """Apply a global phase to an identity body.

    Args:
        target (Qubit): Target qubit.
        angle (Float): Phase angle.

    Returns:
        Qubit: Updated target qubit.
    """
    return qmc.global_phase(_identity, angle)(target)


def _executor(case: Any) -> Any:
    """Return an executor for a cross-backend test case.

    Args:
        case (Any): Backend fixture containing a transpiler and backend name.

    Returns:
        Any: Executor for the selected SDK backend.
    """
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _sample_outcomes(case: Any, kernel: Any) -> set[Any]:
    """Transpile and sample a deterministic test kernel.

    Args:
        case (Any): Backend fixture containing a transpiler.
        kernel (Any): Qkernel to transpile and execute.

    Returns:
        set[Any]: Distinct sampled outcomes.
    """
    executable = case.transpiler.transpile(kernel)
    result = executable.sample(_executor(case), shots=64).result()
    return {bits for bits, _ in result.results}


@pytest.mark.parametrize("control_state", [0, 1, 2, 3])
def test_control_value_truth_table_is_lsb_first(
    sdk_transpiler: Any,
    control_state: int,
) -> None:
    """Value two activates exactly the control pattern ``(0, 1)``.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
        control_state (int): Basis state prepared on the two controls.
    """
    bit_0 = control_state & 1
    bit_1 = (control_state >> 1) & 1

    @qmc.qkernel
    def circuit() -> Bit:
        """Build one basis-state truth-table row.

        Returns:
            Bit: Measured target bit.
        """
        controls = qmc.qubit_array(2, "controls")
        if bit_0:
            controls[0] = qmc.x(controls[0])
        if bit_1:
            controls[1] = qmc.x(controls[1])
        target = qmc.qubit("target")
        controls[0], controls[1], target = qmc.control(
            qmc.x,
            num_controls=2,
            control_value=2,
        )(controls[0], controls[1], target)
        return qmc.measure(target)

    assert _sample_outcomes(sdk_transpiler, circuit) == {int(control_state == 2)}


def test_control_value_accepts_a_whole_vector(sdk_transpiler: Any) -> None:
    """A whole control Vector uses element zero as integer bit zero.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
    """

    @qmc.qkernel
    def circuit() -> Bit:
        """Build a matching whole-Vector control call.

        Returns:
            Bit: Measured target bit.
        """
        controls = qmc.qubit_array(2, "controls")
        controls[1] = qmc.x(controls[1])
        target = qmc.qubit("target")
        controls, target = qmc.control(
            qmc.x,
            num_controls=2,
            control_value=2,
        )(controls, target)
        return qmc.measure(target)

    assert _sample_outcomes(sdk_transpiler, circuit) == {1}


def test_control_value_zero_activates_an_all_zero_register(
    sdk_transpiler: Any,
) -> None:
    """Value zero brackets every control and activates on all zeros.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
    """

    @qmc.qkernel
    def circuit() -> Bit:
        """Build an all-zero two-control call.

        Returns:
            Bit: Measured target bit.
        """
        controls = qmc.qubit_array(2, "controls")
        target = qmc.qubit("target")
        controls, target = qmc.control(
            qmc.x,
            num_controls=2,
            control_value=0,
        )(controls, target)
        return qmc.measure(target)

    assert _sample_outcomes(sdk_transpiler, circuit) == {1}


def test_control_value_preserves_composite_identity_and_executes(
    sdk_transpiler: Any,
) -> None:
    """Patterned composite control remains an InvokeOperation and runs.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
    """

    @qmc.qkernel
    def circuit() -> Bit:
        """Build a matching controlled composite invocation.

        Returns:
            Bit: Measured target bit.
        """
        controls = qmc.qubit_array(2, "controls")
        controls[1] = qmc.x(controls[1])
        target = qmc.qubit("target")
        controls[0], controls[1], target = qmc.control(
            _composite_x,
            num_controls=2,
            control_value=2,
        )(controls[0], controls[1], target)
        return qmc.measure(target)

    [invoke] = [
        operation
        for operation in circuit.block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert invoke.control_value == 2
    assert invoke.target.name == "control_value_test_x"
    assert _sample_outcomes(sdk_transpiler, circuit) == {1}


@pytest.mark.parametrize("outer_state", [0, 1])
def test_control_value_composes_with_an_outer_control(
    sdk_transpiler: Any,
    outer_state: int,
) -> None:
    """Unconditional X brackets cancel when an outer control is inactive.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
        outer_state (int): Basis state prepared on the outer control.
    """

    @qmc.qkernel
    def circuit() -> Bit:
        """Build nested ordinary and value-activated controls.

        Returns:
            Bit: Measured target bit.
        """
        outer = qmc.qubit("outer")
        if outer_state:
            outer = qmc.x(outer)
        inner = qmc.qubit_array(2, "inner")
        inner[1] = qmc.x(inner[1])
        target = qmc.qubit("target")
        outer, inner[0], inner[1], target = qmc.control(_patterned_x_layer)(
            outer,
            inner[0],
            inner[1],
            target,
        )
        return qmc.measure(target)

    assert _sample_outcomes(sdk_transpiler, circuit) == {outer_state}


def test_control_value_inside_static_for_and_if(sdk_transpiler: Any) -> None:
    """A patterned control survives nested static loop and branch lowering.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
    """

    @qmc.qkernel
    def circuit() -> Bit:
        """Apply one matching controlled X in the second loop iteration.

        Returns:
            Bit: Measured target bit.
        """
        controls = qmc.qubit_array(2, "controls")
        controls[1] = qmc.x(controls[1])
        target = qmc.qubit("target")
        for iteration in qmc.range(2):
            if iteration == 1:
                controls[0], controls[1], target = qmc.control(
                    qmc.x,
                    num_controls=2,
                    control_value=2,
                )(controls[0], controls[1], target)
        return qmc.measure(target)

    assert _sample_outcomes(sdk_transpiler, circuit) == {1}


def test_inverse_preserves_control_value(sdk_transpiler: Any) -> None:
    """A patterned controlled layer followed by its inverse is identity.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
    """

    @qmc.qkernel
    def circuit() -> Bit:
        """Build a forward/inverse patterned-control round trip.

        Returns:
            Bit: Measured target bit.
        """
        controls = qmc.qubit_array(2, "controls")
        controls[1] = qmc.x(controls[1])
        target = qmc.qubit("target")
        controls[0], controls[1], target = _patterned_ry_layer(
            controls[0],
            controls[1],
            target,
            0.37,
        )
        controls[0], controls[1], target = qmc.inverse(_patterned_ry_layer)(
            controls[0],
            controls[1],
            target,
            0.37,
        )
        return qmc.measure(target)

    assert _sample_outcomes(sdk_transpiler, circuit) == {0}


def test_inverse_of_patterned_composite_preserves_control_value(
    sdk_transpiler: Any,
) -> None:
    """InverseBlockOperation retains a composite control activation value.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
    """

    @qmc.qkernel
    def circuit() -> Bit:
        """Apply the inverse of a matching patterned composite X.

        Returns:
            Bit: Measured target bit.
        """
        controls = qmc.qubit_array(2, "controls")
        controls[1] = qmc.x(controls[1])
        target = qmc.qubit("target")
        controls[0], controls[1], target = qmc.inverse(_patterned_composite_layer)(
            controls[0],
            controls[1],
            target,
        )
        return qmc.measure(target)

    assert _sample_outcomes(sdk_transpiler, circuit) == {1}


@pytest.mark.parametrize("outer_state", [0, 1])
def test_inverse_control_value_composes_with_an_outer_control(
    sdk_transpiler: Any,
    outer_state: int,
) -> None:
    """Only an inverse block's own controls use its activation value.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
        outer_state (int): Basis state prepared on the outer control.
    """

    @qmc.qkernel
    def circuit() -> Bit:
        """Control an inverse patterned composite with one outer qubit.

        Returns:
            Bit: Measured target bit.
        """
        outer = qmc.qubit("outer")
        if outer_state:
            outer = qmc.x(outer)
        inner = qmc.qubit_array(2, "inner")
        inner[1] = qmc.x(inner[1])
        target = qmc.qubit("target")
        outer, inner[0], inner[1], target = qmc.control(
            _inverse_patterned_composite_layer
        )(
            outer,
            inner[0],
            inner[1],
            target,
        )
        return qmc.measure(target)

    assert _sample_outcomes(sdk_transpiler, circuit) == {outer_state}


def test_control_value_keeps_global_phase_relative(sdk_transpiler: Any) -> None:
    """A phase on value two becomes the expected relative phase.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture.
    """
    theta = 0.43

    @qmc.qkernel
    def circuit(observable: Observable) -> Float:
        """Observe the phase between control-zero and control-one branches.

        Args:
            observable (Observable): Observable evaluated on the register.

        Returns:
            Float: Expectation value.
        """
        qubits = qmc.qubit_array(3, "qubits")
        qubits[0] = qmc.h(qubits[0])
        qubits[1] = qmc.x(qubits[1])
        qubits[0], qubits[1], qubits[2] = qmc.control(
            _phased_identity,
            num_controls=2,
            control_value=2,
        )(qubits[0], qubits[1], qubits[2], theta)
        return qmc.expval(qubits, observable)

    value = (
        sdk_transpiler.transpiler.transpile(
            circuit,
            bindings={"observable": qm_o.Y(0)},
        )
        .run(_executor(sdk_transpiler))
        .result()
    )
    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert np.isclose(value, -np.sin(theta), rtol=0.0, atol=tolerance)


@pytest.mark.parametrize(
    ("control_value", "error_type"),
    [
        (True, TypeError),
        (1.0, TypeError),
        (-1, ValueError),
        (4, ValueError),
    ],
)
def test_control_value_rejects_invalid_values(
    control_value: Any,
    error_type: type[Exception],
) -> None:
    """Invalid activation values fail at controlled-gate construction.

    Args:
        control_value (Any): Invalid activation value.
        error_type (type[Exception]): Expected validation error.
    """
    with pytest.raises(error_type):
        qmc.control(qmc.x, num_controls=2, control_value=control_value)


def test_control_value_rejects_symbolic_width() -> None:
    """A fixed activation value requires a concrete control width."""
    symbolic_width = UInt(value=Value(type=UIntType(), name="width"))
    with pytest.raises(ValueError, match="requires a concrete int"):
        qmc.control(qmc.x, num_controls=symbolic_width, control_value=0)


def test_all_ones_control_value_uses_the_canonical_default() -> None:
    """An explicit all-ones value canonicalizes to ordinary control."""

    @qmc.qkernel
    def circuit(
        control_0: Qubit,
        control_1: Qubit,
        target: Qubit,
    ) -> tuple[Qubit, Qubit, Qubit]:
        """Trace an explicit all-ones controlled X.

        Args:
            control_0 (Qubit): Least-significant control.
            control_1 (Qubit): Most-significant control.
            target (Qubit): Target qubit.

        Returns:
            tuple[Qubit, Qubit, Qubit]: Updated controls and target.
        """
        return qmc.control(qmc.x, num_controls=2, control_value=3)(
            control_0,
            control_1,
            target,
        )

    [operation] = [
        operation
        for operation in circuit.block.operations
        if isinstance(operation, ConcreteControlledU)
    ]
    assert operation.control_value is None


def test_control_value_is_preserved_on_an_opaque_oracle() -> None:
    """Opaque controlled calls retain activation metadata for target emitters."""
    oracle = qmc.opaque("control_value_oracle", num_qubits=1)
    controlled = qmc.control(oracle, num_controls=2, control_value=2)

    @qmc.qkernel
    def circuit(
        control_0: Qubit,
        control_1: Qubit,
        target: Qubit,
    ) -> tuple[Qubit, Qubit, Qubit]:
        """Trace a value-activated opaque call.

        Args:
            control_0 (Qubit): Least-significant control.
            control_1 (Qubit): Most-significant control.
            target (Qubit): Oracle target.

        Returns:
            tuple[Qubit, Qubit, Qubit]: Updated controls and target.
        """
        return controlled(control_0, control_1, target)

    [invoke] = [
        operation
        for operation in circuit.block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert invoke.control_value == 2


def test_inverse_of_patterned_opaque_oracle_stays_controlled() -> None:
    """An opaque inverse retains controlled transform and activation value."""
    oracle = qmc.opaque("inverse_control_value_oracle", num_qubits=1)
    controlled = qmc.control(oracle, num_controls=2, control_value=2)

    @qmc.qkernel
    def oracle_layer(
        control_0: Qubit,
        control_1: Qubit,
        target: Qubit,
    ) -> tuple[Qubit, Qubit, Qubit]:
        """Apply a patterned opaque oracle.

        Args:
            control_0 (Qubit): Least-significant control.
            control_1 (Qubit): Most-significant control.
            target (Qubit): Oracle target.

        Returns:
            tuple[Qubit, Qubit, Qubit]: Updated controls and target.
        """
        return controlled(control_0, control_1, target)

    @qmc.qkernel
    def inverse_layer(
        control_0: Qubit,
        control_1: Qubit,
        target: Qubit,
    ) -> tuple[Qubit, Qubit, Qubit]:
        """Invert the patterned opaque oracle layer.

        Args:
            control_0 (Qubit): Least-significant control.
            control_1 (Qubit): Most-significant control.
            target (Qubit): Oracle target.

        Returns:
            tuple[Qubit, Qubit, Qubit]: Updated controls and target.
        """
        return qmc.inverse(oracle_layer)(control_0, control_1, target)

    [outer_inverse] = [
        operation
        for operation in inverse_layer.block.operations
        if isinstance(operation, InverseBlockOperation)
    ]
    assert outer_inverse.implementation_block is not None
    [invoke] = [
        operation
        for operation in outer_inverse.implementation_block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert invoke.transform is CallTransform.CONTROLLED
    assert invoke.control_value == 2
    assert invoke.target.name == "inverse_control_value_oracle_inv"


def test_control_value_composes_with_existing_oracle_controls() -> None:
    """New low-order control bits compose with existing all-one controls."""
    oracle = qmc.opaque(
        "nested_control_value_oracle",
        num_qubits=1,
        num_control_qubits=1,
    )
    controlled = qmc.control(oracle, num_controls=2, control_value=2)

    @qmc.qkernel
    def circuit(
        new_control_0: Qubit,
        new_control_1: Qubit,
        existing_control: Qubit,
        target: Qubit,
    ) -> tuple[Qubit, Qubit, Qubit, Qubit]:
        """Trace nested opaque controls in flattened call order.

        Args:
            new_control_0 (Qubit): New least-significant control.
            new_control_1 (Qubit): New second control.
            existing_control (Qubit): Oracle's original all-one control.
            target (Qubit): Oracle target.

        Returns:
            tuple[Qubit, Qubit, Qubit, Qubit]: Updated controls and target.
        """
        return controlled(
            new_control_0,
            new_control_1,
            existing_control,
            target,
        )

    [invoke] = [
        operation
        for operation in circuit.block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert invoke.num_control_qubits == 3
    assert invoke.control_value == 0b110


def test_control_value_round_trips_through_qkernel_serialization() -> None:
    """Structural and callable qkernels retain their control values."""
    concrete_restored = deserialize(serialize(_patterned_x_layer)).block
    [concrete] = [
        operation
        for operation in concrete_restored.operations
        if isinstance(operation, ConcreteControlledU)
    ]
    assert concrete.control_value == 2

    invoke_restored = deserialize(serialize(_patterned_composite_layer)).block
    [invoke] = [
        operation
        for operation in invoke_restored.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert invoke.control_value == 2


def test_inverse_block_control_value_round_trips_through_qkernel() -> None:
    """QKernel persistence retains patterned inverse activation metadata."""

    @qmc.qkernel
    def inverse_layer(
        control_0: Qubit,
        control_1: Qubit,
        target: Qubit,
    ) -> tuple[Qubit, Qubit, Qubit]:
        """Invert the patterned composite layer.

        Args:
            control_0 (Qubit): Least-significant control.
            control_1 (Qubit): Most-significant control.
            target (Qubit): Target qubit.

        Returns:
            tuple[Qubit, Qubit, Qubit]: Updated controls and target.
        """
        return qmc.inverse(_patterned_composite_layer)(
            control_0,
            control_1,
            target,
        )

    restored = deserialize(serialize(inverse_layer)).block
    [outer_inverse] = [
        operation
        for operation in restored.operations
        if isinstance(operation, InverseBlockOperation)
    ]
    assert outer_inverse.implementation_block is not None
    [patterned_inverse] = [
        operation
        for operation in outer_inverse.implementation_block.operations
        if isinstance(operation, InverseBlockOperation)
    ]
    assert patterned_inverse.control_value == 2


def test_inverse_of_patterned_qft_preserves_the_controlled_transform() -> None:
    """The QFT-to-IQFT inverse specialization retains its activation value."""

    @qmc.qkernel
    def inverse_layer(
        control_0: Qubit,
        control_1: Qubit,
        targets: Vector[Qubit],
    ) -> tuple[Qubit, Qubit, Vector[Qubit]]:
        """Invert the patterned controlled-QFT layer.

        Args:
            control_0 (Qubit): Least-significant control.
            control_1 (Qubit): Most-significant control.
            targets (Vector[Qubit]): QFT target register.

        Returns:
            tuple[Qubit, Qubit, Vector[Qubit]]: Updated controls and targets.
        """
        return qmc.inverse(_patterned_qft_layer)(
            control_0,
            control_1,
            targets,
        )

    [controlled_iqft] = [
        operation
        for operation in inverse_layer.block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert controlled_iqft.gate_type is CompositeGateType.IQFT
    assert controlled_iqft.transform is CallTransform.CONTROLLED
    assert controlled_iqft.control_value == 2


def test_control_value_participates_in_canonical_hashing() -> None:
    """Different activation values cannot share one canonical fingerprint."""
    block = InlinePass().run(_patterned_x_layer.block)
    [operation] = [
        operation
        for operation in block.operations
        if isinstance(operation, ConcreteControlledU)
    ]
    value_one = dataclasses.replace(operation, control_value=1)
    value_two = dataclasses.replace(operation, control_value=2)
    default = dataclasses.replace(operation, control_value=None)
    explicit_all_ones = dataclasses.replace(operation, control_value=3)

    operation_index = block.operations.index(operation)

    def with_operation(replacement: ConcreteControlledU) -> Any:
        """Return an affine block containing one replacement operation.

        Args:
            replacement (ConcreteControlledU): Controlled operation to insert.

        Returns:
            Any: Rebuilt affine test block.
        """
        operations = list(block.operations)
        operations[operation_index] = replacement
        return dataclasses.replace(block, operations=operations)

    assert content_hash(with_operation(value_one)) != content_hash(
        with_operation(value_two)
    )
    assert content_hash(with_operation(default)) == content_hash(
        with_operation(explicit_all_ones)
    )
