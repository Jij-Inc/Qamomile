"""Regression tests for resource-estimation control decompositions."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qm
import qamomile.observable as qm_o
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    GateOperation,
    GateOperationType,
    ProjectOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import (
    Operation,
    OperationKind,
    QInitOperation,
    Signature,
)
from qamomile.circuit.ir.types.primitives import BitType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.linalg import PauliLCU, PeriodicShiftLCU


@qm.qkernel
def _four_h_body(target: qm.Qubit) -> qm.Qubit:
    """Apply four Hadamard gates to one target."""
    for _index in qm.range(4):
        target = qm.h(target)
    return target


@qm.qkernel
def _scalar_h_body(target: qm.Qubit) -> qm.Qubit:
    """Apply one Hadamard gate to a scalar target."""
    return qm.h(target)


@qm.qkernel
def _loop_local_power_body(
    local_control: qm.Qubit,
    target: qm.Qubit,
    repetitions: qm.UInt,
) -> tuple[qm.Qubit, qm.Qubit]:
    """Use each loop-local index as a controlled-call power."""
    for index in qm.range(repetitions):
        local_control, target = qm.control(_scalar_h_body)(
            local_control,
            target,
            power=index,
        )
    return local_control, target


@qm.qkernel
def _affine_carry_power_body(
    local_control: qm.Qubit,
    target: qm.Qubit,
    repetitions: qm.UInt,
) -> tuple[qm.Qubit, qm.Qubit]:
    """Use an affine loop-carried counter as a controlled-call power."""
    count = qm.uint(0)
    for _index in qm.range(repetitions):
        local_control, target = qm.control(_scalar_h_body)(
            local_control,
            target,
            power=count,
        )
        count = count + 1
    return local_control, target


@qm.qkernel
def _fixed_point_nonlinear_body(
    target: qm.Qubit,
    repetitions: qm.UInt,
) -> qm.Qubit:
    """Keep a nonlinear carry at its fixed point while applying X gates."""
    count = qm.uint(1)
    for _index in qm.range(repetitions):
        if count > 0:
            target = qm.x(target)
        count = count * count
    return target


@qm.qkernel
def _unsupported_nonlinear_resource_body(
    local_control: qm.Qubit,
    target: qm.Qubit,
    repetitions: qm.UInt,
) -> tuple[qm.Qubit, qm.Qubit]:
    """Make controlled-call work depend on a non-fixed nonlinear carry."""
    count = qm.uint(2)
    for index in qm.range(repetitions):
        local_control, target = qm.control(_scalar_h_body)(
            local_control,
            target,
            power=count,
        )
        count = count * index + 1
    return local_control, target


@qm.qkernel
def _loop_local_power_circuit(repetitions: qm.UInt) -> qm.Bit:
    """Control a loop-local powered body with two outer controls."""
    controls = qm.qubit_array(2, "controls")
    local_control = qm.qubit("local_control")
    target = qm.qubit("target")
    *_, local_control, target = qm.control(
        _loop_local_power_body,
        num_controls=2,
    )(controls, local_control, target, repetitions)
    return qm.measure(target)


@qm.qkernel
def _affine_carry_power_circuit(repetitions: qm.UInt) -> qm.Bit:
    """Control an affine-carry powered body with two outer controls."""
    controls = qm.qubit_array(2, "controls")
    local_control = qm.qubit("local_control")
    target = qm.qubit("target")
    *_, local_control, target = qm.control(
        _affine_carry_power_body,
        num_controls=2,
    )(controls, local_control, target, repetitions)
    return qm.measure(target)


@qm.qkernel
def _fixed_point_nonlinear_circuit(repetitions: qm.UInt) -> qm.Bit:
    """Control a fixed-point nonlinear loop with two outer controls."""
    controls = qm.qubit_array(2, "controls")
    target = qm.qubit("target")
    *_, target = qm.control(
        _fixed_point_nonlinear_body,
        num_controls=2,
    )(controls, target, repetitions)
    return qm.measure(target)


@qm.qkernel
def _unsupported_nonlinear_resource_circuit(repetitions: qm.UInt) -> qm.Bit:
    """Control a resource-sensitive nonlinear loop with two outer controls."""
    controls = qm.qubit_array(2, "controls")
    local_control = qm.qubit("local_control")
    target = qm.qubit("target")
    *_, local_control, target = qm.control(
        _unsupported_nonlinear_resource_body,
        num_controls=2,
    )(controls, local_control, target, repetitions)
    return qm.measure(target)


@qm.qkernel
def _scalar_ry_body(target: qm.Qubit, theta: qm.Float) -> qm.Qubit:
    """Apply one parameterized rotation to a scalar target."""
    return qm.ry(target, theta)


@qm.qkernel
def _independent_h_body(
    left: qm.Qubit,
    right: qm.Qubit,
) -> tuple[qm.Qubit, qm.Qubit]:
    """Apply Hadamard gates to two independent targets."""
    return qm.h(left), qm.h(right)


@qm.qkernel
def _identity_body(target: qm.Qubit) -> qm.Qubit:
    """Return one target unchanged."""
    return target


@qm.qkernel
def _renamed_pauli_evolution(
    register: qm.Vector[qm.Qubit],
    observable: qm.Observable,
    time: qm.Float,
) -> qm.Vector[qm.Qubit]:
    """Apply Pauli evolution through renamed formal parameters."""
    return qm.pauli_evolve(register, observable, time)


@qm.qkernel
def _raw_pauli_input_probe(
    observable: qm.Observable,
) -> qm.Vector[qm.Qubit]:
    """Apply one supplied Hamiltonian to a fixed-width local register."""
    register = qm.qubit_array(2, "register")
    return qm.pauli_evolve(register, observable, qm.float_(0.25))


@qm.qkernel
def _raw_width_probe(
    register: qm.Vector[qm.Qubit],
) -> qm.Vector[qm.Qubit]:
    """Return a caller-owned register used to verify raw Block widths."""
    return register


@qm.qkernel
def _generic_block_unitary(
    signal: qm.Vector[qm.Qubit],
    system: qm.Vector[qm.Qubit],
) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
    """Return two arbitrary-width registers unchanged."""
    return signal, system


@qm.qkernel
def _shape_name_collision(
    signal: qm.Vector[qm.Qubit],
    signal_dim0: qm.UInt,
) -> qm.Vector[qm.Qubit]:
    """Exercise a classical argument colliding with a generated shape name."""
    for _ in qm.range(signal_dim0):
        signal[0] = qm.h(signal[0])
    return signal


@qm.qkernel
def _port_shape_name_collision(
    signal: qm.Vector[qm.Qubit],
    signal_dim0: qm.Vector[qm.Qubit],
) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
    """Exercise a generated shape name colliding with another quantum port."""
    return signal, signal_dim0


def _recursive_lcu_resource_encoding() -> qm.LCUBlockEncoding:
    """Build a public heterogeneous recursive LCU estimation fixture.

    Returns:
        qm.LCUBlockEncoding: Two-level encoding with complex phases and child
            descriptors from different public producers.
    """
    identity = qm.identity_block_encoding(1)
    ising = qm.ising_z_block_encoding({(): 1.0j, (0,): 0.5}, 1)
    return qm.lcu_block_encoding(
        (
            qm.LCUBlockEncodingTerm(1.0, ising),
            qm.LCUBlockEncodingTerm(-0.25j, identity),
        )
    )


class _ContextAwareOpaqueCost:
    """Return one symbolically constructed gate in the requested basis."""

    def __call__(self, ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Build a basis-compatible callback cost.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: One basis-sensitive modeled gate.
        """
        callback_work = sp.Symbol(
            "callback_work",
            integer=True,
            nonnegative=True,
        )
        symbolic = qm.ResourceEstimate(
            gates=qm.GateResources(
                total=callback_work,
                non_clifford=callback_work,
            ),
            basis=ctx.basis,
            control_decomposition=ctx.control_decomposition,
            precision=ctx.precision,
        )
        assert symbolic.parameters == {"callback_work": callback_work}
        return symbolic.substitute(callback_work=1)


class _BaseControlOpaqueCost:
    """Return one base gate while recording definition-level context."""

    def __call__(self, ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Build one base cost that does not price external controls.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: One definition-level gate plus context
            diagnostics in the call counters.
        """
        return qm.ResourceEstimate(
            gates=qm.GateResources(
                total=1,
                two_qubit=1,
            ),
            calls=qm.CallResources(
                calls_by_name={
                    f"callable={ctx.callable_name}": 1,
                    (f"definition_control_qubits={ctx.definition_control_qubits}"): 1,
                    f"target_qubits={ctx.target_qubits}": 1,
                }
            ),
        )


class _BaseArityProfileOpaqueCost:
    """Return one definition-level arity profile from a callback."""

    def __call__(self, ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return a base one-/two-qubit gate profile.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: Definition-level profile to which the
            estimator still applies external controls.
        """
        return qm.ResourceEstimate(
            gates=qm.GateResources(
                total=3,
                single_qubit=2,
                two_qubit=1,
            ),
            calls=qm.CallResources(
                calls_by_name={
                    (f"base_definition_controls={ctx.definition_control_qubits}"): 1,
                }
            ),
        )


class _BasisNeutralOpaqueCost:
    """Return a call-only cost that is independent of gate basis."""

    def __call__(self, ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return one explicitly modeled opaque call.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: Basis-neutral call resources.
        """
        del ctx
        return qm.ResourceEstimate(
            calls=qm.CallResources(calls_by_name={"neutral_oracle": 1})
        )


class _NonunitaryOpaqueCallbackCost:
    """Return a non-unitary base cost from a callback."""

    def __call__(self, ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return one definition-level measurement and reset.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: Non-unitary base cost.
        """
        del ctx
        return qm.ResourceEstimate(
            measurements=qm.MeasurementResources(total=1),
            resets=qm.ResetResources(total=1),
            depth=qm.DepthResources(
                depth=2,
                measurement_depth=1,
                reset_depth=1,
            ),
        )


class _UnsupportedQuantumOperation(Operation):
    """Represent a future quantum IR operation unknown to the estimator."""

    @property
    def signature(self) -> Signature:
        """Return an empty test signature."""
        return Signature()

    @property
    def operation_kind(self) -> OperationKind:
        """Classify the test operation as quantum."""
        return OperationKind.QUANTUM


def test_unknown_quantum_operation_fails_closed() -> None:
    """A future quantum operation cannot silently become exact zero cost."""
    with pytest.raises(NotImplementedError, match="exact zero-cost operation"):
        qm.ResourceEstimator().estimate([_UnsupportedQuantumOperation()])


@pytest.mark.parametrize(
    ("axis", "expected_gates", "expected_depth"),
    [("x", 2, 3), ("y", 4, 5), ("z", 0, 1)],
)
def test_raw_projection_axis_includes_semantic_basis_changes(
    axis: str,
    expected_gates: int,
    expected_depth: int,
) -> None:
    """Hand-built X/Y projection IR cannot undercount its basis changes.

    Args:
        axis (str): Projection axis stored in the raw operation.
        expected_gates (int): Basis-change gate count.
        expected_depth (int): Sequential gate-plus-measurement depth.
    """

    @qm.qkernel
    def circuit(target: qm.Qubit) -> tuple[qm.Qubit, qm.Bit]:
        """Build one ordinary Z projection for controlled IR mutation."""
        return qm.project_z(target)

    block = circuit.build()
    projection = next(
        operation
        for operation in block.operations
        if isinstance(operation, ProjectOperation)
    )
    projection.axis = axis

    estimate = qm.estimate_resources(block)

    assert estimate.gates.total == expected_gates
    assert estimate.measurements.total == 1
    assert estimate.depth.depth == expected_depth
    assert estimate.depth.gate_depth == expected_gates
    assert estimate.depth.measurement_depth == 1


def test_clean_ancilla_controlled_qkernel_shares_body_control_ladder() -> None:
    """A controlled multi-gate body computes one shared control conjunction."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Apply a four-gate body under three coherent controls."""
        controls = qm.qubit_array(3, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(_four_h_body, num_controls=3)(controls, target)
        return target

    clean_estimate = circuit.estimate_resources()
    abstract = circuit.estimate_resources(
        control_decomposition=qm.ControlDecomposition.ABSTRACT,
    )

    assert clean_estimate.gates.total == 8
    assert clean_estimate.gates.two_qubit == 4
    assert clean_estimate.gates.multi_qubit == 4
    assert clean_estimate.gates.toffoli == 4
    assert clean_estimate.width.allocated_qubits == 4
    assert clean_estimate.width.clean_ancilla_qubits == 2
    assert clean_estimate.width.peak_qubits == 6
    assert clean_estimate.width.circuit_qubits == 6
    assert clean_estimate.quality is qm.EstimateQuality.CONSERVATIVE

    assert abstract.gates.total == 4
    assert abstract.gates.multi_qubit == 4
    assert abstract.width.clean_ancilla_qubits == 0
    assert abstract.width.peak_qubits == 4


@pytest.mark.parametrize(
    (
        "num_controls",
        "expected_two_qubit",
        "expected_multi_qubit",
        "expected_clifford",
        "expected_toffoli",
    ),
    [
        (1, 2, 2, 2, 1),
        (2, 0, 4, 0, 2),
        (3, 0, 4, 0, 0),
    ],
)
def test_abstract_aggregate_control_shifts_arity_and_bounds_gate_families(
    num_controls: int,
    expected_two_qubit: int,
    expected_multi_qubit: int,
    expected_clifford: int,
    expected_toffoli: int,
) -> None:
    """Abstract aggregate control shifts arity without inventing gate names."""
    base = qm.ResourceEstimate(
        width=qm.WidthResources(
            clean_ancilla_qubits=1,
            peak_qubits=1,
        ),
        gates=qm.GateResources(
            total=4,
            single_qubit=2,
            two_qubit=1,
            multi_qubit=1,
            clifford=2,
            rotation=1,
            t=1,
            toffoli=1,
            non_clifford=2,
        ),
        depth=qm.DepthResources(
            depth=2,
            clifford_depth=1,
            rotation_depth=1,
            t_depth=1,
            toffoli_depth=1,
            non_clifford_depth=1,
            gate_depth=2,
        ),
        control_decomposition=qm.ControlDecomposition.ABSTRACT,
    )

    estimate = base.controlled(num_controls)

    assert estimate.gates.total == 4
    assert estimate.gates.single_qubit == 0
    assert estimate.gates.two_qubit == expected_two_qubit
    assert estimate.gates.multi_qubit == expected_multi_qubit
    assert estimate.gates.clifford == expected_clifford
    assert estimate.gates.rotation == 3
    assert estimate.gates.t == 0
    assert estimate.gates.toffoli == expected_toffoli
    assert estimate.gates.non_clifford == 4
    assert estimate.depth.depth == 4
    assert estimate.depth.gate_depth == 4
    assert estimate.depth.clifford_depth == expected_clifford
    assert estimate.depth.rotation_depth == 3
    assert estimate.depth.t_depth == 0
    assert estimate.depth.toffoli_depth == expected_toffoli
    assert estimate.depth.non_clifford_depth == 4
    assert estimate.width.clean_ancilla_qubits == 1
    assert estimate.width.peak_qubits == 1
    assert estimate.derivation is qm.EstimateDerivation.MODELED
    assert estimate.quality is qm.EstimateQuality.UNKNOWN
    assert any(
        "Gate-family fields are independent field-wise upper bounds"
        in assumption.message
        for assumption in estimate.assumptions
    )


def test_abstract_aggregate_control_keeps_unclassified_arity_explicit() -> None:
    """An incomplete abstract profile does not mislabel its arity remainder."""
    base = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=6,
            single_qubit=2,
            two_qubit=1,
            multi_qubit=1,
        ),
        control_decomposition=qm.ControlDecomposition.ABSTRACT,
    )

    one_control = base.controlled(1)
    two_controls = base.controlled(2)

    assert one_control.gates.total == 6
    assert one_control.gates.single_qubit == 0
    assert one_control.gates.two_qubit == 2
    assert one_control.gates.multi_qubit == 2
    assert two_controls.gates.total == 6
    assert two_controls.gates.single_qubit == 0
    assert two_controls.gates.two_qubit == 0
    assert two_controls.gates.multi_qubit == 4
    assert one_control.quality is qm.EstimateQuality.UNKNOWN
    assert two_controls.quality is qm.EstimateQuality.UNKNOWN
    assert any(
        "2 gate(s) have unclassified arity" in assumption.message
        for assumption in one_control.assumptions
    )
    assert any(
        "transformed arity fields therefore may not sum to total" in assumption.message
        for assumption in two_controls.assumptions
    )


def test_abstract_opaque_costs_apply_external_controls_after_base_cost() -> None:
    """Fixed and callback opaque costs share abstract control projection."""
    fixed_cost = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=3,
            single_qubit=2,
            two_qubit=1,
        ),
        control_decomposition=qm.ControlDecomposition.ABSTRACT,
    )
    fixed_oracle = qm.opaque(
        "abstract_fixed_oracle",
        num_qubits=1,
        cost=fixed_cost,
    )

    def callback_cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return the definition's base cost in the requested gate model."""
        return qm.ResourceEstimate(
            gates=qm.GateResources(
                total=3,
                single_qubit=2,
                two_qubit=1,
            ),
            basis=ctx.basis,
            control_decomposition=ctx.control_decomposition,
            precision=ctx.precision,
        )

    callback_oracle = qm.opaque(
        "abstract_callback_oracle",
        num_qubits=1,
        cost=callback_cost,
    )

    @qm.qkernel
    def fixed_circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply two controls outside the fixed-cost definition."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        target = qm.qubit("target")
        return qm.control(fixed_oracle, num_controls=2)(
            control_0,
            control_1,
            target,
        )

    @qm.qkernel
    def callback_circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply two controls outside the callback-cost definition."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        target = qm.qubit("target")
        return qm.control(callback_oracle, num_controls=2)(
            control_0,
            control_1,
            target,
        )

    fixed = fixed_circuit.estimate_resources(
        control_decomposition=qm.ControlDecomposition.ABSTRACT,
    )
    callback = callback_circuit.estimate_resources(
        control_decomposition=qm.ControlDecomposition.ABSTRACT,
    )

    assert fixed.gates == callback.gates
    assert fixed.depth == callback.depth
    assert fixed.gates.total == 3
    assert fixed.gates.single_qubit == 0
    assert fixed.gates.two_qubit == 0
    assert fixed.gates.multi_qubit == 3
    assert fixed.width.clean_ancilla_qubits == 0
    assert fixed.derivation is qm.EstimateDerivation.MODELED
    assert fixed.quality is qm.EstimateQuality.UNKNOWN
    assert callback.derivation is qm.EstimateDerivation.MODELED
    assert callback.quality is qm.EstimateQuality.UNKNOWN


@pytest.mark.parametrize(
    "control_decomposition",
    [
        qm.ControlDecomposition.ABSTRACT,
        qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI,
    ],
)
def test_clifford_t_aggregate_control_fails_without_gate_names(
    control_decomposition: qm.ControlDecomposition,
) -> None:
    """Clifford+T aggregate costs fail closed for every nonzero control."""
    base = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=1,
            single_qubit=1,
            t=1,
            non_clifford=1,
        ),
        basis=qm.GateBasis.CLIFFORD_T,
        control_decomposition=control_decomposition,
        precision=1e-10,
    )

    assert base.controlled(0) is base
    with pytest.raises(
        ValueError,
        match="aggregate resource projection is not defined.*clifford_t",
    ):
        base.controlled(1)


def test_controlled_logical_predicates_honor_concrete_inputs() -> None:
    """Concrete AND, OR, and NOT branches avoid a spurious shared ladder."""

    @qm.qkernel
    def conditional_body(
        target: qm.Qubit,
        left: qm.Bit,
        right: qm.Bit,
        enabled: qm.Bit,
    ) -> qm.Qubit:
        """Apply gates only when one of three logical predicates is true."""
        if left & right:
            target = qm.x(target)
        if left | right:
            target = qm.h(target)
        if ~enabled:
            target = qm.z(target)
        return target

    @qm.qkernel
    def circuit(left: qm.Bit, right: qm.Bit, enabled: qm.Bit) -> qm.Qubit:
        """Control the conditional body with three qubits."""
        controls = qm.qubit_array(3, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(
            conditional_body,
            num_controls=3,
        )(controls, target, left, right, enabled)
        return target

    estimate = circuit.estimate_resources(
        inputs={
            "left": False,
            "right": False,
            "enabled": True,
        }
    )

    assert estimate.gates.total == 0
    assert estimate.gates.toffoli == 0
    assert estimate.depth.depth == 0
    assert estimate.width.clean_ancilla_qubits == 0
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_clean_ancilla_controlled_loop_hoists_shared_ladder() -> None:
    """A repeated controlled body holds one ladder across all iterations."""

    @qm.qkernel
    def loop_body(target: qm.Qubit) -> qm.Qubit:
        """Apply three rotations in a static loop."""
        for _index in qm.range(3):
            target = qm.ry(target, 0.25)
        return target

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Apply the loop body under two controls."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(loop_body, num_controls=2)(controls, target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 5
    assert estimate.gates.two_qubit == 3
    assert estimate.gates.multi_qubit == 2
    assert estimate.gates.toffoli == 2
    assert estimate.width.clean_ancilla_qubits == 1
    assert estimate.width.peak_qubits == 4


def test_symbolic_controlled_loop_matches_direct_specialization() -> None:
    """Symbolic trip counts retain the same shared-ladder decomposition."""

    @qm.qkernel
    def loop_body(target: qm.Qubit, repetitions: qm.UInt) -> qm.Qubit:
        """Apply one X gate in every requested iteration."""
        for _index in qm.range(repetitions):
            target = qm.x(target)
        return target

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Qubit:
        """Apply the symbolic loop body under three controls."""
        controls = qm.qubit_array(3, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(loop_body, num_controls=3)(
            controls,
            target,
            repetitions,
        )
        return target

    symbolic = circuit.estimate_resources()
    for repetitions in (0, 1, 2, 3):
        specialized = symbolic.substitute(repetitions=repetitions)
        direct = circuit.estimate_resources(inputs={"repetitions": repetitions})
        assert specialized.gates == direct.gates
        assert specialized.depth == direct.depth
        assert specialized.width == direct.width

    active = symbolic.substitute(repetitions=2)
    assert active.gates.total == 6
    assert active.gates.toffoli == 4
    assert active.width.clean_ancilla_qubits == 2
    assert symbolic.substitute(repetitions=0).gates.total == 0


def test_symbolic_controlled_loop_accepts_integer_valued_float_bound() -> None:
    """Integer-valued float bounds fully specialize the batching expression."""

    @qm.qkernel
    def loop_body(target: qm.Qubit, repetitions: qm.UInt) -> qm.Qubit:
        """Apply one X gate in every requested iteration."""
        for _index in qm.range(repetitions):
            target = qm.x(target)
        return target

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Qubit:
        """Apply the symbolic loop body under three controls."""
        controls = qm.qubit_array(3, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(loop_body, num_controls=3)(
            controls,
            target,
            repetitions,
        )
        return target

    symbolic = circuit.estimate_resources()
    integer = symbolic.substitute(repetitions=3)
    float_substitution = symbolic.substitute(repetitions=3.0)
    direct_float = circuit.estimate_resources(inputs={"repetitions": 3.0})

    assert float_substitution.gates == integer.gates == direct_float.gates
    assert float_substitution.depth == integer.depth == direct_float.depth
    assert float_substitution.width == integer.width == direct_float.width
    assert float_substitution.parameters == direct_float.parameters == {}
    assert float_substitution.gates.total.is_number
    assert direct_float.gates.total.is_number


def test_transform_specific_open_control_body_keeps_x_bracket() -> None:
    """A controlled implementation still receives its open-control X pair."""
    formal_control = Value(type=QubitType(), name="formal_control")
    formal_target = Value(type=QubitType(), name="formal_target")
    formal_control_result = formal_control.next_version()
    formal_target_result = formal_target.next_version()
    body = Block(
        input_values=[formal_control, formal_target],
        output_values=[formal_control_result, formal_target_result],
        operations=[
            GateOperation.fixed(
                GateOperationType.CX,
                [formal_control, formal_target],
                [formal_control_result, formal_target_result],
            )
        ],
    )
    ref = CallableRef(namespace="test", name="open_control_implementation")
    implementation = CallableImplementation(
        transform=CallTransform.CONTROLLED,
        body=body,
    )
    control = Value(type=QubitType(), name="control")
    target = Value(type=QubitType(), name="target")
    operation = InvokeOperation(
        operands=[control, target],
        results=[control.next_version(), target.next_version()],
        target=ref,
        transform=CallTransform.CONTROLLED,
        attrs={
            "num_control_qubits": 1,
            "num_target_qubits": 1,
            "control_value": 0,
        },
        definition=CallableDef(ref=ref, implementations=[implementation]),
    )
    root = Block(
        operations=[
            QInitOperation(results=[control]),
            QInitOperation(results=[target]),
            operation,
        ],
        output_values=list(operation.results),
    )

    estimate = qm.ResourceEstimator().estimate(root)

    assert estimate.gates.total == 3
    assert estimate.gates.single_qubit == 2
    assert estimate.gates.two_qubit == 1
    assert estimate.depth.depth == 3
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE


def test_body_backed_oracle_aligns_added_and_declared_controls() -> None:
    """A base Oracle body retains declared controls and excludes added ones."""
    formal_declared = Value(type=QubitType(), name="formal_declared")
    formal_target = Value(type=QubitType(), name="formal_target")
    formal_declared_result = formal_declared.next_version()
    formal_target_result = formal_target.next_version()
    body = Block(
        input_values=[formal_declared, formal_target],
        output_values=[formal_declared_result, formal_target_result],
        operations=[
            GateOperation.fixed(
                GateOperationType.CX,
                [formal_declared, formal_target],
                [formal_declared_result, formal_target_result],
            )
        ],
    )
    ref = CallableRef(namespace="test", name="body_backed_oracle")
    definition_attrs = {
        "kind": "oracle",
        "num_control_qubits": 1,
        "num_declared_control_qubits": 1,
        "num_added_control_qubits": 0,
        "num_target_qubits": 1,
    }
    added = Value(type=QubitType(), name="added")
    declared = Value(type=QubitType(), name="declared")
    target = Value(type=QubitType(), name="target")
    operation = InvokeOperation(
        operands=[added, declared, target],
        results=[
            added.next_version(),
            declared.next_version(),
            target.next_version(),
        ],
        target=ref,
        transform=CallTransform.CONTROLLED,
        attrs={
            **definition_attrs,
            "num_control_qubits": 2,
            "num_added_control_qubits": 1,
        },
        definition=CallableDef(ref=ref, body=body, attrs=definition_attrs),
    )

    selection = operation.select_body()
    assert selection.operands == (declared, target)
    assert selection.results == tuple(operation.results[1:])

    root = Block(
        operations=[
            QInitOperation(results=[added]),
            QInitOperation(results=[declared]),
            QInitOperation(results=[target]),
            operation,
        ],
        output_values=list(operation.results),
    )
    estimate = qm.ResourceEstimator().estimate(root)

    assert estimate.gates.total == 1
    assert estimate.gates.toffoli == 1
    assert estimate.width.clean_ancilla_qubits == 0

    projected_target = formal_target.next_version()
    measured_bit = Value(type=BitType(), name="measured")
    observed_body = Block(
        input_values=[formal_declared, formal_target],
        output_values=[formal_declared, projected_target, measured_bit],
        operations=[
            ProjectOperation(
                operands=[formal_target],
                results=[projected_target, measured_bit],
                axis="z",
            )
        ],
    )
    observed = InvokeOperation(
        operands=[added, declared, target],
        results=[
            added.next_version(),
            declared.next_version(),
            target.next_version(),
            Value(type=BitType(), name="result"),
        ],
        target=ref,
        transform=CallTransform.CONTROLLED,
        attrs={
            **definition_attrs,
            "num_control_qubits": 2,
            "num_added_control_qubits": 1,
        },
        definition=CallableDef(
            ref=ref,
            body=observed_body,
            attrs=definition_attrs,
        ),
    )
    assert observed.measurement_result_indices == frozenset({3})


def test_loop_local_power_specializes_without_bound_symbol_leaks() -> None:
    """Loop-local powers choose the same decomposition before and after binding."""
    symbolic = _loop_local_power_circuit.estimate_resources()

    assert set(symbolic.parameters) == {"repetitions"}
    for repetitions in (0, 1, 2, 3, 3.0, 1000):
        specialized = symbolic.substitute(repetitions=repetitions)
        direct = _loop_local_power_circuit.estimate_resources(
            inputs={"repetitions": repetitions}
        )

        assert specialized.gates == direct.gates
        assert specialized.depth == direct.depth
        assert specialized.width == direct.width
        assert specialized.parameters == direct.parameters == {}
        assert specialized.gates.total.is_number

    assert symbolic.substitute(repetitions=1000).gates.total == 1_498_502


def test_affine_region_carry_matches_loop_local_power_specialization() -> None:
    """An affine RegionArg carry resolves like the equivalent loop index."""
    affine_symbolic = _affine_carry_power_circuit.estimate_resources()
    index_symbolic = _loop_local_power_circuit.estimate_resources()

    for repetitions in (0, 1, 2, 3):
        affine = affine_symbolic.substitute(repetitions=repetitions)
        direct = _affine_carry_power_circuit.estimate_resources(
            inputs={"repetitions": repetitions}
        )
        indexed = index_symbolic.substitute(repetitions=repetitions)

        assert affine.gates == direct.gates == indexed.gates
        assert affine.depth == direct.depth == indexed.depth
        assert affine.width == direct.width == indexed.width
        assert affine.parameters == direct.parameters == {}


def test_fixed_point_nonlinear_carry_uses_fixed_control_model() -> None:
    """A nonlinear fixed point follows the fixed shared-ladder threshold."""
    symbolic = _fixed_point_nonlinear_circuit.estimate_resources()

    expected = {
        0: (0, 0, 0),
        1: (1, 1, 0),
        2: (4, 2, 1),
    }
    for repetitions, (
        expected_total,
        expected_toffoli,
        expected_clean_ancillas,
    ) in expected.items():
        specialized = symbolic.substitute(repetitions=repetitions)
        direct = _fixed_point_nonlinear_circuit.estimate_resources(
            inputs={"repetitions": repetitions}
        )

        assert specialized.gates == direct.gates
        assert specialized.depth == direct.depth
        assert specialized.width == direct.width
        assert specialized.parameters == direct.parameters == {}
        assert direct.gates.total == expected_total
        assert direct.gates.toffoli == expected_toffoli
        assert direct.width.clean_ancilla_qubits == expected_clean_ancillas


def test_resource_sensitive_nonlinear_carry_requires_concrete_bounds() -> None:
    """Unsupported nonlinear work replays any concrete loop exactly."""
    with pytest.raises(
        NotImplementedError,
        match="unsupported nonlinear loop-carried recurrence",
    ):
        _unsupported_nonlinear_resource_circuit.estimate_resources()

    for repetitions, expected_total in ((0, 0), (1, 8), (2, 11)):
        estimate = _unsupported_nonlinear_resource_circuit.estimate_resources(
            inputs={"repetitions": repetitions}
        )

        assert estimate.gates.total == expected_total
        assert estimate.parameters == {}
        assert estimate.gates.total.is_number

    estimate = _unsupported_nonlinear_resource_circuit.estimate_resources(
        inputs={"repetitions": 65}
    )
    count = 2
    expected_total = 2
    for index in range(65):
        expected_total += 3 * count
        count = count * index + 1

    assert estimate.gates.total == expected_total
    assert estimate.parameters == {}


def test_symbolic_control_width_retains_shared_ladder_variants() -> None:
    """Substituting a control width selects the matching concrete recipe."""

    @qm.qkernel
    def two_h(target: qm.Qubit) -> qm.Qubit:
        """Apply two Hadamard gates."""
        return qm.h(qm.h(target))

    @qm.qkernel
    def circuit(width: qm.UInt) -> qm.Qubit:
        """Control the two-gate body with a symbolic-width register."""
        controls = qm.qubit_array(width, "controls")
        target = qm.qubit("target")
        controls, target = qm.control(two_h, num_controls=width)(
            controls,
            target,
        )
        return target

    symbolic = circuit.estimate_resources()
    for width, expected_total, expected_clean in (
        (1, 2, 0),
        (2, 4, 1),
        (3, 6, 2),
    ):
        specialized = symbolic.substitute(width=width)
        direct = circuit.estimate_resources(inputs={"width": width})
        assert specialized.gates == direct.gates
        assert specialized.depth == direct.depth
        assert specialized.width == direct.width
        assert direct.gates.total == expected_total
        assert direct.width.clean_ancilla_qubits == expected_clean


def test_symbolic_loop_activity_selects_body_wide_two_control_ladder() -> None:
    """A conditional loop leaf controls the body-wide batching predicate."""

    @qm.qkernel
    def body(target: qm.Qubit, repetitions: qm.UInt) -> qm.Qubit:
        """Apply X once and then a symbolic number of Hadamards."""
        target = qm.x(target)
        for _index in qm.range(repetitions):
            target = qm.h(target)
        return target

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Qubit:
        """Apply the mixed body under exactly two controls."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(body, num_controls=2)(
            controls,
            target,
            repetitions,
        )
        return target

    symbolic = circuit.estimate_resources()
    for repetitions, expected_total, expected_clean in (
        (0, 1, 0),
        (1, 4, 1),
        (2, 5, 1),
    ):
        specialized = symbolic.substitute(repetitions=repetitions)
        direct = circuit.estimate_resources(inputs={"repetitions": repetitions})
        assert specialized.gates == direct.gates
        assert specialized.depth == direct.depth
        assert specialized.width == direct.width
        assert direct.gates.total == expected_total
        assert direct.width.clean_ancilla_qubits == expected_clean


def test_symbolic_branch_uses_one_body_wide_shared_ladder() -> None:
    """A symbolic branch does not create a branch-local control ladder."""

    @qm.qkernel
    def body(target: qm.Qubit, enabled: qm.UInt) -> qm.Qubit:
        """Apply gates before, inside, and after a symbolic branch."""
        target = qm.x(target)
        if enabled:
            target = qm.h(target)
        return qm.h(target)

    @qm.qkernel
    def circuit(enabled: qm.UInt) -> qm.Qubit:
        """Apply the branch body under exactly two controls."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(body, num_controls=2)(
            controls,
            target,
            enabled,
        )
        return target

    symbolic = circuit.estimate_resources()
    for enabled in (0, 1):
        specialized = symbolic.substitute(enabled=enabled)
        direct = circuit.estimate_resources(inputs={"enabled": enabled})
        assert specialized.gates == direct.gates
        assert specialized.depth == direct.depth
        assert specialized.width == direct.width

    enabled = symbolic.substitute(enabled=1)
    assert enabled.gates.two_qubit == 3
    assert enabled.gates.multi_qubit == 2
    assert enabled.gates.toffoli == 2
    assert enabled.width.clean_ancilla_qubits == 1


def test_two_control_x_body_uses_fixed_shared_ladder() -> None:
    """Two modeled X operations share one ladder under two controls."""

    @qm.qkernel
    def x_body(
        left: qm.Qubit,
        right: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Flip two independent targets."""
        return qm.x(left), qm.x(right)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply both X gates under two controls."""
        controls = qm.qubit_array(2, "controls")
        left = qm.qubit("left")
        right = qm.qubit("right")
        *_, left, right = qm.control(x_body, num_controls=2)(
            controls,
            left,
            right,
        )
        return left, right

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 4
    assert estimate.gates.two_qubit == 2
    assert estimate.gates.multi_qubit == 2
    assert estimate.gates.toffoli == 2
    assert estimate.width.clean_ancilla_qubits == 1
    assert estimate.width.peak_qubits == 5


def test_fixed_shared_ladder_is_invariant_across_ir_boundaries() -> None:
    """Calls, inverse, and loops preserve the fixed work threshold."""

    @qm.qkernel
    def two_x(target: qm.Qubit) -> qm.Qubit:
        """Apply two sequential X gates directly."""
        target = qm.x(target)
        return qm.x(target)

    @qm.qkernel
    def delegated_two_x(target: qm.Qubit) -> qm.Qubit:
        """Delegate the two X gates through a qkernel call."""
        return two_x(target)

    @qm.qkernel
    def inverse_two_x(target: qm.Qubit) -> qm.Qubit:
        """Wrap the same two X gates in an inverse block."""
        return qm.inverse(two_x)(target)

    @qm.qkernel
    def loop_two_x(target: qm.Qubit) -> qm.Qubit:
        """Repeat one X gate through a concrete two-iteration loop."""
        for _index in qm.range(2):
            target = qm.x(target)
        return target

    @qm.qkernel
    def direct_circuit() -> qm.Qubit:
        """Control the direct body with two outer qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(two_x, num_controls=2)(controls, target)
        return target

    @qm.qkernel
    def delegated_circuit() -> qm.Qubit:
        """Control the delegated body with two outer qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(delegated_two_x, num_controls=2)(
            controls,
            target,
        )
        return target

    @qm.qkernel
    def inverse_circuit() -> qm.Qubit:
        """Control the inverse-wrapped body with two outer qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(inverse_two_x, num_controls=2)(
            controls,
            target,
        )
        return target

    @qm.qkernel
    def loop_circuit() -> qm.Qubit:
        """Control the concrete loop body with two outer qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(loop_two_x, num_controls=2)(controls, target)
        return target

    estimates = (
        direct_circuit.estimate_resources(),
        delegated_circuit.estimate_resources(),
        inverse_circuit.estimate_resources(),
        loop_circuit.estimate_resources(),
    )

    for estimate in estimates:
        assert estimate.gates.total == 4
        assert estimate.gates.two_qubit == 2
        assert estimate.gates.multi_qubit == 2
        assert estimate.gates.toffoli == 2
        assert estimate.depth.depth == 4
        assert estimate.width.clean_ancilla_qubits == 1
        assert estimate.width.peak_qubits == 4


def test_zero_cost_slice_markers_do_not_change_control_recipe() -> None:
    """Array-view bookkeeping does not count as controlled quantum work."""

    @qm.qkernel
    def direct_x(register: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
        """Apply one X gate directly to the register."""
        register[0] = qm.x(register[0])
        return register

    @qm.qkernel
    def sliced_x(register: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
        """Apply the same X gate through a one-element slice view."""
        view = register[0:1]
        view[0] = qm.x(view[0])
        register[0:1] = view
        return register

    @qm.qkernel
    def direct_circuit() -> qm.Vector[qm.Qubit]:
        """Control the direct body with two qubits."""
        controls = qm.qubit_array(2, "controls")
        register = qm.qubit_array(1, "register")
        *_, register = qm.control(direct_x, num_controls=2)(
            controls,
            register,
        )
        return register

    @qm.qkernel
    def sliced_circuit() -> qm.Vector[qm.Qubit]:
        """Control the slice-backed body with two qubits."""
        controls = qm.qubit_array(2, "controls")
        register = qm.qubit_array(1, "register")
        *_, register = qm.control(sliced_x, num_controls=2)(
            controls,
            register,
        )
        return register

    direct = direct_circuit.estimate_resources()
    sliced = sliced_circuit.estimate_resources()

    assert sliced.gates == direct.gates
    assert sliced.depth == direct.depth
    assert sliced.width == direct.width
    assert direct.gates.total == 1
    assert direct.gates.toffoli == 1
    assert direct.width.clean_ancilla_qubits == 0


def test_explicit_opaque_calls_share_the_outer_control_ladder() -> None:
    """Fixed and callback costs preserve batching across Invoke boundaries."""
    base_cost = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
    )
    fixed_oracle = qm.opaque(
        "fixed_outer_batch_leaf",
        num_qubits=1,
        cost=base_cost,
    )
    observed: list[qm.OpaqueCostContext] = []

    def callback_cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return one modeled single-qubit operation.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: One single-qubit operation.
        """
        observed.append(ctx)
        return base_cost

    callback_oracle = qm.opaque(
        "callback_outer_batch_leaf",
        num_qubits=1,
        cost=callback_cost,
    )

    @qm.qkernel
    def direct_body(target: qm.Qubit) -> qm.Qubit:
        """Apply two direct Hadamard gates."""
        target = qm.h(target)
        return qm.h(target)

    @qm.qkernel
    def fixed_body(target: qm.Qubit) -> qm.Qubit:
        """Invoke two fixed-cost opaque leaves."""
        (target,) = fixed_oracle(target)
        (target,) = fixed_oracle(target)
        return target

    @qm.qkernel
    def callback_body(target: qm.Qubit) -> qm.Qubit:
        """Invoke two callback-priced opaque leaves."""
        (target,) = callback_oracle(target)
        (target,) = callback_oracle(target)
        return target

    @qm.qkernel
    def direct_circuit() -> qm.Qubit:
        """Control the direct body with two qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(direct_body, num_controls=2)(controls, target)
        return target

    @qm.qkernel
    def fixed_circuit() -> qm.Qubit:
        """Control the fixed-cost body with two qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(fixed_body, num_controls=2)(controls, target)
        return target

    @qm.qkernel
    def callback_circuit() -> qm.Qubit:
        """Control the callback-priced body with two qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(callback_body, num_controls=2)(controls, target)
        return target

    direct = direct_circuit.estimate_resources()
    fixed = fixed_circuit.estimate_resources()
    callback = callback_circuit.estimate_resources()

    for estimate in (fixed, callback):
        assert estimate.gates.total == direct.gates.total
        assert estimate.gates.single_qubit == direct.gates.single_qubit
        assert estimate.gates.two_qubit == direct.gates.two_qubit
        assert estimate.gates.multi_qubit == direct.gates.multi_qubit
        assert estimate.gates.toffoli == direct.gates.toffoli
        assert estimate.depth.depth == direct.depth.depth
        assert estimate.depth.gate_depth == direct.depth.gate_depth
        assert estimate.width.clean_ancilla_qubits == 1
        assert estimate.gates.total == 4
    assert len(observed) == 2


def test_query_only_opaque_call_does_not_trigger_control_batching() -> None:
    """Call/query provenance without gate work stays below the threshold."""
    query_only = qm.opaque(
        "query_only_batch_boundary",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            calls=qm.CallResources(
                queries_by_name={"query_only_batch_boundary": 1},
            )
        ),
    )

    @qm.qkernel
    def with_query(target: qm.Qubit) -> qm.Qubit:
        """Record one query before applying one Hadamard gate."""
        (target,) = query_only(target)
        return qm.h(target)

    @qm.qkernel
    def one_h(target: qm.Qubit) -> qm.Qubit:
        """Apply one Hadamard gate."""
        return qm.h(target)

    @qm.qkernel
    def query_circuit() -> qm.Qubit:
        """Control the query-bearing body with two qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(with_query, num_controls=2)(controls, target)
        return target

    @qm.qkernel
    def gate_circuit() -> qm.Qubit:
        """Control the gate-only body with two qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(one_h, num_controls=2)(controls, target)
        return target

    with_query_estimate = query_circuit.estimate_resources()
    gate_only_estimate = gate_circuit.estimate_resources()

    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.estimator.resource_estimator import (
        ResourceInterpreter,
        _ResourceEstimatorConfig,
    )

    body = with_query.block
    profile = ResourceInterpreter(
        config=_ResourceEstimatorConfig(),
        bindings={},
    )._controlled_body_batch_profile(
        body.operations,
        ExprResolver(body),
    )

    assert profile.work == 1
    assert with_query_estimate.gates == gate_only_estimate.gates
    assert with_query_estimate.depth == gate_only_estimate.depth
    assert with_query_estimate.width == gate_only_estimate.width
    assert with_query_estimate.gates.total == 3
    assert with_query_estimate.gates.toffoli == 2
    assert with_query_estimate.width.clean_ancilla_qubits == 1
    assert with_query_estimate.calls.queries_by_name == {"query_only_batch_boundary": 1}


def test_symbolic_opaque_work_retains_outer_batching_variants() -> None:
    """Opaque symbolic activity selects the matching recipe after substitution."""
    work = sp.Symbol("opaque_work", integer=True, nonnegative=True)
    oracle = qm.opaque(
        "symbolic_outer_batch_leaf",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(
                total=work,
                single_qubit=work,
            )
        ),
    )

    @qm.qkernel
    def body(target: qm.Qubit) -> qm.Qubit:
        """Apply symbolic opaque work followed by one Hadamard gate."""
        (target,) = oracle(target)
        return qm.h(target)

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Control the body with two qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(body, num_controls=2)(controls, target)
        return target

    symbolic = circuit.estimate_resources()
    inactive = symbolic.substitute(opaque_work=0)
    active = symbolic.substitute(opaque_work=1)

    assert inactive.gates.total == 3
    assert active.gates.total == 4
    assert inactive.width.clean_ancilla_qubits == 1
    assert active.width.clean_ancilla_qubits == 1


def test_inactive_controlled_body_does_not_evaluate_opaque_callback() -> None:
    """A zero-power controlled call skips opaque cost profiling entirely."""
    observed: list[qm.OpaqueCostContext] = []

    def callback_cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Record a callback execution.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: One single-qubit operation.
        """
        observed.append(ctx)
        return qm.ResourceEstimate(
            gates=qm.GateResources(total=1, single_qubit=1),
        )

    oracle = qm.opaque(
        "inactive_profile_oracle",
        num_qubits=1,
        cost=callback_cost,
    )

    @qm.qkernel
    def body(target: qm.Qubit) -> qm.Qubit:
        """Invoke the callback-priced opaque Oracle."""
        (target,) = oracle(target)
        return target

    @qm.qkernel
    def circuit(power: qm.UInt) -> qm.Qubit:
        """Apply the controlled body zero times."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(body, num_controls=2)(
            controls,
            target,
            power=power,
        )
        return target

    estimate = circuit.estimate_resources(inputs={"power": 0})

    assert estimate.gates.total == 0
    assert observed == []


def test_zero_iteration_controlled_loop_does_not_evaluate_opaque_callback() -> None:
    """A concretely empty range skips its controlled opaque body entirely."""
    observed: list[qm.OpaqueCostContext] = []

    def callback_cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Record a callback execution.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: One single-qubit operation.
        """
        observed.append(ctx)
        return qm.ResourceEstimate(
            gates=qm.GateResources(total=1, single_qubit=1),
        )

    oracle = qm.opaque(
        "zero_iteration_profile_oracle",
        num_qubits=1,
        cost=callback_cost,
    )

    @qm.qkernel
    def body(target: qm.Qubit, repetitions: qm.UInt) -> qm.Qubit:
        """Invoke the opaque Oracle once per range iteration."""
        for _index in qm.range(repetitions):
            (target,) = oracle(target)
        return target

    @qm.qkernel
    def circuit(repetitions: qm.UInt) -> qm.Qubit:
        """Apply the range body under two coherent controls."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(body, num_controls=2)(
            controls,
            target,
            repetitions,
        )
        return target

    estimate = circuit.estimate_resources(inputs={"repetitions": 0})

    assert estimate.gates.total == 0
    assert estimate.quality is qm.EstimateQuality.EXACT
    assert observed == []


def test_single_intrinsic_cx_uses_per_primitive_recipe() -> None:
    """One modeled operation stays below the fixed sharing threshold."""

    @qm.qkernel
    def cx_body(
        intrinsic: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one intrinsically controlled X gate."""
        return qm.cx(intrinsic, target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Add two surrounding controls to the intrinsic CX."""
        outer = qm.qubit_array(2, "outer")
        intrinsic = qm.qubit("intrinsic")
        target = qm.qubit("target")
        *_, intrinsic, target = qm.control(cx_body, num_controls=2)(
            outer,
            intrinsic,
            target,
        )
        return intrinsic, target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 5
    assert estimate.gates.two_qubit == 1
    assert estimate.gates.multi_qubit == 4
    assert estimate.gates.toffoli == 4
    assert estimate.depth.depth == 5
    assert estimate.width.clean_ancilla_qubits == 2
    assert estimate.width.peak_qubits == 6


def test_nested_controlled_u_uses_shared_outer_ladder() -> None:
    """A local ControlledU composes with one shared outer carrier."""

    @qm.qkernel
    def locally_controlled_x(
        local: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one local control to X."""
        return qm.control(qm.x)(local, target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Add two surrounding controls to the locally controlled body."""
        outer = qm.qubit_array(2, "outer")
        local = qm.qubit("local")
        target = qm.qubit("target")
        *_, local, target = qm.control(
            locally_controlled_x,
            num_controls=2,
        )(outer, local, target)
        return local, target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.toffoli == 3
    assert estimate.depth.depth == 3
    assert estimate.width.clean_ancilla_qubits == 1
    assert estimate.width.peak_qubits == 5


def test_controlled_composite_invoke_uses_shared_outer_ladder() -> None:
    """A controlled composite call preserves its intrinsic control boundary."""

    @qm.composite_gate(name="control_batch_profile_boxed_x")
    def boxed_x(target: qm.Qubit) -> qm.Qubit:
        """Invoke one boxed X gate."""
        return qm.x(target)

    @qm.qkernel
    def locally_controlled_box(
        local: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one local control to the composite invocation."""
        return qm.control(boxed_x)(local, target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Add two surrounding controls to the controlled invocation."""
        outer = qm.qubit_array(2, "outer")
        local = qm.qubit("local")
        target = qm.qubit("target")
        *_, local, target = qm.control(
            locally_controlled_box,
            num_controls=2,
        )(outer, local, target)
        return local, target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.toffoli == 3
    assert estimate.depth.depth == 3
    assert estimate.width.clean_ancilla_qubits == 1
    assert estimate.width.peak_qubits == 5


@pytest.mark.parametrize(
    ("num_controls", "total", "toffoli", "clean_ancillas"),
    [
        (1, 1, 0, 0),
        (2, 1, 1, 0),
        (3, 5, 4, 2),
        (5, 9, 8, 4),
    ],
)
def test_clean_ancilla_controlled_x_scales_with_control_arity(
    num_controls: int,
    total: int,
    toffoli: int,
    clean_ancillas: int,
) -> None:
    """Multi-controlled X reports recipe gates and reusable ancillas.

    Args:
        num_controls (int): Number of coherent controls.
        total (int): Expected clean-ancilla Toffoli recipe gate count.
        toffoli (int): Expected Toffoli count.
        clean_ancillas (int): Expected reusable clean ancillas.
    """

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Apply one X under the requested number of controls."""
        controls = qm.qubit_array(num_controls, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(qm.x, num_controls=num_controls)(controls, target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == total
    assert estimate.gates.toffoli == toffoli
    assert estimate.width.clean_ancilla_qubits == clean_ancillas
    assert estimate.qubits == num_controls + 1 + clean_ancillas


def test_symbolic_control_count_is_validated_and_zero_power_is_identity() -> None:
    """Structural control width persists while symbolic zero power stays legal."""

    @qm.qkernel
    def circuit(width: qm.UInt, power: qm.UInt) -> qm.Qubit:
        """Apply a symbolically controlled and powered Hadamard body."""
        controls = qm.qubit_array(width, "controls")
        target = qm.qubit("target")
        controls, target = qm.control(
            _scalar_h_body,
            num_controls=width,
        )(controls, target, power=power)
        return target

    symbolic = circuit.estimate_resources()
    zero_power = symbolic.substitute(width=2, power=0)

    assert zero_power.gates.total == 0
    assert circuit.estimate_resources(inputs={"width": 2, "power": 0}).gates.total == 0
    clifford_t_zero = circuit.estimate_resources(
        inputs={"width": 2, "power": 0},
        basis=qm.GateBasis.CLIFFORD_T,
    )
    assert clifford_t_zero.gates.total == 0
    assert clifford_t_zero.width.clean_ancilla_qubits == 0
    with pytest.raises(ValueError, match="at least 1 control qubit"):
        symbolic.substitute(width=0, power=1)
    with pytest.raises(ValueError, match="at least 1 control qubit"):
        circuit.estimate_resources(inputs={"width": 0, "power": 1})


def test_symbolic_control_width_matches_operands_and_selected_indices() -> None:
    """Control metadata retains operand width, bounds, and uniqueness rules."""

    @qm.qkernel
    def pooled(
        width: qm.UInt,
        left_index: qm.UInt,
        right_index: qm.UInt,
    ) -> qm.Qubit:
        """Control one gate through two selected slots of a fixed pool."""
        controls = qm.qubit_array(4, "controls")
        target = qm.qubit("target")
        controlled = qm.control(
            _scalar_h_body,
            num_controls=width,
        )
        controls, target = controlled(
            controls,
            target,
            control_indices=(left_index, right_index),
        )
        return target

    @qm.qkernel
    def whole_pool(width: qm.UInt) -> qm.Qubit:
        """Use every slot in a fixed two-qubit control pool."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        controls, target = qm.control(
            _scalar_h_body,
            num_controls=width,
        )(controls, target)
        return target

    assert whole_pool.estimate_resources(inputs={"width": 2}).gates.total == 3
    with pytest.raises(ValueError, match="control operand width must equal 3"):
        whole_pool.estimate_resources(inputs={"width": 3})

    valid = pooled.estimate_resources(
        inputs={"width": 2, "left_index": 0, "right_index": 3}
    )
    assert valid.gates.total == 3
    with pytest.raises(ValueError, match="indices uniqueness"):
        pooled.estimate_resources(
            inputs={"width": 2, "left_index": 1, "right_index": 1}
        )
    with pytest.raises(ValueError, match="index 1 upper bound"):
        pooled.estimate_resources(
            inputs={"width": 2, "left_index": 0, "right_index": 4}
        )


def test_descending_loop_preserves_control_constraints_after_inputs() -> None:
    """Direct inputs retain negative steps and validate every control width."""

    @qm.qkernel
    def circuit(n: qm.UInt) -> qm.Vector[qm.Bit]:
        """Use a nonlinear control width in a descending symbolic range."""
        qubits = qm.qubit_array(10, "qubits")
        for index in qm.range(n, 0, -1):
            width = (index - 2) * (index - 2)
            qubits[0:width], qubits[9] = qm.control(
                _scalar_h_body,
                num_controls=width,
            )(qubits[0:width], qubits[9])
        return qm.measure(qubits)

    with pytest.raises(ValueError, match="at least 1 control qubit"):
        circuit.estimate_resources(inputs={"n": 3})


def test_open_control_brackets_forward_and_inverse_once() -> None:
    """A mixed control value adds one X pair around the complete body."""

    @qm.qkernel
    def patterned(
        control_0: qm.Qubit,
        control_1: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply the four-gate body when the control register equals two."""
        return qm.control(
            _four_h_body,
            num_controls=2,
            control_value=2,
        )(control_0, control_1, target)

    @qm.qkernel
    def inverse_circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply the inverse of the mixed-control body."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        target = qm.qubit("target")
        return qm.inverse(patterned)(control_0, control_1, target)

    forward = patterned.estimate_resources()
    inverse = inverse_circuit.estimate_resources()

    for estimate in (forward, inverse):
        assert estimate.gates.total == 8
        assert estimate.gates.single_qubit == 2
        assert estimate.gates.two_qubit == 4
        assert estimate.gates.multi_qubit == 2
        assert estimate.gates.toffoli == 2
        assert estimate.width.clean_ancilla_qubits == 1


def test_nested_open_control_brackets_remain_unconditional() -> None:
    """Local open-control X brackets remain single-qubit operations."""

    @qm.qkernel
    def inner_open_control(
        local_control: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply X when the local control is zero."""
        return qm.control(
            qm.x,
            num_controls=1,
            control_value=0,
        )(local_control, target)

    @qm.qkernel
    def circuit() -> tuple[qm.Vector[qm.Qubit], qm.Qubit, qm.Qubit]:
        """Place the open-control operation under three outer controls."""
        outer_controls = qm.qubit_array(3, "outer_controls")
        local_control = qm.qubit("local_control")
        target = qm.qubit("target")
        *_, local_control, target = qm.control(
            inner_open_control,
            num_controls=3,
        )(
            outer_controls,
            local_control,
            target,
        )
        return outer_controls, local_control, target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 7
    assert estimate.gates.single_qubit == 2
    assert estimate.gates.two_qubit == 0
    assert estimate.gates.multi_qubit == 5
    assert estimate.gates.toffoli == 5
    assert estimate.depth.depth == 7
    assert estimate.width.clean_ancilla_qubits == 2
    assert estimate.width.allocated_qubits == 5
    assert estimate.width.peak_qubits == 7


def test_single_outer_control_does_not_control_local_open_brackets() -> None:
    """One outer control leaves the local normalization X pair unconditional."""

    @qm.qkernel
    def inner_open_control(
        local_control: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply X when the local control is zero."""
        return qm.control(
            qm.x,
            num_controls=1,
            control_value=0,
        )(local_control, target)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Place the open-control operation under one outer control."""
        outer_control = qm.qubit("outer_control")
        local_control = qm.qubit("local_control")
        target = qm.qubit("target")
        return qm.control(inner_open_control)(
            outer_control,
            local_control,
            target,
        )

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.single_qubit == 2
    assert estimate.gates.two_qubit == 0
    assert estimate.gates.multi_qubit == 1
    assert estimate.gates.toffoli == 1
    assert estimate.depth.depth == 3
    assert estimate.width.clean_ancilla_qubits == 0
    assert estimate.width.allocated_qubits == 3
    assert estimate.width.peak_qubits == 3


def test_controlled_scalar_qkernel_broadcasts_over_vector() -> None:
    """A scalar controlled body is counted once per vector target element."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Qubit]:
        """Broadcast one controlled Hadamard over three targets."""
        control = qm.qubit("control")
        targets = qm.qubit_array(3, "targets")
        control, targets = qm.control(_scalar_h_body)(control, targets)
        return targets

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.two_qubit == 3
    assert estimate.depth.depth == 3
    assert estimate.width.allocated_qubits == 4
    assert estimate.width.peak_qubits == 4


def test_controlled_call_boundary_marks_aggregate_completion_upper_bound() -> None:
    """Unequal controlled-body exits disclose conservative caller scheduling."""

    @qm.qkernel
    def uneven_body(
        left: qm.Qubit,
        right: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Finish the left target one layer after the right target."""
        left = qm.x(left)
        left = qm.x(left)
        right = qm.x(right)
        return left, right

    @qm.qkernel
    def circuit() -> qm.Bit:
        """Consume the earlier body output after a two-control call."""
        controls = qm.qubit_array(2, "controls")
        left = qm.qubit("left")
        right = qm.qubit("right")
        *_, left, right = qm.control(uneven_body, num_controls=2)(
            controls,
            left,
            right,
        )
        left = qm.x(left)
        return qm.measure(left)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 6
    assert estimate.depth.gate_depth == 6
    assert estimate.depth.depth == 7
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert any(
        "aggregate latency" in assumption.message for assumption in estimate.assumptions
    )


def test_scalar_vector_broadcast_discloses_aggregate_element_completion() -> None:
    """A multi-element scalar broadcast labels its conservative exit latency."""

    @qm.qkernel
    def scalar_x(target: qm.Qubit) -> qm.Qubit:
        """Apply one X gate to a scalar target."""
        return qm.x(target)

    @qm.qkernel
    def circuit(width: qm.UInt) -> qm.Bit:
        """Consume the first target after a two-control scalar broadcast."""
        controls = qm.qubit_array(2, "controls")
        targets = qm.qubit_array(width, "targets")
        *_, targets = qm.control(scalar_x, num_controls=2)(controls, targets)
        targets[0] = qm.x(targets[0])
        return qm.measure(targets[0])

    estimate = circuit.estimate_resources(inputs={"width": 2})

    assert estimate.gates.total == 3
    assert estimate.depth.gate_depth == 3
    assert estimate.depth.depth == 4
    assert estimate.quality is qm.EstimateQuality.CONSERVATIVE
    assert any(
        "scalar-to-vector broadcast" in assumption.message
        for assumption in estimate.assumptions
    )


def test_open_control_broadcast_over_empty_vector_is_identity() -> None:
    """An empty scalar-target broadcast emits neither its body nor brackets."""

    @qm.qkernel
    def circuit() -> qm.Bit:
        """Apply an open-controlled X over an empty target register."""
        control = qm.qubit("control")
        targets = qm.qubit_array(0, "targets")
        control, targets = qm.control(qm.x, control_value=0)(control, targets)
        return qm.measure(control)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 0
    assert estimate.gates.single_qubit == 0
    assert estimate.depth.gate_depth == 0
    assert estimate.measurements.total == 1
    assert estimate.width.allocated_qubits == 1
    assert estimate.width.peak_qubits == 1


def test_empty_broadcast_drops_private_body_workspace() -> None:
    """No target elements means private scalar-body workspace never exists."""

    @qm.qkernel
    def scalar_body(target: qm.Qubit) -> qm.Qubit:
        """Use one private workspace qubit before updating the target."""
        workspace = qm.qubit("workspace")
        workspace = qm.h(workspace)
        return qm.x(target)

    @qm.qkernel
    def circuit() -> qm.Bit:
        """Broadcast the workspace body over an empty target register."""
        control = qm.qubit("control")
        targets = qm.qubit_array(0, "targets")
        control, targets = qm.control(scalar_body, control_value=0)(
            control,
            targets,
        )
        return qm.measure(control)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 0
    assert estimate.depth.gate_depth == 0
    assert estimate.width.allocated_qubits == 1
    assert estimate.width.peak_qubits == 1
    assert estimate.width.circuit_qubits == 1


def test_outer_control_ignores_nested_empty_broadcast() -> None:
    """An empty nested broadcast cannot trigger a shared outer control ladder."""

    @qm.qkernel
    def inner(
        local_control: qm.Qubit,
        targets: qm.Vector[qm.Qubit],
    ) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
        """Apply an open-controlled X to every target element."""
        return qm.control(qm.x, control_value=0)(local_control, targets)

    @qm.qkernel
    def circuit() -> tuple[qm.Vector[qm.Qubit], qm.Qubit, qm.Vector[qm.Qubit]]:
        """Place the empty broadcast under three additional controls."""
        outer_controls = qm.qubit_array(3, "outer_controls")
        local_control = qm.qubit("local_control")
        targets = qm.qubit_array(0, "targets")
        *_, local_control, targets = qm.control(inner, num_controls=3)(
            outer_controls,
            local_control,
            targets,
        )
        return outer_controls, local_control, targets

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 0
    assert estimate.gates.toffoli == 0
    assert estimate.depth.depth == 0
    assert estimate.width.clean_ancilla_qubits == 0
    assert estimate.width.allocated_qubits == 4
    assert estimate.width.peak_qubits == 4


def test_symbolic_nested_broadcast_matches_direct_specialization() -> None:
    """A symbolic nested broadcast guards its shared outer ladder by activity."""

    @qm.qkernel
    def inner(
        local_control: qm.Qubit,
        targets: qm.Vector[qm.Qubit],
    ) -> tuple[qm.Qubit, qm.Vector[qm.Qubit]]:
        """Apply a controlled X to every target element."""
        return qm.control(qm.x)(local_control, targets)

    @qm.qkernel
    def circuit(
        width: qm.UInt,
    ) -> tuple[qm.Vector[qm.Qubit], qm.Qubit, qm.Vector[qm.Qubit]]:
        """Place the symbolic broadcast under three additional controls."""
        outer_controls = qm.qubit_array(3, "outer_controls")
        local_control = qm.qubit("local_control")
        targets = qm.qubit_array(width, "targets")
        *_, local_control, targets = qm.control(inner, num_controls=3)(
            outer_controls,
            local_control,
            targets,
        )
        return outer_controls, local_control, targets

    symbolic = circuit.estimate_resources()
    for width in (0, 1, 2):
        specialized = symbolic.substitute(width=width)
        direct = circuit.estimate_resources(inputs={"width": width})
        assert specialized.gates == direct.gates
        assert specialized.depth == direct.depth
        assert specialized.width == direct.width

    empty = symbolic.substitute(width=0)
    assert empty.gates.total == 0
    assert empty.width.clean_ancilla_qubits == 0
    active = symbolic.substitute(width=2)
    assert active.gates.total == 6
    assert active.width.clean_ancilla_qubits == 2


def test_controlled_scalar_qkernel_with_parameter_broadcasts_over_vector() -> None:
    """Classical body arguments do not hide a vector target broadcast."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Qubit]:
        """Broadcast one controlled rotation over three targets."""
        control = qm.qubit("control")
        targets = qm.qubit_array(3, "targets")
        control, targets = qm.control(_scalar_ry_body)(
            control,
            targets,
            qm.float_(0.25),
        )
        return targets

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.two_qubit == 3
    assert estimate.depth.depth == 3


def test_shared_outer_control_serializes_independent_targets() -> None:
    """Independent target gates cannot share a layer when controls are shared."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Control two otherwise independent Hadamard gates."""
        control = qm.qubit("control")
        left = qm.qubit("left")
        right = qm.qubit("right")
        _, left, right = qm.control(_independent_h_body)(control, left, right)
        return left, right

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 2
    assert estimate.gates.two_qubit == 2
    assert estimate.depth.depth == 2


def test_controlled_inverse_preserves_static_control_flow_resources() -> None:
    """Control, inverse, loops, and compile-time branches compose exactly."""

    @qm.qkernel
    def conditional_body(target: qm.Qubit, flag: qm.UInt) -> qm.Qubit:
        """Apply a loop followed by a compile-time-selected branch."""
        for _index in qm.range(2):
            target = qm.h(target)
        if flag:
            target = qm.x(target)
        else:
            target = qm.z(target)
            target = qm.z(target)
        return target

    @qm.qkernel
    def loop_body(target: qm.Qubit) -> qm.Qubit:
        """Apply a statically bounded loop and one final gate."""
        for _index in qm.range(2):
            target = qm.h(target)
        return qm.x(target)

    @qm.qkernel
    def controlled_loop_body(
        control_0: qm.Qubit,
        control_1: qm.Qubit,
        control_2: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply the loop body under three controls."""
        return qm.control(loop_body, num_controls=3)(
            control_0,
            control_1,
            control_2,
            target,
        )

    @qm.qkernel
    def inverse_circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Invert the controlled static-loop body."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        control_2 = qm.qubit("control_2")
        target = qm.qubit("target")
        return qm.inverse(controlled_loop_body)(
            control_0,
            control_1,
            control_2,
            target,
        )

    @qm.qkernel
    def branch_circuit(flag: qm.UInt) -> qm.Qubit:
        """Control a compile-time-selected branch body."""
        controls = qm.qubit_array(3, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(conditional_body, num_controls=3)(
            controls,
            target,
            flag,
        )
        return target

    inverse_estimate = inverse_circuit.estimate_resources()
    symbolic_branch = branch_circuit.estimate_resources()
    true_estimate = branch_circuit.estimate_resources(inputs={"flag": 1})
    false_estimate = branch_circuit.estimate_resources(inputs={"flag": 0})
    assert symbolic_branch.substitute(flag=1).gates == true_estimate.gates
    assert symbolic_branch.substitute(flag=0).gates == false_estimate.gates

    assert inverse_estimate.gates.total == 7
    assert inverse_estimate.gates.two_qubit == 3
    assert inverse_estimate.gates.multi_qubit == 4
    assert inverse_estimate.gates.toffoli == 4
    assert true_estimate.gates.total == 7
    assert true_estimate.gates.two_qubit == 3
    assert true_estimate.gates.multi_qubit == 4
    assert true_estimate.gates.toffoli == 4
    assert false_estimate.gates.total == 8
    assert false_estimate.gates.two_qubit == 4
    assert false_estimate.gates.multi_qubit == 4
    assert false_estimate.gates.toffoli == 4
    assert true_estimate.width.clean_ancilla_qubits == 2
    assert false_estimate.width.clean_ancilla_qubits == 2
    assert inverse_estimate.width.clean_ancilla_qubits == 2


def _controlled_phase_estimate(
    phase: float,
    *,
    num_controls: int = 3,
    control_value: int | None = None,
    basis: qm.GateBasis = qm.GateBasis.LOGICAL,
    precision: float = 1e-10,
) -> qm.ResourceEstimate:
    """Return the estimate for a pure phase under three controls.

    Args:
        phase (float): Global phase in radians.
        num_controls (int): Number of coherent controls. Defaults to three.
        control_value (int | None): Optional activation pattern for the
            controls. Defaults to the all-ones pattern.
        basis (qm.GateBasis): Requested resource basis. Defaults to logical.
        precision (float): Clifford+T rotation precision. Defaults to
            ``1e-10``.

    Returns:
        qm.ResourceEstimate: Controlled phase estimate.
    """

    @qm.qkernel
    def phased(target: qm.Qubit) -> qm.Qubit:
        """Apply a captured phase to an identity body."""
        return qm.global_phase(_identity_body, phase)(target)

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Apply the phased identity under three controls."""
        controls = qm.qubit_array(num_controls, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(
            phased,
            num_controls=num_controls,
            control_value=control_value,
        )(controls, target)
        return target

    return circuit.estimate_resources(basis=basis, precision=precision)


def test_controlled_global_phase_uses_clean_ancilla_phase_lowering() -> None:
    """Controlled global phases retain angle and control-arity semantics."""
    zero = _controlled_phase_estimate(0.0)
    pauli_z = _controlled_phase_estimate(math.pi)
    arbitrary = _controlled_phase_estimate(0.3)

    assert zero.gates.total == 0

    assert pauli_z.gates.total == 3
    assert pauli_z.gates.two_qubit == 1
    assert pauli_z.gates.multi_qubit == 2
    assert pauli_z.gates.toffoli == 2
    assert pauli_z.gates.rotation == 1
    assert pauli_z.width.clean_ancilla_qubits == 1

    assert arbitrary.gates.total == 3
    assert arbitrary.gates.two_qubit == 1
    assert arbitrary.gates.multi_qubit == 2
    assert arbitrary.gates.toffoli == 2
    assert arbitrary.gates.rotation == 1
    assert arbitrary.width.clean_ancilla_qubits == 1


def test_nonzero_tiny_global_phase_is_not_erased() -> None:
    """Clean-ancilla estimation preserves phases below the old fixed tolerance."""
    estimate = _controlled_phase_estimate(5e-13)

    assert estimate.gates.total > 0
    assert estimate.gates.rotation == 1


@pytest.mark.parametrize("num_controls", [2, 3])
@pytest.mark.parametrize("phase", [0.0, math.tau, -math.tau])
def test_controlled_identity_phase_has_no_ladder_or_open_brackets(
    phase: float,
    num_controls: int,
) -> None:
    """Identity phases omit phase gates, shared ladders, and open brackets."""
    estimate = _controlled_phase_estimate(
        phase,
        num_controls=num_controls,
        control_value=0,
    )

    assert estimate.gates.total == 0
    assert estimate.depth.depth == 0
    assert estimate.width.clean_ancilla_qubits == 0


def test_symbolic_open_controlled_phase_specializes_before_batching() -> None:
    """Phase binding selects identity or nonzero open-control resources."""

    @qm.qkernel
    def circuit(theta: qm.Float) -> qm.Qubit:
        """Apply a symbolic phase under three zero-valued controls."""
        controls = qm.qubit_array(3, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(
            _identity_body,
            num_controls=3,
            control_value=0,
        )(controls, target, global_phase=theta)
        return target

    symbolic = circuit.estimate_resources()
    zero = symbolic.substitute(theta=0.0)
    full_turn = symbolic.substitute(theta=math.tau)
    tiny = symbolic.substitute(theta=5e-13)
    direct_zero = circuit.estimate_resources(inputs={"theta": 0.0})
    direct_full_turn = circuit.estimate_resources(inputs={"theta": math.tau})
    direct_tiny = circuit.estimate_resources(inputs={"theta": 5e-13})

    assert zero.gates.total == 0
    assert zero.width.clean_ancilla_qubits == 0
    assert zero.quality is direct_zero.quality is qm.EstimateQuality.EXACT
    assert zero.assumptions == direct_zero.assumptions == ()
    assert full_turn.gates.total == 0
    assert full_turn.width.clean_ancilla_qubits == 0
    assert full_turn.quality is direct_full_turn.quality is qm.EstimateQuality.EXACT
    assert full_turn.assumptions == direct_full_turn.assumptions == ()
    assert direct_zero.gates.total == 0
    assert direct_full_turn.gates.total == 0
    assert tiny.gates.total > 0
    assert tiny.gates.rotation == 1
    assert tiny.quality is direct_tiny.quality
    assert tiny.assumptions == direct_tiny.assumptions


def test_bodyless_controlled_calls_do_not_create_a_shared_ladder() -> None:
    """OPAQUE_CALL records unknown calls without inventing control gates."""
    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.estimator.resource_estimator import (
        ResourceInterpreter,
        _ResourceEstimatorConfig,
    )

    control = Value(type=QubitType(), name="control")
    target = Value(type=QubitType(), name="target")
    first_control = control.next_version()
    first_target = target.next_version()
    first = ConcreteControlledU(
        operands=[control, target],
        results=[first_control, first_target],
        num_controls=1,
        block=None,
    )
    second = ConcreteControlledU(
        operands=[first_control, first_target],
        results=[first_control.next_version(), first_target.next_version()],
        num_controls=1,
        block=None,
    )
    block = Block(
        input_values=[control, target],
        operations=[first, second],
    )
    interpreter = ResourceInterpreter(
        config=_ResourceEstimatorConfig(
            unknown_policy=qm.UnknownResourcePolicy.OPAQUE_CALL,
        ),
        bindings={},
    )

    estimate = interpreter.eval_operations(
        block.operations,
        ExprResolver(block),
        controls=3,
    )

    assert estimate.gates.total == 0
    assert estimate.width.clean_ancilla_qubits == 0
    assert estimate.calls.calls_by_name == {"controlled_u": 2}
    assert estimate.calls.queries_by_name == {"controlled_u": 2}


@pytest.mark.parametrize(
    ("operation", "callable_kind"),
    [
        pytest.param(
            ConcreteControlledU(
                operands=[
                    Value(type=QubitType(), name="control"),
                    Value(type=QubitType(), name="target"),
                ],
                results=[
                    Value(type=QubitType(), name="control_result"),
                    Value(type=QubitType(), name="target_result"),
                ],
                num_controls=1,
                block=None,
            ),
            "controlled callable",
            id="controlled",
        ),
        pytest.param(
            InverseBlockOperation(
                operands=[Value(type=QubitType(), name="target")],
                results=[Value(type=QubitType(), name="target_result")],
                num_target_qubits=1,
                custom_name="bodyless_inverse",
            ),
            "inverse callable",
            id="inverse",
        ),
    ],
)
def test_bodyless_transforms_honor_fail_closed_and_warning_policies(
    operation: Operation,
    callable_kind: str,
) -> None:
    """Bodyless transforms fail by default and honor the warning policy."""
    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.estimator.resource_estimator import (
        ResourceInterpreter,
        _ResourceEstimatorConfig,
    )

    block = Block(operations=[operation])
    with pytest.raises(ValueError, match=rf"resources for {callable_kind}"):
        ResourceInterpreter(
            config=_ResourceEstimatorConfig(),
            bindings={},
        ).eval_operations(block.operations, ExprResolver(block))

    warning = ResourceInterpreter(
        config=_ResourceEstimatorConfig(
            unknown_policy=qm.UnknownResourcePolicy.ZERO_WITH_WARNING,
        ),
        bindings={},
    ).eval_operations(block.operations, ExprResolver(block))

    assert warning.gates.total == 0
    assert warning.derivation is qm.EstimateDerivation.MODELED
    assert any("no implementation body" in item.message for item in warning.assumptions)


@pytest.mark.parametrize(
    "operation",
    [
        ForItemsOperation(),
        WhileOperation(),
    ],
)
def test_unsupported_controlled_loops_fail_closed(
    operation: Operation,
) -> None:
    """Unsupported loop forms fail instead of selecting a control recipe."""
    from qamomile.circuit.estimator._resolver import ExprResolver
    from qamomile.circuit.estimator.resource_estimator import (
        ResourceInterpreter,
        _ResourceEstimatorConfig,
    )

    block = Block(operations=[operation])
    interpreter = ResourceInterpreter(
        config=_ResourceEstimatorConfig(),
        bindings={},
    )

    with pytest.raises(
        ValueError,
        match=rf"Cannot estimate controlled {type(operation).__name__}",
    ):
        interpreter.eval_operations(
            block.operations,
            ExprResolver(block),
            controls=2,
        )


def test_clifford_t_cz_matches_controlled_z_lowering() -> None:
    """Equivalent CZ and CCZ constructions use the same canonical counts."""

    @qm.qkernel
    def z_body(target: qm.Qubit) -> qm.Qubit:
        """Apply one Pauli-Z gate."""
        return qm.z(target)

    @qm.qkernel
    def cz_body(left: qm.Qubit, right: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one controlled-Z gate."""
        return qm.cz(left, right)

    @qm.qkernel
    def direct_cz() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply CZ as a direct two-qubit primitive."""
        return qm.cz(qm.qubit("left"), qm.qubit("right"))

    @qm.qkernel
    def controlled_z() -> tuple[qm.Qubit, qm.Qubit]:
        """Construct CZ by controlling a one-gate Z body."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        return qm.control(z_body)(control, target)

    @qm.qkernel
    def controlled_cz() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Construct CCZ by controlling a CZ body."""
        control = qm.qubit("control")
        left = qm.qubit("left")
        right = qm.qubit("right")
        return qm.control(cz_body)(control, left, right)

    @qm.qkernel
    def doubly_controlled_z() -> qm.Qubit:
        """Construct CCZ by controlling Z with two qubits."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(z_body, num_controls=2)(controls, target)
        return target

    direct = direct_cz.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)
    equivalent = controlled_z.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)
    direct_ccz = controlled_cz.estimate_resources(basis=qm.GateBasis.CLIFFORD_T)
    equivalent_ccz = doubly_controlled_z.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T
    )

    assert (
        direct.gates
        == equivalent.gates
        == qm.GateResources(
            total=3,
            single_qubit=2,
            two_qubit=1,
            clifford=3,
        )
    )
    assert direct.depth.depth == equivalent.depth.depth == 3
    assert direct_ccz.gates == equivalent_ccz.gates
    assert direct_ccz.gates.total == 17


def test_logical_exact_control_wrappers_preserve_recipe_guarantee() -> None:
    """Exact Y, Z, SWAP, and RZZ wrappers stay exact without clean ancillas."""

    @qm.qkernel
    def y_body(target: qm.Qubit) -> qm.Qubit:
        """Apply one Pauli-Y gate."""
        return qm.y(target)

    @qm.qkernel
    def z_body(target: qm.Qubit) -> qm.Qubit:
        """Apply one Pauli-Z gate."""
        return qm.z(target)

    @qm.qkernel
    def swap_body(
        left: qm.Qubit,
        right: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one SWAP gate."""
        return qm.swap(left, right)

    @qm.qkernel
    def rzz_body(
        left: qm.Qubit,
        right: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one fixed-angle RZZ gate."""
        return qm.rzz(left, right, 0.25)

    @qm.qkernel
    def doubly_controlled_y() -> qm.Qubit:
        """Apply a Y gate under two coherent controls."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(y_body, num_controls=2)(controls, target)
        return target

    @qm.qkernel
    def doubly_controlled_z() -> qm.Qubit:
        """Apply a Z gate under two coherent controls."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(z_body, num_controls=2)(controls, target)
        return target

    @qm.qkernel
    def controlled_swap() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply a SWAP gate under one coherent control."""
        control = qm.qubit("control")
        left = qm.qubit("left")
        right = qm.qubit("right")
        _, left, right = qm.control(swap_body)(control, left, right)
        return left, right

    @qm.qkernel
    def controlled_rzz() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply an RZZ gate under one coherent control."""
        control = qm.qubit("control")
        left = qm.qubit("left")
        right = qm.qubit("right")
        _, left, right = qm.control(rzz_body)(control, left, right)
        return left, right

    logical_y = doubly_controlled_y.estimate_resources()
    clifford_t_y = doubly_controlled_y.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T,
    )
    logical_z = doubly_controlled_z.estimate_resources()
    logical_swap = controlled_swap.estimate_resources()
    logical_rzz = controlled_rzz.estimate_resources()

    assert logical_y.gates.total == 3
    assert logical_y.gates.toffoli == 1
    assert logical_y.width.clean_ancilla_qubits == 0
    assert logical_y.quality is qm.EstimateQuality.EXACT
    assert clifford_t_y.width.clean_ancilla_qubits == 0
    assert clifford_t_y.quality is qm.EstimateQuality.EXACT
    for estimate in (logical_z, logical_swap, logical_rzz):
        assert estimate.width.clean_ancilla_qubits == 0
        assert estimate.quality is qm.EstimateQuality.EXACT


def test_clifford_t_supports_multi_controlled_global_phase() -> None:
    """Fixed and arbitrary phases use defined multi-control lowerings."""
    fixed = _controlled_phase_estimate(
        math.pi,
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-3,
    )
    arbitrary = _controlled_phase_estimate(
        0.3,
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-3,
    )

    assert fixed.gates.t > 0
    assert fixed.quality is qm.EstimateQuality.EXACT
    assert fixed.approximation is qm.ApproximationStatus.EXACT
    assert arbitrary.gates.t > fixed.gates.t
    assert arbitrary.width.clean_ancilla_qubits == 1
    assert arbitrary.quality is qm.EstimateQuality.UNKNOWN
    assert arbitrary.approximation is qm.ApproximationStatus.APPROXIMATE


def test_symbolic_controlled_global_phase_retains_angle_classification() -> None:
    """Later phase substitution selects identity and Clifford+T resources."""

    @qm.qkernel
    def circuit(theta: qm.Float) -> qm.Qubit:
        """Apply a parameterized global phase under one coherent control."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        _, target = qm.control(_identity_body)(
            control,
            target,
            global_phase=theta,
        )
        return target

    clean_estimate = circuit.estimate_resources()
    assert set(clean_estimate.parameters) == {"theta"}
    assert clean_estimate.substitute(theta=0).gates.total == 0
    assert clean_estimate.substitute(theta=math.pi).gates.rotation == 1

    clifford_t = circuit.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-3,
    )
    identity = clifford_t.substitute(theta=0)
    pauli_z = clifford_t.substitute(theta=math.pi)
    phase_s = clifford_t.substitute(theta=math.pi / 2)
    phase_t = clifford_t.substitute(theta=math.pi / 4)
    arbitrary = clifford_t.substitute(theta=0.3)

    assert identity.gates.total == 0
    assert pauli_z.gates.clifford == 1
    assert phase_s.gates.clifford == 1
    assert phase_t.gates.t == 1
    assert arbitrary.gates.t > 1
    assert identity.quality is qm.EstimateQuality.EXACT
    assert pauli_z.quality is qm.EstimateQuality.EXACT
    assert phase_s.quality is qm.EstimateQuality.EXACT
    assert phase_t.quality is qm.EstimateQuality.EXACT
    assert arbitrary.quality is qm.EstimateQuality.UNKNOWN
    assert identity.approximation is qm.ApproximationStatus.EXACT
    assert pauli_z.approximation is qm.ApproximationStatus.EXACT
    assert phase_s.approximation is qm.ApproximationStatus.EXACT
    assert phase_t.approximation is qm.ApproximationStatus.EXACT
    assert arbitrary.approximation is qm.ApproximationStatus.APPROXIMATE


def _controlled_pauli_estimate(num_controls: int) -> qm.ResourceEstimate:
    """Return a controlled Pauli-evolution estimate for ``X + 2 I``."""

    @qm.qkernel
    def circuit(
        hamiltonian: qm.Observable,
    ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Control a renamed Pauli-evolution body."""
        controls = qm.qubit_array(num_controls, "controls")
        target = qm.qubit_array(1, "target")
        *_, target = qm.control(
            _renamed_pauli_evolution,
            num_controls=num_controls,
        )(
            controls,
            target,
            hamiltonian,
            qm.float_(0.25),
        )
        return controls, target

    return circuit.estimate_resources(inputs={"hamiltonian": qm_o.X(0) + 2.0})


def test_controlled_pauli_evolve_resolves_binding_and_constant() -> None:
    """Pauli controls affect the axial rotation and identity-term phase."""
    single = _controlled_pauli_estimate(1)
    triple = _controlled_pauli_estimate(3)

    assert single.gates.total == 4
    assert single.gates.single_qubit == 3
    assert single.gates.two_qubit == 1
    assert single.gates.rotation == 2
    assert single.gates.clifford == 2
    assert single.width.clean_ancilla_qubits == 0

    assert triple.gates.total == 8
    assert triple.gates.single_qubit == 3
    assert triple.gates.two_qubit == 1
    assert triple.gates.multi_qubit == 4
    assert triple.gates.toffoli == 4
    assert triple.gates.rotation == 2
    assert triple.width.clean_ancilla_qubits == 2
    assert not any(
        assumption.source == "PauliEvolveOp"
        for estimate in (single, triple)
        for assumption in estimate.assumptions
    )


def test_noncommuting_pauli_evolve_reports_trotter_assumption() -> None:
    """Noncommuting sums distinguish circuit-count exactness from simulation."""
    symbolic = _renamed_pauli_evolution.estimate_resources(
        inputs={
            "register": 1,
            "observable": qm_o.X(0) + qm_o.Z(0),
        },
        trace=True,
    )
    active = symbolic.substitute(time=0.25)
    zero_time = symbolic.substitute(time=0.0)
    expected_message = (
        "Noncommuting Pauli terms are counted as one first-order Lie-Trotter "
        "product-formula step in Hamiltonian term order."
    )

    assert active.gates.total == 4
    assert active.quality is qm.EstimateQuality.EXACT
    assert active.approximation is qm.ApproximationStatus.APPROXIMATE
    assert active.to_dict()["approximation"] == "approximate"
    assert [
        assumption.message
        for assumption in active.assumptions
        if assumption.source == "PauliEvolveOp"
    ] == [expected_message]
    assert expected_message in active.explain()
    assert not any(
        assumption.source == "PauliEvolveOp" for assumption in zero_time.assumptions
    )
    assert zero_time.approximation is qm.ApproximationStatus.EXACT
    assert expected_message not in zero_time.explain()


def test_commuting_pauli_evolve_needs_no_trotter_assumption() -> None:
    """Pairwise-commuting Pauli sums are exact without a product-formula caveat."""
    commuting = qm_o.X(0) * qm_o.X(1) + qm_o.Y(0) * qm_o.Y(1) + qm_o.Z(0) * qm_o.Z(1)
    estimate = _renamed_pauli_evolution.estimate_resources(
        inputs={
            "register": 2,
            "observable": commuting,
            "time": 0.25,
        }
    )

    assert estimate.quality is qm.EstimateQuality.EXACT
    assert estimate.approximation is qm.ApproximationStatus.EXACT
    assert not any(
        assumption.source == "PauliEvolveOp" for assumption in estimate.assumptions
    )


def test_controlled_noncommuting_pauli_evolve_keeps_trotter_assumption() -> None:
    """Coherent control preserves the noncommuting product-formula caveat."""

    @qm.qkernel
    def controlled(
        observable: qm.Observable,
    ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Apply a noncommuting evolution under two coherent controls."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit_array(1, "target")
        *_, target = qm.control(
            _renamed_pauli_evolution,
            num_controls=2,
        )(
            controls,
            target,
            observable,
            qm.float_(0.25),
        )
        return controls, target

    estimate = controlled.estimate_resources(
        inputs={"observable": qm_o.X(0) + qm_o.Z(0)}
    )

    assert any(
        "one first-order Lie-Trotter product-formula step" in assumption.message
        for assumption in estimate.assumptions
        if assumption.source == "PauliEvolveOp"
    )
    assert estimate.approximation is qm.ApproximationStatus.APPROXIMATE


def test_pauli_evolve_zero_time_specialization_removes_all_resources() -> None:
    """Zero time removes both the Pauli gadget and its shared control ladder."""
    base_inputs = {"register": 1, "observable": qm_o.Z(0)}
    symbolic = _renamed_pauli_evolution.estimate_resources(inputs=base_inputs)

    assert symbolic.substitute(time=0.0).gates.total == 0
    assert symbolic.substitute(time=0.0).depth.depth == 0
    assert symbolic.substitute(time=0.25).gates.total == 1
    direct_zero = _renamed_pauli_evolution.estimate_resources(
        inputs={**base_inputs, "time": 0.0}
    )
    assert direct_zero.gates.total == 0
    assert direct_zero.depth.depth == 0
    assert not any(
        "time" in assumption.message for assumption in direct_zero.assumptions
    )

    @qm.qkernel
    def controlled(
        time: qm.Float,
        observable: qm.Observable,
    ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Apply one symbolic-time evolution under three controls."""
        controls = qm.qubit_array(3, "controls")
        target = qm.qubit_array(1, "target")
        *_, target = qm.control(
            _renamed_pauli_evolution,
            num_controls=3,
        )(
            controls,
            target,
            observable,
            time,
        )
        return controls, target

    controlled_symbolic = controlled.estimate_resources(
        inputs={"observable": qm_o.Z(0)}
    )
    controlled_zero = controlled_symbolic.substitute(time=0.0)
    controlled_active = controlled_symbolic.substitute(time=0.25)
    assert controlled_zero.gates.total == 0
    assert controlled_zero.gates.toffoli == 0
    assert controlled_zero.depth.depth == 0
    assert controlled_zero.width.clean_ancilla_qubits == 0
    assert controlled_zero.width.peak_qubits == 4
    assert controlled_active.gates.total > 0
    assert controlled_active.width.clean_ancilla_qubits == 2


def test_raw_ir_inputs_bind_hamiltonian_during_interpretation() -> None:
    """Block and operation-list inputs retain concrete object payloads."""
    inputs = {"observable": qm_o.Z(1)}
    expected = _raw_pauli_input_probe.estimate_resources(inputs=inputs)
    block_estimate = qm.estimate_resources(_raw_pauli_input_probe.block, inputs=inputs)
    operation_estimate = qm.estimate_resources(
        _raw_pauli_input_probe.block.operations,
        inputs=inputs,
    )

    assert block_estimate.gates == expected.gates
    assert operation_estimate.gates == expected.gates
    assert block_estimate.gates.total == 1
    assert block_estimate.width.peak_qubits == 2
    assert operation_estimate.width.peak_qubits == 2

    with pytest.raises(ValueError, match="input names.*observabel"):
        qm.estimate_resources(
            _raw_pauli_input_probe.block.operations,
            inputs={**inputs, "observabel": qm_o.Z(0)},
        )


def test_raw_block_inputs_specialize_array_width_but_reject_scalar_qubits() -> None:
    """Raw Blocks accept register widths without treating qubits as numbers."""
    estimate = qm.estimate_resources(
        _raw_width_probe.block,
        inputs={"register": 3},
    )

    assert estimate.width.input_qubits == 3
    assert estimate.width.peak_qubits == 3
    assert estimate.parameters == {}

    with pytest.raises(ValueError, match="requires an integer width"):
        qm.estimate_resources(
            _raw_width_probe.block,
            inputs={"register": 1.5},
        )
    with pytest.raises(ValueError, match="neither free symbols"):
        qm.estimate_resources(_scalar_h_body.block, inputs={"target": 0})
    with pytest.raises(ValueError, match="neither free symbols"):
        qm.estimate_resources(_scalar_h_body.block.operations, inputs={"target": 0})


def test_pauli_evolve_applies_unknown_policy_and_register_requirement() -> None:
    """Unbound Hamiltonians follow policy and bound widths remain validated."""

    @qm.qkernel
    def circuit(
        width: qm.UInt,
        hamiltonian: qm.Observable,
    ) -> qm.Vector[qm.Qubit]:
        """Evolve a symbolic-width register with a supplied Hamiltonian."""
        register = qm.qubit_array(width, "register")
        return qm.pauli_evolve(register, hamiltonian, qm.float_(0.25))

    with pytest.raises(ValueError, match="without a bound Hamiltonian"):
        circuit.estimate_resources()

    opaque = circuit.estimate_resources(
        unknown_policy=qm.UnknownResourcePolicy.OPAQUE_CALL
    )
    zero = circuit.estimate_resources(
        unknown_policy=qm.UnknownResourcePolicy.ZERO_WITH_WARNING
    )
    assert opaque.calls.calls_by_name == {"pauli_evolve": 1}
    assert opaque.calls.queries_by_name == {"pauli_evolve": 1}
    assert opaque.derivation is qm.EstimateDerivation.MODELED
    assert zero.calls.calls_by_name == {}
    assert zero.gates.total == 0
    assert zero.assumptions

    bound = circuit.estimate_resources(inputs={"hamiltonian": qm_o.Z(1)})
    with pytest.raises(ValueError, match="at least 2 qubits"):
        bound.substitute(width=1)
    with pytest.raises(ValueError, match="at least 2 qubits"):
        circuit.estimate_resources(inputs={"width": 1, "hamiltonian": qm_o.Z(1)})
    assert bound.substitute(width=3).gates.total == 1


def test_fixed_nonunitary_opaque_cost_is_allowed_only_for_direct_calls() -> None:
    """Fixed measurement/reset costs fail closed under unitary transforms."""
    oracle = qm.opaque(
        "fixed_nonunitary_oracle",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            measurements=qm.MeasurementResources(total=2),
            resets=qm.ResetResources(total=1),
            depth=qm.DepthResources(
                depth=3,
                measurement_depth=2,
                reset_depth=1,
            ),
        ),
    )

    @qm.qkernel
    def layer(target: qm.Qubit) -> qm.Qubit:
        """Invoke the fixed-cost bodyless oracle."""
        (target,) = oracle(target)
        return target

    @qm.qkernel
    def direct() -> qm.Qubit:
        """Invoke the fixed cost without a coherent transform."""
        return layer(qm.qubit("target"))

    @qm.qkernel
    def controlled() -> tuple[qm.Qubit, qm.Qubit]:
        """Attempt to control the fixed nonunitary cost."""
        return qm.control(layer)(
            qm.qubit("control"),
            qm.qubit("target"),
        )

    @qm.qkernel
    def inverted() -> qm.Qubit:
        """Attempt to invert the fixed nonunitary cost."""
        return qm.inverse(layer)(qm.qubit("target"))

    direct_estimate = direct.estimate_resources()
    assert direct_estimate.gates.total == 0
    assert direct_estimate.measurements.total == 2
    assert direct_estimate.resets.total == 1
    assert direct_estimate.depth.depth == 3
    assert direct_estimate.depth.gate_depth == 0
    assert direct_estimate.depth.measurement_depth == 2
    assert direct_estimate.depth.reset_depth == 1

    with pytest.raises(ValueError, match="Move measurement and reset"):
        controlled.estimate_resources()
    with pytest.raises(ValueError, match="Move measurement and reset"):
        inverted.estimate_resources()


def test_callback_opaque_cost_cannot_bypass_nonunitary_transform_checks() -> None:
    """A callback follows the same base-cost transform rules as a fixed cost."""
    oracle = qm.opaque(
        "callback_nonunitary_oracle",
        num_qubits=1,
        cost=_NonunitaryOpaqueCallbackCost(),
    )

    @qm.qkernel
    def layer(target: qm.Qubit) -> qm.Qubit:
        """Invoke the context-aware bodyless oracle."""
        (target,) = oracle(target)
        return target

    @qm.qkernel
    def controlled() -> tuple[qm.Qubit, qm.Qubit]:
        """Control the callback-priced invocation."""
        return qm.control(layer)(
            qm.qubit("control"),
            qm.qubit("target"),
        )

    @qm.qkernel
    def inverted() -> qm.Qubit:
        """Invert the callback-priced invocation."""
        return qm.inverse(layer)(qm.qubit("target"))

    with pytest.raises(ValueError, match="Move measurement and reset"):
        controlled.estimate_resources()
    with pytest.raises(ValueError, match="Move measurement and reset"):
        inverted.estimate_resources()


@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(
            qm.ResourceEstimate(
                measurements=qm.MeasurementResources(total=1),
            ),
            id="fixed",
        ),
        pytest.param(_NonunitaryOpaqueCallbackCost(), id="callback"),
    ],
)
def test_declared_controlled_nonunitary_oracle_cost_fails_closed(
    cost: object,
) -> None:
    """Declared coherent controls and their inverse reject nonunitary costs."""
    oracle = qm.opaque(
        "declared_nonunitary_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=cost,
    )

    @qm.qkernel
    def layer(
        control: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Invoke the Oracle through its definition-declared control."""
        return oracle(target, controls=(control,))

    @qm.qkernel
    def direct() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply the declared-controlled Oracle directly."""
        return layer(
            qm.qubit("control"),
            qm.qubit("target"),
        )

    @qm.qkernel
    def inverted() -> tuple[qm.Qubit, qm.Qubit]:
        """Invert the declared-controlled Oracle layer."""
        return qm.inverse(layer)(
            qm.qubit("control"),
            qm.qubit("target"),
        )

    with pytest.raises(ValueError, match="coherently controlled Oracle"):
        direct.estimate_resources()
    with pytest.raises(ValueError, match="coherently controlled Oracle"):
        inverted.estimate_resources()


@pytest.mark.parametrize(
    ("num_controls", "expected_total", "expected_clean_ancillas"),
    [
        (1, 5, 1),
        (2, 7, 2),
        (4, 11, 4),
    ],
)
def test_controlled_fixed_opaque_cost_projects_complete_arity_profile(
    num_controls: int,
    expected_total: int,
    expected_clean_ancillas: int,
) -> None:
    """Complete opaque arity counts receive a controlled projection model."""
    oracle = qm.opaque(
        "arity_oracle",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(
                total=3,
                single_qubit=2,
                two_qubit=1,
            ),
            calls=qm.CallResources(
                calls_by_name={"arity_oracle": 1},
                queries_by_name={"arity_oracle": 1},
            ),
        ),
    )

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Apply the fixed-cost oracle under concrete coherent controls."""
        controls = [qm.qubit(f"control_{index}") for index in range(num_controls)]
        target = qm.qubit("target")
        *_, target = qm.control(
            oracle,
            num_controls=num_controls,
        )(*controls, target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == expected_total
    assert estimate.depth.depth == expected_total
    assert estimate.width.clean_ancilla_qubits == expected_clean_ancillas
    assert estimate.calls.calls_by_name == {"arity_oracle": 1}
    assert estimate.calls.queries_by_name == {"arity_oracle": 1}
    assert estimate.derivation is qm.EstimateDerivation.MODELED
    assert estimate.quality is qm.EstimateQuality.UNKNOWN
    assert any(
        "uses the aggregate clean-ancilla Toffoli batching model" in assumption.message
        for assumption in estimate.assumptions
    )
    assert any(
        "Arity fields are independent field-wise upper bounds and may not sum to total"
        in assumption.message
        for assumption in estimate.assumptions
    )


def test_exact_two_control_aggregate_uses_fixed_shared_recipe() -> None:
    """An aggregate uses the fixed sharing rule without gate names."""

    @qm.qkernel
    def two_cx(
        intrinsic_0: qm.Qubit,
        target_0: qm.Qubit,
        intrinsic_1: qm.Qubit,
        target_1: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply two CX gates whose names are known to the estimator."""
        intrinsic_0, target_0 = qm.cx(intrinsic_0, target_0)
        intrinsic_1, target_1 = qm.cx(intrinsic_1, target_1)
        return intrinsic_0, target_0, intrinsic_1, target_1

    @qm.qkernel
    def controlled_body() -> tuple[
        qm.Qubit,
        qm.Qubit,
        qm.Qubit,
        qm.Qubit,
    ]:
        """Apply the named body under exactly two surrounding controls."""
        outer_0 = qm.qubit("outer_0")
        outer_1 = qm.qubit("outer_1")
        intrinsic_0 = qm.qubit("intrinsic_0")
        target_0 = qm.qubit("target_0")
        intrinsic_1 = qm.qubit("intrinsic_1")
        target_1 = qm.qubit("target_1")
        *_, intrinsic_0, target_0, intrinsic_1, target_1 = qm.control(
            two_cx,
            num_controls=2,
        )(
            outer_0,
            outer_1,
            intrinsic_0,
            target_0,
            intrinsic_1,
            target_1,
        )
        return intrinsic_0, target_0, intrinsic_1, target_1

    named = controlled_body.estimate_resources()
    aggregate = qm.ResourceEstimate(
        gates=qm.GateResources(total=2, two_qubit=2),
    ).controlled(2)

    assert named.gates.total == 4
    assert named.depth.depth == 4
    assert named.width.clean_ancilla_qubits == 1
    assert aggregate.gates.total == 8
    assert aggregate.gates.total >= named.gates.total
    assert aggregate.depth.depth >= named.depth.depth
    assert aggregate.width.clean_ancilla_qubits >= named.width.clean_ancilla_qubits
    assert any(
        "fixed aggregate recipe may differ" in assumption.message
        for assumption in aggregate.assumptions
    )


def test_fixed_opaque_projection_matches_shared_qkernel_ladder_cost() -> None:
    """A multi-gate opaque profile shares the same outer ladder as a qkernel."""
    oracle = qm.opaque(
        "four_h_cost",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(
                total=4,
                single_qubit=4,
            ),
        ),
    )

    @qm.qkernel
    def opaque_circuit() -> qm.Qubit:
        """Apply the aggregate four-gate profile under three controls."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        control_2 = qm.qubit("control_2")
        target = qm.qubit("target")
        *_, target = qm.control(oracle, num_controls=3)(
            control_0,
            control_1,
            control_2,
            target,
        )
        return target

    @qm.qkernel
    def body_circuit() -> qm.Qubit:
        """Apply the equivalent body-backed qkernel under three controls."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        control_2 = qm.qubit("control_2")
        target = qm.qubit("target")
        *_, target = qm.control(_four_h_body, num_controls=3)(
            control_0,
            control_1,
            control_2,
            target,
        )
        return target

    opaque_estimate = opaque_circuit.estimate_resources()
    body_estimate = body_circuit.estimate_resources()

    assert opaque_estimate.gates.total == body_estimate.gates.total == 8
    assert opaque_estimate.depth.depth == body_estimate.depth.depth == 8
    assert (
        opaque_estimate.width.clean_ancilla_qubits
        == body_estimate.width.clean_ancilla_qubits
        == 2
    )


def test_open_control_brackets_wrap_projected_fixed_opaque_cost() -> None:
    """Open controls add X brackets around the projected opaque gate cost."""
    oracle = qm.opaque(
        "open_arity_oracle",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(
                total=3,
                single_qubit=2,
                two_qubit=1,
            ),
            calls=qm.CallResources(
                queries_by_name={"open_arity_oracle": 1},
            ),
        ),
    )

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Apply one two-control oracle with one zero-valued control."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        target = qm.qubit("target")
        *_, target = qm.control(
            oracle,
            num_controls=2,
            control_value=2,
        )(control_0, control_1, target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 9
    assert estimate.depth.depth == 9
    assert estimate.width.clean_ancilla_qubits == 2
    assert estimate.calls.queries_by_name == {"open_arity_oracle": 1}


def test_declared_open_control_brackets_base_oracle_cost() -> None:
    """A declared open control brackets an otherwise complete base cost."""
    oracle = qm.opaque(
        "declared_open_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(total=1, single_qubit=1),
            depth=qm.DepthResources(
                depth=1,
                clifford_depth=1,
                gate_depth=1,
            ),
        ),
    )

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Invoke one declared control on its zero state."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        return oracle(
            target,
            controls=(control,),
            control_value=0,
        )

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.single_qubit == 3
    assert estimate.depth.depth == 3
    assert estimate.width.clean_ancilla_qubits == 0


def test_added_open_control_brackets_only_external_partition() -> None:
    """An added open control brackets the externally controlled base cost."""
    oracle = qm.opaque(
        "added_open_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(total=1, single_qubit=1),
        ),
    )
    controlled = qm.control(oracle, num_controls=2, control_value=2)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Add two controls while leaving the declared control all-one."""
        added_0 = qm.qubit("added_0")
        added_1 = qm.qubit("added_1")
        declared = qm.qubit("declared")
        target = qm.qubit("target")
        return controlled(added_0, added_1, declared, target)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 5
    assert estimate.depth.depth == 5
    assert estimate.width.clean_ancilla_qubits == 1


def test_declared_and_added_open_controls_share_one_depth_bracket() -> None:
    """One Oracle invocation normalizes all local zero controls in parallel."""
    oracle = qm.opaque(
        "combined_open_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(total=1, single_qubit=1),
            depth=qm.DepthResources(
                depth=1,
                clifford_depth=1,
                gate_depth=1,
            ),
        ),
    )
    controlled = qm.control(oracle, num_controls=2)

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Trace one Oracle with two added and one declared control."""
        added_0 = qm.qubit("added_0")
        added_1 = qm.qubit("added_1")
        declared = qm.qubit("declared")
        target = qm.qubit("target")
        return controlled(added_0, added_1, declared, target)

    invoke = next(
        operation
        for operation in circuit.block.operations
        if isinstance(operation, InvokeOperation)
    )
    patterned_invoke = dataclasses.replace(
        invoke,
        attrs={
            **invoke.attrs,
            "control_value": 0b010,
        },
    )
    patterned_block = dataclasses.replace(
        circuit.block,
        operations=[
            patterned_invoke if operation is invoke else operation
            for operation in circuit.block.operations
        ],
    )

    estimate = qm.estimate_resources(patterned_block)

    assert estimate.gates.total == 7
    assert estimate.gates.single_qubit == 6
    assert estimate.gates.two_qubit == 1
    assert estimate.gates.multi_qubit == 2
    assert estimate.gates.toffoli == 2
    assert estimate.depth.depth == 5
    assert estimate.depth.clifford_depth == 4
    assert estimate.depth.toffoli_depth == 2
    assert estimate.width.clean_ancilla_qubits == 1
    assert estimate.width.allocated_qubits == 4
    assert estimate.width.peak_qubits == 5


def test_incomplete_opaque_arity_profile_projects_known_gate_counts() -> None:
    """Known arities are projected while an unknown remainder stays explicit."""
    oracle = qm.opaque(
        "incomplete_arity_oracle",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(
                total=5,
                single_qubit=2,
                two_qubit=1,
            ),
        ),
    )

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Control an opaque cost whose arity profile has two unknown gates."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        target = qm.qubit("target")
        *_, target = qm.control(
            oracle,
            num_controls=2,
        )(control_0, control_1, target)
        return target

    estimate = circuit.estimate_resources()
    complete = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=3,
            single_qubit=2,
            two_qubit=1,
        )
    ).controlled(2)

    assert estimate.gates.total == 9
    assert estimate.depth.depth == 9
    assert estimate.width.clean_ancilla_qubits == 2
    assert estimate.gates.single_qubit == complete.gates.single_qubit
    assert estimate.gates.two_qubit == complete.gates.two_qubit
    assert estimate.gates.multi_qubit == complete.gates.multi_qubit
    assert (
        estimate.gates.single_qubit
        + estimate.gates.two_qubit
        + estimate.gates.multi_qubit
        != estimate.gates.total
    )
    assert any(
        "2 gate(s) with unclassified arity" in assumption.message
        for assumption in estimate.assumptions
    )
    assert any(
        "Arity fields are independent field-wise upper bounds and may not sum to total"
        in assumption.message
        for assumption in estimate.assumptions
    )


def test_partial_opaque_projection_preserves_declared_multi_qubit_gates() -> None:
    """Known multi-qubit gates remain classified but are not decomposed."""
    partial = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=4,
            single_qubit=2,
            two_qubit=1,
            multi_qubit=1,
        )
    ).controlled(2)
    complete = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=3,
            single_qubit=2,
            two_qubit=1,
        )
    ).controlled(2)

    assert partial.gates.total == 8
    assert partial.depth.depth == 8
    assert partial.width.clean_ancilla_qubits == 2
    assert partial.gates.multi_qubit == complete.gates.multi_qubit + 1
    assert any(
        "0 gate(s) with unclassified arity" in assumption.message
        for assumption in partial.assumptions
    )


def test_partial_opaque_projection_preserves_declared_family_floors() -> None:
    """Unresolved gates do not silently erase declared family information."""
    estimate = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=2,
            single_qubit=1,
            toffoli=1,
            non_clifford=1,
        ),
        depth=qm.DepthResources(
            depth=2,
            toffoli_depth=1,
            non_clifford_depth=1,
        ),
    ).controlled(1)

    assert estimate.gates.total == 2
    assert estimate.gates.toffoli == 1
    assert estimate.gates.non_clifford >= 1
    assert estimate.depth.toffoli_depth == 1
    assert estimate.derivation is qm.EstimateDerivation.MODELED
    assert estimate.quality is qm.EstimateQuality.UNKNOWN
    assert any(
        "gate-family counts are retained only as field-wise floors"
        in assumption.message
        for assumption in estimate.assumptions
    )


def test_total_only_opaque_cost_is_not_presented_as_a_partial_projection() -> None:
    """A total without any supported arity bucket remains unchanged."""
    base = qm.ResourceEstimate(
        gates=qm.GateResources(total=5),
        depth=qm.DepthResources(depth=3),
    )

    estimate = base.controlled(2)

    assert estimate.gates == base.gates
    assert estimate.depth == base.depth
    assert estimate.derivation is qm.EstimateDerivation.MODELED
    assert estimate.quality is qm.EstimateQuality.UNKNOWN
    assert any(
        "no declared one- or two-qubit gate profile" in assumption.message
        for assumption in estimate.assumptions
    )


def test_opaque_control_projection_adds_to_declared_clean_workspace() -> None:
    """A one-operation profile keeps per-primitive recipe workspace."""
    estimate = qm.ResourceEstimate(
        width=qm.WidthResources(
            clean_ancilla_qubits=3,
            peak_qubits=3,
        ),
        gates=qm.GateResources(
            total=1,
            two_qubit=1,
        ),
    ).controlled(2)

    assert estimate.gates.total == 7
    assert estimate.width.clean_ancilla_qubits == 5
    assert estimate.width.peak_qubits == 5


@pytest.mark.parametrize(
    ("num_controls", "expected_total", "expected_clean_ancillas"),
    [
        (2, 7, 2),
        (3, 9, 3),
    ],
)
def test_single_opaque_primitive_does_not_trigger_shared_ladder(
    num_controls: int,
    expected_total: int,
    expected_clean_ancillas: int,
) -> None:
    """One modeled operation stays on the per-primitive decomposition."""
    estimate = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=1,
            two_qubit=1,
        ),
    ).controlled(num_controls)

    assert estimate.gates.total == expected_total
    assert estimate.depth.depth == expected_total
    assert estimate.width.clean_ancilla_qubits == expected_clean_ancillas


def test_symbolic_opaque_gate_count_uses_fixed_shared_threshold() -> None:
    """A symbolic aggregate selects sharing from controls and modeled work."""
    total = sp.Symbol("total", integer=True, nonnegative=True)
    controls = sp.Symbol("controls", integer=True, nonnegative=True)
    base = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=total,
            two_qubit=total,
        ),
    )
    estimate = base.controlled(controls)

    two_control_single = estimate.substitute(total=1, controls=2)
    two_control_choice = estimate.substitute(total=2, controls=2)
    three_control_single = estimate.substitute(total=1, controls=3)
    three_control_shared = estimate.substitute(total=2, controls=3)
    controlled_zero_profile = estimate.substitute(total=0, controls=2)
    zero_control = estimate.substitute(total=2, controls=0)

    assert two_control_single.gates.total == 7
    assert two_control_single.width.clean_ancilla_qubits == 2
    assert two_control_choice.gates.total == 8
    assert two_control_choice.width.clean_ancilla_qubits == 2
    assert three_control_single.gates.total == 9
    assert three_control_single.width.clean_ancilla_qubits == 3
    assert three_control_shared.gates.total == 10
    assert three_control_shared.width.clean_ancilla_qubits == 3
    assert controlled_zero_profile.gates.total == 0
    assert controlled_zero_profile.quality is qm.EstimateQuality.UNKNOWN
    assert zero_control.gates == qm.GateResources(total=2, two_qubit=2)


def test_symbolic_opaque_control_projection_preserves_zero_control_cost() -> None:
    """A symbolic zero-control branch remains the original aggregate cost."""
    controls = sp.Symbol("controls", integer=True, nonnegative=True)
    base = qm.ResourceEstimate(
        width=qm.WidthResources(clean_ancilla_qubits=1, peak_qubits=1),
        gates=qm.GateResources(
            total=3,
            single_qubit=2,
            two_qubit=1,
            clifford=2,
            rotation=1,
            non_clifford=1,
        ),
        depth=qm.DepthResources(
            depth=2,
            clifford_depth=1,
            rotation_depth=1,
            non_clifford_depth=1,
        ),
        calls=qm.CallResources(
            calls_by_name={"symbolic_arity_oracle": 1},
            queries_by_name={"symbolic_arity_oracle": 1},
        ),
        assumptions=(qm.ResourceAssumption("declared aggregate model"),),
        derivation=qm.EstimateDerivation.MODELED,
    )

    symbolic = base.controlled(controls)
    zero = symbolic.substitute(controls=0)

    assert zero.width == base.width
    assert zero.gates == base.gates
    assert zero.depth == base.depth
    assert zero.calls == base.calls
    assert zero.assumptions == base.assumptions
    assert zero.derivation is base.derivation
    assert zero.quality is base.quality
    assert symbolic.substitute(controls=2).gates.total == 7


def test_symbolic_opaque_arity_constraints_reject_invalid_specialization() -> None:
    """Deferred family bounds prevent a symbolic aggregate undercount."""
    gates = sp.Symbol("gates", integer=True, nonnegative=True)
    rotations = sp.Symbol("rotations", integer=True, nonnegative=True)
    controls = sp.Symbol("controls", integer=True, nonnegative=True)
    estimate = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=gates,
            single_qubit=gates,
            rotation=rotations,
        )
    ).controlled(controls)

    valid = estimate.substitute(gates=2, rotations=1, controls=2)

    assert valid.gates.total == 4
    assert "rotations" in estimate.parameters
    assert any(
        requirement["label"]
        == "Clean-ancilla aggregate rotation count within total gate count"
        for requirement in estimate.to_dict()["requirements"]
    )
    with pytest.raises(ValueError, match="rotation count within total"):
        estimate.substitute(gates=1, rotations=2, controls=2)
    assert estimate.substitute(gates=1, rotations=2, controls=0).gates.rotation == 2


def test_symbolic_partial_arity_remainder_is_validated_only_under_control() -> None:
    """A symbolic unclassified remainder is constrained on controlled branches."""
    total = sp.Symbol("total", integer=True, nonnegative=True)
    single = sp.Symbol("single", integer=True, nonnegative=True)
    controls = sp.Symbol("controls", integer=True, nonnegative=True)
    base = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=total,
            single_qubit=single,
        )
    )

    estimate = base.controlled(controls)
    valid = estimate.substitute(total=5, single=2, controls=2)

    assert valid.gates.total == 7
    assert valid.depth.depth == 7
    assert any(
        requirement["label"] == "Clean-ancilla aggregate unclassified arity remainder"
        for requirement in estimate.to_dict()["requirements"]
    )
    with pytest.raises(ValueError, match="unclassified arity remainder"):
        estimate.substitute(total=1, single=2, controls=2)
    zero = estimate.substitute(total=1, single=2, controls=0)
    assert zero.gates.total == 1
    assert zero.gates.single_qubit == 2


def test_symbolic_zero_known_arity_uses_total_only_control_branch() -> None:
    """Specializing known arities to zero matches direct total-only modeling."""
    single = sp.Symbol("single", integer=True, nonnegative=True)
    controls = sp.Symbol("controls", integer=True, nonnegative=True)
    base = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=5,
            single_qubit=single,
        ),
        depth=qm.DepthResources(depth=1),
    )

    symbolic = base.controlled(controls)
    zero = symbolic.substitute(single=0, controls=2)
    direct = qm.ResourceEstimate(
        gates=qm.GateResources(total=5),
        depth=qm.DepthResources(depth=1),
    ).controlled(2)
    zero_controls = symbolic.substitute(single=0, controls=0)
    positive = symbolic.substitute(single=1, controls=2)
    complete = symbolic.substitute(single=5, controls=2)

    assert zero.gates == direct.gates
    assert zero.depth == direct.depth
    assert zero.derivation is direct.derivation
    assert zero.quality is direct.quality
    assert any(
        "no declared one- or two-qubit gate profile" in assumption.message
        and "active controls" in assumption.message
        for assumption in zero.assumptions
    )
    assert any(
        "no declared one- or two-qubit gate profile" in assumption.message
        for assumption in direct.assumptions
    )
    assert zero_controls.gates == qm.GateResources(total=5)
    assert zero_controls.depth == qm.DepthResources(depth=1)
    assert zero_controls.quality is qm.EstimateQuality.EXACT
    assert zero_controls.assumptions == ()
    assert positive.gates.total == 7
    assert positive.depth.depth == 7
    assert not any(
        "no declared one- or two-qubit gate profile" in assumption.message
        for assumption in positive.assumptions
    )
    assert complete.gates.total == 7
    assert any(
        "uses the aggregate clean-ancilla Toffoli batching model" in assumption.message
        for assumption in complete.assumptions
    )
    assert not any(
        "with unclassified arity" in assumption.message
        for assumption in complete.assumptions
    )


def test_symbolic_arity_branch_preserves_internal_metadata() -> None:
    """Partial projection retains caller metadata through its symbolic branch."""
    single = sp.Dummy("single", integer=True, nonnegative=True)
    base = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=5,
            single_qubit=single,
        ),
        _allocation_sites={"site": sp.Integer(3)},
        _output_sizes={"output": sp.Integer(1)},
        _input_sizes={"input": sp.Integer(2)},
        _has_output_summary=True,
        _dependency_keys=frozenset({("wire", None)}),
        _symbol_aliases={single: "declared_single"},
    )

    controlled = base.controlled(2)

    assert controlled._allocation_sites == base._allocation_sites
    assert controlled._output_sizes == base._output_sizes
    assert controlled._input_sizes == base._input_sizes
    assert controlled._has_output_summary is True
    assert controlled._dependency_keys == base._dependency_keys
    assert controlled._symbol_aliases == base._symbol_aliases
    assert controlled.parameters == {"declared_single": single}


def test_query_only_opaque_cost_is_not_treated_as_complete_gate_profile() -> None:
    """A query declaration without gate counts cannot bound control overhead."""
    base = qm.ResourceEstimate(
        calls=qm.CallResources(
            calls_by_name={"query_only_oracle": 1},
            queries_by_name={"query_only_oracle": 1},
        )
    )

    estimate = base.controlled(3)

    assert estimate.gates == base.gates
    assert estimate.depth == base.depth
    assert estimate.calls == base.calls
    assert estimate.derivation is qm.EstimateDerivation.MODELED
    assert estimate.quality is qm.EstimateQuality.UNKNOWN
    assert any(
        "no declared one- or two-qubit gate profile" in assumption.message
        for assumption in estimate.assumptions
    )
    assert not any(
        "for all declared one- and two-qubit gates" in assumption.message
        for assumption in estimate.assumptions
    )


def test_zero_aggregate_cost_does_not_claim_controlled_upper_bound() -> None:
    """An empty aggregate cannot reveal hidden controlled global phase."""
    estimate = qm.ResourceEstimate.zero().controlled(3)

    assert estimate.gates.total == 0
    assert estimate.derivation is qm.EstimateDerivation.MODELED
    assert estimate.quality is qm.EstimateQuality.UNKNOWN
    assert any(
        "cannot distinguish an exact identity from an undeclared global phase"
        in assumption.message
        for assumption in estimate.assumptions
    )


@pytest.mark.parametrize(
    "gates",
    [
        qm.GateResources(total=1, single_qubit=1, toffoli=1),
        qm.GateResources(total=1, two_qubit=1, t=1),
    ],
)
def test_incompatible_opaque_gate_family_arity_is_not_projected(
    gates: qm.GateResources,
) -> None:
    """Known family/arity contradictions keep their declared aggregate cost."""
    estimate = qm.ResourceEstimate(gates=gates).controlled(4)

    assert estimate.gates == gates
    assert estimate.derivation is qm.EstimateDerivation.MODELED
    assert estimate.quality is qm.EstimateQuality.UNKNOWN
    assert any("exceeds the declared" in item.message for item in estimate.assumptions)


def test_opaque_callback_separates_declared_and_inherited_controls() -> None:
    """A callback returns base cost before inherited controls are projected."""
    oracle = qm.opaque(
        "base_control_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=_BaseControlOpaqueCost(),
    )

    @qm.qkernel
    def invoke_oracle(
        own_control: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Invoke the oracle with its one explicit control."""
        own_control, target = oracle(target, controls=(own_control,))
        return own_control, target

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Invoke the controlled oracle under two inherited controls."""
        surrounding_0 = qm.qubit("surrounding_0")
        surrounding_1 = qm.qubit("surrounding_1")
        own_control = qm.qubit("own_control")
        target = qm.qubit("target")
        return qm.control(
            invoke_oracle,
            num_controls=2,
        )(surrounding_0, surrounding_1, own_control, target)

    estimate = circuit.estimate_resources()
    expected = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, two_qubit=1),
    ).controlled(2)

    assert estimate.gates == expected.gates
    assert estimate.width.clean_ancilla_qubits == expected.width.clean_ancilla_qubits
    assert estimate.calls.calls_by_name == {
        "callable=base_control_oracle": 1,
        "definition_control_qubits=1": 1,
        "target_qubits=1": 1,
    }


def test_opaque_cost_context_hides_call_site_transforms() -> None:
    """A callback sees definition targets but not added or inherited controls."""
    observed: list[qm.OpaqueCostContext] = []

    def cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Record the definition-level context and return one base gate.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: One two-qubit base gate whose arity includes
            the definition-declared control.
        """
        observed.append(ctx)
        return qm.ResourceEstimate(
            gates=qm.GateResources(total=1, two_qubit=1),
            basis=ctx.basis,
            control_decomposition=ctx.control_decomposition,
            precision=ctx.precision,
        )

    oracle = qm.opaque(
        "context_boundary_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=cost,
    )
    controlled_oracle = qm.control(oracle, num_controls=2)

    @qm.qkernel
    def layer(
        added_0: qm.Qubit,
        added_1: qm.Qubit,
        declared: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply two later controls to the declared-controlled Oracle."""
        return controlled_oracle(added_0, added_1, declared, target)

    @qm.qkernel
    def circuit() -> tuple[
        qm.Qubit,
        qm.Qubit,
        qm.Qubit,
        qm.Qubit,
        qm.Qubit,
    ]:
        """Apply one inherited control outside the controlled Oracle layer."""
        return qm.control(layer)(
            qm.qubit("inherited"),
            qm.qubit("added_0"),
            qm.qubit("added_1"),
            qm.qubit("declared"),
            qm.qubit("target"),
        )

    circuit.estimate_resources()

    assert len(observed) == 1
    context = observed[0]
    assert context.callable_name == "context_boundary_oracle"
    assert context.definition_control_qubits == 1
    assert context.target_shapes == {"target_0": ()}
    assert context.target_qubits == 1
    for call_site_field in (
        "added_controls",
        "inherited_controls",
        "external_controls",
        "total_controls",
        "transform",
        "power",
    ):
        assert not hasattr(context, call_site_field)


def test_declared_oracle_controls_are_already_in_fixed_base_cost() -> None:
    """A direct declared-controlled Oracle does not control its cost again."""
    oracle = qm.opaque(
        "declared_control_cost",
        num_qubits=1,
        num_control_qubits=2,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(total=1, multi_qubit=1),
        ),
    )

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Invoke the costed Oracle with its two declared controls."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        target = qm.qubit("target")
        return oracle(
            target,
            controls=(control_0, control_1),
        )

    estimate = circuit.estimate_resources()

    assert estimate.gates == qm.GateResources(total=1, multi_qubit=1)
    assert estimate.width.clean_ancilla_qubits == 0


def test_opaque_cost_projects_only_added_and_inherited_controls() -> None:
    """Declared controls stay in base cost while later controls are projected."""
    oracle = qm.opaque(
        "partitioned_control_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=_BaseControlOpaqueCost(),
    )
    controlled_oracle = qm.control(oracle, num_controls=2)

    @qm.qkernel
    def layer(
        added_0: qm.Qubit,
        added_1: qm.Qubit,
        declared: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Apply the Oracle with two controls added after its definition."""
        return controlled_oracle(added_0, added_1, declared, target)

    @qm.qkernel
    def circuit() -> tuple[
        qm.Qubit,
        qm.Qubit,
        qm.Qubit,
        qm.Qubit,
        qm.Qubit,
    ]:
        """Apply the layer under one additional inherited control."""
        inherited = qm.qubit("inherited")
        added_0 = qm.qubit("added_0")
        added_1 = qm.qubit("added_1")
        declared = qm.qubit("declared")
        target = qm.qubit("target")
        return qm.control(layer)(
            inherited,
            added_0,
            added_1,
            declared,
            target,
        )

    estimate = circuit.estimate_resources()
    expected = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, two_qubit=1),
    ).controlled(3)

    assert estimate.gates == expected.gates
    assert estimate.calls.calls_by_name == {
        "callable=partitioned_control_oracle": 1,
        "definition_control_qubits=1": 1,
        "target_qubits=1": 1,
    }


def test_opaque_cost_discards_foreign_dependency_masks() -> None:
    """Fixed and callback costs never reuse dependency UUIDs from their source."""

    @qm.qkernel
    def priced(
        declared: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Price one operation that already includes its declared control."""
        return qm.control(qm.h)(declared, target)

    base_cost = priced.estimate_resources()
    fixed_oracle = qm.opaque(
        "fixed_dependency_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=base_cost,
    )

    def callback_cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return the same definition-level cost through the callback path."""
        assert ctx.definition_control_qubits == 1
        assert ctx.target_qubits == 1
        return base_cost

    callback_oracle = qm.opaque(
        "callback_dependency_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=callback_cost,
    )

    @qm.qkernel
    def fixed_layer(
        added: qm.Qubit,
        declared: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Add one call-site control to the fixed-cost Oracle."""
        return qm.control(fixed_oracle)(added, declared, target)

    @qm.qkernel
    def callback_layer(
        added: qm.Qubit,
        declared: qm.Qubit,
        target: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Add one call-site control to the callback-cost Oracle."""
        return qm.control(callback_oracle)(added, declared, target)

    @qm.qkernel
    def fixed_circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Add one inherited control and then touch the Oracle target again."""
        inherited = qm.qubit("inherited")
        added = qm.qubit("added")
        declared = qm.qubit("declared")
        target = qm.qubit("target")
        inherited, added, declared, target = qm.control(fixed_layer)(
            inherited,
            added,
            declared,
            target,
        )
        target = qm.h(target)
        return inherited, added, declared, target

    @qm.qkernel
    def callback_circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit, qm.Qubit]:
        """Exercise the same dependency boundary through a callback cost."""
        inherited = qm.qubit("inherited")
        added = qm.qubit("added")
        declared = qm.qubit("declared")
        target = qm.qubit("target")
        inherited, added, declared, target = qm.control(callback_layer)(
            inherited,
            added,
            declared,
            target,
        )
        target = qm.h(target)
        return inherited, added, declared, target

    fixed = fixed_circuit.estimate_resources()
    callback = callback_circuit.estimate_resources()
    restored = qm.estimate_resources(deserialize(serialize(fixed_circuit)))

    expected_gates = qm.GateResources(
        total=8,
        single_qubit=1,
        two_qubit=3,
        multi_qubit=4,
        clifford=4,
        rotation=1,
        toffoli=4,
        non_clifford=5,
    )
    expected_depth = qm.DepthResources(
        depth=8,
        clifford_depth=4,
        rotation_depth=1,
        toffoli_depth=4,
        non_clifford_depth=5,
        gate_depth=8,
    )
    for estimate in (fixed, callback, restored):
        assert estimate.gates == expected_gates
        assert estimate.depth == expected_depth
        assert estimate.width.clean_ancilla_qubits == 2
        assert estimate.width.peak_qubits == 6


def test_opaque_cost_width_is_relative_to_the_call_boundary() -> None:
    """Fixed and callback costs do not recount their source qkernel inputs."""

    @qm.qkernel
    def priced(
        left: qm.Qubit,
        right: qm.Qubit,
    ) -> tuple[qm.Qubit, qm.Qubit]:
        """Use one internal workspace qubit above two caller-owned inputs."""
        workspace = qm.qubit("workspace")
        workspace = qm.h(workspace)
        return left, right

    base_cost = priced.estimate_resources()
    assert base_cost.width.input_qubits == 2
    assert base_cost.width.allocated_qubits == 1
    assert base_cost.width.peak_qubits == 3

    fixed_oracle = qm.opaque(
        "fixed_width_boundary_oracle",
        num_qubits=2,
        cost=base_cost,
    )

    def callback_cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return the same standalone qkernel cost through a callback."""
        assert ctx.target_qubits == 2
        return base_cost

    callback_oracle = qm.opaque(
        "callback_width_boundary_oracle",
        num_qubits=2,
        cost=callback_cost,
    )

    @qm.qkernel
    def body_circuit() -> tuple[qm.Bit, qm.Bit, qm.Qubit, qm.Qubit]:
        """Call the body, consume its inputs, then allocate two later qubits."""
        left = qm.qubit("left")
        right = qm.qubit("right")
        left, right = priced(left, right)
        measured_left = qm.measure(left)
        measured_right = qm.measure(right)
        later_left = qm.qubit("later_left")
        later_right = qm.qubit("later_right")
        return measured_left, measured_right, later_left, later_right

    @qm.qkernel
    def fixed_circuit() -> tuple[qm.Bit, qm.Bit, qm.Qubit, qm.Qubit]:
        """Exercise the same lifetime pattern through a fixed opaque cost."""
        left = qm.qubit("left")
        right = qm.qubit("right")
        left, right = fixed_oracle(left, right)
        measured_left = qm.measure(left)
        measured_right = qm.measure(right)
        later_left = qm.qubit("later_left")
        later_right = qm.qubit("later_right")
        return measured_left, measured_right, later_left, later_right

    @qm.qkernel
    def callback_circuit() -> tuple[qm.Bit, qm.Bit, qm.Qubit, qm.Qubit]:
        """Exercise the same lifetime pattern through a callback opaque cost."""
        left = qm.qubit("left")
        right = qm.qubit("right")
        left, right = callback_oracle(left, right)
        measured_left = qm.measure(left)
        measured_right = qm.measure(right)
        later_left = qm.qubit("later_left")
        later_right = qm.qubit("later_right")
        return measured_left, measured_right, later_left, later_right

    body = body_circuit.estimate_resources()
    fixed = fixed_circuit.estimate_resources()
    callback = callback_circuit.estimate_resources()

    for estimate in (body, fixed, callback):
        assert estimate.width.input_qubits == 0
        assert estimate.width.allocated_qubits == 5
        assert estimate.width.peak_qubits == 3
        assert estimate.width.circuit_qubits == 5
        assert not any(
            "anonymous allocated workspace" in assumption.message
            for assumption in estimate.assumptions
        )


def test_opaque_peak_residual_becomes_anonymous_workspace() -> None:
    """An input-plus-peak declaration survives call-boundary scheduling."""
    base_cost = qm.ResourceEstimate(
        width=qm.WidthResources(
            input_qubits=1,
            peak_qubits=3,
        )
    )
    fixed_oracle = qm.opaque(
        "fixed_peak_only_oracle",
        num_qubits=1,
        cost=base_cost,
    )
    observed: list[qm.OpaqueCostContext] = []

    def callback_cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return the same partial width declaration.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: Input-plus-peak opaque width.
        """
        observed.append(ctx)
        return base_cost

    callback_oracle = qm.opaque(
        "callback_peak_only_oracle",
        num_qubits=1,
        cost=callback_cost,
    )

    @qm.qkernel
    def fixed_direct() -> qm.Qubit:
        """Invoke the fixed-cost Oracle directly."""
        target = qm.qubit("target")
        (target,) = fixed_oracle(target)
        return target

    @qm.qkernel
    def callback_direct() -> qm.Qubit:
        """Invoke the callback-priced Oracle directly."""
        target = qm.qubit("target")
        (target,) = callback_oracle(target)
        return target

    @qm.qkernel
    def fixed_inverse() -> qm.Qubit:
        """Invoke the inverse fixed-cost Oracle."""
        target = qm.qubit("target")
        (target,) = qm.inverse(fixed_oracle)(target)
        return target

    @qm.qkernel
    def fixed_controlled() -> qm.Qubit:
        """Invoke the fixed-cost Oracle under two controls."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(fixed_oracle, num_controls=2)(
            controls[0],
            controls[1],
            target,
        )
        return target

    direct_estimates = (
        fixed_direct.estimate_resources(),
        callback_direct.estimate_resources(),
        fixed_inverse.estimate_resources(),
    )
    for estimate in direct_estimates:
        assert estimate.width.allocated_qubits == 3
        assert estimate.width.peak_qubits == 3
        assert estimate.width.circuit_qubits == 3
        assert any(
            "anonymous allocated workspace" in assumption.message
            for assumption in estimate.assumptions
        )

    controlled = fixed_controlled.estimate_resources()
    assert controlled.width.allocated_qubits == 5
    assert controlled.width.clean_ancilla_qubits == 0
    assert controlled.width.peak_qubits == 5
    assert controlled.width.circuit_qubits == 5
    assert len(observed) == 1


def test_opaque_peak_only_workspace_remains_distinct_from_reusable_ancilla() -> None:
    """Peak-only scratch stays conservative across distinct static call sites."""
    peak_only = qm.opaque(
        "peak_only_static_workspace",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            width=qm.WidthResources(
                input_qubits=1,
                peak_qubits=3,
            )
        ),
    )
    reusable_clean = qm.opaque(
        "declared_reusable_clean_workspace",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            width=qm.WidthResources(
                input_qubits=1,
                clean_ancilla_qubits=2,
                peak_qubits=3,
            )
        ),
    )

    @qm.qkernel
    def peak_only_circuit() -> qm.Qubit:
        """Call the peak-only Oracle at two distinct source locations."""
        target = qm.qubit("target")
        (target,) = peak_only(target)
        (target,) = peak_only(target)
        return target

    @qm.qkernel
    def reusable_clean_circuit() -> qm.Qubit:
        """Call the clean-workspace Oracle at two distinct source locations."""
        target = qm.qubit("target")
        (target,) = reusable_clean(target)
        (target,) = reusable_clean(target)
        return target

    conservative = peak_only_circuit.estimate_resources()
    reusable = reusable_clean_circuit.estimate_resources()

    assert conservative.width.peak_qubits == reusable.width.peak_qubits == 3
    assert conservative.width.allocated_qubits == 5
    assert conservative.width.clean_ancilla_qubits == 0
    assert conservative.width.circuit_qubits == 5
    assert reusable.width.allocated_qubits == 1
    assert reusable.width.clean_ancilla_qubits == 2
    assert reusable.width.circuit_qubits == 3


@pytest.mark.parametrize("uses_callback", [False, True], ids=["fixed", "callback"])
def test_opaque_ancilla_declaration_implies_peak_workspace(
    uses_callback: bool,
) -> None:
    """Opaque clean ancillas contribute to peak even when peak is omitted."""
    base_cost = qm.ResourceEstimate(
        width=qm.WidthResources(clean_ancilla_qubits=2),
    )

    def callback_cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return the same partial width declaration for one Oracle call."""
        assert ctx.target_qubits == 1
        return base_cost

    oracle = qm.opaque(
        f"ancilla_without_peak_{uses_callback}",
        num_qubits=1,
        cost=callback_cost if uses_callback else base_cost,
    )

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Invoke an Oracle whose cost declares only clean workspace."""
        (target,) = oracle(qm.qubit("target"))
        return target

    estimate = circuit.estimate_resources()

    assert estimate.width.allocated_qubits == 1
    assert estimate.width.clean_ancilla_qubits == 2
    assert estimate.width.peak_qubits == 3
    assert estimate.width.circuit_qubits == 3


def test_opaque_callback_arity_profile_uses_fixed_cost_projection() -> None:
    """Callback and fixed arity profiles receive the same external controls."""
    callback_oracle = qm.opaque(
        "callback_arity_oracle",
        num_qubits=1,
        cost=_BaseArityProfileOpaqueCost(),
    )
    fixed_cost = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=3,
            single_qubit=2,
            two_qubit=1,
        ),
    )
    fixed_oracle = qm.opaque(
        "fixed_arity_oracle",
        num_qubits=1,
        cost=fixed_cost,
    )

    @qm.qkernel
    def invoke_callback(target: qm.Qubit) -> qm.Qubit:
        """Invoke the callback-priced opaque Oracle."""
        (target,) = callback_oracle(target)
        return target

    @qm.qkernel
    def callback_circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Invoke the callback Oracle under two inherited controls."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        target = qm.qubit("target")
        return qm.control(
            invoke_callback,
            num_controls=2,
        )(control_0, control_1, target)

    @qm.qkernel
    def fixed_circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Invoke the fixed-cost Oracle with two added controls."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        target = qm.qubit("target")
        return qm.control(fixed_oracle, num_controls=2)(
            control_0,
            control_1,
            target,
        )

    callback_estimate = callback_circuit.estimate_resources()
    fixed_estimate = fixed_circuit.estimate_resources()

    assert callback_estimate.gates == fixed_estimate.gates
    assert callback_estimate.depth == fixed_estimate.depth
    assert (
        callback_estimate.width.clean_ancilla_qubits
        == fixed_estimate.width.clean_ancilla_qubits
    )
    assert callback_estimate.calls.calls_by_name == {
        "base_definition_controls=0": 1,
    }


def test_opaque_aggregate_and_enclosing_body_share_recipe_predicate() -> None:
    """An Invoke boundary cannot change the symbolic shared-ladder choice."""
    known_work = sp.Symbol(
        "known_arity_work",
        integer=True,
        nonnegative=True,
    )
    base_cost = qm.ResourceEstimate(
        gates=qm.GateResources(
            total=2,
            single_qubit=known_work,
        ),
    )
    oracle = qm.opaque(
        "shared_recipe_predicate_oracle",
        num_qubits=1,
        cost=base_cost,
    )

    @qm.qkernel
    def body(target: qm.Qubit) -> qm.Qubit:
        """Invoke the aggregate-cost Oracle once."""
        (target,) = oracle(target)
        return target

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Apply two outer controls around the invocation body."""
        controls = qm.qubit_array(2, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(body, num_controls=2)(controls, target)
        return target

    through_body = circuit.estimate_resources()
    direct_aggregate = base_cost.controlled(2)

    for concrete_work in (0, 1, 2):
        body_estimate = through_body.substitute(known_arity_work=concrete_work)
        aggregate_estimate = direct_aggregate.substitute(known_arity_work=concrete_work)
        assert body_estimate.gates == aggregate_estimate.gates
        assert body_estimate.depth == aggregate_estimate.depth
        assert (
            body_estimate.width.clean_ancilla_qubits
            == aggregate_estimate.width.clean_ancilla_qubits
        )


def test_opaque_callback_nested_estimate_restores_manual_parameter_metadata() -> None:
    """A nested estimate returns to eager callback-local parameter derivation."""

    @qm.qkernel
    def symbolic_body(iterations: qm.UInt) -> qm.Qubit:
        """Apply one gate per symbolic iteration."""
        target = qm.qubit("nested_target")
        for _index in qm.range(iterations):
            target = qm.h(target)
        return target

    def nested_cost(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Compose a nested estimate with a callback-local symbolic cost.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: Three concrete one-qubit gates.
        """
        nested = symbolic_body.estimate_resources(
            basis=ctx.basis,
            control_decomposition=ctx.control_decomposition,
        )
        assert set(nested.parameters) == {"iterations"}

        callback_work = sp.Symbol(
            "callback_work",
            integer=True,
            nonnegative=True,
        )
        manual = qm.ResourceEstimate(
            gates=qm.GateResources(
                total=callback_work,
                single_qubit=callback_work,
            ),
            basis=ctx.basis,
            control_decomposition=ctx.control_decomposition,
            precision=ctx.precision,
        )
        assert manual.parameters == {"callback_work": callback_work}
        return nested.substitute(iterations=2).seq(manual.substitute(callback_work=1))

    oracle = qm.opaque(
        "nested_estimate_oracle",
        num_qubits=1,
        cost=nested_cost,
    )

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Invoke the callback that performs a nested estimate."""
        target = qm.qubit("target")
        (target,) = oracle(target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.parameters == {}
    assert estimate.gates.total == 3
    assert estimate.gates.single_qubit == 3


def test_opaque_callback_uses_requested_provenance_and_allows_neutral_cost() -> None:
    """Callbacks can return matching provenance or basis-neutral resources."""
    compatible = qm.opaque(
        "compatible_oracle",
        num_qubits=1,
        cost=_ContextAwareOpaqueCost(),
    )
    neutral = qm.opaque(
        "neutral_oracle",
        num_qubits=1,
        cost=_BasisNeutralOpaqueCost(),
    )

    @qm.qkernel
    def compatible_circuit() -> qm.Qubit:
        """Invoke the context-aware opaque callable."""
        (target,) = compatible(qm.qubit("target"))
        return target

    @qm.qkernel
    def neutral_circuit() -> qm.Qubit:
        """Invoke the basis-neutral opaque callable."""
        (target,) = neutral(qm.qubit("target"))
        return target

    lowered = compatible_circuit.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-4,
    )
    neutral_logical = neutral_circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert lowered.basis is qm.GateBasis.CLIFFORD_T
    assert lowered.precision == 1e-4
    assert neutral_logical.basis is qm.GateBasis.LOGICAL
    assert neutral_logical.calls.calls_by_name == {"neutral_oracle": 1}


@pytest.mark.parametrize("use_callback", [False, True], ids=["fixed", "callback"])
@pytest.mark.parametrize(
    "label_with_requested_model",
    [False, True],
    ids=["default-neutral", "requested-model-neutral"],
)
def test_controlled_basis_neutral_opaque_cost_has_fixed_callback_parity(
    use_callback: bool,
    label_with_requested_model: bool,
) -> None:
    """Calls-only costs need no gate projection under Clifford+T controls."""

    def callback(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return one call-only definition cost.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: Basis-neutral call resource.
        """
        options = (
            {
                "basis": ctx.basis,
                "control_decomposition": ctx.control_decomposition,
                "precision": ctx.precision,
            }
            if label_with_requested_model
            else {}
        )
        return qm.ResourceEstimate(
            calls=qm.CallResources(calls_by_name={"neutral_controlled": 1}),
            **options,
        )

    fixed_options = (
        {
            "basis": qm.GateBasis.CLIFFORD_T,
            "control_decomposition": (qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI),
            "precision": 1e-4,
        }
        if label_with_requested_model
        else {}
    )
    fixed = qm.ResourceEstimate(
        calls=qm.CallResources(calls_by_name={"neutral_controlled": 1}),
        **fixed_options,
    )
    oracle = qm.opaque(
        "neutral_controlled",
        num_qubits=1,
        cost=callback if use_callback else fixed,
    )

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply one added control to the calls-only Oracle."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        return qm.control(oracle)(control, target)

    estimate = circuit.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-4,
    )

    assert estimate.basis is qm.GateBasis.CLIFFORD_T
    assert estimate.calls.calls_by_name == {"neutral_controlled": 1}
    assert estimate.gates.total == 0
    assert estimate.derivation is qm.EstimateDerivation.MODELED
    assert any(
        "no declared one- or two-qubit gate profile" in assumption.message
        for assumption in estimate.assumptions
    )


@pytest.mark.parametrize("use_callback", [False, True], ids=["fixed", "callback"])
@pytest.mark.parametrize(
    ("cost_estimate", "estimator_options", "error_match"),
    [
        pytest.param(
            qm.ResourceEstimate(
                gates=qm.GateResources(total=1, single_qubit=1),
                basis=qm.GateBasis.LOGICAL,
            ),
            {"basis": qm.GateBasis.CLIFFORD_T},
            "uses basis 'logical'",
            id="basis",
        ),
        pytest.param(
            qm.ResourceEstimate(
                gates=qm.GateResources(total=1, single_qubit=1),
                control_decomposition=qm.ControlDecomposition.ABSTRACT,
            ),
            {"control_decomposition": (qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI)},
            "uses control decomposition 'abstract'",
            id="control-decomposition",
        ),
        pytest.param(
            qm.ResourceEstimate(
                gates=qm.GateResources(total=1, t=1),
                basis=qm.GateBasis.CLIFFORD_T,
                precision=1e-3,
            ),
            {
                "basis": qm.GateBasis.CLIFFORD_T,
                "precision": 1e-4,
            },
            "uses precision",
            id="precision",
        ),
    ],
)
def test_opaque_cost_rejects_mismatched_provenance(
    use_callback: bool,
    cost_estimate: qm.ResourceEstimate,
    estimator_options: dict[str, object],
    error_match: str,
) -> None:
    """Fixed and callback costs reject incompatible estimator provenance."""

    def callback(ctx: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return the deliberately incompatible definition-level cost.

        Args:
            ctx (qm.OpaqueCostContext): Definition-level cost context.

        Returns:
            qm.ResourceEstimate: Cost carrying incompatible provenance.
        """
        del ctx
        return cost_estimate

    oracle = qm.opaque(
        "incompatible_provenance_oracle",
        num_qubits=1,
        cost=callback if use_callback else cost_estimate,
    )

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Invoke the opaque callable with incompatible cost provenance."""
        (target,) = oracle(qm.qubit("target"))
        return target

    with pytest.raises(ValueError, match=error_match):
        circuit.estimate_resources(**estimator_options)


def test_formal_quantum_input_is_not_counted_as_body_allocation() -> None:
    """A formal quantum input contributes width once, not twice through QInit."""

    @qm.qkernel
    def circuit(target: qm.Qubit) -> qm.Qubit:
        """Apply one gate to a caller-owned qubit."""
        return qm.h(target)

    estimate = circuit.estimate_resources()

    assert estimate.width.input_qubits == 1
    assert estimate.width.allocated_qubits == 0
    assert estimate.width.peak_qubits == 1
    assert estimate.width.circuit_qubits == 1


def test_metadata_axes_and_model_settings_survive_resource_algebra() -> None:
    """Independent metadata axes and estimator settings remain visible."""
    modeled = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        derivation=qm.EstimateDerivation.MODELED,
        approximation=qm.ApproximationStatus.APPROXIMATE,
    )

    transformed = modeled.controlled(2).inverse().repeat(3).choice(modeled)
    assert transformed.derivation is qm.EstimateDerivation.MODELED
    assert transformed.approximation is qm.ApproximationStatus.APPROXIMATE

    default = _four_h_body.estimate_resources()
    lowered = _four_h_body.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-4,
    )
    assert default.basis is qm.GateBasis.LOGICAL
    assert (
        default.control_decomposition is qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
    )
    assert default.precision is None
    assert default.to_dict()["basis"] == "logical"
    assert default.to_dict()["control_decomposition"] == "clean_ancilla_toffoli"
    assert default.to_dict()["approximation"] == "exact"
    assert default.to_dict()["width"]["circuit_qubits"] == "1"
    assert lowered.basis is qm.GateBasis.CLIFFORD_T
    assert lowered.precision == 1e-4
    assert lowered.to_dict()["precision"] == 1e-4
    assert lowered.approximation is qm.ApproximationStatus.EXACT


def test_nonunitary_resources_compose_substitute_and_serialize() -> None:
    """Measurement/reset fields survive composition and symbolic rewrites."""
    events = sp.Symbol("events", integer=True, nonnegative=True)
    primitive = qm.ResourceEstimate(
        measurements=qm.MeasurementResources(total=1),
        resets=qm.ResetResources(total=1),
        depth=qm.DepthResources(
            depth=2,
            measurement_depth=1,
            reset_depth=1,
        ),
    )

    sequential = primitive.seq(primitive)
    parallel = primitive.parallel(primitive)
    repeated = primitive.repeat(events)

    assert sequential.measurements.total == 2
    assert sequential.resets.total == 2
    assert sequential.depth.depth == 4
    assert sequential.depth.measurement_depth == 2
    assert sequential.depth.reset_depth == 2

    assert parallel.measurements.total == 2
    assert parallel.resets.total == 2
    assert parallel.depth.depth == 2
    assert parallel.depth.measurement_depth == 1
    assert parallel.depth.reset_depth == 1

    assert repeated.parameters == {"events": events}
    assert repeated.measurements.total == events
    assert repeated.resets.total == events
    assert repeated.depth.depth == 2 * events
    assert repeated.depth.measurement_depth == events
    assert repeated.depth.reset_depth == events

    substituted = repeated.substitute(events=3)
    assert substituted.parameters == {}
    assert substituted.measurements.total == 3
    assert substituted.resets.total == 3
    assert substituted.depth.depth == 6
    assert substituted.depth.measurement_depth == 3
    assert substituted.depth.reset_depth == 3

    serialized = repeated.to_dict()
    assert serialized["measurements"] == {"total": "events"}
    assert serialized["resets"] == {"total": "events"}
    assert serialized["depth"]["gate_depth"] == "0"
    assert serialized["depth"]["measurement_depth"] == "events"
    assert serialized["depth"]["reset_depth"] == "events"


def test_resource_algebra_rejects_mixed_gate_bases() -> None:
    """Basis-sensitive estimates cannot be silently combined and relabeled."""
    clifford_t = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-3,
    )
    logical = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        basis=qm.GateBasis.LOGICAL,
    )

    with pytest.raises(ValueError, match="different gate bases"):
        clifford_t.seq(logical)


def test_resource_algebra_rejects_mixed_control_decompositions() -> None:
    """Gate-sensitive estimates cannot hide different control recipes."""
    clean = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
    )
    abstract = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        control_decomposition=qm.ControlDecomposition.ABSTRACT,
    )

    with pytest.raises(ValueError, match="different control decompositions"):
        clean.seq(abstract)


def test_legacy_unclassified_depth_remains_gate_basis_sensitive() -> None:
    """A pre-gate-depth opaque cost cannot be silently relabeled."""
    clifford_t = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-3,
    )
    legacy = qm.ResourceEstimate(
        depth=qm.DepthResources(depth=1),
        basis=qm.GateBasis.LOGICAL,
    )

    with pytest.raises(ValueError, match="different gate bases"):
        clifford_t.seq(legacy)


@pytest.mark.parametrize(
    "resource_type",
    [qm.MeasurementResources, qm.ResetResources],
)
@pytest.mark.parametrize(
    "invalid_count",
    [True, False, np.bool_(True), -1, 0.5, sp.oo],
)
def test_event_resources_reject_invalid_concrete_counts(
    resource_type: type[qm.MeasurementResources] | type[qm.ResetResources],
    invalid_count: object,
) -> None:
    """Public event resources reject non-count concrete values."""
    with pytest.raises(ValueError, match="nonnegative integer"):
        resource_type(total=invalid_count)


@pytest.mark.parametrize(
    ("field_name", "resource_type"),
    [
        ("measurements", qm.MeasurementResources),
        ("resets", qm.ResetResources),
    ],
)
def test_event_resources_accept_nonnegative_symbols_and_validate_substitution(
    field_name: str,
    resource_type: type[qm.MeasurementResources] | type[qm.ResetResources],
) -> None:
    """Symbolic event counts retain their nonnegative integer domain."""
    events = sp.Symbol("events", integer=True, nonnegative=True)
    resource = resource_type(total=events)
    estimate = qm.ResourceEstimate(**{field_name: resource})

    assert resource_type(total=0).total == 0
    assert estimate.substitute(events=0).parameters == {}
    with pytest.raises(ValueError, match="Cannot substitute negative"):
        estimate.substitute(events=-1)


def test_basis_provenance_does_not_simplify_accumulated_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resource-algebra fold does not simplify its full symbolic history."""
    flag = sp.Symbol("flag", integer=True, nonnegative=True)
    conditional_count = sp.Piecewise((1, sp.Eq(flag, 1)), (0, True))
    decomposed = qm.ResourceEstimate(
        gates=qm.GateResources(total=conditional_count),
    )
    neutral = qm.ResourceEstimate(
        depth=qm.DepthResources(depth=1, measurement_depth=1),
        basis=qm.GateBasis.LOGICAL,
        control_decomposition=qm.ControlDecomposition.ABSTRACT,
    )

    def fail_on_simplify(_expression: sp.Basic) -> None:
        """Reject an unexpected whole-expression simplification.

        Args:
            _expression (sp.Basic): Expression passed by the implementation.

        Raises:
            AssertionError: Always, because provenance checks must be
                structural.
        """
        raise AssertionError("resource provenance unexpectedly called simplify")

    monkeypatch.setattr(sp, "simplify", fail_on_simplify)

    combined = decomposed.seq(neutral)

    assert combined.basis is qm.GateBasis.LOGICAL
    assert (
        combined.control_decomposition is qm.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
    )
    assert combined.gates.total == conditional_count
    assert combined.depth.measurement_depth == 1


def test_measurement_only_resources_are_gate_basis_neutral() -> None:
    """Measurement depth composes with a gate estimate from any basis."""
    measurement = qm.ResourceEstimate(
        depth=qm.DepthResources(depth=1, measurement_depth=1),
        basis=qm.GateBasis.LOGICAL,
    )
    decomposed = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        depth=qm.DepthResources(depth=1, clifford_depth=1),
    )

    combined = measurement.seq(decomposed)

    assert combined.basis is qm.GateBasis.LOGICAL
    assert combined.depth.depth == 2
    assert combined.depth.measurement_depth == 1


def test_controlled_nonunitary_body_fails_closed() -> None:
    """The estimator rejects controlled measurement like the emitter does."""

    @qm.qkernel
    def nonunitary(target: qm.Qubit) -> qm.Qubit:
        """Measure a target inside an otherwise callable body."""
        qm.measure(target)
        return target

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Attempt to place the nonunitary body under coherent control."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        return qm.control(nonunitary)(control, target)

    with pytest.raises(
        ValueError,
        match=r"non-unitary kernel effects.*MEASUREMENT",
    ):
        circuit.estimate_resources()


def test_pauli_lcu_block_encoding_composes_forward_inverse_and_control() -> None:
    """Block encoding exposes meaningful costs under inverse and control."""
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    encoding = qm.pauli_lcu_block_encoding(
        PauliLCU.from_matrix(1j * identity + 0.5 * pauli_x)
    )

    @qm.qkernel
    def direct() -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Apply the block encoding directly."""
        signal = qm.qubit_array(encoding.num_signal_qubits, "signal")
        system = qm.qubit_array(encoding.num_system_qubits, "system")
        return encoding.unitary(signal, system)

    @qm.qkernel
    def inverse() -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Apply the inverse block encoding."""
        signal = qm.qubit_array(encoding.num_signal_qubits, "signal")
        system = qm.qubit_array(encoding.num_system_qubits, "system")
        return qm.inverse(encoding.unitary)(signal, system)

    @qm.qkernel
    def controlled() -> tuple[
        qm.Vector[qm.Qubit],
        qm.Vector[qm.Qubit],
        qm.Vector[qm.Qubit],
    ]:
        """Apply the block encoding under three controls."""
        controls = qm.qubit_array(3, "controls")
        signal = qm.qubit_array(encoding.num_signal_qubits, "signal")
        system = qm.qubit_array(encoding.num_system_qubits, "system")
        *_, signal, system = qm.control(
            encoding.unitary,
            num_controls=3,
        )(controls, signal, system)
        return controls, signal, system

    direct_estimate = direct.estimate_resources()
    inverse_estimate = inverse.estimate_resources()
    controlled_estimate = controlled.estimate_resources()
    root_estimate = encoding.unitary.estimate_resources()

    assert direct_estimate.gates == inverse_estimate.gates
    assert direct_estimate.gates.total == 6
    assert direct_estimate.width.allocated_qubits == 2
    assert direct_estimate.width.peak_qubits == 2

    assert controlled_estimate.gates.total == 10
    assert controlled_estimate.gates.two_qubit == 5
    assert controlled_estimate.gates.multi_qubit == 5
    assert controlled_estimate.gates.toffoli == 5
    assert controlled_estimate.width.allocated_qubits == 5
    assert controlled_estimate.width.clean_ancilla_qubits == 2
    assert controlled_estimate.width.peak_qubits == 7
    assert controlled_estimate.width.circuit_qubits == 7

    assert root_estimate.width.allocated_qubits == 0
    assert root_estimate.width.input_qubits == 2
    assert root_estimate.width.peak_qubits == 2
    assert root_estimate.width.circuit_qubits == 2
    assert "signal_dim0" not in root_estimate.parameters
    assert "system_dim0" not in root_estimate.parameters


def _block_encoding_callers(
    encoding: qm.LCUBlockEncoding,
) -> tuple[qm.QKernel, qm.QKernel, qm.QKernel]:
    """Build direct, inverse, and controlled symbolic-width callers.

    Args:
        encoding (qm.LCUBlockEncoding): Encoding whose callable metadata is
            exercised.

    Returns:
        tuple[qm.QKernel, qm.QKernel, qm.QKernel]: Direct, inverse, and
            controlled callers.
    """

    @qm.qkernel
    def direct(
        signal: qm.Vector[qm.Qubit],
        system: qm.Vector[qm.Qubit],
    ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Apply the captured block encoding directly."""
        return encoding.unitary(signal, system)

    @qm.qkernel
    def inverse(
        signal: qm.Vector[qm.Qubit],
        system: qm.Vector[qm.Qubit],
    ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Apply the captured block encoding inversely."""
        return qm.inverse(encoding.unitary)(signal, system)

    @qm.qkernel
    def controlled(
        control: qm.Qubit,
        signal: qm.Vector[qm.Qubit],
        system: qm.Vector[qm.Qubit],
    ) -> tuple[qm.Qubit, qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Apply the captured block encoding under one control."""
        return qm.control(encoding.unitary)(control, signal, system)

    return direct, inverse, controlled


def test_recursive_lcu_expands_through_inverse_control_and_serialization() -> None:
    """New public LCU encodings retain recursive resource semantics."""
    encoding = _recursive_lcu_resource_encoding()
    direct, inverse, controlled = _block_encoding_callers(encoding)
    inputs = {
        "signal": encoding.num_signal_qubits,
        "system": encoding.num_system_qubits,
    }

    direct_estimate = direct.estimate_resources(inputs=inputs)
    inverse_estimate = inverse.estimate_resources(inputs=inputs)
    controlled_estimate = controlled.estimate_resources(inputs=inputs)
    root_estimate = encoding.unitary.estimate_resources()
    restored_estimate = qm.estimate_resources(
        deserialize(serialize(encoding.unitary)),
    )
    restored_inverse = qm.estimate_resources(
        deserialize(serialize(inverse)),
        inputs=inputs,
    )
    restored_controlled = qm.estimate_resources(
        deserialize(serialize(controlled)),
        inputs=inputs,
    )

    assert (encoding.num_signal_qubits, encoding.num_system_qubits) == (2, 1)
    # Two outer RY preparation gates surround a coherent SELECT. Its Ising
    # branch contributes two X brackets plus eight child gates; its identity
    # branch contributes one relative-phase gate, for 2 + 10 + 1 = 13.
    assert direct_estimate.gates == qm.GateResources(
        total=13,
        single_qubit=7,
        two_qubit=5,
        multi_qubit=1,
        clifford=6,
        rotation=6,
        t=0,
        toffoli=1,
        non_clifford=7,
    )
    assert inverse_estimate.gates == direct_estimate.gates
    assert inverse_estimate.width == direct_estimate.width
    assert inverse_estimate.depth == direct_estimate.depth
    assert inverse_estimate.calls == direct_estimate.calls
    assert inverse_estimate.derivation is direct_estimate.derivation
    assert inverse_estimate.quality is direct_estimate.quality
    assert direct_estimate.width.input_qubits == 3
    assert direct_estimate.width.peak_qubits == 3

    assert controlled_estimate.gates == qm.GateResources(
        total=15,
        single_qubit=2,
        two_qubit=10,
        multi_qubit=3,
        clifford=6,
        rotation=6,
        t=0,
        toffoli=3,
        non_clifford=9,
    )
    assert controlled_estimate.width.input_qubits == 4
    assert controlled_estimate.width.clean_ancilla_qubits == 1
    assert controlled_estimate.width.peak_qubits == 5

    assert root_estimate.gates == direct_estimate.gates
    assert root_estimate.width == direct_estimate.width
    assert root_estimate.depth == direct_estimate.depth
    assert restored_estimate.gates == root_estimate.gates
    assert restored_estimate.width == root_estimate.width
    assert restored_estimate.depth == root_estimate.depth
    assert direct_estimate.calls == qm.CallResources()
    assert controlled_estimate.calls == qm.CallResources()
    for restored, expected in (
        (restored_inverse, inverse_estimate),
        (restored_controlled, controlled_estimate),
    ):
        assert restored.width == expected.width
        assert restored.gates == expected.gates
        assert restored.depth == expected.depth
        assert restored.calls == expected.calls
        assert restored.derivation is expected.derivation
        assert restored.quality is expected.quality


@pytest.mark.parametrize(
    "encoding",
    [
        qm.pauli_lcu_block_encoding(
            PauliLCU.from_matrix(np.eye(4, dtype=np.complex128))
        ),
        qm.periodic_shift_lcu_block_encoding(
            PeriodicShiftLCU.from_coefficients({1: 1.0}, register_sizes=(2,))
        ),
        qm.identity_block_encoding(2),
        qm.ising_z_block_encoding(
            {(): 1.0, (0,): -0.5, (1,): 0.25j},
            2,
        ),
        _recursive_lcu_resource_encoding(),
    ],
    ids=("pauli", "periodic-shift", "identity", "ising-z", "recursive"),
)
def test_block_encoding_width_contract_survives_all_call_transforms(
    encoding: qm.LCUBlockEncoding,
) -> None:
    """Every callable form rejects incorrect signal and system widths."""
    direct, inverse, controlled = _block_encoding_callers(encoding)
    correct = {
        "signal": encoding.num_signal_qubits,
        "system": encoding.num_system_qubits,
    }

    for caller in (encoding.unitary, direct, inverse):
        caller.estimate_resources(inputs=correct)
        with pytest.raises(ValueError, match="signal register width must equal"):
            caller.estimate_resources(
                inputs={**correct, "signal": encoding.num_signal_qubits + 1}
            )
        with pytest.raises(ValueError, match="system register width must equal"):
            caller.estimate_resources(
                inputs={**correct, "system": encoding.num_system_qubits + 1}
            )

    controlled_inputs = dict(correct)
    controlled.estimate_resources(inputs=controlled_inputs)
    with pytest.raises(ValueError, match="signal register width must equal"):
        controlled.estimate_resources(
            inputs={
                **controlled_inputs,
                "signal": encoding.num_signal_qubits + 1,
            }
        )
    with pytest.raises(ValueError, match="system register width must equal"):
        controlled.estimate_resources(
            inputs={
                **controlled_inputs,
                "system": encoding.num_system_qubits + 1,
            }
        )


def test_block_encoding_width_contract_is_descriptor_local() -> None:
    """Descriptors neither mutate nor conflict through one generic qkernel."""
    narrow = qm.LCUBlockEncoding(_generic_block_unitary, 1.0, 1, 2)
    wide = qm.LCUBlockEncoding(_generic_block_unitary, 1.0, 3, 4)

    assert narrow.unitary is not _generic_block_unitary
    assert wide.unitary is not _generic_block_unitary
    assert narrow.unitary is not wide.unitary

    _generic_block_unitary.estimate_resources(inputs={"signal": 5, "system": 6})
    narrow.unitary.estimate_resources(inputs={"signal": 1, "system": 2})
    wide.unitary.estimate_resources(inputs={"signal": 3, "system": 4})
    with pytest.raises(ValueError, match="signal register width must equal 1"):
        narrow.unitary.estimate_resources(inputs={"signal": 3, "system": 2})
    with pytest.raises(ValueError, match="system register width must equal 4"):
        wide.unitary.estimate_resources(inputs={"signal": 3, "system": 2})


def test_block_encoding_width_contract_survives_serialization() -> None:
    """Serialized periodic encodings retain exact register requirements."""
    encoding = qm.periodic_shift_lcu_block_encoding(
        PeriodicShiftLCU.from_coefficients({1: 1.0}, register_sizes=(2,))
    )
    restored = deserialize(serialize(encoding.unitary))

    automatic = qm.estimate_resources(restored)
    assert automatic.width.input_qubits == 3
    assert "signal_dim0" not in automatic.parameters
    assert "system_dim0" not in automatic.parameters
    qm.estimate_resources(restored, inputs={"signal": 1, "system": 2})
    with pytest.raises(ValueError, match="signal register width must equal 1"):
        qm.estimate_resources(restored, inputs={"signal": 2, "system": 2})
    with pytest.raises(ValueError, match="system register width must equal 2"):
        qm.estimate_resources(restored, inputs={"signal": 1, "system": 3})


def test_root_width_aliases_must_agree_when_both_are_supplied() -> None:
    """Port widths cannot silently overwrite explicit dimension aliases."""
    encoding = _recursive_lcu_resource_encoding()

    estimate = encoding.unitary.estimate_resources(
        inputs={"signal": 2, "signal_dim0": 2}
    )
    assert estimate.width.input_qubits == 3
    with pytest.raises(ValueError, match="signal_dim0=2.*also specify"):
        encoding.unitary.estimate_resources(inputs={"signal": 2, "signal_dim0": 1})
    with pytest.raises(ValueError, match="signal_dim0=None"):
        encoding.unitary.estimate_resources(inputs={"signal": 2, "signal_dim0": None})
    with pytest.raises(ValueError, match="bool is not a dimension"):
        encoding.unitary.estimate_resources(inputs={"signal_dim0": True})


def test_shape_alias_collision_keeps_classical_and_width_inputs_independent() -> None:
    """Generated shape aliases never capture a same-named classical input."""
    attrs = {
        "resource_contract": {
            "quantum_operand_widths": [{"index": 0, "name": "signal", "width": 1}]
        }
    }
    contracted = _shape_name_collision._clone_with_callable_attrs(attrs)

    symbolic = _shape_name_collision.estimate_resources()
    assert set(symbolic.parameters) == {"signal_dim0", "signal_dim0__shape"}
    assert symbolic.gates.total == symbolic.parameters["signal_dim0"]
    assert symbolic.width.input_qubits == symbolic.parameters["signal_dim0__shape"]

    automatic = contracted.estimate_resources(inputs={"signal_dim0": 3})
    explicit = contracted.estimate_resources(inputs={"signal": 1, "signal_dim0": 3})
    plain = _shape_name_collision.estimate_resources(
        inputs={"signal": 1, "signal_dim0": 3}
    )
    for estimate in (automatic, explicit, plain):
        assert estimate.width.input_qubits == 1
        assert estimate.gates.total == 3
        assert estimate.parameters == {}


def test_shape_alias_collision_keeps_quantum_ports_independent() -> None:
    """Generated shape aliases never capture another quantum port name."""
    attrs = {
        "resource_contract": {
            "quantum_operand_widths": [
                {"index": 0, "name": "signal", "width": 2},
                {"index": 1, "name": "signal_dim0", "width": 3},
            ]
        }
    }
    contracted = _port_shape_name_collision._clone_with_callable_attrs(attrs)

    symbolic = _port_shape_name_collision.estimate_resources()
    assert set(symbolic.parameters) == {
        "signal_dim0__shape",
        "signal_dim0_dim0",
    }
    automatic = contracted.estimate_resources()
    explicit = _port_shape_name_collision.estimate_resources(
        inputs={"signal": 2, "signal_dim0": 3}
    )
    assert automatic.width.input_qubits == 5
    assert explicit.width.input_qubits == 5
    with pytest.raises(ValueError, match="signal register width must equal 2"):
        contracted.estimate_resources(inputs={"signal": 1, "signal_dim0": 3})


def test_root_width_inference_skips_a_constant_array_dimension() -> None:
    """Fixed IR dimensions satisfy exact contracts without synthetic inputs."""
    dimension = Value(type=UIntType(), name="register_dim0").with_const(3)
    register = ArrayValue(
        type=QubitType(),
        name="register",
        shape=(dimension,),
    )
    kernel = SimpleNamespace(
        name="fixed_width",
        block=Block(
            name="fixed_width",
            label_args=["register"],
            input_values=[register],
        ),
        _callable_attrs_override={
            "resource_contract": {
                "quantum_operand_widths": [{"index": 0, "name": "register", "width": 3}]
            }
        },
    )

    estimate = qm.estimate_resources(kernel)

    assert estimate.width.input_qubits == 3
    assert estimate.parameters == {}
    assert qm.estimate_resources(kernel, inputs={"register": 3}).width == estimate.width
    assert (
        qm.estimate_resources(
            kernel,
            inputs={"register_dim0": 3},
        ).width
        == estimate.width
    )
    assert (
        qm.estimate_resources(
            kernel.block,
            inputs={"register": 3},
        ).width
        == estimate.width
    )
    with pytest.raises(ValueError, match="fixed at 3"):
        qm.estimate_resources(kernel, inputs={"register": 2})
    with pytest.raises(ValueError, match="fixed at 3"):
        qm.estimate_resources(kernel, inputs={"register_dim0": 2})
    with pytest.raises(ValueError, match="fixed at 3"):
        qm.estimate_resources(kernel.block, inputs={"register": 2})


@pytest.mark.parametrize("value", [1.5, "1", None])
def test_quantum_port_width_rejects_non_integer_scalars(value: object) -> None:
    """One-dimensional quantum ports diagnose invalid scalar widths directly."""
    encoding = _recursive_lcu_resource_encoding()

    with pytest.raises(ValueError, match="requires an integer width"):
        encoding.unitary.estimate_resources(inputs={"signal": value})


def test_block_encoding_width_contract_survives_select_and_serialization() -> None:
    """SELECT preserves every case callable's exact target-register ABI."""
    encoding = qm.periodic_shift_lcu_block_encoding(
        PeriodicShiftLCU.from_coefficients({}, register_sizes=(2,))
    )

    @qm.qkernel
    def circuit(
        index: qm.Qubit,
        signal: qm.Vector[qm.Qubit],
        system: qm.Vector[qm.Qubit],
    ) -> tuple[qm.Qubit, qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Select between two copies of one periodic block encoding."""
        return qm.select([encoding.unitary, encoding.unitary])(
            index,
            signal,
            system,
        )

    @qm.qkernel
    def inverse_circuit(
        index: qm.Qubit,
        signal: qm.Vector[qm.Qubit],
        system: qm.Vector[qm.Qubit],
    ) -> tuple[qm.Qubit, qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Invert the SELECT while preserving each case width contract."""
        return qm.inverse(circuit)(index, signal, system)

    @qm.qkernel
    def controlled_inverse_circuit(
        control: qm.Qubit,
        index: qm.Qubit,
        signal: qm.Vector[qm.Qubit],
        system: qm.Vector[qm.Qubit],
    ) -> tuple[
        qm.Qubit,
        qm.Qubit,
        qm.Vector[qm.Qubit],
        qm.Vector[qm.Qubit],
    ]:
        """Control the inverse SELECT and preserve its case width contracts."""
        return qm.control(inverse_circuit)(control, index, signal, system)

    candidates = (circuit, inverse_circuit, controlled_inverse_circuit)
    restored = tuple(deserialize(serialize(candidate)) for candidate in candidates)
    direct_estimate = qm.estimate_resources(
        circuit,
        inputs={"signal": 1, "system": 2},
    )
    inverse_estimate = qm.estimate_resources(
        inverse_circuit,
        inputs={"signal": 1, "system": 2},
    )
    assert inverse_estimate.gates == direct_estimate.gates
    assert inverse_estimate.width == direct_estimate.width
    assert inverse_estimate.depth == direct_estimate.depth

    for candidate in (*candidates, *restored):
        qm.estimate_resources(candidate, inputs={"signal": 1, "system": 2})
        with pytest.raises(ValueError, match="signal register width must equal 1"):
            qm.estimate_resources(candidate, inputs={"signal": 2, "system": 2})
        with pytest.raises(ValueError, match="system register width must equal 2"):
            qm.estimate_resources(candidate, inputs={"signal": 1, "system": 3})


def test_static_block_encoding_binding_preserves_width_contract() -> None:
    """A descriptor-bound member call enforces its concrete ABI widths."""
    encoding = qm.periodic_shift_lcu_block_encoding(
        PeriodicShiftLCU.from_coefficients({}, register_sizes=(2,))
    )

    @qm.qkernel
    def apply_encoding(
        descriptor: qm.LCUBlockEncoding,
        signal: qm.Vector[qm.Qubit],
        system: qm.Vector[qm.Qubit],
    ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
        """Apply a compile-time block-encoding descriptor."""
        return descriptor.unitary(signal, system)

    correct = {"descriptor": encoding, "signal": 1, "system": 2}
    apply_encoding.estimate_resources(inputs=correct)
    with pytest.raises(ValueError, match="signal register width must equal 1"):
        apply_encoding.estimate_resources(inputs={**correct, "signal": 2})
    with pytest.raises(ValueError, match="system register width must equal 2"):
        apply_encoding.estimate_resources(inputs={**correct, "system": 3})


def test_fixed_width_opaque_vector_validates_total_target_width() -> None:
    """Legacy opaque metadata rejects symbolic vectors with the wrong width."""
    oracle = qm.opaque("three_qubit_oracle", num_qubits=3)

    @qm.qkernel
    def circuit(register: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
        """Apply a fixed-width bodyless oracle to a symbolic register."""
        return oracle(register)

    estimate = circuit.estimate_resources(
        inputs={"register": 3},
        unknown_policy=qm.UnknownResourcePolicy.OPAQUE_CALL,
    )
    assert estimate.calls.calls_by_name == {"three_qubit_oracle": 1}
    with pytest.raises(ValueError, match="target register width must equal 3"):
        circuit.estimate_resources(
            inputs={"register": 2},
            unknown_policy=qm.UnknownResourcePolicy.OPAQUE_CALL,
        )
    with pytest.raises(ValueError, match="target register width must equal 3"):
        circuit.estimate_resources(
            inputs={"register": 4},
            unknown_policy=qm.UnknownResourcePolicy.OPAQUE_CALL,
        )
