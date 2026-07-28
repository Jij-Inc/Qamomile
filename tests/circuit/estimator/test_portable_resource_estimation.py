"""Regression tests for portable algorithmic resource estimation."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qm
import qamomile.observable as qm_o
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.gate import ProjectOperation
from qamomile.circuit.ir.operation.operation import (
    Operation,
    OperationKind,
    Signature,
)
from qamomile.circuit.ir.types.primitives import QubitType, UIntType
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

    def __call__(self, ctx: qm.OpaqueCallContext) -> qm.ResourceEstimate:
        """Build a basis-compatible callback cost.

        Args:
            ctx (qm.OpaqueCallContext): Active estimator context.

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
            precision=ctx.precision,
        )
        assert symbolic.parameters == {"callback_work": callback_work}
        return symbolic.substitute(callback_work=1)


class _AuthoritativeControlOpaqueCost:
    """Price all explicit and inherited controls exactly once."""

    def __call__(self, ctx: qm.OpaqueCallContext) -> qm.ResourceEstimate:
        """Build a cost that exposes every control-context component.

        Args:
            ctx (qm.OpaqueCallContext): Complete opaque invocation context.

        Returns:
            qm.ResourceEstimate: Complete callback-owned invocation cost.
        """
        gate_count = ctx.total_controls + 1
        return qm.ResourceEstimate(
            gates=qm.GateResources(
                total=gate_count,
                single_qubit=gate_count,
            ),
            calls=qm.CallResources(
                calls_by_name={
                    f"own_controls={ctx.own_controls}": 1,
                    f"surrounding_controls={ctx.controls}": 1,
                    f"total_controls={ctx.total_controls}": 1,
                }
            ),
        )


class _AuthoritativeArityProfileOpaqueCost:
    """Return one complete arity profile already priced by the callback."""

    def __call__(self, ctx: qm.OpaqueCallContext) -> qm.ResourceEstimate:
        """Return a callback-owned one-/two-qubit gate profile.

        Args:
            ctx (qm.OpaqueCallContext): Complete opaque invocation context.

        Returns:
            qm.ResourceEstimate: Complete profile that needs no further
            control projection.
        """
        return qm.ResourceEstimate(
            gates=qm.GateResources(
                total=3,
                single_qubit=2,
                two_qubit=1,
            ),
            calls=qm.CallResources(
                calls_by_name={
                    f"authoritative_total_controls={ctx.total_controls}": 1,
                }
            ),
        )


class _LogicalOnlyOpaqueCost:
    """Return a deliberately logical-basis callback cost."""

    def __call__(self, ctx: qm.OpaqueCallContext) -> qm.ResourceEstimate:
        """Ignore the requested basis to exercise provenance validation.

        Args:
            ctx (qm.OpaqueCallContext): Active estimator context.

        Returns:
            qm.ResourceEstimate: One logical basis-sensitive gate.
        """
        del ctx
        return qm.ResourceEstimate(
            gates=qm.GateResources(total=1, single_qubit=1),
            basis=qm.GateBasis.LOGICAL,
        )


class _BasisNeutralOpaqueCost:
    """Return a call-only cost that is independent of gate basis."""

    def __call__(self, ctx: qm.OpaqueCallContext) -> qm.ResourceEstimate:
        """Return one explicitly modeled opaque call.

        Args:
            ctx (qm.OpaqueCallContext): Active estimator context.

        Returns:
            qm.ResourceEstimate: Basis-neutral call resources.
        """
        del ctx
        return qm.ResourceEstimate(
            calls=qm.CallResources(calls_by_name={"neutral_oracle": 1})
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
    assert estimate.depth.depth == expected_depth
    assert estimate.depth.measurement_depth == 1


def test_portable_controlled_qkernel_shares_body_control_ladder() -> None:
    """A controlled multi-gate body computes one shared control conjunction."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Apply a four-gate body under three coherent controls."""
        controls = qm.qubit_array(3, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(_four_h_body, num_controls=3)(controls, target)
        return target

    portable = circuit.estimate_resources()
    abstract = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert portable.gates.total == 8
    assert portable.gates.two_qubit == 4
    assert portable.gates.multi_qubit == 4
    assert portable.gates.toffoli == 4
    assert portable.width.allocated_qubits == 4
    assert portable.width.clean_ancilla_qubits == 2
    assert portable.width.peak_qubits == 6
    assert portable.width.circuit_qubits == 6
    assert portable.quality is qm.EstimateQuality.UPPER_BOUND

    assert abstract.gates.total == 4
    assert abstract.gates.multi_qubit == 4
    assert abstract.width.clean_ancilla_qubits == 0
    assert abstract.width.peak_qubits == 4


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


def test_portable_controlled_loop_hoists_shared_ladder() -> None:
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


def test_two_control_native_x_body_skips_shared_ladder() -> None:
    """Two-control X leaves remain direct Toffolis without an ancilla."""

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

    assert estimate.gates.total == 2
    assert estimate.gates.multi_qubit == 2
    assert estimate.gates.toffoli == 2
    assert estimate.width.clean_ancilla_qubits == 0
    assert estimate.width.peak_qubits == 4


@pytest.mark.parametrize(
    ("num_controls", "total", "toffoli", "clean_ancillas"),
    [
        (1, 1, 0, 0),
        (2, 1, 1, 0),
        (3, 5, 4, 2),
        (5, 9, 8, 4),
    ],
)
def test_portable_controlled_x_scales_with_control_arity(
    num_controls: int,
    total: int,
    toffoli: int,
    clean_ancillas: int,
) -> None:
    """Multi-controlled X reports fallback gates and reusable ancillas.

    Args:
        num_controls (int): Number of coherent controls.
        total (int): Expected portable gate count.
        toffoli (int): Expected Toffoli count.
        clean_ancillas (int): Expected clean fallback ancillas.
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


def test_nested_open_control_brackets_inherit_outer_controls() -> None:
    """Open-control X brackets are controlled by the surrounding region."""

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

    assert estimate.gates.total == 17
    assert estimate.gates.toffoli == 14
    assert estimate.width.clean_ancilla_qubits == 3
    assert estimate.width.allocated_qubits == 5
    assert estimate.width.peak_qubits == 8


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
    basis: qm.GateBasis = qm.GateBasis.PORTABLE,
    precision: float = 1e-10,
) -> qm.ResourceEstimate:
    """Return the estimate for a pure phase under three controls.

    Args:
        phase (float): Global phase in radians.
        basis (qm.GateBasis): Requested resource basis. Defaults to portable.
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
        controls = qm.qubit_array(3, "controls")
        target = qm.qubit("target")
        *_, target = qm.control(phased, num_controls=3)(controls, target)
        return target

    return circuit.estimate_resources(basis=basis, precision=precision)


def test_controlled_global_phase_uses_portable_phase_lowering() -> None:
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
    """Portable estimation preserves phases below the old fixed tolerance."""
    estimate = _controlled_phase_estimate(5e-13)

    assert estimate.gates.total > 0
    assert estimate.gates.rotation == 1


def test_clifford_t_supports_multi_controlled_global_phase() -> None:
    """Fixed and arbitrary phases lower through multi-control fallbacks."""
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
    assert arbitrary.gates.t > fixed.gates.t
    assert arbitrary.width.clean_ancilla_qubits == 1
    assert arbitrary.quality is qm.EstimateQuality.UPPER_BOUND


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

    portable = circuit.estimate_resources()
    assert set(portable.parameters) == {"theta"}
    assert portable.substitute(theta=0).gates.total == 0
    assert portable.substitute(theta=math.pi).gates.rotation == 1

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
    assert arbitrary.quality is qm.EstimateQuality.UPPER_BOUND


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


def test_pauli_evolve_zero_time_specialization_removes_all_resources() -> None:
    """Zero time removes the gadget but not a statically selected shared ladder."""
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
    assert controlled_zero.gates.total == 4
    assert controlled_zero.gates.toffoli == 4
    assert controlled_zero.depth.depth == 4
    assert controlled_zero.width.clean_ancilla_qubits == 2
    assert controlled_zero.width.peak_qubits == 6
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
    assert opaque.quality is qm.EstimateQuality.MODELED
    assert zero.calls.calls_by_name == {}
    assert zero.gates.total == 0
    assert zero.assumptions

    bound = circuit.estimate_resources(inputs={"hamiltonian": qm_o.Z(1)})
    with pytest.raises(ValueError, match="at least 2 qubits"):
        bound.substitute(width=1)
    with pytest.raises(ValueError, match="at least 2 qubits"):
        circuit.estimate_resources(inputs={"width": 1, "hamiltonian": qm_o.Z(1)})
    assert bound.substitute(width=3).gates.total == 1


@pytest.mark.parametrize(
    ("num_controls", "expected_total", "expected_clean_ancillas"),
    [
        (1, 5, 1),
        (2, 13, 2),
        (4, 25, 4),
    ],
)
def test_controlled_fixed_opaque_cost_projects_complete_arity_profile(
    num_controls: int,
    expected_total: int,
    expected_clean_ancillas: int,
) -> None:
    """Complete opaque arity counts receive a portable controlled upper bound."""
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
    assert estimate.quality is qm.EstimateQuality.MODELED
    assert any(
        "complete one- and two-qubit counts" in assumption.message
        for assumption in estimate.assumptions
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

    assert estimate.gates.total == 15
    assert estimate.depth.depth == 15
    assert estimate.width.clean_ancilla_qubits == 2
    assert estimate.calls.queries_by_name == {"open_arity_oracle": 1}


def test_incomplete_opaque_arity_profile_stays_explicitly_unprojected() -> None:
    """An unclassified gate remainder preserves cost with a clear assumption."""
    oracle = qm.opaque(
        "incomplete_arity_oracle",
        num_qubits=1,
        cost=qm.ResourceEstimate(
            gates=qm.GateResources(
                total=3,
                single_qubit=2,
            ),
        ),
    )

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Control an opaque cost whose arity profile has one unknown gate."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        _, target = qm.control(oracle)(control, target)
        return target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.gates.single_qubit == 2
    assert any(
        "not fully partitioned" in assumption.message
        for assumption in estimate.assumptions
    )


def test_opaque_control_projection_adds_to_declared_clean_workspace() -> None:
    """Fallback ancillas remain separate from opaque scratch workspace."""
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
        quality=qm.EstimateQuality.MODELED,
    )

    symbolic = base.controlled(controls)
    zero = symbolic.substitute(controls=0)

    assert zero.width == base.width
    assert zero.gates == base.gates
    assert zero.depth == base.depth
    assert zero.calls == base.calls
    assert zero.assumptions == base.assumptions
    assert zero.quality is base.quality
    assert symbolic.substitute(controls=2).gates.total == 13


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

    assert valid.gates.total == 6
    assert "rotations" in estimate.parameters
    assert any(
        requirement["label"]
        == "Portable aggregate rotation count within total gate count"
        for requirement in estimate.to_dict()["requirements"]
    )
    with pytest.raises(ValueError, match="rotation count within total"):
        estimate.substitute(gates=1, rotations=2, controls=2)
    assert estimate.substitute(gates=1, rotations=2, controls=0).gates.rotation == 2


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
    assert estimate.quality is qm.EstimateQuality.MODELED
    assert any(
        "no declared one- or two-qubit gate profile" in assumption.message
        for assumption in estimate.assumptions
    )
    assert not any(
        "complete one- and two-qubit counts" in assumption.message
        for assumption in estimate.assumptions
    )


def test_zero_aggregate_cost_does_not_claim_controlled_upper_bound() -> None:
    """An empty aggregate cannot reveal hidden controlled global phase."""
    estimate = qm.ResourceEstimate.zero().controlled(3)

    assert estimate.gates.total == 0
    assert estimate.quality is qm.EstimateQuality.MODELED
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
    assert estimate.quality is qm.EstimateQuality.MODELED
    assert any("exceeds the declared" in item.message for item in estimate.assumptions)


def test_opaque_callback_prices_own_and_surrounding_controls_once() -> None:
    """A callback owns explicit and inherited control pricing exactly once."""
    oracle = qm.opaque(
        "authoritative_control_oracle",
        num_qubits=1,
        num_control_qubits=1,
        cost=_AuthoritativeControlOpaqueCost(),
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

    assert estimate.gates.total == 4
    assert estimate.gates.single_qubit == 4
    assert estimate.calls.calls_by_name == {
        "own_controls=1": 1,
        "surrounding_controls=2": 1,
        "total_controls=3": 1,
    }
    assert not any(
        "controlled aggregate cost" in assumption.message
        for assumption in estimate.assumptions
    )


def test_opaque_callback_arity_profile_is_not_controlled_twice() -> None:
    """A complete callback-owned arity profile receives no second projection."""
    oracle = qm.opaque(
        "authoritative_arity_oracle",
        num_qubits=1,
        cost=_AuthoritativeArityProfileOpaqueCost(),
    )

    @qm.qkernel
    def invoke_oracle(target: qm.Qubit) -> qm.Qubit:
        """Invoke the direct opaque oracle."""
        (target,) = oracle(target)
        return target

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Invoke the oracle under two inherited controls."""
        control_0 = qm.qubit("control_0")
        control_1 = qm.qubit("control_1")
        target = qm.qubit("target")
        return qm.control(
            invoke_oracle,
            num_controls=2,
        )(control_0, control_1, target)

    estimate = circuit.estimate_resources()

    assert estimate.gates == qm.GateResources(
        total=3,
        single_qubit=2,
        two_qubit=1,
    )
    assert estimate.calls.calls_by_name == {
        "authoritative_total_controls=2": 1,
    }
    assert not any(
        "controlled aggregate cost" in assumption.message
        for assumption in estimate.assumptions
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

    def nested_cost(ctx: qm.OpaqueCallContext) -> qm.ResourceEstimate:
        """Compose a nested estimate with a callback-local symbolic cost.

        Args:
            ctx (qm.OpaqueCallContext): Active opaque invocation context.

        Returns:
            qm.ResourceEstimate: Three concrete one-qubit gates.
        """
        nested = symbolic_body.estimate_resources(basis=ctx.basis)
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


def test_opaque_callback_cost_validates_basis_and_precision_provenance() -> None:
    """Callback costs receive provenance and cannot be silently relabeled."""
    compatible = qm.opaque(
        "compatible_oracle",
        num_qubits=1,
        cost=_ContextAwareOpaqueCost(),
    )
    incompatible = qm.opaque(
        "incompatible_oracle",
        num_qubits=1,
        cost=_LogicalOnlyOpaqueCost(),
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
    def incompatible_circuit() -> qm.Qubit:
        """Invoke the logical-only opaque callable."""
        (target,) = incompatible(qm.qubit("target"))
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
    with pytest.raises(ValueError, match="uses basis 'logical'"):
        incompatible_circuit.estimate_resources()


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


def test_quality_and_basis_provenance_survive_resource_algebra() -> None:
    """Modeled quality and estimator provenance remain visible to users."""
    modeled = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        quality=qm.EstimateQuality.MODELED,
    )

    transformed = modeled.controlled(2).inverse().repeat(3).choice(modeled)
    assert transformed.quality is qm.EstimateQuality.MODELED

    default = _four_h_body.estimate_resources()
    lowered = _four_h_body.estimate_resources(
        basis=qm.GateBasis.CLIFFORD_T,
        precision=1e-4,
    )
    assert default.basis is qm.GateBasis.PORTABLE
    assert default.precision is None
    assert default.to_dict()["basis"] == "portable"
    assert default.to_dict()["width"]["circuit_qubits"] == "1"
    assert lowered.basis is qm.GateBasis.CLIFFORD_T
    assert lowered.precision == 1e-4
    assert lowered.to_dict()["precision"] == 1e-4


def test_resource_algebra_rejects_mixed_gate_bases() -> None:
    """Basis-sensitive estimates cannot be silently combined and relabeled."""
    portable = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        basis=qm.GateBasis.PORTABLE,
    )
    logical = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        basis=qm.GateBasis.LOGICAL,
    )

    with pytest.raises(ValueError, match="different gate bases"):
        portable.seq(logical)


def test_basis_provenance_does_not_simplify_accumulated_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resource-algebra fold does not simplify its full symbolic history."""
    flag = sp.Symbol("flag", integer=True, nonnegative=True)
    conditional_count = sp.Piecewise((1, sp.Eq(flag, 1)), (0, True))
    portable = qm.ResourceEstimate(
        gates=qm.GateResources(total=conditional_count),
        basis=qm.GateBasis.PORTABLE,
    )
    neutral = qm.ResourceEstimate(
        depth=qm.DepthResources(depth=1, measurement_depth=1),
        basis=qm.GateBasis.LOGICAL,
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

    combined = portable.seq(neutral)

    assert combined.basis is qm.GateBasis.PORTABLE
    assert combined.gates.total == conditional_count
    assert combined.depth.measurement_depth == 1


def test_measurement_only_resources_are_gate_basis_neutral() -> None:
    """Measurement depth composes with a gate estimate from any basis."""
    measurement = qm.ResourceEstimate(
        depth=qm.DepthResources(depth=1, measurement_depth=1),
        basis=qm.GateBasis.LOGICAL,
    )
    portable = qm.ResourceEstimate(
        gates=qm.GateResources(total=1, single_qubit=1),
        depth=qm.DepthResources(depth=1, clifford_depth=1),
        basis=qm.GateBasis.PORTABLE,
    )

    combined = measurement.seq(portable)

    assert combined.basis is qm.GateBasis.PORTABLE
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
    assert inverse_estimate.quality is direct_estimate.quality
    assert direct_estimate.width.input_qubits == 3
    assert direct_estimate.width.peak_qubits == 3

    assert controlled_estimate.gates == qm.GateResources(
        total=17,
        single_qubit=3,
        two_qubit=9,
        multi_qubit=5,
        clifford=6,
        rotation=6,
        t=0,
        toffoli=5,
        non_clifford=11,
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
