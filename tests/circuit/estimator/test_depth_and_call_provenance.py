"""Regression tests for physical-wire depth and call-boundary provenance."""

from __future__ import annotations

import dataclasses

import pytest
import sympy as sp

import qamomile.circuit as qm
import qamomile.observable as qm_o
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import Value


@qm.qkernel
def _runtime_branch_helper(target: qm.Qubit, measured: qm.Bit) -> qm.Qubit:
    """Apply a one- or two-layer branch selected by a runtime bit."""
    if measured:
        target = qm.h(target)
        target = qm.z(target)
    else:
        target = qm.x(target)
    return target


@qm.qkernel
def _identity_runtime_case(target: qm.Qubit, measured: qm.Bit) -> qm.Qubit:
    """Return a SELECT target while accepting the shared runtime bit."""
    return target


@qm.qkernel
def _runtime_invoke_circuit() -> qm.Qubit:
    """Pass a measurement result through an ordinary qkernel boundary."""
    predicate = qm.h(qm.qubit("predicate"))
    measured = qm.measure(predicate)
    return _runtime_branch_helper(qm.qubit("target"), measured)


@qm.qkernel
def _runtime_controlled_circuit() -> tuple[qm.Qubit, qm.Qubit]:
    """Pass a measurement result into a coherently controlled helper."""
    predicate = qm.h(qm.qubit("predicate"))
    measured = qm.measure(predicate)
    control = qm.qubit("control")
    target = qm.qubit("target")
    return qm.control(_runtime_branch_helper)(control, target, measured)


@qm.qkernel
def _runtime_select_circuit() -> tuple[qm.Qubit, qm.Qubit]:
    """Pass a measurement result into one SELECT case body."""
    predicate = qm.h(qm.qubit("predicate"))
    measured = qm.measure(predicate)
    index = qm.qubit("index")
    target = qm.qubit("target")
    return qm.select([_runtime_branch_helper, _identity_runtime_case])(
        index,
        target,
        measured=measured,
    )


@qm.qkernel
def _left_only(left: qm.Qubit, right: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
    """Touch only the first of two pass-through quantum arguments."""
    return qm.h(left), right


@qm.qkernel
def _identity_pair(
    left: qm.Qubit,
    right: qm.Qubit,
) -> tuple[qm.Qubit, qm.Qubit]:
    """Return two quantum arguments without touching either one."""
    return left, right


@qm.qkernel
def _uneven_pair(
    left: qm.Qubit,
    right: qm.Qubit,
) -> tuple[qm.Qubit, qm.Qubit]:
    """Finish two independent arguments at different internal layers."""
    left = qm.h(left)
    left = qm.z(left)
    right = qm.x(right)
    return left, right


@qm.qkernel
def _fresh_hadamard() -> qm.Qubit:
    """Allocate, touch, and return one callee-owned qubit."""
    return qm.h(qm.qubit("fresh"))


@qm.qkernel
def _identity_qubit(target: qm.Qubit) -> qm.Qubit:
    """Return one qubit without touching it."""
    return target


@qm.qkernel
def _single_hadamard(target: qm.Qubit) -> qm.Qubit:
    """Apply one Hadamard gate to a scalar target."""
    return qm.h(target)


@qm.qkernel
def _phased_identity(target: qm.Qubit) -> qm.Qubit:
    """Apply a global phase while passing the target through unchanged."""
    return qm.global_phase(_identity_qubit, 0.25)(target)


@qm.qkernel
def _left_call_with_sibling_gate() -> tuple[qm.Qubit, qm.Qubit]:
    """Run a left-only nested call beside a gate on its idle argument."""
    left = qm.qubit("left")
    right = qm.qubit("right")
    left, right = _left_only(left, right)
    right = qm.x(right)
    return left, right


@qm.qkernel
def _vector_hx_measure(width: qm.UInt) -> qm.Vector[qm.Bit]:
    """Apply two broadcast layers and measure a symbolic-width register."""
    targets = qm.qubit_array(width, "targets")
    targets = qm.h(targets)
    targets = qm.x(targets)
    return qm.measure(targets)


@qm.qkernel
def _resource_recursive_leaf(target: qm.Qubit) -> qm.Qubit:
    """Apply the one gate reached by the recursive base case."""
    return qm.h(target)


@qm.qkernel
def _resource_recursive_body(k: qm.UInt, target: qm.Qubit) -> qm.Qubit:
    """Recurse toward zero before applying one base-case gate."""
    if k == 0:
        target = _resource_recursive_leaf(target)
    else:
        target = _resource_recursive_body(k - 1, target)
    return target


@qm.qkernel
def _resource_recursive_circuit(k: qm.UInt) -> qm.Bit:
    """Allocate and measure the target of a self-recursive qkernel."""
    target = _resource_recursive_body(k, qm.qubit("target"))
    return qm.measure(target)


def test_array_aliases_share_physical_wire_dependencies() -> None:
    """Element gates precede whole-register measurement without serializing siblings."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Bit]:
        """Gate two independent elements before measuring their root register."""
        register = qm.qubit_array(3, "register")
        register[0] = qm.h(register[0])
        register[1] = qm.x(register[1])
        return qm.measure(register)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 2
    assert estimate.depth.depth == 2
    assert estimate.depth.clifford_depth == 1
    assert estimate.depth.measurement_depth == 1


def test_array_view_aliases_its_root_register() -> None:
    """An element reached through a slice precedes root-register measurement."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Bit]:
        """Gate an element through a view and then measure that view."""
        register = qm.qubit_array(3, "register")
        view = register[1:3]
        view[0] = qm.h(view[0])
        return qm.measure(view)

    assert circuit.estimate_resources().depth.depth == 2


def test_disjoint_concrete_array_view_does_not_alias_the_whole_root() -> None:
    """A resolved view expands to only its physical root-array slots."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Bit]:
        """Measure slots two and three beside a gate on slot zero."""
        register = qm.qubit_array(4, "register")
        register[0] = qm.h(register[0])
        return qm.measure(register[2:4])

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_supplied_array_index_sharpens_depth_dependencies() -> None:
    """Direct inputs resolve physical indices while symbolic estimates stay safe."""

    @qm.qkernel
    def circuit(index: qm.UInt) -> qm.Vector[qm.Qubit]:
        """Apply two gates whose aliasing depends on a supplied index."""
        register = qm.qubit_array(2, "register")
        register[0] = qm.h(register[0])
        register[index] = qm.x(register[index])
        return register

    disjoint = circuit.estimate_resources(
        inputs={"index": 1},
        basis=qm.GateBasis.LOGICAL,
    )
    overlapping = circuit.estimate_resources(
        inputs={"index": 0},
        basis=qm.GateBasis.LOGICAL,
    )
    symbolic = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)
    substituted = symbolic.substitute(index=1)

    assert disjoint.depth.depth == 1
    assert disjoint.quality is qm.EstimateQuality.EXACT
    assert overlapping.depth.depth == 2
    assert overlapping.quality is qm.EstimateQuality.EXACT
    assert symbolic.depth.depth == 2
    assert symbolic.quality is qm.EstimateQuality.UPPER_BOUND
    assert substituted.depth.depth == 2
    assert substituted.quality is qm.EstimateQuality.UPPER_BOUND


def test_symbolic_vector_broadcast_has_layer_depth_not_element_depth() -> None:
    """Injective affine broadcast loops parallelize across vector slots."""
    width = sp.Symbol("width", integer=True, nonnegative=True)

    symbolic = _vector_hx_measure.estimate_resources(basis=qm.GateBasis.LOGICAL)
    concrete = _vector_hx_measure.estimate_resources(
        inputs={"width": 3},
        basis=qm.GateBasis.LOGICAL,
    )
    empty = _vector_hx_measure.estimate_resources(
        inputs={"width": 0},
        basis=qm.GateBasis.LOGICAL,
    )

    assert symbolic.gates.total == 2 * width
    assert symbolic.depth.depth.subs(width, 3) == 3
    assert symbolic.depth.depth.subs(width, 0) == 0
    assert symbolic.quality is qm.EstimateQuality.EXACT
    assert concrete.gates.total == 6
    assert concrete.depth.depth == 3
    assert concrete.quality is qm.EstimateQuality.EXACT
    assert empty.gates.total == 0
    assert empty.depth.depth == 0
    assert empty.quality is qm.EstimateQuality.EXACT


def test_concrete_loop_parallelizes_disjoint_array_elements() -> None:
    """Concrete iterations on distinct slots share one gate layer."""

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Bit]:
        """Apply one Hadamard per array slot before a vector measurement."""
        register = qm.qubit_array(4, "register")
        for index in qm.range(4):
            register[index] = qm.h(register[index])
        return qm.measure(register)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 4
    assert estimate.depth.depth == 2
    assert estimate.depth.clifford_depth == 1
    assert estimate.depth.measurement_depth == 1


def test_reusable_clean_ancilla_pool_serializes_independent_fallbacks() -> None:
    """Depth and reusable-width estimates describe one realizable schedule."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply independent three-control X fallbacks to two targets."""
        left_controls = qm.qubit_array(3, "left_controls")
        left_target = qm.qubit("left_target")
        *_, left_target = qm.control(qm.x, num_controls=3)(
            left_controls,
            left_target,
        )
        right_controls = qm.qubit_array(3, "right_controls")
        right_target = qm.qubit("right_target")
        *_, right_target = qm.control(qm.x, num_controls=3)(
            right_controls,
            right_target,
        )
        return left_target, right_target

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 10
    assert estimate.width.clean_ancilla_qubits == 2
    assert estimate.depth.depth == 10


def test_ordinary_call_uses_only_body_touched_arguments_for_depth() -> None:
    """An unused pass-through argument remains parallel with the call body."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Apply a body to the left qubit and a sibling gate to the right."""
        left = qm.qubit("left")
        right = qm.qubit("right")
        left, right = _left_only(left, right)
        right = qm.x(right)
        return left, right

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_inverse_call_uses_only_body_touched_arguments_for_depth() -> None:
    """Inverse body traversal preserves its caller-mapped touched-wire mask."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Invert a left-only body while gating the unused right argument."""
        left = qm.qubit("left")
        right = qm.qubit("right")
        left, right = qm.inverse(_left_only)(left, right)
        right = qm.x(right)
        return left, right

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_controlled_call_uses_only_control_and_body_touched_arguments() -> None:
    """A controlled body does not block an unused pass-through target."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Control the left-only body and gate its unused right argument."""
        control = qm.qubit("control")
        left = qm.qubit("left")
        right = qm.qubit("right")
        control, left, right = qm.control(_left_only)(control, left, right)
        right = qm.x(right)
        return control, left, right

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_returned_callee_allocation_blocks_its_caller_consumer() -> None:
    """A touched fresh output carries body latency across the call boundary."""

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Consume a qubit allocated and prepared by a nested qkernel."""
        target = _fresh_hadamard()
        return qm.z(target)

    assert circuit.estimate_resources(basis=qm.GateBasis.LOGICAL).depth.depth == 2


def test_multi_wire_call_boundary_reports_conservative_depth_quality() -> None:
    """Unequal body exit layers are disclosed as an upper-bound estimate."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Gate the earlier-finishing output of a two-path nested body."""
        left = qm.qubit("left")
        right = qm.qubit("right")
        left, right = _uneven_pair(left, right)
        right = qm.h(right)
        return left, right

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 3
    assert estimate.quality is qm.EstimateQuality.UPPER_BOUND
    assert any("aggregate latency" in note.message for note in estimate.assumptions)


def test_multi_wire_if_boundary_reports_conservative_depth_quality() -> None:
    """A selected multi-path branch discloses aggregate exit latency."""

    @qm.qkernel
    def circuit(flag: qm.UInt) -> tuple[qm.Qubit, qm.Qubit]:
        """Gate the earlier-finishing output after a compile-time branch."""
        left = qm.qubit("left")
        right = qm.qubit("right")
        if flag:
            left = qm.h(left)
            left = qm.z(left)
            right = qm.x(right)
        right = qm.h(right)
        return left, right

    estimate = circuit.estimate_resources(
        inputs={"flag": 1},
        basis=qm.GateBasis.LOGICAL,
    )

    assert estimate.depth.depth == 3
    assert estimate.quality is qm.EstimateQuality.UPPER_BOUND
    assert any("control-flow boundary" in note.message for note in estimate.assumptions)


def test_multi_wire_for_boundary_reports_conservative_depth_quality() -> None:
    """A loop with unequal wire exits is explicitly an upper-bound boundary."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Gate the earlier-finishing output after one loop iteration."""
        left = qm.qubit("left")
        right = qm.qubit("right")
        for _index in qm.range(1):
            left = qm.h(left)
            left = qm.z(left)
            right = qm.x(right)
        right = qm.h(right)
        return left, right

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 3
    assert estimate.quality is qm.EstimateQuality.UPPER_BOUND
    assert any("control-flow boundary" in note.message for note in estimate.assumptions)


def test_select_does_not_block_an_unused_pass_through_target() -> None:
    """SELECT maps each case's touched targets plus its index controls."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
        """Gate a target unused by every non-identity SELECT case."""
        index = qm.qubit("index")
        left = qm.qubit("left")
        right = qm.qubit("right")
        index, left, right = qm.select([_left_only, _identity_pair])(
            index,
            left,
            right,
        )
        right = qm.x(right)
        return index, left, right

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 3
    assert estimate.quality is qm.EstimateQuality.UPPER_BOUND


def test_selected_control_pool_slot_does_not_block_a_sibling_slot() -> None:
    """Only resolved control_indices entries participate in call depth."""

    @qm.qkernel
    def circuit(
        width: qm.UInt,
        index: qm.UInt,
    ) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
        """Control through a selected pool slot and gate slot one."""
        pool = qm.qubit_array(2, "pool")
        target = qm.qubit("target")
        pool, target = qm.control(
            _single_hadamard,
            num_controls=width,
        )(pool, target, control_indices=(index,))
        pool[1] = qm.x(pool[1])
        return pool, target

    disjoint = circuit.estimate_resources(
        inputs={"width": 1, "index": 0},
        basis=qm.GateBasis.LOGICAL,
    )
    overlapping = circuit.estimate_resources(
        inputs={"width": 1, "index": 1},
        basis=qm.GateBasis.LOGICAL,
    )
    symbolic = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)
    substituted = symbolic.substitute(width=1, index=0)

    assert disjoint.depth.depth == 1
    assert disjoint.quality is qm.EstimateQuality.EXACT
    assert overlapping.depth.depth == 2
    assert overlapping.quality is qm.EstimateQuality.EXACT
    assert symbolic.depth.depth == 2
    assert symbolic.quality is qm.EstimateQuality.UPPER_BOUND
    assert substituted.depth.depth == 2
    assert substituted.quality is qm.EstimateQuality.UPPER_BOUND


def test_zero_power_body_does_not_join_independent_wire_timelines() -> None:
    """A specialized zero-depth controlled call is absent from scheduling."""

    @qm.qkernel
    def circuit(
        width: qm.UInt,
        power: qm.UInt,
    ) -> tuple[qm.Vector[qm.Qubit], qm.Qubit, qm.Qubit]:
        """Place a potentially multi-wire call between independent gates."""
        controls = qm.qubit_array(1, "controls")
        left = qm.h(qm.qubit("left"))
        right = qm.qubit("right")
        controls, left, right = qm.control(
            _uneven_pair,
            num_controls=width,
        )(controls, left, right, power=power)
        right = qm.x(right)
        return controls, left, right

    estimate = circuit.estimate_resources(
        inputs={"width": 1, "power": 0},
        basis=qm.GateBasis.LOGICAL,
    )
    substituted = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL).substitute(
        width=1, power=0
    )

    assert estimate.depth.depth == 1
    assert substituted.depth.depth == 1
    assert substituted.quality is qm.EstimateQuality.EXACT


def test_zero_iteration_loop_preserves_independent_wire_timelines() -> None:
    """A symbolic loop becomes dependency-neutral when specialized to zero."""

    @qm.qkernel
    def circuit(iterations: qm.UInt) -> tuple[qm.Qubit, qm.Qubit]:
        """Place a multi-wire loop between two otherwise parallel gates."""
        left = qm.h(qm.qubit("left"))
        right = qm.qubit("right")
        for _index in qm.range(iterations):
            left = qm.h(left)
            left = qm.z(left)
            right = qm.x(right)
        right = qm.z(right)
        return left, right

    direct = circuit.estimate_resources(
        inputs={"iterations": 0},
        basis=qm.GateBasis.LOGICAL,
    )
    substituted = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL).substitute(
        iterations=0
    )

    assert direct.depth.depth == 1
    assert substituted.depth.depth == 1
    assert direct.quality is qm.EstimateQuality.EXACT
    assert substituted.quality is qm.EstimateQuality.EXACT


def test_zero_depth_classical_operation_does_not_disable_wire_scheduling() -> None:
    """Classical arithmetic between disjoint gates is depth-neutral."""

    @qm.qkernel
    def circuit(value: qm.UInt) -> tuple[qm.Qubit, qm.Qubit]:
        """Create an unused classical BinOp between independent gates."""
        left = qm.h(qm.qubit("left"))
        _unused = value + 1
        right = qm.x(qm.qubit("right"))
        return left, right

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 1


def test_serialized_return_operation_is_depth_neutral() -> None:
    """A restored trailing ReturnOperation keeps dependency scheduling enabled."""
    from qamomile.circuit.serialization import deserialize, serialize

    original = _left_call_with_sibling_gate.estimate_resources(
        basis=qm.GateBasis.LOGICAL
    )
    restored = deserialize(serialize(_left_call_with_sibling_gate))
    restored_estimate = qm.ResourceEstimator(basis=qm.GateBasis.LOGICAL).estimate(
        restored.block
    )

    assert original.depth.depth == 1
    assert restored_estimate.depth.depth == 1


def test_open_control_boundary_discloses_aggregate_depth_bound() -> None:
    """Open-control exit latency is conservatively classified as upper-bound."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Gate the target while the final open-control bracket can run."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        control, target = qm.control(
            _single_hadamard,
            control_value=0,
        )(control, target)
        target = qm.z(target)
        return control, target

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 4
    assert estimate.quality is qm.EstimateQuality.UPPER_BOUND
    assert any("open-control" in note.message for note in estimate.assumptions)


def test_controlled_global_phase_does_not_block_pass_through_target() -> None:
    """A controlled relative phase depends on its control, not an idle target."""

    @qm.qkernel
    def circuit() -> tuple[qm.Qubit, qm.Qubit]:
        """Run a target gate beside a control-only relative phase."""
        control = qm.qubit("control")
        target = qm.qubit("target")
        control, target = qm.control(_phased_identity)(control, target)
        target = qm.x(target)
        return control, target

    estimate = circuit.estimate_resources(basis=qm.GateBasis.LOGICAL)

    assert estimate.depth.depth == 1
    assert estimate.quality is qm.EstimateQuality.EXACT


def test_measurement_provenance_sets_runtime_choice_and_feed_forward_depth() -> None:
    """A measured actual remaps to the helper formal and delays its two layers."""
    estimate = _runtime_invoke_circuit.estimate_resources()

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 4
    assert not estimate.gates.total.has(sp.Piecewise)
    assert estimate.parameters == {}


def test_measurement_provenance_crosses_uncontrolled_inverse_boundary() -> None:
    """An inverse body receives runtime provenance when no coherent control exists."""
    root = _runtime_invoke_circuit.build()
    operations = list(root.operations)
    invoke_index, invoke = next(
        (index, operation)
        for index, operation in enumerate(operations)
        if isinstance(operation, InvokeOperation)
    )
    body = invoke.effective_body()
    assert isinstance(body, Block)
    operations[invoke_index] = InverseBlockOperation(
        operands=list(invoke.operands),
        results=list(invoke.results),
        num_target_qubits=1,
        custom_name="runtime_branch_helper_inverse",
        source_block=body,
        implementation_block=body,
    )
    inverse_root = dataclasses.replace(root, operations=operations)

    estimate = qm.ResourceEstimator().estimate(inverse_root)

    assert estimate.gates.total == 3
    assert estimate.depth.depth == 4
    assert not estimate.gates.total.has(sp.Piecewise)
    assert estimate.parameters == {}


def test_measurement_branch_under_controlled_inverse_fails_closed() -> None:
    """An inverse with coherent controls rejects its measurement-selected body."""
    root = _runtime_invoke_circuit.build()
    operations = list(root.operations)
    invoke_index, invoke = next(
        (index, operation)
        for index, operation in enumerate(operations)
        if isinstance(operation, InvokeOperation)
    )
    body = invoke.effective_body()
    assert isinstance(body, Block)
    control = Value(type=QubitType(), name="control")
    inverse = InverseBlockOperation(
        operands=[control, *invoke.operands],
        results=[control.next_version(), *invoke.results],
        num_control_qubits=1,
        num_target_qubits=1,
        custom_name="controlled_runtime_branch_helper_inverse",
        source_block=body,
        implementation_block=body,
    )
    operations[invoke_index : invoke_index + 1] = [
        QInitOperation(results=[control]),
        inverse,
    ]
    inverse_root = dataclasses.replace(root, operations=operations)

    with pytest.raises(
        ValueError,
        match="Cannot estimate controlled IfOperation",
    ):
        qm.ResourceEstimator().estimate(inverse_root)


@pytest.mark.parametrize(
    "kernel",
    [_runtime_controlled_circuit, _runtime_select_circuit],
    ids=["controlled_u", "select"],
)
def test_measurement_branch_under_coherent_control_fails_closed(
    kernel: qm.QKernel,
) -> None:
    """Controlled wrappers reject measurement-selected case bodies explicitly."""
    with pytest.raises(
        ValueError,
        match="Cannot estimate controlled IfOperation",
    ):
        kernel.estimate_resources()


def test_qfixed_measurement_tracks_cast_carriers_and_measurement_depth() -> None:
    """A QFixed alias measures its source wires after their preceding gates."""

    @qm.qkernel
    def circuit() -> qm.Float:
        """Gate one carrier, cast the register, and measure the alias."""
        register = qm.qubit_array(3, "register")
        register[0] = qm.h(register[0])
        fixed = qm.cast(register, qm.QFixed, int_bits=0)
        return qm.measure(fixed)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 1
    assert estimate.depth.depth == 2
    assert estimate.depth.measurement_depth == 1
    assert estimate.width.allocated_qubits == 3
    assert estimate.width.peak_qubits == 3


def test_qfixed_view_measurement_aliases_only_covered_root_wires() -> None:
    """A cast view blocks covered carriers without blocking an idle sibling."""

    @qm.qkernel
    def covered() -> qm.Float:
        """Gate a covered root slot before measuring a strided cast view."""
        register = qm.qubit_array(4, "register")
        register[1] = qm.h(register[1])
        fixed = qm.cast(register[1::2], qm.QFixed, int_bits=0)
        return qm.measure(fixed)

    @qm.qkernel
    def sibling() -> tuple[qm.Float, qm.Qubit]:
        """Gate an uncovered sibling beside the cast-view measurement."""
        register = qm.qubit_array(4, "register")
        register[0] = qm.h(register[0])
        fixed = qm.cast(register[1::2], qm.QFixed, int_bits=0)
        measured = qm.measure(fixed)
        return measured, register[0]

    assert covered.estimate_resources().depth.depth == 2
    assert sibling.estimate_resources().depth.depth == 1


def test_empty_qfixed_measurement_has_zero_depth() -> None:
    """Measuring an empty cast register creates no measurement layer."""

    @qm.qkernel
    def circuit() -> qm.Float:
        """Cast and measure a zero-width register."""
        register = qm.qubit_array(0, "register")
        return qm.measure(qm.cast(register, qm.QFixed, int_bits=0))

    estimate = circuit.estimate_resources()

    assert estimate.qubits == 0
    assert estimate.depth.depth == 0
    assert estimate.depth.measurement_depth == 0


def test_expval_is_a_modeled_runtime_observation() -> None:
    """Expectation values expose query cost and runtime branch provenance."""

    @qm.qkernel
    def circuit(observable: qm.Observable) -> qm.Bit:
        """Choose a gate sequence from an expectation-value result."""
        observed = qm.expval(qm.qubit("observed"), observable)
        target = qm.qubit("target")
        if observed > 0.0:
            target = qm.x(target)
        else:
            target = qm.h(target)
            target = qm.z(target)
        return qm.measure(target)

    estimate = circuit.estimate_resources(inputs={"observable": qm_o.Z(0)})

    assert estimate.calls.calls_by_name == {"expval": 1}
    assert estimate.calls.queries_by_name == {"expval": 1}
    assert estimate.gates.total == 2
    assert estimate.parameters == {}
    assert estimate.quality is qm.EstimateQuality.MODELED
    assert any(assumption.source == "ExpvalOp" for assumption in estimate.assumptions)


def test_expval_runtime_taint_does_not_change_kernel_effects() -> None:
    """Estimator-local expval taint preserves the executable effect contract."""

    @qm.qkernel
    def circuit(observable: qm.Observable) -> qm.Float:
        """Return one expectation value without sample-only measurement."""
        return qm.expval(qm.qubit("observed"), observable)

    assert circuit.effects is qm.KernelEffect.NONE
    estimate = circuit.estimate_resources(inputs={"observable": qm_o.Z(0)})
    assert estimate.calls.queries_by_name == {"expval": 1}


def test_expval_destructively_releases_input_liveness() -> None:
    """A register consumed by expval does not inflate a later live allocation."""

    @qm.qkernel
    def circuit(observable: qm.Observable) -> tuple[qm.Float, qm.Bit]:
        """Observe a three-qubit register before allocating one later qubit."""
        register = qm.qubit_array(3, "register")
        expectation = qm.expval(register, observable)
        later = qm.qubit("later")
        return expectation, qm.measure(later)

    estimate = circuit.estimate_resources(inputs={"observable": qm_o.Z(0)})

    assert estimate.width.allocated_qubits == 4
    assert estimate.width.peak_qubits == 3
    assert estimate.width.circuit_qubits == 4


def test_tuple_expval_forms_a_barrier_and_releases_each_carrier() -> None:
    """Tuple carriers parallelize around expval and are all released after it."""

    @qm.qkernel
    def circuit(
        observable: qm.Observable,
    ) -> tuple[qm.Float, qm.Bit, qm.Bit]:
        """Observe two prepared scalars before measuring two fresh scalars."""
        left = qm.h(qm.qubit("left"))
        right = qm.x(qm.qubit("right"))
        expectation = qm.expval((left, right), observable)
        later_left = qm.qubit("later_left")
        later_right = qm.qubit("later_right")
        return (
            expectation,
            qm.measure(later_left),
            qm.measure(later_right),
        )

    observable = qm_o.Z(0) + qm_o.Z(1)
    estimate = circuit.estimate_resources(inputs={"observable": observable})

    assert estimate.depth.depth == 3
    assert estimate.depth.measurement_depth == 2
    assert estimate.width.allocated_qubits == 4
    assert estimate.width.peak_qubits == 2
    assert estimate.width.circuit_qubits == 4


def test_tuple_expval_consumption_crosses_control_flow_summary() -> None:
    """Both branches propagate tuple-carrier destruction to outer liveness."""

    @qm.qkernel
    def circuit(
        flag: qm.UInt,
        observable: qm.Observable,
    ) -> tuple[qm.Float, qm.Bit, qm.Bit]:
        """Observe the same two scalars in either compile-time branch."""
        left = qm.qubit("left")
        right = qm.qubit("right")
        if flag:
            expectation = qm.expval((left, right), observable)
        else:
            expectation = qm.expval((left, right), observable)
        later_left = qm.qubit("later_left")
        later_right = qm.qubit("later_right")
        return (
            expectation,
            qm.measure(later_left),
            qm.measure(later_right),
        )

    observable = qm_o.Z(0) + qm_o.Z(1)
    symbolic = circuit.estimate_resources(inputs={"observable": observable})
    concrete = circuit.estimate_resources(inputs={"flag": 1, "observable": observable})

    for estimate in (symbolic, concrete):
        assert estimate.width.allocated_qubits == 4
        assert estimate.width.peak_qubits == 2
        assert estimate.width.circuit_qubits == 4


@pytest.mark.parametrize("k", [0, 1, 3, 64])
def test_concrete_recursive_resource_driver_reaches_base_case(k: int) -> None:
    """Concrete recursion inputs expand only the terminating call path.

    Args:
        k (int): Number of recursive steps before the base case.
    """
    estimate = _resource_recursive_circuit.estimate_resources(inputs={"k": k})

    assert estimate.gates.total == 1
    assert estimate.depth.depth == 2
    assert estimate.depth.measurement_depth == 1


def test_symbolic_recursive_resource_driver_fails_with_guidance() -> None:
    """Unresolved recursion reports how to provide a terminating estimate."""
    with pytest.raises(
        ValueError,
        match="Supply a concrete recursion-driving value in inputs",
    ):
        _resource_recursive_circuit.estimate_resources()
