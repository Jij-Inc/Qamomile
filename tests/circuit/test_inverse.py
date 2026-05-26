"""Tests for the inverse frontend operation."""

import math

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    IndexSpecControlledU,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import Value
from tests.circuit.conftest import run_statevector

_HAS_QISKIT = True
try:  # pragma: no cover - dependency-presence guard.
    from qamomile.qiskit import QiskitTranspiler
except ImportError:  # pragma: no cover - covered when qiskit is absent.
    _HAS_QISKIT = False
    QiskitTranspiler = None  # type: ignore[assignment]

_HAS_QURI_PARTS = True
try:  # pragma: no cover - dependency-presence guard.
    import quri_parts.qulacs  # noqa: F401

    from qamomile.quri_parts import QuriPartsTranspiler
except ImportError:  # pragma: no cover - covered when quri_parts is absent.
    _HAS_QURI_PARTS = False
    QuriPartsTranspiler = None  # type: ignore[assignment]

_HAS_CUDAQ = True
try:  # pragma: no cover - dependency-presence guard.
    import cudaq  # noqa: F401

    from qamomile.cudaq import CudaqTranspiler
except ImportError:  # pragma: no cover - covered when cudaq is absent.
    _HAS_CUDAQ = False
    CudaqTranspiler = None  # type: ignore[assignment]

BACKENDS = [
    pytest.param(
        QiskitTranspiler,
        id="qiskit",
        marks=pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed"),
    ),
    pytest.param(
        QuriPartsTranspiler,
        id="quri_parts",
        marks=pytest.mark.skipif(
            not _HAS_QURI_PARTS,
            reason="quri_parts/qulacs not installed",
        ),
    ),
    pytest.param(
        CudaqTranspiler,
        id="cudaq",
        marks=pytest.mark.skipif(not _HAS_CUDAQ, reason="cudaq not installed"),
    ),
]

ANGLE_CASES = [
    pytest.param(0.0, id="zero"),
    pytest.param(math.pi, id="pi"),
    pytest.param(2.0 * math.pi, id="two-pi"),
    pytest.param(("random", 0), id="seed-0"),
    pytest.param(("random", 42), id="seed-42"),
]


@qmc.qkernel
def _inverse_layer(q: qmc.Qubit, rotation_angle: qmc.Float) -> qmc.Qubit:
    """Apply a small two-gate layer used by inverse tests."""
    q = qmc.h(q)
    q = qmc.rz(q, rotation_angle)
    return q


@qmc.qkernel
def _inverse_inner_call_layer(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a nested helper used to verify CallBlock inversion."""
    q = qmc.h(q)
    return q


@qmc.qkernel
def _inverse_call_then_gate_layer(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a nested call followed by a gate for inverse tests."""
    q = _inverse_inner_call_layer(q)
    q = qmc.x(q)
    return q


@qmc.qkernel
def _inverse_loop_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply a looped layer used to verify ForOperation inversion."""
    for i in qmc.range(3):
        qs[i] = qmc.h(qs[i])
    return qs


@qmc.qkernel
def _inverse_loop_with_tail_layer(
    qs: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply gates around a loop to verify inverse loop value flow."""
    qs[0] = qmc.x(qs[0])
    for i in qmc.range(2):
        qs[i] = qmc.h(qs[i])
    qs[0] = qmc.z(qs[0])
    return qs


@qmc.qkernel
def _inverse_branch_layer(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
    """Apply a conditional layer used to verify unsupported IfOperation."""
    if flag:
        q = qmc.x(q)
    return q


@qmc.qkernel
def _inverse_while_layer(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
    """Apply a while-loop layer used to verify unsupported WhileOperation."""
    while flag:
        q = qmc.x(q)
    return q


@qmc.qkernel
def _inverse_for_items_layer(
    q: qmc.Qubit,
    angles: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Qubit:
    """Apply a dict-item loop used to verify unsupported ForItemsOperation."""
    for _idx, angle in qmc.items(angles):
        q = qmc.rz(q, angle)
    return q


@qmc.qkernel
def _phase_layer(q: qmc.Qubit, rotation_angle: qmc.Float) -> qmc.Qubit:
    """Apply one phase rotation for controlled inverse tests."""
    q = qmc.rz(q, rotation_angle)
    return q


@qmc.qkernel
def _inverse_controlled_concrete_layer(
    ctrl: qmc.Qubit,
    target: qmc.Qubit,
    rotation_angle: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply a concrete ControlledUOperation for inverse tests."""
    ctrl, target = qmc.controlled(_phase_layer)(
        ctrl,
        target,
        rotation_angle=rotation_angle,
    )
    return ctrl, target


@qmc.qkernel
def _inverse_controlled_symbolic_layer(
    controls: qmc.Vector[qmc.Qubit],
    target: qmc.Qubit,
    control_count: qmc.UInt,
    rotation_angle: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Apply a symbolic-control ControlledUOperation for inverse tests."""
    controls, target = qmc.controlled(_phase_layer, num_controls=control_count)(
        controls,
        target,
        rotation_angle=rotation_angle,
    )
    return controls, target


@qmc.qkernel
def _inverse_controlled_index_layer(
    qs: qmc.Vector[qmc.Qubit],
    rotation_angle: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply an index-spec ControlledUOperation for inverse tests."""
    qs = qmc.controlled(_phase_layer, num_controls=1)(
        qs,
        target_indices=[1],
        rotation_angle=rotation_angle,
    )
    return qs


@qmc.qkernel
def _inverse_controlled_symbolic_index_layer(
    qs: qmc.Vector[qmc.Qubit],
    control_count: qmc.UInt,
    target_index: qmc.UInt,
    rotation_angle: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply an index-spec ControlledUOperation with symbolic fields."""
    qs = qmc.controlled(_phase_layer, num_controls=control_count)(
        qs,
        target_indices=[target_index],
        rotation_angle=rotation_angle,
    )
    return qs


@qmc.qkernel
def _inverse_pauli_evolve_layer(
    qs: qmc.Vector[qmc.Qubit],
    observable: qmc.Observable,
    evolution_time: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a Pauli evolution for inverse tests."""
    qs = qmc.pauli_evolve(qs, observable, evolution_time)
    return qs


@qmc.qkernel
def _custom_composite_impl(q: qmc.Qubit) -> qmc.Qubit:
    """Apply one custom composite implementation gate."""
    q = qmc.h(q)
    return q


_custom_composite_gate = qmc.composite_gate(name="custom_h")(_custom_composite_impl)


@qmc.qkernel
def _inverse_custom_composite_layer(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a custom composite gate for inverse tests."""
    (q,) = _custom_composite_gate(q)
    return q


@qmc.composite_gate(stub=True, name="stub_inverse_gate", num_qubits=1)
def _stub_composite_gate() -> None:
    """Define a stub composite gate for inverse rejection tests."""


@qmc.qkernel
def _inverse_stub_composite_layer(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a stub composite gate for inverse tests."""
    (q,) = _stub_composite_gate(q)
    return q


def _angle_from_case(angle_case: float | tuple[str, int]) -> float:
    """Return a deterministic angle for a boundary or seeded-random case.

    Args:
        angle_case (float | tuple[str, int]): Explicit boundary angle in
            radians, or a `("random", seed)` pair.

    Returns:
        float: Resolved angle in radians.
    """
    if isinstance(angle_case, tuple):
        _kind, seed = angle_case
        rng = np.random.default_rng(seed)
        return float(rng.uniform(-2.0 * math.pi, 2.0 * math.pi))
    return angle_case


def _sum_z_hamiltonian(num_qubits: int) -> qm_o.Hamiltonian:
    """Build the sum-Z Hamiltonian on `num_qubits` qubits.

    Args:
        num_qubits (int): Number of qubits covered by the Hamiltonian.

    Returns:
        qm_o.Hamiltonian: Hamiltonian equal to the sum of per-qubit Z terms.
    """
    hamiltonian = qm_o.Hamiltonian.zero(num_qubits=num_qubits)
    for idx in range(num_qubits):
        hamiltonian += qm_o.Z(idx)
    return hamiltonian


def _inverse_roundtrip_sample_kernel(num_qubits: int) -> qmc.QKernel:
    """Build a measured inverse-roundtrip kernel for backend tests.

    Args:
        num_qubits (int): Register size for the generated kernel.

    Returns:
        qmc.QKernel: Kernel that applies a layer and its inverse before
        measuring all qubits.
    """

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(num_qubits, "qs")
        for idx in qmc.range(num_qubits):
            qs[idx] = _inverse_layer(qs[idx], rotation_angle)
            qs[idx] = qmc.inverse(_inverse_layer)(qs[idx], rotation_angle)
        return qmc.measure(qs)

    return circuit


def _inverse_roundtrip_expval_kernel(num_qubits: int) -> qmc.QKernel:
    """Build an expval inverse-roundtrip kernel for backend tests.

    Args:
        num_qubits (int): Register size for the generated kernel.

    Returns:
        qmc.QKernel: Kernel that applies a layer and its inverse before
        computing an expectation value.
    """

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float, observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(num_qubits, "qs")
        for idx in qmc.range(num_qubits):
            qs[idx] = _inverse_layer(qs[idx], rotation_angle)
            qs[idx] = qmc.inverse(_inverse_layer)(qs[idx], rotation_angle)
        return qmc.expval(qs, observable)

    return circuit


def test_inverse_native_rotation_negates_angle() -> None:
    """inverse(rx) emits another RX gate with the negated angle."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.rx(q, 0.25)
        q = qmc.inverse(qmc.rx)(q, 0.25)
        return q

    block = circuit.build()
    gates = [op for op in block.operations if isinstance(op, GateOperation)]

    assert [gate.gate_type for gate in gates] == [
        GateOperationType.RX,
        GateOperationType.RX,
    ]
    assert gates[0].theta is not None
    assert gates[1].theta is not None
    assert gates[0].theta.get_const() == 0.25
    assert gates[1].theta.get_const() == -0.25


def test_inverse_native_dagger_gate() -> None:
    """inverse(s) emits SDG as the dagger gate."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.s(q)
        q = qmc.inverse(qmc.s)(q)
        return q

    block = circuit.build()
    gates = [op for op in block.operations if isinstance(op, GateOperation)]

    assert [gate.gate_type for gate in gates] == [
        GateOperationType.S,
        GateOperationType.SDG,
    ]


def test_inverse_qkernel_inlines_reverse_operations() -> None:
    """inverse(qkernel) emits the wrapped block's operations in reverse order."""

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Qubit:
        q = qmc.qubit("q")
        q = _inverse_layer(q, rotation_angle)
        q = qmc.inverse(_inverse_layer)(q, rotation_angle)
        return q

    block = circuit.build(parameters=["rotation_angle"])
    call_ops = [op for op in block.operations if isinstance(op, CallBlockOperation)]
    angle_ops = [
        op
        for op in block.operations
        if isinstance(op, BinOp) and op.kind is BinOpKind.MUL
    ]
    gates = [op for op in block.operations if isinstance(op, GateOperation)]

    assert len(call_ops) == 1
    assert len(angle_ops) == 1
    assert [gate.gate_type for gate in gates] == [
        GateOperationType.RZ,
        GateOperationType.H,
    ]


def test_inverse_qft_function_maps_to_iqft() -> None:
    """inverse(qft) returns iqft directly."""
    assert qmc.inverse(qmc.qft) is qmc.iqft
    assert qmc.inverse(qmc.iqft) is qmc.qft


def test_inverse_qft_emits_iqft_composite() -> None:
    """inverse(qft) preserves native composite emission by using IQFT."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.qft(qs)
        qs = qmc.inverse(qmc.qft)(qs)
        return qs

    block = circuit.build()
    composites = [
        op for op in block.operations if isinstance(op, CompositeGateOperation)
    ]

    assert [op.gate_type for op in composites] == [
        CompositeGateType.QFT,
        CompositeGateType.IQFT,
    ]


def test_inverse_for_operation_reverses_constant_range() -> None:
    """inverse(qkernel) reverses constant ForOperation bounds."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.inverse(_inverse_loop_layer)(qs)
        return qs

    block = circuit.build()
    loops = [op for op in block.operations if isinstance(op, ForOperation)]

    assert len(loops) == 1
    assert [operand.get_const() for operand in loops[0].operands] == [2, -1, -1]


def test_inverse_for_operation_with_surrounding_gates_transpiles_to_identity(
    qiskit_transpiler,
) -> None:
    """inverse(qkernel) keeps value flow correct around inverted loops."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = _inverse_loop_with_tail_layer(qs)
        qs = qmc.inverse(_inverse_loop_with_tail_layer)(qs)
        return qmc.measure(qs)

    qc = qiskit_transpiler.to_circuit(circuit)
    statevector = run_statevector(qc)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0

    assert np.allclose(statevector, expected, atol=1e-8)


def test_inverse_if_operation_raises() -> None:
    """inverse(qkernel) reports unsupported IfOperation explicitly."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        flag = qmc.bit(True)
        q = qmc.inverse(_inverse_branch_layer)(q, flag)
        return q

    with pytest.raises(NotImplementedError, match="IfOperation"):
        circuit.build()


def test_inverse_while_operation_raises() -> None:
    """inverse(qkernel) reports unsupported WhileOperation explicitly."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        flag = qmc.bit(True)
        q = qmc.inverse(_inverse_while_layer)(q, flag)
        return q

    with pytest.raises(NotImplementedError, match="WhileOperation"):
        circuit.build()


def test_inverse_for_items_operation_raises() -> None:
    """inverse(qkernel) reports unsupported ForItemsOperation explicitly."""

    @qmc.qkernel
    def circuit(angles: qmc.Dict[qmc.UInt, qmc.Float]) -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(_inverse_for_items_layer)(q, angles)
        return q

    with pytest.raises(NotImplementedError, match="ForItemsOperation"):
        circuit.build()


def test_inverse_rejects_internal_qubit_allocation() -> None:
    """inverse(qkernel) rejects blocks that allocate their own qubits."""

    @qmc.qkernel
    def allocates_qubit(q: qmc.Qubit) -> qmc.Qubit:
        ancilla = qmc.qubit("ancilla")
        ancilla = qmc.h(ancilla)
        q, ancilla = qmc.cx(q, ancilla)
        return q

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(allocates_qubit)(q)
        return q

    with pytest.raises(NotImplementedError, match="allocate qubits internally"):
        circuit.build()


def test_inverse_qkernel_keeps_outer_qubit_initialization() -> None:
    """inverse(qkernel) emits no extra QInitOperation for input qubits."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(_inverse_layer)(q, 0.5)
        return q

    block = circuit.build()
    qinit_ops = [op for op in block.operations if isinstance(op, QInitOperation)]

    assert len(qinit_ops) == 1


def test_inverse_pauli_evolve_negates_evolution_time() -> None:
    """inverse(qkernel) inverts PauliEvolveOp by negating its time parameter."""

    @qmc.qkernel
    def circuit(observable: qmc.Observable) -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(1, "qs")
        qs = qmc.inverse(_inverse_pauli_evolve_layer)(qs, observable, 0.25)
        return qs

    block = circuit.build()
    evolves = [op for op in block.operations if isinstance(op, PauliEvolveOp)]

    assert len(evolves) == 1
    assert evolves[0].gamma.get_const() == -0.25


def test_inverse_controlled_concrete_operation() -> None:
    """inverse(qkernel) inverts concrete ControlledUOperation blocks."""

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
        ctrl = qmc.qubit("ctrl")
        target = qmc.qubit("target")
        ctrl, target = qmc.inverse(_inverse_controlled_concrete_layer)(
            ctrl,
            target,
            0.25,
        )
        return ctrl, target

    block = circuit.build()
    ctrl_ops = [op for op in block.operations if isinstance(op, ControlledUOperation)]

    assert len(ctrl_ops) == 1
    assert isinstance(ctrl_ops[0], ConcreteControlledU)
    assert ctrl_ops[0].block is not None
    assert ctrl_ops[0].block.name == "_phase_layer_inverse"
    assert any(isinstance(op, BinOp) for op in ctrl_ops[0].block.operations)


def test_inverse_controlled_symbolic_operation() -> None:
    """inverse(qkernel) preserves SymbolicControlledUOperation shape."""

    @qmc.qkernel
    def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
        controls = qmc.qubit_array(2, "controls")
        target = qmc.qubit("target")
        controls, target = qmc.inverse(_inverse_controlled_symbolic_layer)(
            controls,
            target,
            2,
            0.25,
        )
        return controls, target

    block = circuit.build()
    ctrl_ops = [op for op in block.operations if isinstance(op, ControlledUOperation)]

    assert len(ctrl_ops) == 1
    assert isinstance(ctrl_ops[0], SymbolicControlledU)
    assert ctrl_ops[0].block is not None
    assert ctrl_ops[0].block.name == "_phase_layer_inverse"


def test_inverse_controlled_index_operation() -> None:
    """inverse(qkernel) preserves IndexSpecControlledUOperation shape."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(2, "qs")
        qs = qmc.inverse(_inverse_controlled_index_layer)(qs, 0.25)
        return qs

    block = circuit.build()
    ctrl_ops = [op for op in block.operations if isinstance(op, ControlledUOperation)]

    assert len(ctrl_ops) == 1
    assert isinstance(ctrl_ops[0], IndexSpecControlledU)
    assert ctrl_ops[0].block is not None
    assert ctrl_ops[0].block.name == "_phase_layer_inverse"


def test_inverse_controlled_index_substitutes_symbolic_fields() -> None:
    """inverse(qkernel) substitutes symbolic IndexSpecControlledU fields."""

    @qmc.qkernel
    def circuit(
        control_count: qmc.UInt,
        target_index: qmc.UInt,
    ) -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(2, "qs")
        qs = qmc.inverse(_inverse_controlled_symbolic_index_layer)(
            qs,
            control_count,
            target_index,
            0.25,
        )
        return qs

    block = circuit.build(parameters=["control_count", "target_index"])
    ctrl_ops = [op for op in block.operations if isinstance(op, ControlledUOperation)]
    block_inputs = {value.name: value for value in block.input_values}

    assert len(ctrl_ops) == 1
    assert isinstance(ctrl_ops[0], IndexSpecControlledU)
    assert isinstance(ctrl_ops[0].num_controls, Value)
    assert ctrl_ops[0].num_controls.uuid == block_inputs["control_count"].uuid
    assert ctrl_ops[0].target_indices is not None
    assert ctrl_ops[0].target_indices[0].uuid == block_inputs["target_index"].uuid


def test_inverse_custom_composite_gate_inverts_implementation() -> None:
    """inverse(qkernel) inverts custom composites that carry implementations."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(_inverse_custom_composite_layer)(q)
        return q

    block = circuit.build()
    composites = [
        op for op in block.operations if isinstance(op, CompositeGateOperation)
    ]

    assert len(composites) == 1
    assert composites[0].gate_type is CompositeGateType.CUSTOM
    assert composites[0].custom_name == "custom_h_inverse"
    assert composites[0].has_implementation
    assert composites[0].implementation_block is not None
    assert composites[0].implementation_block.name.endswith("_inverse")


def test_inverse_stub_composite_gate_raises() -> None:
    """inverse(qkernel) rejects stub composites without implementations."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(_inverse_stub_composite_layer)(q)
        return q

    with pytest.raises(NotImplementedError, match="stub composite gate"):
        circuit.build()


def test_inverse_qkernel_transpiles_to_identity(qiskit_transpiler) -> None:
    """inverse(qkernel) survives the full transpiler path as an identity."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = _inverse_layer(q, 0.37)
        q = qmc.inverse(_inverse_layer)(q, 0.37)
        return qmc.measure(q)

    qc = qiskit_transpiler.to_circuit(circuit)
    statevector = run_statevector(qc)

    assert np.allclose(statevector, np.array([1.0, 0.0]), atol=1e-8)


def test_inverse_nested_qkernel_call_transpiles_to_identity(qiskit_transpiler) -> None:
    """inverse(qkernel) preserves the current value across nested calls."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = _inverse_call_then_gate_layer(q)
        q = qmc.inverse(_inverse_call_then_gate_layer)(q)
        return qmc.measure(q)

    qc = qiskit_transpiler.to_circuit(circuit)
    statevector = run_statevector(qc)

    assert np.allclose(statevector, np.array([1.0, 0.0]), atol=1e-8)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5])
@pytest.mark.parametrize("angle_case", ANGLE_CASES)
def test_inverse_roundtrip_cross_backend_sampling_and_expval(
    transpiler_factory,
    num_qubits: int,
    angle_case: float | tuple[str, int],
) -> None:
    """inverse(qkernel) executes as identity across sampling and expval paths."""
    rotation_angle = _angle_from_case(angle_case)
    transpiler = transpiler_factory()

    sample_kernel = _inverse_roundtrip_sample_kernel(num_qubits)
    sample_executable = transpiler.transpile(
        sample_kernel,
        parameters=["rotation_angle"],
    )
    sample_result = sample_executable.sample(
        transpiler.executor(),
        shots=64,
        bindings={"rotation_angle": rotation_angle},
    ).result()
    expected_bits = (0,) * num_qubits
    for value, _count in sample_result.results:
        assert value == expected_bits

    expval_kernel = _inverse_roundtrip_expval_kernel(num_qubits)
    observable = _sum_z_hamiltonian(num_qubits)
    expval_executable = transpiler.transpile(
        expval_kernel,
        parameters=["rotation_angle"],
        bindings={"observable": observable},
    )
    expval_result = expval_executable.run(
        transpiler.executor(),
        bindings={"rotation_angle": rotation_angle},
    ).result()

    assert np.isclose(expval_result, float(num_qubits), atol=1e-6)
