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
    InverseBlockOperation,
    ResourceMetadata,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import Value
from qamomile.circuit.stdlib import IQFT, QFT
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

UNARY_NATIVE_NONPARAMETRIC_GATES = ("h", "x", "y", "z", "s", "sdg", "t", "tdg")
UNARY_NATIVE_PARAMETRIC_GATES = ("p", "rx", "ry", "rz")
TWO_QUBIT_NATIVE_NONPARAMETRIC_GATES = ("cx", "cz", "swap")
TWO_QUBIT_NATIVE_PARAMETRIC_GATES = ("cp", "rzz")
THREE_QUBIT_NATIVE_GATES = ("ccx",)

UNARY_NATIVE_CASES = [
    pytest.param(gate_name, None, id=gate_name)
    for gate_name in UNARY_NATIVE_NONPARAMETRIC_GATES
] + [
    pytest.param(gate_name, angle_case, id=f"{gate_name}-{angle_case_id}")
    for gate_name in UNARY_NATIVE_PARAMETRIC_GATES
    for angle_case, angle_case_id in [
        (0.0, "zero"),
        (math.pi, "pi"),
        (2.0 * math.pi, "two-pi"),
        (("random", 0), "seed-0"),
        (("random", 42), "seed-42"),
    ]
]

MULTI_QUBIT_NATIVE_CASES = (
    [
        pytest.param(gate_name, None, id=gate_name)
        for gate_name in TWO_QUBIT_NATIVE_NONPARAMETRIC_GATES
    ]
    + [
        pytest.param(gate_name, angle_case, id=f"{gate_name}-{angle_case_id}")
        for gate_name in TWO_QUBIT_NATIVE_PARAMETRIC_GATES
        for angle_case, angle_case_id in [
            (0.0, "zero"),
            (math.pi, "pi"),
            (2.0 * math.pi, "two-pi"),
            (("random", 0), "seed-0"),
            (("random", 42), "seed-42"),
        ]
    ]
    + [
        pytest.param(gate_name, None, id=gate_name)
        for gate_name in THREE_QUBIT_NATIVE_GATES
    ]
)


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
    ctrl, target = qmc.control(_phase_layer)(
        ctrl,
        target,
        rotation_angle=rotation_angle,
    )
    return ctrl, target


@qmc.qkernel
def _inverse_controlled_native_layer(
    ctrl: qmc.Qubit,
    target: qmc.Qubit,
    rotation_angle: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply a controlled built-in native gate for inverse tests."""
    controlled_rx = qmc.control(qmc.rx)
    ctrl, target = controlled_rx(ctrl, target, rotation_angle)
    return ctrl, target


@qmc.qkernel
def _inverse_controlled_symbolic_layer(
    controls: qmc.Vector[qmc.Qubit],
    target: qmc.Qubit,
    control_count: qmc.UInt,
    rotation_angle: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Apply a symbolic-control ControlledUOperation for inverse tests."""
    controls, target = qmc.control(_phase_layer, num_controls=control_count)(
        controls,
        target,
        rotation_angle=rotation_angle,
    )
    return controls, target


@qmc.qkernel
def _inverse_controlled_index_layer(
    controls: qmc.Vector[qmc.Qubit],
    target: qmc.Qubit,
    control_count: qmc.UInt,
    rotation_angle: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Apply a control-indices SymbolicControlledUOperation."""
    controls, target = qmc.control(_phase_layer, num_controls=control_count)(
        controls,
        target,
        control_indices=[0],
        rotation_angle=rotation_angle,
    )
    return controls, target


@qmc.qkernel
def _inverse_controlled_symbolic_index_layer(
    controls: qmc.Vector[qmc.Qubit],
    target: qmc.Qubit,
    control_count: qmc.UInt,
    control_index: qmc.UInt,
    rotation_angle: qmc.Float,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Apply a control-indices SymbolicControlledUOperation."""
    controls, target = qmc.control(_phase_layer, num_controls=control_count)(
        controls,
        target,
        control_indices=[control_index],
        rotation_angle=rotation_angle,
    )
    return controls, target


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
def _inverse_vector_param_layer(
    qs: qmc.Vector[qmc.Qubit],
    rotation_angle: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a vector layer with a classical parameter."""
    qs[0] = qmc.h(qs[0])
    qs[1] = qmc.rz(qs[1], rotation_angle)
    qs[0], qs[2] = qmc.cx(qs[0], qs[2])
    return qs


@qmc.qkernel
def _inverse_custom_composite_layer(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a custom composite gate for inverse tests."""
    (q,) = _custom_composite_gate(q)
    return q


_STUB_RESOURCE_METADATA = ResourceMetadata(
    query_complexity=7,
    t_gates=3,
    total_gates=5,
)


@qmc.composite_gate(
    stub=True,
    name="stub_inverse_gate",
    num_qubits=1,
    resource_metadata=_STUB_RESOURCE_METADATA,
)
def _stub_composite_gate() -> None:
    """Define a stub composite gate for inverse tests."""


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


def _assert_all_zero_samples(sample_result: object, width: int) -> None:
    """Assert that every sampled bitstring is all zero.

    Args:
        sample_result (object): Backend sample result exposing a `results`
            iterable of `(bitstring, count)` pairs.
        width (int): Expected bitstring width.

    Returns:
        None.
    """
    expected_bits: object = (0,) * width
    expected_values = {0, expected_bits} if width == 1 else {expected_bits}
    for value, count in sample_result.results:  # type: ignore[attr-defined]
        assert count > 0
        assert value in expected_values


def _apply_unary_native_gate(
    gate_name: str,
    target: object,
    angle: float | None,
) -> object:
    """Apply a unary native gate by name.

    Args:
        gate_name (str): Native unary gate name.
        target (object): Qubit, Vector, or VectorView target.
        angle (float | None): Rotation angle for parametric gates, or None.

    Returns:
        object: Updated target handle.
    """
    gate = getattr(qmc, gate_name)
    if angle is None:
        return gate(target)
    return gate(target, angle)


def _apply_inverse_unary_native_gate(
    gate_name: str,
    target: object,
    angle: float | None,
) -> object:
    """Apply the inverse of a unary native gate by name.

    Args:
        gate_name (str): Native unary gate name.
        target (object): Qubit, Vector, or VectorView target.
        angle (float | None): Rotation angle for parametric gates, or None.

    Returns:
        object: Updated target handle.
    """
    inverse_gate = qmc.inverse(getattr(qmc, gate_name))
    if angle is None:
        return inverse_gate(target)
    return inverse_gate(target, angle)


def _apply_multi_native_gate(
    gate_name: str,
    qs: qmc.Vector[qmc.Qubit],
    angle: float | None,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a multi-qubit native gate to the leading qubits.

    Args:
        gate_name (str): Native multi-qubit gate name.
        qs (qmc.Vector[qmc.Qubit]): Register to update.
        angle (float | None): Rotation angle for parametric gates, or None.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated register.

    Raises:
        AssertionError: If `gate_name` is not a supported native gate.
    """
    match gate_name:
        case "cx":
            qs[0], qs[1] = qmc.cx(qs[0], qs[1])
        case "cz":
            qs[0], qs[1] = qmc.cz(qs[0], qs[1])
        case "swap":
            qs[0], qs[1] = qmc.swap(qs[0], qs[1])
        case "cp":
            assert angle is not None
            qs[0], qs[1] = qmc.cp(qs[0], qs[1], angle)
        case "rzz":
            assert angle is not None
            qs[0], qs[1] = qmc.rzz(qs[0], qs[1], angle)
        case "ccx":
            qs[0], qs[1], qs[2] = qmc.ccx(qs[0], qs[1], qs[2])
        case _:
            raise AssertionError(f"unsupported native gate {gate_name!r}")
    return qs


def _apply_inverse_multi_native_gate(
    gate_name: str,
    qs: qmc.Vector[qmc.Qubit],
    angle: float | None,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the inverse of a multi-qubit native gate to leading qubits.

    Args:
        gate_name (str): Native multi-qubit gate name.
        qs (qmc.Vector[qmc.Qubit]): Register to update.
        angle (float | None): Rotation angle for parametric gates, or None.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated register.

    Raises:
        AssertionError: If `gate_name` is not a supported native gate.
    """
    inverse_gate = qmc.inverse(getattr(qmc, gate_name))
    match gate_name:
        case "cx" | "cz" | "swap":
            qs[0], qs[1] = inverse_gate(qs[0], qs[1])
        case "cp" | "rzz":
            assert angle is not None
            qs[0], qs[1] = inverse_gate(qs[0], qs[1], angle)
        case "ccx":
            qs[0], qs[1], qs[2] = inverse_gate(qs[0], qs[1], qs[2])
        case _:
            raise AssertionError(f"unsupported native gate {gate_name!r}")
    return qs


def _prepare_multi_native_roundtrip_state(
    gate_name: str,
    qs: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a state that makes the native gate action observable.

    Args:
        gate_name (str): Native multi-qubit gate name.
        qs (qmc.Vector[qmc.Qubit]): Register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Prepared register.

    Raises:
        AssertionError: If `gate_name` is not supported.
    """
    match gate_name:
        case "cx":
            qs[0] = qmc.h(qs[0])
        case "cz" | "cp" | "rzz":
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.h(qs[1])
        case "swap":
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.x(qs[1])
        case "ccx":
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.h(qs[1])
        case _:
            raise AssertionError(f"unsupported native gate {gate_name!r}")
    return qs


def _unprepare_multi_native_roundtrip_state(
    gate_name: str,
    qs: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Undo the native-gate test-state preparation.

    Args:
        gate_name (str): Native multi-qubit gate name.
        qs (qmc.Vector[qmc.Qubit]): Register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Unprepared register.

    Raises:
        AssertionError: If `gate_name` is not supported.
    """
    match gate_name:
        case "cx":
            qs[0] = qmc.h(qs[0])
        case "cz" | "cp" | "rzz":
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.h(qs[1])
        case "swap":
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.x(qs[1])
        case "ccx":
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.h(qs[1])
        case _:
            raise AssertionError(f"unsupported native gate {gate_name!r}")
    return qs


def _native_unary_scalar_roundtrip_kernel(
    gate_name: str,
    angle: float | None,
) -> qmc.QKernel:
    """Build a scalar unary native-gate inverse roundtrip kernel.

    Args:
        gate_name (str): Native unary gate name.
        angle (float | None): Rotation angle for parametric gates, or None.

    Returns:
        qmc.QKernel: Kernel that samples zero iff the roundtrip is identity.
    """

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = qmc.h(q)
        q = _apply_unary_native_gate(gate_name, q, angle)
        q = _apply_inverse_unary_native_gate(gate_name, q, angle)
        q = qmc.h(q)
        return qmc.measure(q)

    return circuit


def _native_unary_vector_roundtrip_kernel(
    gate_name: str,
    angle: float | None,
) -> qmc.QKernel:
    """Build a Vector-broadcast unary native inverse roundtrip kernel.

    Args:
        gate_name (str): Native unary gate name.
        angle (float | None): Rotation angle for parametric gates, or None.

    Returns:
        qmc.QKernel: Kernel that samples all-zero after the roundtrip.
    """

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.h(qs)
        qs = _apply_unary_native_gate(gate_name, qs, angle)
        qs = _apply_inverse_unary_native_gate(gate_name, qs, angle)
        qs = qmc.h(qs)
        return qmc.measure(qs)

    return circuit


def _native_unary_vector_view_roundtrip_kernel(
    gate_name: str,
    angle: float | None,
) -> qmc.QKernel:
    """Build a VectorView-broadcast unary native inverse roundtrip kernel.

    Args:
        gate_name (str): Native unary gate name.
        angle (float | None): Rotation angle for parametric gates, or None.

    Returns:
        qmc.QKernel: Kernel that samples all-zero after the sliced roundtrip.
    """

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(5, "qs")
        view = qs[1:4]
        view = qmc.h(view)
        view = _apply_unary_native_gate(gate_name, view, angle)
        view = _apply_inverse_unary_native_gate(gate_name, view, angle)
        view = qmc.h(view)
        qs[1:4] = view
        return qmc.measure(qs)

    return circuit


def _native_multi_roundtrip_kernel(
    gate_name: str,
    angle: float | None,
) -> qmc.QKernel:
    """Build a multi-qubit native inverse roundtrip kernel.

    Args:
        gate_name (str): Native multi-qubit gate name.
        angle (float | None): Rotation angle for parametric gates, or None.

    Returns:
        qmc.QKernel: Kernel that samples all-zero after the roundtrip.
    """
    width = 3 if gate_name in THREE_QUBIT_NATIVE_GATES else 2

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(width, "qs")
        qs = _prepare_multi_native_roundtrip_state(gate_name, qs)
        qs = _apply_multi_native_gate(gate_name, qs, angle)
        qs = _apply_inverse_multi_native_gate(gate_name, qs, angle)
        qs = _unprepare_multi_native_roundtrip_state(gate_name, qs)
        return qmc.measure(qs)

    return circuit


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


def _controlled_qkernel_roundtrip_kernel() -> qmc.QKernel:
    """Build a controlled-qkernel inverse roundtrip kernel.

    Returns:
        qmc.QKernel: Kernel that samples all-zero after controlled-U and
            inverse controlled-U are applied.
    """

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs[0] = qmc.h(qs[0])
        qs[0], qs[1] = _inverse_controlled_concrete_layer(qs[0], qs[1], 0.37)
        qs[0], qs[1] = qmc.inverse(_inverse_controlled_concrete_layer)(
            qs[0],
            qs[1],
            0.37,
        )
        qs[0] = qmc.h(qs[0])
        return qmc.measure(qs)

    return circuit


def _controlled_native_roundtrip_kernel() -> qmc.QKernel:
    """Build a directly controlled native-gate inverse roundtrip kernel.

    Returns:
        qmc.QKernel: Kernel that samples all-zero after `control(qmc.rx)` and
            its inverse are applied.
    """

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs[0] = qmc.h(qs[0])
        qs[0], qs[1] = _inverse_controlled_native_layer(qs[0], qs[1], 0.37)
        qs[0], qs[1] = qmc.inverse(_inverse_controlled_native_layer)(
            qs[0],
            qs[1],
            0.37,
        )
        qs[0] = qmc.h(qs[0])
        return qmc.measure(qs)

    return circuit


def _fixed_scalar_qkernel_roundtrip_kernel() -> qmc.QKernel:
    """Build a scalar qkernel inverse roundtrip kernel with fixed angles.

    Returns:
        qmc.QKernel: Kernel that samples zero after a qkernel and its
            inverse are applied.
    """

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = qmc.h(q)
        q = _inverse_layer(q, 0.37)
        q = qmc.inverse(_inverse_layer)(q, 0.37)
        q = qmc.h(q)
        return qmc.measure(q)

    return circuit


def _custom_composite_qkernel_roundtrip_kernel() -> qmc.QKernel:
    """Build a custom-composite inverse roundtrip kernel.

    Returns:
        qmc.QKernel: Kernel that samples zero after composite and inverse
            composite are applied.
    """

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = qmc.x(q)
        q = _inverse_custom_composite_layer(q)
        q = qmc.inverse(_inverse_custom_composite_layer)(q)
        q = qmc.x(q)
        return qmc.measure(q)

    return circuit


def _vector_param_qkernel_roundtrip_kernel() -> qmc.QKernel:
    """Build a vector qkernel inverse roundtrip with a classical parameter.

    Returns:
        qmc.QKernel: Kernel that samples zero after a vector layer and its
            inverse are applied.
    """

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs[0] = qmc.x(qs[0])
        qs[2] = qmc.x(qs[2])
        qs = _inverse_vector_param_layer(qs, 0.41)
        qs = qmc.inverse(_inverse_vector_param_layer)(qs, 0.41)
        qs[0] = qmc.x(qs[0])
        qs[2] = qmc.x(qs[2])
        return qmc.measure(qs)

    return circuit


def _vector_loop_qkernel_roundtrip_kernel() -> qmc.QKernel:
    """Build a Vector-loop qkernel inverse roundtrip kernel.

    Returns:
        qmc.QKernel: Kernel that samples all-zero after a loop body and its
            inverse are applied.
    """

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = _inverse_loop_layer(qs)
        qs = qmc.inverse(_inverse_loop_layer)(qs)
        return qmc.measure(qs)

    return circuit


QKERNEL_ROUNDTRIP_CASES = [
    pytest.param(_fixed_scalar_qkernel_roundtrip_kernel, 1, id="scalar-layer"),
    pytest.param(_controlled_qkernel_roundtrip_kernel, 2, id="controlled"),
    pytest.param(_controlled_native_roundtrip_kernel, 2, id="controlled-native"),
    pytest.param(_custom_composite_qkernel_roundtrip_kernel, 1, id="custom-composite"),
    pytest.param(_vector_param_qkernel_roundtrip_kernel, 3, id="vector-param"),
    pytest.param(_vector_loop_qkernel_roundtrip_kernel, 3, id="vector-loop"),
]


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


def test_inverse_native_rotation_broadcasts_vector() -> None:
    """inverse(rz) preserves native Vector broadcast semantics."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.rz(qs, 0.25)
        qs = qmc.inverse(qmc.rz)(qs, 0.25)
        return qs

    block = circuit.build()
    loops = [op for op in block.operations if isinstance(op, ForOperation)]
    inverse_gates = [op for op in loops[1].operations if isinstance(op, GateOperation)]

    assert len(loops) == 2
    assert len(inverse_gates) == 1
    assert inverse_gates[0].gate_type is GateOperationType.RZ
    assert inverse_gates[0].theta is not None
    assert inverse_gates[0].theta.get_const() == -0.25


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


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("gate_name, angle_case", UNARY_NATIVE_CASES)
def test_inverse_native_unary_scalar_roundtrip_cross_backend(
    transpiler_factory,
    gate_name: str,
    angle_case: float | tuple[str, int] | None,
) -> None:
    """Every unary native gate followed by its inverse samples zero."""
    angle = None if angle_case is None else _angle_from_case(angle_case)
    transpiler = transpiler_factory()
    executable = transpiler.transpile(
        _native_unary_scalar_roundtrip_kernel(gate_name, angle)
    )
    sample_result = executable.sample(transpiler.executor(), shots=32).result()

    _assert_all_zero_samples(sample_result, 1)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("gate_name, angle_case", UNARY_NATIVE_CASES)
def test_inverse_native_unary_vector_broadcast_roundtrip_cross_backend(
    transpiler_factory,
    gate_name: str,
    angle_case: float | tuple[str, int] | None,
) -> None:
    """Every Vector-broadcast unary native inverse roundtrip samples zero."""
    angle = None if angle_case is None else _angle_from_case(angle_case)
    transpiler = transpiler_factory()
    executable = transpiler.transpile(
        _native_unary_vector_roundtrip_kernel(gate_name, angle)
    )
    sample_result = executable.sample(transpiler.executor(), shots=32).result()

    _assert_all_zero_samples(sample_result, 3)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("gate_name, angle_case", UNARY_NATIVE_CASES)
def test_inverse_native_unary_vector_view_broadcast_roundtrip_cross_backend(
    transpiler_factory,
    gate_name: str,
    angle_case: float | tuple[str, int] | None,
) -> None:
    """Every VectorView-broadcast unary native inverse roundtrip samples zero."""
    angle = None if angle_case is None else _angle_from_case(angle_case)
    transpiler = transpiler_factory()
    executable = transpiler.transpile(
        _native_unary_vector_view_roundtrip_kernel(gate_name, angle)
    )
    sample_result = executable.sample(transpiler.executor(), shots=32).result()

    _assert_all_zero_samples(sample_result, 5)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("gate_name, angle_case", MULTI_QUBIT_NATIVE_CASES)
def test_inverse_native_multi_qubit_roundtrip_cross_backend(
    transpiler_factory,
    gate_name: str,
    angle_case: float | tuple[str, int] | None,
) -> None:
    """Every multi-qubit native gate followed by its inverse samples zero."""
    angle = None if angle_case is None else _angle_from_case(angle_case)
    transpiler = transpiler_factory()
    executable = transpiler.transpile(_native_multi_roundtrip_kernel(gate_name, angle))
    sample_result = executable.sample(transpiler.executor(), shots=32).result()
    width = 3 if gate_name in THREE_QUBIT_NATIVE_GATES else 2

    _assert_all_zero_samples(sample_result, width)


def test_inverse_qkernel_can_be_assigned_before_calling() -> None:
    """inverse(qkernel) returns a reusable callable wrapper."""

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Qubit:
        q = qmc.qubit("q")
        inverse_layer = qmc.inverse(_inverse_layer)
        q = inverse_layer(q, rotation_angle)
        return q

    block = circuit.build(parameters=["rotation_angle"])
    inverse_ops = [
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    ]

    assert len(inverse_ops) == 1
    assert inverse_ops[0].source_block is _inverse_layer.block
    assert inverse_ops[0].implementation_block is not None
    gates = [
        op
        for op in inverse_ops[0].implementation_block.operations
        if isinstance(op, GateOperation)
    ]
    assert [gate.gate_type for gate in gates] == [
        GateOperationType.RZ,
        GateOperationType.H,
    ]


def test_inverse_qkernel_rejects_vector_for_scalar_input() -> None:
    """inverse(qkernel) rejects shape-mismatched quantum inputs."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.inverse(_inverse_layer)(qs, 0.25)
        return qs

    with pytest.raises(TypeError, match="expected scalar, got Vector"):
        circuit.build()


def test_inverse_qkernel_rejects_reordered_quantum_outputs() -> None:
    """inverse(qkernel) rejects kernels that only reorder output wires."""

    @qmc.qkernel
    def reorder(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
        """Return the same quantum inputs in a different order."""
        return b, a

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
        a = qmc.qubit("a")
        b = qmc.qubit("b")
        a, b = qmc.inverse(reorder)(a, b)
        return a, b

    with pytest.raises(TypeError, match="preserve the input order"):
        circuit.build()


def test_inverse_qkernel_keeps_inverse_fallback_block() -> None:
    """inverse(qkernel) stores a backend-native source and fallback block."""

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Qubit:
        q = qmc.qubit("q")
        q = _inverse_layer(q, rotation_angle)
        q = qmc.inverse(_inverse_layer)(q, rotation_angle)
        return q

    block = circuit.build(parameters=["rotation_angle"])
    call_ops = [op for op in block.operations if isinstance(op, CallBlockOperation)]
    inverse_ops = [
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    ]

    assert len(call_ops) == 1
    assert len(inverse_ops) == 1
    assert inverse_ops[0].source_block is _inverse_layer.block
    assert inverse_ops[0].implementation_block is not None

    angle_ops = [
        op
        for op in inverse_ops[0].implementation_block.operations
        if isinstance(op, BinOp) and op.kind is BinOpKind.MUL
    ]
    gates = [
        op
        for op in inverse_ops[0].implementation_block.operations
        if isinstance(op, GateOperation)
    ]
    assert len(angle_ops) == 1
    assert [gate.gate_type for gate in gates] == [
        GateOperationType.RZ,
        GateOperationType.H,
    ]


def test_inverse_qft_function_maps_to_iqft() -> None:
    """inverse(qft) returns iqft directly."""
    assert qmc.inverse(qmc.qft) is qmc.iqft
    assert qmc.inverse(qmc.iqft) is qmc.qft


@pytest.mark.parametrize(
    "gate",
    [
        pytest.param(QFT(2), id="qft-instance"),
        pytest.param(IQFT(2), id="iqft-instance"),
        pytest.param(_custom_composite_gate, id="custom-composite"),
        pytest.param(_stub_composite_gate, id="stub-composite"),
    ],
)
def test_inverse_rejects_direct_composite_gate_instances(
    gate: qmc.CompositeGate,
) -> None:
    """inverse() rejects direct CompositeGate instances with guidance."""
    with pytest.raises(TypeError, match="direct CompositeGate instances"):
        qmc.inverse(gate)


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
    inverse_ops = [
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    ]

    assert len(inverse_ops) == 1
    assert inverse_ops[0].implementation_block is not None
    ctrl_ops = [
        op
        for op in inverse_ops[0].implementation_block.operations
        if isinstance(op, ControlledUOperation)
    ]
    assert len(ctrl_ops) == 1
    assert isinstance(ctrl_ops[0], ConcreteControlledU)
    assert ctrl_ops[0].block is not None
    assert ctrl_ops[0].block.name == "_phase_layer_inverse"
    inverse_gates = [
        op for op in ctrl_ops[0].block.operations if isinstance(op, GateOperation)
    ]
    assert len(inverse_gates) == 1
    assert inverse_gates[0].gate_type is GateOperationType.RZ
    assert inverse_gates[0].theta.is_constant()
    assert inverse_gates[0].theta.get_const() == pytest.approx(-0.25)


def test_inverse_controlled_concrete_roundtrip_statevector(qiskit_transpiler) -> None:
    """controlled-U followed by inverse(controlled-U) executes as identity."""

    @qmc.qkernel
    def controlled_roundtrip() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs[0] = qmc.x(qs[0])
        qs[0], qs[1] = _inverse_controlled_concrete_layer(qs[0], qs[1], 0.37)
        qs[0], qs[1] = qmc.inverse(_inverse_controlled_concrete_layer)(
            qs[0],
            qs[1],
            0.37,
        )
        return qmc.measure(qs)

    qc = qiskit_transpiler.to_circuit(controlled_roundtrip)
    statevector = run_statevector(qc)
    expected = np.zeros(4, dtype=complex)
    expected[1] = 1.0

    assert np.allclose(statevector, expected, atol=1e-8)


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
    """inverse(qkernel) preserves control_indices on SymbolicControlledU."""

    @qmc.qkernel
    def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
        controls = qmc.qubit_array(2, "controls")
        target = qmc.qubit("target")
        controls, target = qmc.inverse(_inverse_controlled_index_layer)(
            controls,
            target,
            1,
            0.25,
        )
        return controls, target

    block = circuit.build()
    ctrl_ops = [op for op in block.operations if isinstance(op, ControlledUOperation)]

    assert len(ctrl_ops) == 1
    assert isinstance(ctrl_ops[0], SymbolicControlledU)
    assert ctrl_ops[0].block is not None
    assert ctrl_ops[0].block.name == "_phase_layer_inverse"
    assert ctrl_ops[0].control_indices is not None
    assert ctrl_ops[0].control_indices[0].get_const() == 0


def test_inverse_controlled_index_substitutes_symbolic_fields() -> None:
    """inverse(qkernel) substitutes symbolic control_indices fields."""

    @qmc.qkernel
    def circuit(
        control_count: qmc.UInt,
        control_index: qmc.UInt,
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
        controls = qmc.qubit_array(2, "controls")
        target = qmc.qubit("target")
        controls, target = qmc.inverse(_inverse_controlled_symbolic_index_layer)(
            controls,
            target,
            control_count,
            control_index,
            0.25,
        )
        return controls, target

    block = circuit.build(parameters=["control_count", "control_index"])
    ctrl_ops = [op for op in block.operations if isinstance(op, ControlledUOperation)]
    block_inputs = {value.name: value for value in block.input_values}

    assert len(ctrl_ops) == 1
    assert isinstance(ctrl_ops[0], SymbolicControlledU)
    assert isinstance(ctrl_ops[0].num_controls, Value)
    assert ctrl_ops[0].num_controls.uuid == block_inputs["control_count"].uuid
    assert ctrl_ops[0].control_indices is not None
    assert ctrl_ops[0].control_indices[0].uuid == block_inputs["control_index"].uuid


def test_inverse_custom_composite_gate_inverts_implementation() -> None:
    """inverse(qkernel) inverts custom composites that carry implementations."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(_inverse_custom_composite_layer)(q)
        return q

    block = circuit.build()
    inverse_ops = [
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    ]

    assert len(inverse_ops) == 1
    outer = inverse_ops[0]
    assert outer.source_block is _inverse_custom_composite_layer.block
    assert outer.implementation_block is not None
    inner_inverse_ops = [
        op
        for op in outer.implementation_block.operations
        if isinstance(op, InverseBlockOperation)
    ]
    assert len(inner_inverse_ops) == 1
    assert inner_inverse_ops[0].custom_name == "custom_h_inverse"
    assert inner_inverse_ops[0].source_block is _custom_composite_impl.block
    assert inner_inverse_ops[0].implementation_block is not None
    assert inner_inverse_ops[0].implementation_block.name.endswith("_inverse")


def test_inverse_stub_composite_gate_builds_opaque_inverse() -> None:
    """inverse(qkernel) keeps stub composite resources on an opaque inverse."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(_inverse_stub_composite_layer)(q)
        return q

    block = circuit.build()
    inverse_ops = [
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    ]

    assert len(inverse_ops) == 1
    outer = inverse_ops[0]
    assert outer.implementation_block is not None
    inner_composites = [
        op
        for op in outer.implementation_block.operations
        if isinstance(op, CompositeGateOperation)
    ]
    assert len(inner_composites) == 1
    assert inner_composites[0].custom_name == "stub_inverse_gate_inv"
    assert not inner_composites[0].has_implementation
    assert inner_composites[0].implementation_block is None
    assert inner_composites[0].resource_metadata == _STUB_RESOURCE_METADATA


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


def test_inverse_qkernel_prefers_qiskit_backend_inverse(qiskit_transpiler) -> None:
    """inverse(qkernel) uses Qiskit's reusable-gate inverse when available."""

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = qmc.inverse(_inverse_layer)(q, 0.37)
        return qmc.measure(q)

    qc = qiskit_transpiler.to_circuit(circuit)
    quantum_ops = [
        instruction.operation
        for instruction in qc.data
        if instruction.operation.name != "measure"
    ]

    assert len(quantum_ops) == 1
    assert quantum_ops[0].name.endswith("_dg")


def test_inverse_qkernel_prefers_quri_parts_backend_inverse(monkeypatch) -> None:
    """inverse(qkernel) uses QURI Parts inverse_circuit when available."""
    pytest.importorskip("quri_parts")
    pytest.importorskip("quri_parts.qulacs")

    import quri_parts.circuit as qp_c

    from qamomile.quri_parts import QuriPartsTranspiler

    original_inverse_circuit = qp_c.inverse_circuit
    inverse_calls = []

    def inverse_circuit_spy(circuit):
        """Record backend inverse calls while preserving QURI Parts behavior."""
        inverse_calls.append(circuit)
        return original_inverse_circuit(circuit)

    monkeypatch.setattr(qp_c, "inverse_circuit", inverse_circuit_spy)

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs[1] = _inverse_layer(qs[1], 0.37)
        qs[1] = qmc.inverse(_inverse_layer)(qs[1], 0.37)
        return qmc.measure(qs)

    transpiler = QuriPartsTranspiler()
    executable = transpiler.transpile(circuit)
    gates = executable.compiled_quantum[0].circuit.gates

    assert len(inverse_calls) == 1
    assert [(gate.name, gate.target_indices) for gate in gates] == [
        ("H", (1,)),
        ("RZ", (1,)),
        ("RZ", (1,)),
        ("H", (1,)),
    ]


def test_inverse_qkernel_prefers_cudaq_backend_adjoint() -> None:
    """inverse(qkernel) uses CUDA-Q cudaq.adjoint when available."""
    cudaq = pytest.importorskip("cudaq")

    from qamomile.cudaq import CudaqTranspiler

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs[1] = _inverse_layer(qs[1], rotation_angle)
        qs[1] = qmc.inverse(_inverse_layer)(qs[1], rotation_angle)
        return qmc.measure(qs)

    transpiler = CudaqTranspiler()
    executable = transpiler.transpile(circuit, parameters=["rotation_angle"])
    quantum_step = executable.compiled_quantum[0]
    cudaq_circuit = quantum_step.circuit

    assert "def _qamomile_adjoint_0(t0: cudaq.qubit, thetas: list[float]):" in (
        cudaq_circuit.source
    )
    assert "cudaq.adjoint(_qamomile_adjoint_0, q[1], thetas)" in cudaq_circuit.source

    bound = transpiler.executor().bind_parameters(
        cudaq_circuit,
        {"rotation_angle": 0.37},
        quantum_step.parameter_metadata,
    )
    statevector = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0

    assert np.allclose(statevector, expected, atol=1e-8)


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
@pytest.mark.parametrize("kernel_factory, width", QKERNEL_ROUNDTRIP_CASES)
def test_inverse_allowed_qkernel_roundtrip_cross_backend(
    transpiler_factory,
    kernel_factory,
    width: int,
) -> None:
    """Allowed qkernel inverse roundtrips sample all-zero on every backend."""
    transpiler = transpiler_factory()
    executable = transpiler.transpile(kernel_factory())
    sample_result = executable.sample(transpiler.executor(), shots=32).result()

    _assert_all_zero_samples(sample_result, width)


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
    _assert_all_zero_samples(sample_result, num_qubits)

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
