"""Tests for the inverse frontend operation."""

import math

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.arithmetic import (
    modular_decrement,
    modular_increment,
)
from qamomile.circuit.algorithm.basic import (
    cx_entangling_layer,
    cz_entangling_layer,
    phase_gadget,
    rx_layer,
    ry_layer,
    rz_layer,
    superposition_vector,
)
from qamomile.circuit.algorithm.fqaoa import (
    givens_rotation,
    givens_rotations,
    hopping_gate,
    initial_occupations,
    mixer_layer,
)
from qamomile.circuit.algorithm.qaoa import x_mixer
from qamomile.circuit.algorithm.state_preparation import (
    amplitude_encoding,
    amplitude_encoding_from_angles,
    computational_basis_state,
)
from qamomile.circuit.estimator import estimate_resources
from qamomile.circuit.frontend.operation.inverse import (
    _BlockInverter,
    _InverseRotationCallable,
    _static_quantum_width,
)
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.types.primitives import QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, Value
from qamomile.circuit.stdlib import IQFT, QFT
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.style import CircuitStyle
from tests.circuit.conftest import run_statevector

# ---------------------------------------------------------------------------
# Backend availability and parametrization tables
# ---------------------------------------------------------------------------


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
    # The lazy accessor raises ImportError when cudaq is missing, without
    # loading the cudaq runtime at collection time when it is installed
    # (see tests/_cudaq_isolation.py).
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
        # The cudaq mark keeps this leg out of default sessions, where
        # loading cudaq is unsafe (see tests/_cudaq_isolation.py).
        marks=[
            pytest.mark.skipif(not _HAS_CUDAQ, reason="cudaq not installed"),
            pytest.mark.cudaq,
        ],
    ),
]


# (angle_case, id) pairs covering boundary angles plus seeded-random draws.
# `("random", seed)` markers are resolved to concrete radians by
# `_angle_from_case` inside each test body.
ANGLE_CASE_PAIRS = [
    (0.0, "zero"),
    (math.pi, "pi"),
    (2.0 * math.pi, "two-pi"),
    (("random", 0), "seed-0"),
    (("random", 42), "seed-42"),
]

ANGLE_CASES = [
    pytest.param(angle_case, id=angle_case_id)
    for angle_case, angle_case_id in ANGLE_CASE_PAIRS
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
    for angle_case, angle_case_id in ANGLE_CASE_PAIRS
]

MULTI_QUBIT_NATIVE_CASES = (
    [
        pytest.param(gate_name, None, id=gate_name)
        for gate_name in TWO_QUBIT_NATIVE_NONPARAMETRIC_GATES
    ]
    + [
        pytest.param(gate_name, angle_case, id=f"{gate_name}-{angle_case_id}")
        for gate_name in TWO_QUBIT_NATIVE_PARAMETRIC_GATES
        for angle_case, angle_case_id in ANGLE_CASE_PAIRS
    ]
    + [
        pytest.param(gate_name, None, id=gate_name)
        for gate_name in THREE_QUBIT_NATIVE_GATES
    ]
)


# IR-level expectations mirroring the frontend gate maps in
# `qamomile.circuit.frontend.operation.inverse`: `_ROTATION_GATES` (angle
# negation, split here by arity into the single- and two-qubit tables),
# `_DAGGER_GATES` (S/T-family counterparts), and the direct callable map in
# `_inverse_native_gate_target` (self-inverse gates plus the dagger pairs).
# Together the tables cover every native gate.
SINGLE_QUBIT_ROTATION_GATE_CASES = [
    pytest.param("p", GateOperationType.P, id="p"),
    pytest.param("rx", GateOperationType.RX, id="rx"),
    pytest.param("ry", GateOperationType.RY, id="ry"),
    pytest.param("rz", GateOperationType.RZ, id="rz"),
]

TWO_QUBIT_ROTATION_GATE_CASES = [
    pytest.param("cp", GateOperationType.CP, id="cp"),
    pytest.param("rzz", GateOperationType.RZZ, id="rzz"),
]

DAGGER_GATE_CASES = [
    pytest.param("s", GateOperationType.S, GateOperationType.SDG, id="s"),
    pytest.param("sdg", GateOperationType.SDG, GateOperationType.S, id="sdg"),
    pytest.param("t", GateOperationType.T, GateOperationType.TDG, id="t"),
    pytest.param("tdg", GateOperationType.TDG, GateOperationType.T, id="tdg"),
]

NATIVE_CALLABLE_INVERSE_CASES = [
    pytest.param(gate_name, gate_name, id=gate_name)
    for gate_name in ("h", "x", "y", "z", "cx", "cz", "swap", "ccx")
] + [
    pytest.param(forward_name, inverse_name, id=forward_name)
    for forward_name, inverse_name in (
        ("s", "sdg"),
        ("sdg", "s"),
        ("t", "tdg"),
        ("tdg", "t"),
    )
]


# ---------------------------------------------------------------------------
# Shared layer kernels and helpers
# ---------------------------------------------------------------------------


@qmc.qkernel
def _inverse_layer(q: qmc.Qubit, rotation_angle: qmc.Float) -> qmc.Qubit:
    """Apply a small two-gate layer used by inverse tests."""
    q = qmc.h(q)
    q = qmc.rz(q, rotation_angle)
    return q


@qmc.qkernel
def _inverse_vector_layer(
    qs: qmc.Vector[qmc.Qubit],
    rotation_angle: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a vector layer used to verify atomic inverse emission."""
    for idx in qmc.range(qs.shape[0]):
        qs[idx] = qmc.rx(qs[idx], rotation_angle)
    return qs


@qmc.qkernel
def _inverse_loop_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply a looped layer used to verify ForOperation inversion."""
    for i in qmc.range(3):
        qs[i] = qmc.h(qs[i])
    return qs


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
def _custom_composite_impl(q: qmc.Qubit) -> qmc.Qubit:
    """Apply one custom composite implementation gate."""
    q = qmc.h(q)
    return q


_custom_composite_gate = qmc.composite_gate(name="custom_h")(_custom_composite_impl)


@qmc.qkernel
def _inverse_custom_composite_layer(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a custom composite gate for inverse tests."""
    q = _custom_composite_gate(q)
    return q


def _single_inverse_implementation(block: Block) -> Block:
    """Return the only inverse implementation block in a test block."""
    inverse_ops = [
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    ]

    assert len(inverse_ops) == 1
    assert inverse_ops[0].implementation_block is not None
    return inverse_ops[0].implementation_block


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


def _assert_all_zero_samples(
    sample_result: object,
    width: int,
    expected_shots: int,
) -> None:
    """Assert that every sampled bitstring is all zero.

    Args:
        sample_result (object): Backend sample result exposing a `results`
            iterable of `(bitstring, count)` pairs.
        width (int): Expected bitstring width.
        expected_shots (int): Expected total number of sampled shots.

    Returns:
        None.
    """
    expected_bits: object = (0,) * width
    expected_values = {0, expected_bits} if width == 1 else {expected_bits}
    results = list(sample_result.results)  # type: ignore[attr-defined]
    assert results
    assert sum(count for _, count in results) == expected_shots
    for value, count in results:
        assert count > 0
        assert value in expected_values


def _sum_z_observable(width: int) -> qm_o.Hamiltonian:
    """Build the sum of single-qubit Z observables over a register.

    Args:
        width (int): Register width in qubits.

    Returns:
        qm_o.Hamiltonian: Observable summing `Z(idx)` over every qubit
            index, so the all-zero state has expectation `width`.
    """
    observable = qm_o.Hamiltonian.zero(num_qubits=width)
    for idx in range(width):
        observable += qm_o.Z(idx)
    return observable


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


# ---------------------------------------------------------------------------
# Native-gate inverse mapping (IR level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forward_name, inverse_name", NATIVE_CALLABLE_INVERSE_CASES)
def test_inverse_native_callable_mapping(forward_name: str, inverse_name: str) -> None:
    """inverse(<self-inverse or dagger gate>) returns the counterpart callable.

    The frontend maps non-rotation native gates directly to an existing
    frontend callable (itself for self-inverse gates, the dagger function for
    the S/T family), so object identity pins the mapping exactly.
    """
    assert qmc.inverse(getattr(qmc, forward_name)) is getattr(qmc, inverse_name)


@pytest.mark.parametrize(
    "gate_name, expected_gate_type",
    SINGLE_QUBIT_ROTATION_GATE_CASES,
)
def test_inverse_native_rotation_negates_angle(
    gate_name: str,
    expected_gate_type: GateOperationType,
) -> None:
    """inverse(<rotation gate>) emits the same gate type with the negated angle."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = _apply_unary_native_gate(gate_name, q, 0.5)
        q = _apply_inverse_unary_native_gate(gate_name, q, 0.5)
        return q

    block = circuit.build()
    gates = [op for op in block.operations if isinstance(op, GateOperation)]

    assert [gate.gate_type for gate in gates] == [
        expected_gate_type,
        expected_gate_type,
    ]
    assert gates[0].theta is not None
    assert gates[1].theta is not None
    assert gates[0].theta.get_const() == 0.5
    assert gates[1].theta.get_const() == -0.5


@pytest.mark.parametrize(
    "gate_name, expected_gate_type",
    TWO_QUBIT_ROTATION_GATE_CASES,
)
def test_inverse_native_two_qubit_rotation_negates_angle(
    gate_name: str,
    expected_gate_type: GateOperationType,
) -> None:
    """inverse(<two-qubit rotation>) emits the same gate type, negated angle."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(2, "qs")
        qs = _apply_multi_native_gate(gate_name, qs, 0.5)
        qs = _apply_inverse_multi_native_gate(gate_name, qs, 0.5)
        return qs

    block = circuit.build()
    gates = [op for op in block.operations if isinstance(op, GateOperation)]

    assert [gate.gate_type for gate in gates] == [
        expected_gate_type,
        expected_gate_type,
    ]
    assert gates[0].theta is not None
    assert gates[1].theta is not None
    assert gates[0].theta.get_const() == 0.5
    assert gates[1].theta.get_const() == -0.5


def test_inverse_native_rotation_broadcasts_vector() -> None:
    """inverse(rz) preserves native Vector broadcast semantics."""
    # A single representative rotation suffices: broadcast dispatch is
    # gate-independent, and the execution matrix (UNARY_NATIVE_CASES) covers
    # every native gate through the same broadcast path.

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


@pytest.mark.parametrize("gate_name, forward_type, inverse_type", DAGGER_GATE_CASES)
def test_inverse_native_dagger_gate(
    gate_name: str,
    forward_type: GateOperationType,
    inverse_type: GateOperationType,
) -> None:
    """inverse(<S/T-family gate>) emits the dagger counterpart gate."""

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = _apply_unary_native_gate(gate_name, q, None)
        q = _apply_inverse_unary_native_gate(gate_name, q, None)
        return q

    block = circuit.build()
    gates = [op for op in block.operations if isinstance(op, GateOperation)]

    assert [gate.gate_type for gate in gates] == [forward_type, inverse_type]


def test_inverse_rotation_normalizes_signed_zero() -> None:
    """inverse(rotation) stores positive zero for zero-angle inverses."""

    @qmc.qkernel
    def direct_native() -> qmc.Qubit:
        a = qmc.qubit("a")
        a = qmc.inverse(qmc.rz)(a, 0.0)
        return a

    block = direct_native.build()
    gates = [op for op in block.operations if isinstance(op, GateOperation)]
    assert gates[0].theta is not None
    theta = gates[0].theta.get_const()
    assert theta == 0.0
    assert math.copysign(1.0, float(theta)) == 1.0


def test_inverse_rotation_callable_applies_defaults() -> None:
    """Inverse rotation callables negate defaulted angles."""

    def rotation(q: object, angle: float = 0.0) -> tuple[object, float]:
        """Return the supplied qubit-like object and angle."""
        return q, angle

    inverse_rotation = _InverseRotationCallable(rotation, "angle")

    assert inverse_rotation("q") == ("q", 0.0)


# ---------------------------------------------------------------------------
# Native-gate execution roundtrip matrices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("gate_name, angle_case", UNARY_NATIVE_CASES)
def test_inverse_native_unary_scalar_roundtrip_cross_backend(
    transpiler_factory,
    gate_name: str,
    angle_case: float | tuple[str, int] | None,
) -> None:
    """Every unary native gate followed by its inverse samples zero."""
    angle = None if angle_case is None else _angle_from_case(angle_case)

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = qmc.h(q)
        q = _apply_unary_native_gate(gate_name, q, angle)
        q = _apply_inverse_unary_native_gate(gate_name, q, angle)
        q = qmc.h(q)
        return qmc.measure(q)

    transpiler = transpiler_factory()
    executable = transpiler.transpile(circuit)
    sample_result = executable.sample(transpiler.executor(), shots=32).result()

    _assert_all_zero_samples(sample_result, 1, 32)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("gate_name, angle_case", UNARY_NATIVE_CASES)
def test_inverse_native_unary_vector_broadcast_roundtrip_cross_backend(
    transpiler_factory,
    gate_name: str,
    angle_case: float | tuple[str, int] | None,
) -> None:
    """Every Vector-broadcast unary native inverse roundtrip samples zero."""
    angle = None if angle_case is None else _angle_from_case(angle_case)

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.h(qs)
        qs = _apply_unary_native_gate(gate_name, qs, angle)
        qs = _apply_inverse_unary_native_gate(gate_name, qs, angle)
        qs = qmc.h(qs)
        return qmc.measure(qs)

    transpiler = transpiler_factory()
    executable = transpiler.transpile(circuit)
    sample_result = executable.sample(transpiler.executor(), shots=32).result()

    _assert_all_zero_samples(sample_result, 3, 32)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("gate_name, angle_case", UNARY_NATIVE_CASES)
def test_inverse_native_unary_vector_view_broadcast_roundtrip_cross_backend(
    transpiler_factory,
    gate_name: str,
    angle_case: float | tuple[str, int] | None,
) -> None:
    """Every VectorView-broadcast unary native inverse roundtrip samples zero."""
    angle = None if angle_case is None else _angle_from_case(angle_case)

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

    transpiler = transpiler_factory()
    executable = transpiler.transpile(circuit)
    sample_result = executable.sample(transpiler.executor(), shots=32).result()

    _assert_all_zero_samples(sample_result, 5, 32)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("gate_name, angle_case", MULTI_QUBIT_NATIVE_CASES)
def test_inverse_native_multi_qubit_roundtrip_cross_backend(
    transpiler_factory,
    gate_name: str,
    angle_case: float | tuple[str, int] | None,
) -> None:
    """Every multi-qubit native gate followed by its inverse samples zero."""
    angle = None if angle_case is None else _angle_from_case(angle_case)
    width = 3 if gate_name in THREE_QUBIT_NATIVE_GATES else 2

    def prepare_state(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Prepare a state that makes the native gate action observable."""
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

    def unprepare_state(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Undo the native-gate test-state preparation."""
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

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(width, "qs")
        qs = prepare_state(qs)
        qs = _apply_multi_native_gate(gate_name, qs, angle)
        qs = _apply_inverse_multi_native_gate(gate_name, qs, angle)
        qs = unprepare_state(qs)
        return qmc.measure(qs)

    transpiler = transpiler_factory()
    executable = transpiler.transpile(circuit)
    sample_result = executable.sample(transpiler.executor(), shots=32).result()

    _assert_all_zero_samples(sample_result, width, 32)


# ---------------------------------------------------------------------------
# QKernel inverse core behavior
# ---------------------------------------------------------------------------


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


def test_inverse_qkernel_keeps_inverse_fallback_block() -> None:
    """inverse(qkernel) stores a backend-native source and fallback block."""

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Qubit:
        q = qmc.qubit("q")
        q = _inverse_layer(q, rotation_angle)
        q = qmc.inverse(_inverse_layer)(q, rotation_angle)
        return q

    block = circuit.build(parameters=["rotation_angle"])
    call_ops = [
        op
        for op in block.operations
        if isinstance(op, InvokeOperation)
        and op.body is not None
        and op.body.name == "_inverse_layer"
    ]
    inverse_ops = [
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    ]

    assert len(call_ops) == 1
    assert len(inverse_ops) == 1
    assert call_ops[0].body is _inverse_layer.block
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


def test_inverse_qkernel_atomic_inverse_accepts_dict_parameter() -> None:
    """inverse(qkernel) preserves DictValue operands for emit-time binding."""

    @qmc.qkernel
    def dict_parameter_layer(
        q: qmc.Qubit,
        angles: qmc.Dict[qmc.UInt, qmc.Float],
    ) -> qmc.Qubit:
        """Apply H while keeping an intentionally unused dict parameter.

        `angles` is never read; it stays in the signature only so the
        traced call carries a DictValue operand for the inverse op.
        """
        q = qmc.h(q)
        return q

    @qmc.qkernel
    def circuit(angles: qmc.Dict[qmc.UInt, qmc.Float]) -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(dict_parameter_layer)(q, angles)
        return q

    block = circuit.build()
    inverse_op = next(
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    )

    assert isinstance(inverse_op.operands[1], DictValue)


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


def test_inverse_qkernel_rejects_vector_for_scalar_input() -> None:
    """inverse(qkernel) rejects shape-mismatched quantum inputs."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.inverse(_inverse_layer)(qs, 0.25)
        return qs

    with pytest.raises(TypeError, match="scalar Qubit but received"):
        circuit.build()


def test_inverse_qkernel_allows_reordered_quantum_outputs(qiskit_transpiler) -> None:
    """inverse(qkernel) treats pure output reordering as a wire permutation."""

    @qmc.qkernel
    def reorder(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
        """Return the same quantum inputs in a different order."""
        return b, a

    @qmc.qkernel
    def circuit() -> tuple[qmc.Bit, qmc.Bit]:
        a = qmc.qubit("a")
        b = qmc.qubit("b")
        a = qmc.x(a)
        a, b = qmc.inverse(reorder)(a, b)
        return qmc.measure(a), qmc.measure(b)

    executable = qiskit_transpiler.transpile(circuit)
    sample_result = executable.sample(qiskit_transpiler.executor(), shots=32).result()

    assert sample_result.results == [((False, True), 32)]


def test_inverse_qkernel_builds_symbolic_vector_broadcast_inverse() -> None:
    """inverse(qkernel) specializes symmetric Vector broadcast loop bounds."""

    @qmc.qkernel
    def layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply a broadcast gate to a symbolic-width vector."""
        qs = qmc.h(qs)
        return qs

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.inverse(layer)(qs)
        return qs

    implementation = _single_inverse_implementation(circuit.build())
    loops = [op for op in implementation.operations if isinstance(op, ForOperation)]

    assert len(loops) == 1
    assert [operand.get_const() for operand in loops[0].operands] == [2, -1, -1]


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


@qmc.qkernel
def _inverse_pauli_evolve_layer(
    qs: qmc.Vector[qmc.Qubit],
    observable: qmc.Observable,
    evolution_time: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a Pauli evolution for inverse tests."""
    qs = qmc.pauli_evolve(qs, observable, evolution_time)
    return qs


def test_inverse_pauli_evolve_negates_evolution_time() -> None:
    """inverse(qkernel) inverts PauliEvolveOp by negating its time parameter."""

    @qmc.qkernel
    def circuit(observable: qmc.Observable) -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(1, "qs")
        qs = qmc.inverse(_inverse_pauli_evolve_layer)(qs, observable, 0.25)
        return qs

    block = _single_inverse_implementation(circuit.build())
    evolves = [op for op in block.operations if isinstance(op, PauliEvolveOp)]

    assert len(evolves) == 1
    assert evolves[0].gamma.get_const() == -0.25


def test_inverse_pauli_evolve_matches_manual_negative_time(qiskit_transpiler) -> None:
    """inverse(pauli_evolve) executes like pauli_evolve with negative time."""
    observable = qm_o.Hamiltonian()
    observable.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 0.5)

    @qmc.qkernel
    def inverse_circuit(
        observable: qmc.Observable,
        gamma: qmc.Float,
    ) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(1, "qs")
        qs = qmc.inverse(_inverse_pauli_evolve_layer)(qs, observable, gamma)
        return qmc.measure(qs)

    @qmc.qkernel
    def manual_circuit(
        observable: qmc.Observable,
        gamma: qmc.Float,
    ) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(1, "qs")
        qs = qmc.pauli_evolve(qs, observable, -gamma)
        return qmc.measure(qs)

    inverse_qc = qiskit_transpiler.to_circuit(
        inverse_circuit,
        bindings={"observable": observable, "gamma": 0.25},
    )
    manual_qc = qiskit_transpiler.to_circuit(
        manual_circuit,
        bindings={"observable": observable, "gamma": 0.25},
    )

    assert np.allclose(run_statevector(inverse_qc), run_statevector(manual_qc))


# ---------------------------------------------------------------------------
# Inverse of inverse
# ---------------------------------------------------------------------------


def test_inverse_of_inverse_restores_source_operations() -> None:
    """inverse() cancels a nested inverse block back to its source body."""

    @qmc.qkernel
    def inverse_layer(rotation_angle: qmc.Float, q: qmc.Qubit) -> qmc.Qubit:
        q = qmc.inverse(_inverse_layer)(q, rotation_angle)
        return q

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(inverse_layer)(rotation_angle, q)
        return q

    block = circuit.build(parameters=["rotation_angle"])
    implementation = _single_inverse_implementation(block)

    assert not any(
        isinstance(op, InverseBlockOperation) for op in implementation.operations
    )
    gates = [op for op in implementation.operations if isinstance(op, GateOperation)]
    assert [gate.gate_type for gate in gates] == [
        GateOperationType.H,
        GateOperationType.RZ,
    ]


def test_inverse_of_controlled_inverse_restores_controlled_source() -> None:
    """Double inversion preserves controls on first-class inverse blocks."""
    source_input = Value(type=QubitType(), name="target")
    source_output = source_input.next_version()
    source_block = Block(
        name="source",
        input_values=[source_input],
        output_values=[source_output],
        operations=[
            GateOperation.fixed(
                GateOperationType.X,
                [source_input],
                [source_output],
            )
        ],
        kind=BlockKind.HIERARCHICAL,
    )
    control = Value(type=QubitType(), name="control")
    target = Value(type=QubitType(), name="target")
    inverse_control = control.next_version()
    inverse_target = target.next_version()
    inverse_op = InverseBlockOperation(
        operands=[control, target],
        results=[inverse_control, inverse_target],
        num_control_qubits=1,
        num_target_qubits=1,
        custom_name="source_inverse",
        source_block=source_block,
        implementation_block=source_block,
    )
    block = Block(
        name="controlled_inverse",
        input_values=[control, target],
        output_values=[inverse_control, inverse_target],
        operations=[inverse_op],
        kind=BlockKind.HIERARCHICAL,
    )

    inverted = _BlockInverter().invert_block(block)

    assert len(inverted.operations) == 1
    restored = inverted.operations[0]
    assert isinstance(restored, ConcreteControlledU)
    assert restored.num_controls == 1
    assert restored.block is source_block
    assert restored.control_operands == [control]
    assert restored.target_operands[:1] == [target]
    assert inverted.output_values == restored.results


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


def test_inverse_for_operation_reverses_constant_range() -> None:
    """inverse(qkernel) reverses constant ForOperation bounds."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(3, "qs")
        qs = qmc.inverse(_inverse_loop_layer)(qs)
        return qs

    block = _single_inverse_implementation(circuit.build())
    loops = [op for op in block.operations if isinstance(op, ForOperation)]

    assert len(loops) == 1
    assert [operand.get_const() for operand in loops[0].operands] == [2, -1, -1]


def test_inverse_for_operation_with_surrounding_gates_transpiles_to_identity(
    qiskit_transpiler,
) -> None:
    """inverse(qkernel) keeps value flow correct around inverted loops."""

    @qmc.qkernel
    def loop_with_tail_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply gates around a loop to verify inverse loop value flow."""
        qs[0] = qmc.x(qs[0])
        for i in qmc.range(2):
            qs[i] = qmc.h(qs[i])
        qs[0] = qmc.z(qs[0])
        return qs

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = loop_with_tail_layer(qs)
        qs = qmc.inverse(loop_with_tail_layer)(qs)
        return qmc.measure(qs)

    qc = qiskit_transpiler.to_circuit(circuit)
    statevector = run_statevector(qc)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0

    assert np.allclose(statevector, expected, atol=1e-8)


def test_inverse_for_operation_with_bindings_resolved_bound_transpiles_to_identity(
    qiskit_transpiler,
) -> None:
    """inverse(qkernel) supports loop bounds fed by bindings-resolved UInt args.

    Bindings are baked into the IR at trace time, so a bound loop bound is
    already a compile-time constant when the inverse walker resolves the
    range, not the documented symbolic-bounds limitation.
    """

    @qmc.qkernel
    def layer(qs: qmc.Vector[qmc.Qubit], n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Apply H to the first `n` qubits of the register."""
        for i in qmc.range(n):
            qs[i] = qmc.h(qs[i])
        return qs

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = layer(qs, n)
        qs = qmc.inverse(layer)(qs, n)
        return qmc.measure(qs)

    qc = qiskit_transpiler.to_circuit(circuit, bindings={"n": 2})
    statevector = run_statevector(qc)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0

    assert np.allclose(statevector, expected, atol=1e-8)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
def test_inverse_nested_for_operation_roundtrip_cross_backend(
    transpiler_factory,
) -> None:
    """Nested loop inverse preserves the carried qubit wire across backends."""

    @qmc.qkernel
    def nested_loop_layer(q: qmc.Qubit, rotation_angle: qmc.Float) -> qmc.Qubit:
        """Apply nested loops over one carried qubit wire."""
        for _i in qmc.range(2):
            q = qmc.rx(q, rotation_angle)
            for _j in qmc.range(2):
                q = qmc.rz(q, rotation_angle)
        return q

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Bit:
        q = qmc.qubit("q")
        q = nested_loop_layer(q, rotation_angle)
        q = qmc.inverse(nested_loop_layer)(q, rotation_angle)
        return qmc.measure(q)

    transpiler = transpiler_factory()
    executable = transpiler.transpile(circuit, parameters=["rotation_angle"])
    sample_result = executable.sample(
        transpiler.executor(),
        shots=32,
        bindings={"rotation_angle": 0.37},
    ).result()

    _assert_all_zero_samples(sample_result, 1, 32)


def test_inverse_negative_step_loop_roundtrip_statevector(qiskit_transpiler) -> None:
    """inverse() reverses a negative-step compile-time loop to identity."""

    @qmc.qkernel
    def negative_step_loop_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Loop layer with a negative compile-time step."""
        for i in qmc.range(2, -1, -1):
            qs[i] = qmc.h(qs[i])
            qs[i] = qmc.rx(qs[i], 0.37)
        return qs

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = negative_step_loop_layer(qs)
        qs = qmc.inverse(negative_step_loop_layer)(qs)
        return qmc.measure(qs)

    statevector = run_statevector(qiskit_transpiler.to_circuit(circuit))
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1.0

    assert np.allclose(statevector, expected, atol=1e-8)


def test_inverse_loop_with_huge_compile_time_bound_builds_in_constant_space() -> None:
    """Reversing a loop must not materialize its compile-time iteration space.

    The reversed bounds are derived arithmetically from the forward
    ``range``; materializing ``10**9`` iterations at trace time would
    hang or exhaust memory even though the IR keeps the loop symbolic.
    """

    @qmc.qkernel
    def huge_range_layer(q: qmc.Qubit) -> qmc.Qubit:
        """Loop layer whose compile-time bound is far too large to unroll."""
        for _i in qmc.range(10**9):
            q = qmc.h(q)
        return q

    @qmc.qkernel
    def circuit(q: qmc.Qubit) -> qmc.Qubit:
        q = huge_range_layer(q)
        q = qmc.inverse(huge_range_layer)(q)
        return q

    block = circuit.build()
    inverse_op = next(
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    )
    assert inverse_op.implementation_block is not None
    loop = next(
        op
        for op in inverse_op.implementation_block.operations
        if isinstance(op, ForOperation)
    )
    bounds = [operand.get_const() for operand in loop.operands]
    assert bounds == [10**9 - 1, -1, -1]


# ---------------------------------------------------------------------------
# Unsupported control flow and non-unitary rejection
# ---------------------------------------------------------------------------


def test_inverse_if_operation_raises() -> None:
    """inverse(qkernel) reports unsupported IfOperation explicitly."""

    @qmc.qkernel
    def branch_layer(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
        """Apply a conditional layer used to verify unsupported IfOperation."""
        if flag:
            q = qmc.x(q)
        return q

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        flag = qmc.bit(True)
        q = qmc.inverse(branch_layer)(q, flag)
        return q

    with pytest.raises(NotImplementedError, match="IfOperation"):
        circuit.build()


def test_inverse_if_operation_raises_even_with_compile_time_bindings(
    qiskit_transpiler,
) -> None:
    """A classical-value `if` raises even when bindings could resolve it.

    The inverse fallback block is built eagerly at trace time, before
    `CompileTimeIfLoweringPass` runs in partial_eval, so bindings that would
    fold the branch cannot rescue inverse() from the IfOperation.
    """

    @qmc.qkernel
    def conditional_layer(q: qmc.Qubit, flag: qmc.UInt) -> qmc.Qubit:
        """Apply H behind a classical-value branch."""
        if flag > qmc.uint(0):
            q = qmc.h(q)
        return q

    @qmc.qkernel
    def circuit(flag: qmc.UInt) -> qmc.Bit:
        q = qmc.qubit("q")
        q = conditional_layer(q, flag)
        q = qmc.inverse(conditional_layer)(q, flag)
        return qmc.measure(q)

    with pytest.raises(NotImplementedError, match="IfOperation"):
        qiskit_transpiler.transpile(circuit, bindings={"flag": 1})


def test_inverse_while_operation_raises() -> None:
    """inverse(qkernel) reports unsupported WhileOperation explicitly."""

    @qmc.qkernel
    def while_layer(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
        """Apply a while-loop layer used to verify unsupported WhileOperation."""
        while flag:
            q = qmc.x(q)
        return q

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        flag = qmc.bit(True)
        q = qmc.inverse(while_layer)(q, flag)
        return q

    with pytest.raises(NotImplementedError, match="WhileOperation"):
        circuit.build()


def test_inverse_for_items_operation_raises() -> None:
    """inverse(qkernel) reports unsupported ForItemsOperation explicitly."""

    @qmc.qkernel
    def for_items_layer(
        q: qmc.Qubit,
        angles: qmc.Dict[qmc.UInt, qmc.Float],
    ) -> qmc.Qubit:
        """Apply a dict-item loop used to verify unsupported ForItemsOperation."""
        for _idx, angle in qmc.items(angles):
            q = qmc.rz(q, angle)
        return q

    @qmc.qkernel
    def circuit(angles: qmc.Dict[qmc.UInt, qmc.Float]) -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(for_items_layer)(q, angles)
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


def test_inverse_rejects_superposition_vector() -> None:
    """inverse() rejects algorithm kernels that return internally allocated qubits.

    `superposition_vector` allocates its register internally and returns it,
    so it has one quantum output but zero quantum inputs. The input/output
    preservation check fires before the QInit walk, so the rejection
    surfaces as TypeError rather than the internal-allocation
    NotImplementedError.
    """

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.inverse(superposition_vector)(2)
        return qmc.measure(qs)

    with pytest.raises(TypeError, match="preserve every quantum input"):
        circuit.build()


# ---------------------------------------------------------------------------
# Controlled operations
# ---------------------------------------------------------------------------


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
    def controlled_symbolic_layer(
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
    def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
        controls = qmc.qubit_array(2, "controls")
        target = qmc.qubit("target")
        controls, target = qmc.inverse(controlled_symbolic_layer)(
            controls,
            target,
            2,
            0.25,
        )
        return controls, target

    block = _single_inverse_implementation(circuit.build())
    ctrl_ops = [op for op in block.operations if isinstance(op, ControlledUOperation)]

    assert len(ctrl_ops) == 1
    assert isinstance(ctrl_ops[0], SymbolicControlledU)
    assert ctrl_ops[0].block is not None
    assert ctrl_ops[0].block.name == "_phase_layer_inverse"


def test_inverse_controlled_index_operation() -> None:
    """inverse(qkernel) preserves control_indices on SymbolicControlledU."""

    @qmc.qkernel
    def controlled_index_layer(
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
    def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
        controls = qmc.qubit_array(2, "controls")
        target = qmc.qubit("target")
        controls, target = qmc.inverse(controlled_index_layer)(
            controls,
            target,
            1,
            0.25,
        )
        return controls, target

    block = _single_inverse_implementation(circuit.build())
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
    def controlled_symbolic_index_layer(
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
    def circuit(
        control_count: qmc.UInt,
        control_index: qmc.UInt,
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
        controls = qmc.qubit_array(2, "controls")
        target = qmc.qubit("target")
        controls, target = qmc.inverse(controlled_symbolic_index_layer)(
            controls,
            target,
            control_count,
            control_index,
            0.25,
        )
        return controls, target

    top_block = circuit.build(parameters=["control_count", "control_index"])
    inverse_ops = [
        op for op in top_block.operations if isinstance(op, InverseBlockOperation)
    ]
    assert len(inverse_ops) == 1

    block = _single_inverse_implementation(top_block)
    ctrl_ops = [op for op in block.operations if isinstance(op, ControlledUOperation)]
    block_inputs = {value.name: value for value in top_block.input_values}
    impl_inputs = {value.name: value for value in block.input_values}

    assert len(ctrl_ops) == 1
    assert isinstance(ctrl_ops[0], SymbolicControlledU)
    assert inverse_ops[0].parameters[0].uuid == block_inputs["control_count"].uuid
    assert inverse_ops[0].parameters[1].uuid == block_inputs["control_index"].uuid
    assert isinstance(ctrl_ops[0].num_controls, Value)
    assert ctrl_ops[0].num_controls.uuid == impl_inputs["control_count"].uuid
    assert ctrl_ops[0].control_indices is not None
    assert ctrl_ops[0].control_indices[0].uuid == impl_inputs["control_index"].uuid


# ---------------------------------------------------------------------------
# Composite gates and opaque oracles (QFT / IQFT / custom / QPE)
# ---------------------------------------------------------------------------


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
    invokes = [op for op in block.operations if isinstance(op, InvokeOperation)]

    assert [op.attrs["gate_type"] for op in invokes] == ["QFT", "IQFT"]
    assert [op.target.name for op in invokes] == ["qft", "iqft"]


def test_inverse_qft_rejects_strategy_without_iqft_counterpart(monkeypatch) -> None:
    """inverse(qkernel) rejects QFT strategies missing on IQFT."""
    # Register a strategy name that exists only on QFT: inverting to IQFT
    # must fail because IQFT defines no strategy of the same name.
    monkeypatch.setitem(QFT._strategies, "qft_only", QFT._strategies["standard"])

    @qmc.qkernel
    def qft_only_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        q0, q1 = QFT(2)(qs[0], qs[1], strategy="qft_only")
        qs[0] = q0
        qs[1] = q1
        return qs

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        qs = qmc.qubit_array(2, "qs")
        qs = qmc.inverse(qft_only_layer)(qs)
        return qs

    with pytest.raises(NotImplementedError, match="IQFT does not define"):
        circuit.build()


def test_inverse_qft_builds_iqft_invocation() -> None:
    """QFT inversion should build an IQFT invocation without resource metadata."""
    q = Value(type=QubitType(), name="q")
    q_after = q.next_version()
    ref = CallableRef(namespace="qamomile.stdlib", name="qft")
    attrs = {
        "kind": "composite",
        "gate_type": CompositeGateType.QFT.name,
        "num_control_qubits": 0,
        "num_target_qubits": 1,
        "custom_name": "qft",
    }
    op = InvokeOperation(
        operands=[q],
        results=[q_after],
        target=ref,
        attrs=attrs,
        definition=CallableDef(ref=ref, attrs=attrs),
    )
    block = Block(
        name="qft_layer",
        input_values=[q],
        output_values=[q_after],
        operations=[op],
        kind=BlockKind.HIERARCHICAL,
    )

    inverted = _BlockInverter().invert_block(block)
    inverse_op = inverted.operations[0]

    assert isinstance(inverse_op, InvokeOperation)
    assert inverse_op.target.name == "iqft"
    assert inverse_op.transform.name == "DIRECT"
    assert inverse_op.definition is not None
    assert inverse_op.definition.resource_models == []


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


_OPAQUE_RESOURCE_MODEL = qmc.FixedResourceModel(
    gates=qmc.GateResources(t=3, total=5),
    calls=qmc.CallResources(queries_by_name={"opaque_inverse_gate": 7}),
)


_opaque_oracle = qmc.Oracle(
    name="opaque_inverse_gate",
    num_qubits=1,
    resource_model=_OPAQUE_RESOURCE_MODEL,
)


def test_inverse_oracle_builds_opaque_inverse() -> None:
    """inverse(qkernel) keeps oracle resource models on an opaque inverse."""

    @qmc.qkernel
    def oracle_layer(q: qmc.Qubit) -> qmc.Qubit:
        """Apply an oracle for inverse tests."""
        (q,) = _opaque_oracle(q)
        return q

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(oracle_layer)(q)
        return q

    block = circuit.build()
    inverse_ops = [
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    ]

    assert len(inverse_ops) == 1
    outer = inverse_ops[0]
    assert outer.implementation_block is not None
    inner_invokes = [
        op
        for op in outer.implementation_block.operations
        if isinstance(op, InvokeOperation)
    ]
    assert len(inner_invokes) == 1
    assert inner_invokes[0].attrs["custom_name"] == "opaque_inverse_gate_inv"
    assert inner_invokes[0].body is None
    assert inner_invokes[0].definition is not None
    assert (
        inner_invokes[0].definition.resource_models[0].model is _OPAQUE_RESOURCE_MODEL
    )


def test_inverse_accepts_qkernel_backed_composite_decorator() -> None:
    """inverse(composite-decorated qkernel) follows the qkernel-like path."""

    @qmc.composite_gate(name="boxed_h")
    @qmc.qkernel
    def boxed_h(q: qmc.Qubit) -> qmc.Qubit:
        q = qmc.h(q)
        return q

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(boxed_h)(q)
        return q

    block = circuit.build()
    inverse_ops = [
        op for op in block.operations if isinstance(op, InverseBlockOperation)
    ]

    assert len(inverse_ops) == 1
    assert inverse_ops[0].source_block is boxed_h.block
    assert inverse_ops[0].implementation_block is not None
    assert inverse_ops[0].callable_ref is not None
    assert inverse_ops[0].callable_ref.namespace == "user.composite"
    assert inverse_ops[0].callable_ref.name == "boxed_h"
    assert inverse_ops[0].callable_attrs["kind"] == "composite"
    assert inverse_ops[0].callable_attrs["custom_name"] == "boxed_h"
    assert inverse_ops[0].callable_attrs["default_policy"] == "PRESERVE_BOX"


@pytest.mark.parametrize(
    "gate",
    [
        pytest.param(QFT(2), id="qft-instance"),
        pytest.param(IQFT(2), id="iqft-instance"),
    ],
)
def test_inverse_rejects_direct_composite_gate_instances(
    gate: qmc.CompositeGate,
) -> None:
    """inverse() rejects direct CompositeGate instances with guidance."""
    with pytest.raises(TypeError, match="CompositeGate objects"):
        qmc.inverse(gate)


def test_inverse_rejects_native_qpe_composite_marker() -> None:
    """inverse(qkernel) rejects native QPE composites without an inverse."""

    def emit_native_qpe_marker(q: qmc.Qubit) -> qmc.Qubit:
        """Emit a synthetic native QPE composite marker for inverse tests."""
        result = q.value.next_version()
        ref = CallableRef(namespace="qamomile.stdlib", name="qpe")
        attrs = {
            "kind": "composite",
            "gate_type": CompositeGateType.QPE.name,
            "num_control_qubits": 0,
            "num_target_qubits": 1,
            "custom_name": "qpe",
        }
        op = InvokeOperation(
            operands=[q.value],
            results=[result],
            target=ref,
            attrs=attrs,
            definition=CallableDef(ref=ref, attrs=attrs),
        )
        get_current_tracer().add_operation(op)
        return qmc.Qubit(result)

    @qmc.qkernel
    def qpe_marker_layer(q: qmc.Qubit) -> qmc.Qubit:
        q = emit_native_qpe_marker(q)
        return q

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(qpe_marker_layer)(q)
        return q

    with pytest.raises(NotImplementedError, match="opaque native InvokeOperation"):
        circuit.build()


# ---------------------------------------------------------------------------
# Estimator and visualization integration
# ---------------------------------------------------------------------------


def test_inverse_resource_estimator_binds_mixed_inputs_by_formal_order() -> None:
    """Resource estimation binds classical inverse inputs despite quantum-first IR."""
    repetition_count = Value(type=UIntType(), name="repetition_count")
    formal_q = Value(type=QubitType(), name="formal_q")
    target_q = Value(type=QubitType(), name="target_q")
    target_q_next = target_q.next_version()
    controlled_block = Block(
        name="controlled",
        input_values=[target_q],
        output_values=[target_q_next],
        operations=[
            GateOperation.fixed(GateOperationType.H, [target_q], [target_q_next])
        ],
        kind=BlockKind.AFFINE,
    )
    controlled_result = formal_q.next_version()
    impl = Block(
        name="impl",
        input_values=[repetition_count, formal_q],
        output_values=[controlled_result],
        operations=[
            SymbolicControlledU(
                operands=[formal_q],
                results=[controlled_result],
                block=controlled_block,
                num_controls=repetition_count,
                num_control_args=0,
            )
        ],
        kind=BlockKind.AFFINE,
    )
    actual_q = Value(type=QubitType(), name="actual_q")
    actual_n = Value(type=UIntType(), name="repetition_count")
    inverse_op = InverseBlockOperation(
        operands=[actual_q, actual_n],
        results=[actual_q.next_version()],
        num_target_qubits=1,
        implementation_block=impl,
    )
    block = Block(operations=[inverse_op], kind=BlockKind.AFFINE)

    counts = estimate_resources(block).gates
    count_expr = str(counts.two_qubit)

    assert "repetition_count" in count_expr
    assert "actual_q" not in count_expr


def test_inverse_block_visual_label_uses_source_name() -> None:
    """InverseBlockOperation visualization labels the source block inverse."""

    q = Value(type=QubitType(), name="q")
    source = Block(
        name="sub",
        input_values=[q],
        output_values=[q],
        operations=[],
        kind=BlockKind.HIERARCHICAL,
    )
    op = InverseBlockOperation(
        operands=[q],
        results=[q.next_version()],
        num_target_qubits=1,
        custom_name="sub_inverse",
        source_block=source,
        implementation_block=source,
    )
    analyzer = CircuitAnalyzer(Block(), CircuitStyle())

    gate = analyzer._build_vgate(op, ("inverse",), {}, {}, {})

    assert gate.label == "SUB^-1"


# ---------------------------------------------------------------------------
# Backend-native inverse preference
# ---------------------------------------------------------------------------


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


def test_inverse_vector_qkernel_prefers_qiskit_backend_inverse(
    qiskit_transpiler,
) -> None:
    """inverse(qkernel) keeps Vector inputs atomic for Qiskit inversion."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = qmc.inverse(_inverse_vector_layer)(qs, 0.37)
        return qmc.measure(qs)

    qc = qiskit_transpiler.to_circuit(circuit)
    quantum_ops = [
        instruction.operation
        for instruction in qc.data
        if instruction.operation.name != "measure"
    ]

    assert len(quantum_ops) == 1
    assert quantum_ops[0].name.endswith("_dg")


def test_qiskit_emitter_falls_back_on_qiskit_error() -> None:
    """Qiskit reusable-gate probes return None on QiskitError."""
    pytest.importorskip("qiskit")

    from qiskit.exceptions import QiskitError

    from qamomile.qiskit.emitter import QiskitGateEmitter

    class BadCircuit:
        """Raise a QiskitError from circuit_to_gate."""

        def to_gate(self, label: str) -> None:
            """Reject reusable conversion like a Qiskit circuit."""
            del label
            raise QiskitError("cannot convert")

    class BadGate:
        """Raise a QiskitError from gate_inverse."""

        def inverse(self) -> None:
            """Reject inversion like a Qiskit gate."""
            raise QiskitError("cannot invert")

    emitter = QiskitGateEmitter()

    assert emitter.circuit_to_gate(BadCircuit()) is None
    assert emitter.gate_inverse(BadGate()) is None


def test_inverse_qkernel_prefers_quri_parts_backend_inverse(monkeypatch) -> None:
    """inverse(qkernel) uses QURI Parts inverse_circuit when available."""
    pytest.importorskip("quri_parts")
    pytest.importorskip("quri_parts.qulacs")

    import quri_parts.circuit as qp_c

    from qamomile.quri_parts import QuriPartsTranspiler

    # The emitter looks up `qp_c.inverse_circuit` at call time, so patching
    # the quri_parts.circuit module attribute observes the backend-native
    # inverse path without changing its behavior.
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
    assert [gate.params for gate in gates] == [
        (),
        (0.37,),
        (-0.37,),
        (),
    ]


def test_inverse_vector_qkernel_prefers_quri_parts_backend_inverse(monkeypatch) -> None:
    """inverse(qkernel) keeps Vector inputs atomic for QURI Parts inversion."""
    pytest.importorskip("quri_parts")
    pytest.importorskip("quri_parts.qulacs")

    import quri_parts.circuit as qp_c

    from qamomile.quri_parts import QuriPartsTranspiler

    # The emitter looks up `qp_c.inverse_circuit` at call time, so patching
    # the quri_parts.circuit module attribute observes the backend-native
    # inverse path without changing its behavior.
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
        qs = qmc.inverse(_inverse_vector_layer)(qs, 0.37)
        return qmc.measure(qs)

    transpiler = QuriPartsTranspiler()
    executable = transpiler.transpile(circuit)
    gates = executable.compiled_quantum[0].circuit.gates

    assert len(inverse_calls) == 1
    assert [(gate.name, gate.target_indices, gate.params) for gate in gates] == [
        ("RX", (1,), (-0.37,)),
        ("RX", (0,), (-0.37,)),
    ]


@pytest.mark.cudaq
def test_inverse_qkernel_prefers_cudaq_backend_adjoint() -> None:
    """inverse(qkernel) uses CUDA-Q cudaq.adjoint when available.

    Runs in ``-m cudaq`` sessions only: loading cudaq into a default
    session is unsafe — see tests/_cudaq_isolation.py.
    """
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


@pytest.mark.cudaq
def test_inverse_vector_qkernel_prefers_cudaq_backend_adjoint() -> None:
    """inverse(qkernel) keeps Vector inputs atomic for CUDA-Q adjoint.

    Runs in ``-m cudaq`` sessions only: loading cudaq into a default
    session is unsafe — see tests/_cudaq_isolation.py.
    """
    cudaq = pytest.importorskip("cudaq")

    from qamomile.cudaq import CudaqTranspiler

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = _inverse_vector_layer(qs, rotation_angle)
        qs = qmc.inverse(_inverse_vector_layer)(qs, rotation_angle)
        return qmc.measure(qs)

    transpiler = CudaqTranspiler()
    executable = transpiler.transpile(circuit, parameters=["rotation_angle"])
    quantum_step = executable.compiled_quantum[0]
    cudaq_circuit = quantum_step.circuit

    assert (
        "def _qamomile_adjoint_0(t0: cudaq.qubit, t1: cudaq.qubit, "
        "thetas: list[float]):"
    ) in cudaq_circuit.source
    assert (
        "cudaq.adjoint(_qamomile_adjoint_0, q[0], q[1], thetas)" in cudaq_circuit.source
    )

    bound = transpiler.executor().bind_parameters(
        cudaq_circuit,
        {"rotation_angle": 0.37},
        quantum_step.parameter_metadata,
    )
    statevector = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0

    assert np.allclose(statevector, expected, atol=1e-8)


@pytest.mark.cudaq
def test_inverse_of_inverse_cudaq_falls_back_without_nested_adjoint() -> None:
    """CUDA-Q inverse-of-inverse falls back before nested adjoint emission.

    Runs in ``-m cudaq`` sessions only: loading cudaq into a default
    session is unsafe — see tests/_cudaq_isolation.py.
    """
    cudaq = pytest.importorskip("cudaq")

    from qamomile.cudaq import CudaqTranspiler

    @qmc.qkernel
    def inverse_layer(rotation_angle: qmc.Float, q: qmc.Qubit) -> qmc.Qubit:
        q = qmc.inverse(_inverse_layer)(q, rotation_angle)
        return q

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Bit:
        q = qmc.qubit("q")
        q = inverse_layer(rotation_angle, q)
        q = qmc.inverse(inverse_layer)(rotation_angle, q)
        return qmc.measure(q)

    transpiler = CudaqTranspiler()
    executable = transpiler.transpile(circuit, parameters=["rotation_angle"])
    quantum_step = executable.compiled_quantum[0]
    cudaq_circuit = quantum_step.circuit

    helper_source, _entry_source = cudaq_circuit.source.split(
        "@cudaq.kernel\ndef _qamomile_kernel",
        maxsplit=1,
    )
    assert "cudaq.adjoint(" not in helper_source

    bound = transpiler.executor().bind_parameters(
        cudaq_circuit,
        {"rotation_angle": 0.37},
        quantum_step.parameter_metadata,
    )
    statevector = np.array(cudaq.get_state(bound.kernel_func, bound.param_values))
    expected = np.zeros(2, dtype=complex)
    expected[0] = 1.0

    assert np.allclose(statevector, expected, atol=1e-8)


# The two layers below are shared by the CUDA-Q adjoint-source test in this
# section and by the nested-call wire-availability test in the nested
# qkernel call section.
@qmc.qkernel
def _inverse_runtime_inner_call_layer(
    q: qmc.Qubit,
    rotation_angle: qmc.Float,
) -> qmc.Qubit:
    """Apply a runtime rotation inside a nested inverse helper."""
    q = qmc.rx(q, rotation_angle)
    return q


@qmc.qkernel
def _inverse_runtime_call_then_gate_layer(
    q: qmc.Qubit,
    rotation_angle: qmc.Float,
) -> qmc.Qubit:
    """Apply a gate and a nested runtime-parameter helper."""
    q = qmc.h(q)
    q = _inverse_runtime_inner_call_layer(q, rotation_angle)
    return q


@pytest.mark.cudaq
def test_inverse_cudaq_adjoint_inlines_nested_source_block() -> None:
    """CUDA-Q adjoint helpers include nested qkernel call bodies.

    Runs in ``-m cudaq`` sessions only: loading cudaq into a default
    session is unsafe — see tests/_cudaq_isolation.py.
    """
    pytest.importorskip("cudaq")

    from qamomile.cudaq import CudaqTranspiler

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Bit:
        q = qmc.qubit("q")
        q = qmc.inverse(_inverse_runtime_call_then_gate_layer)(q, rotation_angle)
        return qmc.measure(q)

    executable = CudaqTranspiler().transpile(circuit, parameters=["rotation_angle"])
    source = executable.compiled_quantum[0].circuit.source

    assert "def _qamomile_adjoint_0(t0: cudaq.qubit, thetas: list[float]):" in source
    assert "h(t0)" in source
    assert "rx(thetas[0], t0)" in source
    assert "cudaq.adjoint(_qamomile_adjoint_0, q[0], thetas)" in source


# ---------------------------------------------------------------------------
# Nested qkernel calls
# ---------------------------------------------------------------------------


def test_inverse_nested_qkernel_call_transpiles_to_identity(qiskit_transpiler) -> None:
    """inverse(qkernel) preserves the current value across nested calls."""

    @qmc.qkernel
    def inner_call_layer(q: qmc.Qubit) -> qmc.Qubit:
        """Apply a nested helper used to verify CallBlock inversion."""
        q = qmc.h(q)
        return q

    @qmc.qkernel
    def call_then_gate_layer(q: qmc.Qubit) -> qmc.Qubit:
        """Apply a nested call followed by a gate for inverse tests."""
        q = inner_call_layer(q)
        q = qmc.x(q)
        return q

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = call_then_gate_layer(q)
        q = qmc.inverse(call_then_gate_layer)(q)
        return qmc.measure(q)

    qc = qiskit_transpiler.to_circuit(circuit)
    statevector = run_statevector(qc)

    assert np.allclose(statevector, np.array([1.0, 0.0]), atol=1e-8)


def test_inverse_nested_call_implementation_uses_available_wires() -> None:
    """Nested-call inverse fallback block should not reference source-only wires."""

    @qmc.qkernel
    def circuit(rotation_angle: qmc.Float) -> qmc.Qubit:
        q = qmc.qubit("q")
        q = qmc.inverse(_inverse_runtime_call_then_gate_layer)(q, rotation_angle)
        return q

    implementation = _single_inverse_implementation(
        circuit.build(parameters=["rotation_angle"])
    )
    available = {value.uuid for value in implementation.input_values}
    for op in implementation.operations:
        if isinstance(op, GateOperation):
            for operand in op.qubit_operands:
                assert operand.uuid in available
        available.update(result.uuid for result in op.results)


# ---------------------------------------------------------------------------
# Cross-backend qkernel roundtrip matrices
# ---------------------------------------------------------------------------


def _fixed_scalar_qkernel_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build scalar qkernel inverse roundtrip kernels with fixed angles.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures zero
            and expval kernel whose sum-Z expectation equals the width.
    """

    def roundtrip(q: qmc.Qubit) -> qmc.Qubit:
        """Apply the scalar layer and its inverse inside an H sandwich."""
        q = qmc.h(q)
        q = _inverse_layer(q, 0.37)
        q = qmc.inverse(_inverse_layer)(q, 0.37)
        q = qmc.h(q)
        return q

    @qmc.qkernel
    def sample_circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = roundtrip(q)
        return qmc.measure(q)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(1, "qs")
        qs[0] = roundtrip(qs[0])
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _controlled_qkernel_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build controlled-qkernel inverse roundtrip kernels.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply controlled-U and inverse controlled-U inside an H sandwich."""
        qs[0] = qmc.h(qs[0])
        qs[0], qs[1] = _inverse_controlled_concrete_layer(qs[0], qs[1], 0.37)
        qs[0], qs[1] = qmc.inverse(_inverse_controlled_concrete_layer)(
            qs[0],
            qs[1],
            0.37,
        )
        qs[0] = qmc.h(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _controlled_native_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build directly controlled native-gate inverse roundtrip kernels.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after `control(qmc.rx)` and its inverse are applied.
    """

    @qmc.qkernel
    def controlled_native_layer(
        ctrl: qmc.Qubit,
        target: qmc.Qubit,
        rotation_angle: qmc.Float,
    ) -> tuple[qmc.Qubit, qmc.Qubit]:
        """Apply a controlled built-in native gate for inverse tests."""
        controlled_rx = qmc.control(qmc.rx)
        ctrl, target = controlled_rx(ctrl, target, rotation_angle)
        return ctrl, target

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply the controlled layer and its inverse inside an H sandwich."""
        qs[0] = qmc.h(qs[0])
        qs[0], qs[1] = controlled_native_layer(qs[0], qs[1], 0.37)
        qs[0], qs[1] = qmc.inverse(controlled_native_layer)(
            qs[0],
            qs[1],
            0.37,
        )
        qs[0] = qmc.h(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _custom_composite_qkernel_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build custom-composite inverse roundtrip kernels.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures zero
            and expval kernel whose sum-Z expectation equals the width.
    """

    def roundtrip(q: qmc.Qubit) -> qmc.Qubit:
        """Apply the composite layer and its inverse inside an X sandwich."""
        q = qmc.x(q)
        q = _inverse_custom_composite_layer(q)
        q = qmc.inverse(_inverse_custom_composite_layer)(q)
        q = qmc.x(q)
        return q

    @qmc.qkernel
    def sample_circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = roundtrip(q)
        return qmc.measure(q)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(1, "qs")
        qs[0] = roundtrip(qs[0])
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _generic_vector_qkernel_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build size-generic Vector qkernel inverse roundtrip kernels.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after a generic vector layer and its inverse are applied.
    """

    @qmc.qkernel
    def layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply a broadcast gate to a symbolic-width vector."""
        qs = qmc.h(qs)
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = layer(qs)
        qs = qmc.inverse(layer)(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = layer(qs)
        qs = qmc.inverse(layer)(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _vector_param_qkernel_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build vector qkernel inverse roundtrip kernels with a classical parameter.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after a vector layer and its inverse are applied.
    """

    @qmc.qkernel
    def vector_param_layer(
        qs: qmc.Vector[qmc.Qubit],
        rotation_angle: qmc.Float,
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply a vector layer with a classical parameter."""
        qs[0] = qmc.h(qs[0])
        qs[1] = qmc.rz(qs[1], rotation_angle)
        qs[0], qs[2] = qmc.cx(qs[0], qs[2])
        return qs

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply the parametric layer and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs[2] = qmc.x(qs[2])
        qs = vector_param_layer(qs, 0.41)
        qs = qmc.inverse(vector_param_layer)(qs, 0.41)
        qs[0] = qmc.x(qs[0])
        qs[2] = qmc.x(qs[2])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _vector_loop_qkernel_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build Vector-loop qkernel inverse roundtrip kernels.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after a loop body and its inverse are applied.
    """

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = _inverse_loop_layer(qs)
        qs = qmc.inverse(_inverse_loop_layer)(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = _inverse_loop_layer(qs)
        qs = qmc.inverse(_inverse_loop_layer)(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


QKERNEL_ROUNDTRIP_CASES = [
    pytest.param(_fixed_scalar_qkernel_roundtrip_kernels, 1, id="scalar-layer"),
    pytest.param(_controlled_qkernel_roundtrip_kernels, 2, id="controlled"),
    pytest.param(_controlled_native_roundtrip_kernels, 2, id="controlled-native"),
    pytest.param(_custom_composite_qkernel_roundtrip_kernels, 1, id="custom-composite"),
    pytest.param(_generic_vector_qkernel_roundtrip_kernels, 3, id="generic-vector"),
    pytest.param(_vector_param_qkernel_roundtrip_kernels, 3, id="vector-param"),
    pytest.param(_vector_loop_qkernel_roundtrip_kernels, 3, id="vector-loop"),
]


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("kernel_factory, width", QKERNEL_ROUNDTRIP_CASES)
def test_inverse_allowed_qkernel_roundtrip_cross_backend(
    transpiler_factory,
    kernel_factory,
    width: int,
) -> None:
    """Allowed qkernel inverse roundtrips execute on sampling and expval paths.

    Sampling and expectation values go through different backend primitives
    (sampler vs estimator) and regress independently, so both legs run for
    every case: samples must be all-zero and the sum-Z expectation must
    equal the register width.
    """
    sample_kernel, expval_kernel = kernel_factory()
    transpiler = transpiler_factory()

    executable = transpiler.transpile(sample_kernel)
    sample_result = executable.sample(transpiler.executor(), shots=32).result()
    _assert_all_zero_samples(sample_result, width, 32)

    expval_executable = transpiler.transpile(
        expval_kernel,
        bindings={"observable": _sum_z_observable(width)},
    )
    expval_result = expval_executable.run(transpiler.executor()).result()

    assert np.isclose(expval_result, float(width), atol=1e-6)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5])
@pytest.mark.parametrize("angle_case", ANGLE_CASES)
def test_inverse_roundtrip_cross_backend_sampling_and_expval(
    transpiler_factory,
    num_qubits: int,
    angle_case: float | tuple[str, int],
) -> None:
    """inverse(qkernel) executes as identity across sampling and expval paths."""

    @qmc.qkernel
    def sample_circuit(rotation_angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(num_qubits, "qs")
        for idx in qmc.range(num_qubits):
            qs[idx] = _inverse_layer(qs[idx], rotation_angle)
            qs[idx] = qmc.inverse(_inverse_layer)(qs[idx], rotation_angle)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(
        rotation_angle: qmc.Float,
        observable: qmc.Observable,
    ) -> qmc.Float:
        qs = qmc.qubit_array(num_qubits, "qs")
        for idx in qmc.range(num_qubits):
            qs[idx] = _inverse_layer(qs[idx], rotation_angle)
            qs[idx] = qmc.inverse(_inverse_layer)(qs[idx], rotation_angle)
        return qmc.expval(qs, observable)

    rotation_angle = _angle_from_case(angle_case)
    transpiler = transpiler_factory()

    sample_executable = transpiler.transpile(
        sample_circuit,
        parameters=["rotation_angle"],
    )
    sample_result = sample_executable.sample(
        transpiler.executor(),
        shots=64,
        bindings={"rotation_angle": rotation_angle},
    ).result()
    _assert_all_zero_samples(sample_result, num_qubits, 64)

    observable = qm_o.Hamiltonian.zero(num_qubits=num_qubits)
    for idx in range(num_qubits):
        observable += qm_o.Z(idx)
    expval_executable = transpiler.transpile(
        expval_circuit,
        parameters=["rotation_angle"],
        bindings={"observable": observable},
    )
    expval_result = expval_executable.run(
        transpiler.executor(),
        bindings={"rotation_angle": rotation_angle},
    ).result()

    assert np.isclose(expval_result, float(num_qubits), atol=1e-6)


def _qft_qkernel_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build stdlib-QFT qkernel inverse roundtrip kernels.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after a QFT layer and its inverse are applied.
    """

    @qmc.qkernel
    def qft_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply the stdlib QFT to the whole register."""
        qs = qmc.qft(qs)
        return qs

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply the QFT layer and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs = qft_layer(qs)
        qs = qmc.inverse(qft_layer)(qs)
        qs[0] = qmc.x(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _iqft_qkernel_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build stdlib-IQFT qkernel inverse roundtrip kernels.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after an IQFT layer and its inverse are applied.
    """

    @qmc.qkernel
    def iqft_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply the stdlib IQFT to the whole register."""
        qs = qmc.iqft(qs)
        return qs

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply the IQFT layer and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs = iqft_layer(qs)
        qs = qmc.inverse(iqft_layer)(qs)
        qs[0] = qmc.x(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _qft_direct_callable_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build direct `inverse(qmc.qft)` callable roundtrip kernels.

    `qmc.inverse(qmc.qft)` maps directly to `qmc.iqft` (no qkernel wrapper),
    so this exercises the direct-callable QFT inverse path end to end.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after `qmc.qft` and `qmc.inverse(qmc.qft)` are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply qmc.qft and its direct inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs = qmc.qft(qs)
        qs = qmc.inverse(qmc.qft)(qs)
        qs[0] = qmc.x(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _rx_layer_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build algorithm rx_layer inverse roundtrip kernels.

    The `thetas` angle vector stays a kernel argument (rx_layer consumes a
    `Vector[Float]`), so the table binds it at transpile time.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after rx_layer and its inverse are applied.
    """

    def roundtrip(
        qs: qmc.Vector[qmc.Qubit],
        thetas: qmc.Vector[qmc.Float],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply rx_layer and its inverse with the same angles."""
        qs = rx_layer(qs, thetas, 0)
        qs = qmc.inverse(rx_layer)(qs, thetas, 0)
        return qs

    @qmc.qkernel
    def sample_circuit(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, thetas)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(
        thetas: qmc.Vector[qmc.Float],
        observable: qmc.Observable,
    ) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, thetas)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _modular_increment_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build algorithm modular_increment inverse roundtrip kernels.

    The register stays at two qubits so the QURI Parts modular-arithmetic
    support documented in LIMITATIONS.md is not exceeded by the forward op.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after modular_increment and its inverse are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply modular_increment and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs = modular_increment(qs)
        qs = qmc.inverse(modular_increment)(qs)
        qs[0] = qmc.x(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _ry_layer_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build algorithm ry_layer inverse roundtrip kernels.

    The `thetas` angle vector stays a kernel argument (ry_layer consumes a
    `Vector[Float]`), so the table binds it at transpile time.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after ry_layer and its inverse are applied.
    """

    def roundtrip(
        qs: qmc.Vector[qmc.Qubit],
        thetas: qmc.Vector[qmc.Float],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply ry_layer and its inverse with the same angles."""
        qs = ry_layer(qs, thetas, 0)
        qs = qmc.inverse(ry_layer)(qs, thetas, 0)
        return qs

    @qmc.qkernel
    def sample_circuit(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, thetas)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(
        thetas: qmc.Vector[qmc.Float],
        observable: qmc.Observable,
    ) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, thetas)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _rz_layer_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build algorithm rz_layer inverse roundtrip kernels.

    RZ leaves |0..0> invariant up to phase, so the layer pair runs inside
    an X sandwich to make the forward act on |1..1> instead.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after rz_layer and its inverse are applied.
    """

    def roundtrip(
        qs: qmc.Vector[qmc.Qubit],
        thetas: qmc.Vector[qmc.Float],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply rz_layer and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs[1] = qmc.x(qs[1])
        qs = rz_layer(qs, thetas, 0)
        qs = qmc.inverse(rz_layer)(qs, thetas, 0)
        qs[0] = qmc.x(qs[0])
        qs[1] = qmc.x(qs[1])
        return qs

    @qmc.qkernel
    def sample_circuit(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, thetas)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(
        thetas: qmc.Vector[qmc.Float],
        observable: qmc.Observable,
    ) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, thetas)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _cz_entangling_layer_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build algorithm cz_entangling_layer inverse roundtrip kernels.

    CZ leaves |0..0> invariant, so the layer pair runs inside an X sandwich
    on the first two qubits, where CZ(0, 1) picks up a phase.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after cz_entangling_layer and its inverse are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply cz_entangling_layer and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs[1] = qmc.x(qs[1])
        qs = cz_entangling_layer(qs)
        qs = qmc.inverse(cz_entangling_layer)(qs)
        qs[0] = qmc.x(qs[0])
        qs[1] = qmc.x(qs[1])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _cx_entangling_layer_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build algorithm cx_entangling_layer inverse roundtrip kernels.

    The X sandwich on qs[0] makes the forward CX ladder propagate the
    excitation across the register, so the inverse must undo a
    nontrivial basis permutation.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after cx_entangling_layer and its inverse are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply cx_entangling_layer and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs = cx_entangling_layer(qs)
        qs = qmc.inverse(cx_entangling_layer)(qs)
        qs[0] = qmc.x(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _phase_gadget_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build algorithm phase_gadget inverse roundtrip kernels.

    The gadget applies a multi-body Z phase, which leaves basis states
    invariant up to phase, so the pair runs inside an X sandwich. The
    `indices` vector stays a kernel argument bound by the table.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after phase_gadget and its inverse are applied.
    """

    def roundtrip(
        qs: qmc.Vector[qmc.Qubit],
        indices: qmc.Vector[qmc.UInt],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply phase_gadget and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs = phase_gadget(qs, indices, 0.37)
        qs = qmc.inverse(phase_gadget)(qs, indices, 0.37)
        qs[0] = qmc.x(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit(indices: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs, indices)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(
        indices: qmc.Vector[qmc.UInt],
        observable: qmc.Observable,
    ) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs, indices)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _x_mixer_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build algorithm x_mixer inverse roundtrip kernels.

    RX(2*beta) with a nonzero beta rotates every qubit away from |0>, so
    the forward acts nontrivially without a sandwich.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after x_mixer and its inverse are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply x_mixer and its inverse with the same beta."""
        qs = x_mixer(qs, 0.37)
        qs = qmc.inverse(x_mixer)(qs, 0.37)
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _modular_decrement_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build algorithm modular_decrement inverse roundtrip kernels.

    The register stays at two qubits so the QURI Parts modular-arithmetic
    support documented in LIMITATIONS.md is not exceeded by the forward op.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after modular_decrement and its inverse are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply modular_decrement and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs = modular_decrement(qs)
        qs = qmc.inverse(modular_decrement)(qs)
        qs[0] = qmc.x(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _computational_basis_state_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build state-preparation computational_basis_state roundtrip kernels.

    With bits = [1, 0] the forward prepares |10> (up to global phase) from
    |00>, so the inverse must restore the all-zero state without a sandwich.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after computational_basis_state and its inverse are
            applied.
    """

    def roundtrip(
        qs: qmc.Vector[qmc.Qubit],
        bits: qmc.Vector[qmc.UInt],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply computational_basis_state and its inverse with the same bits."""
        qs = computational_basis_state(qs, bits)
        qs = qmc.inverse(computational_basis_state)(qs, bits)
        return qs

    @qmc.qkernel
    def sample_circuit(bits: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, bits)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(
        bits: qmc.Vector[qmc.UInt],
        observable: qmc.Observable,
    ) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, bits)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _amplitude_encoding_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build state-preparation amplitude_encoding inverse roundtrip kernels.

    Both legs go through a register-parameterized qkernel wrapper,
    exercising Möttönen custom-composite inlining and inversion.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after amplitude encoding and its inverse are applied.
    """
    amplitudes = [0.2, 0.4, 0.4, 0.8]

    @qmc.qkernel
    def amplitude_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply Mottonen amplitude encoding to the whole register."""
        qs = amplitude_encoding(qs, amplitudes)
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = amplitude_layer(qs)
        qs = qmc.inverse(amplitude_layer)(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = amplitude_layer(qs)
        qs = qmc.inverse(amplitude_layer)(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _amplitude_encoding_from_angles_roundtrip_kernels() -> tuple[
    qmc.QKernel, qmc.QKernel
]:
    """Build state-preparation amplitude_encoding_from_angles roundtrip kernels.

    The parametric companion to `amplitude_encoding`: pre-computed Mottonen
    Ry angles are emitted as elementary gates rather than through a custom
    composite operation. The wrapper path is included for symmetry with
    `amplitude_encoding`, but it does not exercise `_inline_composite`.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after angle-driven amplitude encoding and its inverse
            are applied.
    """
    from qamomile.linalg import compute_mottonen_amplitude_encoding_ry_angles

    ry_angles = compute_mottonen_amplitude_encoding_ry_angles([0.2, 0.4, 0.4, 0.8])

    @qmc.qkernel
    def amplitude_angles_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply angle-driven Mottonen amplitude encoding to the register."""
        qs = amplitude_encoding_from_angles(qs, ry_angles)
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = amplitude_angles_layer(qs)
        qs = qmc.inverse(amplitude_angles_layer)(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = amplitude_angles_layer(qs)
        qs = qmc.inverse(amplitude_angles_layer)(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _initial_occupations_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build FQAOA initial_occupations inverse roundtrip kernels.

    The forward applies X to the first two qubits, so the inverse must
    restore the all-zero state without a sandwich.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after initial_occupations and its inverse are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply initial_occupations and its inverse for two fermions."""
        qs = initial_occupations(qs, 2)
        qs = qmc.inverse(initial_occupations)(qs, 2)
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _givens_rotation_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build FQAOA givens_rotation inverse roundtrip kernels.

    The X sandwich on qs[0] activates the controlled-RY inside the Givens
    rotation, so the inverse must undo a ControlledUOperation block.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after givens_rotation and its inverse are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply givens_rotation and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs = givens_rotation(qs, 0, 1, 0.37)
        qs = qmc.inverse(givens_rotation)(qs, 0, 1, 0.37)
        qs[0] = qmc.x(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _givens_rotations_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build FQAOA givens_rotations inverse roundtrip kernels.

    The `(N, 2)` index matrix and angle vector stay kernel arguments bound
    by the table, exercising Matrix[UInt]-indexed inversion.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after givens_rotations and its inverse are applied.
    """

    def roundtrip(
        qs: qmc.Vector[qmc.Qubit],
        givens_ij: qmc.Matrix[qmc.UInt],
        givens_theta: qmc.Vector[qmc.Float],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply givens_rotations and its inverse inside an X sandwich."""
        qs[0] = qmc.x(qs[0])
        qs = givens_rotations(qs, givens_ij, givens_theta)
        qs = qmc.inverse(givens_rotations)(qs, givens_ij, givens_theta)
        qs[0] = qmc.x(qs[0])
        return qs

    @qmc.qkernel
    def sample_circuit(
        givens_ij: qmc.Matrix[qmc.UInt],
        givens_theta: qmc.Vector[qmc.Float],
    ) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, givens_ij, givens_theta)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(
        givens_ij: qmc.Matrix[qmc.UInt],
        givens_theta: qmc.Vector[qmc.Float],
        observable: qmc.Observable,
    ) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs, givens_ij, givens_theta)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _hopping_gate_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build FQAOA hopping_gate inverse roundtrip kernels.

    The RX(pi/2) basis changes inside the hopping gate act nontrivially on
    |00>, so no sandwich is needed.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after hopping_gate and its inverse are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply hopping_gate and its inverse with the same parameters."""
        qs = hopping_gate(qs, 0, 1, 0.37, 1.0)
        qs = qmc.inverse(hopping_gate)(qs, 0, 1, 0.37, 1.0)
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


def _mixer_layer_roundtrip_kernels() -> tuple[qmc.QKernel, qmc.QKernel]:
    """Build FQAOA mixer_layer inverse roundtrip kernels.

    Covers the even/odd/boundary hopping sweeps, including the stepped
    `qmc.range(start, stop, 2)` loops, on a three-qubit register.

    Returns:
        tuple[qmc.QKernel, qmc.QKernel]: Sampling kernel that measures
            all-zero and expval kernel whose sum-Z expectation equals the
            width, after mixer_layer and its inverse are applied.
    """

    def roundtrip(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply mixer_layer and its inverse with the same parameters."""
        qs = mixer_layer(qs, 0.37, 1.0, 3)
        qs = qmc.inverse(mixer_layer)(qs, 0.37, 1.0, 3)
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        qs = qmc.qubit_array(3, "qs")
        qs = roundtrip(qs)
        return qmc.expval(qs, observable)

    return sample_circuit, expval_circuit


STDLIB_ALGO_ROUNDTRIP_CASES = [
    pytest.param(_qft_qkernel_roundtrip_kernels, 3, {}, id="qft"),
    pytest.param(_iqft_qkernel_roundtrip_kernels, 3, {}, id="iqft"),
    pytest.param(
        _qft_direct_callable_roundtrip_kernels,
        3,
        {},
        id="qft-direct-callable",
    ),
    pytest.param(
        _rx_layer_roundtrip_kernels,
        2,
        {"thetas": [0.37, 0.37]},
        id="algorithm-rx-layer",
    ),
    pytest.param(
        _ry_layer_roundtrip_kernels,
        2,
        {"thetas": [0.41, 0.23]},
        id="algorithm-ry-layer",
    ),
    pytest.param(
        _rz_layer_roundtrip_kernels,
        2,
        {"thetas": [0.41, 0.23]},
        id="algorithm-rz-layer",
    ),
    pytest.param(
        _cz_entangling_layer_roundtrip_kernels,
        3,
        {},
        id="algorithm-cz-entangling-layer",
    ),
    pytest.param(
        _cx_entangling_layer_roundtrip_kernels,
        3,
        {},
        id="algorithm-cx-entangling-layer",
    ),
    pytest.param(
        _phase_gadget_roundtrip_kernels,
        3,
        {"indices": [0, 1, 2]},
        id="algorithm-phase-gadget",
    ),
    pytest.param(
        _x_mixer_roundtrip_kernels,
        2,
        {},
        id="algorithm-x-mixer",
    ),
    pytest.param(
        _modular_increment_roundtrip_kernels,
        2,
        {},
        id="algorithm-modular-increment",
    ),
    pytest.param(
        _modular_decrement_roundtrip_kernels,
        2,
        {},
        id="algorithm-modular-decrement",
    ),
    pytest.param(
        _computational_basis_state_roundtrip_kernels,
        2,
        {"bits": [1, 0]},
        id="algorithm-computational-basis-state",
    ),
    pytest.param(
        _amplitude_encoding_roundtrip_kernels,
        2,
        {},
        id="algorithm-amplitude-encoding",
    ),
    pytest.param(
        _amplitude_encoding_from_angles_roundtrip_kernels,
        2,
        {},
        id="algorithm-amplitude-encoding-from-angles",
    ),
    pytest.param(
        _initial_occupations_roundtrip_kernels,
        3,
        {},
        id="algorithm-initial-occupations",
    ),
    pytest.param(
        _givens_rotation_roundtrip_kernels,
        2,
        {},
        id="algorithm-givens-rotation",
    ),
    pytest.param(
        _givens_rotations_roundtrip_kernels,
        2,
        {"givens_ij": np.array([[0, 1]]), "givens_theta": [0.37]},
        id="algorithm-givens-rotations",
    ),
    pytest.param(
        _hopping_gate_roundtrip_kernels,
        2,
        {},
        id="algorithm-hopping-gate",
    ),
    pytest.param(
        _mixer_layer_roundtrip_kernels,
        3,
        {},
        id="algorithm-mixer-layer",
    ),
]


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize(
    "kernel_factory, width, case_bindings",
    STDLIB_ALGO_ROUNDTRIP_CASES,
)
def test_inverse_stdlib_algorithm_roundtrip_cross_backend(
    transpiler_factory,
    kernel_factory,
    width: int,
    case_bindings: dict[str, object],
) -> None:
    """Stdlib and algorithm qkernel inverse roundtrips execute on backends.

    This is the execution matrix for inverses of stdlib kernels (qmc.qft /
    qmc.iqft, both qkernel-wrapped and as the direct callable mapping) and
    every invertible algorithm kernel: the basic rotation and entangling
    layers, phase_gadget, the QAOA x_mixer, modular increment/decrement,
    computational_basis_state, Mottonen amplitude encoding, and the FQAOA
    initial-state/mixer building blocks. Both backend primitives run per
    case: samples must be all-zero and the sum-Z expectation must equal
    the register width.
    """
    sample_kernel, expval_kernel = kernel_factory()
    transpiler = transpiler_factory()

    executable = transpiler.transpile(sample_kernel, bindings=case_bindings or None)
    sample_result = executable.sample(transpiler.executor(), shots=32).result()
    _assert_all_zero_samples(sample_result, width, 32)

    expval_executable = transpiler.transpile(
        expval_kernel,
        bindings={**case_bindings, "observable": _sum_z_observable(width)},
    )
    expval_result = expval_executable.run(transpiler.executor()).result()

    assert np.isclose(expval_result, float(width), atol=1e-6)


def test_inverse_nested_invoke_keeps_vector_target_width() -> None:
    """Nested inverse invokes count vector target width without parameters."""
    if QiskitTranspiler is None:
        pytest.skip("qiskit not installed")

    sample_kernel, _ = _givens_rotations_roundtrip_kernels()
    block = QiskitTranspiler().to_block(
        sample_kernel,
        bindings={"givens_ij": np.array([[0, 1]]), "givens_theta": [0.37]},
    )
    inverse_ops: list[InverseBlockOperation] = []

    def collect(operations: list[object]) -> None:
        """Collect inverse operations recursively from nested control flow."""
        for operation in operations:
            if isinstance(operation, InverseBlockOperation):
                inverse_ops.append(operation)
                if operation.implementation_block is not None:
                    collect(list(operation.implementation_block.operations))
            elif hasattr(operation, "nested_op_lists"):
                for nested in operation.nested_op_lists():
                    collect(list(nested))

    collect(list(block.operations))

    inner = next(
        op for op in inverse_ops if op.custom_name == "givens_rotation_inverse"
    )
    assert inner.num_target_qubits == 2
    assert len(inner.target_qubits) == 1
    assert len(inner.parameters) == 3


# ---------------------------------------------------------------------------
# InverseBlockOperation construction validation and _static_quantum_width
# ---------------------------------------------------------------------------


def test_inverse_block_operation_rejects_interleaved_parameters() -> None:
    """InverseBlockOperation rejects quantum targets after parameters."""
    q0 = Value(type=QubitType(), name="q0")
    parameter = Value(type=UIntType(), name="theta")
    q1 = Value(type=QubitType(), name="q1")

    with pytest.raises(ValueError, match="quantum target operands must precede"):
        InverseBlockOperation(
            operands=[q0, parameter, q1],
            results=[q0.next_version(), q1.next_version()],
            num_control_qubits=0,
            num_target_qubits=2,
        )


def test_inverse_block_operation_rejects_missing_result() -> None:
    """The constructor rejects a result list shorter than the operand layout.

    Downstream passes pair operands with results by ``zip``; a malformed
    op (e.g. from hand-built IR) must fail at construction instead of
    being silently part-processed.
    """
    ctrl = Value(type=QubitType(), name="ctrl")
    target = Value(type=QubitType(), name="target")

    with pytest.raises(ValueError, match="results must mirror"):
        InverseBlockOperation(
            operands=[ctrl, target],
            results=[ctrl.next_version()],
            num_control_qubits=1,
            num_target_qubits=1,
            custom_name="malformed",
        )


def test_inverse_block_operation_rejects_classical_result() -> None:
    """The constructor rejects non-quantum values in the result list."""
    target = Value(type=QubitType(), name="target")

    with pytest.raises(ValueError, match="must be quantum"):
        InverseBlockOperation(
            operands=[target],
            results=[Value(type=UIntType(), name="bad")],
            num_control_qubits=0,
            num_target_qubits=1,
            custom_name="malformed",
        )


def test_inverse_block_operation_rejects_arrayness_mismatch() -> None:
    """The constructor rejects a scalar result for a vector target operand."""
    dim = Value(type=UIntType(), name="dim").with_const(2)
    target = ArrayValue(type=QubitType(), name="target", shape=(dim,))

    with pytest.raises(ValueError, match="array-ness"):
        InverseBlockOperation(
            operands=[target],
            results=[Value(type=QubitType(), name="scalar")],
            num_control_qubits=0,
            num_target_qubits=2,
            custom_name="malformed",
        )


def test_static_quantum_width_multiplies_all_dimensions() -> None:
    """``_static_quantum_width`` counts scalar qubits across every dimension.

    The width helper must stay correct for any array rank so that
    ``InverseBlockOperation.num_target_qubits`` can never understate a
    register's scalar width (a rank-2 ``(2, 3)`` value is 6 qubits, not
    the first dimension's 2).
    """

    def dim(value: int) -> Value:
        return Value(type=UIntType(), name="dim").with_const(value)

    scalar = Value(type=QubitType(), name="q")
    assert _static_quantum_width(scalar) == 1

    vector = ArrayValue(type=QubitType(), name="v", shape=(dim(3),))
    assert _static_quantum_width(vector) == 3

    matrix = ArrayValue(type=QubitType(), name="m", shape=(dim(2), dim(3)))
    assert _static_quantum_width(matrix) == 6

    symbolic = ArrayValue(
        type=QubitType(),
        name="s",
        shape=(dim(2), Value(type=UIntType(), name="n")),
    )
    assert _static_quantum_width(symbolic) is None
