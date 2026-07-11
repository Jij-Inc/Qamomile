"""Tests for the uncompute frontend primitive (``qmc.uncompute``).

``qmc.uncompute(target, *args, **kwargs)`` applies the adjoint of ``target``
at the current trace site (the explicit *uncompute* half of the
compute--use--uncompute idiom). It is a thin wrapper over ``qmc.inverse``,
so these tests focus on (1) end-to-end execution across every supported
backend on both the sampling and expectation-value paths, (2) equivalence
with the ``qmc.inverse(target)(...)`` spelling it replaces, and (3) faithful
propagation of ``inverse``'s diagnostics.
"""

import math

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.stdlib import QFT
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

# Self-inverse and dagger-pair native gates whose uncompute restores |0>.
NATIVE_NONPARAMETRIC_GATES = ("h", "x", "y", "z", "s", "sdg", "t", "tdg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _angle_from_case(angle_case: float | tuple[str, int]) -> float:
    """Return a deterministic angle for a boundary or seeded-random case.

    Args:
        angle_case (float | tuple[str, int]): Explicit boundary angle in
            radians, or a ``("random", seed)`` pair.

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
        sample_result (object): Backend sample result exposing a ``results``
            iterable of ``(bitstring, count)`` pairs.
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
        qm_o.Hamiltonian: Observable summing ``Z(idx)`` over every qubit
            index, so the all-zero state has expectation ``width``.
    """
    observable = qm_o.Hamiltonian.zero(num_qubits=width)
    for idx in range(width):
        observable += qm_o.Z(idx)
    return observable


@qmc.qkernel
def _prep_layer(q: qmc.Qubit, rotation_angle: qmc.Float) -> qmc.Qubit:
    """Apply a small non-trivial layer used as the compute step.

    Args:
        q (qmc.Qubit): Target qubit.
        rotation_angle (qmc.Float): Z-rotation angle in radians.

    Returns:
        qmc.Qubit: The transformed qubit handle.
    """
    q = qmc.h(q)
    q = qmc.rz(q, rotation_angle)
    q = qmc.t(q)
    return q


@qmc.qkernel
def _two_qubit_layer(
    a: qmc.Qubit,
    b: qmc.Qubit,
    rotation_angle: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply an entangling layer over two qubits used as the compute step.

    Args:
        a (qmc.Qubit): First target qubit.
        b (qmc.Qubit): Second target qubit.
        rotation_angle (qmc.Float): Z-rotation angle in radians.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: The transformed qubit handles.
    """
    a = qmc.h(a)
    a, b = qmc.cx(a, b)
    b = qmc.rz(b, rotation_angle)
    return a, b


# ---------------------------------------------------------------------------
# Cross-backend execution: compute -> uncompute is the identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5])
@pytest.mark.parametrize("angle_case", ANGLE_CASES)
def test_uncompute_roundtrip_cross_backend_sampling_and_expval(
    transpiler_factory,
    num_qubits: int,
    angle_case: float | tuple[str, int],
) -> None:
    """uncompute(qkernel) undoes the compute step on sampling and expval paths.

    Sampling and expectation values go through different backend primitives
    (sampler vs estimator) and regress independently, so both legs run for
    every case: samples must be all-zero and the sum-Z expectation must
    equal the register width.
    """

    @qmc.qkernel
    def sample_circuit(rotation_angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(num_qubits, "qs")
        for idx in qmc.range(num_qubits):
            qs[idx] = _prep_layer(qs[idx], rotation_angle)
            qs[idx] = qmc.uncompute(_prep_layer, qs[idx], rotation_angle)
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(
        rotation_angle: qmc.Float,
        observable: qmc.Observable,
    ) -> qmc.Float:
        qs = qmc.qubit_array(num_qubits, "qs")
        for idx in qmc.range(num_qubits):
            qs[idx] = _prep_layer(qs[idx], rotation_angle)
            qs[idx] = qmc.uncompute(_prep_layer, qs[idx], rotation_angle)
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

    expval_executable = transpiler.transpile(
        expval_circuit,
        parameters=["rotation_angle"],
        bindings={"observable": _sum_z_observable(num_qubits)},
    )
    expval_result = expval_executable.run(
        transpiler.executor(),
        bindings={"rotation_angle": rotation_angle},
    ).result()

    assert np.isclose(expval_result, float(num_qubits), atol=1e-6)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("gate_name", NATIVE_NONPARAMETRIC_GATES)
def test_uncompute_native_gate_roundtrip_cross_backend(
    transpiler_factory,
    gate_name: str,
) -> None:
    """uncompute(native gate) restores |0> on sampling and expval paths."""
    gate = getattr(qmc, gate_name)

    @qmc.qkernel
    def sample_circuit() -> qmc.Bit:
        q = qmc.qubit("q")
        q = gate(q)
        q = qmc.uncompute(gate, q)
        return qmc.measure(q)

    @qmc.qkernel
    def expval_circuit(observable: qmc.Observable) -> qmc.Float:
        q = qmc.qubit("q")
        q = gate(q)
        q = qmc.uncompute(gate, q)
        return qmc.expval(q, observable)

    transpiler = transpiler_factory()

    sample_executable = transpiler.transpile(sample_circuit)
    sample_result = sample_executable.sample(
        transpiler.executor(),
        shots=64,
    ).result()
    _assert_all_zero_samples(sample_result, 1, 64)

    expval_executable = transpiler.transpile(
        expval_circuit,
        bindings={"observable": _sum_z_observable(1)},
    )
    expval_result = expval_executable.run(transpiler.executor()).result()
    assert np.isclose(expval_result, 1.0, atol=1e-6)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("angle_case", ANGLE_CASES)
def test_uncompute_native_rotation_kwargs_roundtrip_cross_backend(
    transpiler_factory,
    angle_case: float | tuple[str, int],
) -> None:
    """uncompute(rotation gate) negates the angle and forwards keyword args.

    Exercises the native ``_InverseRotationCallable`` path directly (rather
    than through a wrapping kernel) and confirms keyword-argument forwarding
    through ``uncompute``'s ``**kwargs``.
    """

    @qmc.qkernel
    def sample_circuit(rotation_angle: qmc.Float) -> qmc.Bit:
        q = qmc.qubit("q")
        q = qmc.rx(q, rotation_angle)
        q = qmc.uncompute(qmc.rx, q, angle=rotation_angle)
        return qmc.measure(q)

    @qmc.qkernel
    def expval_circuit(
        rotation_angle: qmc.Float,
        observable: qmc.Observable,
    ) -> qmc.Float:
        q = qmc.qubit("q")
        q = qmc.rx(q, rotation_angle)
        q = qmc.uncompute(qmc.rx, q, angle=rotation_angle)
        return qmc.expval(q, observable)

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
    _assert_all_zero_samples(sample_result, 1, 64)

    expval_executable = transpiler.transpile(
        expval_circuit,
        parameters=["rotation_angle"],
        bindings={"observable": _sum_z_observable(1)},
    )
    expval_result = expval_executable.run(
        transpiler.executor(),
        bindings={"rotation_angle": rotation_angle},
    ).result()
    assert np.isclose(expval_result, 1.0, atol=1e-6)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
@pytest.mark.parametrize("angle_case", ANGLE_CASES)
def test_uncompute_multi_input_tuple_roundtrip_cross_backend(
    transpiler_factory,
    angle_case: float | tuple[str, int],
) -> None:
    """uncompute returns the tuple shape of a multi-quantum-input kernel."""

    @qmc.qkernel
    def sample_circuit(rotation_angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        qs[0], qs[1] = _two_qubit_layer(qs[0], qs[1], rotation_angle)
        qs[0], qs[1] = qmc.uncompute(
            _two_qubit_layer,
            qs[0],
            qs[1],
            rotation_angle,
        )
        return qmc.measure(qs)

    @qmc.qkernel
    def expval_circuit(
        rotation_angle: qmc.Float,
        observable: qmc.Observable,
    ) -> qmc.Float:
        qs = qmc.qubit_array(2, "qs")
        qs[0], qs[1] = _two_qubit_layer(qs[0], qs[1], rotation_angle)
        qs[0], qs[1] = qmc.uncompute(
            _two_qubit_layer,
            qs[0],
            qs[1],
            rotation_angle,
        )
        return qmc.expval(qs, observable)

    rotation_angle = _angle_from_case(angle_case)
    transpiler = transpiler_factory()

    executable = transpiler.transpile(sample_circuit, parameters=["rotation_angle"])
    sample_result = executable.sample(
        transpiler.executor(),
        shots=64,
        bindings={"rotation_angle": rotation_angle},
    ).result()
    _assert_all_zero_samples(sample_result, 2, 64)

    expval_executable = transpiler.transpile(
        expval_circuit,
        parameters=["rotation_angle"],
        bindings={"observable": _sum_z_observable(2)},
    )
    expval_result = expval_executable.run(
        transpiler.executor(),
        bindings={"rotation_angle": rotation_angle},
    ).result()
    assert np.isclose(expval_result, 2.0, atol=1e-6)


@pytest.mark.parametrize("transpiler_factory", BACKENDS)
def test_uncompute_stdlib_qft_roundtrip_cross_backend(transpiler_factory) -> None:
    """uncompute(qmc.qft) applies the inverse QFT and restores the state."""

    @qmc.qkernel
    def qft_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply the stdlib QFT to the whole register."""
        qs = qmc.qft(qs)
        return qs

    @qmc.qkernel
    def sample_circuit() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(3, "qs")
        qs[0] = qmc.x(qs[0])
        qs = qft_layer(qs)
        qs = qmc.uncompute(qft_layer, qs)
        qs[0] = qmc.x(qs[0])
        return qmc.measure(qs)

    transpiler = transpiler_factory()
    executable = transpiler.transpile(sample_circuit)
    sample_result = executable.sample(transpiler.executor(), shots=64).result()
    _assert_all_zero_samples(sample_result, 3, 64)


# ---------------------------------------------------------------------------
# Equivalence with the inverse(...) spelling it replaces
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
@pytest.mark.parametrize("angle_case", ANGLE_CASES)
def test_uncompute_matches_inverse_spelling_statevector(
    angle_case: float | tuple[str, int],
) -> None:
    """uncompute(layer, q, a) produces the same circuit as inverse(layer)(q, a)."""

    @qmc.qkernel
    def uncompute_circuit(rotation_angle: qmc.Float) -> qmc.Bit:
        q = qmc.qubit("q")
        q = _prep_layer(q, rotation_angle)
        q = qmc.uncompute(_prep_layer, q, rotation_angle)
        return qmc.measure(q)

    @qmc.qkernel
    def inverse_circuit(rotation_angle: qmc.Float) -> qmc.Bit:
        q = qmc.qubit("q")
        q = _prep_layer(q, rotation_angle)
        q = qmc.inverse(_prep_layer)(q, rotation_angle)
        return qmc.measure(q)

    rotation_angle = _angle_from_case(angle_case)
    transpiler = QiskitTranspiler()
    uncompute_qc = transpiler.to_circuit(
        uncompute_circuit,
        bindings={"rotation_angle": rotation_angle},
    )
    inverse_qc = transpiler.to_circuit(
        inverse_circuit,
        bindings={"rotation_angle": rotation_angle},
    )
    assert np.allclose(run_statevector(uncompute_qc), run_statevector(inverse_qc))


# ---------------------------------------------------------------------------
# Diagnostics are inherited from inverse()
# ---------------------------------------------------------------------------


def test_uncompute_rejects_direct_composite_gate_instance() -> None:
    """uncompute() surfaces inverse()'s guidance for direct CompositeGate use."""
    with pytest.raises(TypeError, match="direct CompositeGate instances"):
        qmc.uncompute(QFT(2))


def test_uncompute_propagates_unsupported_if_operation() -> None:
    """uncompute(qkernel) reports unsupported IfOperation explicitly."""

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
        q = qmc.uncompute(branch_layer, q, flag)
        return q

    with pytest.raises(NotImplementedError, match="IfOperation"):
        circuit.build()
