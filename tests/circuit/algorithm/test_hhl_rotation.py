"""Tests for qamomile/circuit/algorithm/hhl_rotation.py.

``reciprocal_rotation_2clock_le`` rotates the ancilla by ``Ry(theta_raw)``
when the 2-qubit clock register is in basis state ``|raw>``;
``hhl_middle_block_2clock_le`` wraps the same rotation in ``IQFT``/``QFT``.

For a clock basis state ``|raw>`` the ancilla ends in
``Ry(theta_raw) |0>``, an unentangled single-qubit rotation, so the
results are checked against the analytic ``P(anc=1) = sin^2(theta/2)``
(sampling) and ``<Z_anc> = cos(theta)`` (expectation value).  Every
check runs on each installed quantum backend (Qiskit, QURI Parts,
CUDA-Q); the latter two are guarded by ``importorskip`` and the
``quri_parts`` / ``cudaq`` pytest markers.
"""

import math

import numpy as np
import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.circuit.algorithm import (  # noqa: E402
    computational_basis_state,
    hhl_middle_block_2clock_le,
    reciprocal_rotation_2clock_le,
)
from qamomile.circuit.stdlib.qft import qft  # noqa: E402
from qamomile.circuit.transpiler.job import SampleResult  # noqa: E402

# Index of the ancilla qubit inside the (c0, c1, ancilla) tuple handed
# to ``qmc.expval`` -- the observable acts as Z on this position.
_ANCILLA_INDEX = 2


# ----------------------------------------------------------------------
# Backend fixture
# ----------------------------------------------------------------------


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def backend(request):
    """Yield ``(name, transpiler, executor)`` for each installed backend.

    Args:
        request (pytest.FixtureRequest): The pytest request object whose
            ``param`` field selects the backend (``"qiskit"``,
            ``"quri_parts"`` or ``"cudaq"``).

    Returns:
        tuple[str, object, object]: The backend name, a transpiler for
            that backend, and a fresh executor obtained from it.

    Raises:
        AssertionError: If ``request.param`` is not a recognised backend
            name (an internal invariant -- the fixture's ``params`` list
            is the only source of values).
    """
    name = request.param
    if name == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "quri_parts":
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        transpiler = CudaqTranspiler()
        return name, transpiler, transpiler.executor()
    raise AssertionError(f"unknown backend {name}")


# ----------------------------------------------------------------------
# Wrapper kernels (entry points that allocate qubits and measure)
# ----------------------------------------------------------------------


@qmc.qkernel
def _recip_sample(
    bits: qmc.Vector[qmc.UInt],
    theta1: qmc.Float,
    theta2: qmc.Float,
    theta3: qmc.Float,
) -> qmc.Bit:
    """Prepare clock ``|bits>``, apply the reciprocal rotation, sample anc."""
    clock = qmc.qubit_array(2, name="clock")
    clock = computational_basis_state(clock, bits)
    anc = qmc.qubit("anc")
    clock[0], clock[1], anc = reciprocal_rotation_2clock_le(
        clock[0], clock[1], anc, theta1, theta2, theta3
    )
    return qmc.measure(anc)


@qmc.qkernel
def _recip_expval(
    bits: qmc.Vector[qmc.UInt],
    theta1: qmc.Float,
    theta2: qmc.Float,
    theta3: qmc.Float,
    obs: qmc.Observable,
) -> qmc.Float:
    """Prepare clock ``|bits>``, apply the reciprocal rotation, estimate ``obs``."""
    clock = qmc.qubit_array(2, name="clock")
    clock = computational_basis_state(clock, bits)
    anc = qmc.qubit("anc")
    clock[0], clock[1], anc = reciprocal_rotation_2clock_le(
        clock[0], clock[1], anc, theta1, theta2, theta3
    )
    return qmc.expval((clock[0], clock[1], anc), obs)


@qmc.qkernel
def _block_sample(
    bits: qmc.Vector[qmc.UInt],
    theta1: qmc.Float,
    theta2: qmc.Float,
    theta3: qmc.Float,
) -> qmc.Bit:
    """Phase-encode clock ``|bits>``, apply the middle block, sample anc."""
    clock = qmc.qubit_array(2, name="clock")
    clock = computational_basis_state(clock, bits)
    clock = qft(clock)
    anc = qmc.qubit("anc")
    clock[0], clock[1], anc = hhl_middle_block_2clock_le(
        clock[0], clock[1], anc, theta1, theta2, theta3
    )
    return qmc.measure(anc)


@qmc.qkernel
def _block_expval(
    bits: qmc.Vector[qmc.UInt],
    theta1: qmc.Float,
    theta2: qmc.Float,
    theta3: qmc.Float,
    obs: qmc.Observable,
) -> qmc.Float:
    """Phase-encode clock ``|bits>``, apply the middle block, estimate ``obs``."""
    clock = qmc.qubit_array(2, name="clock")
    clock = computational_basis_state(clock, bits)
    clock = qft(clock)
    anc = qmc.qubit("anc")
    clock[0], clock[1], anc = hhl_middle_block_2clock_le(
        clock[0], clock[1], anc, theta1, theta2, theta3
    )
    return qmc.expval((clock[0], clock[1], anc), obs)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _raw_to_bits(raw: int) -> list[int]:
    """Return the little-endian ``[c0, c1]`` bit list for clock value ``raw``.

    Args:
        raw (int): Clock register value in ``range(4)``.

    Returns:
        list[int]: The two-element ``[c0, c1]`` bit list, little-endian,
            so that ``raw == c0 + 2 * c1``.
    """
    return [raw & 1, (raw >> 1) & 1]


def _expected(raw: int, thetas: np.ndarray) -> tuple[float, float]:
    """Return analytic ``(P(anc=1), <Z_anc>)`` for clock ``|raw>``.

    Bin ``raw == 0`` is the zero-eigenvalue bin and is left untouched, so
    the ancilla stays in ``|0>``.

    Args:
        raw (int): Clock register value in ``range(4)``.
        thetas (np.ndarray): The three ``Ry`` angles, in radians, for
            clock bins 1, 2 and 3.

    Returns:
        tuple[float, float]: The analytic ancilla excitation probability
            ``P(anc=1) = sin^2(theta_raw/2)`` and expectation value
            ``<Z_anc> = cos(theta_raw)`` -- ``(0.0, 1.0)`` for ``raw == 0``.
    """
    if raw == 0:
        return 0.0, 1.0
    theta = float(thetas[raw - 1])
    return math.sin(theta / 2.0) ** 2, math.cos(theta)


def _ancilla_observable() -> qm_o.Hamiltonian:
    """Return the single-Pauli ``Z`` observable acting on the ancilla qubit.

    Returns:
        qm_o.Hamiltonian: The ``Z`` observable on qubit
            :data:`_ANCILLA_INDEX` of the ``(c0, c1, ancilla)`` tuple
            passed to ``qmc.expval``.
    """
    return qm_o.Z(_ANCILLA_INDEX)


def _sampled_p1(result: SampleResult) -> float:
    """Return the measured ``P(anc=1)`` from a single-bit ``SampleResult``.

    Args:
        result (SampleResult): Result of sampling a kernel whose single
            classical output is the ancilla bit.

    Returns:
        float: Fraction of shots in which the ancilla measured ``1``.
    """
    ones = sum(count for value, count in result.results if int(value) == 1)
    return ones / result.shots


# Shot-noise tolerance: at 8192 shots the worst-case (p = 0.5) standard
# deviation is ~0.0055, so 0.04 keeps the assertion ~7 sigma from a
# correct distribution while still catching a genuine miscompilation.
_SAMPLING_ATOL = 0.04
# Statevector estimators are exact up to floating point.
_EXPVAL_ATOL = 1e-6
_SHOTS = 8192

_SEEDS = [0, 1, 2, 42]
_RAW_VALUES = [0, 1, 2, 3]
_BOUNDARY_THETAS = [0.0, math.pi, 2.0 * math.pi]


# ----------------------------------------------------------------------
# reciprocal_rotation_2clock_le
# ----------------------------------------------------------------------


class TestReciprocalRotation:
    """``reciprocal_rotation_2clock_le`` rotates the ancilla per clock bin."""

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("raw", _RAW_VALUES)
    def test_sampling_matches_analytic(self, backend, seed, raw):
        """Sampled ``P(anc=1)`` matches ``sin^2(theta_raw/2)``."""
        _, transpiler, executor = backend
        thetas = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, size=3)
        exe = transpiler.transpile(
            _recip_sample,
            bindings={
                "bits": _raw_to_bits(raw),
                "theta1": thetas[0],
                "theta2": thetas[1],
                "theta3": thetas[2],
            },
        )
        result = exe.sample(executor, shots=_SHOTS).result()
        expected_p1, _ = _expected(raw, thetas)
        assert math.isclose(_sampled_p1(result), expected_p1, abs_tol=_SAMPLING_ATOL)

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("raw", _RAW_VALUES)
    def test_expval_matches_analytic(self, backend, seed, raw):
        """Estimated ``<Z_anc>`` matches ``cos(theta_raw)``."""
        _, transpiler, executor = backend
        thetas = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, size=3)
        exe = transpiler.transpile(
            _recip_expval,
            bindings={
                "bits": _raw_to_bits(raw),
                "theta1": thetas[0],
                "theta2": thetas[1],
                "theta3": thetas[2],
                "obs": _ancilla_observable(),
            },
        )
        got = exe.run(executor).result()
        _, expected_z = _expected(raw, thetas)
        assert math.isclose(float(got), expected_z, abs_tol=_EXPVAL_ATOL)

    @pytest.mark.parametrize("raw", [1, 2, 3])
    @pytest.mark.parametrize("theta", _BOUNDARY_THETAS)
    def test_boundary_thetas(self, backend, raw, theta):
        """Boundary angles 0, pi, 2pi give the exact ``<Z_anc>`` (1, -1, 1)."""
        _, transpiler, executor = backend
        exe = transpiler.transpile(
            _recip_expval,
            bindings={
                "bits": _raw_to_bits(raw),
                "theta1": theta,
                "theta2": theta,
                "theta3": theta,
                "obs": _ancilla_observable(),
            },
        )
        got = exe.run(executor).result()
        assert math.isclose(float(got), math.cos(theta), abs_tol=_EXPVAL_ATOL)

    def test_zero_bin_leaves_ancilla_untouched(self, backend):
        """Clock ``|raw=0>`` never rotates the ancilla, for any angles."""
        _, transpiler, executor = backend
        exe = transpiler.transpile(
            _recip_expval,
            bindings={
                "bits": _raw_to_bits(0),
                "theta1": 1.3,
                "theta2": 2.1,
                "theta3": 0.7,
                "obs": _ancilla_observable(),
            },
        )
        got = exe.run(executor).result()
        assert math.isclose(float(got), 1.0, abs_tol=_EXPVAL_ATOL)


# ----------------------------------------------------------------------
# hhl_middle_block_2clock_le
# ----------------------------------------------------------------------


class TestHhlMiddleBlock:
    """``hhl_middle_block_2clock_le`` = IQFT -> reciprocal rotation -> QFT.

    Fed a phase-encoded clock ``qft(|raw>)``, the leading IQFT recovers
    the computational basis state ``|raw>``, so the ancilla outcome is
    identical to the bare reciprocal rotation.
    """

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("raw", _RAW_VALUES)
    def test_sampling_matches_analytic(self, backend, seed, raw):
        """Sampled ``P(anc=1)`` matches ``sin^2(theta_raw/2)``."""
        _, transpiler, executor = backend
        thetas = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, size=3)
        exe = transpiler.transpile(
            _block_sample,
            bindings={
                "bits": _raw_to_bits(raw),
                "theta1": thetas[0],
                "theta2": thetas[1],
                "theta3": thetas[2],
            },
        )
        result = exe.sample(executor, shots=_SHOTS).result()
        expected_p1, _ = _expected(raw, thetas)
        assert math.isclose(_sampled_p1(result), expected_p1, abs_tol=_SAMPLING_ATOL)

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("raw", _RAW_VALUES)
    def test_expval_matches_analytic(self, backend, seed, raw):
        """Estimated ``<Z_anc>`` matches ``cos(theta_raw)``."""
        _, transpiler, executor = backend
        thetas = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, size=3)
        exe = transpiler.transpile(
            _block_expval,
            bindings={
                "bits": _raw_to_bits(raw),
                "theta1": thetas[0],
                "theta2": thetas[1],
                "theta3": thetas[2],
                "obs": _ancilla_observable(),
            },
        )
        got = exe.run(executor).result()
        _, expected_z = _expected(raw, thetas)
        assert math.isclose(float(got), expected_z, abs_tol=_EXPVAL_ATOL)

    @pytest.mark.parametrize("raw", [1, 2, 3])
    @pytest.mark.parametrize("theta", _BOUNDARY_THETAS)
    def test_boundary_thetas(self, backend, raw, theta):
        """Boundary angles 0, pi, 2pi give the exact ``<Z_anc>`` (1, -1, 1)."""
        _, transpiler, executor = backend
        exe = transpiler.transpile(
            _block_expval,
            bindings={
                "bits": _raw_to_bits(raw),
                "theta1": theta,
                "theta2": theta,
                "theta3": theta,
                "obs": _ancilla_observable(),
            },
        )
        got = exe.run(executor).result()
        assert math.isclose(float(got), math.cos(theta), abs_tol=_EXPVAL_ATOL)
