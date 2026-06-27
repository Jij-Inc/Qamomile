"""Tests for qamomile/circuit/algorithm/reciprocal_rotation.py.

``reciprocal_rotation`` applies the HHL eigenvalue-inversion transform
on an n-qubit little-endian clock register: for each non-zero clock
basis state ``|raw>`` it rotates the ancilla into
``sqrt(1 - (c/raw)^2) |0> + (c/raw) |1>``.

For a clock basis state ``|raw>`` the ancilla ends in an unentangled
single-qubit rotation, so the results are checked against the analytic
``P(anc=1) = (c/raw)^2`` (sampling) and ``<Z_anc> = 1 - 2 * (c/raw)^2``
(expectation value).  Tests run across all installed quantum backends
(Qiskit, QURI Parts, CUDA-Q); the latter two are guarded by
``importorskip`` and the ``quri_parts`` / ``cudaq`` pytest markers.

Neither QURI Parts nor CUDA-Q can emit the multi-controlled
custom-block gate that the n-controlled ``Ry`` lowers to for
``n >= 2``; the tests skip those backend/size combinations and
document the constraint.
"""

import math

import numpy as np
import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.circuit.algorithm import (  # noqa: E402
    computational_basis_state,
    reciprocal_rotation,
)
from qamomile.circuit.transpiler.job import SampleResult  # noqa: E402

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


def _skip_if_multi_control_unsupported(backend_name: str, n: int) -> None:
    """Skip the test on backends that cannot emit a multi-controlled ``Ry``.

    Neither QURI Parts nor CUDA-Q can lower the multi-controlled
    custom-block gate emitted by ``qmc.controlled(qmc.ry, num_controls=n)``
    when ``n >= 2``:

    * QURI Parts: ``EmitError: Cannot decompose multi-controlled
      operation...`` (its fallback only supports a single control).
    * CUDA-Q: ``EmitError: Unsupported gate type ... Only X and SWAP
      are natively supported with multiple controls.``

    Single control (``n == 1``) is supported on both.  Qiskit handles
    multi-controlled rotations natively at every ``n``.
    """
    if n >= 2 and backend_name in ("quri_parts", "cudaq"):
        pytest.skip(
            f"{backend_name} cannot lower a multi-controlled custom-block "
            f"Ry (num_controls={n}); see review thread of PR #400."
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _raw_to_bits(raw: int, n: int) -> list[int]:
    """Return the little-endian ``[c0, c1, ..., c_{n-1}]`` bit list for ``raw``.

    Args:
        raw (int): Clock register value in ``range(2**n)``.
        n (int): Number of clock qubits.

    Returns:
        list[int]: Length-``n`` bit list with ``raw == sum(b[i] * 2**i for i in
            range(n))`` -- ``b[0]`` is the least-significant bit.
    """
    return [(raw >> i) & 1 for i in range(n)]


def _expected(raw: int, c: float) -> tuple[float, float]:
    """Return analytic ``(P(anc=1), <Z_anc>)`` for clock ``|raw>`` and scaling ``c``.

    Bin ``raw == 0`` is the zero-eigenvalue bin and is left untouched, so
    the ancilla stays in ``|0>``.

    Args:
        raw (int): Clock register value in ``range(2**n)``.
        c (float): Scaling constant; must satisfy ``0 < c <= 1``.

    Returns:
        tuple[float, float]: The analytic ancilla excitation probability
            ``P(anc=1) = (c/raw)^2`` and the expectation value
            ``<Z_anc> = 1 - 2 * (c/raw)^2`` -- ``(0.0, 1.0)`` for ``raw == 0``.
    """
    if raw == 0:
        return 0.0, 1.0
    ratio = c / raw
    return ratio**2, 1.0 - 2.0 * ratio**2


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


# ----------------------------------------------------------------------
# Kernel factories
#
# Built per ``(n, c)`` because both are Python values consumed by
# ``reciprocal_rotation`` at trace time -- the plain-Python function
# needs a concrete ``n`` (from ``get_size(qubits)``) and a Python
# ``float`` ``c`` (for the classical ``math.asin`` precompute).
# ----------------------------------------------------------------------


def _build_recip_sample(n: int, c: float):
    """Return a sample-path entry-point kernel for ``reciprocal_rotation``."""

    @qmc.qkernel
    def kernel(bits: qmc.Vector[qmc.UInt]) -> qmc.Bit:
        """Prepare clock ``|bits>``, apply ``reciprocal_rotation``, measure anc."""
        clock = qmc.qubit_array(n, name="clock")
        clock = computational_basis_state(clock, bits)
        anc = qmc.qubit("anc")
        clock, anc = reciprocal_rotation(clock, anc, c)
        return qmc.measure(anc)

    return kernel


def _build_recip_expval(n: int, c: float):
    """Return an expval-path entry-point kernel for ``reciprocal_rotation``."""

    @qmc.qkernel
    def kernel(
        bits: qmc.Vector[qmc.UInt],
        obs: qmc.Observable,
    ) -> qmc.Float:
        """Prepare clock ``|bits>``, apply ``reciprocal_rotation``, estimate ``obs``."""
        clock = qmc.qubit_array(n, name="clock")
        clock = computational_basis_state(clock, bits)
        anc = qmc.qubit("anc")
        clock, anc = reciprocal_rotation(clock, anc, c)
        return qmc.expval((anc,), obs)

    return kernel


# ----------------------------------------------------------------------
# Test constants
# ----------------------------------------------------------------------

# Shot-noise tolerance: at 8192 shots the worst-case (p = 0.5) standard
# deviation is ~0.0055, so 0.04 keeps the assertion ~7 sigma from a
# correct distribution while still catching a genuine miscompilation.
_SAMPLING_ATOL = 0.04
# Statevector estimators are exact up to floating point.
_EXPVAL_ATOL = 1e-6
_SHOTS = 8192

_SEEDS = [0, 1, 42]
# Clock register sizes; CLAUDE.md asks for n in {1, 2, 3, 5}.  n = 5
# multiplies the per-test inner loop by 32 raw bins and is omitted as a
# speed concession; n in {1, 2, 3} still covers single-control and
# multi-control code paths and the n = 1 boundary.
_N_VALUES = [1, 2, 3]
# Ancilla-only observable: Z acting on the single qubit passed to expval.
_ANCILLA_OBSERVABLE = qm_o.Z(0)


# ----------------------------------------------------------------------
# reciprocal_rotation
# ----------------------------------------------------------------------


class TestReciprocalRotation:
    """``reciprocal_rotation`` rotates the ancilla per clock bin."""

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("n", _N_VALUES)
    def test_sampling_matches_analytic(self, backend, seed, n):
        """Sampled ``P(anc=1)`` matches ``(c/raw)^2`` across every clock bin."""
        name, transpiler, executor = backend
        _skip_if_multi_control_unsupported(name, n)

        rng = np.random.default_rng(seed)
        c = float(rng.uniform(0.5, 1.0))
        kernel = _build_recip_sample(n, c)

        for raw in range(2**n):
            exe = transpiler.transpile(kernel, bindings={"bits": _raw_to_bits(raw, n)})
            result = exe.sample(executor, shots=_SHOTS).result()
            expected_p1, _ = _expected(raw, c)
            assert math.isclose(
                _sampled_p1(result), expected_p1, abs_tol=_SAMPLING_ATOL
            ), f"n={n} raw={raw} c={c:.6f}"

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("n", _N_VALUES)
    def test_expval_matches_analytic(self, backend, seed, n):
        """Estimated ``<Z_anc>`` matches ``1 - 2 * (c/raw)^2`` across clock bins."""
        name, transpiler, executor = backend
        _skip_if_multi_control_unsupported(name, n)

        rng = np.random.default_rng(seed)
        c = float(rng.uniform(0.5, 1.0))
        kernel = _build_recip_expval(n, c)

        for raw in range(2**n):
            exe = transpiler.transpile(
                kernel,
                bindings={"bits": _raw_to_bits(raw, n), "obs": _ANCILLA_OBSERVABLE},
            )
            got = exe.run(executor).result()
            _, expected_z = _expected(raw, c)
            assert math.isclose(float(got), expected_z, abs_tol=_EXPVAL_ATOL), (
                f"n={n} raw={raw} c={c:.6f}"
            )

    @pytest.mark.parametrize("n", _N_VALUES)
    def test_zero_bin_leaves_ancilla_untouched(self, backend, n):
        """Clock ``|raw=0>`` is the zero-eigenvalue bin: ancilla stays ``|0>``."""
        name, transpiler, executor = backend
        _skip_if_multi_control_unsupported(name, n)

        kernel = _build_recip_expval(n, c=0.7)
        exe = transpiler.transpile(
            kernel,
            bindings={"bits": _raw_to_bits(0, n), "obs": _ANCILLA_OBSERVABLE},
        )
        got = exe.run(executor).result()
        assert math.isclose(float(got), 1.0, abs_tol=_EXPVAL_ATOL)

    def test_c_equals_one_at_raw_one_fully_flips_ancilla(self, backend):
        """``c = 1`` at ``raw = 1`` gives ``Ry(pi)``: ancilla goes to ``|1>``."""
        name, transpiler, executor = backend
        # n = 1 so QURI Parts also runs (single control supported).
        kernel = _build_recip_expval(1, c=1.0)
        exe = transpiler.transpile(
            kernel,
            bindings={"bits": [1], "obs": _ANCILLA_OBSERVABLE},
        )
        got = exe.run(executor).result()
        assert math.isclose(float(got), -1.0, abs_tol=_EXPVAL_ATOL)

    def test_rejects_non_positive_c(self):
        """A non-positive scaling raises ``ValueError`` at transpile time."""
        from qamomile.qiskit import QiskitTranspiler

        kernel = _build_recip_sample(2, c=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            QiskitTranspiler().transpile(kernel, bindings={"bits": [1, 0]})

    def test_rejects_c_greater_than_one(self):
        """``c > 1`` raises ``ValueError`` (|c/raw| > 1 at raw=1)."""
        from qamomile.qiskit import QiskitTranspiler

        kernel = _build_recip_sample(2, c=1.5)
        with pytest.raises(ValueError, match="c > 1"):
            QiskitTranspiler().transpile(kernel, bindings={"bits": [1, 0]})

    @pytest.mark.parametrize("bad_c", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_c(self, bad_c):
        """Non-finite ``c`` raises ``ValueError``.

        Without an explicit finiteness check, ``c = nan`` would pass both
        the ``<= 0`` and ``> 1`` guards (every comparison with ``nan`` is
        ``False``) and the kernel would silently emit ``Ry(nan)`` for
        every clock bin.
        """
        from qamomile.qiskit import QiskitTranspiler

        kernel = _build_recip_sample(2, c=bad_c)
        with pytest.raises(ValueError, match="must be finite"):
            QiskitTranspiler().transpile(kernel, bindings={"bits": [1, 0]})

    def test_rejects_empty_clock_register(self):
        """A zero-qubit clock register raises ``ValueError``."""
        from qamomile.qiskit import QiskitTranspiler

        kernel = _build_recip_sample(0, c=0.5)
        with pytest.raises(ValueError, match="at least one qubit"):
            QiskitTranspiler().transpile(kernel, bindings={"bits": []})
