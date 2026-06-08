"""Tests for the unary-minus operator on the ``qamomile.circuit.Float`` handle.

``Float.__neg__`` lets users write the natural ``-x`` inside a ``@qkernel``
instead of the awkward ``0 - x`` idiom (GitHub issue #329). Negation is
lowered to the existing ``MUL`` IR op as ``self * -1.0``, so it adds no new
IR node and rides on every backend's existing multiplication support. When
the operand is a compile-time-bound Float, ``partial_eval`` folds the
``MUL`` into the literal negated constant baked into the emitted circuit.

These tests exercise three layers of evidence:

1. **Build / transpile**: a kernel using ``-theta`` compiles on each backend.
2. **Execute**: both ``sample`` and ``run`` (expval) paths produce results
   matching an analytic baseline. Per CLAUDE.md, sampling and expval go
   through different backend primitives and must both pass.
3. **Equivalence**: ``-theta`` is numerically identical to ``0.0 - theta``.

Note on scope: arithmetic on a *runtime* parameter feeding a gate angle
(e.g. ``rx(q, -theta)`` with ``theta`` left symbolic) is a pre-existing
unsupported path — plain ``theta - 0.5`` collapses to ``Rx(0)`` the same
way. ``__neg__`` is consistent with that existing behaviour, so these
tests bind ``theta`` at compile time, which is the supported path.

Cross-backend coverage spans Qiskit, QuriParts (Qulacs), and CUDA-Q with
``skipif`` guards so a missing SDK skips rather than errors.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations.
"""

import math

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o

# ---------------------------------------------------------------------------
# Backend matrix
# ---------------------------------------------------------------------------

_HAS_QISKIT = True
try:  # pragma: no cover - presence check, not behaviour
    from qamomile.qiskit import QiskitTranspiler
except ImportError:  # pragma: no cover - covered when qiskit is absent
    _HAS_QISKIT = False
    QiskitTranspiler = None  # type: ignore[assignment]

_HAS_QURI_PARTS = True
try:  # pragma: no cover - presence check, not behaviour
    import quri_parts.qulacs  # noqa: F401

    from qamomile.quri_parts import QuriPartsTranspiler
except ImportError:  # pragma: no cover - covered when quri_parts is absent
    _HAS_QURI_PARTS = False
    QuriPartsTranspiler = None  # type: ignore[assignment]

_HAS_CUDAQ = True
try:  # pragma: no cover - presence check, not behaviour
    import cudaq  # noqa: F401

    from qamomile.cudaq import CudaqTranspiler
except ImportError:  # pragma: no cover - covered when cudaq is absent
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
            not _HAS_QURI_PARTS, reason="quri_parts/qulacs not installed"
        ),
    ),
    pytest.param(
        CudaqTranspiler,
        id="cudaq",
        marks=pytest.mark.skipif(not _HAS_CUDAQ, reason="cudaq not installed"),
    ),
]

# Boundary angles exercised alongside random ones drawn per-test by the RNG.
_BOUNDARY_ANGLES = [0.0, math.pi, 2.0 * math.pi, -0.7]


def _manual_cry_half_angle(
    ctrl: qmc.Qubit, tgt: qmc.Qubit, angle: float
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Decompose CRY with the same symbolic half-angle form users write.

    Args:
        ctrl (qmc.Qubit): Control qubit consumed by the decomposition.
        tgt (qmc.Qubit): Target qubit consumed by the decomposition.
        angle (float): Rotation angle. Tests intentionally pass a
            ``qmc.Float`` value through this annotation to mirror user
            helper code written with a Python ``float`` annotation.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Updated control and target qubits.
    """
    tgt = qmc.ry(tgt, angle / 2)
    ctrl, tgt = qmc.cx(ctrl, tgt)
    tgt = qmc.ry(tgt, -angle / 2)
    ctrl, tgt = qmc.cx(ctrl, tgt)
    return ctrl, tgt


def test_float_handle_exposes_neg():
    """``qamomile.circuit.Float`` implements ``__neg__`` (issue #329 fix)."""
    assert hasattr(qmc.Float, "__neg__")


def test_vector_float_element_helper_supports_negated_half_angle():
    """A ``Vector[qmc.Float]`` element can flow through ``-angle / 2``."""

    @qmc.qkernel
    def circuit(slopes_p: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, name="qs")
        qs[0], qs[1] = _manual_cry_half_angle(qs[0], qs[1], slopes_p[0])
        return qmc.measure(qs)

    block = circuit.build(slopes_p=np.array([0.3]))

    assert block.operations


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
def test_vector_float_element_helper_transpiles_with_bindings():
    """Qiskit transpilation accepts the user-reported ``-angle / 2`` pattern."""

    @qmc.qkernel
    def circuit(slopes_p: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, name="qs")
        qs[0], qs[1] = _manual_cry_half_angle(qs[0], qs[1], slopes_p[0])
        return qmc.measure(qs)

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(circuit, bindings={"slopes_p": np.array([0.3])})
    result = exe.sample(transpiler.executor(), shots=16).result()

    assert exe.quantum_circuit.num_qubits == 2
    assert sum(count for _value, count in result.results) == 16


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
def test_vector_float_slice_element_helper_transpiles_with_bindings():
    """A sliced ``Vector[qmc.Float]`` element resolves through ``-angle / 2``."""

    @qmc.qkernel
    def circuit(slopes_p: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(1, name="qs")
        view = slopes_p[1:3]
        qs[0] = qmc.ry(qs[0], -view[0] / 2)
        return qmc.measure(qs)

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        circuit, bindings={"slopes_p": np.array([0.1, 0.3, 0.5])}
    )
    ry_angles = [
        float(instruction.operation.params[0])
        for instruction in exe.quantum_circuit.data
        if instruction.operation.name == "ry"
    ]

    assert ry_angles == pytest.approx([-0.15])


class TestVectorFloatSliceElementRegression:
    """Sliced bound ``Vector[qmc.Float]`` elements compile and execute."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    def test_sliced_vector_float_element_expval_runs(self, transpiler_factory):
        """A main-regression slice element emits the expected rotation angle.

        The user-facing pattern is ``view = slopes_p[1:3]`` followed by
        ``-view[0] / 2``. On main, the sliced element does not resolve to
        the root bound value before transpilation; this test pins the
        fixed path by executing the emitted circuit on every available
        SDK backend.
        """

        @qmc.qkernel
        def circuit(slopes_p: qmc.Vector[qmc.Float], obs: qmc.Observable) -> qmc.Float:
            qs = qmc.qubit_array(1, name="qs")
            view = slopes_p[1:3]
            qs[0] = qmc.ry(qs[0], -view[0] / 2)
            return qmc.expval(qs, obs)

        slopes = np.array([0.1, 0.3, 0.5])
        expected = math.cos(-0.15)
        obs = qm_o.Hamiltonian.zero(num_qubits=1) + qm_o.Z(0)
        transpiler = transpiler_factory()
        exe = transpiler.transpile(circuit, bindings={"slopes_p": slopes, "obs": obs})
        got = exe.run(transpiler.executor()).result()

        assert np.isclose(got, expected, atol=1e-6), (
            f"[{transpiler_factory.__name__}] expected <Z>={expected}, got {got}"
        )

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    def test_sliced_vector_float_element_sampling_runs(self, transpiler_factory):
        """The same slice-element regression deterministically samples all-zero."""

        @qmc.qkernel
        def circuit(slopes_p: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(1, name="qs")
            view = slopes_p[1:3]
            qs[0] = qmc.ry(qs[0], -view[0] / 2)
            return qmc.measure(qs)

        transpiler = transpiler_factory()
        exe = transpiler.transpile(
            circuit, bindings={"slopes_p": np.array([0.1, 0.0, 0.5])}
        )
        result = exe.sample(transpiler.executor(), shots=64).result()

        assert sum(count for _value, count in result.results) == 64
        assert result.results == [((0,), 64)]


class TestNegCancelsRotation:
    """``ry(theta)`` followed by ``ry(-theta)`` is the identity."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_cancellation_expval(self, transpiler_factory, seed):
        """``RY(theta)`` then ``RY(-theta)`` returns the register to |0>, so <Z> == 1.

        If ``__neg__`` were wrong (e.g. a no-op leaving ``+theta``), the
        circuit would be ``RY(2*theta)`` and <Z> would equal
        ``cos(2*theta)`` instead of 1, so this pins down the negation.
        Boundary angles (0, pi, 2*pi) are checked alongside random ones.
        """
        rng = np.random.default_rng(seed)
        angles = _BOUNDARY_ANGLES + rng.uniform(-2 * math.pi, 2 * math.pi, 3).tolist()

        @qmc.qkernel
        def cancel(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(1, name="q")
            q = qmc.ry(q, theta)
            q = qmc.ry(q, -theta)
            return qmc.expval(q, obs)

        H = qm_o.Hamiltonian.zero(num_qubits=1) + qm_o.Z(0)
        t = transpiler_factory()
        for theta in angles:
            exe = t.transpile(cancel, bindings={"theta": theta, "obs": H})
            out = exe.run(t.executor()).result()
            assert np.isclose(out, 1.0, atol=1e-6), (
                f"[{transpiler_factory.__name__}, seed={seed}, theta={theta}] "
                f"expected <Z>=1.0 after RY(theta)+RY(-theta), got {out}"
            )

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("seed", [0, 7])
    def test_cancellation_sampling(self, transpiler_factory, seed):
        """Sampling the cancelling circuit yields outcome 0 on every shot.

        Exercises the sampler primitive (distinct from the estimator path
        covered above). Since ``RY(theta)+RY(-theta)`` restores |0>, the
        circuit has no quantum randomness and every shot must be 0.
        """
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-2 * math.pi, 2 * math.pi))

        @qmc.qkernel
        def cancel_sample(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            q = qmc.ry(q, -theta)
            return qmc.measure(q)

        t = transpiler_factory()
        exe = t.transpile(cancel_sample, bindings={"theta": theta})
        result = exe.sample(t.executor(), shots=256).result()
        total = sum(count for _val, count in result.results)
        assert total == 256
        for value, count in result.results:
            assert value == 0, (
                f"[{transpiler_factory.__name__}, seed={seed}, theta={theta}] "
                f"expected only outcome 0, got {value} ({count} times)"
            )


class TestNegMatchesZeroMinusX:
    """``-theta`` is numerically identical to the old ``0.0 - theta`` idiom."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_neg_equals_zero_minus_x_and_analytic(self, transpiler_factory, seed):
        """``rx(q, -theta)`` matches ``rx(q, 0.0 - theta)`` and the analytic <Y>.

        For ``RX(a)|0>`` the expectation ``<Y> = -sin(a)``; with
        ``a = -theta`` this is ``sin(theta)``. The sign of the rotation
        is observable through the ``Y`` observable (a ``Z``-only check
        would be sign-blind), so this confirms ``__neg__`` truly negates
        rather than merely leaving the angle unchanged.
        """
        rng = np.random.default_rng(seed)
        angles = _BOUNDARY_ANGLES + rng.uniform(-2 * math.pi, 2 * math.pi, 3).tolist()

        @qmc.qkernel
        def neg_kernel(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(1, name="q")
            q = qmc.rx(q, -theta)
            return qmc.expval(q, obs)

        @qmc.qkernel
        def sub_kernel(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(1, name="q")
            q = qmc.rx(q, 0.0 - theta)
            return qmc.expval(q, obs)

        H = qm_o.Hamiltonian.zero(num_qubits=1) + qm_o.Y(0)
        t = transpiler_factory()
        for theta in angles:
            exe_neg = t.transpile(neg_kernel, bindings={"theta": theta, "obs": H})
            exe_sub = t.transpile(sub_kernel, bindings={"theta": theta, "obs": H})
            out_neg = exe_neg.run(t.executor()).result()
            out_sub = exe_sub.run(t.executor()).result()
            expected = math.sin(theta)
            assert np.isclose(out_neg, expected, atol=1e-6), (
                f"[{transpiler_factory.__name__}, seed={seed}, theta={theta}] "
                f"-theta gave <Y>={out_neg}, expected {expected}"
            )
            assert np.isclose(out_neg, out_sub, atol=1e-8), (
                f"[{transpiler_factory.__name__}, seed={seed}, theta={theta}] "
                f"-theta ({out_neg}) differs from 0.0-theta ({out_sub})"
            )


class TestNegConstantFold:
    """A negated compile-time-bound Float is folded into the emitted circuit."""

    @pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("theta", [0.0, 0.5, math.pi, -1.3])
    def test_bound_neg_folds_to_constant_param(self, theta):
        """``-theta`` with ``theta`` bound emits a single ``rz`` with param ``-theta``.

        When both operands of the underlying ``MUL`` are constants
        (``-1.0`` and the bound ``theta``), ``partial_eval`` folds the op,
        so the emitted Qiskit circuit carries the literal ``-theta`` as
        the rotation parameter rather than a symbolic expression.
        """

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, -theta)
            return qmc.measure(q)

        exe = QiskitTranspiler().transpile(circuit, bindings={"theta": theta})
        rz_params = [
            instr.operation.params[0]
            for instr in exe.quantum_circuit.data
            if instr.operation.name == "rz"
        ]
        assert len(rz_params) == 1
        assert abs(float(rz_params[0]) - (-theta)) < 1e-10
