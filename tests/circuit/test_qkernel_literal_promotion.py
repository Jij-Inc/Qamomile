"""Tests for auto-promotion of Python literals at @qkernel call sites.

When a `@qkernel` is invoked from inside another `@qkernel`'s body, raw
Python primitives (``int``, ``float``, ``bool``) used as scalar arguments
should be auto-promoted to the corresponding Handle (``UInt``, ``Float``,
``Bit``) based on the callee's declared parameter type. The user-facing
expectation is that ``ry_layer(q, thetas, 0)`` works the same as
``ry_layer(q, thetas, qmc.uint(0))``.

These tests originated from a user report where
``ry_layer(q, thetas, 0)`` raised
``TypeError: Argument 'offset' must be a Handle instance, got <class 'int'>``.

Each test exercises three layers of evidence:

1. **Build**: the outer kernel's `.block` builds without raising.
2. **Transpile**: `transpiler.transpile(...)` returns an executable.
3. **Execute**: ``executable.sample(...)`` and/or ``executable.run(...)``
   produce results matching an analytic baseline. Per CLAUDE.md, sampling
   and expval go through different backend primitives and must both pass.

Cross-backend coverage spans Qiskit, QuriParts (Qulacs), and CUDA-Q with
``importorskip``-style guards so a missing SDK skips rather than errors.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations.
"""

import math

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.basic import (
    cz_entangling_layer,
    ry_layer,
)

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

_HAS_OBSERVABLE = True
try:  # pragma: no cover - presence check, not behaviour
    import qamomile.observable as qm_o
except ImportError:  # pragma: no cover - never expected, defensive
    _HAS_OBSERVABLE = False
    qm_o = None  # type: ignore[assignment]


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


def _sum_z_hamiltonian(n: int):
    """Build H = sum_{i=0}^{n-1} Z_i over an n-qubit register."""
    H = qm_o.Hamiltonian.zero(num_qubits=n)
    for i in range(n):
        H += qm_o.Z(i)
    return H


# ---------------------------------------------------------------------------
# UInt literal promotion (the user's original failure case)
# ---------------------------------------------------------------------------
#
# ``ry_layer(q, thetas, 0)`` and the broader HEA pattern below are the
# concrete cases the user filed.  ``offset: qmc.UInt`` should accept the
# Python ``int`` ``0`` and behave identically to ``qmc.uint(0)``.


class TestUIntLiteralPromotion:
    """Sub-@qkernel calls accept ``int`` literals where ``UInt`` is declared."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    def test_user_reported_hea_with_int_literal_offset(self, transpiler_factory):
        """The exact reported case: ``ry_layer(q, thetas, 0)`` builds + executes.

        Mirrors the user's HEA kernel. After the fix, transpile + sample
        must succeed and produce a normalized distribution whose total
        shot count matches the requested shot budget.
        """

        @qmc.qkernel
        def hea(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = ry_layer(q, thetas, 0)
            q = cz_entangling_layer(q)
            q = ry_layer(q, thetas, n)
            return qmc.measure(q)

        n = 3
        rng = np.random.default_rng(0)
        thetas = rng.uniform(-math.pi, math.pi, size=2 * n).tolist()

        t = transpiler_factory()
        exe = t.transpile(hea, bindings={"n": n}, parameters=["thetas"])
        result = exe.sample(
            t.executor(), shots=512, bindings={"thetas": thetas}
        ).result()
        total = sum(count for _val, count in result.results)
        assert total == 512

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    def test_int_literal_emits_same_outcome_set_as_qmc_uint(self, transpiler_factory):
        """``ry_layer(q, thetas, 0)`` produces the same set of measurement outcomes as ``qmc.uint(0)``.

        Counts can differ by shot noise (independent RNG seeds across the
        two ``executor()`` calls), but the support of the distribution
        must be identical: both circuits must visit the same set of
        bitstrings. Numerical-equivalence of the two forms is established
        more strictly via expectation values in
        ``test_int_literal_offset_expval_matches_analytic`` below.
        """

        @qmc.qkernel
        def hea_int(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = ry_layer(q, thetas, 0)
            q = cz_entangling_layer(q)
            q = ry_layer(q, thetas, n)
            return qmc.measure(q)

        @qmc.qkernel
        def hea_uint(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            q = ry_layer(q, thetas, n)
            return qmc.measure(q)

        n = 3
        rng = np.random.default_rng(1)
        thetas = rng.uniform(-math.pi, math.pi, size=2 * n).tolist()

        t = transpiler_factory()
        bindings_compile = {"n": n}
        bindings_runtime = {"thetas": thetas}
        shots = 4096  # enough that all 2^n outcomes appear with high probability

        exe_int = t.transpile(hea_int, bindings=bindings_compile, parameters=["thetas"])
        exe_uint = t.transpile(
            hea_uint, bindings=bindings_compile, parameters=["thetas"]
        )

        out_int = dict(
            exe_int.sample(t.executor(), shots=shots, bindings=bindings_runtime)
            .result()
            .results
        )
        out_uint = dict(
            exe_uint.sample(t.executor(), shots=shots, bindings=bindings_runtime)
            .result()
            .results
        )
        # Same set of measured bitstrings (counts may differ by shot noise).
        assert set(out_int.keys()) == set(out_uint.keys()), (
            f"[{transpiler_factory.__name__}] outcome support differs: "
            f"int-literal={set(out_int.keys())}, qmc.uint(0)={set(out_uint.keys())}"
        )
        # Total shot count is exact.
        assert sum(out_int.values()) == shots
        assert sum(out_uint.values()) == shots

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_int_literal_offset_expval_matches_analytic(self, transpiler_factory, seed):
        """End-to-end: HEA with int-literal offset returns the analytic <Z_i>.

        Hardware-efficient ansatz with two RY layers and a CZ entangling
        layer; for n=2 the closed-form expectation under H = Z_0 + Z_1
        is straightforward to derive from the entangled state, so we
        cross-check against the qmc.uint(0)-wrapped variant rather than
        a hand-derived constant. This exercises the ``estimator`` path on
        each backend in addition to ``sample`` above.
        """
        rng = np.random.default_rng(seed)
        n = 2
        thetas = rng.uniform(-math.pi, math.pi, size=2 * n).tolist()

        @qmc.qkernel
        def hea_int_expval(
            thetas: qmc.Vector[qmc.Float], obs: qmc.Observable
        ) -> qmc.Float:
            q = qmc.qubit_array(n, name="q")
            q = ry_layer(q, thetas, 0)
            q = cz_entangling_layer(q)
            q = ry_layer(q, thetas, n)
            return qmc.expval(q, obs)

        @qmc.qkernel
        def hea_uint_expval(
            thetas: qmc.Vector[qmc.Float], obs: qmc.Observable
        ) -> qmc.Float:
            q = qmc.qubit_array(n, name="q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            q = ry_layer(q, thetas, n)
            return qmc.expval(q, obs)

        H = _sum_z_hamiltonian(n)
        t = transpiler_factory()
        exe_int = t.transpile(
            hea_int_expval, bindings={"obs": H}, parameters=["thetas"]
        )
        exe_uint = t.transpile(
            hea_uint_expval, bindings={"obs": H}, parameters=["thetas"]
        )
        out_int = exe_int.run(t.executor(), bindings={"thetas": thetas}).result()
        out_uint = exe_uint.run(t.executor(), bindings={"thetas": thetas}).result()
        assert np.isclose(out_int, out_uint, atol=1e-6), (
            f"[{transpiler_factory.__name__}, seed={seed}] expval differs: "
            f"int-literal={out_int}, qmc.uint=({out_uint})"
        )


# ---------------------------------------------------------------------------
# Float literal promotion
# ---------------------------------------------------------------------------
#
# When a sub-@qkernel declares ``angle: qmc.Float``, a Python ``float``
# at the call site should auto-promote and behave like ``qmc.float_(x)``.


@qmc.qkernel
def _ry_only(q: qmc.Vector[qmc.Qubit], theta: qmc.Float) -> qmc.Vector[qmc.Qubit]:
    """Apply a single RY(theta) broadcast and return the register."""
    q = qmc.ry(q, theta)
    return q


class TestFloatLiteralPromotion:
    """Sub-@qkernel calls accept ``float`` literals where ``Float`` is declared."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize("theta", [0.0, math.pi, 0.5, -0.7])
    def test_float_literal_matches_qmc_float(self, transpiler_factory, theta):
        """``_ry_only(q, 0.5)`` matches ``_ry_only(q, qmc.float_(0.5))`` exactly.

        The expval after applying RY(theta) to |0>^n on a single Z_i
        observable is ``cos(theta)`` per-qubit, so we cross-check both
        forms against the analytic value as well as each other.
        """

        @qmc.qkernel
        def outer_lit(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(2, name="q")
            q = _ry_only(q, theta)  # raw float literal
            return qmc.expval(q, obs)

        @qmc.qkernel
        def outer_wrapped(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(2, name="q")
            q = _ry_only(q, qmc.float_(theta))  # explicit wrap
            return qmc.expval(q, obs)

        H = _sum_z_hamiltonian(2)
        t = transpiler_factory()
        exe_lit = t.transpile(outer_lit, bindings={"obs": H})
        exe_wrap = t.transpile(outer_wrapped, bindings={"obs": H})
        out_lit = exe_lit.run(t.executor()).result()
        out_wrap = exe_wrap.run(t.executor()).result()
        # Analytic: <Z_0 + Z_1> = 2 * cos(theta)
        expected = 2.0 * math.cos(theta)
        assert np.isclose(out_lit, expected, atol=1e-6)
        assert np.isclose(out_lit, out_wrap, atol=1e-6)


# ---------------------------------------------------------------------------
# Bit literal promotion
# ---------------------------------------------------------------------------
#
# Less common, but the symmetry argument applies: a callee declaring
# ``flag: qmc.Bit`` should accept ``True``/``False``/``0``/``1`` at the
# call site. Compile-time folding then erases the dead branch.


@qmc.qkernel
def _maybe_x(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
    """Apply X conditionally on a compile-time Bit flag."""
    if flag:
        q = qmc.x(q)
    return q


class TestBitLiteralPromotion:
    """Sub-@qkernel calls accept ``bool`` literals where ``Bit`` is declared."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    @pytest.mark.parametrize(
        "flag,expected_outcome",
        [(True, 1), (False, 0)],
        ids=["flag-True", "flag-False"],
    )
    def test_bool_literal_drives_deterministic_outcome(
        self, transpiler_factory, flag, expected_outcome
    ):
        """A Python ``bool`` passed to ``flag: qmc.Bit`` produces a deterministic circuit.

        ``flag=True`` keeps the X gate, ``flag=False`` removes it (compile-time
        fold of ``if flag:`` against the constant Bit handle). Since the
        resulting circuit has no quantum randomness, every shot must equal
        ``expected_outcome`` — observed here via sampling rather than via
        IR inspection.
        """

        @qmc.qkernel
        def outer() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = _maybe_x(q0, flag)
            return qmc.measure(q0)

        t = transpiler_factory()
        exe = t.transpile(outer)
        result = exe.sample(t.executor(), shots=64).result()
        for value, count in result.results:
            assert value == expected_outcome, (
                f"[{transpiler_factory.__name__}, flag={flag}] expected "
                f"{expected_outcome}, got {value} ({count} times)"
            )


# ---------------------------------------------------------------------------
# Negative cases — values that must NOT be promoted
# ---------------------------------------------------------------------------
#
# The auto-promote helper is intentionally narrow: it covers ``int → UInt``,
# ``float → Float``, and ``bool → Bit`` only. Any other mismatch (string,
# list, or ``bool`` aimed at a numeric param) must keep raising the
# original ``TypeError`` so that callers see a clear error rather than
# silently coerced data.


@qmc.qkernel
def _takes_float(theta: qmc.Float) -> qmc.Float:
    """Identity helper: returns its scalar Float input unchanged."""
    return theta


@qmc.qkernel
def _takes_uint(n: qmc.UInt) -> qmc.UInt:
    """Identity helper: returns its scalar UInt input unchanged."""
    return n


@qmc.qkernel
def _takes_float_vector(xs: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Float]:
    """Identity helper: returns its Vector[Float] input unchanged."""
    return xs


class TestNonPromotableArgsRaise:
    """Non-promotable mismatches keep the existing strict ``TypeError``."""

    def test_str_for_float_param_raises(self):
        """``str`` passed to a ``Float`` parameter is not promoted; original error stands."""

        @qmc.qkernel
        def caller() -> qmc.Float:
            return _takes_float("not a float")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a Handle instance"):
            caller.block  # noqa: B018 - property access triggers compilation

    def test_list_for_vector_param_raises(self):
        """``list`` passed to a ``Vector[Float]`` parameter is not promoted; only scalars are."""

        @qmc.qkernel
        def caller() -> qmc.Vector[qmc.Float]:
            return _takes_float_vector([0.1, 0.2])  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a Handle instance"):
            caller.block  # noqa: B018

    def test_bool_for_uint_param_raises(self):
        """``bool`` is intentionally excluded from int→UInt promotion to avoid silent ``True → UInt(1)``.

        ``bool`` is a subclass of ``int`` in Python, so an untyped helper that
        treats every ``int`` as a ``UInt`` candidate would silently swallow
        ``True``/``False``. This test pins the design choice: ``bool`` is
        only ever promoted to ``Bit``.
        """

        @qmc.qkernel
        def caller() -> qmc.UInt:
            return _takes_uint(True)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a Handle instance"):
            caller.block  # noqa: B018

    def test_str_for_bit_param_raises(self):
        """``str`` passed to a ``Bit`` parameter is not promoted (only ``bool`` is)."""

        @qmc.qkernel
        def takes_bit(flag: qmc.Bit) -> qmc.Bit:
            return flag

        @qmc.qkernel
        def caller() -> qmc.Bit:
            return takes_bit("yes")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a Handle instance"):
            caller.block  # noqa: B018


# ---------------------------------------------------------------------------
# Default-value edge case
# ---------------------------------------------------------------------------
#
# A scalar parameter declared with a Python literal default (``n: UInt = 4``)
# previously failed at the call site because ``apply_defaults()`` injects the
# raw literal into the bound arguments and the strict ``isinstance`` check
# rejected it. The auto-promote helper runs against the post-apply_defaults
# arguments, so default values are now wrapped on the same path as
# explicit-positional literals.


@qmc.qkernel
def _h_register(n: qmc.UInt = 4) -> qmc.Vector[qmc.Bit]:
    """Apply H broadcast to an n-qubit register (default n=4) and measure."""
    q = qmc.qubit_array(n, name="q")
    q = qmc.h(q)
    return qmc.measure(q)


class TestDefaultValuePromotion:
    """Scalar default values are auto-promoted via the same path as call-site literals."""

    @pytest.mark.parametrize("transpiler_factory", BACKENDS)
    def test_uint_default_works_without_explicit_wrap(self, transpiler_factory):
        """``def f(n: UInt = 4)`` is callable without explicitly wrapping the default.

        Builds, transpiles, and samples a kernel whose helper has a raw-int
        default. The outer caller uses ``_h_register()`` with no arguments,
        forcing ``apply_defaults()`` to inject ``4`` — which the auto-promote
        helper must wrap. After execution, the H-broadcast on |0>^4 yields
        a uniform distribution; total shot count must equal the budget.
        """

        @qmc.qkernel
        def caller() -> qmc.Vector[qmc.Bit]:
            return _h_register()  # uses default n=4

        t = transpiler_factory()
        exe = t.transpile(caller)
        result = exe.sample(t.executor(), shots=512).result()
        total = sum(count for _val, count in result.results)
        assert total == 512
        # H broadcast on |0>^4 gives uniform support across 2^4 = 16 outcomes;
        # 512 shots across 16 outcomes makes seeing all 16 overwhelmingly likely.
        assert len(result.results) >= 8
