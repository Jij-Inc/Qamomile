"""Regression: ``Transpiler.transpile`` rejects ``parameters``/``bindings`` overlap.

A parameter name appearing in both ``bindings`` and ``parameters`` is
fundamentally ambiguous (placeholder value vs runtime symbol) and used to
silently miscompile control-flow predicates that depended on the parameter
array's elements (Issue #354 B-series). After Phase 1, the overlap raises
``ValueError`` at the public API entry, eliminating the silent miscompilation
class entirely.
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc


@pytest.fixture
def qiskit_transpiler():
    """Return a QiskitTranspiler, skipping the test if qiskit is unavailable."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


@qmc.qkernel
def _identity_kernel(theta: qmc.Float) -> qmc.Bit:
    """Trivial kernel used to exercise the disjointness check.

    Returns:
        Measurement of a Hadamard-prepared qubit.
    """
    q = qmc.qubit(name="q")
    q = qmc.rx(q, theta)
    return qmc.measure(q)


class TestParamBindingsDisjoint:
    """Validates the API-level disjointness contract."""

    def test_overlap_single_name_rejected(self, qiskit_transpiler) -> None:
        """A single shared name between bindings and parameters raises ValueError."""
        with pytest.raises(ValueError, match=r"appear in both"):
            qiskit_transpiler.transpile(
                _identity_kernel,
                bindings={"theta": 0.5},
                parameters=["theta"],
            )

    def test_overlap_multiple_names_listed_in_message(self, qiskit_transpiler) -> None:
        """Multiple shared names are surfaced in the error message for debuggability."""

        @qmc.qkernel
        def two_param(a: qmc.Float, b: qmc.Float) -> qmc.Bit:
            """Two-parameter kernel for the multi-name overlap test."""
            q = qmc.qubit(name="q")
            q = qmc.rx(q, a)
            q = qmc.ry(q, b)
            return qmc.measure(q)

        with pytest.raises(ValueError) as excinfo:
            qiskit_transpiler.transpile(
                two_param,
                bindings={"a": 0.1, "b": 0.2},
                parameters=["a", "b"],
            )
        message = str(excinfo.value)
        assert "'a'" in message and "'b'" in message

    def test_disjoint_bindings_and_parameters_compile(self, qiskit_transpiler) -> None:
        """The QAOA-style pattern (compile-time scalar + runtime array) compiles cleanly."""

        @qmc.qkernel
        def qaoa_like(p: qmc.UInt, gamma: qmc.Float) -> qmc.Bit:
            """Compile-time depth ``p`` plus runtime rotation angle ``gamma``."""
            q = qmc.qubit(name="q")
            for _ in qmc.range(p):
                q = qmc.rx(q, gamma)
            return qmc.measure(q)

        # No overlap: ``p`` is compile-time bound, ``gamma`` is runtime symbolic.
        exe = qiskit_transpiler.transpile(
            qaoa_like,
            bindings={"p": 2},
            parameters=["gamma"],
        )
        # Sanity-check that the runtime parameter actually flows through.
        result = exe.sample(
            qiskit_transpiler.executor(),
            shots=8,
            bindings={"gamma": 0.5},
        ).result()
        assert sum(count for _, count in result.results) == 8

    def test_only_bindings_no_overlap_check(self, qiskit_transpiler) -> None:
        """``bindings`` alone (no ``parameters``) is unaffected by the disjointness check."""
        exe = qiskit_transpiler.transpile(_identity_kernel, bindings={"theta": 0.0})
        result = exe.sample(qiskit_transpiler.executor(), shots=4).result()
        assert sum(count for _, count in result.results) == 4

    def test_only_parameters_no_overlap_check(self, qiskit_transpiler) -> None:
        """``parameters`` alone (no ``bindings``) is unaffected by the disjointness check."""
        exe = qiskit_transpiler.transpile(_identity_kernel, parameters=["theta"])
        result = exe.sample(
            qiskit_transpiler.executor(),
            shots=4,
            bindings={"theta": 0.3},
        ).result()
        assert sum(count for _, count in result.results) == 4


class TestIssue354BSeriesRepro:
    """The exact repro from Issue #354 B-series no longer silently miscompiles.

    Pre-fix: ``transpile(basis_only, bindings={'input': [0]*4}, parameters=['input'])``
    silently dropped the X gates (sample returned ``[((0,0,0,0), shots)]`` regardless
    of the runtime ``input`` binding). Post-fix: the same call raises ``ValueError``
    at the API boundary, making the silent-wrong-answer impossible.
    """

    def test_repro_now_raises_value_error(self, qiskit_transpiler) -> None:
        """The B-series pattern is rejected with a clear ValueError."""

        @qmc.qkernel
        def computational_basis_state(
            q: qmc.Vector[qmc.Qubit],
            input: qmc.Vector[qmc.UInt],
        ) -> qmc.Vector[qmc.Qubit]:
            """Apply X to qubits where the corresponding input bit is 1."""
            n = q.shape[0]
            for i in qmc.range(n):
                if input[i] == 1:
                    q[i] = qmc.x(q[i])
            return q

        @qmc.qkernel
        def basis_only(input: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
            """Wrap ``computational_basis_state`` in a measure-all entrypoint."""
            q = qmc.qubit_array(input.shape[0], name="q")
            q = computational_basis_state(q, input)
            return qmc.measure(q)

        with pytest.raises(ValueError, match=r"appear in both"):
            qiskit_transpiler.transpile(
                basis_only,
                bindings={"input": [0] * 4},
                parameters=["input"],
            )
