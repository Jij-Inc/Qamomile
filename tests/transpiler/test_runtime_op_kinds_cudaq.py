"""End-to-end coverage of ``RuntimeOpKind`` on the CUDA-Q backend.

Mirrors the Qiskit-side tests in ``tests/transpiler/test_runtime_op_kinds.py``
for the AND / OR / NOT runtime classical expressions that the frontend
constructs under measurement taint. Per H-bis, every algorithm /
backend-relevant change must be exercised end-to-end on each supported
backend; this file pins that contract for CUDA-Q's
``_emit_runtime_classical_expr`` lowering.

The Qamomile CUDA-Q emit pass lowers each ``RuntimeClassicalExpr`` to a
Python source-text expression (e.g. ``"(__b0 and __b1)"``) inside the
generated ``@cudaq.kernel`` body. The kernel then runs via
``cudaq.run`` (auto-selected by ``CudaqEmitPass`` whenever the IR
contains runtime-conditional control flow).
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc

cudaq = pytest.importorskip("cudaq")

# Gate the whole module on the CUDA-Q job: the same pattern
# ``tests/transpiler/backends/test_cudaq.py`` uses, so ``pytest -m cudaq``
# selects this file too.
pytestmark = pytest.mark.cudaq


@pytest.fixture
def cudaq_transpiler():
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


class TestCudaqRuntimeClassicalExpr:
    """RuntimeOpKind AND / OR / NOT lowering on the CUDA-Q backend.

    The kernel pattern is: prepare deterministic bit values via
    ``qmc.x``, measure, then apply a corrective X gate iff the
    runtime predicate is true. Because the input bits are deterministic
    the final measurement is also deterministic, so we can sample with
    a small shot count and assert every outcome is the expected value.
    """

    def test_and_runtime_executes(self, cudaq_transpiler):
        """``a & b`` over two measurement bits drives a conditional X."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            if a & b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = cudaq_transpiler.transpile(kernel)
        result = exe.sample(cudaq_transpiler.executor(), shots=50).result()
        assert result.results == [(1, 50)]

    def test_or_runtime_executes(self, cudaq_transpiler):
        """``a | b`` with one bit forced to 1 fires the conditional X."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            if a | b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = cudaq_transpiler.transpile(kernel)
        result = exe.sample(cudaq_transpiler.executor(), shots=50).result()
        assert result.results == [(1, 50)]

    def test_not_runtime_executes(self, cudaq_transpiler):
        """``~a`` over a 0-prepared bit fires the conditional X."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            a = qmc.measure(q0)
            if ~a:
                q1 = qmc.x(q1)
            return qmc.measure(q1)

        exe = cudaq_transpiler.transpile(kernel)
        result = exe.sample(cudaq_transpiler.executor(), shots=50).result()
        assert result.results == [(1, 50)]

    def test_nested_and_not_executes(self, cudaq_transpiler):
        """Nested predicate ``(~a) & b`` exercises sub-expression reuse."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            # q0 stays |0> -> a=0; q1 flipped -> b=1 -> (~a) & b = 1
            q1 = qmc.x(q1)
            a = qmc.measure(q0)
            b = qmc.measure(q1)
            if (~a) & b:
                q2 = qmc.x(q2)
            return qmc.measure(q2)

        exe = cudaq_transpiler.transpile(kernel)
        result = exe.sample(cudaq_transpiler.executor(), shots=50).result()
        assert result.results == [(1, 50)]
