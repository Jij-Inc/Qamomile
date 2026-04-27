"""Regression: helper @qkernel that returns Vector[Qubit] from a body with
loop/if must produce a fully-emitted measure on the caller side.

Bug summary: inline pass propagated callee shape dim UUIDs only into its
local body-substitution map, not into the outer value_map. Outer Values
(e.g., ``MeasureVectorOperation.result``) whose shape inherited the inner
kernel's symbolic dim UUID at frontend tracing time were left
unsubstituted; resource allocator silently failed to allocate clbits,
dropping the measure entirely (returning ``[(None, shots)]``).

Before the fix, the kernel below emitted 0 clbits and 0 measure ops.
After the fix it emits the expected 3 clbits and 3 measures.
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc

qiskit = pytest.importorskip("qiskit")
from qamomile.qiskit import QiskitTranspiler  # noqa: E402


def test_helper_with_loop_returning_vector_emits_measure():
    """Helper with a for/if body returning Vector[Qubit] — outer measure must fire."""

    @qmc.qkernel
    def helper(data: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        for j in qmc.range(3):
            if j == 0:
                data[j] = qmc.x(data[j])
        return data

    @qmc.qkernel
    def runner() -> qmc.Vector[qmc.Bit]:
        data = qmc.qubit_array(3, name="data")
        data = helper(data)
        return qmc.measure(data)

    tp = QiskitTranspiler()
    exe = tp.transpile(runner)
    qc = exe.compiled_quantum[0].circuit
    assert qc.num_clbits == 3, (
        f"Expected 3 clbits for measuring 3-qubit Vector, got {qc.num_clbits}. "
        f"Pre-fix bug: clbits=0 because callee shape dim wasn't propagated to "
        f"outer value_map."
    )
    measure_count = sum(1 for inst in qc.data if inst.operation.name == "measure")
    assert measure_count == 3

    result = exe.sample(tp.executor(), shots=8).result()
    # Helper applies X to data[0] only. Big-endian-ish tuple; deterministic.
    assert result.results == [((1, 0, 0), 8)]


def test_helper_with_compile_time_param_returning_vector():
    """Mirrors the Steane tutorial's ``inject_single_pauli`` helper pattern.

    This was the original failing scenario that triggered the investigation.
    """
    X_ERROR = 1
    Y_ERROR = 2
    Z_ERROR = 3

    @qmc.qkernel
    def inject(
        data: qmc.Vector[qmc.Qubit],
        error_type: qmc.UInt,
        error_pos: qmc.UInt,
    ) -> qmc.Vector[qmc.Qubit]:
        for j in qmc.range(7):
            if (error_type == X_ERROR) & (error_pos == j):
                data[j] = qmc.x(data[j])
            if (error_type == Y_ERROR) & (error_pos == j):
                data[j] = qmc.y(data[j])
            if (error_type == Z_ERROR) & (error_pos == j):
                data[j] = qmc.z(data[j])
        return data

    @qmc.qkernel
    def runner(error_type: qmc.UInt, error_pos: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        data = qmc.qubit_array(7, name="data")
        data = inject(data, error_type, error_pos)
        return qmc.measure(data)

    tp = QiskitTranspiler()
    exe = tp.transpile(runner, bindings={"error_type": X_ERROR, "error_pos": 3})
    qc = exe.compiled_quantum[0].circuit
    assert qc.num_clbits == 7

    result = exe.sample(tp.executor(), shots=8).result()
    # X applied to data[3] only, others remain |0⟩.
    assert result.results == [((0, 0, 0, 1, 0, 0, 0), 8)]


def test_nested_helper_with_loop_returning_vector():
    """Two levels of helpers, each carrying the Vector through a loop."""

    @qmc.qkernel
    def inner(data: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        for j in qmc.range(2):
            if j == 0:
                data[j] = qmc.x(data[j])
        return data

    @qmc.qkernel
    def outer(data: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        data = inner(data)
        for j in qmc.range(2):
            if j == 1:
                data[j] = qmc.x(data[j])
        return data

    @qmc.qkernel
    def runner() -> qmc.Vector[qmc.Bit]:
        data = qmc.qubit_array(2, name="data")
        data = outer(data)
        return qmc.measure(data)

    tp = QiskitTranspiler()
    exe = tp.transpile(runner)
    qc = exe.compiled_quantum[0].circuit
    assert qc.num_clbits == 2

    result = exe.sample(tp.executor(), shots=8).result()
    # Inner X on data[0], outer X on data[1] → both ones.
    assert result.results == [((1, 1), 8)]
