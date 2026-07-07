"""Regression tests for call-site-unique sub-kernel result Values (issue #563).

``Block.call`` used to return the callee block's cached output Value
instance for non-pass-through outputs, so every call site of the same
sub-kernel shared one result UUID. With a Bit-returning sub-kernel both
measurements collapsed onto a single clbit and always agreed.
"""

from __future__ import annotations

from typing import Any

import qamomile.circuit as qmc
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation


@qmc.qkernel
def measure_one(q: qmc.Qubit) -> qmc.Bit:
    """Measure a single qubit inside a sub-kernel."""
    return qmc.measure(q)


@qmc.qkernel
def double_call() -> tuple[qmc.Bit, qmc.Bit]:
    """Call the same Bit-returning sub-kernel at two call sites."""
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q1 = qmc.x(q1)
    b0 = measure_one(q0)
    b1 = measure_one(q1)
    return b0, b1


@qmc.qkernel
def measure_pair(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Bit]:
    """Measure a two-qubit register inside a sub-kernel."""
    return qmc.measure(qs)


@qmc.qkernel
def double_call_vector() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Call the same Vector[Bit]-returning sub-kernel at two call sites."""
    a = qubit_array(2, "a")
    b = qubit_array(2, "b")
    b[0] = qmc.x(b[0])
    r0 = measure_pair(a)
    r1 = measure_pair(b)
    return r0, r1


def _call_ops(kernel: QKernel[Any, Any]) -> list[CallBlockOperation]:
    """Collect top-level CallBlockOperations from a kernel's traced block.

    Args:
        kernel (QKernel[Any, Any]): A qkernel whose built block is inspected.

    Returns:
        list[CallBlockOperation]: Call operations in program order.
    """
    block = kernel.build()
    return [op for op in block.operations if isinstance(op, CallBlockOperation)]


class TestSubkernelCallSiteResults:
    """Each call site of a sub-kernel owns fresh result Value identities."""

    def test_subkernel_double_call_distinct_results(self):
        """Two calls to one Bit-returning sub-kernel get distinct result Values."""
        call_ops = _call_ops(double_call)
        assert len(call_ops) == 2
        first, second = (op.results[0] for op in call_ops)
        assert first.uuid != second.uuid
        assert first.logical_id != second.logical_id

    def test_subkernel_double_call_vector_distinct_results(self):
        """Two calls returning Vector[Bit] get distinct results with shape intact."""
        call_ops = _call_ops(double_call_vector)
        assert len(call_ops) == 2
        first, second = (op.results[0] for op in call_ops)
        assert first.uuid != second.uuid
        assert first.logical_id != second.logical_id
        assert first.shape and second.shape

    def test_call_results_do_not_alias_callee_outputs(self):
        """Call results never reuse the callee block's cached output instances."""
        call_ops = _call_ops(double_call)
        callee_output_uuids = {
            v.uuid for op in call_ops for v in op.block.output_values
        }
        for op in call_ops:
            assert op.results[0].uuid not in callee_output_uuids

    def test_subkernel_double_call_two_clbits(self, qiskit_transpiler):
        """The transpiled circuit allocates one clbit per call site."""
        qc = qiskit_transpiler.to_circuit(double_call)
        assert qc.num_clbits == 2, f"Expected 2 clbits, got {qc.num_clbits}"

    def test_subkernel_double_call_independent_measurements(self, sdk_transpiler):
        """X applied to one qubit flips only that call site's measurement."""
        transpiler = sdk_transpiler.transpiler
        executable = transpiler.transpile(double_call)
        result = executable.sample(transpiler.executor(), shots=128).result()

        assert result.results == [((False, True), 128)]

    def test_subkernel_double_call_vector_independent_measurements(
        self, sdk_transpiler
    ):
        """Register-returning sub-kernel calls measure their own registers."""
        transpiler = sdk_transpiler.transpiler
        executable = transpiler.transpile(double_call_vector)
        result = executable.sample(transpiler.executor(), shots=128).result()

        assert len(result.results) == 1
        (outputs, count) = result.results[0]
        r0, r1 = outputs
        assert count == 128
        assert tuple(bool(bit) for bit in r0) == (False, False)
        assert tuple(bool(bit) for bit in r1) == (True, False)
