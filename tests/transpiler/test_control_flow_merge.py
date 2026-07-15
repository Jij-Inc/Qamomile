"""Regression tests for emit-time classical merge-output binding.

The frontend's ``emit_if`` builder creates a merge slot for **every**
captured variable in an if-branch — including read-only ones. For
classical merge outputs (UInt loop indices, Float angles, Bit flags) that
end up identifying the SAME IR Value on both branches, the merge is a
no-op merge and the output should be bound to that value at emit time.

The bug this guards against: a pattern like

    for j in qmc.range(7):
        if cond:
            data[j] = qmc.x(data[j])

would fail at emit with ``symbolic_index_not_bound`` because
``emit_for_unrolled`` binds ``j`` per iteration but the if-branch
merge-versions ``j`` to ``j_merge_4``, which has no entry in bindings.
``register_classical_merge_aliases`` (called from ``emit_if``) now binds
merge outputs whose true_value and false_value are the same IR Value,
making subsequent ``data[j_merge_4]`` indexing resolvable.
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc


@pytest.fixture
def transpiler():
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


class TestForRangeIfArrayIndex:
    """The original failing pattern from QEC tutorial 11."""

    def test_for_range_if_data_index(self, transpiler):
        """Iteration over qmc.range with conditional array assignment indexed
        by the loop variable. Pre-fix this raised
        ``QubitIndexResolutionError: symbolic_index_not_bound``."""

        X_ERROR = 1
        Y_ERROR = 2
        Z_ERROR = 3

        @qmc.qkernel
        def kernel(error_type: qmc.UInt, error_pos: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            data = qmc.qubit_array(7, name="data")
            for j in qmc.range(7):
                if (error_type == X_ERROR) & (error_pos == j):
                    data[j] = qmc.x(data[j])
                if (error_type == Y_ERROR) & (error_pos == j):
                    data[j] = qmc.y(data[j])
                if (error_type == Z_ERROR) & (error_pos == j):
                    data[j] = qmc.z(data[j])
            return qmc.measure(data)

        # Each (error_type, error_pos) should produce exactly one Pauli
        # gate at position error_pos and 7 measurements.
        for et_name, et_code in [("x", X_ERROR), ("y", Y_ERROR), ("z", Z_ERROR)]:
            for pos in range(7):
                exe = transpiler.transpile(
                    kernel, bindings={"error_type": et_code, "error_pos": pos}
                )
                qc = exe.compiled_quantum[0].circuit
                gates = [i.operation.name for i in qc.data]
                assert gates.count(et_name) == 1, (
                    f"{et_name} error at q[{pos}]: expected 1 {et_name} gate, "
                    f"got {gates.count(et_name)}"
                )
                assert gates.count("measure") == 7

    def test_for_range_if_no_assignment(self, transpiler):
        """Read-only loop variable in if-branch: no failure."""

        @qmc.qkernel
        def kernel(target: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            data = qmc.qubit_array(5, name="data")
            for j in qmc.range(5):
                if target == j:
                    data[j] = qmc.h(data[j])
            return qmc.measure(data)

        for tgt in range(5):
            exe = transpiler.transpile(kernel, bindings={"target": tgt})
            qc = exe.compiled_quantum[0].circuit
            gates = [i.operation.name for i in qc.data]
            assert gates.count("h") == 1


class TestRuntimeIfReadOnlyLoopVar:
    """Runtime if (measurement-driven) inside a for-loop, with the loop
    variable read in the if-condition. The merge alias must bind so emit
    can locate the right qubit per iteration."""

    def test_runtime_if_uses_loop_var_in_array_index(self, transpiler):
        """A runtime if inside a for-loop where the loop variable is used
        as the array index in the if-body."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            data = qmc.qubit_array(3, name="data")
            ctrl = qmc.qubit("ctrl")
            ctrl = qmc.x(ctrl)
            flag = qmc.measure(ctrl)
            for j in qmc.range(3):
                if flag:
                    data[j] = qmc.x(data[j])
            return qmc.measure(data)

        exe = transpiler.transpile(kernel)
        result = exe.sample(transpiler.executor(), shots=100).result()
        # flag == 1 (always), so all data qubits should be flipped.
        for outcome, count in result.results:
            bits = (
                outcome
                if isinstance(outcome, (list, tuple))
                else [(outcome >> i) & 1 for i in range(3)]
            )
            # data measured first 3 bits. Should all be 1.
            data_bits = bits[:3]
            assert all(b == 1 for b in data_bits), f"Expected all 1s, got {data_bits}"


class TestBitMergeAliasTypeCheck:
    """``register_classical_merge_aliases`` must detect Bit merges by type.

    Regression guard for a dead type check: the Bit detection used
    ``hasattr(output.type, "_is_bit_marker")``, an attribute no IR type
    defines, so ``is_bit`` was always ``False``. Runtime Bit merges then
    fell through to the identity-binding path (they belong to the clbit
    mapping instead), and the static-Bit bool coercion never ran. The
    check is now ``isinstance(output.type, BitType)``; these tests pin
    both behaviors directly at the helper level.
    """

    @staticmethod
    def _emit_pass_stub():
        """Build the minimal emit-pass surface the helper consumes.

        Returns:
            SimpleNamespace: Object exposing ``_resolver`` with a real
                ``ValueResolver``, the only attribute
                ``register_classical_merge_aliases`` touches.
        """
        from types import SimpleNamespace

        from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
            ValueResolver,
        )

        return SimpleNamespace(_resolver=ValueResolver())

    def test_runtime_bit_merge_is_left_to_clbit_mapping(self):
        """resolved=None: a Bit identity merge is skipped, a UInt one binds."""
        from qamomile.circuit.ir.operation.control_flow import IfOperation
        from qamomile.circuit.ir.types.primitives import BitType, UIntType
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
            register_classical_merge_aliases,
        )

        shared_bit = Value(type=BitType(), name="flag").with_const(1)
        shared_uint = Value(type=UIntType(), name="j").with_const(1)
        bit_result = Value(type=BitType(), name="flag_merge_0")
        uint_result = Value(type=UIntType(), name="j_merge_0")
        if_op = IfOperation(operands=[Value(type=BitType(), name="cond")])
        if_op.add_merge(shared_bit, shared_bit, bit_result)
        if_op.add_merge(shared_uint, shared_uint, uint_result)

        bindings: dict = {}
        register_classical_merge_aliases(self._emit_pass_stub(), if_op, bindings, None)

        assert bit_result.uuid not in bindings, (
            "runtime Bit merge must stay on the clbit-mapping path"
        )
        assert bindings[uint_result.uuid] == 1, (
            "non-Bit identity merge must still be pre-bound"
        )

    def test_static_bit_merge_binds_python_bool(self):
        """resolved=True: a statically selected Bit merge binds as bool."""
        from qamomile.circuit.ir.operation.control_flow import IfOperation
        from qamomile.circuit.ir.types.primitives import BitType
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
            register_classical_merge_aliases,
        )

        true_bit = Value(type=BitType(), name="t").with_const(1)
        false_bit = Value(type=BitType(), name="f").with_const(0)
        result = Value(type=BitType(), name="bit_merge_0")
        if_op = IfOperation(operands=[Value(type=BitType(), name="cond")])
        if_op.add_merge(true_bit, false_bit, result)

        bindings: dict = {}
        register_classical_merge_aliases(self._emit_pass_stub(), if_op, bindings, True)

        assert bindings[result.uuid] is True, (
            "static Bit merge must bind the selected branch as a Python bool"
        )
