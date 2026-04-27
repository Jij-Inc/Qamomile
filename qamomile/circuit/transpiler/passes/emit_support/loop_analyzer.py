"""Loop analysis helpers for emission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
from qamomile.circuit.ir.operation.control_flow import ForOperation, HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation, GateOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


class LoopAnalyzer:
    """Analyzes loop structures to determine emission strategy."""

    def should_unroll(
        self,
        op: ForOperation,
        bindings: dict[str, object],
    ) -> bool:
        if self._has_dynamic_nested_loop(op.operations, bindings, op.loop_var):
            return True
        if self._has_array_element_access(op.operations, op.loop_var):
            return True
        if self._has_loop_var_binop(op.operations, op.loop_var):
            return True
        return False

    def _has_loop_var_binop(
        self,
        operations: list[Operation],
        loop_var: str,
    ) -> bool:
        from qamomile.circuit.ir.value import Value

        for op in operations:
            if isinstance(op, BinOp):
                for operand in op.operands:
                    if isinstance(operand, Value) and operand.name == loop_var:
                        return True
            if isinstance(op, HasNestedOps):
                if any(
                    self._has_loop_var_binop(op_list, loop_var)
                    for op_list in op.nested_op_lists()
                ):
                    return True
        return False

    def _has_dynamic_nested_loop(
        self,
        operations: list[Operation],
        bindings: dict[str, object],
        parent_loop_var: str,
    ) -> bool:
        for op in operations:
            if isinstance(op, ForOperation):
                for bound_val in op.operands[:3]:
                    if hasattr(bound_val, "name"):
                        if bound_val.name == parent_loop_var:
                            return True
                        if bound_val.name and bound_val.name in bindings:
                            bound = bindings[bound_val.name]
                            if not isinstance(bound, (int, float)):
                                return True
            if isinstance(op, HasNestedOps):
                if any(
                    self._has_dynamic_nested_loop(op_list, bindings, parent_loop_var)
                    for op_list in op.nested_op_lists()
                ):
                    return True
        return False

    def _has_array_element_access(
        self,
        operations: list[Operation],
        loop_var: str,
    ) -> bool:
        from qamomile.circuit.ir.value import Value as _Value

        for op in operations:
            if isinstance(op, GateOperation):
                for v in op.operands:
                    if v.parent_array is not None and v.element_indices:
                        for idx in v.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var):
                                return True

                if isinstance(op.theta, _Value):
                    if op.theta.parent_array is not None and op.theta.element_indices:
                        for idx in op.theta.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var):
                                return True

            elif isinstance(op, BinOp):
                for operand in [op.lhs, op.rhs]:
                    if operand.parent_array is not None and operand.element_indices:
                        for idx in operand.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var):
                                return True

            elif isinstance(op, ControlledUOperation):
                for v in op.operands:
                    if isinstance(v, _Value):
                        if v.parent_array is not None and v.element_indices:
                            for idx in v.element_indices:
                                if self._index_depends_on_loop_var(idx, loop_var):
                                    return True

            elif isinstance(op, PauliEvolveOp):
                # pauli_evolve gamma may be arr[loop_var], which requires
                # unrolling to materialise each layer's backend parameter.
                gamma = op.gamma
                if (
                    isinstance(gamma, _Value)
                    and gamma.parent_array is not None
                    and gamma.element_indices
                ):
                    for idx in gamma.element_indices:
                        if self._index_depends_on_loop_var(idx, loop_var):
                            return True
                # The observable operand may be Hs[loop_var] — a concrete
                # Hamiltonian is required at emit, so unroll too.
                observable = op.observable
                if (
                    isinstance(observable, _Value)
                    and observable.parent_array is not None
                    and observable.element_indices
                ):
                    for idx in observable.element_indices:
                        if self._index_depends_on_loop_var(idx, loop_var):
                            return True

            if isinstance(op, HasNestedOps):
                if any(
                    self._has_array_element_access(op_list, loop_var)
                    for op_list in op.nested_op_lists()
                ):
                    return True

        return False

    def _index_depends_on_loop_var(self, idx: "Value", loop_var: str) -> bool:
        if idx.name == loop_var:
            return True
        if idx.parent_array is not None and idx.element_indices:
            for sub_idx in idx.element_indices:
                if self._index_depends_on_loop_var(sub_idx, loop_var):
                    return True
        return False
