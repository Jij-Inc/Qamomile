"""Loop analysis helpers for emission.

Identity policy: loop variables are identified by their IR ``Value``
UUID (``ForOperation.loop_var_value.uuid``), **not** by their display
name. Two nested or sibling loops with identical user-chosen names
(e.g. ``for i in range(N): for i in range(M):``) are distinct here
because each ``ForOperation`` carries its own ``loop_var_value`` with
a fresh UUID, and ``UUIDRemapper`` clones that field consistently with
body references via the ``all_input_values`` / ``replace_values``
protocol.

There is **no name fallback**: if ``loop_var_value`` is ``None`` (legacy
IR built before the field existed) we skip the loop-var checks entirely.
Comparing by name was the soil for nested-loop name-collision bugs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    RuntimeClassicalExpr,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation, GateOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.select import SelectOperation

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


def _is_loop_var(value: "Value", loop_var_uuid: str | None) -> bool:
    """Check whether ``value`` is the loop variable of the surrounding for.

    Pure UUID comparison. ``loop_var_uuid`` comes from
    ``ForOperation.loop_var_value`` and is preserved through inline
    cloning by ``UUIDRemapper`` (which uses the ``all_input_values`` /
    ``replace_values`` protocol). When the IR predates the
    ``loop_var_value`` field (``loop_var_uuid is None``), this returns
    ``False`` rather than falling back to name comparison.
    """
    if loop_var_uuid is None:
        return False
    return value.uuid == loop_var_uuid


class LoopAnalyzer:
    """Analyzes loop structures to determine emission strategy."""

    def should_unroll(
        self,
        op: ForOperation,
        bindings: dict[str, object],
    ) -> bool:
        loop_uuid = op.loop_var_value.uuid if op.loop_var_value is not None else None
        if self._has_dynamic_nested_loop(op.operations, bindings, loop_uuid):
            return True
        if self._has_array_element_access(op.operations, loop_uuid):
            return True
        if self._has_loop_var_binop(op.operations, loop_uuid):
            return True
        if self._has_loop_var_condition(op.operations, loop_uuid):
            return True
        return False

    def _has_loop_var_condition(
        self,
        operations: list[Operation],
        loop_var_uuid: str | None,
    ) -> bool:
        """True when a runtime control-flow condition depends on the loop var.

        A native backend loop (e.g. Qiskit ``for_loop``) keeps the loop
        variable as a backend loop parameter that classical-register
        indexing cannot consume. So when an ``IfOperation`` / ``WhileOperation``
        condition — or a ``RuntimeClassicalExpr`` operand — reads a measured
        ``Vector[Bit]`` element indexed by the loop variable (``if s[j]:``)
        or sliced by it (``if s[j:j+1][0]:``), the loop must be unrolled so
        the index resolves to a concrete clbit per iteration. Without this
        the condition's clbit address cannot be resolved and emit fails.

        Args:
            operations (list[Operation]): Loop-body operations to scan.
            loop_var_uuid (str | None): UUID of the enclosing loop variable,
                or None for legacy IR without ``loop_var_value``.

        Returns:
            bool: True if any runtime condition depends on the loop variable.
        """
        for op in operations:
            if isinstance(op, IfOperation):
                if self._value_depends_on_loop_var(op.condition, loop_var_uuid):
                    return True
            elif isinstance(op, WhileOperation):
                if any(
                    self._value_depends_on_loop_var(operand, loop_var_uuid)
                    for operand in op.operands
                ):
                    return True
            elif isinstance(op, RuntimeClassicalExpr):
                if any(
                    self._value_depends_on_loop_var(operand, loop_var_uuid)
                    for operand in op.operands
                ):
                    return True
            if isinstance(op, HasNestedOps):
                if any(
                    self._has_loop_var_condition(op_list, loop_var_uuid)
                    for op_list in op.nested_op_lists()
                ):
                    return True
        return False

    def _value_depends_on_loop_var(
        self,
        value: object,
        loop_var_uuid: str | None,
    ) -> bool:
        """True when a condition Value reads the loop variable.

        Covers three forms: the value *is* the loop variable; its element
        index depends on it (``s[j]``); or a ``slice_start`` / ``slice_step``
        along its ``parent_array``'s ``slice_of`` chain depends on it
        (``s[j:j+1][0]``).

        Args:
            value (object): Candidate condition operand (an IR ``Value`` or
                a non-Value, which never depends on the loop variable).
            loop_var_uuid (str | None): UUID of the enclosing loop variable.

        Returns:
            bool: True if ``value`` depends on the loop variable.
        """
        from qamomile.circuit.ir.value import Value as _Value

        if not isinstance(value, _Value):
            return False
        if _is_loop_var(value, loop_var_uuid):
            return True
        if value.element_indices:
            for idx in value.element_indices:
                if self._index_depends_on_loop_var(idx, loop_var_uuid):
                    return True
        parent = value.parent_array
        while parent is not None:
            for bound in (parent.slice_start, parent.slice_step):
                if isinstance(bound, _Value) and self._index_depends_on_loop_var(
                    bound, loop_var_uuid
                ):
                    return True
            parent = parent.slice_of
        return False

    def _has_loop_var_binop(
        self,
        operations: list[Operation],
        loop_var_uuid: str | None,
    ) -> bool:
        from qamomile.circuit.ir.value import Value

        for op in operations:
            if isinstance(op, BinOp):
                for operand in op.operands:
                    if isinstance(operand, Value) and _is_loop_var(
                        operand, loop_var_uuid
                    ):
                        return True
            if isinstance(op, HasNestedOps):
                if any(
                    self._has_loop_var_binop(op_list, loop_var_uuid)
                    for op_list in op.nested_op_lists()
                ):
                    return True
        return False

    def _has_dynamic_nested_loop(
        self,
        operations: list[Operation],
        bindings: dict[str, object],
        parent_loop_var_uuid: str | None,
    ) -> bool:
        from qamomile.circuit.ir.value import Value

        for op in operations:
            if isinstance(op, ForOperation):
                for bound_val in op.operands[:3]:
                    if isinstance(bound_val, Value):
                        if _is_loop_var(bound_val, parent_loop_var_uuid):
                            return True
                        # UUID-keyed bindings lookup. Names are kept around
                        # only for legacy migration-shim parameters; since
                        # those use is_parameter()-flagged Values, the
                        # name path is exercised by the dedicated parameter
                        # binding mechanism and not here.
                        if bound_val.uuid in bindings:
                            bound = bindings[bound_val.uuid]
                            if not isinstance(bound, (int, float)):
                                return True
                        elif bound_val.is_parameter():
                            param_name = bound_val.parameter_name()
                            if param_name and param_name in bindings:
                                bound = bindings[param_name]
                                if not isinstance(bound, (int, float)):
                                    return True
            if isinstance(op, HasNestedOps):
                if any(
                    self._has_dynamic_nested_loop(
                        op_list, bindings, parent_loop_var_uuid
                    )
                    for op_list in op.nested_op_lists()
                ):
                    return True
        return False

    def _has_array_element_access(
        self,
        operations: list[Operation],
        loop_var_uuid: str | None,
    ) -> bool:
        from qamomile.circuit.ir.value import Value as _Value

        for op in operations:
            if isinstance(op, GateOperation):
                for v in op.operands:
                    if v.parent_array is not None and v.element_indices:
                        for idx in v.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var_uuid):
                                return True

                if isinstance(op.theta, _Value):
                    if op.theta.parent_array is not None and op.theta.element_indices:
                        for idx in op.theta.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var_uuid):
                                return True

            elif isinstance(op, BinOp):
                for operand in [op.lhs, op.rhs]:
                    if operand.parent_array is not None and operand.element_indices:
                        for idx in operand.element_indices:
                            if self._index_depends_on_loop_var(idx, loop_var_uuid):
                                return True

            elif isinstance(op, (ControlledUOperation, SelectOperation)):
                # Both carry per-physical-qubit scalar control / index
                # operands plus target operands that may be ``reg[loop_var]``
                # element accesses; a loop-var-indexed operand forces
                # unrolling so the element resolves at emit.
                for v in op.operands:
                    if isinstance(v, _Value):
                        if v.parent_array is not None and v.element_indices:
                            for idx in v.element_indices:
                                if self._index_depends_on_loop_var(idx, loop_var_uuid):
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
                        if self._index_depends_on_loop_var(idx, loop_var_uuid):
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
                        if self._index_depends_on_loop_var(idx, loop_var_uuid):
                            return True

            if isinstance(op, HasNestedOps):
                if any(
                    self._has_array_element_access(op_list, loop_var_uuid)
                    for op_list in op.nested_op_lists()
                ):
                    return True

        return False

    def _index_depends_on_loop_var(
        self,
        idx: "Value",
        loop_var_uuid: str | None,
    ) -> bool:
        if _is_loop_var(idx, loop_var_uuid):
            return True
        if idx.parent_array is not None and idx.element_indices:
            for sub_idx in idx.element_indices:
                if self._index_depends_on_loop_var(sub_idx, loop_var_uuid):
                    return True
        return False
