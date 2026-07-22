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
IR built before the field existed), analysis conservatively requests
unrolling and the emit entry point rejects the malformed IR. Comparing by
name was the soil for nested-loop name-collision bugs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    RuntimeClassicalExpr,
)
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.classical_ops import DictGetItemOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
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

    Args:
        value (Value): Candidate IR value.
        loop_var_uuid (str | None): Enclosing loop-variable UUID, or None for
            legacy IR.

    Returns:
        bool: True when ``value`` is the enclosing loop variable.
    """
    if loop_var_uuid is None:
        return False
    return value.uuid == loop_var_uuid


class LoopAnalyzer:
    """Analyze loop structures to determine an emission strategy."""

    def should_unroll(
        self,
        op: ForOperation,
        bindings: dict[str, object],
    ) -> bool:
        """Determine whether a for-loop requires emit-time unrolling.

        Args:
            op (ForOperation): Loop to analyze.
            bindings (dict[str, object]): Compile-time bindings available to
                nested-bound analysis.

        Returns:
            bool: True when identities, carries, indices, predicates,
                dictionary lookups, structural body parameters, or nested
                bounds require concrete per-iteration evaluation.
        """
        if op.loop_var_value is None:
            # Never recover compiler identity from a display label. The emit
            # entry point rejects this legacy/malformed IR, while returning
            # True here prevents standalone analyzer callers from selecting a
            # native path that cannot bind the loop index correctly.
            return True
        if op.region_args:
            # A native backend loop keeps one static body per iteration and
            # cannot thread a classical value between iterations.
            return True
        loop_uuid = op.loop_var_value.uuid
        if self._has_dynamic_nested_loop(op.operations, bindings, loop_uuid):
            return True
        return self._has_loop_var_dependency(op.operations, loop_uuid)

    def _has_loop_var_dependency(
        self,
        operations: list[Operation],
        loop_var_uuid: str | None,
    ) -> bool:
        """Return whether a loop body needs a concrete Python index.

        Preserve native backend-loop parameters for ordinary gate angles,
        while forcing unrolling for boxed-call parameters and structural
        decisions: classical expression evaluation, runtime conditions,
        dictionary lookup, operation-owned controlled/inverse bodies, phase
        normalization, and any array/slice/container address derived from the
        induction value. The generic ``all_input_values`` scan is retained for
        structural ancestry so subclass-specific fields such as symbolic
        control indices cannot be omitted.

        Args:
            operations (list[Operation]): Loop-body operations to scan.
            loop_var_uuid (str | None): UUID of the enclosing loop variable,
                or None for legacy IR without ``loop_var_value``.

        Returns:
            bool: True if any operation input depends on the loop variable.
        """
        for op in operations:
            if self._operation_requires_concrete_loop_var(op, loop_var_uuid):
                return True
            if isinstance(op, HasNestedOps) and any(
                self._has_loop_var_dependency(op_list, loop_var_uuid)
                for op_list in op.nested_op_lists()
            ):
                return True
        return False

    def _operation_requires_concrete_loop_var(
        self,
        op: Operation,
        loop_var_uuid: str | None,
    ) -> bool:
        """Return whether one operation cannot consume a backend loop value.

        Args:
            op (Operation): Loop-body operation to inspect.
            loop_var_uuid (str | None): UUID of the enclosing loop variable.

        Returns:
            bool: True when ``op`` needs the induction value as a concrete
                Python scalar during emission.
        """
        requires_direct = False
        if isinstance(op, (BinOp, CompOp, CondOp, NotOp, RuntimeClassicalExpr)):
            requires_direct = any(
                self._value_depends_on_loop_var(value, loop_var_uuid)
                for value in op.all_input_values()
            )
        elif isinstance(op, IfOperation):
            requires_direct = self._value_depends_on_loop_var(
                op.condition, loop_var_uuid
            )
        elif isinstance(op, WhileOperation):
            requires_direct = any(
                self._value_depends_on_loop_var(value, loop_var_uuid)
                for value in op.operands
            )
            requires_direct = requires_direct or any(
                self._value_depends_on_loop_var(arg.init, loop_var_uuid)
                for arg in op.region_args
            )
        elif isinstance(op, (ForOperation, ForItemsOperation)):
            # A nested carry seeded from the outer induction value may feed any
            # downstream operation through its distinct block-argument UUID.
            # Unroll at the seed edge instead of attempting partial dataflow
            # analysis through every carried-value consumer.
            requires_direct = any(
                self._value_depends_on_loop_var(arg.init, loop_var_uuid)
                for arg in op.region_args
            )
        elif isinstance(op, DictGetItemOperation):
            requires_direct = any(
                self._value_depends_on_loop_var(value, loop_var_uuid)
                for value in op.operands[1:]
            )
        elif isinstance(op, ControlledUOperation):
            requires_direct = any(
                self._value_depends_on_loop_var(value, loop_var_uuid)
                for value in op.all_input_values()
            )
        elif isinstance(op, InverseBlockOperation):
            requires_direct = any(
                self._value_depends_on_loop_var(value, loop_var_uuid)
                for value in op.parameters
            )
        elif isinstance(op, GlobalPhaseOperation):
            requires_direct = any(
                self._value_depends_on_loop_var(value, loop_var_uuid)
                for value in op.operands
            )
        elif isinstance(op, PauliEvolveOp):
            requires_direct = self._value_depends_on_loop_var(
                op.gamma,
                loop_var_uuid,
            )
        elif isinstance(op, InvokeOperation):
            requires_direct = any(
                self._value_depends_on_loop_var(value, loop_var_uuid)
                for value in op.parameters
            )
        elif isinstance(op, SelectOperation):
            # SELECT case bodies become independent reusable circuits at the
            # CircuitProgram boundary and therefore cannot capture a caller's
            # native loop variable. Its index width also fixes reusable-circuit
            # arity. Materialize a concrete body per iteration whenever either
            # the width or a shared case parameter depends on that variable.
            requires_direct = any(
                self._value_depends_on_loop_var(value, loop_var_uuid)
                for value in [*op.param_operands, op.num_index_qubits]
            )
        if requires_direct:
            return True

        # Ordinary gate operands may consume a backend-native loop parameter
        # directly. Boxed callable bodies are materialized independently and
        # cannot capture the caller's loop-variable scope, so their direct
        # parameters were handled above. Structural descendants (array
        # indices, slice bounds, shapes, tuple/dict members) always force a
        # concrete Python iteration value.
        return any(
            self._value_depends_on_loop_var(
                value,
                loop_var_uuid,
                include_direct=False,
            )
            for value in op.all_input_values()
        )

    def _value_depends_on_loop_var(
        self,
        value: object,
        loop_var_uuid: str | None,
        visiting: frozenset[int] = frozenset(),
        *,
        include_direct: bool = True,
    ) -> bool:
        """Return whether an IR value structurally reads the loop variable.

        Args:
            value (object): Candidate operation input or nested IR value.
            loop_var_uuid (str | None): UUID of the enclosing loop variable.
            visiting (frozenset[int]): Object identities already visited on
                the current recursive path. Defaults to an empty set.
            include_direct (bool): Whether ``value`` itself matching the loop
                variable counts. Defaults to True; callers pass False when a
                backend may consume a direct loop parameter but not a value
                nested inside an address/container structure.

        Returns:
            bool: True if ``value`` depends on the loop variable.
        """
        from qamomile.circuit.ir.value import (
            ArrayValue,
            DictValue,
            TupleValue,
            Value,
            ValueBase,
        )

        if not isinstance(value, ValueBase):
            return False
        object_id = id(value)
        if object_id in visiting:
            return False
        next_visiting = visiting | {object_id}

        if (
            include_direct
            and isinstance(value, Value)
            and _is_loop_var(value, loop_var_uuid)
        ):
            return True

        children: list[object] = []
        if isinstance(value, Value):
            children.extend(value.element_indices)
            if value.parent_array is not None:
                children.append(value.parent_array)
        if isinstance(value, ArrayValue):
            children.extend(value.shape)
            if value.slice_of is not None:
                children.append(value.slice_of)
            if value.slice_start is not None:
                children.append(value.slice_start)
            if value.slice_step is not None:
                children.append(value.slice_step)
        elif isinstance(value, TupleValue):
            children.extend(value.elements)
        elif isinstance(value, DictValue):
            for key, item in value.entries:
                children.extend((key, item))

        return any(
            self._value_depends_on_loop_var(child, loop_var_uuid, next_visiting)
            for child in children
        )

    def _has_dynamic_nested_loop(
        self,
        operations: list[Operation],
        bindings: dict[str, object],
        parent_loop_var_uuid: str | None,
    ) -> bool:
        """Return whether a nested loop needs the parent's concrete index.

        Args:
            operations (list[Operation]): Operations in the parent loop body.
            bindings (dict[str, object]): Compile-time bindings available for
                nested loop bounds.
            parent_loop_var_uuid (str | None): UUID of the parent loop variable,
                or None for legacy IR without ``loop_var_value``.

        Returns:
            bool: True if a nested bound directly uses the parent index or
                resolves to a non-numeric binding.
        """
        from qamomile.circuit.ir.value import Value

        for op in operations:
            if isinstance(op, ForOperation):
                for bound_val in op.operands[:3]:
                    if isinstance(bound_val, Value):
                        if _is_loop_var(bound_val, parent_loop_var_uuid):
                            return True
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
            if isinstance(op, HasNestedOps) and any(
                self._has_dynamic_nested_loop(op_list, bindings, parent_loop_var_uuid)
                for op_list in op.nested_op_lists()
            ):
                return True
        return False
