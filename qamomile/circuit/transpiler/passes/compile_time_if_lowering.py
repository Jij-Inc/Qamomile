"""Compile-time IfOperation lowering pass.

Lowers compile-time resolvable ``IfOperation``s before the separate pass,
replacing them with selected-branch operations and substituting phi outputs
with selected-branch values throughout the block.

This prevents ``SeparatePass`` from seeing classical-only compile-time
``IfOperation``s that would otherwise split quantum segments.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    PhiOp,
)
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase

from . import Pass
from .emit_base import resolve_if_condition


class CompileTimeIfLoweringPass(Pass[Block, Block]):
    """Lowers compile-time resolvable IfOperations before separation.

    After constant folding, some ``IfOperation`` conditions are statically
    known but remain as control-flow nodes.  ``SeparatePass`` treats them
    as segment boundaries, causing ``MultipleQuantumSegmentsError`` for
    classical-only compile-time ``if`` after quantum init.

    This pass:

    1. Evaluates conditions including expression-derived ones
       (``CompOp``, ``CondOp``, ``NotOp`` chains).
    2. Replaces resolved ``IfOperation``s with selected-branch operations.
    3. Substitutes phi output UUIDs with selected-branch values in all
       subsequent operations and block outputs.
    """

    def __init__(self, bindings: dict[str, Any] | None = None):
        self._bindings = bindings or {}

    @property
    def name(self) -> str:
        return "compile_time_if_lowering"

    def run(self, input: Block) -> Block:
        """Run the compile-time if lowering pass."""
        # Track concrete values produced by classical ops.
        # Maps UUID → concrete Python value (int, float, bool).
        concrete_values: dict[str, Any] = {}

        # Seed concrete_values from bindings (by UUID and by name).
        for key, val in self._bindings.items():
            concrete_values[key] = val

        # Seed from block input_values that are constants or bound.
        for iv in input.input_values:
            self._try_seed_value(iv, concrete_values)

        # Process operations, lowering compile-time ifs.
        new_ops, phi_subst, dead_uuids = self._lower_operations(
            input.operations, concrete_values
        )

        # Apply phi substitution to block output_values.
        new_outputs = self._substitute_output_values(input.output_values, phi_subst)

        # Remove dead operations whose results are only used by lowered ifs.
        if dead_uuids:
            new_ops = self._eliminate_dead_ops(new_ops, dead_uuids, new_outputs)

        return dataclasses.replace(
            input,
            operations=new_ops,
            output_values=new_outputs,
        )

    # ------------------------------------------------------------------
    # Condition evaluation
    # ------------------------------------------------------------------

    def _try_resolve_condition(
        self,
        condition: Any,
        concrete_values: dict[str, Any],
    ) -> bool | None:
        """Resolve an IfOperation condition to a compile-time bool.

        Tries ``resolve_if_condition`` first (handles plain Python values,
        constants, direct bindings).  Then falls back to expression-aware
        evaluation using the accumulated ``concrete_values`` map.
        """
        # Fast path: direct resolution.
        resolved = resolve_if_condition(condition, self._bindings)
        if resolved is not None:
            return resolved

        # Expression-aware path: check if the condition's UUID is in
        # concrete_values (produced by a prior CompOp/CondOp/NotOp).
        if hasattr(condition, "uuid") and condition.uuid in concrete_values:
            return bool(concrete_values[condition.uuid])

        return None

    def _try_evaluate_classical_op(
        self,
        op: Operation,
        concrete_values: dict[str, Any],
    ) -> None:
        """Try to evaluate a classical op and record its result.

        Handles ``CompOp``, ``CondOp``, ``NotOp``, and ``BinOp``.
        Only evaluates if all operands are resolvable.
        """
        if isinstance(op, CompOp):
            lhs = self._resolve_operand(op.operands[0], concrete_values)
            rhs = self._resolve_operand(op.operands[1], concrete_values)
            if lhs is not None and rhs is not None and op.results:
                result = self._eval_comp(op.kind, lhs, rhs)
                if result is not None:
                    concrete_values[op.results[0].uuid] = result

        elif isinstance(op, CondOp):
            lhs = self._resolve_operand(op.operands[0], concrete_values)
            rhs = self._resolve_operand(op.operands[1], concrete_values)
            if lhs is not None and rhs is not None and op.results:
                result = self._eval_cond(op.kind, lhs, rhs)
                if result is not None:
                    concrete_values[op.results[0].uuid] = result

        elif isinstance(op, NotOp):
            operand = self._resolve_operand(op.operands[0], concrete_values)
            if operand is not None and op.results:
                concrete_values[op.results[0].uuid] = not operand

        elif isinstance(op, BinOp):
            lhs = self._resolve_operand(op.operands[0], concrete_values)
            rhs = self._resolve_operand(op.operands[1], concrete_values)
            if lhs is not None and rhs is not None and op.results:
                result = self._eval_binop(op.kind, lhs, rhs)
                if result is not None:
                    concrete_values[op.results[0].uuid] = result

    def _resolve_operand(
        self,
        value: Any,
        concrete_values: dict[str, Any],
    ) -> Any:
        """Resolve an operand to a concrete value, or None."""
        if not hasattr(value, "uuid"):
            return value

        # Check concrete_values map first (includes bindings by UUID).
        if value.uuid in concrete_values:
            return concrete_values[value.uuid]

        # Check constant.
        if hasattr(value, "is_constant") and value.is_constant():
            return value.get_const()

        # Check binding by name.
        if hasattr(value, "name") and value.name and value.name in self._bindings:
            return self._bindings[value.name]

        # Check parameter binding.
        if hasattr(value, "is_parameter") and value.is_parameter():
            param_name = value.parameter_name()
            if param_name and param_name in self._bindings:
                return self._bindings[param_name]

        # Check params["const"].
        if hasattr(value, "params") and value.params and "const" in value.params:
            return value.params["const"]

        return None

    @staticmethod
    def _eval_comp(kind: Any, lhs: Any, rhs: Any) -> bool | None:
        """Evaluate a comparison operation."""
        try:
            match kind:
                case CompOpKind.EQ:
                    return lhs == rhs
                case CompOpKind.NEQ:
                    return lhs != rhs
                case CompOpKind.LT:
                    return lhs < rhs
                case CompOpKind.LE:
                    return lhs <= rhs
                case CompOpKind.GT:
                    return lhs > rhs
                case CompOpKind.GE:
                    return lhs >= rhs
                case _:
                    return None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _eval_cond(kind: Any, lhs: Any, rhs: Any) -> bool | None:
        """Evaluate a conditional logical operation."""
        try:
            match kind:
                case CondOpKind.AND:
                    return bool(lhs and rhs)
                case CondOpKind.OR:
                    return bool(lhs or rhs)
                case _:
                    return None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _eval_binop(kind: Any, lhs: Any, rhs: Any) -> Any:
        """Evaluate a binary arithmetic operation."""
        try:
            match kind:
                case BinOpKind.ADD:
                    return lhs + rhs
                case BinOpKind.SUB:
                    return lhs - rhs
                case BinOpKind.MUL:
                    return lhs * rhs
                case BinOpKind.DIV:
                    return lhs / rhs if rhs != 0 else None
                case BinOpKind.FLOORDIV:
                    return lhs // rhs if rhs != 0 else None
                case BinOpKind.POW:
                    return lhs**rhs
                case _:
                    return None
        except (TypeError, ValueError, OverflowError):
            return None

    # ------------------------------------------------------------------
    # Operation lowering
    # ------------------------------------------------------------------

    def _lower_operations(
        self,
        operations: list[Operation],
        concrete_values: dict[str, Any],
    ) -> tuple[list[Operation], dict[str, ValueBase], set[str]]:
        """Process operations, lowering compile-time IfOperations.

        Returns:
            Tuple of (new operations list, phi substitution map,
            dead UUIDs from lowered condition values).
        """
        new_ops: list[Operation] = []
        phi_subst: dict[str, ValueBase] = {}
        dead_uuids: set[str] = set()

        for op in operations:
            # First apply any accumulated phi substitutions to this op.
            op = self._apply_substitution(op, phi_subst)

            # Track classical op results for expression-aware evaluation.
            self._try_evaluate_classical_op(op, concrete_values)

            if isinstance(op, IfOperation):
                resolved = self._try_resolve_condition(op.condition, concrete_values)
                if resolved is not None:
                    # Mark the condition value as dead (its producer can
                    # be removed if no other op uses it).
                    if hasattr(op.condition, "uuid"):
                        dead_uuids.add(op.condition.uuid)

                    # Build phi substitution map.
                    for phi in op.phi_ops:
                        if not isinstance(phi, PhiOp):
                            continue
                        selected_val = phi.true_value if resolved else phi.false_value
                        phi_subst[phi.output.uuid] = selected_val

                    # Inline selected branch operations.
                    selected_ops = (
                        op.true_operations if resolved else op.false_operations
                    )
                    # Recursively lower any nested compile-time ifs in the
                    # selected branch.
                    lowered, nested_subst, nested_dead = self._lower_operations(
                        selected_ops, concrete_values
                    )
                    phi_subst.update(nested_subst)
                    dead_uuids.update(nested_dead)
                    new_ops.extend(lowered)
                    continue

            # For non-lowered IfOperations, recurse into branches.
            if isinstance(op, IfOperation):
                lowered_true, subst_true, dead_true = self._lower_operations(
                    op.true_operations, dict(concrete_values)
                )
                lowered_false, subst_false, dead_false = self._lower_operations(
                    op.false_operations, dict(concrete_values)
                )
                # Apply nested substitutions to phi_ops.
                nested_subst_if = {**subst_true, **subst_false}
                new_phi_ops = [
                    self._apply_substitution(p, nested_subst_if) for p in op.phi_ops
                ]
                op = dataclasses.replace(
                    op,
                    true_operations=lowered_true,
                    false_operations=lowered_false,
                    phi_ops=new_phi_ops,
                )
                dead_uuids.update(dead_true)
                dead_uuids.update(dead_false)

            new_ops.append(op)

        return new_ops, phi_subst, dead_uuids

    # ------------------------------------------------------------------
    # Substitution
    # ------------------------------------------------------------------

    def _apply_substitution(
        self,
        op: Operation,
        subst: dict[str, ValueBase],
    ) -> Operation:
        """Apply phi substitution map to an operation's operands and results."""
        if not subst:
            return op

        new_operands = [
            self._substitute_value(v, subst) if isinstance(v, ValueBase) else v
            for v in op.operands
        ]

        new_results = list(op.results)

        changed = any(
            new_operands[i] is not op.operands[i]
            for i in range(len(op.operands))
            if isinstance(op.operands[i], ValueBase)
        )

        # Handle control-flow recursion.
        if isinstance(op, IfOperation):
            new_true = [self._apply_substitution(o, subst) for o in op.true_operations]
            new_false = [
                self._apply_substitution(o, subst) for o in op.false_operations
            ]
            new_phi = [self._apply_substitution(p, subst) for p in op.phi_ops]
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                true_operations=new_true,
                false_operations=new_false,
                phi_ops=new_phi,
            )

        from qamomile.circuit.ir.operation.control_flow import (
            ForOperation,
            WhileOperation,
        )
        from qamomile.circuit.ir.operation.gate import GateOperation

        if isinstance(op, ForOperation):
            new_body = [self._apply_substitution(o, subst) for o in op.operations]
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                operations=new_body,
            )

        if isinstance(op, WhileOperation):
            new_body = [self._apply_substitution(o, subst) for o in op.operations]
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                operations=new_body,
            )

        result_op = op
        if changed:
            result_op = dataclasses.replace(op, operands=new_operands)

        # Handle GateOperation.theta.
        if isinstance(result_op, GateOperation) and isinstance(result_op.theta, Value):
            new_theta = self._substitute_value(result_op.theta, subst)
            if new_theta is not result_op.theta and isinstance(new_theta, Value):
                result_op = dataclasses.replace(result_op, theta=new_theta)

        return result_op

    def _substitute_value(
        self,
        v: ValueBase,
        subst: dict[str, ValueBase],
    ) -> ValueBase:
        """Substitute a value using the phi substitution map."""
        if v.uuid in subst:
            return subst[v.uuid]

        # Handle array elements whose parent_array should be substituted.
        if isinstance(v, Value) and v.parent_array is not None:
            if v.parent_array.uuid in subst:
                new_parent = subst[v.parent_array.uuid]
                if isinstance(new_parent, ArrayValue):
                    return dataclasses.replace(v, parent_array=new_parent)

        # Handle element_indices substitution.
        if isinstance(v, Value) and v.element_indices:
            new_indices = []
            indices_changed = False
            for idx in v.element_indices:
                if idx.uuid in subst:
                    sub_idx = subst[idx.uuid]
                    if isinstance(sub_idx, Value):
                        new_indices.append(sub_idx)
                        indices_changed = True
                    else:
                        new_indices.append(idx)
                else:
                    new_indices.append(idx)
            if indices_changed:
                return dataclasses.replace(v, element_indices=tuple(new_indices))

        return v

    def _substitute_output_values(
        self,
        output_values: list[Value],
        subst: dict[str, ValueBase],
    ) -> list[Value]:
        """Apply phi substitution to block output values."""
        if not subst:
            return output_values

        new_outputs = []
        for ov in output_values:
            substituted = self._substitute_value(ov, subst)
            if isinstance(substituted, Value):
                new_outputs.append(substituted)
            else:
                new_outputs.append(ov)
        return new_outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _eliminate_dead_ops(
        self,
        operations: list[Operation],
        dead_uuids: set[str],
        output_values: list[Value] | None = None,
    ) -> list[Operation]:
        """Remove operations whose results are only consumed by dead UUIDs.

        An operation is removable if ALL of its result UUIDs are in the
        dead set AND none of those results are used by any remaining
        operation's operands (or block outputs).

        This is applied iteratively to propagate: if removing a CompOp
        makes a BinOp's result dead, the BinOp is also removed.
        """
        # Build use set: UUIDs referenced as operands in remaining ops.
        used_uuids: set[str] = set()
        for op in operations:
            self._collect_used_uuids(op, used_uuids)
        # Also include block output UUIDs as used.
        if output_values:
            for ov in output_values:
                used_uuids.add(ov.uuid)

        result: list[Operation] = []
        for op in operations:
            # Check if all results are dead and unused.
            if op.results and all(
                r.uuid in dead_uuids and r.uuid not in used_uuids
                for r in op.results
                if hasattr(r, "uuid")
            ):
                # This op can be safely removed. Mark its operands as
                # potentially dead too (for iterative propagation).
                for operand in op.operands:
                    if hasattr(operand, "uuid"):
                        dead_uuids.add(operand.uuid)
                continue
            result.append(op)

        # If we removed anything, do another pass for propagation.
        if len(result) < len(operations):
            return self._eliminate_dead_ops(result, dead_uuids, output_values)
        return result

    @staticmethod
    def _collect_used_uuids(op: Operation, used: set[str]) -> None:
        """Collect all UUIDs used as operands in an operation (recursive)."""
        for operand in op.operands:
            if hasattr(operand, "uuid"):
                used.add(operand.uuid)
            # Also collect element_indices and parent_array references.
            if isinstance(operand, Value):
                if operand.parent_array is not None:
                    used.add(operand.parent_array.uuid)
                for idx in operand.element_indices:
                    used.add(idx.uuid)

        # GateOperation.theta
        from qamomile.circuit.ir.operation.gate import GateOperation

        if isinstance(op, GateOperation) and isinstance(op.theta, Value):
            used.add(op.theta.uuid)

        # Recurse into control flow.
        if isinstance(op, IfOperation):
            for inner in op.true_operations:
                CompileTimeIfLoweringPass._collect_used_uuids(inner, used)
            for inner in op.false_operations:
                CompileTimeIfLoweringPass._collect_used_uuids(inner, used)
            for phi in op.phi_ops:
                CompileTimeIfLoweringPass._collect_used_uuids(phi, used)

        from qamomile.circuit.ir.operation.control_flow import (
            ForOperation,
            WhileOperation,
        )

        if isinstance(op, ForOperation):
            for inner in op.operations:
                CompileTimeIfLoweringPass._collect_used_uuids(inner, used)
        if isinstance(op, WhileOperation):
            for inner in op.operations:
                CompileTimeIfLoweringPass._collect_used_uuids(inner, used)

    def _try_seed_value(
        self,
        value: ValueBase,
        concrete_values: dict[str, Any],
    ) -> None:
        """Seed concrete_values from a block input value if it's constant or bound."""
        if hasattr(value, "is_constant") and value.is_constant():
            const = value.get_const() if hasattr(value, "get_const") else None
            if const is not None:
                concrete_values[value.uuid] = const

        if hasattr(value, "is_parameter") and value.is_parameter():
            param_name = (
                value.parameter_name() if hasattr(value, "parameter_name") else None
            )
            if param_name and param_name in self._bindings:
                concrete_values[value.uuid] = self._bindings[param_name]

        if hasattr(value, "name") and value.name and value.name in self._bindings:
            concrete_values[value.uuid] = self._bindings[value.name]
