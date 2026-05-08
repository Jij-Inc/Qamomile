"""Compile-time IfOperation lowering pass.

Lowers compile-time resolvable ``IfOperation``s before the segmentation pass,
replacing them with selected-branch operations and substituting phi outputs
with selected-branch values throughout the block.

This prevents ``SegmentationPass`` from seeing classical-only compile-time
``IfOperation``s that would otherwise split quantum segments.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    PhiOp,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps, IfOperation
from qamomile.circuit.ir.value import Value, ValueBase
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.value_resolver import (
    ValueResolver as UnifiedValueResolver,
)

from . import Pass
from .emit_support import resolve_if_condition
from .eval_utils import FoldPolicy, fold_classical_op
from .value_mapping import ValueSubstitutor


class CompileTimeIfLoweringPass(Pass[Block, Block]):
    """Lowers compile-time resolvable IfOperations before separation.

    After constant folding, some ``IfOperation`` conditions are statically
    known but remain as control-flow nodes.  ``SegmentationPass`` treats them
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
        # HIERARCHICAL is accepted during the self-recursion unroll loop;
        # surviving CallBlockOperations are passed through untouched.
        if input.kind not in (BlockKind.AFFINE, BlockKind.HIERARCHICAL):
            raise ValidationError(
                f"CompileTimeIfLoweringPass expects AFFINE or "
                f"HIERARCHICAL block, got {input.kind}",
            )

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

        Supported operation types:

        - ``CompOp``  — comparison (==, !=, <, <=, >, >=)
        - ``CondOp``  — logical connective (and, or)
        - ``NotOp``   — logical negation
        - ``BinOp``   — arithmetic (+, -, *, /, //, %)

        Delegates the actual fold to ``fold_classical_op`` under the
        ``COMPILE_TIME`` policy, which bypasses the runtime-parameter
        guard. At this stage there are no runtime parameters in the
        concrete values map; everything in ``self._bindings`` is treated
        as a real compile-time value.

        Other operation types are silently ignored. If all operands of a
        supported op are concrete but evaluation still fails, the result
        is not recorded and downstream IfOperations referencing it will
        remain unresolved (emitted as runtime branches).
        """
        if not isinstance(op, (CompOp, CondOp, NotOp, BinOp)):
            return
        if not op.results:
            return
        result = fold_classical_op(
            op,
            lambda v: self._resolve_operand(v, concrete_values),
            parameters=set(),
            policy=FoldPolicy.COMPILE_TIME,
        )
        if result is not None:
            concrete_values[op.results[0].uuid] = result

    def _resolve_operand(
        self,
        value: Any,
        concrete_values: dict[str, Any],
    ) -> Any:
        """Resolve an operand to a concrete value, or None."""
        return UnifiedValueResolver(
            context=concrete_values, bindings=self._bindings
        ).resolve(value)

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
                new_phi_ops = cast(
                    list[PhiOp],
                    [self._apply_substitution(p, nested_subst_if) for p in op.phi_ops],
                )
                op = dataclasses.replace(
                    op,
                    true_operations=lowered_true,
                    false_operations=lowered_false,
                    phi_ops=new_phi_ops,
                )
                dead_uuids.update(dead_true)
                dead_uuids.update(dead_false)

            elif isinstance(op, HasNestedOps):
                # Generic recursion for For/ForItems/While bodies.
                new_lists: list[list[Operation]] = []
                for body in op.nested_op_lists():
                    lowered_body, nested_subst, nested_dead = self._lower_operations(
                        body, dict(concrete_values)
                    )
                    new_lists.append(lowered_body)
                    phi_subst.update(nested_subst)
                    dead_uuids.update(nested_dead)
                op = op.rebuild_nested(new_lists)

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

        substitutor = ValueSubstitutor(subst, transitive=True)

        new_operands = cast(
            list[Value],
            [
                substitutor.substitute_value(v) if isinstance(v, ValueBase) else v
                for v in op.operands
            ],
        )

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
            new_phi = cast(
                list[PhiOp],
                [self._apply_substitution(p, subst) for p in op.phi_ops],
            )
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                true_operations=new_true,
                false_operations=new_false,
                phi_ops=new_phi,
            )

        from qamomile.circuit.ir.operation.gate import (
            ControlledUOperation,
            IndexSpecControlledU,
            SymbolicControlledU,
        )

        if isinstance(op, HasNestedOps):
            # Generic recursion for For/ForItems/While bodies.
            new_lists = [
                [self._apply_substitution(o, subst) for o in body]
                for body in op.nested_op_lists()
            ]
            rebuilt = op.rebuild_nested(new_lists)
            return dataclasses.replace(
                cast(Any, rebuilt),
                operands=new_operands,
                results=new_results,
            )

        result_op = op
        if changed:
            result_op = dataclasses.replace(op, operands=new_operands)

        # theta is now part of operands, handled by the operands
        # substitution above.

        # Handle ControlledUOperation non-operand fields per subclass.
        if isinstance(result_op, ControlledUOperation):
            extra_kwargs: dict[str, Any] = {}
            # power is shared across all subclasses.
            if isinstance(result_op.power, Value):
                new_power = substitutor.substitute_value(result_op.power)
                if new_power is not result_op.power:
                    extra_kwargs["power"] = new_power
            if isinstance(result_op, IndexSpecControlledU):
                if isinstance(result_op.num_controls, Value):
                    new_nc = substitutor.substitute_value(result_op.num_controls)
                    if new_nc is not result_op.num_controls:
                        extra_kwargs["num_controls"] = new_nc
                if result_op.target_indices is not None:
                    new_ti = self._substitute_value_list(
                        result_op.target_indices, substitutor
                    )
                    if new_ti is not None:
                        extra_kwargs["target_indices"] = new_ti
                if result_op.controlled_indices is not None:
                    new_ci = self._substitute_value_list(
                        result_op.controlled_indices, substitutor
                    )
                    if new_ci is not None:
                        extra_kwargs["controlled_indices"] = new_ci
            elif isinstance(result_op, SymbolicControlledU):
                new_nc = substitutor.substitute_value(result_op.num_controls)
                if new_nc is not result_op.num_controls:
                    extra_kwargs["num_controls"] = new_nc
            # ConcreteControlledU: num_controls is int, nothing to substitute.
            if extra_kwargs:
                result_op = dataclasses.replace(result_op, **extra_kwargs)

        # Handle CastOperation source provenance sync.
        from qamomile.circuit.ir.operation.cast import CastOperation

        if isinstance(result_op, CastOperation) and changed:
            new_source = new_operands[0] if new_operands else None
            if (
                new_source is not None
                and hasattr(new_source, "uuid")
                and result_op.results
            ):
                result_val = result_op.results[0]
                if result_val.is_cast_result():
                    num_bits = result_val.get_qfixed_num_bits()
                    new_result = result_val.with_cast_metadata(
                        source_uuid=new_source.uuid,
                        source_logical_id=getattr(
                            new_source, "logical_id", new_source.uuid
                        ),
                        qubit_uuids=(
                            [f"{new_source.uuid}_{i}" for i in range(num_bits)]
                            if num_bits is not None
                            else result_val.get_cast_qubit_uuids() or ()
                        ),
                        qubit_logical_ids=(
                            [
                                f"{getattr(new_source, 'logical_id', new_source.uuid)}_{i}"
                                for i in range(num_bits)
                            ]
                            if num_bits is not None
                            else result_val.get_cast_qubit_logical_ids() or ()
                        ),
                    )
                    if num_bits is not None:
                        new_result = new_result.with_qfixed_metadata(
                            qubit_uuids=[
                                f"{new_source.uuid}_{i}" for i in range(num_bits)
                            ],
                            num_bits=num_bits,
                            int_bits=result_val.get_qfixed_int_bits() or 0,
                        )
                    new_mapping = (
                        list(new_result.get_qfixed_qubit_uuids())
                        or result_op.qubit_mapping
                    )
                    result_op = dataclasses.replace(
                        result_op,
                        results=[new_result],
                        qubit_mapping=new_mapping,
                    )

        return result_op

    def _substitute_value_list(
        self,
        values: list[Value],
        substitutor: ValueSubstitutor,
    ) -> list[Value] | None:
        """Substitute values in a list, returning new list if changed, None otherwise."""
        new_values: list[Value] = []
        changed = False
        for v in values:
            new_v = substitutor.substitute_value(v)
            if new_v is not v and isinstance(new_v, Value):
                new_values.append(new_v)
                changed = True
            else:
                new_values.append(v)
        return new_values if changed else None

    def _substitute_output_values(
        self,
        output_values: list[Value],
        subst: dict[str, ValueBase],
    ) -> list[Value]:
        """Apply phi substitution to block output values."""
        if not subst:
            return output_values

        substitutor = ValueSubstitutor(subst, transitive=True)
        new_outputs = []
        for ov in output_values:
            substituted = substitutor.substitute_value(ov)
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

        # theta is now part of operands — its UUID is collected above.

        # ControlledUOperation non-operand fields (per subclass).
        from qamomile.circuit.ir.operation.gate import (
            ControlledUOperation,
            IndexSpecControlledU,
            SymbolicControlledU,
        )

        if isinstance(op, ControlledUOperation):
            if isinstance(op.power, Value):
                used.add(op.power.uuid)
            if isinstance(op, IndexSpecControlledU):
                if isinstance(op.num_controls, Value):
                    used.add(op.num_controls.uuid)
                if op.target_indices is not None:
                    for v in op.target_indices:
                        used.add(v.uuid)
                if op.controlled_indices is not None:
                    for v in op.controlled_indices:
                        used.add(v.uuid)
            elif isinstance(op, SymbolicControlledU):
                used.add(op.num_controls.uuid)

        # Recurse into control flow (For/ForItems/While/If).
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                for inner in body:
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
