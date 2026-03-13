"""Constant folding pass for compile-time expression evaluation."""

from __future__ import annotations

import dataclasses
from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.gate import ControlledUOperation, GateOperation
from qamomile.circuit.ir.value import Value, ValueBase

from . import Pass
from .control_flow_visitor import OperationTransformer
from .value_mapping import substitute_value_recursive


class ConstantFoldingPass(Pass[Block, Block]):
    """Evaluates constant expressions at compile time.

    This pass folds BinOp operations when all operands are constants
    or bound parameters, eliminating unnecessary classical operations
    that would otherwise split quantum segments.

    Example:
        Before (with bindings={"phase": 0.5}):
            BinOp(phase * 2) -> classical segment split

        After:
            Constant 1.0 -> no segment split
    """

    def __init__(self, bindings: dict[str, Any] | None = None):
        self._bindings = bindings or {}

    @property
    def name(self) -> str:
        return "constant_fold"

    def run(self, input: Block) -> Block:
        """Run constant folding on the block."""
        # Track folded values: uuid -> constant Value
        folded_values: dict[str, Value] = {}

        # Process operations
        new_ops = self._fold_operations(input.operations, folded_values)

        return dataclasses.replace(input, operations=new_ops)

    def _fold_operations(
        self,
        operations: list[Operation],
        folded_values: dict[str, Value],
    ) -> list[Operation]:
        """Process operations, folding constant BinOps."""
        outer_self = self

        class ConstantFoldingTransformer(OperationTransformer):
            def transform_operation(self, op: Operation) -> Operation | None:
                if isinstance(op, BinOp):
                    folded = outer_self._try_fold_binop(op, folded_values)
                    if folded is not None:
                        # BinOp was folded to a constant - remove it
                        # Just record the mapping for later substitution
                        folded_values[op.results[0].uuid] = folded
                        return None

                # Substitute folded values in operands
                return outer_self._substitute_folded_operands(op, folded_values)

        transformer = ConstantFoldingTransformer()
        return transformer.transform_operations(operations)

    def _try_fold_binop(
        self,
        op: BinOp,
        folded_values: dict[str, Value],
    ) -> Value | None:
        """Try to fold a BinOp to a constant. Returns None if not foldable."""
        if len(op.operands) != 2:
            return None

        left = self._resolve_value(op.operands[0], folded_values)
        right = self._resolve_value(op.operands[1], folded_values)

        if left is None or right is None:
            return None

        # Both operands are constants, evaluate
        result_value = self._evaluate_binop(op.kind, left, right)
        if result_value is None:
            return None

        # Create constant Value with same uuid for substitution
        result_type = op.results[0].type
        return Value(
            type=result_type,
            name=f"folded_{op.results[0].name}",
            params={"const": result_value},
            uuid=op.results[0].uuid,  # Keep same UUID for substitution
        )

    def _resolve_value(
        self,
        value: Value,
        folded_values: dict[str, Value],
    ) -> float | int | None:
        """Resolve a Value to a constant, or None if not resolvable."""
        # Check if already folded
        if value.uuid in folded_values:
            folded = folded_values[value.uuid]
            return folded.get_const()

        # Check if constant
        if value.is_constant():
            return value.get_const()

        # Check if bound parameter
        if value.is_parameter():
            param_name = value.parameter_name()
            if param_name and param_name in self._bindings:
                return self._bindings[param_name]

        return None

    def _evaluate_binop(
        self,
        kind: BinOpKind | None,
        left: float | int,
        right: float | int,
    ) -> float | int | None:
        """Evaluate a binary operation on two constants."""
        if kind is None:
            return None

        try:
            match kind:
                case BinOpKind.ADD:
                    return left + right
                case BinOpKind.SUB:
                    return left - right
                case BinOpKind.MUL:
                    return left * right
                case BinOpKind.DIV:
                    return left / right if right != 0 else None
                case BinOpKind.FLOORDIV:
                    return left // right if right != 0 else None
                case BinOpKind.POW:
                    return left**right
                case _:
                    return None
        except (TypeError, ValueError, OverflowError):
            return None

    def _substitute_folded_operands(
        self,
        op: Operation,
        folded_values: dict[str, Value],
    ) -> Operation:
        """Substitute folded constant values in operation operands and results.

        Uses ``substitute_value_recursive`` to propagate folded values
        through nested ``element_indices`` (index-of-index), ``parent_array``,
        and other Value-tree structures.

        For ``ControlledUOperation``, also folds ``num_controls``,
        ``target_indices``, ``controlled_indices``, and ``power`` fields.
        """
        # Cast folded_values to the broader type expected by the helper.
        value_map: dict[str, ValueBase] = dict(folded_values)

        # Substitute operands recursively.
        new_operands: list[Any] = []
        operands_changed = False
        for operand in op.operands:
            if isinstance(operand, ValueBase):
                new_operand = substitute_value_recursive(operand, value_map)
                new_operands.append(new_operand)
                if new_operand is not operand:
                    operands_changed = True
            else:
                new_operands.append(operand)

        # Substitute results recursively.
        new_results: list[Any] = []
        results_changed = False
        for result in op.results:
            if isinstance(result, ValueBase):
                new_result = substitute_value_recursive(result, value_map)
                new_results.append(new_result)
                if new_result is not result:
                    results_changed = True
            else:
                new_results.append(result)

        changed = operands_changed or results_changed
        replacements: dict[str, Any] = {}
        if operands_changed:
            replacements["operands"] = new_operands
        if results_changed:
            replacements["results"] = new_results

        result_op = dataclasses.replace(op, **replacements) if replacements else op

        # Fold ControlledUOperation-specific dataclass fields.
        if isinstance(result_op, ControlledUOperation):
            extra_kwargs: dict[str, Any] = {}

            # Fold num_controls: Value -> int if resolvable.
            if isinstance(result_op.num_controls, Value):
                new_nc = self._resolve_field_value(
                    result_op.num_controls, folded_values
                )
                if new_nc is not result_op.num_controls:
                    extra_kwargs["num_controls"] = new_nc
                    changed = True

            # Fold target_indices list.
            if result_op.target_indices is not None:
                new_ti = self._fold_value_list(result_op.target_indices, folded_values)
                if new_ti is not None:
                    extra_kwargs["target_indices"] = new_ti
                    changed = True

            # Fold controlled_indices list.
            if result_op.controlled_indices is not None:
                new_ci = self._fold_value_list(
                    result_op.controlled_indices, folded_values
                )
                if new_ci is not None:
                    extra_kwargs["controlled_indices"] = new_ci
                    changed = True

            # Fold power: Value -> int if resolvable (strict-integer-only).
            if isinstance(result_op.power, Value):
                new_power = self._resolve_power_field_value(
                    result_op.power, folded_values
                )
                if new_power is not result_op.power:
                    extra_kwargs["power"] = new_power
                    changed = True

            if extra_kwargs:
                result_op = dataclasses.replace(result_op, **extra_kwargs)

        # Substitute GateOperation.theta recursively.
        if isinstance(result_op, GateOperation) and isinstance(result_op.theta, Value):
            new_theta = substitute_value_recursive(result_op.theta, value_map)
            if isinstance(new_theta, Value) and new_theta is not result_op.theta:
                result_op = dataclasses.replace(result_op, theta=new_theta)
                changed = True

        return result_op

    def _resolve_field_value(
        self,
        value: Value,
        folded_values: dict[str, Value],
    ) -> Value | int:
        """Resolve a Value in a ControlledUOperation field.

        Returns a concrete ``int`` if resolvable, or the original
        ``Value`` unchanged.
        """
        if value.uuid in folded_values:
            folded = folded_values[value.uuid]
            const = folded.get_const()
            if const is not None:
                return int(const)
            return folded

        resolved = self._resolve_value(value, folded_values)
        if resolved is not None:
            return int(resolved)

        return value

    def _resolve_power_field_value(
        self,
        value: Value,
        folded_values: dict[str, Value],
    ) -> Value | int:
        """Resolve a ``ControlledUOperation.power`` Value to a concrete ``int``.

        Unlike :meth:`_resolve_field_value`, this uses strict-integer
        semantics: ``bool`` and non-integer ``float`` constants are
        rejected via :meth:`_strict_int_cast` instead of being silently
        coerced through ``int(...)``.

        Args:
            value: The symbolic ``Value`` stored in the ``power`` field.
            folded_values: UUID → folded ``Value`` map built by the pass.

        Returns:
            A concrete ``int`` if *value* can be resolved, or the original
            ``Value`` unchanged.
        """
        if value.uuid in folded_values:
            folded = folded_values[value.uuid]
            const = folded.get_const()
            if const is not None:
                return self._strict_int_cast(const)
            return folded

        resolved = self._resolve_value(value, folded_values)
        if resolved is not None:
            return self._strict_int_cast(resolved)

        return value

    @staticmethod
    def _strict_int_cast(value: object) -> int:
        """Cast *value* to ``int`` with strict validation for power fields.

        Only true integer values (or whole ``float`` like ``4.0``) are
        accepted.  ``bool``, non-integer ``float``, and non-positive
        integers are rejected.

        Args:
            value: The resolved constant to cast.

        Returns:
            A validated positive ``int``.

        Raises:
            ValueError: If *value* is ``bool``, a non-integer ``float``,
                a non-``int`` type, or ``<= 0``.
        """
        if isinstance(value, bool):
            raise ValueError(
                f"ControlledU power must be a positive integer, got bool ({value})."
            )
        if isinstance(value, float):
            if value != int(value):
                raise ValueError(
                    f"ControlledU power must be an integer, "
                    f"got non-integer float {value}."
                )
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(
                f"ControlledU power must be an integer, got {type(value).__name__}."
            )
        if value <= 0:
            raise ValueError(
                f"ControlledU power must be strictly positive, got {value}."
            )
        return value

    def _fold_value_list(
        self,
        values: list[Value],
        folded_values: dict[str, Value],
    ) -> list[Value] | None:
        """Fold a list of Values (e.g. target_indices).

        Returns a new list if any element changed, or ``None`` if unchanged.
        """
        new_values: list[Value] = []
        list_changed = False
        for v in values:
            if v.uuid in folded_values:
                new_values.append(folded_values[v.uuid])
                list_changed = True
            else:
                resolved = self._resolve_value(v, folded_values)
                if resolved is not None:
                    new_values.append(
                        Value(
                            type=v.type,
                            name=f"folded_{v.name}",
                            params={"const": int(resolved)},
                            uuid=v.uuid,
                        )
                    )
                    list_changed = True
                else:
                    new_values.append(v)
        return new_values if list_changed else None
