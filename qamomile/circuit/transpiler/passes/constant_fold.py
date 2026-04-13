"""Constant folding pass for compile-time expression evaluation."""

from __future__ import annotations

import dataclasses
from typing import Any

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    IndexSpecControlledU,
    SymbolicControlledU,
)
from qamomile.circuit.ir.value import Value, ValueBase
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.value_resolver import (
    ValueResolver as UnifiedValueResolver,
)

from . import Pass
from .control_flow_visitor import OperationTransformer


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
        if input.kind not in (BlockKind.AFFINE,):
            raise ValidationError(
                f"ConstantFoldingPass expects AFFINE block, got {input.kind}",
            )

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
            uuid=op.results[0].uuid,  # Keep same UUID for substitution
        ).with_const(result_value)

    def _resolve_value(
        self,
        value: Value,
        folded_values: dict[str, Value],
    ) -> float | int | None:
        """Resolve a Value to a constant, or None if not resolvable."""
        return UnifiedValueResolver(
            context=folded_values, bindings=self._bindings
        ).resolve(value)

    @staticmethod
    def _evaluate_binop(
        kind: BinOpKind | None,
        left: float | int,
        right: float | int,
    ) -> float | int | None:
        """Evaluate a binary operation on two constants."""
        from qamomile.circuit.transpiler.passes.eval_utils import (
            evaluate_binop_values,
        )

        return evaluate_binop_values(kind, left, right)

    def _substitute_folded_operands(
        self,
        op: Operation,
        folded_values: dict[str, Value],
    ) -> Operation:
        """Substitute folded constant values in operation operands.

        Also propagates folded values into ``element_indices`` of Value
        operands so that gate operations referencing a folded BinOp result
        as a qubit index are updated correctly.

        For ``ControlledUOperation``, also folds ``num_controls``,
        ``target_indices``, and ``controlled_indices`` fields.
        """
        new_operands: list[Any] = []
        changed = False

        for operand in op.operands:
            if isinstance(operand, ValueBase) and operand.uuid in folded_values:
                new_operands.append(folded_values[operand.uuid])
                changed = True
            elif isinstance(operand, Value) and operand.element_indices:
                new_indices = []
                indices_changed = False
                for idx in operand.element_indices:
                    if idx.uuid in folded_values:
                        new_indices.append(folded_values[idx.uuid])
                        indices_changed = True
                    else:
                        new_indices.append(idx)
                if indices_changed:
                    new_operands.append(
                        dataclasses.replace(operand, element_indices=tuple(new_indices))
                    )
                    changed = True
                else:
                    new_operands.append(operand)
            else:
                new_operands.append(operand)

        result_op = dataclasses.replace(op, operands=new_operands) if changed else op

        # Fold ControlledUOperation-specific dataclass fields per subclass.
        if isinstance(result_op, ControlledUOperation):
            extra_kwargs: dict[str, Any] = {}

            # Fold power (shared across all subclasses).
            if isinstance(result_op.power, Value):
                new_power = self._resolve_power_field_value(
                    result_op.power, folded_values
                )
                if new_power is not result_op.power:
                    extra_kwargs["power"] = new_power
                    changed = True

            if isinstance(result_op, IndexSpecControlledU):
                # Fold num_controls if symbolic.
                if isinstance(result_op.num_controls, Value):
                    new_nc = self._resolve_field_value(
                        result_op.num_controls, folded_values
                    )
                    if new_nc is not result_op.num_controls:
                        extra_kwargs["num_controls"] = new_nc
                        changed = True
                # Fold target_indices list.
                if result_op.target_indices is not None:
                    new_ti = self._fold_value_list(
                        result_op.target_indices, folded_values
                    )
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
            elif isinstance(result_op, SymbolicControlledU):
                # Fold num_controls: Value -> int.  If resolved to int,
                # promote to ConcreteControlledU.
                new_nc = self._resolve_field_value(
                    result_op.num_controls, folded_values
                )
                if new_nc is not result_op.num_controls:
                    if isinstance(new_nc, int):
                        # Promote to ConcreteControlledU.
                        power = extra_kwargs.get("power", result_op.power)
                        result_op = ConcreteControlledU(
                            operands=new_operands
                            if changed
                            else list(result_op.operands),
                            results=list(result_op.results),
                            num_controls=new_nc,
                            power=power,
                            block=result_op.block,
                        )
                        extra_kwargs = {}  # Already applied
                        changed = True
                    else:
                        extra_kwargs["num_controls"] = new_nc
                        changed = True
            # ConcreteControlledU: num_controls is already int, nothing to fold.

            if extra_kwargs:
                result_op = dataclasses.replace(result_op, **extra_kwargs)

        # theta is now part of operands, so the operands substitution
        # above already handles it.  No GateOperation-specific code needed.

        if changed:
            return dataclasses.replace(result_op, operands=new_operands)
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
                            uuid=v.uuid,
                        ).with_const(int(resolved))
                    )
                    list_changed = True
                else:
                    new_values.append(v)
        return new_values if list_changed else None
