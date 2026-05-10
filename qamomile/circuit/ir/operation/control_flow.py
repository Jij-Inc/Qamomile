from __future__ import annotations

import dataclasses
import typing
from typing import cast

from qamomile.circuit.ir.types.primitives import BitType, BlockType, UIntType
from qamomile.circuit.ir.value import Value, ValueBase

from .operation import Operation, OperationKind, ParamHint, Signature

if typing.TYPE_CHECKING:
    from .arithmetic_operations import PhiOp


class HasNestedOps:
    """Mixin for operations that contain nested operation lists.

    Subclasses implement ``nested_op_lists()`` and ``rebuild_nested()``
    so that generic passes can recurse into control flow without
    isinstance chains.
    """

    def nested_op_lists(self) -> list[list[Operation]]:
        """Return all nested operation lists in this control flow op."""
        raise NotImplementedError

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        """Return a copy with nested operation lists replaced.

        ``new_lists`` must have the same length/order as ``nested_op_lists()``.
        """
        raise NotImplementedError


@dataclasses.dataclass
class WhileOperation(HasNestedOps, Operation):
    """Represents a while loop operation.

    Only measurement-backed conditions are supported: the condition must
    be a ``Bit`` value produced by ``qmc.measure()``.  Non-measurement
    conditions (classical variables, constants, comparisons) are rejected
    by ``ValidateWhileContractPass`` before reaching backend emit.

    Example::

        bit = qmc.measure(q)
        while bit:
            q = qmc.h(q)
            bit = qmc.measure(q)

    Attributes:
        operations: List of operations in the loop body.
        operands[0]: Initial condition (required). Must be a measurement
            result (``Bit`` from ``qmc.measure()``).
        operands[1]: Loop-carried condition (optional). When the loop body
            reassigns the condition variable (e.g., ``bit = qmc.measure(q)``),
            this captures the updated handle so that the transpiler can alias
            both UUIDs to the same classical bit during resource allocation.
            Without this alias, SSA-like value versioning would allocate a
            separate clbit for the body measurement, causing the while
            condition to never update (infinite loop on Qiskit) or read from
            the wrong qubit (wrong return value on QURI Parts).
    """

    operations: list[Operation] = dataclasses.field(default_factory=list)
    max_iterations: int | None = None

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint("condition", BlockType()),
                ParamHint("loop_carried", BlockType()),
            ],
            results=[],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class ForOperation(HasNestedOps, Operation):
    """Represents a for loop operation.

    Example:
        for i in range(start, stop, step):
            body

    Attributes:
        loop_var: Display name of the loop variable (e.g., "i"). After
            the UUID-keyed migration this is **display-only**: it appears
            in IR printers, error messages, and the legacy name-fallback
            ``bindings`` view, but never participates in identity or
            lookup. Identity is carried by ``loop_var_value.uuid``.
        loop_var_value: The loop variable's IR ``Value`` (typically a
            ``UInt``). Carries the UUID that identifies this loop
            variable across the IR — the binding written by the emit
            pass and the binding read inside the loop body both key on
            ``loop_var_value.uuid``. ``None`` only for legacy IR
            constructed before the migration; new construction must
            always provide it.
        operations: List of operations in the loop body
        operands[0]: start (UInt type)
        operands[1]: stop (UInt type)
        operands[2]: step (UInt type)
    """

    loop_var: str = ""
    loop_var_value: Value | None = None
    operations: list[Operation] = dataclasses.field(default_factory=list)

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    def all_input_values(self) -> list[ValueBase]:
        """Include ``loop_var_value`` so cloning/substitution stays consistent.

        Without this override, ``UUIDRemapper`` would clone every body
        reference to the loop variable to a fresh UUID, but leave
        ``loop_var_value`` pointing at the un-cloned original — emit-time
        UUID-keyed lookups for the loop variable would then miss.
        """
        values = super().all_input_values()
        if self.loop_var_value is not None:
            values.append(self.loop_var_value)
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        result = super().replace_values(mapping)
        assert isinstance(result, ForOperation)
        if result.loop_var_value is not None and result.loop_var_value.uuid in mapping:
            mapped = mapping[result.loop_var_value.uuid]
            if isinstance(mapped, Value):
                result = dataclasses.replace(result, loop_var_value=mapped)
        return result

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint("start", UIntType()),
                ParamHint("stop", UIntType()),
                ParamHint("step", UIntType()),
            ],
            results=[],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class ForItemsOperation(HasNestedOps, Operation):
    """Represents iteration over dict/iterable items.

    Example:
        for (i, j), Jij in qmc.items(ising):
            body

    Attributes:
        key_vars: Display names of key unpacking variables (e.g.,
            ``["i", "j"]``). Display-only; identity comes from
            ``key_var_values``.
        value_var: Display name of value variable. Display-only.
        key_var_values: IR ``Value`` objects for the key variables. For
            scalar keys, one entry per ``key_vars`` entry. For
            ``key_is_vector=True`` (Vector keys), a single
            ``ArrayValue``-typed entry. Carries the UUIDs used to bind
            keys at emit time. ``None`` only for legacy IR.
        value_var_value: IR ``Value`` for the value variable. ``None``
            only for legacy IR.
        operations: List of operations in the loop body.
        operands[0]: The dict/iterable value (DictValue type).

    Note:
        This operation is always unrolled at transpile time since quantum
        backends cannot natively iterate over classical data structures.
    """

    key_vars: list[str] = dataclasses.field(default_factory=list)
    value_var: str = ""
    key_is_vector: bool = False
    key_var_values: tuple[Value, ...] | None = None
    value_var_value: Value | None = None
    operations: list[Operation] = dataclasses.field(default_factory=list)

    def nested_op_lists(self) -> list[list[Operation]]:
        return [self.operations]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(self, operations=new_lists[0])

    def all_input_values(self) -> list[ValueBase]:
        """Include the per-key/value ``Value`` fields for cloning/substitution.

        Same rationale as ``ForOperation.all_input_values``: keep the IR
        identity fields in lockstep with body references so UUID-keyed
        lookups stay valid after inline cloning.
        """
        values = super().all_input_values()
        if self.key_var_values is not None:
            values.extend(self.key_var_values)
        if self.value_var_value is not None:
            values.append(self.value_var_value)
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        result = super().replace_values(mapping)
        assert isinstance(result, ForItemsOperation)
        # Substitute key_var_values element-wise.
        if result.key_var_values is not None:
            new_keys: list[Value] = []
            keys_changed = False
            for kv in result.key_var_values:
                if kv.uuid in mapping:
                    mapped = mapping[kv.uuid]
                    if isinstance(mapped, Value):
                        new_keys.append(mapped)
                        keys_changed = True
                        continue
                new_keys.append(kv)
            if keys_changed:
                result = dataclasses.replace(result, key_var_values=tuple(new_keys))
        # Substitute value_var_value.
        if (
            result.value_var_value is not None
            and result.value_var_value.uuid in mapping
        ):
            mapped = mapping[result.value_var_value.uuid]
            if isinstance(mapped, Value):
                result = dataclasses.replace(result, value_var_value=mapped)
        return result

    @property
    def signature(self) -> Signature:
        # Signature is flexible - operand is the dict/iterable being iterated
        return Signature(
            operands=[ParamHint("iterable", BlockType())],  # BlockType as placeholder
            results=[],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL


@dataclasses.dataclass
class IfOperation(HasNestedOps, Operation):
    """Represents an if-else conditional operation.

    Example:
        if condition:
            true_body
        else:
            false_body

    Attributes:
        true_operations: List of operations in the true branch
        false_operations: List of operations in the false branch (may be empty)
        phi_ops: List of PhiOp instances merging values from both branches
        operands[0]: condition (Bit type from measurement or comparison)
        results: Phi-merged output values from both branches
    """

    true_operations: list[Operation] = dataclasses.field(default_factory=list)
    false_operations: list[Operation] = dataclasses.field(default_factory=list)
    phi_ops: list[PhiOp] = dataclasses.field(default_factory=list)

    def nested_op_lists(self) -> list[list[Operation]]:
        return [
            self.true_operations,
            self.false_operations,
            cast(list[Operation], self.phi_ops),
        ]

    def rebuild_nested(self, new_lists: list[list[Operation]]) -> Operation:
        return dataclasses.replace(
            self,
            true_operations=new_lists[0],
            false_operations=new_lists[1],
            phi_ops=cast("list[PhiOp]", new_lists[2]),
        )

    @property
    def condition(self) -> Value:
        return self.operands[0]

    @property
    def signature(self) -> Signature:
        result_hints = [
            ParamHint(name=f"result_{i}", type=r.type)
            for i, r in enumerate(self.results)
        ]
        return Signature(
            operands=[ParamHint("condition", BitType())], results=result_hints
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CONTROL
