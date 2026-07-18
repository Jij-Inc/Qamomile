"""Unified block representation for all pipeline stages."""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import TYPE_CHECKING

from qamomile.circuit.ir.parameter import ParamSlot
from qamomile.circuit.ir.static_binding import StaticBindingSlot
from qamomile.circuit.ir.value import Value, ValueLike, collect_value_like_uuids

if TYPE_CHECKING:
    from qamomile.circuit.ir.operation import Operation
    from qamomile.circuit.ir.operation.callable import InvokeOperation


class BlockKind(Enum):
    """Classification of block structure for pipeline stages."""

    TRACED = auto()  # Direct output of frontend tracing / build()
    HIERARCHICAL = auto()  # May contain inline callable invocations
    AFFINE = auto()  # No inline callable invocations, For/If preserved
    ANALYZED = auto()  # Validated and dependency-analyzed


@dataclasses.dataclass
class Block:
    """Unified block representation for all pipeline stages.

    Replaces the older traced and callable IR wrappers with a single structure.
    The `kind` field indicates which pipeline stage this block is at.
    """

    name: str = ""
    label_args: list[str] = dataclasses.field(default_factory=list)
    input_values: list[ValueLike] = dataclasses.field(default_factory=list)
    output_values: list[ValueLike] = dataclasses.field(default_factory=list)
    output_names: list[str] = dataclasses.field(default_factory=list)
    operations: list["Operation"] = dataclasses.field(default_factory=list)

    # Pipeline stage indicator
    kind: BlockKind = BlockKind.HIERARCHICAL

    # Parameters (unbound values for circuit parameters)
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)

    # Per-classical-argument metadata describing the kernel's parameter
    # contract (name, type, runtime-or-bound kind, default, bound_value,
    # differentiability hint). Populated by the frontend (``func_to_block``
    # / ``QKernel.build``) and preserved by every pass. Empty for
    # synthetic blocks that have no Python-level classical interface
    # (e.g., nested composite-gate implementation blocks).
    param_slots: tuple[ParamSlot, ...] = dataclasses.field(default_factory=tuple)

    # Compile-time object dependencies that are deliberately absent from the
    # runtime/quantum Block ABI. Hierarchical qkernels may carry unresolved
    # slots; build() materializes and removes them before compilation.
    static_bindings: tuple[StaticBindingSlot, ...] = dataclasses.field(
        default_factory=tuple
    )

    def __post_init__(self):
        """Validate the block interface and compile-time manifests.

        Raises:
            ValueError: If labels and inputs disagree, names or field UUIDs
                overlap, parameter slots are duplicated, or a static binding
                is malformed or attached to a non-hierarchical block.
        """
        if self.label_args and len(self.label_args) != len(self.input_values):
            raise ValueError(
                f"label_args length ({len(self.label_args)}) must match "
                f"input_values length ({len(self.input_values)})"
            )
        seen: set[str] = set(self.label_args)
        if len(seen) != len(self.label_args):
            raise ValueError("Block.label_args must contain unique names")
        seen_param_slots: set[str] = set()
        if self.param_slots:
            for slot in self.param_slots:
                if slot.name in seen_param_slots:
                    raise ValueError(
                        f"Duplicate ParamSlot name {slot.name!r} in Block.param_slots; "
                        f"every classical kernel argument may appear at most once."
                    )
                seen_param_slots.add(slot.name)
        formal_uuids: set[str] = set()
        for value in (*self.input_values, *self.parameters.values()):
            formal_uuids.update(collect_value_like_uuids(value))
        static_names: set[str] = set()
        static_field_uuids: set[str] = set()
        for slot in self.static_bindings:
            if self.kind is not BlockKind.HIERARCHICAL:
                raise ValueError(
                    "Static bindings are only valid on HIERARCHICAL Blocks."
                )
            if not slot.name or not slot.type_key:
                raise ValueError(
                    "Static binding names and type keys must be non-empty."
                )
            if slot.name in static_names:
                raise ValueError(
                    f"Duplicate static binding name {slot.name!r} in Block."
                )
            if slot.name in seen:
                raise ValueError(
                    f"Static binding {slot.name!r} must not appear in Block.label_args."
                )
            if slot.name in seen_param_slots or slot.name in self.parameters:
                raise ValueError(
                    f"Static binding {slot.name!r} overlaps a classical parameter."
                )
            field_names = [field.name for field in slot.fields]
            if len(set(field_names)) != len(field_names):
                raise ValueError(
                    f"Static binding {slot.name!r} has duplicate field names."
                )
            if any(
                not field.name
                or type(field.value) is not Value
                or field.value.parent_array is not None
                or field.value.element_indices
                or not field.value.type.is_classical()
                for field in slot.fields
            ):
                raise ValueError(
                    f"Static binding {slot.name!r} fields must be named "
                    "classical scalar Values."
                )
            for field in slot.fields:
                if field.value.uuid in formal_uuids:
                    raise ValueError(
                        f"Static binding {slot.name!r} field {field.name!r} "
                        "must not alias an ordinary Block formal."
                    )
                if field.value.uuid in static_field_uuids:
                    raise ValueError(
                        f"Static binding field UUID {field.value.uuid!r} is "
                        "used more than once."
                    )
                static_field_uuids.add(field.value.uuid)
            static_names.add(slot.name)

    def unbound_parameters(self) -> list[str]:
        """Return list of unbound parameter names."""
        return list(self.parameters.keys())

    def is_affine(self) -> bool:
        """Return whether this block has passed affine validation.

        Returns:
            bool: True for ``AFFINE`` and ``ANALYZED`` blocks.
        """
        return self.kind in (BlockKind.AFFINE, BlockKind.ANALYZED)

    def call(self, **kwargs: ValueLike) -> "InvokeOperation":
        """Create an inline callable invocation against this block.

        Args:
            **kwargs (ValueLike): Actual argument values keyed by
                ``self.label_args``.

        Returns:
            InvokeOperation: Inline-policy invocation whose callable
                definition points at this block.

        Raises:
            KeyError: If a required label in ``self.label_args`` is missing
                from ``kwargs``.
        """
        from qamomile.circuit.ir.operation.callable import (
            CallableDef,
            CallableRef,
            CallPolicy,
            InvokeOperation,
            block_call_operands_and_results,
            signature_from_block,
        )

        inputs, results = block_call_operands_and_results(self, kwargs)

        name = self.name or "anonymous"
        attrs = {"kind": "block", "default_policy": CallPolicy.INLINE.name}
        ref = CallableRef(namespace="user.block", name=name)
        return InvokeOperation(
            operands=inputs,
            results=results,
            target=ref,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                signature=signature_from_block(self),
                body=self,
                default_policy=CallPolicy.INLINE,
                attrs=attrs,
            ),
        )
