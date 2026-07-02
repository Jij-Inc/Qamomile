"""Unified block representation for all pipeline stages."""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from qamomile.circuit.ir.parameter import ParamSlot
from qamomile.circuit.ir.value import Value, ValueLike

if TYPE_CHECKING:
    from qamomile.circuit.ir.operation import Operation
    from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation


class BlockKind(Enum):
    """Classification of block structure for pipeline stages."""

    TRACED = auto()  # Direct output of frontend tracing / build()
    HIERARCHICAL = auto()  # May contain CallBlockOperations
    AFFINE = auto()  # No CallBlockOperations, For/If preserved
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

    def __post_init__(self):
        """Validate label_args / input_values agreement and param_slots disjointness.

        Raises:
            ValueError: If ``label_args`` is non-empty and its length
                does not match ``input_values``, or if any
                ``ParamSlot.name`` appears more than once across
                ``param_slots``.
        """
        if self.label_args and len(self.label_args) != len(self.input_values):
            raise ValueError(
                f"label_args length ({len(self.label_args)}) must match "
                f"input_values length ({len(self.input_values)})"
            )
        if self.param_slots:
            seen: set[str] = set()
            for slot in self.param_slots:
                if slot.name in seen:
                    raise ValueError(
                        f"Duplicate ParamSlot name {slot.name!r} in Block.param_slots; "
                        f"every classical kernel argument may appear at most once."
                    )
                seen.add(slot.name)

    def unbound_parameters(self) -> list[str]:
        """Return list of unbound parameter names."""
        return list(self.parameters.keys())

    def is_affine(self) -> bool:
        """Check if block contains no CallBlockOperations."""
        return self.kind in (BlockKind.AFFINE, BlockKind.ANALYZED)

    def call(self, **kwargs: ValueLike) -> "CallBlockOperation":
        """Create a CallBlockOperation against this block."""
        from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation

        inputs = [kwargs[label] for label in self.label_args]
        dummy_inputs = {v.logical_id: idx for idx, v in enumerate(self.input_values)}

        results = []
        for dummy_return in self.output_values:
            if dummy_return.logical_id in dummy_inputs:
                input_idx = dummy_inputs[dummy_return.logical_id]
                results.append(inputs[input_idx].next_version())
            else:
                results.append(dummy_return)

        return CallBlockOperation(
            block=self,
            operands=cast(list[Value], inputs),
            results=cast(list[Value], results),
        )
