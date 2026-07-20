"""Unified block representation for all pipeline stages."""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import TYPE_CHECKING

from qamomile.circuit.ir.parameter import ParamSlot
from qamomile.circuit.ir.value import Value, ValueLike

if TYPE_CHECKING:
    from qamomile.circuit.ir.effect import KernelEffect
    from qamomile.circuit.ir.operation import Operation
    from qamomile.circuit.ir.operation.callable import InvokeOperation


def _empty_kernel_effect() -> "KernelEffect":
    """Return the empty effect set without introducing an import cycle.

    Returns:
        KernelEffect: ``KernelEffect.NONE``.
    """
    from qamomile.circuit.ir.effect import KernelEffect

    return KernelEffect.NONE


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

    # Derived semantic metadata. The cache is populated on first access, so
    # compiler passes that construct transient Blocks do not repeatedly scan
    # bodies whose effects they never inspect. ``dataclasses.replace`` resets
    # init=False fields and therefore invalidates the cache automatically.
    _effects: "KernelEffect" = dataclasses.field(
        default_factory=_empty_kernel_effect,
        init=False,
        repr=False,
    )
    _measurement_result_indices: frozenset[int] = dataclasses.field(
        default_factory=frozenset,
        init=False,
        repr=False,
    )
    _effects_valid: bool = dataclasses.field(default=False, init=False, repr=False)
    _effects_refreshing: bool = dataclasses.field(
        default=False,
        init=False,
        repr=False,
    )

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

    def _ensure_effects(self) -> None:
        """Populate derived effect metadata when it is first requested."""
        if self._effects_valid or self._effects_refreshing:
            return
        from qamomile.circuit.ir.effect import refresh_block_effects

        refresh_block_effects(self)

    @property
    def effects(self) -> "KernelEffect":
        """Return lazily cached semantic effects for this block.

        Returns:
            KernelEffect: Aggregated measurement, reset, and feed-forward
                effects reachable from the block.
        """
        self._ensure_effects()
        return self._effects

    @property
    def measurement_result_indices(self) -> frozenset[int]:
        """Return public output positions derived from measurement.

        Returns:
            frozenset[int]: Indices of measurement-derived block outputs.
        """
        self._ensure_effects()
        return self._measurement_result_indices

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
