"""First-class semantic effects for qkernel bodies and invocations."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING

from qamomile.circuit.ir.dataflow import (
    build_dependency_graph,
    find_measurement_derived_values,
    find_measurement_results,
    walk_operations,
)
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import CallTransform, InvokeOperation
from qamomile.circuit.ir.operation.control_flow import IfOperation, WhileOperation
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.select import SelectOperation

if TYPE_CHECKING:
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.operation.callable import CallableDef


class KernelEffect(enum.Flag):
    """Describe non-unitary behavior reachable from a kernel body.

    ``KernelEffect.NONE`` is the empty effect set and denotes unitary behavior.
    Flags compose with bitwise union so one kernel can expose measurement,
    reset, and measurement-backed feed-forward together.
    """

    NONE = 0
    MEASUREMENT = enum.auto()
    RESET = enum.auto()
    FEED_FORWARD = enum.auto()

    @property
    def is_unitary(self) -> bool:
        """Return whether this is the empty effect set.

        Returns:
            bool: ``True`` only for ``KernelEffect.NONE``.
        """
        return self == KernelEffect.NONE

    def labels(self) -> tuple[str, ...]:
        """Return stable effect names for diagnostics and serialization.

        Returns:
            tuple[str, ...]: Active flag names in declaration order.
        """
        return tuple(
            effect.name
            for effect in (
                KernelEffect.MEASUREMENT,
                KernelEffect.RESET,
                KernelEffect.FEED_FORWARD,
            )
            if effect in self and effect.name is not None
        )


def callable_bodies(
    definition: "CallableDef",
    transform: CallTransform,
) -> tuple["Block", ...]:
    """Return cached semantic bodies relevant to one call transform.

    An explicit transform implementation takes precedence. Otherwise inverse
    and controlled calls conservatively inherit the direct body's effects,
    which keeps the API ready for a future explicit controlled implementation
    without treating today's generic structural fallback as unitary.

    Args:
        definition (CallableDef): Callable definition referenced by a call.
        transform (CallTransform): Requested direct, inverse, or controlled
            transform.

    Returns:
        tuple[Block, ...]: Candidate bodies whose cached metadata applies.
    """
    matching_implementations = tuple(
        implementation
        for implementation in definition.implementations
        if implementation.transform is transform
    )
    if matching_implementations:
        return tuple(
            implementation.body
            for implementation in matching_implementations
            if implementation.body is not None
        )
    if definition.body is not None:
        return (definition.body,)
    return ()


def callable_effects(
    definition: "CallableDef | None",
    transform: CallTransform = CallTransform.DIRECT,
) -> KernelEffect:
    """Return cached effects for a callable invocation.

    Args:
        definition (CallableDef | None): Referenced callable definition.
        transform (CallTransform): Requested call transform. Defaults to
            ``CallTransform.DIRECT``.

    Returns:
        KernelEffect: Union of the relevant cached body effects.
    """
    if definition is None:
        return KernelEffect.NONE
    effects = KernelEffect.NONE
    for body in callable_bodies(definition, transform):
        effects |= body.effects
    return effects


def callable_measurement_result_indices(
    definition: "CallableDef | None",
    transform: CallTransform = CallTransform.DIRECT,
) -> frozenset[int]:
    """Return callable result positions carrying measurement provenance.

    Args:
        definition (CallableDef | None): Referenced callable definition.
        transform (CallTransform): Requested call transform. Defaults to
            ``CallTransform.DIRECT``.

    Returns:
        frozenset[int]: Result positions derived from measurement in any
            applicable body.
    """
    if definition is None:
        return frozenset()
    indices: set[int] = set()
    for body in callable_bodies(definition, transform):
        indices.update(body.measurement_result_indices)
    return frozenset(indices)


def _operation_owned_effects(operation: Operation) -> KernelEffect:
    """Return cached effects of operation-owned callable bodies.

    Args:
        operation (Operation): Semantic operation to inspect.

    Returns:
        KernelEffect: Effects inherited from owned or referenced bodies.
    """
    if isinstance(operation, InvokeOperation):
        return callable_effects(operation.definition, operation.transform)
    if isinstance(operation, ControlledUOperation) and operation.block is not None:
        return operation.block.effects
    if isinstance(operation, InverseBlockOperation):
        if operation.implementation_block is not None:
            return operation.implementation_block.effects
        if operation.source_block is not None:
            return operation.source_block.effects
    if isinstance(operation, SelectOperation):
        effects = KernelEffect.NONE
        for case_block in operation.case_blocks:
            effects |= case_block.effects
        return effects
    return KernelEffect.NONE


def _invocation_measurement_seeds(operations: Sequence[Operation]) -> set[str]:
    """Return caller result UUIDs mapped from measured callable outputs.

    Args:
        operations (Sequence[Operation]): Top-level semantic operations.

    Returns:
        set[str]: Invocation-result UUIDs with measurement provenance.
    """
    seeds: set[str] = set()
    for operation in walk_operations(operations):
        if not isinstance(operation, InvokeOperation):
            continue
        indices = callable_measurement_result_indices(
            operation.definition,
            operation.transform,
        )
        seeds.update(
            operation.results[index].uuid
            for index in indices
            if index < len(operation.results)
        )
    return seeds


def summarize_block_effects(
    operations: Sequence[Operation],
    output_values: Sequence[object],
) -> tuple[KernelEffect, frozenset[int]]:
    """Summarize kernel effects and measurement-derived public outputs.

    Args:
        operations (Sequence[Operation]): Block operation tree.
        output_values (Sequence[object]): Ordered block output values.

    Returns:
        tuple[KernelEffect, frozenset[int]]: Aggregated effects and output
            positions carrying measurement provenance.
    """
    effects = KernelEffect.NONE
    for operation in walk_operations(operations):
        if isinstance(operation, ResetOperation):
            effects |= KernelEffect.RESET
        effects |= _operation_owned_effects(operation)

    measurement_seeds = find_measurement_results(operations)
    measurement_seeds.update(_invocation_measurement_seeds(operations))
    if measurement_seeds:
        effects |= KernelEffect.MEASUREMENT

    derived = find_measurement_derived_values(
        build_dependency_graph(operations),
        measurement_seeds,
    )
    for operation in walk_operations(operations):
        if isinstance(operation, (IfOperation, WhileOperation)):
            condition = operation.operands[0] if operation.operands else None
            if getattr(condition, "uuid", None) in derived:
                effects |= KernelEffect.FEED_FORWARD

    measurement_outputs = frozenset(
        index
        for index, output in enumerate(output_values)
        if getattr(output, "uuid", None) in derived
    )
    return effects, measurement_outputs


def refresh_block_effects(block: "Block") -> None:
    """Refresh one block's cached effect and output-provenance metadata.

    Args:
        block (Block): Mutable semantic block whose operations are finalized.
    """
    if block._effects_refreshing:
        return
    block._effects_refreshing = True
    try:
        effects, result_indices = summarize_block_effects(
            block.operations,
            block.output_values,
        )
        block._effects = effects
        block._measurement_result_indices = result_indices
        block._effects_valid = True
    finally:
        block._effects_refreshing = False


def format_kernel_effects(effects: KernelEffect) -> str:
    """Format an effect set for deterministic user-facing diagnostics.

    Args:
        effects (KernelEffect): Effect set to format.

    Returns:
        str: Comma-separated flag names, or ``NONE`` for a unitary kernel.
    """
    labels = effects.labels()
    return ", ".join(labels) if labels else KernelEffect.NONE.name


def require_unitary_effects(
    effects: KernelEffect,
    *,
    operation: str,
    target: str,
    alternative: str,
) -> None:
    """Reject non-unitary effects with a uniform early diagnostic.

    Args:
        effects (KernelEffect): Cached target effects to validate.
        operation (str): User-facing meta-operation name.
        target (str): Target kernel or callable name.
        alternative (str): Actionable compatible API guidance.

    Raises:
        ValueError: If ``effects`` is not the empty unitary set.
    """
    if effects.is_unitary:
        return
    raise ValueError(
        f"{operation} cannot transform kernel {target!r} because it has "
        f"non-unitary kernel effects [{format_kernel_effects(effects)}]. "
        f"{alternative}"
    )
