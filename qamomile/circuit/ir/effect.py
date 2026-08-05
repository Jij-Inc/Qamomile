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

    An explicit implementation of the complete transform takes precedence.
    A controlled-inverse call then reuses explicit inverse-body metadata when
    generic lowering only needs to add controls; all other structural
    fallbacks conservatively inherit the direct body's effects.

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
        if implementation.transform is transform and implementation.body is not None
    )
    bodies: list["Block"] = []
    for implementation in matching_implementations:
        assert implementation.body is not None
        bodies.append(implementation.body)
    if any(
        implementation.backend is None and implementation.strategy is None
        for implementation in matching_implementations
    ):
        return tuple(bodies)
    if transform is CallTransform.CONTROLLED_INVERSE:
        inverse_implementations = tuple(
            implementation
            for implementation in definition.implementations
            if implementation.transform is CallTransform.INVERSE
            and implementation.body is not None
        )
        for implementation in inverse_implementations:
            assert implementation.body is not None
            bodies.append(implementation.body)
        if any(
            implementation.backend is None and implementation.strategy is None
            for implementation in inverse_implementations
        ):
            return tuple(bodies)
    if definition.body is not None:
        bodies.append(definition.body)
    return tuple(bodies)


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
    effects = KernelEffect.NONE
    for body in _operation_owned_effect_bodies(operation):
        effects |= body.effects
    return effects


def _operation_owned_effect_bodies(operation: Operation) -> tuple["Block", ...]:
    """Return callable bodies whose effects are inherited by an operation.

    This structural helper never reads a body's cached effects, so it can also
    discover recursive callable graphs before fixed-point evaluation begins.

    Args:
        operation (Operation): Semantic operation to inspect.

    Returns:
        tuple[Block, ...]: Referenced or owned bodies relevant to the
            operation's effect contract.
    """
    if isinstance(operation, InvokeOperation):
        if operation.definition is None:
            return ()
        return callable_bodies(operation.definition, operation.transform)
    if isinstance(operation, ControlledUOperation) and operation.block is not None:
        return (operation.block,)
    if isinstance(operation, InverseBlockOperation):
        if operation.implementation_block is not None:
            return (operation.implementation_block,)
        if operation.source_block is not None:
            return (operation.source_block,)
    if isinstance(operation, SelectOperation):
        return tuple(operation.case_blocks)
    return ()


def _reachable_effect_blocks(root: "Block") -> tuple["Block", ...]:
    """Collect the finite block graph participating in ``root`` effects.

    Args:
        root (Block): Block whose reachable callable graph should be walked.

    Returns:
        tuple[Block, ...]: Strongly referenced blocks in deterministic preorder.
    """
    blocks: dict[int, "Block"] = {}
    pending = [root]
    while pending:
        block = pending.pop()
        identity = id(block)
        if identity in blocks:
            continue
        blocks[identity] = block
        children = [
            child
            for operation in walk_operations(block.operations)
            for child in _operation_owned_effect_bodies(operation)
        ]
        pending.extend(reversed(children))
    return tuple(blocks.values())


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
        indices = operation.measurement_result_indices
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
    """Refresh reachable effect metadata as a least fixed point.

    Recursive and mutually recursive callables are valid serialized IR. Their
    semantic effects therefore cannot be populated with an ordinary recursive
    cache: a cycle would expose and then persist a partial ``NONE`` result.
    The effect and measured-output lattices are finite, so this routine starts
    every reachable body at the empty summary and repeatedly applies the
    ordinary local equations until no flag or result index grows.

    Args:
        block (Block): Mutable semantic block whose operations are finalized.
    """
    if block._effects_refreshing:
        return
    reachable = _reachable_effect_blocks(block)
    previous = {
        id(candidate): (
            candidate._effects,
            candidate._measurement_result_indices,
            candidate._effects_valid,
            candidate._effects_refreshing,
        )
        for candidate in reachable
    }
    for candidate in reachable:
        candidate._effects = KernelEffect.NONE
        candidate._measurement_result_indices = frozenset()
        candidate._effects_valid = True
        candidate._effects_refreshing = True
    try:
        while True:
            summaries = {
                id(candidate): summarize_block_effects(
                    candidate.operations,
                    candidate.output_values,
                )
                for candidate in reachable
            }
            changed = False
            for candidate in reachable:
                effects, result_indices = summaries[id(candidate)]
                effects |= candidate._effects
                result_indices |= candidate._measurement_result_indices
                if (
                    effects != candidate._effects
                    or result_indices != candidate._measurement_result_indices
                ):
                    changed = True
                candidate._effects = effects
                candidate._measurement_result_indices = result_indices
            if not changed:
                break
    except Exception:
        for candidate in reachable:
            (
                candidate._effects,
                candidate._measurement_result_indices,
                candidate._effects_valid,
                candidate._effects_refreshing,
            ) = previous[id(candidate)]
        raise
    else:
        for candidate in reachable:
            candidate._effects_valid = True
            candidate._effects_refreshing = False


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
