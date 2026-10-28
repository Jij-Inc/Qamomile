"""Summarize runtime-observation provenance across selected call bodies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import TypeAlias

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.dataflow import (
    build_dependency_graph,
    find_measurement_derived_values,
    find_measurement_results,
    walk_operations,
)
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.operation import Operation

_RuntimeObservationSummary: TypeAlias = tuple[frozenset[int], bool]
_RuntimeObservationCacheEntry: TypeAlias = tuple[Block, frozenset[int], bool]
_RuntimeObservationCache: TypeAlias = MutableMapping[
    int,
    _RuntimeObservationCacheEntry,
]
_StrategyFor: TypeAlias = Callable[[InvokeOperation], str | None]


def _find_runtime_observation_results(
    operations: Sequence[Operation],
) -> set[str]:
    """Return classical results that must remain runtime-dependent.

    Expectation values behave like observations for estimator scheduling and
    branch specialization, but they deliberately are not sample-only
    ``KernelEffect.MEASUREMENT`` effects. Keeping the additional seeds local to
    the estimator avoids changing compiler effect validation. Invoke results
    are deliberately excluded because each caller adds provenance from the
    implementation selected by the current resource strategy.

    Args:
        operations (Sequence[Operation]): Operation tree to inspect.

    Returns:
        set[str]: Measurement, projection-bit, and expectation-result UUIDs.
    """
    results = find_measurement_results(operations)
    for operation in walk_operations(operations):
        if isinstance(operation, ExpvalOp):
            results.update(result.uuid for result in operation.results)
    return results


def _block_runtime_observation_summary(
    block: Block,
    *,
    strategy_for: _StrategyFor,
    cache: _RuntimeObservationCache,
) -> _RuntimeObservationSummary:
    """Summarize selected runtime observations as a least fixed point.

    Measurement provenance is cached in the IR, while expectation values
    intentionally are not a ``KernelEffect``. This estimator-local summary
    follows the same selected Invoke bodies used for resource evaluation.
    Recursive callable graphs start at the empty summary and iterate until no
    output provenance or observation flag grows, so a cycle never caches a
    partial result based on access order.

    Args:
        block (Block): Selected callable body to inspect.
        strategy_for (_StrategyFor): Stable run-scoped callback selecting the
            implementation strategy for each invocation.
        cache (_RuntimeObservationCache): Run-scoped identity cache that keeps
            strong references to previously summarized blocks.

    Returns:
        _RuntimeObservationSummary: Body output indices derived from a runtime
        observation and whether any observation occurs in the body.
    """
    cached = cache.get(id(block))
    if cached is not None and cached[0] is block:
        return cached[1], cached[2]

    reachable = _selected_runtime_observation_blocks(
        block,
        strategy_for=strategy_for,
    )
    summaries: dict[int, _RuntimeObservationSummary] = {
        id(candidate): (frozenset(), False) for candidate in reachable
    }
    while True:
        changed = False
        updated: dict[int, _RuntimeObservationSummary] = {}
        for candidate in reachable:
            output_indices, has_observation = _runtime_observation_equation(
                candidate,
                summaries,
                strategy_for=strategy_for,
            )
            previous_indices, previous_observation = summaries[id(candidate)]
            summary = (
                previous_indices | output_indices,
                previous_observation or has_observation,
            )
            updated[id(candidate)] = summary
            changed = changed or summary != summaries[id(candidate)]
        summaries = updated
        if not changed:
            break

    for candidate in reachable:
        output_indices, has_observation = summaries[id(candidate)]
        cache[id(candidate)] = (
            candidate,
            output_indices,
            has_observation,
        )
    return summaries[id(block)]


def _selected_runtime_observation_blocks(
    root: Block,
    *,
    strategy_for: _StrategyFor,
) -> tuple[Block, ...]:
    """Collect blocks reachable through selected Invoke implementations.

    Args:
        root (Block): Selected body at the root of the callable graph.
        strategy_for (_StrategyFor): Stable run-scoped callback selecting the
            implementation strategy for each invocation.

    Returns:
        tuple[Block, ...]: Strongly referenced blocks in deterministic preorder.
    """
    blocks: dict[int, Block] = {}
    pending = [root]
    while pending:
        block = pending.pop()
        identity = id(block)
        if identity in blocks:
            continue
        blocks[identity] = block
        children: list[Block] = []
        for operation in walk_operations(block.operations):
            if not isinstance(operation, InvokeOperation):
                continue
            selection = operation.select_body(strategy=strategy_for(operation))
            if isinstance(selection.body, Block):
                children.append(selection.body)
        pending.extend(reversed(children))
    return tuple(blocks.values())


def _runtime_observation_equation(
    block: Block,
    summaries: Mapping[int, _RuntimeObservationSummary],
    *,
    strategy_for: _StrategyFor,
) -> _RuntimeObservationSummary:
    """Apply one selected-observation equation to a block.

    Args:
        block (Block): Body whose local equation should be evaluated.
        summaries (Mapping[int, _RuntimeObservationSummary]): Previous
            fixed-point approximation for every reachable selected body.
        strategy_for (_StrategyFor): Stable run-scoped callback selecting the
            implementation strategy for each invocation.

    Returns:
        _RuntimeObservationSummary: Output provenance and observation flag
        derived in this iteration.
    """
    roots = _find_runtime_observation_results(block.operations)
    has_observation = bool(roots)
    for operation in walk_operations(block.operations):
        if not isinstance(operation, InvokeOperation):
            continue
        selection = operation.select_body(strategy=strategy_for(operation))
        if not isinstance(selection.body, Block):
            continue
        body_indices, nested_has_observation = summaries[id(selection.body)]
        mapped_indices = selection.map_result_indices(
            body_indices,
            operation.results,
        )
        roots.update(
            operation.results[index].uuid
            for index in mapped_indices
            if index < len(operation.results)
        )
        has_observation = has_observation or nested_has_observation

    derived = find_measurement_derived_values(
        build_dependency_graph(block.operations),
        roots,
    )
    derived.update(roots)
    return (
        frozenset(
            index
            for index, output in enumerate(block.output_values)
            if output.uuid in derived
        ),
        has_observation,
    )


def _invoke_runtime_observation_summary(
    operation: InvokeOperation,
    *,
    strategy_for: _StrategyFor,
    cache: _RuntimeObservationCache,
) -> _RuntimeObservationSummary:
    """Map a selected body's runtime observations to Invoke results.

    Args:
        operation (InvokeOperation): Callable invocation to inspect.
        strategy_for (_StrategyFor): Stable run-scoped callback selecting the
            implementation strategy for each invocation.
        cache (_RuntimeObservationCache): Run-scoped identity cache that keeps
            strong references to previously summarized blocks.

    Returns:
        _RuntimeObservationSummary: Caller result indices derived from an
        observation and whether the selected body contains an observation.
    """
    strategy = strategy_for(operation)
    selection = operation.select_body(strategy=strategy)
    body = selection.body
    if not isinstance(body, Block):
        indices = operation.measurement_result_indices_for(strategy=strategy)
        return indices, bool(indices)
    body_indices, has_observation = _block_runtime_observation_summary(
        body,
        strategy_for=strategy_for,
        cache=cache,
    )
    return (
        selection.map_result_indices(body_indices, operation.results),
        has_observation,
    )
