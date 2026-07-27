"""Materialize explicit capture lists on structured semantic IR regions."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    genuine_input_values,
)
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
)
from qamomile.circuit.transpiler.passes import Pass


class RegionCapturePass(Pass[Block, Block]):
    """Populate explicit captures for every structured control-flow region.

    The pass derives captures from the current semantic IR, so it can normalize
    hand-built and deserialized blocks as well as frontend output. The pass
    preserves existing block identity while replacing structured operations
    with capture-annotated values. Running it repeatedly is idempotent.
    """

    def __init__(self) -> None:
        """Initialize an empty reachable-block visitation set."""
        self._visited_blocks: set[int] = set()

    @property
    def name(self) -> str:
        """Return the compiler-visible pass name.

        Returns:
            str: Stable pass name used in diagnostics.
        """
        return "region_capture"

    def run(self, input: Block) -> Block:
        """Populate explicit captures throughout one block graph.

        Args:
            input (Block): Semantic entrypoint whose reachable regions should
                be normalized.

        Returns:
            Block: The input entrypoint with capture lists populated on every
                reachable structured-control operation.
        """
        self._visited_blocks = set()
        self._normalize_block(input)
        return input

    def _normalize_block(self, block: Block) -> None:
        """Normalize one block and every callable block it owns.

        Args:
            block (Block): Block to annotate in place.
        """
        identity = id(block)
        if identity in self._visited_blocks:
            return
        self._visited_blocks.add(identity)

        available: dict[str, ValueBase] = {}
        for value in block.input_values:
            self._register_value_graph(value, available)
        for value in block.parameters.values():
            self._register_value_graph(value, available)
        for slot in block.static_bindings:
            for field in slot.fields:
                self._register_value_graph(field.value, available)

        normalized: list[Operation] = []
        for operation in block.operations:
            operation = self._normalize_operation(operation, available)
            normalized.append(operation)
            for result in operation.results:
                self._register_value_graph(result, available)
            self._normalize_owned_blocks(operation)
        block.operations = normalized

    def _normalize_owned_blocks(self, operation: Operation) -> None:
        """Normalize callable and SELECT blocks owned by one operation.

        Args:
            operation (Operation): Operation whose nested block ownership
                should be traversed.
        """
        if isinstance(operation, InvokeOperation) and operation.definition is not None:
            definition = operation.definition
            if definition.body is not None:
                self._normalize_block(definition.body)
            for implementation in definition.implementations:
                if implementation.body is not None:
                    self._normalize_block(implementation.body)
        if isinstance(operation, SelectOperation):
            for case_block in operation.case_blocks:
                self._normalize_block(case_block)

    def _normalize_operation(
        self,
        operation: Operation,
        available: dict[str, ValueBase],
    ) -> Operation:
        """Normalize regions nested directly under one operation.

        Args:
            operation (Operation): Operation to inspect and possibly rebuild.
            available (dict[str, ValueBase]): Values defined in the enclosing
                scope before ``operation``.

        Returns:
            Operation: Original operation for a leaf, or a rebuilt structured
                operation with normalized bodies and capture lists.
        """
        if isinstance(operation, HasNestedOps):
            regions = []
            for region in operation.nested_regions():
                body, captures, yields = self._normalize_region(
                    list(region.operations),
                    available,
                    region.block_args,
                    region.captures,
                    region.yields,
                )
                regions.append(
                    dataclasses.replace(
                        region,
                        operations=tuple(body),
                        captures=captures,
                        yields=yields,
                    )
                )
            return operation.rebuild_regions(regions)
        return operation

    def _normalize_region(
        self,
        operations: list[Operation],
        outer_available: dict[str, ValueBase],
        block_args: Sequence[ValueBase],
        declared_captures: Sequence[ValueBase],
        yields: Sequence[ValueBase],
    ) -> tuple[
        list[Operation],
        tuple[ValueBase, ...],
        tuple[ValueBase, ...],
    ]:
        """Normalize one region and derive captures in first-use order.

        Args:
            operations (list[Operation]): Region body in program order.
            outer_available (dict[str, ValueBase]): Values visible immediately
                outside the region.
            block_args (list[ValueBase]): Values defined at region entry.
            declared_captures (Sequence[ValueBase]): Frontend or serialized
                capture declarations that may identify aggregate snapshots
                not materialized by a single producer operation.
            yields (list[ValueBase]): Values crossing the region exit.

        Returns:
            tuple[list[Operation], tuple[ValueBase, ...], tuple[ValueBase, ...]]:
                Rebuilt body, explicit capture sequence, and normalized yields.
        """
        local: dict[str, ValueBase] = {}
        for block_arg in block_args:
            self._register_value_graph(block_arg, local)
        scope = dict(outer_available)
        scope.update(local)
        captures: list[ValueBase] = []
        captured_ids: set[str] = set()
        declared_by_id = {value.uuid: value for value in declared_captures}
        normalized: list[Operation] = []

        for operation in operations:
            operation = self._normalize_operation(operation, scope)
            normalized.append(operation)
            for value in genuine_input_values(operation):
                self._record_capture(
                    value,
                    local,
                    outer_available,
                    declared_by_id,
                    captures,
                    captured_ids,
                )
            for result in operation.results:
                self._register_value_graph(result, local)
                self._register_value_graph(result, scope)
            self._normalize_owned_blocks(operation)
        normalized_yields: list[ValueBase] = []
        for value in yields:
            normalized_yields.append(value)
            self._record_capture(
                value,
                local,
                outer_available,
                declared_by_id,
                captures,
                captured_ids,
            )
        return normalized, tuple(captures), tuple(normalized_yields)

    @staticmethod
    def _matching_outer_value(
        value: ValueBase,
        outer_available: dict[str, ValueBase],
    ) -> ValueBase | None:
        """Resolve a branch-local handle copy to its outer SSA value.

        Branch tracing uses independent frontend handle copies so affine
        consumption in mutually exclusive branches is isolated. Legacy traces
        can therefore expose a fresh UUID at the first operation in a branch,
        even though its logical identity denotes an enclosing value. When
        several SSA versions of that identity dominate the region, insertion
        order reflects program order, so the latest compatible version is the
        region-entry value.

        Args:
            value (ValueBase): Candidate branch-entry operand.
            outer_available (dict[str, ValueBase]): Enclosing values.

        Returns:
            ValueBase | None: Latest type-compatible dominating value, or
                ``None``.
        """
        if value.uuid in outer_available:
            return outer_available[value.uuid]
        matches = [
            candidate
            for candidate in outer_available.values()
            if candidate.logical_id == value.logical_id and candidate.type == value.type
        ]
        return matches[-1] if matches else None

    @staticmethod
    def _record_capture(
        value: ValueBase,
        local: dict[str, ValueBase],
        outer_available: dict[str, ValueBase],
        declared_captures: dict[str, ValueBase],
        captures: list[ValueBase],
        captured_ids: set[str],
    ) -> None:
        """Append one non-local, non-constant value as a capture.

        Args:
            value (ValueBase): Candidate region input.
            local (dict[str, ValueBase]): Values defined inside the region or
                by its block arguments.
            outer_available (dict[str, ValueBase]): Values defined before the
                region in the enclosing scope.
            declared_captures (dict[str, ValueBase]): Existing capture values
                keyed by UUID, used to preserve frontend array snapshots.
            captures (list[ValueBase]): Ordered capture accumulator.
            captured_ids (set[str]): UUIDs already present in ``captures``.
        """
        if value.uuid in local or value.uuid in captured_ids or value.is_constant():
            return
        if value.uuid not in outer_available:
            declared = declared_captures.get(value.uuid)
            if isinstance(declared, ArrayValue):
                value = declared
            else:
                logical_match = RegionCapturePass._matching_outer_value(
                    value, outer_available
                )
                if logical_match is not None:
                    value = logical_match
                else:
                    root = value if isinstance(value, ArrayValue) else None
                    if isinstance(value, Value) and value.parent_array is not None:
                        root = value.parent_array
                    while root is not None and root.slice_of is not None:
                        root = root.slice_of
                    if root is None:
                        return
                    if root.uuid in declared_captures:
                        value = declared_captures[root.uuid]
                    elif root.uuid in outer_available:
                        value = outer_available[root.uuid]
                    else:
                        root_match = RegionCapturePass._matching_outer_value(
                            root, outer_available
                        )
                        if root_match is None:
                            return
                        value = root_match
        if value.uuid in captured_ids:
            return
        captures.append(value)
        captured_ids.add(value.uuid)

    @classmethod
    def _register_value_graph(
        cls,
        value: ValueBase,
        destination: dict[str, ValueBase],
    ) -> None:
        """Register one definition and its owned aggregate components.

        Array shape formals are owned by an array definition. Slice ancestry,
        element indices, and ``parent_array`` are dependencies; registering
        those would incorrectly make an enclosing array look region-local
        when an operation produces only one element version.

        Args:
            value (ValueBase): Defined value-like root to traverse.
            destination (dict[str, ValueBase]): Registry updated in place.
        """
        if value.uuid in destination:
            return
        destination[value.uuid] = value
        if isinstance(value, TupleValue):
            for element in value.elements:
                cls._register_value_graph(element, destination)
        elif isinstance(value, DictValue):
            for key, entry_value in value.entries:
                cls._register_value_graph(key, destination)
                cls._register_value_graph(entry_value, destination)
        elif isinstance(value, ArrayValue):
            # Shape formals are owned metadata of an array definition (for
            # example a symbolic ForItems vector key). Slice ancestry and
            # indices remain ordinary dependencies and are not registered.
            for dimension in value.shape:
                cls._register_value_graph(dimension, destination)
