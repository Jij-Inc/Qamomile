"""Program-level semantic input for target-specific compilation pipelines."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.transpiler.segments import ProgramABI


@dataclasses.dataclass(frozen=True)
class PreparedModule:
    """Hold a prepared entrypoint and its reachable callable definitions.

    Args:
        entrypoint_ref (CallableRef): Stable symbol assigned to the program
            entrypoint.
        entrypoint (Block): Hierarchical semantic block for the entrypoint.
        definitions (Mapping[CallableRef, CallableDef]): Reachable callable
            definitions keyed by their stable symbols.
        call_graph (Mapping[CallableRef, frozenset[CallableRef]]): Directed
            caller-to-callee relation, including the entrypoint symbol.
        abi (ProgramABI): Classical public input and output contract.
    """

    entrypoint_ref: CallableRef
    entrypoint: Block
    definitions: Mapping[CallableRef, CallableDef]
    call_graph: Mapping[CallableRef, frozenset[CallableRef]]
    abi: ProgramABI

    def body(self, ref: CallableRef) -> Block:
        """Return the semantic body associated with a program symbol.

        Args:
            ref (CallableRef): Entrypoint or callable symbol to resolve.

        Returns:
            Block: Hierarchical semantic body for ``ref``.

        Raises:
            KeyError: If ``ref`` is neither the entrypoint nor a reachable
                body-backed callable definition.
        """
        if ref == self.entrypoint_ref:
            return self.entrypoint
        definition = self.definitions[ref]
        if definition.body is None:
            raise KeyError(f"Callable {ref.namespace}.{ref.name} has no body")
        return definition.body


def prepare_module(entrypoint: Block) -> PreparedModule:
    """Collect a hierarchical block into an immutable program-level view.

    The collector follows calls in nested control-flow regions and in every
    body carried by a callable definition. Definitions remain Qamomile
    semantic IR; this function does not inline, clone, or lower operations.

    Args:
        entrypoint (Block): Hierarchical entrypoint after target-independent
            frontend preparation.

    Returns:
        PreparedModule: Entrypoint, reachable definitions, call graph, and
            public ABI packaged for a target-specific compiler pipeline.
    """
    entrypoint_ref = CallableRef(
        namespace="qamomile.entrypoint",
        name=entrypoint.name or "main",
    )
    definitions: dict[CallableRef, CallableDef] = {}
    edges: dict[CallableRef, set[CallableRef]] = {entrypoint_ref: set()}
    visited_bodies: set[tuple[CallableRef, int]] = set()

    def visit_block(owner: CallableRef, block: Block) -> None:
        """Visit one callable body and collect its reachable calls.

        Args:
            owner (CallableRef): Symbol whose body is being traversed.
            block (Block): Semantic body to inspect recursively.
        """
        body_identity = (owner, id(block))
        if body_identity in visited_bodies:
            return
        visited_bodies.add(body_identity)
        edges.setdefault(owner, set())
        visit_operations(owner, block.operations)

    def visit_definition(definition: CallableDef) -> None:
        """Register one callable definition and traverse all available bodies.

        Args:
            definition (CallableDef): Definition referenced by an invocation.
        """
        existing = definitions.get(definition.ref)
        if existing is None or (existing.body is None and definition.body is not None):
            definitions[definition.ref] = definition
        edges.setdefault(definition.ref, set())
        if definition.body is not None:
            visit_block(definition.ref, definition.body)
        for implementation in definition.implementations:
            if implementation.body is not None:
                visit_block(definition.ref, implementation.body)

    def visit_operations(owner: CallableRef, operations: list[Operation]) -> None:
        """Visit operations, nested regions, and callable definitions.

        Args:
            owner (CallableRef): Symbol containing ``operations``.
            operations (list[Operation]): Operation sequence to inspect.
        """
        for operation in operations:
            if isinstance(operation, InvokeOperation):
                edges.setdefault(owner, set()).add(operation.target)
                if operation.definition is not None:
                    visit_definition(operation.definition)
            if isinstance(operation, HasNestedOps):
                for nested in operation.nested_op_lists():
                    visit_operations(owner, nested)

    visit_block(entrypoint_ref, entrypoint)
    abi = ProgramABI(
        public_inputs=dict(entrypoint.parameters),
        output_refs=[value.uuid for value in entrypoint.output_values],
    )
    frozen_graph = {caller: frozenset(callees) for caller, callees in edges.items()}
    return PreparedModule(
        entrypoint_ref=entrypoint_ref,
        entrypoint=entrypoint,
        definitions=MappingProxyType(definitions),
        call_graph=MappingProxyType(frozen_graph),
        abi=abi,
    )
