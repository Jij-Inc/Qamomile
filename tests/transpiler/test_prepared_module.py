"""Tests for program-level semantic preparation before target lowering."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    InvokeOperation,
)
from qamomile.circuit.transpiler.artifact import (
    CompilationMetadata,
    CompiledProgram,
)
from qamomile.circuit.transpiler.compiler import QamomileCompiler
from qamomile.circuit.transpiler.errors import CallableDefinitionConflictError
from qamomile.circuit.transpiler.prepared import PreparedModule, prepare_module
from qamomile.hugr.lowerer import HugrTarget
from qamomile.qiskit import QiskitTranspiler


@qmc.composite_gate(name="prepared_bell")
def _prepared_bell(
    left: qmc.Qubit,
    right: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Build a reusable Bell-state operation."""
    left = qmc.h(left)
    return qmc.cx(left, right)


@qmc.qkernel
def _prepared_entrypoint() -> tuple[qmc.Bit, qmc.Bit]:
    """Invoke a boxed callable from the preparation test entrypoint."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left, right = _prepared_bell(left, right)
    return qmc.measure(left), qmc.measure(right)


class _MutatingTarget:
    """Mutate its owned semantic input to test compiler isolation."""

    @property
    def name(self) -> str:
        """Return the synthetic target name.

        Returns:
            str: Stable test target name.
        """
        return "mutating-test"

    def plan(self, program: PreparedModule) -> None:
        """Clear nested semantic state inside the target-owned snapshot.

        Args:
            program (PreparedModule): Target-owned prepared snapshot.
        """
        program.entrypoint.operations.clear()
        for definition in program.definitions.values():
            if definition.body is not None:
                definition.body.operations.clear()

    def compile(
        self,
        program: PreparedModule,
        plan: None,
    ) -> CompiledProgram[object]:
        """Package a synthetic artifact after mutation.

        Args:
            program (PreparedModule): Mutated target-owned prepared snapshot.
            plan (None): Synthetic empty plan.

        Returns:
            CompiledProgram[object]: Synthetic compiled artifact.
        """
        del plan
        return CompiledProgram(
            artifact=object(),
            abi=program.abi,
            metadata=CompilationMetadata(self.name, "test"),
        )

    def validate(self, artifact: object) -> None:
        """Accept the synthetic artifact.

        Args:
            artifact (object): Synthetic artifact to discard.
        """
        del artifact


def test_prepare_preserves_hierarchy_and_collects_call_graph() -> None:
    """Preparation retains calls while collecting their definitions."""
    prepared = QiskitTranspiler().prepare(_prepared_entrypoint)

    assert prepared.entrypoint.kind is BlockKind.HIERARCHICAL
    invokes = [
        operation
        for operation in prepared.entrypoint.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert len(invokes) == 1
    call = invokes[0]
    assert call.target in prepared.definitions
    assert call.target in prepared.call_graph[prepared.entrypoint_ref]
    assert prepared.body(call.target) is _prepared_bell.block


def test_prepare_exposes_program_abi_before_segmentation() -> None:
    """Preparation records public outputs without creating a ProgramPlan."""
    prepared = QiskitTranspiler().prepare(_prepared_entrypoint)

    assert prepared.abi.public_inputs == {}
    assert prepared.abi.output_refs == [
        value.uuid for value in prepared.entrypoint.output_values
    ]


def test_prepare_definition_mapping_is_read_only() -> None:
    """Target lowering cannot mutate the prepared definition registry."""
    prepared = QiskitTranspiler().prepare(_prepared_entrypoint)

    with pytest.raises(TypeError):
        prepared.definitions[prepared.entrypoint_ref] = next(  # type: ignore[index]
            iter(prepared.definitions.values())
        )


def test_prepared_snapshot_isolates_nested_semantic_mutation() -> None:
    """A target-owned snapshot cannot mutate the shared prepared module."""
    prepared = QiskitTranspiler().prepare(_prepared_entrypoint)
    snapshot = prepared.owned_snapshot()
    original_operation_count = len(prepared.entrypoint.operations)

    snapshot.entrypoint.operations.clear()
    snapshot_definition = next(iter(snapshot.definitions.values()))
    assert snapshot_definition.body is not None
    snapshot_definition.body.operations.clear()

    assert len(prepared.entrypoint.operations) == original_operation_count
    original_definition = next(iter(prepared.definitions.values()))
    assert original_definition.body is not None
    assert original_definition.body.operations


def test_compiler_gives_each_target_an_owned_semantic_snapshot() -> None:
    """Target mutation during compilation cannot change the cached qkernel IR."""
    original_operation_count = len(_prepared_entrypoint.block.operations)
    original_body_count = len(_prepared_bell.block.operations)

    QamomileCompiler().compile(_prepared_entrypoint, _MutatingTarget())

    assert len(_prepared_entrypoint.block.operations) == original_operation_count
    assert len(_prepared_bell.block.operations) == original_body_count


def test_prepare_collects_edges_for_shared_body_under_each_owner() -> None:
    """Call-graph collection keys body visits by symbol as well as identity."""
    leaf_ref = CallableRef("test", "leaf")
    leaf_definition = CallableDef(ref=leaf_ref, body=Block(name="leaf"))
    shared_body = Block(
        name="shared",
        operations=[InvokeOperation(definition=leaf_definition)],
    )
    left_definition = CallableDef(
        ref=CallableRef("test", "left"),
        body=shared_body,
    )
    right_definition = CallableDef(
        ref=CallableRef("test", "right"),
        body=shared_body,
    )
    entrypoint = Block(
        name="entrypoint",
        operations=[
            InvokeOperation(definition=left_definition),
            InvokeOperation(definition=right_definition),
        ],
    )

    prepared = prepare_module(entrypoint)

    assert prepared.call_graph[left_definition.ref] == frozenset({leaf_ref})
    assert prepared.call_graph[right_definition.ref] == frozenset({leaf_ref})


def test_prepare_records_variants_for_target_specific_resolution() -> None:
    """Preparation retains same-symbol bodies for the target to resolve."""
    shared_ref = CallableRef("test", "shared")
    left_definition = CallableDef(ref=shared_ref, body=Block(name="left"))
    right_definition = CallableDef(ref=shared_ref, body=Block(name="right"))
    entrypoint = Block(
        name="entrypoint",
        operations=[
            InvokeOperation(definition=left_definition),
            InvokeOperation(definition=right_definition),
        ],
    )

    prepared = prepare_module(entrypoint)

    assert len(prepared.definition_variants[shared_ref]) == 2
    with pytest.raises(CallableDefinitionConflictError, match="test.shared"):
        HugrTarget().plan(prepared)


def test_existing_transpile_pipeline_uses_prepared_entrypoint() -> None:
    """The circuit pipeline still executes after preparation extraction."""
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(_prepared_entrypoint)
    result = executable.sample(transpiler.executor(), shots=16).result()

    assert sum(count for _, count in result.results) == 16
    assert {bits for bits, _ in result.results} <= {(0, 0), (1, 1)}
