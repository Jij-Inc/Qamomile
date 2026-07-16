"""Regression tests for compiler consumers of SELECT-owned case Blocks."""

from __future__ import annotations

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.canonical import content_hash
from qamomile.circuit.ir.operation import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types.primitives import FloatType, QubitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    CompileTimeIfLoweringPass,
)
from qamomile.circuit.transpiler.passes.inline import (
    InlinePass,
    count_inline_invokes,
)
from qamomile.circuit.transpiler.passes.partial_eval import PartialEvaluationPass
from qamomile.circuit.transpiler.passes.substitution import (
    SubstitutionConfig,
    SubstitutionPass,
    SubstitutionRule,
)
from qamomile.circuit.transpiler.prepared import prepare_module
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer


@qmc.qkernel
def _identity_case(q: qmc.Qubit) -> qmc.Qubit:
    """Return one target unchanged."""
    return q


@qmc.qkernel
def _leaf_x(q: qmc.Qubit) -> qmc.Qubit:
    """Apply X in the leaf case helper."""
    return qmc.x(q)


@qmc.qkernel
def _delegating_x_case(q: qmc.Qubit) -> qmc.Qubit:
    """Delegate a SELECT case to an inline-policy qkernel."""
    return _leaf_x(q)


@qmc.qkernel
def _select_with_delegating_case(
    index: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Embed a qkernel invocation in the selected case Block."""
    return qmc.select([_identity_case, _delegating_x_case])(index, target)


@qmc.qkernel
def _conditional_x_case(q: qmc.Qubit, selector: qmc.UInt) -> qmc.Qubit:
    """Apply X when the case-local selector is zero."""
    if selector == 0:
        q = qmc.x(q)
    return q


@qmc.qkernel
def _parameterized_identity_case(
    q: qmc.Qubit,
    selector: qmc.UInt,
) -> qmc.Qubit:
    """Accept the shared selector while preserving the target."""
    _ = selector
    return q


@qmc.qkernel
def _select_with_shadowed_selector(
    selector: qmc.UInt,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Pass literal zero despite an outer same-named parameter."""
    index = qmc.qubit("index")
    target = qmc.qubit("target")
    return qmc.select([_conditional_x_case, _parameterized_identity_case])(
        index,
        target,
        selector=0,
    )


@qmc.composite_gate(name="select_strategy_case")
def _strategy_case(q: qmc.Qubit) -> qmc.Qubit:
    """Provide a boxed case whose strategy can be substituted."""
    return qmc.x(q)


@qmc.qkernel
def _delegating_strategy_case(q: qmc.Qubit) -> qmc.Qubit:
    """Keep the boxed strategy call inside a qkernel case body."""
    return _strategy_case(q)


@qmc.qkernel
def _select_with_strategy_case(
    index: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Embed a boxed strategy-selectable call in a SELECT case."""
    return qmc.select([_identity_case, _delegating_strategy_case])(index, target)


def _only_select(block: Block) -> SelectOperation:
    """Return the sole SELECT operation from a test block.

    Args:
        block (Block): Block expected to contain exactly one SELECT.

    Returns:
        SelectOperation: The block's sole SELECT operation.
    """
    selects = [
        operation
        for operation in block.operations
        if isinstance(operation, SelectOperation)
    ]
    assert len(selects) == 1
    return selects[0]


def _fresh_select_block() -> Block:
    """Build one structurally fixed SELECT with fresh case-formal UUIDs.

    Returns:
        Block: Affine block containing identity and X SELECT cases.
    """
    identity_target = Value(type=QubitType(), name="identity_target")
    identity_case = Block(
        input_values=[identity_target],
        output_values=[identity_target],
        kind=BlockKind.AFFINE,
        label_args=["q"],
    )

    x_target = Value(type=QubitType(), name="x_target")
    x_result = x_target.next_version()
    x_case = Block(
        input_values=[x_target],
        output_values=[x_result],
        operations=[
            GateOperation(
                operands=[x_target],
                results=[x_result],
                gate_type=GateOperationType.X,
            )
        ],
        kind=BlockKind.AFFINE,
        label_args=["q"],
    )

    index = Value(type=QubitType(), name="index")
    target = Value(type=QubitType(), name="target")
    index_result = index.next_version()
    target_result = target.next_version()
    select = SelectOperation(
        operands=[index, target],
        results=[index_result, target_result],
        num_index_qubits=1,
        case_blocks=[identity_case, x_case],
    )
    return Block(
        input_values=[index, target],
        output_values=[index_result, target_result],
        operations=[select],
        kind=BlockKind.AFFINE,
        label_args=["index", "target"],
    )


def test_inline_descends_into_select_case_blocks() -> None:
    """Inline-policy calls inside cases are expanded and counted correctly."""
    inlined = InlinePass().run(_select_with_delegating_case.block)
    select = _only_select(inlined)

    assert count_inline_invokes(inlined.operations) == 0
    assert not any(
        isinstance(operation, InvokeOperation)
        for case_block in select.case_blocks
        for operation in case_block.operations
    )
    assert any(
        isinstance(operation, GateOperation)
        and operation.gate_type is GateOperationType.X
        for operation in select.case_blocks[1].operations
    )


def test_prepare_collects_calls_inside_select_case_blocks() -> None:
    """PreparedModule includes call-graph edges reached only through a case."""
    entrypoint = _select_with_delegating_case.block
    select = _only_select(entrypoint)
    calls = [
        operation
        for operation in select.case_blocks[1].operations
        if isinstance(operation, InvokeOperation)
    ]
    assert len(calls) == 1

    prepared = prepare_module(entrypoint)

    assert calls[0].target in prepared.definitions
    assert calls[0].target in prepared.call_graph[prepared.entrypoint_ref]


def test_compile_time_if_uses_select_actual_not_outer_same_name() -> None:
    """Case lowering binds formals from SELECT actuals, not outer names."""
    inlined = InlinePass().run(_select_with_shadowed_selector.block)
    lowered = CompileTimeIfLoweringPass({"selector": 5}).run(inlined)
    first_case = _only_select(lowered).case_blocks[0]

    assert not any(
        isinstance(operation, IfOperation) for operation in first_case.operations
    )
    assert any(
        isinstance(operation, GateOperation)
        and operation.gate_type is GateOperationType.X
        for operation in first_case.operations
    )


def test_partial_eval_scopes_select_case_bindings() -> None:
    """Partial evaluation gives each case a fresh formal binding scope."""
    inlined = InlinePass().run(_select_with_shadowed_selector.block)
    lowered = PartialEvaluationPass({"selector": 5}).run(inlined)
    first_case = _only_select(lowered).case_blocks[0]

    assert not any(
        isinstance(operation, IfOperation) for operation in first_case.operations
    )
    assert any(
        isinstance(operation, GateOperation)
        and operation.gate_type is GateOperationType.X
        for operation in first_case.operations
    )


def test_substitution_descends_into_select_case_blocks() -> None:
    """Strategy substitutions reach boxed calls owned by SELECT cases."""
    transformed = SubstitutionPass(
        SubstitutionConfig(
            rules=[
                SubstitutionRule(
                    source_name="select_strategy_case",
                    strategy="test_strategy",
                )
            ]
        )
    ).run(_select_with_strategy_case.block)
    case = _only_select(transformed).case_blocks[1]
    invoke = next(
        operation
        for operation in case.operations
        if isinstance(operation, InvokeOperation)
    )

    assert invoke.strategy_name == "test_strategy"


def test_canonical_hash_ignores_fresh_case_formal_uuids() -> None:
    """Equivalent independently built SELECT cases have one content hash."""
    assert content_hash(_fresh_select_block()) == content_hash(_fresh_select_block())


def test_visualization_finds_global_phase_inside_select_case() -> None:
    """Global-phase discovery descends into SELECT-owned case Blocks."""
    phase = Value(type=FloatType(), name="phase").with_const(0.25)
    phased_target = Value(type=QubitType(), name="phased_target")
    phased_case = Block(
        input_values=[phased_target],
        output_values=[phased_target],
        operations=[GlobalPhaseOperation(operands=[phase], results=[])],
        kind=BlockKind.AFFINE,
    )
    identity_target = Value(type=QubitType(), name="identity_target")
    identity_case = Block(
        input_values=[identity_target],
        output_values=[identity_target],
        kind=BlockKind.AFFINE,
    )
    index = Value(type=QubitType(), name="index")
    target = Value(type=QubitType(), name="target")
    select = SelectOperation(
        operands=[index, target],
        results=[index.next_version(), target.next_version()],
        num_index_qubits=1,
        case_blocks=[identity_case, phased_case],
    )

    assert CircuitAnalyzer._contains_global_phase([select])
