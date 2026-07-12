"""Regression tests for structural invocation values in circuit drawings."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import CompOp, CompOpKind
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    InvokeOperation,
    signature_from_values,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VGate,
    VInlineBlock,
    VisualNode,
    VSkip,
    VUnfoldedSequence,
)


@qmc.qkernel
def _allocate_one_qubit() -> qmc.Qubit:
    """Allocate one qubit inside a reusable callable body.

    Returns:
        qmc.Qubit: Newly allocated qubit.
    """
    return qmc.qubit("allocated")


@qmc.qkernel
def _invoke_allocator_twice() -> tuple[qmc.Bit, qmc.Bit]:
    """Measure distinct qubits returned by two calls to one allocator.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Measurements of the two allocated qubits.
    """
    first = _allocate_one_qubit()
    second = _allocate_one_qubit()
    return qmc.measure(first), qmc.measure(second)


@qmc.qkernel
def _extend_with_allocated_qubit(
    q: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Return an input qubit together with a nested allocation.

    Args:
        q (qmc.Qubit): Input qubit to forward unchanged.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Input qubit and new nested qubit.
    """
    return q, qmc.qubit("nested")


@qmc.qkernel
def _forward_extended_qubits(
    q: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Forward the result of a nested allocating callable.

    Args:
        q (qmc.Qubit): Input qubit to forward through the nested call.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Input qubit and nested allocation.
    """
    return _extend_with_allocated_qubit(q)


@qmc.qkernel
def _draw_nested_allocator() -> tuple[qmc.Bit, qmc.Bit]:
    """Measure an input and an allocation returned through two call levels.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Measurements of both returned qubits.
    """
    q = qmc.qubit("input")
    q, nested = _forward_extended_qubits(q)
    return qmc.measure(q), qmc.measure(nested)


@qmc.qkernel
def _recursive_draw_helper(
    k: qmc.UInt,
    angle: qmc.Float,
    q: qmc.Qubit,
) -> qmc.Qubit:
    """Apply one Hadamard after a finite recursive countdown.

    Args:
        k (qmc.UInt): Remaining recursive depth.
        angle (qmc.Float): Symbolic angle carried unchanged through recursion.
        q (qmc.Qubit): Qubit forwarded through the recursion.

    Returns:
        qmc.Qubit: Qubit after one Hadamard at the base case.
    """
    if k == 0:
        q = qmc.h(q)
        q = qmc.rx(q, angle)
    else:
        q = _recursive_draw_helper(k - 1, angle, q)
    return q


@qmc.qkernel
def _draw_recursive_countdown(k: qmc.UInt, angle: qmc.Float) -> qmc.Bit:
    """Measure a qubit after a recursive helper invocation.

    Args:
        k (qmc.UInt): Recursive countdown depth.
        angle (qmc.Float): Symbolic angle forwarded through every frame.

    Returns:
        qmc.Bit: Measurement after the recursive helper.
    """
    q = qmc.qubit("q")
    q = _recursive_draw_helper(k, angle, q)
    return qmc.measure(q)


@qmc.qkernel
def _branching_nonterminating_draw_helper(
    k: qmc.UInt,
    q: qmc.Qubit,
) -> qmc.Qubit:
    """Recurse twice with an increasing classical driver.

    Args:
        k (qmc.UInt): Increasing nonterminating recursion driver.
        q (qmc.Qubit): Qubit forwarded through both recursive calls.

    Returns:
        qmc.Qubit: Qubit returned only if recursive expansion is truncated.
    """
    if k == 0:
        q = qmc.h(q)
    else:
        q = _branching_nonterminating_draw_helper(k + 1, q)
        q = _branching_nonterminating_draw_helper(k + 1, q)
    return q


@qmc.qkernel
def _draw_branching_nonterminating_recursion(k: qmc.UInt) -> qmc.Bit:
    """Measure after a deliberately nonterminating branching call.

    Args:
        k (qmc.UInt): Initial recursion driver.

    Returns:
        qmc.Bit: Measurement after the guarded recursive visualization.
    """
    q = qmc.qubit("q")
    q = _branching_nonterminating_draw_helper(k, q)
    return qmc.measure(q)


@qmc.qkernel
def _allocate_after_compile_time_if(
    flag: qmc.UInt,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Allocate through a call whose operand is a compile-time IF merge.

    Args:
        flag (qmc.UInt): Compile-time branch selector.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Selected input qubit and nested allocation.
    """
    q = qmc.qubit("conditional_input")
    if flag == qmc.uint(0):
        q = qmc.x(q)
    else:
        q = qmc.h(q)
    return _extend_with_allocated_qubit(q)


@qmc.qkernel
def _draw_compile_time_if_allocator() -> tuple[qmc.Bit, qmc.Bit]:
    """Measure results allocated after compile-time branch lowering.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Measurements of the selected and nested qubits.
    """
    q, nested = _allocate_after_compile_time_if(qmc.uint(0))
    return qmc.measure(q), qmc.measure(nested)


@qmc.qkernel
def _use_branch_local_ancilla(
    flag: qmc.UInt,
    q: qmc.Qubit,
) -> qmc.Qubit:
    """Measure a distinct branch-local ancilla in each compile-time branch.

    Args:
        flag (qmc.UInt): Compile-time branch selector.
        q (qmc.Qubit): Qubit forwarded through the helper.

    Returns:
        qmc.Qubit: Forwarded input qubit.
    """
    if flag == qmc.uint(0):
        selected = qmc.qubit("selected")
        _ = qmc.measure(selected)
    else:
        unselected = qmc.qubit("unselected")
        _ = qmc.measure(unselected)
    return q


@qmc.qkernel
def _draw_selected_branch_ancilla() -> qmc.Bit:
    """Inline a helper whose compile-time branch owns one ancilla.

    Returns:
        qmc.Bit: Measurement of the helper's forwarded qubit.
    """
    q = qmc.qubit("q")
    q = _use_branch_local_ancilla(qmc.uint(0), q)
    return qmc.measure(q)


@qmc.qkernel
def _draw_loop_selected_branch_ancilla() -> qmc.Bit:
    """Pass a concrete loop value into a callable compile-time branch.

    Returns:
        qmc.Bit: Measurement of the helper's forwarded qubit.
    """
    q = qmc.qubit("q")
    for i in qmc.range(1):
        q = _use_branch_local_ancilla(i, q)
    return qmc.measure(q)


@qmc.qkernel
def _draw_symbolic_branch_ancilla(flag: qmc.UInt) -> qmc.Bit:
    """Forward an unbound parameter into a callable runtime branch.

    Args:
        flag (qmc.UInt): Unbound selector that must remain symbolic.

    Returns:
        qmc.Bit: Measurement of the helper's forwarded qubit.
    """
    q = qmc.qubit("q")
    q = _use_branch_local_ancilla(flag, q)
    return qmc.measure(q)


@qmc.qkernel
def _forward_shadowed_symbolic_branch(
    flag: qmc.UInt,
    other: qmc.UInt,
    q: qmc.Qubit,
) -> qmc.Qubit:
    """Forward a differently named symbolic parameter through a nested call.

    Args:
        flag (qmc.UInt): Concrete outer binding that shadows the inner formal
            name but is otherwise unused.
        other (qmc.UInt): Symbolic selector forwarded to the inner callable.
        q (qmc.Qubit): Qubit forwarded through both call levels.

    Returns:
        qmc.Qubit: Forwarded input qubit.
    """
    return _use_branch_local_ancilla(other, q)


@qmc.qkernel
def _draw_shadowed_symbolic_branch(other: qmc.UInt) -> qmc.Bit:
    """Pass concrete and symbolic same-name bindings through nested calls.

    Args:
        other (qmc.UInt): Unbound selector forwarded to the inner callable.

    Returns:
        qmc.Bit: Measurement of the nested helper's forwarded qubit.
    """
    q = qmc.qubit("q")
    for i in qmc.range(1):
        q = _forward_shadowed_symbolic_branch(i, other, q)
    return qmc.measure(q)


@qmc.qkernel
def _allocate_and_return_element() -> qmc.Qubit:
    """Return one element from a register allocated inside the callable.

    Returns:
        qmc.Qubit: Third qubit from the callable-local register.
    """
    q = qmc.qubit_array(4, "inner")
    return q[2]


@qmc.qkernel
def _use_allocated_element() -> qmc.Bit:
    """Apply and measure a gate on an element returned by a callable.

    Returns:
        qmc.Bit: Measurement of the returned register element.
    """
    q = _allocate_and_return_element()
    q = qmc.h(q)
    return qmc.measure(q)


@qmc.qkernel
def _allocate_and_return_slice() -> qmc.Vector[qmc.Qubit]:
    """Return a slice from a register allocated inside the callable.

    Returns:
        qmc.Vector[qmc.Qubit]: Two-qubit slice of the local register.
    """
    q = qmc.qubit_array(4, "inner")
    return q[1:3]


@qmc.qkernel
def _use_allocated_slice() -> qmc.Vector[qmc.Bit]:
    """Apply and measure gates on a slice returned by a callable.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements of the returned slice.
    """
    q = _allocate_and_return_slice()
    q = qmc.h(q)
    return qmc.measure(q)


@qmc.qkernel
def _allocate_and_return_partitioned_slices() -> tuple[
    qmc.Vector[qmc.Qubit],
    qmc.Vector[qmc.Qubit],
]:
    """Return two disjoint slices of one callable-local register.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Left and right
            halves of the local register.
    """
    q = qmc.qubit_array(4, "inner")
    return q[0:2], q[2:4]


@qmc.qkernel
def _use_partitioned_slices() -> tuple[
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
]:
    """Apply distinct gates to two callable-returned register slices.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measurements of the
            left and right slices.
    """
    left, right = _allocate_and_return_partitioned_slices()
    left = qmc.h(left)
    right = qmc.x(right)
    return qmc.measure(left), qmc.measure(right)


def _resultless_allocator_definition() -> CallableDef:
    """Build a shared resultless callable that allocates one local qubit.

    Returns:
        CallableDef: Inline callable definition with a body-local allocation.
    """
    local_qubit = Value(type=QubitType(), name="local")
    body = Block(
        name="side_effect",
        operations=[QInitOperation(results=[local_qubit])],
    )
    ref = CallableRef(namespace="user", name="side_effect")
    return CallableDef(
        ref=ref,
        body=body,
        default_policy=CallPolicy.INLINE,
    )


def _zero_length_allocator_definition() -> CallableDef:
    """Build a resultless callable allocating an empty quantum vector.

    Returns:
        CallableDef: Inline callable definition with a zero-width allocation.
    """
    zero = Value(type=UIntType(), name="zero").with_const(0)
    empty = ArrayValue(
        type=QubitType(),
        name="empty",
        shape=(zero,),
    )
    body = Block(
        name="empty_side_effect",
        operations=[QInitOperation(results=[empty])],
    )
    ref = CallableRef(namespace="user", name="empty_side_effect")
    return CallableDef(
        ref=ref,
        body=body,
        default_policy=CallPolicy.INLINE,
    )


def _build_shared_resultless_invokes_graph() -> Block:
    """Build two call sites that intentionally share one callable definition.

    Returns:
        Block: Graph containing two resultless inline invocations.
    """
    definition = _resultless_allocator_definition()
    return Block(
        name="main",
        operations=[
            InvokeOperation(target=definition.ref, definition=definition),
            InvokeOperation(target=definition.ref, definition=definition),
        ],
    )


def _build_resultless_control_flow_graph(kind: str) -> Block:
    """Build control flow containing a resultless allocating invocation.

    Args:
        kind (str): Control-flow kind, either ``"if"`` or ``"for"``.

    Returns:
        Block: Graph with one outer wire and one callable-local wire.

    Raises:
        ValueError: If ``kind`` is not a supported control-flow kind.
    """
    outer_qubit = Value(type=QubitType(), name="outer")
    definition = _resultless_allocator_definition()
    invoke = InvokeOperation(target=definition.ref, definition=definition)
    if kind == "if":
        condition = Value(type=BitType(), name="condition")
        control_flow = IfOperation(
            operands=[condition],
            true_operations=[invoke],
        )
    elif kind == "for":
        start = Value(type=UIntType(), name="start").with_const(0)
        stop = Value(type=UIntType(), name="stop").with_const(2)
        step = Value(type=UIntType(), name="step").with_const(1)
        loop_var = Value(type=UIntType(), name="i")
        control_flow = ForOperation(
            operands=[start, stop, step],
            loop_var="i",
            loop_var_value=loop_var,
            operations=[invoke],
        )
    else:
        raise ValueError(f"Unsupported control-flow kind: {kind}")
    return Block(
        name="main",
        operations=[
            QInitOperation(results=[outer_qubit]),
            control_flow,
        ],
    )


def _build_analyzed_body_invoke_graph() -> Block:
    """Build an analyzed graph containing an analyzed callable body.

    Returns:
        Block: Analyzed caller whose invocation returns one allocated qubit.
    """
    local_qubit = Value(type=QubitType(), name="q")
    body = Block(
        name="alloc",
        kind=BlockKind.ANALYZED,
        operations=[QInitOperation(results=[local_qubit])],
        output_values=[local_qubit],
        output_names=["q"],
    )
    return Block(
        name="main",
        kind=BlockKind.ANALYZED,
        operations=[body.call()],
    )


def _build_local_qinit_for_graph() -> Block:
    """Build a for loop whose body directly allocates one qubit.

    Returns:
        Block: Graph containing one loop-local quantum allocation.
    """
    start = Value(type=UIntType(), name="start").with_const(0)
    stop = Value(type=UIntType(), name="stop").with_const(1)
    step = Value(type=UIntType(), name="step").with_const(1)
    loop_var = Value(type=UIntType(), name="i")
    loop_local = Value(type=QubitType(), name="loop_local")
    return Block(
        name="main",
        operations=[
            ForOperation(
                operands=[start, stop, step],
                loop_var="i",
                loop_var_value=loop_var,
                operations=[QInitOperation(results=[loop_local])],
            )
        ],
    )


def _build_zero_length_callable_for_graph() -> Block:
    """Build a for loop invoking one zero-width allocating callable.

    Returns:
        Block: Graph whose loop body owns no physical wires.
    """
    start = Value(type=UIntType(), name="start").with_const(0)
    stop = Value(type=UIntType(), name="stop").with_const(1)
    step = Value(type=UIntType(), name="step").with_const(1)
    loop_var = Value(type=UIntType(), name="i")
    definition = _zero_length_allocator_definition()
    invoke = InvokeOperation(target=definition.ref, definition=definition)
    return Block(
        name="main",
        operations=[
            ForOperation(
                operands=[start, stop, step],
                loop_var="i",
                loop_var_value=loop_var,
                operations=[invoke],
            )
        ],
    )


def _zero_width_qubit_array(name: str) -> ArrayValue:
    """Build a concrete zero-width quantum array value.

    Args:
        name (str): Display name for the empty array.

    Returns:
        ArrayValue: One-dimensional quantum array with concrete length zero.
    """
    zero = Value(type=UIntType(), name="zero").with_const(0)
    return ArrayValue(type=QubitType(), name=name, shape=(zero,))


def _build_zero_width_result_graph(kind: str) -> tuple[Block, dict[int, str]]:
    """Build a callable operation forwarding one empty quantum array.

    Args:
        kind (str): Callable kind: ``"invoke"``, ``"controlled"``, or
            ``"inverse"``.

    Returns:
        tuple[Block, dict[int, str]]: Graph and expected physical wire names.

    Raises:
        ValueError: If ``kind`` is not supported.
    """
    formal = _zero_width_qubit_array("formal")
    body = Block(
        name="empty_body",
        label_args=["target"],
        input_values=[formal],
        output_values=[formal],
        output_names=["target"],
    )
    target = _zero_width_qubit_array("target")

    if kind == "invoke":
        operation = InvokeOperation(
            operands=[target],
            results=[target.next_version()],
            target=CallableRef(namespace="user", name="opaque_empty"),
        )
        operations = [QInitOperation(results=[target]), operation]
        expected_names: dict[int, str] = {}
    elif kind == "controlled":
        control = Value(type=QubitType(), name="control")
        operation = ConcreteControlledU(
            operands=[control, target],
            results=[control.next_version(), target.next_version()],
            block=body,
            num_controls=1,
        )
        operations = [
            QInitOperation(results=[control]),
            QInitOperation(results=[target]),
            operation,
        ]
        expected_names = {0: "control"}
    elif kind == "inverse":
        operation = InverseBlockOperation(
            operands=[target],
            results=[target.next_version()],
            num_target_qubits=0,
            custom_name="empty_body",
            source_block=body,
            implementation_block=body,
        )
        operations = [QInitOperation(results=[target]), operation]
        expected_names = {}
    else:
        raise ValueError(f"Unsupported zero-width callable kind: {kind}")

    return Block(name=f"{kind}_empty", operations=operations), expected_names


def _build_zero_trip_qinit_for_graph() -> Block:
    """Build a concrete zero-trip for loop containing one local allocation.

    Returns:
        Block: Graph whose loop body must not allocate a display wire.
    """
    start = Value(type=UIntType(), name="start").with_const(0)
    stop = Value(type=UIntType(), name="stop").with_const(0)
    step = Value(type=UIntType(), name="step").with_const(1)
    loop_var = Value(type=UIntType(), name="i")
    local = Value(type=QubitType(), name="local")
    return Block(
        name="zero_trip_for",
        operations=[
            ForOperation(
                operands=[start, stop, step],
                loop_var="i",
                loop_var_value=loop_var,
                operations=[QInitOperation(results=[local])],
            )
        ],
    )


def _build_bound_empty_qinit_for_items_graph() -> Block:
    """Build a bound-empty items loop containing one local allocation.

    Returns:
        Block: Graph whose loop body must not allocate a display wire.
    """
    empty = DictValue(name="items", entries=()).with_dict_runtime_metadata({})
    key = Value(type=UIntType(), name="key")
    item = Value(type=FloatType(), name="item")
    local = Value(type=QubitType(), name="local")
    return Block(
        name="zero_trip_for_items",
        operations=[
            ForItemsOperation(
                operands=[empty],
                key_vars=["key"],
                value_var="item",
                key_var_values=(key,),
                value_var_value=item,
                operations=[QInitOperation(results=[local])],
            )
        ],
    )


def _build_root_quantum_input_graph() -> Block:
    """Build a root block operating directly on one quantum input.

    Returns:
        Block: Graph whose input is a physical wire without a QInit operation.
    """
    qubit = Value(type=QubitType(), name="input")
    result = qubit.next_version()
    return Block(
        name="root_input",
        label_args=["qubit"],
        input_values=[qubit],
        output_values=[result],
        output_names=["qubit"],
        operations=[
            GateOperation.fixed(
                GateOperationType.H,
                [qubit],
                [result],
            )
        ],
    )


def _build_symbolic_element_opaque_graph(
    *,
    strided_view: bool,
) -> tuple[Block, int]:
    """Build an opaque pass-through call on a symbolic array element.

    Args:
        strided_view (bool): Whether the symbolic element belongs to
            ``root[1:5:2]`` instead of the root register itself.

    Returns:
        tuple[Block, int]: Graph and conservative existing wire expected for
            the unresolved element.
    """
    root_size = Value(type=UIntType(), name="root_size").with_const(5)
    root = ArrayValue(type=QubitType(), name="q", shape=(root_size,))
    parent = root
    expected_wire = 0
    if strided_view:
        view_size = Value(type=UIntType(), name="view_size").with_const(2)
        start = Value(type=UIntType(), name="start").with_const(1)
        step = Value(type=UIntType(), name="step").with_const(2)
        parent = ArrayValue(
            type=QubitType(),
            name="view",
            shape=(view_size,),
            slice_of=root,
            slice_start=start,
            slice_step=step,
        )
        expected_wire = 1

    index = Value(type=UIntType(), name="index").with_parameter("index")
    element = Value(
        type=QubitType(),
        name="element",
        parent_array=parent,
        element_indices=(index,),
    )
    call_result = element.next_version()
    call = InvokeOperation(
        operands=[element],
        results=[call_result],
        target=CallableRef(namespace="user", name="opaque_element"),
    )
    gate_result = call_result.next_version()
    gate = GateOperation.fixed(
        GateOperationType.H,
        [call_result],
        [gate_result],
    )
    return (
        Block(
            name="symbolic_element",
            operations=[QInitOperation(results=[root]), call, gate],
        ),
        expected_wire,
    )


def _build_unresolved_slice_opaque_graph(
    fresh_result_size: int | None = None,
) -> Block:
    """Build an opaque call on an unresolved whole slice.

    Args:
        fresh_result_size (int | None): Width of a fresh caller-local result.
            Defaults to None, which returns a pass-through next version of the
            slice. A concrete width also adds a downstream gate.

    Returns:
        Block: Graph whose slice must conservatively reuse root wires and
            allocate only a genuinely wider result suffix.
    """
    root_size = Value(type=UIntType(), name="root_size").with_const(5)
    root = ArrayValue(type=QubitType(), name="q", shape=(root_size,))
    view_size = Value(type=UIntType(), name="view_size").with_const(2)
    start = Value(type=UIntType(), name="start").with_parameter("start")
    step = Value(type=UIntType(), name="step").with_const(1)
    view = ArrayValue(
        type=QubitType(),
        name="view",
        shape=(view_size,),
        slice_of=root,
        slice_start=start,
        slice_step=step,
    )
    if fresh_result_size is None:
        result = view.next_version()
    else:
        result_size = Value(type=UIntType(), name="result_size").with_const(
            fresh_result_size
        )
        result = ArrayValue(type=QubitType(), name="fresh", shape=(result_size,))
    call = InvokeOperation(
        operands=[view],
        results=[result],
        target=CallableRef(namespace="user", name="opaque_slice"),
    )
    operations = [QInitOperation(results=[root]), call]
    if fresh_result_size is not None:
        gate_operand: Value = result
        if fresh_result_size == 3:
            suffix_index = Value(type=UIntType(), name="suffix_index").with_const(2)
            gate_operand = Value(
                type=QubitType(),
                name="fresh[2]",
                parent_array=result,
                element_indices=(suffix_index,),
            )
        operations.append(
            GateOperation.fixed(
                GateOperationType.H,
                [gate_operand],
                [gate_operand.next_version()],
            )
        )
    return Block(
        name="unresolved_slice",
        operations=operations,
    )


def _build_unresolved_slice_inline_graph() -> Block:
    """Build an inline call operating on an unresolved whole slice.

    Returns:
        Block: Graph whose inline body must conservatively cover root wires.
    """
    root_size = Value(type=UIntType(), name="root_size").with_const(5)
    root = ArrayValue(type=QubitType(), name="q", shape=(root_size,))
    view_size = Value(type=UIntType(), name="view_size").with_const(2)
    start = Value(type=UIntType(), name="start").with_parameter("start")
    step = Value(type=UIntType(), name="step").with_const(1)
    view = ArrayValue(
        type=QubitType(),
        name="view",
        shape=(view_size,),
        slice_of=root,
        slice_start=start,
        slice_step=step,
    )

    formal = ArrayValue(type=QubitType(), name="target", shape=(view_size,))
    formal_result = formal.next_version()
    body = Block(
        name="unresolved_slice_body",
        label_args=["target"],
        input_values=[formal],
        output_values=[formal_result],
        output_names=["target"],
        operations=[
            GateOperation.fixed(
                GateOperationType.H,
                [formal],
                [formal_result],
            ),
            ReturnOperation(operands=[formal_result]),
        ],
    )
    call = body.call(target=view)
    return Block(
        name="unresolved_slice_inline",
        operations=[QInitOperation(results=[root]), call],
    )


def _build_nested_unresolved_slice_inline_graph(leaf_kind: str) -> Block:
    """Build a symbolic slice forwarded through two inline callables.

    Args:
        leaf_kind (str): Final call kind, either ``"inline"`` for a body-backed
            qkernel call or ``"opaque"`` for a bodyless pass-through call.

    Returns:
        Block: Graph whose nested bodies must conservatively cover every root
            wire without concretizing the unresolved view to a prefix.

    Raises:
        ValueError: If ``leaf_kind`` is not ``"inline"`` or ``"opaque"``.
    """
    root_size = Value(type=UIntType(), name="root_size").with_const(5)
    root = ArrayValue(type=QubitType(), name="q", shape=(root_size,))
    view_size = Value(type=UIntType(), name="view_size").with_const(2)
    start = Value(type=UIntType(), name="start").with_parameter("start")
    step = Value(type=UIntType(), name="step").with_const(1)
    view = ArrayValue(
        type=QubitType(),
        name="view",
        shape=(view_size,),
        slice_of=root,
        slice_start=start,
        slice_step=step,
    )

    middle_formal = ArrayValue(type=QubitType(), name="target", shape=(view_size,))
    if leaf_kind == "inline":
        leaf_formal = ArrayValue(
            type=QubitType(),
            name="target",
            shape=(view_size,),
        )
        leaf_result = leaf_formal.next_version()
        leaf = Block(
            name="unresolved_slice_leaf",
            label_args=["target"],
            input_values=[leaf_formal],
            output_values=[leaf_result],
            output_names=["target"],
            operations=[
                GateOperation.fixed(
                    GateOperationType.H,
                    [leaf_formal],
                    [leaf_result],
                ),
                ReturnOperation(operands=[leaf_result]),
            ],
        )
        leaf_call = leaf.call(target=middle_formal)
    elif leaf_kind == "opaque":
        leaf_call = InvokeOperation(
            operands=[middle_formal],
            results=[middle_formal.next_version()],
            target=CallableRef(namespace="user", name="opaque_slice_leaf"),
        )
    else:
        raise ValueError(f"Unsupported leaf kind: {leaf_kind}")
    middle_result = leaf_call.results[0]
    middle = Block(
        name="unresolved_slice_middle",
        label_args=["target"],
        input_values=[middle_formal],
        output_values=[middle_result],
        output_names=["target"],
        operations=[leaf_call, ReturnOperation(operands=[middle_result])],
    )

    call = middle.call(target=view)
    return Block(
        name="nested_unresolved_slice_inline",
        operations=[QInitOperation(results=[root]), call],
    )


def _build_nested_unresolved_slice_widening_graph(leaf_kind: str) -> Block:
    """Build widening and forwarding calls rooted at an unresolved slice.

    Args:
        leaf_kind (str): Forwarding call kind, either ``"inline"`` for a
            body-backed qkernel or ``"opaque"`` for a bodyless invocation.

    Returns:
        Block: Graph whose known widening suffix must remain exact while the
            unresolved input positions retain their conservative footprint.

    Raises:
        ValueError: If ``leaf_kind`` is not ``"inline"`` or ``"opaque"``.
    """
    root_size = Value(type=UIntType(), name="root_size").with_const(5)
    root = ArrayValue(type=QubitType(), name="q", shape=(root_size,))
    view_size = Value(type=UIntType(), name="view_size").with_const(2)
    view_start = Value(type=UIntType(), name="view_start").with_parameter("view_start")
    view_step = Value(type=UIntType(), name="view_step").with_const(1)
    view = ArrayValue(
        type=QubitType(),
        name="view",
        shape=(view_size,),
        slice_of=root,
        slice_start=view_start,
        slice_step=view_step,
    )

    middle_formal = ArrayValue(
        type=QubitType(),
        name="target",
        shape=(view_size,),
    )
    widened_size = Value(type=UIntType(), name="widened_size").with_const(3)
    widened_body_result = ArrayValue(
        type=QubitType(),
        name="fresh",
        shape=(widened_size,),
    )
    widening_call = InvokeOperation(
        operands=[middle_formal],
        results=[widened_body_result],
        target=CallableRef(namespace="user", name="opaque_slice_widening"),
    )
    middle = Block(
        name="unresolved_slice_widening_middle",
        label_args=["target"],
        input_values=[middle_formal],
        output_values=[widened_body_result],
        output_names=["fresh"],
        operations=[
            widening_call,
            ReturnOperation(operands=[widened_body_result]),
        ],
    )
    middle_call = middle.call(target=view)
    widened_result = middle_call.results[0]
    suffix_index = Value(type=UIntType(), name="suffix_index").with_const(2)
    direct_suffix = Value(
        type=QubitType(),
        name="fresh[2]",
        parent_array=widened_result,
        element_indices=(suffix_index,),
    )
    direct_gate = GateOperation.fixed(
        GateOperationType.H,
        [direct_suffix],
        [direct_suffix.next_version()],
    )

    if leaf_kind == "inline":
        leaf_formal = ArrayValue(
            type=QubitType(),
            name="target",
            shape=(widened_size,),
        )
        leaf_element = Value(
            type=QubitType(),
            name="target[2]",
            parent_array=leaf_formal,
            element_indices=(suffix_index,),
        )
        leaf_result = leaf_formal.next_version()
        leaf = Block(
            name="widened_slice_leaf",
            label_args=["target"],
            input_values=[leaf_formal],
            output_values=[leaf_result],
            output_names=["target"],
            operations=[
                GateOperation.fixed(
                    GateOperationType.X,
                    [leaf_element],
                    [leaf_element.next_version()],
                ),
                ReturnOperation(operands=[leaf_result]),
            ],
        )
        forwarding_call = leaf.call(target=widened_result)
    elif leaf_kind == "opaque":
        forwarded_result = ArrayValue(
            type=QubitType(),
            name="forwarded",
            shape=(widened_size,),
        )
        forwarding_call = InvokeOperation(
            operands=[widened_result],
            results=[forwarded_result],
            target=CallableRef(namespace="user", name="opaque_widened_slice"),
        )
    else:
        raise ValueError(f"Unsupported leaf kind: {leaf_kind}")

    forwarded_result = forwarding_call.results[0]
    forwarded_suffix = Value(
        type=QubitType(),
        name="forwarded[2]",
        parent_array=forwarded_result,
        element_indices=(suffix_index,),
    )
    forwarded_gate = GateOperation.fixed(
        GateOperationType.Z,
        [forwarded_suffix],
        [forwarded_suffix.next_version()],
    )
    return Block(
        name="nested_unresolved_slice_widening",
        operations=[
            QInitOperation(results=[root]),
            middle_call,
            direct_gate,
            forwarding_call,
            forwarded_gate,
        ],
    )


def _build_opaque_widening_result_graph(
    *,
    occupy_next_wire: bool = False,
) -> tuple[Block, ArrayValue]:
    """Build an opaque call whose result vector is wider than its input.

    Args:
        occupy_next_wire (bool): Whether an unrelated scalar allocation should
            occupy the wire immediately after the input vector. Defaults to
            False.

    Returns:
        tuple[Block, ArrayValue]: Graph and three-qubit opaque result vector.
    """
    input_size = Value(type=UIntType(), name="input_size").with_const(2)
    result_size = Value(type=UIntType(), name="result_size").with_const(3)
    input_vector = ArrayValue(
        type=QubitType(),
        name="input",
        shape=(input_size,),
    )
    result_vector = ArrayValue(
        type=QubitType(),
        name="result",
        shape=(result_size,),
    )
    call = InvokeOperation(
        operands=[input_vector],
        results=[result_vector],
        target=CallableRef(namespace="user", name="opaque_widening"),
    )
    operations = [QInitOperation(results=[input_vector])]
    if occupy_next_wire:
        unrelated = Value(type=QubitType(), name="unrelated")
        operations.append(QInitOperation(results=[unrelated]))
    operations.append(call)
    return (
        Block(
            name="opaque_widening",
            operations=operations,
        ),
        result_vector,
    )


def _build_legacy_stack_parameter_graph() -> Block:
    """Build a callable whose parameter name matches the former stack key.

    Returns:
        Block: Graph whose concrete zero binding must select only the X branch.
    """
    legacy_stack_name = "__qamomile_visual_call_stack__"
    flag = Value(type=UIntType(), name="flag").with_parameter(legacy_stack_name)
    formal_qubit = Value(type=QubitType(), name="q")
    zero = Value(type=UIntType(), name="zero").with_const(0)
    condition = Value(type=BitType(), name="condition")
    compare = CompOp(
        operands=[flag, zero],
        results=[condition],
        kind=CompOpKind.EQ,
    )
    true_qubit = formal_qubit.next_version()
    false_qubit = formal_qubit.next_version()
    true_gate = GateOperation.fixed(
        GateOperationType.X,
        [formal_qubit],
        [true_qubit],
    )
    false_gate = GateOperation.fixed(
        GateOperationType.H,
        [formal_qubit],
        [false_qubit],
    )
    merged = Value(type=QubitType(), name="merged")
    conditional = IfOperation(
        operands=[condition],
        true_operations=[true_gate],
        false_operations=[false_gate],
        true_yields=[true_qubit],
        false_yields=[false_qubit],
        results=[merged],
    )
    body = Block(
        name="legacy_stack_parameter",
        label_args=["flag", "q"],
        input_values=[flag, formal_qubit],
        operations=[
            compare,
            conditional,
            ReturnOperation(operands=[merged]),
        ],
        output_values=[merged],
        output_names=["q"],
    )

    actual_flag = Value(type=UIntType(), name="actual_flag").with_const(0)
    actual_qubit = Value(type=QubitType(), name="q")
    call = body.call(flag=actual_flag, q=actual_qubit)
    return Block(
        name="legacy_stack_parameter_caller",
        operations=[QInitOperation(results=[actual_qubit]), call],
    )


def _build_structural_input_graph() -> Block:
    """Build a valid invocation whose tuple input owns one quantum leaf.

    Returns:
        Block: Caller graph containing the structural invocation.
    """
    formal_qubit = Value(type=QubitType(), name="formal_qubit")
    formal_result = formal_qubit.next_version()
    formal_tuple = TupleValue(name="args", elements=(formal_qubit,))
    callee = Block(
        name="touch",
        label_args=["args"],
        input_values=[formal_tuple],
        output_values=[formal_result],
        output_names=["qubit"],
        operations=[
            GateOperation.fixed(
                GateOperationType.H,
                [formal_qubit],
                [formal_result],
            ),
            ReturnOperation(operands=[formal_result]),
        ],
    )
    caller_qubit = Value(type=QubitType(), name="q")
    actual_tuple = TupleValue(name="args", elements=(caller_qubit,))
    invoke = callee.call(args=actual_tuple)
    return Block(
        name="main",
        operations=[
            QInitOperation(results=[caller_qubit]),
            invoke,
        ],
    )


def _build_mixed_tuple_input_graph() -> Block:
    """Build an invocation whose tuple carries a qubit and bound angle.

    Returns:
        Block: Caller graph containing the mixed tuple invocation.
    """
    formal_qubit = Value(type=QubitType(), name="formal_qubit")
    formal_angle = Value(type=FloatType(), name="theta")
    formal_result = formal_qubit.next_version()
    formal_tuple = TupleValue(
        name="args",
        elements=(formal_qubit, formal_angle),
    )
    callee = Block(
        name="rotate",
        label_args=["args"],
        input_values=[formal_tuple],
        output_values=[formal_result],
        output_names=["qubit"],
        operations=[
            GateOperation.rotation(
                GateOperationType.RZ,
                [formal_qubit],
                formal_angle,
                [formal_result],
            ),
            ReturnOperation(operands=[formal_result]),
        ],
    )

    caller_qubit = Value(type=QubitType(), name="q")
    actual_angle = Value(type=FloatType(), name="theta").with_const(0.5)
    actual_tuple = TupleValue(
        name="args",
        elements=(caller_qubit, actual_angle),
    )
    invoke = callee.call(args=actual_tuple)
    return Block(
        name="main",
        operations=[
            QInitOperation(results=[caller_qubit]),
            invoke,
        ],
    )


def _build_swapped_tuple_io_graph() -> tuple[Block, InvokeOperation]:
    """Build an invocation that returns its tuple qubits in reverse order.

    Returns:
        tuple[Block, InvokeOperation]: Caller graph and swapped-output call.
    """
    formal_first = Value(type=QubitType(), name="first")
    formal_second = Value(type=QubitType(), name="second")
    formal_tuple = TupleValue(
        name="inputs",
        elements=(formal_first, formal_second),
    )
    swapped_output = TupleValue(
        name="outputs",
        elements=(formal_second, formal_first),
    )
    callee = Block(
        name="swap_order",
        label_args=["inputs"],
        input_values=[formal_tuple],
        output_values=[swapped_output],
        output_names=["outputs"],
        operations=[ReturnOperation(operands=[swapped_output])],
    )

    caller_first = Value(type=QubitType(), name="first")
    caller_second = Value(type=QubitType(), name="second")
    actual_tuple = TupleValue(
        name="inputs",
        elements=(caller_first, caller_second),
    )
    invoke = callee.call(inputs=actual_tuple)
    return (
        Block(
            name="main",
            operations=[
                QInitOperation(results=[caller_first]),
                QInitOperation(results=[caller_second]),
                invoke,
            ],
        ),
        invoke,
    )


def _build_passthrough_dict_graph() -> Block:
    """Build a call returning a populated actual through an empty formal.

    Returns:
        Block: Caller graph containing the dictionary pass-through call.

    Raises:
        AssertionError: If invocation materialization loses dictionary entries.
    """
    formal_dict = DictValue(name="mapping", entries=()).with_parameter("mapping")
    callee = Block(
        name="pass_mapping",
        label_args=["mapping"],
        input_values=[formal_dict],
        output_values=[formal_dict],
        output_names=["mapping"],
        operations=[ReturnOperation(operands=[formal_dict])],
    )

    actual_key = Value(type=UIntType(), name="key").with_const(0)
    actual_value = Value(type=FloatType(), name="value").with_const(1.0)
    actual_dict = DictValue(
        name="mapping",
        entries=((actual_key, actual_value),),
    )
    invoke = callee.call(mapping=actual_dict)
    result = invoke.results[0]
    assert isinstance(result, DictValue)
    assert len(result.entries) == 1

    caller_qubit = Value(type=QubitType(), name="q")
    return Block(
        name="main",
        operations=[
            QInitOperation(results=[caller_qubit]),
            invoke,
        ],
    )


def _build_input_and_new_output_graph() -> Block:
    """Build a call that returns its input plus a newly allocated qubit.

    Returns:
        Block: Caller graph containing the allocating call.
    """
    formal_qubit = Value(type=QubitType(), name="formal_qubit")
    new_qubit = Value(type=QubitType(), name="new_qubit")
    callee = Block(
        name="extend_register",
        label_args=["qubit"],
        input_values=[formal_qubit],
        output_values=[formal_qubit, new_qubit],
        output_names=["qubit", "new_qubit"],
        operations=[
            QInitOperation(results=[new_qubit]),
            ReturnOperation(operands=[formal_qubit, new_qubit]),
        ],
    )

    caller_qubit = Value(type=QubitType(), name="q")
    invoke = callee.call(qubit=caller_qubit)
    return Block(
        name="main",
        operations=[
            QInitOperation(results=[caller_qubit]),
            invoke,
        ],
    )


def _build_structural_result_graph(kind: str) -> tuple[Block, ArrayValue]:
    """Build a graph whose invocation returns a structural quantum value.

    Args:
        kind (str): Structural wrapper kind, either ``"tuple"`` or ``"dict"``.

    Returns:
        tuple[Block, ArrayValue]: Mutated caller graph and nested array result.

    Raises:
        AssertionError: If ``kind`` is unsupported or the constructed call no
            longer has the expected structural result shape.
    """

    @qmc.qkernel
    def make_register() -> qmc.Vector[qmc.Qubit]:
        """Allocate a two-qubit register.

        Returns:
            qmc.Vector[qmc.Qubit]: Newly allocated register.
        """
        q = qmc.qubit_array(2, "q")
        return q

    @qmc.qkernel
    def touch_register(
        q: qmc.Vector[qmc.Qubit],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply a Hadamard gate to a register.

        Args:
            q (qmc.Vector[qmc.Qubit]): Register to update.

        Returns:
            qmc.Vector[qmc.Qubit]: Updated register.
        """
        q = qmc.h(q)
        return q

    @qmc.qkernel
    def main() -> qmc.Vector[qmc.Bit]:
        """Allocate, update, and measure a nested register.

        Returns:
            qmc.Vector[qmc.Bit]: Register measurements.
        """
        q = make_register()
        q = touch_register(q)
        return qmc.measure(q)

    graph = main._build_graph_for_visualization()
    invokes = [op for op in graph.operations if isinstance(op, InvokeOperation)]
    assert len(invokes) == 2
    producer = invokes[0]
    body = producer.effective_body()
    assert isinstance(body, Block)
    assert len(body.output_values) == 1
    assert len(producer.results) == 1

    body_result = body.output_values[0]
    call_result = producer.results[0]
    assert isinstance(body_result, ArrayValue)
    assert isinstance(call_result, ArrayValue)

    if kind == "tuple":
        structural_body_result = TupleValue(
            name="register_tuple",
            elements=(body_result,),
        )
        structural_call_result = TupleValue(
            name="register_tuple",
            elements=(call_result,),
        )
    elif kind == "dict":
        body_key = Value(type=UIntType(), name="key").with_const(0)
        call_key = body_key.next_version()
        structural_body_result = DictValue(
            name="register_dict",
            entries=((body_key, body_result),),
        )
        structural_call_result = DictValue(
            name="register_dict",
            entries=((call_key, call_result),),
        )
    else:
        raise AssertionError(f"Unsupported structural result kind: {kind}")

    body.output_values = [structural_body_result]
    producer.results = [structural_call_result]
    return_op = next(
        op for op in reversed(body.operations) if isinstance(op, ReturnOperation)
    )
    return_op.operands = [structural_body_result]
    assert producer.definition is not None
    producer.definition.signature = signature_from_values(
        producer.operands,
        producer.results,
    )
    return graph, call_result


def _iter_visual_nodes(nodes: list[VisualNode]) -> Iterator[VisualNode]:
    """Yield visual nodes recursively from nested drawing containers.

    Args:
        nodes (list[VisualNode]): Root nodes to traverse.

    Returns:
        Iterator[VisualNode]: Depth-first visual-node iterator.
    """
    for node in nodes:
        yield node
        children = getattr(node, "children", None)
        if children:
            yield from _iter_visual_nodes(children)
        if isinstance(node, VUnfoldedSequence):
            for iteration in node.iterations:
                yield from _iter_visual_nodes(iteration)


def _assert_structural_result_uses_source_wires(
    graph: Block,
    call_result: ArrayValue,
) -> None:
    """Assert a structural quantum result aliases its callee-owned wires.

    Args:
        graph (Block): Caller graph to analyze.
        call_result (ArrayValue): Caller-local array result to resolve.

    Raises:
        AssertionError: If any result wire or rendered gate is incorrect.
    """
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)

    assert num_qubits == 2, (num_qubits, qubit_names)
    assert sorted(qubit_names.values()) == ["q[0]", "q[1]"]
    assert [qubit_map[f"{call_result.logical_id}_[{index}]"] for index in range(2)] == [
        0,
        1,
    ]

    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )
    gates = [
        node for node in _iter_visual_nodes(circuit.children) if isinstance(node, VGate)
    ]
    assert sorted(
        tuple(gate.qubit_indices) for gate in gates if gate.label == r"$H$"
    ) == [(0,), (1,)]
    assert any(gate.label == "M" and gate.qubit_indices == [0, 1] for gate in gates)


def test_shared_resultless_invoke_definitions_use_distinct_scopes() -> None:
    """Shared definitions keep separate body-owned wires at each call site."""
    graph = _build_shared_resultless_invokes_graph()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 2
    assert qubit_names == {0: "local", 1: "local"}
    calls = [node for node in circuit.children if isinstance(node, VInlineBlock)]
    assert [node.affected_qubits for node in calls] == [[0], [1]]


@pytest.mark.parametrize("inline", [False, True])
def test_analyzed_callable_body_skips_compile_time_lowering(inline: bool) -> None:
    """Analyzed callable bodies draw without rerunning an affine-only pass."""
    graph = _build_analyzed_body_invoke_graph()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=inline,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 1
    assert qubit_names == {0: "q"}
    assert len(circuit.children) == 1
    call = circuit.children[0]
    if inline:
        assert isinstance(call, VInlineBlock)
        assert call.affected_qubits == [0]
    else:
        assert isinstance(call, VGate)
        assert call.qubit_indices == [0]


@pytest.mark.parametrize(
    ("kind", "folded"),
    [
        pytest.param("if", False, id="unfolded-if"),
        pytest.param("if", True, id="folded-if"),
        pytest.param("for", False, id="unfolded-for"),
        pytest.param("for", True, id="folded-for"),
    ],
)
def test_control_flow_includes_resultless_callable_local_wire(
    kind: str,
    folded: bool,
) -> None:
    """Folded and unfolded control-flow borders include child call wires."""
    graph = _build_resultless_control_flow_graph(kind)
    map_analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=folded,
        fold_ifs=folded,
    )
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=folded,
        fold_ifs=folded,
    )
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 2
    assert qubit_names == {0: "outer", 1: "local"}
    control_flow = circuit.children[1]
    assert isinstance(control_flow, (VFoldedBlock, VUnfoldedSequence))
    assert control_flow.affected_qubits == [1]


@pytest.mark.parametrize("folded", [False, True])
def test_for_border_includes_direct_body_allocation(folded: bool) -> None:
    """Folded and unfolded for borders include direct body allocations."""
    graph = _build_local_qinit_for_graph()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=folded,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 1
    assert qubit_names == {0: "loop_local"}
    loop = circuit.children[0]
    assert isinstance(loop, (VFoldedBlock, VUnfoldedSequence))
    assert loop.affected_qubits == [0]


def test_zero_length_callable_allocation_adds_no_wire() -> None:
    """A folded callable with a zero-width vector has an empty footprint."""
    graph = _build_zero_length_callable_for_graph()
    map_analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=True,
    )
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=True,
    )
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 0
    assert qubit_map == {}
    assert qubit_names == {}
    loop = circuit.children[0]
    assert isinstance(loop, VFoldedBlock)
    assert loop.affected_qubits == []


@pytest.mark.parametrize(
    ("kind", "inline", "expand_composite"),
    [
        pytest.param("invoke", False, False, id="opaque-invoke"),
        pytest.param("controlled", True, False, id="controlled-u"),
        pytest.param("inverse", False, True, id="expanded-inverse"),
    ],
)
def test_zero_width_callable_results_add_no_phantom_wire(
    kind: str,
    inline: bool,
    expand_composite: bool,
) -> None:
    """Zero-width pass-through results never allocate an unlabeled wire."""
    graph, expected_names = _build_zero_width_result_graph(kind)
    map_analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=inline,
        expand_composite=expand_composite,
    )
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=inline,
        expand_composite=expand_composite,
    )
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == len(expected_names)
    assert qubit_names == expected_names
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    for node in _iter_visual_nodes(circuit.children):
        wires = (
            node.qubit_indices
            if isinstance(node, VGate)
            else getattr(node, "affected_qubits", [])
        )
        assert all(0 <= wire < num_qubits for wire in wires)


def test_opaque_wider_result_allocates_only_missing_named_wires() -> None:
    """A wider opaque result adds named wires without out-of-range indices."""
    graph, result_vector = _build_opaque_widening_result_graph()
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 3
    assert set(qubit_names) == set(range(num_qubits))
    assert all(qubit_names.values())
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    result_wires = [
        qubit_map[f"{result_vector.logical_id}_[{index}]"] for index in range(3)
    ]
    assert result_wires == [0, 1, 2]
    gates = [node for node in circuit.children if isinstance(node, VGate)]
    assert [gate.qubit_indices for gate in gates] == [[0, 1, 2]]


def test_opaque_wider_result_does_not_claim_an_unrelated_neighbor_wire() -> None:
    """A partial result alias never consumes an existing unrelated wire."""
    graph, result_vector = _build_opaque_widening_result_graph(
        occupy_next_wire=True,
    )
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 4
    assert qubit_names[2] == "unrelated"
    assert set(qubit_names) == set(range(num_qubits))
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    result_wires = [
        qubit_map[f"{result_vector.logical_id}_[{index}]"] for index in range(3)
    ]
    assert result_wires == [0, 1, 3]
    gates = [node for node in circuit.children if isinstance(node, VGate)]
    assert [gate.qubit_indices for gate in gates] == [[0, 1, 3]]


@pytest.mark.parametrize("kind", ["for", "for-items"])
@pytest.mark.parametrize("folded", [False, True])
def test_zero_trip_loop_body_adds_no_phantom_wire(
    kind: str,
    folded: bool,
) -> None:
    """Concrete zero-trip loops skip body-local quantum allocations."""
    graph = (
        _build_zero_trip_qinit_for_graph()
        if kind == "for"
        else _build_bound_empty_qinit_for_items_graph()
    )
    map_analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=False,
        fold_loops=folded,
    )
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=False,
        fold_loops=folded,
    )
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 0
    assert qubit_map == {}
    assert qubit_names == {}
    assert len(circuit.children) == 1
    assert isinstance(circuit.children[0], VSkip)


def test_root_quantum_input_registers_a_physical_wire() -> None:
    """A root Block quantum input is drawable without a QInit operation."""
    graph = _build_root_quantum_input_graph()
    input_qubit = graph.input_values[0]
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 1
    assert qubit_names == {0: "input"}
    assert qubit_map[input_qubit.logical_id] == 0
    [gate] = [node for node in circuit.children if isinstance(node, VGate)]
    assert gate.label == r"$H$"
    assert gate.qubit_indices == [0]


@pytest.mark.parametrize("strided_view", [False, True])
def test_symbolic_element_pass_through_reuses_an_existing_wire(
    strided_view: bool,
) -> None:
    """Opaque symbolic element calls never allocate a phantom result wire."""
    graph, expected_wire = _build_symbolic_element_opaque_graph(
        strided_view=strided_view
    )
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 5
    assert qubit_names == {index: f"q[{index}]" for index in range(5)}
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    gates = [node for node in circuit.children if isinstance(node, VGate)]
    assert [gate.qubit_indices for gate in gates] == [
        [expected_wire],
        [expected_wire],
    ]


def test_unresolved_whole_slice_reuses_root_wires_conservatively() -> None:
    """An unresolved whole slice never materializes phantom view wires."""
    graph = _build_unresolved_slice_opaque_graph()
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 5
    assert qubit_names == {index: f"q[{index}]" for index in range(5)}
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    [call] = [node for node in circuit.children if isinstance(node, VGate)]
    assert call.qubit_indices == list(range(5))


@pytest.mark.parametrize(
    ("result_size", "expected_num_qubits"),
    [(2, 5), (3, 6)],
)
def test_unresolved_slice_fresh_opaque_result_preserves_and_extends_wires(
    result_size: int,
    expected_num_qubits: int,
) -> None:
    """Fresh opaque results keep root candidates and allocate only widening."""
    graph = _build_unresolved_slice_opaque_graph(
        fresh_result_size=result_size,
    )
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    expected_wires = list(range(expected_num_qubits))
    assert num_qubits == expected_num_qubits
    assert set(qubit_names) == set(expected_wires)
    assert all(qubit_names.values())
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    gates = [node for node in circuit.children if isinstance(node, VGate)]
    followup_wires = expected_wires if result_size == 2 else [5]
    assert [gate.qubit_indices for gate in gates] == [
        expected_wires,
        followup_wires,
    ]
    if result_size == 3:
        assert qubit_names[5] == "fresh[2]"


def test_unresolved_whole_slice_inline_body_reuses_root_wires() -> None:
    """An inline body preserves conservative unresolved-slice aliases."""
    graph = _build_unresolved_slice_inline_graph()
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 5
    assert qubit_names == {index: f"q[{index}]" for index in range(5)}
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    [call] = [node for node in circuit.children if isinstance(node, VInlineBlock)]
    assert call.affected_qubits == list(range(5))
    gates = [
        node
        for node in _iter_visual_nodes(call.children)
        if isinstance(node, VGate) and node.gate_type is GateOperationType.H
    ]
    assert [gate.qubit_indices for gate in gates] == [list(range(5))]


@pytest.mark.parametrize("leaf_kind", ["inline", "opaque"])
def test_nested_unresolved_slice_inline_bodies_reuse_all_root_wires(
    leaf_kind: str,
) -> None:
    """Nested forwarding keeps an unresolved slice conservative."""
    graph = _build_nested_unresolved_slice_inline_graph(leaf_kind)
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 5
    assert qubit_names == {index: f"q[{index}]" for index in range(5)}
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    inline_blocks = [
        node
        for node in _iter_visual_nodes(circuit.children)
        if isinstance(node, VInlineBlock)
    ]
    assert len(inline_blocks) == (2 if leaf_kind == "inline" else 1)
    assert all(block.affected_qubits == list(range(5)) for block in inline_blocks)
    gates = [
        node for node in _iter_visual_nodes(circuit.children) if isinstance(node, VGate)
    ]
    if leaf_kind == "inline":
        assert [
            gate.qubit_indices
            for gate in gates
            if gate.gate_type is GateOperationType.H
        ] == [list(range(5))]
    else:
        [opaque_call] = [gate for gate in gates if gate.label == "OPAQUE_SLICE_LEAF"]
        assert opaque_call.qubit_indices == list(range(5))


@pytest.mark.parametrize("leaf_kind", ["inline", "opaque"])
def test_nested_unresolved_slice_widening_preserves_exact_suffix(
    leaf_kind: str,
) -> None:
    """Nested calls preserve conservative roots and an exact new suffix."""
    graph = _build_nested_unresolved_slice_widening_graph(leaf_kind)
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    expected_wires = list(range(6))
    assert num_qubits == 6
    assert set(qubit_names) == set(expected_wires)
    assert qubit_names[5] == "fresh[2]"
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    inline_blocks = [
        node
        for node in _iter_visual_nodes(circuit.children)
        if isinstance(node, VInlineBlock)
    ]
    assert len(inline_blocks) == (2 if leaf_kind == "inline" else 1)
    assert all(block.affected_qubits == expected_wires for block in inline_blocks)

    gates = [
        node for node in _iter_visual_nodes(circuit.children) if isinstance(node, VGate)
    ]
    [widening_call] = [gate for gate in gates if gate.label == "OPAQUE_SLICE_WIDENING"]
    assert widening_call.qubit_indices == expected_wires
    assert [
        gate.qubit_indices for gate in gates if gate.gate_type is GateOperationType.H
    ] == [[5]]
    assert [
        gate.qubit_indices for gate in gates if gate.gate_type is GateOperationType.Z
    ] == [[5]]
    if leaf_kind == "inline":
        assert [
            gate.qubit_indices
            for gate in gates
            if gate.gate_type is GateOperationType.X
        ] == [[5]]
    else:
        [opaque_forward] = [
            gate for gate in gates if gate.label == "OPAQUE_WIDENED_SLICE"
        ]
        assert opaque_forward.qubit_indices == expected_wires


def test_legacy_stack_parameter_name_does_not_shadow_compile_time_binding() -> None:
    """The former internal stack key remains a valid user parameter name."""
    graph = _build_legacy_stack_parameter_graph()
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    gate_types = [
        node.gate_type
        for node in _iter_visual_nodes(circuit.children)
        if isinstance(node, VGate) and node.gate_type is not None
    ]
    assert gate_types == [GateOperationType.X]
    assert not any(
        isinstance(node, VUnfoldedSequence)
        for node in _iter_visual_nodes(circuit.children)
    )


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_concrete_self_recursive_invoke_unrolls_without_recursion_error(
    depth: int,
) -> None:
    """A concrete recursive countdown reaches its single base-case gate."""
    graph = _draw_recursive_countdown._build_graph_for_visualization(k=depth)
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 1
    assert qubit_names == {0: "q"}
    gates = [
        node for node in _iter_visual_nodes(circuit.children) if isinstance(node, VGate)
    ]
    assert [
        gate.qubit_indices for gate in gates if gate.gate_type is GateOperationType.H
    ] == [[0]]
    assert [
        gate.qubit_indices for gate in gates if gate.gate_type is GateOperationType.RX
    ] == [[0]]
    assert [gate.qubit_indices for gate in gates if gate.label == "M"] == [[0]]


def test_symbolic_self_recursive_invoke_falls_back_to_a_box() -> None:
    """An unresolved recursive state stops inline expansion at one cycle."""
    graph = _draw_recursive_countdown._build_graph_for_visualization()
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 1
    assert qubit_names == {0: "q"}
    assert all(0 <= wire < num_qubits for wire in qubit_map.values())
    recursive_boxes = [
        node
        for node in _iter_visual_nodes(circuit.children)
        if isinstance(node, VGate) and "RECURSIVE_DRAW_HELPER" in node.label
    ]
    assert len(recursive_boxes) == 1
    assert recursive_boxes[0].qubit_indices == [0]


def test_branching_nonterminating_recursion_uses_shared_expansion_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A branching recursive tree stops at one shared expansion budget."""
    monkeypatch.setattr(
        "qamomile.circuit.visualization.analyzer._MAX_RECURSIVE_INLINE_EXPANSIONS",
        8,
    )
    # Keep the path-depth fallback small so a broken shared budget fails with
    # a bounded tree instead of exhausting CI memory before the assertions.
    monkeypatch.setattr(
        "qamomile.circuit.visualization.analyzer._MAX_RECURSIVE_INLINE_STATES",
        8,
    )
    graph = _draw_branching_nonterminating_recursion._build_graph_for_visualization(k=1)
    map_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    qubit_map, qubit_names, num_qubits = map_analyzer.build_qubit_map(graph)
    visual_analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
    circuit = visual_analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    nodes = list(_iter_visual_nodes(circuit.children))
    recursive_inlines = [
        node
        for node in nodes
        if isinstance(node, VInlineBlock)
        and node.label == "_branching_nonterminating_draw_helper"
    ]
    recursive_boxes = [
        node
        for node in nodes
        if isinstance(node, VGate)
        and "BRANCHING_NONTERMINATING_DRAW_HELPER" in node.label
    ]

    assert num_qubits == 1
    assert qubit_names == {0: "q"}
    assert all(wire == 0 for wire in qubit_map.values())
    assert len(recursive_inlines) <= 20
    assert recursive_boxes
    assert all(box.qubit_indices == [0] for box in recursive_boxes)
    assert len(nodes) < 50


def test_tuple_invoke_input_resolves_quantum_leaf_in_box_and_inline_body() -> None:
    """TupleValue inputs keep their quantum leaf on its allocated wire."""
    graph = _build_structural_input_graph()
    caller_qubit = graph.operations[0].results[0]

    for inline in (False, True):
        analyzer = CircuitAnalyzer(
            graph,
            DEFAULT_STYLE,
            inline=inline,
            fold_loops=False,
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)

        assert num_qubits == 1, (inline, num_qubits, qubit_names)
        assert qubit_names == {0: "q"}
        assert qubit_map[caller_qubit.logical_id] == 0

        circuit = analyzer.build_visual_ir(
            graph,
            qubit_map,
            qubit_names,
            num_qubits,
        )
        invoke_node = circuit.children[1]
        if inline:
            assert isinstance(invoke_node, VInlineBlock)
            assert invoke_node.affected_qubits == [0]
            assert len(invoke_node.children) == 1
            child_gate = invoke_node.children[0]
            assert isinstance(child_gate, VGate)
            assert child_gate.label == r"$H$"
            assert child_gate.qubit_indices == [0]
        else:
            assert isinstance(invoke_node, VGate)
            assert invoke_node.qubit_indices == [0]


def test_mixed_tuple_invoke_input_forwards_quantum_and_classical_leaves() -> None:
    """Tuple inputs resolve both a qubit wire and its bound gate parameter."""
    graph = _build_mixed_tuple_input_graph()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 1
    assert qubit_names == {0: "q"}
    invoke_node = circuit.children[1]
    assert isinstance(invoke_node, VInlineBlock)
    child_gate = invoke_node.children[0]
    assert isinstance(child_gate, VGate)
    assert child_gate.gate_type is GateOperationType.RZ
    assert child_gate.qubit_indices == [0]
    assert child_gate.has_param
    assert "0.5" in child_gate.label


def test_tuple_invoke_output_order_preserves_each_source_wire() -> None:
    """Invoke output provenance wins over positional operand/result pairing."""
    graph, invoke = _build_swapped_tuple_io_graph()
    result = invoke.results[0]
    assert isinstance(result, TupleValue)

    for inline in (False, True):
        analyzer = CircuitAnalyzer(
            graph,
            DEFAULT_STYLE,
            inline=inline,
            fold_loops=False,
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)

        assert num_qubits == 2
        assert qubit_names == {0: "first", 1: "second"}
        assert [qubit_map[element.logical_id] for element in result.elements] == [
            1,
            0,
        ]


@pytest.mark.parametrize("inline", [False, True])
def test_passthrough_dict_allows_different_formal_and_actual_entries(
    inline: bool,
) -> None:
    """A symbolic empty Dict formal may return its populated caller actual."""
    graph = _build_passthrough_dict_graph()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=inline,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 1
    assert qubit_names == {0: "q"}
    assert len(circuit.children) == 2


def test_body_backed_invoke_includes_new_result_wires_in_box_and_inline() -> None:
    """Invoke geometry includes callee-owned result wires alongside inputs."""
    graph = _build_input_and_new_output_graph()

    for inline in (False, True):
        analyzer = CircuitAnalyzer(
            graph,
            DEFAULT_STYLE,
            inline=inline,
            fold_loops=False,
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        circuit = analyzer.build_visual_ir(
            graph,
            qubit_map,
            qubit_names,
            num_qubits,
        )

        assert num_qubits == 2
        assert qubit_names == {0: "q", 1: "new_qubit"}
        invoke_node = circuit.children[1]
        if inline:
            assert isinstance(invoke_node, VInlineBlock)
            assert invoke_node.affected_qubits == [0, 1]
        else:
            assert isinstance(invoke_node, VGate)
            assert invoke_node.qubit_indices == [0, 1]


def test_repeated_allocator_invokes_keep_caller_local_wires() -> None:
    """Inlining one allocator body twice preserves two distinct result wires."""
    for inline in (False, True):
        graph = _invoke_allocator_twice._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(
            graph,
            DEFAULT_STYLE,
            inline=inline,
            fold_loops=False,
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        circuit = analyzer.build_visual_ir(
            graph,
            qubit_map,
            qubit_names,
            num_qubits,
        )

        assert num_qubits == 2
        assert qubit_names == {0: "allocated", 1: "allocated"}
        if inline:
            calls = [
                node for node in circuit.children if isinstance(node, VInlineBlock)
            ]
            assert [node.affected_qubits for node in calls] == [[0], [1]]
        else:
            calls = [
                node
                for node in circuit.children
                if isinstance(node, VGate) and node.label == "_ALLOCATE_ONE_QUBIT"
            ]
            assert [node.qubit_indices for node in calls] == [[0], [1]]
        measurements = [
            node
            for node in circuit.children
            if isinstance(node, VGate) and node.label == "M"
        ]
        assert [node.qubit_indices for node in measurements] == [[0], [1]]


def test_nested_allocator_result_expands_parent_inline_border() -> None:
    """A parent inline border includes wires allocated by a nested call."""
    graph = _draw_nested_allocator._build_graph_for_visualization()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 2
    assert qubit_names == {0: "input", 1: "nested"}
    outer = next(node for node in circuit.children if isinstance(node, VInlineBlock))
    assert outer.affected_qubits == [0, 1]
    inner = next(node for node in outer.children if isinstance(node, VInlineBlock))
    assert inner.affected_qubits == [0, 1]


def test_call_scope_survives_compile_time_if_operation_cloning() -> None:
    """Compile-time IF substitution preserves nested callable wire scope."""
    graph = _draw_compile_time_if_allocator._build_graph_for_visualization()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 2
    assert qubit_names == {0: "conditional_input", 1: "nested"}
    outer = next(node for node in circuit.children if isinstance(node, VInlineBlock))
    assert outer.affected_qubits == [0, 1]
    inner = next(node for node in outer.children if isinstance(node, VInlineBlock))
    assert inner.affected_qubits == [0, 1]
    gates = [
        node for node in _iter_visual_nodes(circuit.children) if isinstance(node, VGate)
    ]
    assert [
        gate.qubit_indices for gate in gates if gate.gate_type is GateOperationType.X
    ] == [[0]]
    assert not any(gate.gate_type is GateOperationType.H for gate in gates)
    assert [gate.qubit_indices for gate in gates if gate.label == "M"] == [
        [0],
        [1],
    ]


def test_compile_time_if_registers_only_selected_branch_qubits() -> None:
    """Inline mapping omits qubits owned only by an unselected branch."""
    graph = _draw_selected_branch_ancilla._build_graph_for_visualization()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 2
    assert qubit_names == {0: "q", 1: "selected"}
    inline_call = next(
        node for node in circuit.children if isinstance(node, VInlineBlock)
    )
    assert inline_call.affected_qubits == [0, 1]
    gates = [
        node for node in _iter_visual_nodes(circuit.children) if isinstance(node, VGate)
    ]
    assert [gate.qubit_indices for gate in gates if gate.label == "M"] == [
        [1],
        [0],
    ]


def test_loop_value_selects_only_one_callable_compile_time_branch() -> None:
    """Loop bindings omit the unselected callable branch and its ancilla."""
    graph = _draw_loop_selected_branch_ancilla._build_graph_for_visualization()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=True,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 2
    assert qubit_names == {0: "q", 1: "selected"}
    loop = next(node for node in circuit.children if isinstance(node, VFoldedBlock))
    assert loop.affected_qubits == [0, 1]


def test_symbolic_callable_parameter_keeps_both_branches() -> None:
    """A forwarded symbolic name never becomes a compile-time binding."""
    graph = _draw_symbolic_branch_ancilla._build_graph_for_visualization()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 3
    assert qubit_names == {0: "q", 1: "selected", 2: "unselected"}
    inline_call = next(
        node for node in circuit.children if isinstance(node, VInlineBlock)
    )
    branch = next(
        node for node in inline_call.children if isinstance(node, VUnfoldedSequence)
    )
    assert branch.affected_qubits == [1, 2]


def test_nested_parameter_name_shadowing_keeps_symbolic_inner_branch() -> None:
    """An outer same-name binding never folds an inner symbolic parameter."""
    graph = _draw_shadowed_symbolic_branch._build_graph_for_visualization()
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=True,
        fold_loops=True,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 3
    assert qubit_names == {0: "q", 1: "selected", 2: "unselected"}
    loop = next(node for node in circuit.children if isinstance(node, VFoldedBlock))
    assert loop.affected_qubits == [0, 1, 2]


def test_callable_local_array_element_result_reuses_its_source_wire() -> None:
    """An element returned from a local register never creates a phantom wire."""
    graph = _use_allocated_element._build_graph_for_visualization()
    invoke = next(op for op in graph.operations if isinstance(op, InvokeOperation))
    result = invoke.results[0]

    for inline in (False, True):
        analyzer = CircuitAnalyzer(
            graph,
            DEFAULT_STYLE,
            inline=inline,
            fold_loops=False,
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        circuit = analyzer.build_visual_ir(
            graph,
            qubit_map,
            qubit_names,
            num_qubits,
        )

        result_wire = 2 if inline else 0
        assert num_qubits == (4 if inline else 1)
        assert analyzer._resolve_operand_to_qubit_indices(
            result,
            qubit_map,
        ) == [result_wire]
        gates = [
            node
            for node in _iter_visual_nodes(circuit.children)
            if isinstance(node, VGate)
        ]
        assert [
            gate.qubit_indices
            for gate in gates
            if gate.gate_type is GateOperationType.H
        ] == [[result_wire]]
        assert [gate.qubit_indices for gate in gates if gate.label == "M"] == [
            [result_wire]
        ]
        call = circuit.children[0]
        if inline:
            assert isinstance(call, VInlineBlock)
            assert call.affected_qubits == [0, 1, 2, 3]
        else:
            assert isinstance(call, VGate)
            assert call.qubit_indices == [0]


def test_callable_local_array_slice_result_reuses_its_source_wires() -> None:
    """A slice returned from a local register aliases its internal wires."""
    graph = _use_allocated_slice._build_graph_for_visualization()
    invoke = next(op for op in graph.operations if isinstance(op, InvokeOperation))
    result = invoke.results[0]

    for inline in (False, True):
        analyzer = CircuitAnalyzer(
            graph,
            DEFAULT_STYLE,
            inline=inline,
            fold_loops=False,
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        circuit = analyzer.build_visual_ir(
            graph,
            qubit_map,
            qubit_names,
            num_qubits,
        )

        result_wires = [1, 2] if inline else [0, 1]
        assert num_qubits == (4 if inline else 2)
        assert (
            analyzer._resolve_operand_to_qubit_indices(
                result,
                qubit_map,
            )
            == result_wires
        )
        gates = [
            node
            for node in _iter_visual_nodes(circuit.children)
            if isinstance(node, VGate)
        ]
        assert sorted(
            gate.qubit_indices
            for gate in gates
            if gate.gate_type is GateOperationType.H
        ) == [[wire] for wire in result_wires]
        assert [gate.qubit_indices for gate in gates if gate.label == "M"] == [
            result_wires
        ]
        call = circuit.children[0]
        if inline:
            assert isinstance(call, VInlineBlock)
            assert call.affected_qubits == [0, 1, 2, 3]
        else:
            assert isinstance(call, VGate)
            assert call.qubit_indices == [0, 1]


def test_partitioned_callable_local_slices_keep_disjoint_source_wires() -> None:
    """Disjoint slice results preserve one four-wire internal allocation."""
    graph = _use_partitioned_slices._build_graph_for_visualization()
    invoke = next(op for op in graph.operations if isinstance(op, InvokeOperation))

    for inline in (False, True):
        analyzer = CircuitAnalyzer(
            graph,
            DEFAULT_STYLE,
            inline=inline,
            fold_loops=False,
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        circuit = analyzer.build_visual_ir(
            graph,
            qubit_map,
            qubit_names,
            num_qubits,
        )

        assert num_qubits == 4
        assert [
            analyzer._resolve_operand_to_qubit_indices(result, qubit_map)
            for result in invoke.results
        ] == [[0, 1], [2, 3]]
        gates = [
            node
            for node in _iter_visual_nodes(circuit.children)
            if isinstance(node, VGate)
        ]
        assert sorted(
            gate.qubit_indices
            for gate in gates
            if gate.gate_type is GateOperationType.H
        ) == [[0], [1]]
        assert sorted(
            gate.qubit_indices
            for gate in gates
            if gate.gate_type is GateOperationType.X
        ) == [[2], [3]]
        assert [gate.qubit_indices for gate in gates if gate.label == "M"] == [
            [0, 1],
            [2, 3],
        ]
        call = circuit.children[0]
        if inline:
            assert isinstance(call, VInlineBlock)
            assert call.affected_qubits == [0, 1, 2, 3]
        else:
            assert isinstance(call, VGate)
            assert call.qubit_indices == [0, 1, 2, 3]


@pytest.mark.parametrize("kind", ["tuple", "dict"])
def test_boxed_structural_invoke_registers_new_quantum_result_wires(
    kind: str,
) -> None:
    """A boxed Invoke shows newly allocated quantum leaves in its result."""
    graph, _ = _build_structural_result_graph(kind)
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=False,
        fold_loops=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    circuit = analyzer.build_visual_ir(
        graph,
        qubit_map,
        qubit_names,
        num_qubits,
    )

    assert num_qubits == 2
    assert qubit_names == {0: "q[0]", 1: "q[1]"}
    gates = [node for node in circuit.children if isinstance(node, VGate)]
    assert [gate.qubit_indices for gate in gates] == [[0, 1], [0, 1], [0, 1]]


def test_tuple_invoke_result_aliases_nested_quantum_wires() -> None:
    """TupleValue invocation results keep nested arrays on source wires."""
    graph, call_result = _build_structural_result_graph("tuple")
    _assert_structural_result_uses_source_wires(graph, call_result)


def test_dict_invoke_result_aliases_nested_quantum_wires() -> None:
    """DictValue invocation results keep nested arrays on source wires."""
    graph, call_result = _build_structural_result_graph("dict")
    _assert_structural_result_uses_source_wires(graph, call_result)
