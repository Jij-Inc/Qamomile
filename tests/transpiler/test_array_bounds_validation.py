"""Tests for post-fold array element bounds validation."""

from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    RegionArg,
)
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.operation.slice_array import SliceArrayOperation
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
)
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes.array_bounds_validation import (
    ArrayBoundsValidationPass,
)
from qamomile.circuit.transpiler.passes.partial_eval import PartialEvaluationPass
from qamomile.circuit.transpiler.value_resolver import ValueResolver


def _uint(name: str, value: int) -> Value:
    """Create one constant UInt value for synthetic IR tests.

    Args:
        name (str): Display name for the synthetic value.
        value (int): Constant integer payload.

    Returns:
        Value: Constant UInt IR value.
    """
    return Value(type=UIntType(), name=name).with_const(value)


def _view_element(*, view_length: int, index: int) -> tuple[Value, BinOp]:
    """Create one float element access through a named compile-time view.

    Args:
        view_length (int): Resolved length of the sliced view.
        index (int): Constant view-local element index.

    Returns:
        tuple[Value, BinOp]: Element value and an operation that consumes it.
    """
    root_length = _uint("phases_length", 1)
    root = ArrayValue(
        type=FloatType(),
        name="phases",
        shape=(root_length,),
    ).with_array_runtime_metadata(const_array=(0.25,))
    view = ArrayValue(
        type=FloatType(),
        name="phases[slice]",
        shape=(_uint("phase_count", view_length),),
        slice_of=root,
        slice_start=_uint("slice_start", 0),
        slice_step=_uint("slice_step", 1),
    )
    element = Value(
        type=FloatType(),
        name="selected_phase",
        parent_array=view,
        element_indices=(_uint("phase_index", index),),
    )
    operation = BinOp(
        kind=BinOpKind.MUL,
        operands=[element, Value(type=FloatType(), name="two").with_const(2.0)],
        results=[Value(type=FloatType(), name="doubled")],
    )
    return element, operation


def _post_loop_carried_index_block(
    *,
    extent: int,
    trips: int,
    read_in_body: bool = False,
) -> Block:
    """Build a loop whose carried result indexes an array after the loop.

    Args:
        extent (int): Concrete extent of the indexed array.
        trips (int): Concrete number of carried increments.
        read_in_body (bool): Whether the loop body also indexes the array with
            the entering carried value. Defaults to ``False``.

    Returns:
        Block: Affine block containing the loop and post-loop access.
    """
    initial = _uint("initial_index", 0)
    block_arg = Value(type=UIntType(), name="carried_index")
    yielded = Value(type=UIntType(), name="next_index")
    result = Value(type=UIntType(), name="final_index")
    increment = BinOp(
        kind=BinOpKind.ADD,
        operands=[block_arg, _uint("increment", 1)],
        results=[yielded],
    )
    values = ArrayValue(
        type=FloatType(),
        name="values",
        shape=(_uint("values_length", extent),),
    )
    body_operations: list[BinOp] = []
    if read_in_body:
        body_element = Value(
            type=FloatType(),
            name="values[carried_index]",
            parent_array=values,
            element_indices=(block_arg,),
        )
        body_operations.append(
            BinOp(
                kind=BinOpKind.MUL,
                operands=[
                    body_element,
                    Value(type=FloatType(), name="body_two").with_const(2.0),
                ],
                results=[Value(type=FloatType(), name="body_doubled")],
            )
        )
    body_operations.append(increment)
    loop = ForOperation(
        operands=[_uint("start", 0), _uint("stop", trips), _uint("step", 1)],
        results=[result],
        loop_var="iteration",
        loop_var_value=Value(type=UIntType(), name="iteration"),
        operations=body_operations,
        region_args=(
            RegionArg(
                var_name="index",
                init=initial,
                block_arg=block_arg,
                yielded=yielded,
                result=result,
            ),
        ),
    )
    element = Value(
        type=FloatType(),
        name="values[final_index]",
        parent_array=values,
        element_indices=(result,),
    )
    consume = BinOp(
        kind=BinOpKind.MUL,
        operands=[element, Value(type=FloatType(), name="two").with_const(2.0)],
        results=[Value(type=FloatType(), name="doubled")],
    )
    return Block(operations=[loop, consume], kind=BlockKind.AFFINE)


def _nested_bound_array_block(
    *,
    stop: int,
    constant_body_index: int | None = None,
) -> Block:
    """Build a nested loop whose inner stop comes from a bound array element.

    Args:
        stop (int): Compile-time inner-loop stop stored in the bound array.
        constant_body_index (int | None): Fixed body index, or ``None`` to use
            the inner induction value. Defaults to ``None``.

    Returns:
        Block: Affine block containing the outer and inner loops.
    """
    outer_index = Value(type=UIntType(), name="outer_index")
    inner_index = Value(type=UIntType(), name="inner_index")
    bounds = ArrayValue(
        type=UIntType(),
        name="bounds",
        shape=(_uint("bounds_length", 1),),
    ).with_array_runtime_metadata(const_array=(stop,))
    inner_stop = Value(
        type=UIntType(),
        name="bounds[outer_index]",
        parent_array=bounds,
        element_indices=(outer_index,),
    )
    values = ArrayValue(
        type=FloatType(),
        name="values",
        shape=(_uint("values_length", 1),),
    ).with_array_runtime_metadata(const_array=(0.25,))
    element_index = (
        inner_index
        if constant_body_index is None
        else _uint("constant_body_index", constant_body_index)
    )
    element = Value(
        type=FloatType(),
        name="values[index]",
        parent_array=values,
        element_indices=(element_index,),
    )
    consume = BinOp(
        kind=BinOpKind.MUL,
        operands=[element, Value(type=FloatType(), name="two").with_const(2.0)],
        results=[Value(type=FloatType(), name="doubled")],
    )
    inner = ForOperation(
        operands=[_uint("inner_start", 0), inner_stop, _uint("inner_step", 1)],
        loop_var="inner_index",
        loop_var_value=inner_index,
        operations=[consume],
    )
    outer = ForOperation(
        operands=[
            _uint("outer_start", 0),
            _uint("outer_stop", 1),
            _uint("outer_step", 1),
        ],
        loop_var="outer_index",
        loop_var_value=outer_index,
        operations=[inner],
    )
    return Block(operations=[outer], kind=BlockKind.AFFINE)


def _transpile_serialized(
    kernel: Any,
    *,
    bindings: dict[str, Any],
    parameters: list[str] | None = None,
) -> Any:
    """Round-trip and transpile one qkernel with Qiskit.

    Args:
        kernel (Any): Qkernel template to serialize before binding.
        bindings (dict[str, Any]): Compile-time bindings applied after restore.
        parameters (list[str] | None): Runtime parameter names. Defaults to
            ``None``.

    Returns:
        Any: Qiskit executable produced by the transpiler.
    """
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    restored = deserialize(serialize(kernel))
    return QiskitTranspiler().transpile(
        restored,
        bindings=bindings,
        parameters=parameters,
    )


@qmc.qkernel
def _serialized_for_items_view_access(
    values: qmc.Vector[qmc.Float],
    count: qmc.UInt,
    items: qmc.Dict[qmc.UInt, qmc.UInt],
) -> qmc.Bit:
    """Read a bound-length view only from an items-loop body.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime rotation values.
        count (qmc.UInt): Compile-time view length.
        items (qmc.Dict[qmc.UInt, qmc.UInt]): Compile-time loop entries.

    Returns:
        qmc.Bit: Measurement of the rotated target.
    """
    target = qmc.qubit("target")
    selected = values[0:count]
    for _key, _value in qmc.items(items):
        target = qmc.rx(target, selected[0])
    return qmc.measure(target)


@qmc.qkernel
def _serialized_carried_array_index(
    values: qmc.Vector[qmc.Float],
    initial: qmc.UInt,
    increment: qmc.UInt,
    repetitions: qmc.UInt,
) -> qmc.Bit:
    """Index an array through a loop-carried UInt value.

    Args:
        values (qmc.Vector[qmc.Float]): Compile-time rotation values.
        initial (qmc.UInt): First array index.
        increment (qmc.UInt): Per-iteration index increment.
        repetitions (qmc.UInt): Number of loop iterations.

    Returns:
        qmc.Bit: Measurement of the rotated target.
    """
    target = qmc.qubit("target")
    index = initial
    for _iteration in qmc.range(repetitions):
        target = qmc.rx(target, values[index])
        index = index + increment
    return qmc.measure(target)


@qmc.qkernel
def _serialized_induction_array_index(
    values: qmc.Vector[qmc.Float],
    repetitions: qmc.UInt,
) -> qmc.Bit:
    """Index an array with a counted-loop induction value.

    Args:
        values (qmc.Vector[qmc.Float]): Compile-time rotation values.
        repetitions (qmc.UInt): Number of indexed iterations.

    Returns:
        qmc.Bit: Measurement of the rotated target.
    """
    target = qmc.qubit("target")
    for index in qmc.range(repetitions):
        target = qmc.rx(target, values[index])
    return qmc.measure(target)


@qmc.qkernel
def _serialized_for_items_carried_array_index(
    values: qmc.Vector[qmc.Float],
    items: qmc.Dict[qmc.UInt, qmc.UInt],
) -> qmc.Bit:
    """Index an array through a carried items-loop counter.

    Args:
        values (qmc.Vector[qmc.Float]): Compile-time rotation values.
        items (qmc.Dict[qmc.UInt, qmc.UInt]): Compile-time loop entries.

    Returns:
        qmc.Bit: Measurement of the rotated target.
    """
    target = qmc.qubit("target")
    index = qmc.uint(0)
    for _key, _value in qmc.items(items):
        target = qmc.rx(target, values[index])
        index = index + 1
    return qmc.measure(target)


@qmc.qkernel
def _serialized_vector_key_array_index(
    values: qmc.Vector[qmc.Float],
    items: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Bit:
    """Index an array through every bound vector-key first element.

    Args:
        values (qmc.Vector[qmc.Float]): Compile-time rotation values.
        items (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Compile-time mapping
            whose vector keys supply array indices.

    Returns:
        qmc.Bit: Measurement of the rotated target.
    """
    target = qmc.qubit("target")
    for key, _value in qmc.items(items):
        target = qmc.rx(target, values[key[0]])
    return qmc.measure(target)


@qmc.qkernel
def _serialized_vector_key_constant_access(
    items: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Bit:
    """Read a fixed position from every bound vector key.

    Args:
        items (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Compile-time mapping
            whose keys may be shorter than the accessed position.

    Returns:
        qmc.Bit: Measurement of the loop target.
    """
    target = qmc.qubit("target")
    for key, _value in qmc.items(items):
        for _index in qmc.range(key[1]):
            target = qmc.x(target)
    return qmc.measure(target)


def test_value_resolver_does_not_fold_past_empty_view() -> None:
    """An empty view cannot resolve its index zero from a non-empty root."""
    element, _ = _view_element(view_length=0, index=0)

    assert ValueResolver().resolve(element) is None


def test_reachable_empty_view_access_names_root_view_and_extent() -> None:
    """A reachable invalid access reports every user-relevant provenance name."""
    _, operation = _view_element(view_length=0, index=0)
    block = Block(operations=[operation], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError) as exc_info:
        ArrayBoundsValidationPass().run(block)

    message = str(exc_info.value)
    assert "Index 0" in message
    assert "phases[slice]" in message
    assert "root array 'phases'" in message
    assert "extent 'phase_count' resolved to 0" in message
    assert "at least 1" in message


def test_root_bound_is_checked_after_view_affine_mapping() -> None:
    """A locally valid view index must also fit the underlying root array."""
    _, operation = _view_element(view_length=2, index=1)
    block = Block(operations=[operation], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="phases_length.*resolved to 1"):
        ArrayBoundsValidationPass().run(block)


def test_reachable_view_descriptor_must_fit_physical_root() -> None:
    """A concrete view cannot declare more physical slots than its root."""
    root = ArrayValue(
        type=FloatType(),
        name="phases",
        shape=(_uint("phases_length", 1),),
    )
    start = _uint("slice_start", 0)
    step = _uint("slice_step", 1)
    view = ArrayValue(
        type=FloatType(),
        name="phases[slice]",
        shape=(_uint("phase_count", 5),),
        slice_of=root,
        slice_start=start,
        slice_step=step,
    )
    operation = SliceArrayOperation(
        operands=[root, start, step],
        results=[view],
    )
    block = Block(operations=[operation], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError) as exc_info:
        ArrayBoundsValidationPass().run(block)

    message = str(exc_info.value)
    assert "Index 4" in message
    assert "phase_count" in message
    assert "phases_length" in message


def test_top_level_output_element_is_validated() -> None:
    """An array access used only as a public output is still reachable."""
    values = ArrayValue(
        type=FloatType(),
        name="values",
        shape=(_uint("values_length", 0),),
    )
    output = Value(
        type=FloatType(),
        name="values[0]",
        parent_array=values,
        element_indices=(_uint("output_index", 0),),
    )
    block = Block(output_values=[output], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="Index 0.*values"):
        ArrayBoundsValidationPass().run(block)


def test_tuple_output_validates_nested_array_access() -> None:
    """A reachable tuple output recursively validates its element values."""
    element, _ = _view_element(view_length=0, index=0)
    output = TupleValue(name="output", elements=(element,))
    block = Block(output_values=[output], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="Index 0.*phase_count"):
        ArrayBoundsValidationPass().run(block)


def test_dict_capture_validates_nested_array_access() -> None:
    """A selected dictionary capture recursively validates its entries."""
    element, _ = _view_element(view_length=0, index=0)
    capture = DictValue(
        name="capture",
        entries=((_uint("key", 0), element),),
    )
    conditional = IfOperation(
        operands=[Value(type=BitType(), name="condition").with_const(True)],
        true_captures=(capture,),
    )
    block = Block(operations=[conditional], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="Index 0.*phase_count"):
        ArrayBoundsValidationPass().run(block)


def test_array_output_validates_shape_value_access() -> None:
    """An array output recursively validates accesses embedded in its shape."""
    lengths = ArrayValue(
        type=UIntType(),
        name="lengths",
        shape=(_uint("lengths_length", 0),),
    )
    selected_length = Value(
        type=UIntType(),
        name="lengths[0]",
        parent_array=lengths,
        element_indices=(_uint("length_index", 0),),
    )
    output = ArrayValue(
        type=FloatType(),
        name="output",
        shape=(selected_length,),
    )
    block = Block(output_values=[output], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="Index 0.*lengths"):
        ArrayBoundsValidationPass().run(block)


def test_array_output_validates_slice_metadata_access() -> None:
    """An array output recursively validates accesses in slice metadata."""
    offsets = ArrayValue(
        type=UIntType(),
        name="offsets",
        shape=(_uint("offsets_length", 0),),
    )
    selected_start = Value(
        type=UIntType(),
        name="offsets[0]",
        parent_array=offsets,
        element_indices=(_uint("offset_index", 0),),
    )
    root = ArrayValue(
        type=FloatType(),
        name="root",
        shape=(_uint("root_length", 1),),
    )
    output = ArrayValue(
        type=FloatType(),
        name="output",
        shape=(_uint("output_length", 1),),
        slice_of=root,
        slice_start=selected_start,
        slice_step=_uint("slice_step", 1),
    )
    block = Block(output_values=[output], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="Index 0.*offsets"):
        ArrayBoundsValidationPass().run(block)


def test_negative_index_is_rejected_with_unresolved_extent() -> None:
    """A constant negative index is invalid even before its extent resolves."""
    values = ArrayValue(
        type=FloatType(),
        name="values",
        shape=(Value(type=UIntType(), name="values_length"),),
    )
    output = Value(
        type=FloatType(),
        name="values[-1]",
        parent_array=values,
        element_indices=(_uint("output_index", -1),),
    )
    block = Block(output_values=[output], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="Index -1.*non-negative"):
        ArrayBoundsValidationPass().run(block)


def test_owned_block_output_is_specialized_before_validation() -> None:
    """A case output uses its call-site array extent, not its formal shape."""
    formal = ArrayValue(
        type=FloatType(),
        name="formal_values",
        shape=(Value(type=UIntType(), name="formal_length"),),
    )
    case_output = Value(
        type=FloatType(),
        name="formal_values[0]",
        parent_array=formal,
        element_indices=(_uint("case_output_index", 0),),
    )
    case = Block(
        input_values=[formal],
        output_values=[case_output],
        kind=BlockKind.AFFINE,
    )
    actual = ArrayValue(
        type=FloatType(),
        name="actual_values",
        shape=(_uint("actual_length", 0),),
    )
    select = SelectOperation(
        operands=[Value(type=QubitType(), name="index"), actual],
        num_index_qubits=1,
        num_index_args=1,
        case_blocks=[case, case],
    )
    block = Block(operations=[select], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="Index 0.*actual_values"):
        ArrayBoundsValidationPass().run(block)


def test_zero_trip_loop_skips_unreachable_out_of_bounds_access() -> None:
    """A post-fold invalid access in a statically empty loop is not diagnosed."""
    _, operation = _view_element(view_length=0, index=0)
    loop = ForOperation(
        operands=[_uint("start", 0), _uint("stop", 0), _uint("step", 1)],
        loop_var="index",
        loop_var_value=Value(type=UIntType(), name="index"),
        operations=[operation],
    )
    block = Block(operations=[loop], kind=BlockKind.AFFINE)
    folded = PartialEvaluationPass().run(block)

    assert ArrayBoundsValidationPass().run(folded) is folded


def test_zero_trip_loop_skips_unreachable_region_yield() -> None:
    """An invalid body yield is unreachable when the loop executes no times."""
    element, _ = _view_element(view_length=0, index=0)
    init = Value(type=FloatType(), name="init").with_const(0.0)
    block_arg = Value(type=FloatType(), name="carried")
    result = Value(type=FloatType(), name="result")
    loop = ForOperation(
        operands=[_uint("start", 0), _uint("stop", 0), _uint("step", 1)],
        results=[result],
        loop_var="index",
        loop_var_value=Value(type=UIntType(), name="index"),
        operations=[],
        region_args=(
            RegionArg(
                var_name="carried",
                init=init,
                block_arg=block_arg,
                yielded=element,
                result=result,
            ),
        ),
    )
    block = Block(operations=[loop], kind=BlockKind.AFFINE)

    assert ArrayBoundsValidationPass().run(block) is block


def test_constant_if_skips_unreachable_branch_yield() -> None:
    """An invalid merge source in a statically dead branch is not diagnosed."""
    element, _ = _view_element(view_length=0, index=0)
    condition = Value(type=BitType(), name="condition").with_const(True)
    valid = Value(type=FloatType(), name="valid").with_const(0.0)
    conditional = IfOperation(
        operands=[condition],
        results=[Value(type=FloatType(), name="merged")],
        true_operations=[],
        false_operations=[],
        true_yields=[valid],
        false_yields=[element],
    )
    block = Block(operations=[conditional], kind=BlockKind.AFFINE)

    assert ArrayBoundsValidationPass().run(block) is block


def test_constant_if_validates_selected_branch_yield() -> None:
    """An invalid merge source in a statically selected branch is rejected."""
    element, _ = _view_element(view_length=0, index=0)
    condition = Value(type=BitType(), name="condition").with_const(False)
    valid = Value(type=FloatType(), name="valid").with_const(0.0)
    conditional = IfOperation(
        operands=[condition],
        results=[Value(type=FloatType(), name="merged")],
        true_operations=[],
        false_operations=[],
        true_yields=[valid],
        false_yields=[element],
    )
    block = Block(operations=[conditional], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="phase_count.*resolved to 0"):
        ArrayBoundsValidationPass().run(block)


def test_nonempty_loop_validates_its_reachable_body() -> None:
    """The same invalid access is rejected when its loop executes once."""
    _, operation = _view_element(view_length=0, index=0)
    loop = ForOperation(
        operands=[_uint("start", 0), _uint("stop", 1), _uint("step", 1)],
        loop_var="index",
        loop_var_value=Value(type=UIntType(), name="index"),
        operations=[operation],
    )
    block = Block(operations=[loop], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="phase_count.*resolved to 0"):
        ArrayBoundsValidationPass().run(block)


def test_constant_if_validates_selected_branch_capture() -> None:
    """A selected branch capture is a reachable semantic array read."""
    element, _ = _view_element(view_length=0, index=0)
    conditional = IfOperation(
        operands=[Value(type=BitType(), name="condition").with_const(True)],
        true_captures=(element,),
    )
    block = Block(operations=[conditional], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="phase_count.*resolved to 0"):
        ArrayBoundsValidationPass().run(block)


def test_constant_if_skips_unreachable_branch_capture() -> None:
    """A statically dead branch capture is not a reachable array read."""
    element, _ = _view_element(view_length=0, index=0)
    conditional = IfOperation(
        operands=[Value(type=BitType(), name="condition").with_const(True)],
        false_captures=(element,),
    )
    block = Block(operations=[conditional], kind=BlockKind.AFFINE)

    assert ArrayBoundsValidationPass().run(block) is block


@pytest.mark.parametrize("flag", [False, True])
def test_replay_resolves_bound_array_element_if_condition(flag: bool) -> None:
    """Replay validates only the branch selected by a bound array element.

    Args:
        flag (bool): Compile-time condition selected from the bound array.
    """
    outer_index = Value(type=UIntType(), name="outer_index")
    flags = ArrayValue(
        type=BitType(),
        name="flags",
        shape=(_uint("flags_length", 1),),
    ).with_array_runtime_metadata(const_array=(flag,))
    condition = Value(
        type=BitType(),
        name="flags[outer_index]",
        parent_array=flags,
        element_indices=(outer_index,),
    )
    values = ArrayValue(
        type=FloatType(),
        name="values",
        shape=(_uint("values_length", 1),),
    )
    unreachable = Value(
        type=FloatType(),
        name="values[1]",
        parent_array=values,
        element_indices=(_uint("bad_index", 1),),
    )
    conditional = IfOperation(
        operands=[condition],
        true_captures=() if flag else (unreachable,),
        false_captures=(unreachable,) if flag else (),
    )
    outer = ForOperation(
        operands=[
            _uint("outer_start", 0),
            _uint("outer_stop", 1),
            _uint("outer_step", 1),
        ],
        loop_var="outer_index",
        loop_var_value=outer_index,
        operations=[conditional],
    )
    block = Block(operations=[outer], kind=BlockKind.AFFINE)

    assert ArrayBoundsValidationPass().run(block) is block


def test_nonempty_loop_validates_region_capture() -> None:
    """A capture entering an executing range-loop body is validated."""
    element, _ = _view_element(view_length=0, index=0)
    loop = ForOperation(
        operands=[_uint("start", 0), _uint("stop", 1), _uint("step", 1)],
        loop_var="index",
        loop_var_value=Value(type=UIntType(), name="index"),
        captures=(element,),
    )
    block = Block(operations=[loop], kind=BlockKind.AFFINE)

    with pytest.raises(ValidationError, match="phase_count.*resolved to 0"):
        ArrayBoundsValidationPass().run(block)


def test_bound_array_zero_trip_nested_loop_skips_body() -> None:
    """An outer-indexed zero bound makes the inner invalid body unreachable."""
    block = _nested_bound_array_block(stop=0, constant_body_index=1)

    assert ArrayBoundsValidationPass().run(block) is block


def test_bound_array_one_trip_nested_loop_accepts_covered_index() -> None:
    """An outer-indexed one-trip bound resolves the inner induction value."""
    block = _nested_bound_array_block(stop=1)

    assert ArrayBoundsValidationPass().run(block) is block


def test_bound_array_nested_loop_rejects_later_trip_oob() -> None:
    """An outer-indexed two-trip bound exposes the second inner index."""
    block = _nested_bound_array_block(stop=2)

    with pytest.raises(ValidationError, match="Index 1.*values"):
        ArrayBoundsValidationPass().run(block)


def test_zero_trip_loop_skips_unreachable_region_capture() -> None:
    """A capture entering a statically empty range loop remains unread."""
    element, _ = _view_element(view_length=0, index=0)
    loop = ForOperation(
        operands=[_uint("start", 0), _uint("stop", 0), _uint("step", 1)],
        loop_var="index",
        loop_var_value=Value(type=UIntType(), name="index"),
        captures=(element,),
    )
    block = Block(operations=[loop], kind=BlockKind.AFFINE)

    assert ArrayBoundsValidationPass().run(block) is block


def test_serialized_empty_for_items_skips_unreachable_view_access() -> None:
    """An empty bound Dict makes its items body unreachable after restore."""
    executable = _transpile_serialized(
        _serialized_for_items_view_access,
        bindings={"count": 0, "items": {}},
        parameters=["values"],
    )

    assert executable.get_first_circuit() is not None


def test_serialized_nonempty_for_items_validates_view_access() -> None:
    """The same items-body view access is invalid for a nonempty Dict."""
    with pytest.raises(ValidationError, match="count.*resolved to 0"):
        _transpile_serialized(
            _serialized_for_items_view_access,
            bindings={"count": 0, "items": {0: 1}},
            parameters=["values"],
        )


def test_multitrip_carried_zero_index_rejects_empty_array() -> None:
    """A carried index zero is invalid for every trip over an empty array."""
    with pytest.raises(ValidationError, match="Index 0.*values"):
        _transpile_serialized(
            _serialized_carried_array_index,
            bindings={
                "values": [],
                "initial": 0,
                "increment": 0,
                "repetitions": 2,
            },
        )


def test_multitrip_carried_negative_index_is_rejected() -> None:
    """A negative constant remains invalid through a carried index slot."""

    @qmc.qkernel
    def kernel(values: qmc.Vector[qmc.Float]) -> qmc.Bit:
        """Index an array through a negative loop-carried constant.

        Args:
            values (qmc.Vector[qmc.Float]): Compile-time rotation values.

        Returns:
            qmc.Bit: Measurement of the rotated target.
        """
        target = qmc.qubit("target")
        index = qmc.uint(-1)
        for _iteration in qmc.range(2):
            target = qmc.rx(target, values[index])
            index = index + 0
        return qmc.measure(target)

    with pytest.raises(ValidationError, match="Index -1.*non-negative"):
        _transpile_serialized(
            kernel,
            bindings={"values": [0.25]},
        )


def test_multitrip_carried_index_rejects_later_out_of_bounds_value() -> None:
    """A carried increment that escapes the extent is rejected pre-emit."""
    with pytest.raises(ValidationError, match="Index 1.*values"):
        _transpile_serialized(
            _serialized_carried_array_index,
            bindings={
                "values": [0.25],
                "initial": 0,
                "increment": 1,
                "repetitions": 2,
            },
        )


def test_multitrip_carried_index_accepts_covering_extent() -> None:
    """Every carried index is valid when the bound array covers both trips."""
    executable = _transpile_serialized(
        _serialized_carried_array_index,
        bindings={
            "values": [0.25, 0.5],
            "initial": 0,
            "increment": 1,
            "repetitions": 2,
        },
    )

    assert executable.get_first_circuit() is not None


def test_induction_index_rejects_later_out_of_bounds_value() -> None:
    """A loop induction index is checked at every concrete iteration."""
    with pytest.raises(ValidationError, match="Index 1.*values"):
        _transpile_serialized(
            _serialized_induction_array_index,
            bindings={"values": [0.25], "repetitions": 2},
        )


def test_induction_index_accepts_covering_extent() -> None:
    """A loop induction index passes when every iteration is in bounds."""
    executable = _transpile_serialized(
        _serialized_induction_array_index,
        bindings={"values": [0.25, 0.5], "repetitions": 2},
    )

    assert executable.get_first_circuit() is not None


def test_replay_budget_fallback_does_not_publish_symbolic_loop_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skipped large-loop replay leaves its post-loop result unresolved."""
    monkeypatch.setattr(ArrayBoundsValidationPass, "_MAX_REPLAY_TRIPS", 1)
    block = _post_loop_carried_index_block(extent=0, trips=2)

    assert ArrayBoundsValidationPass().run(block) is block


def test_replay_budget_fallback_validates_first_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-budget loop still validates its known first carried value."""
    monkeypatch.setattr(ArrayBoundsValidationPass, "_MAX_REPLAY_TRIPS", 1)
    block = _post_loop_carried_index_block(
        extent=0,
        trips=2,
        read_in_body=True,
    )

    with pytest.raises(ValidationError, match="Index 0.*values"):
        ArrayBoundsValidationPass().run(block)


def test_for_items_budget_fallback_materializes_first_iteration_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-budget items loop binds only its known first iteration."""
    original = ArrayBoundsValidationPass._for_items_iteration_bindings
    call_count = 0

    def spy(
        validation_pass: ArrayBoundsValidationPass,
        operation: ForItemsOperation,
        key: Any,
        value: Any,
    ) -> dict[str, ValueBase]:
        """Count per-entry bindings built during array-bounds validation.

        Args:
            validation_pass (ArrayBoundsValidationPass): Pass under test.
            operation (ForItemsOperation): For-items operation being replayed.
            key (Any): Concrete item key.
            value (Any): Concrete item value.

        Returns:
            dict[str, ValueBase]: Original iteration bindings.
        """
        nonlocal call_count
        call_count += 1
        return original(validation_pass, operation, key, value)

    monkeypatch.setattr(ArrayBoundsValidationPass, "_MAX_REPLAY_TRIPS", 1)
    monkeypatch.setattr(
        ArrayBoundsValidationPass,
        "_for_items_iteration_bindings",
        spy,
    )

    executable = _transpile_serialized(
        _serialized_for_items_carried_array_index,
        bindings={"values": [0.25, 0.5], "items": {0: 10, 1: 20}},
    )

    assert executable.get_first_circuit() is not None
    assert call_count == 1


def test_replay_within_budget_validates_post_loop_region_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact replay publishes a carried result for following bounds checks."""
    monkeypatch.setattr(ArrayBoundsValidationPass, "_MAX_REPLAY_TRIPS", 2)
    block = _post_loop_carried_index_block(extent=0, trips=2)

    with pytest.raises(ValidationError, match="Index 2.*values"):
        ArrayBoundsValidationPass().run(block)


def test_for_items_carried_index_rejects_later_out_of_bounds_value() -> None:
    """A two-entry items loop validates its second carried array index."""
    with pytest.raises(ValidationError, match="Index 1.*values"):
        _transpile_serialized(
            _serialized_for_items_carried_array_index,
            bindings={"values": [0.25], "items": {0: 10, 1: 20}},
        )


def test_for_items_carried_index_accepts_covering_extent() -> None:
    """A two-entry items loop passes when both carried indices are covered."""
    executable = _transpile_serialized(
        _serialized_for_items_carried_array_index,
        bindings={"values": [0.25, 0.5], "items": {0: 10, 1: 20}},
    )

    assert executable.get_first_circuit() is not None


def test_vector_key_rejects_later_out_of_bounds_array_index() -> None:
    """A later vector-key entry cannot escape a bound array extent."""
    with pytest.raises(ValidationError, match="Index 1.*values"):
        _transpile_serialized(
            _serialized_vector_key_array_index,
            bindings={
                "values": [0.25],
                "items": {(0,): 1.0, (1,): 2.0},
            },
        )


def test_vector_key_accepts_covering_array_extent() -> None:
    """All vector-key indices pass when the bound array covers them."""
    executable = _transpile_serialized(
        _serialized_vector_key_array_index,
        bindings={
            "values": [0.25, 0.5],
            "items": {(0,): 1.0, (1,): 2.0},
        },
    )

    assert executable.get_first_circuit() is not None


def test_vector_key_rejects_shorter_later_entry() -> None:
    """Each entry's concrete key length bounds fixed key-element reads."""
    with pytest.raises(ValidationError, match="Index 1.*key"):
        _transpile_serialized(
            _serialized_vector_key_constant_access,
            bindings={"items": {(0, 1): 1.0, (0,): 2.0}},
        )
