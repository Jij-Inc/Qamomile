"""Tests for post-fold array element bounds validation."""

import pytest

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.control_flow import (
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
from qamomile.circuit.ir.value import ArrayValue, Value
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
