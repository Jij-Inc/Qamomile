"""Test structural identity for array physical regions."""

from qamomile.circuit.ir.types.primitives import QubitType, UIntType
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    array_physical_region,
    array_static_length,
    arrays_share_physical_region,
)


def _uint(value: int) -> Value:
    """Create a constant UInt IR value for a shape or slice bound.

    Args:
        value (int): Constant integer payload.

    Returns:
        Value: Constant UInt value.
    """
    return Value(type=UIntType(), name="").with_const(value)


def _root(name: str = "q", size: int = 4) -> ArrayValue:
    """Create a one-dimensional quantum root array.

    Args:
        name (str): Display name. Defaults to ``"q"``.
        size (int): Static array length. Defaults to 4.

    Returns:
        ArrayValue: Fresh root array.
    """
    return ArrayValue(type=QubitType(), name=name, shape=(_uint(size),))


def _view(
    root: ArrayValue,
    *,
    start: int,
    step: int,
    size: int,
) -> ArrayValue:
    """Create a constant-bounded one-dimensional view.

    Args:
        root (ArrayValue): Root array being sliced.
        start (int): Root-space start index.
        step (int): Root-space stride.
        size (int): Number of elements in the view.

    Returns:
        ArrayValue: Slice view with constant bounds.
    """
    return ArrayValue(
        type=QubitType(),
        name="view",
        shape=(_uint(size),),
        slice_of=root,
        slice_start=_uint(start),
        slice_step=_uint(step),
    )


def test_root_and_full_view_share_physical_region() -> None:
    """A root register and its full slice have identical ordered coverage."""
    root = _root()
    full = _view(root, start=0, step=1, size=4)

    assert array_physical_region(root) == (root.logical_id, (0, 1, 2, 3))
    assert array_physical_region(full) == (root.logical_id, (0, 1, 2, 3))
    assert arrays_share_physical_region(root, full)


def test_partial_and_strided_views_are_distinct() -> None:
    """Different ordered root coverage is not treated as one region."""
    root = _root()
    tail = _view(root, start=1, step=1, size=3)
    evens = _view(root, start=0, step=2, size=2)

    assert not arrays_share_physical_region(root, tail)
    assert not arrays_share_physical_region(root, evens)
    assert not arrays_share_physical_region(tail, evens)


def test_same_named_distinct_roots_do_not_alias() -> None:
    """Display labels cannot make distinct root allocations compare equal."""
    left = _root(name="shared")
    right = _root(name="shared")

    assert not arrays_share_physical_region(left, right)


def test_distinct_empty_roots_have_one_observational_region() -> None:
    """Empty arrays share a region because they contain no physical slot."""
    left = _root(name="left", size=0)
    right = _root(name="right", size=0)

    assert array_static_length(left) == 0
    assert array_physical_region(left) == (left.logical_id, ())
    assert array_physical_region(right) == (right.logical_id, ())
    assert arrays_share_physical_region(left, right)

    symbolic_start = Value(type=UIntType(), name="start")
    symbolic_empty = ArrayValue(
        type=QubitType(),
        name="symbolic_empty",
        shape=(_uint(0),),
        slice_of=left,
        slice_start=symbolic_start,
        slice_step=_uint(1),
    )
    assert array_physical_region(symbolic_empty) is None
    assert arrays_share_physical_region(symbolic_empty, right)


def test_array_ssa_versions_keep_physical_identity() -> None:
    """A new SSA version of one array keeps its logical physical region."""
    root = _root()

    assert arrays_share_physical_region(root, root.next_version())


def test_symbolic_view_region_stays_unresolved() -> None:
    """Symbolic slice bounds fail closed instead of guessing coverage."""
    root = _root()
    symbolic = ArrayValue(
        type=QubitType(),
        name="symbolic",
        shape=(_uint(4),),
        slice_of=root,
        slice_start=Value(type=UIntType(), name="start"),
        slice_step=_uint(1),
    )

    assert array_physical_region(symbolic) is None
    assert not arrays_share_physical_region(root, symbolic)
