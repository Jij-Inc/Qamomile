"""Pure tests for persistent classical provenance state projection."""

import sympy as sp

from qamomile.circuit.estimator import _array_state, _classical_facts, _resolver
from qamomile.circuit.ir.types.primitives import BitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value


def _uint(value: int) -> Value:
    """Return one constant unsigned-integer IR value.

    Args:
        value (int): Constant value to carry.

    Returns:
        Value: Constant UInt IR value.
    """
    return Value(type=UIntType(), name="").with_const(value)


def _bit_array(name: str, contents: tuple[bool, ...]) -> ArrayValue:
    """Return one constant Bit-array IR value.

    Args:
        name (str): Debug label for the array.
        contents (tuple[bool, ...]): Initial array contents.

    Returns:
        ArrayValue: Bit array carrying immutable runtime contents.
    """
    return ArrayValue(
        type=BitType(),
        name=name,
        shape=(_uint(len(contents)),),
    ).with_array_runtime_metadata(const_array=contents)


def _element(array: ArrayValue, index: int) -> Value:
    """Return one scalar element read from an array.

    Args:
        array (ArrayValue): Parent array.
        index (int): Constant element index.

    Returns:
        Value: Scalar Bit value carrying array-element provenance.
    """
    return Value(
        type=BitType(),
        name="",
        parent_array=array,
        element_indices=(_uint(index),),
    )


def _fact(
    value: sp.Basic | int | bool,
    token: str | None = None,
) -> _classical_facts._ResolvedClassicalFact:
    """Return a resolved fact with an optional unconditional dependency.

    Args:
        value (sp.Basic | int | bool): Resolved fact value.
        token (str | None): Optional source token. Defaults to ``None``.

    Returns:
        _ResolvedClassicalFact: Normalized test fact.
    """
    dependencies = {} if token is None else {token: sp.true}
    return _classical_facts._ResolvedClassicalFact.create(value, dependencies)


def test_strong_store_overwrite_kills_previous_dependencies() -> None:
    """A definite store replaces old element dependencies instead of unioning."""
    array = _bit_array("bits", (False,))
    resolver = _resolver.ExprResolver()
    resolver.bind_array_state(
        array,
        _array_state._ArrayStoreState(
            _array_state._ArrayStoreState(
                _array_state._ArrayConstantState((False,)),
                _fact(True, "old"),
                (_fact(0),),
            ),
            _fact(False, "new"),
            (_fact(0),),
        ),
    )

    result = resolver.resolve_classical_fact(_element(array, 0))

    assert bool(result.value) is False
    assert result.dependencies == {"new": sp.true}


def test_precise_element_projection_replaces_whole_array_fallback() -> None:
    """A proven sibling element does not inherit a broad array fallback."""
    array = _bit_array("bits", (False, False))
    resolver = _resolver.ExprResolver()
    resolver.bind_classical_fact(array, _fact(sp.Symbol("bits"), "broad"))
    resolver.bind_array_state(
        array,
        _array_state._ArrayStoreState(
            _array_state._ArrayConstantState((False, False)),
            _fact(True, "updated"),
            (_fact(0),),
        ),
    )

    result = resolver.resolve_classical_fact(_element(array, 1))

    assert bool(result.value) is False
    assert result.dependencies == {}


def test_symbolic_store_alias_guards_old_and_new_dependencies() -> None:
    """A symbolic store index guards both overwrite alternatives precisely."""
    array = _bit_array("bits", (False,))
    store_index = sp.Symbol("store", integer=True, nonnegative=True)
    resolver = _resolver.ExprResolver()
    resolver.bind_array_state(
        array,
        _array_state._ArrayStoreState(
            _array_state._ArrayStoreState(
                _array_state._ArrayConstantState((False,)),
                _fact(True, "old"),
                (_fact(0),),
            ),
            _fact(False, "new"),
            (
                _classical_facts._ResolvedClassicalFact.create(
                    store_index,
                    {"index_source": sp.true},
                ),
            ),
        ),
    )

    result = resolver.resolve_classical_fact(_element(array, 0))
    aliases = sp.Eq(0, store_index)

    assert result.value == sp.Piecewise((sp.false, aliases), (sp.true, True))
    assert result.dependencies == {
        "index_source": sp.true,
        "new": aliases,
        "old": sp.Not(aliases),
    }


def test_choice_drops_selector_only_for_structurally_identical_facts() -> None:
    """Only an observationally identical branch pair ignores its selector."""
    selector = _classical_facts._ResolvedClassicalFact.create(
        sp.Symbol("measured", integer=True, nonnegative=True),
        {"measurement": sp.true},
    )
    identical = _fact(True, "shared")

    collapsed = _classical_facts._choice_classical_fact(identical, identical, selector)
    distinct = _classical_facts._choice_classical_fact(
        identical,
        _fact(True, "other"),
        selector,
    )

    assert collapsed == identical
    assert distinct.dependencies["measurement"] is sp.true
    assert distinct.dependencies["shared"] == selector.value
    assert distinct.dependencies["other"] == sp.Not(selector.value)


def test_bind_classical_selection_preserves_guards_with_value_override() -> None:
    """A separately derived merge value keeps selected branch dependencies."""
    result = Value(type=BitType(), name="merged")
    selector = _classical_facts._ResolvedClassicalFact.create(
        sp.Symbol("condition", integer=True, nonnegative=True),
        {"selector": sp.true},
    )
    resolver = _resolver.ExprResolver()

    resolver.bind_classical_selection(
        result,
        _fact(True, "true_source"),
        _fact(False, "false_source"),
        selector,
        value_override=sp.Symbol("merged_value", integer=True, nonnegative=True),
    )
    selected = resolver.resolve_classical_fact(result)

    assert selected.value == sp.Symbol(
        "merged_value",
        integer=True,
        nonnegative=True,
    )
    assert selected.dependencies == {
        "false_source": sp.Not(selector.value),
        "selector": sp.true,
        "true_source": selector.value,
    }


def test_bind_classical_selection_drops_irrelevant_selector() -> None:
    """Identical branch facts remain independent of their selector token."""
    result = Value(type=BitType(), name="merged")
    selector = _classical_facts._ResolvedClassicalFact.create(
        sp.Symbol("condition", integer=True, nonnegative=True),
        {"selector": sp.true},
    )
    identical = _fact(True, "shared")
    resolver = _resolver.ExprResolver()

    resolver.bind_classical_selection(
        result,
        identical,
        identical,
        selector,
    )

    assert resolver.resolve_classical_fact(result) == identical


def test_bind_array_state_selection_projects_each_element_independently() -> None:
    """Detached array branches retain selectors only for differing elements."""
    result = _bit_array("merged", (False, False))
    selector = _classical_facts._ResolvedClassicalFact.create(
        sp.Symbol("condition", integer=True, nonnegative=True),
        {"selector": sp.true},
    )
    false_state = _array_state._ArrayConstantState((False, False))
    true_state = _array_state._ArrayStoreState(
        false_state,
        _fact(True, "updated"),
        (_fact(0),),
    )
    resolver = _resolver.ExprResolver()

    resolver.bind_array_state_selection(
        result,
        true_state,
        false_state,
        selector,
    )
    changed = resolver.resolve_classical_fact(_element(result, 0))
    unchanged = resolver.resolve_classical_fact(_element(result, 1))

    assert changed.dependencies == {
        "selector": sp.true,
        "updated": selector.value,
    }
    assert unchanged.dependencies == {}


def test_array_context_fork_export_and_import_are_isolated() -> None:
    """Persistent context APIs share state nodes without sharing UUID maps."""
    left = _bit_array("left", (False,))
    right = _bit_array("right", (True,))
    parent = _resolver.ExprResolver()
    parent.bind_array_state(left, _array_state._ArrayConstantState((False,)))
    parent.bind_array_state(right, _array_state._ArrayConstantState((True,)))

    forked = parent.fork_array_context()
    forked.pop(left.uuid)
    selected = parent.export_array_context((left,))
    child = _resolver.ExprResolver()
    child.import_array_context(selected)

    assert left.uuid in parent.export_array_context()
    assert left.uuid not in forked
    assert set(selected) == {left.uuid}
    assert bool(child.resolve_classical_fact(_element(left, 0)).value) is False
    assert right.uuid not in child.export_array_context()
