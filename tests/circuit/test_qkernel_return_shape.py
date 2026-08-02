"""Frontend validation for qkernel return annotations."""

from __future__ import annotations

from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import QubitRebindError


@qmc.qkernel
def _scalar_returns_array() -> qmc.Qubit:
    """Return an array behind an intentionally scalar annotation.

    Returns:
        qmc.Qubit: Declared scalar type intentionally contradicted by the
        traced return value.
    """
    return qmc.qubit_array(2, "q")  # type: ignore[return-value]


@qmc.qkernel
def _array_returns_scalar() -> qmc.Vector[qmc.Qubit]:
    """Return a scalar behind an intentionally array annotation.

    Returns:
        qmc.Vector[qmc.Qubit]: Declared array type intentionally contradicted
        by the traced return value.
    """
    return qmc.qubit("q")  # type: ignore[return-value]


@qmc.qkernel
def _bit_returns_qubit() -> qmc.Bit:
    """Return a Qubit behind an intentionally Bit annotation.

    Returns:
        qmc.Bit: Declared classical type intentionally contradicted by the
        traced quantum value.
    """
    return qmc.qubit("q")  # type: ignore[return-value]


@qmc.qkernel
def _bit_vector_returns_qubit_vector() -> qmc.Vector[qmc.Bit]:
    """Return quantum elements behind a classical vector annotation.

    Returns:
        qmc.Vector[qmc.Bit]: Declared classical element type intentionally
        contradicted by the traced quantum array.
    """
    return qmc.qubit_array(2, "q")  # type: ignore[return-value]


@qmc.qkernel
def _matrix_returns_vector() -> qmc.Matrix[qmc.Bit]:
    """Return a vector behind an intentionally matrix annotation.

    Returns:
        qmc.Matrix[qmc.Bit]: Declared rank intentionally contradicted by the
        traced vector.
    """
    return qmc.bit_array(2, "bits")  # type: ignore[return-value]


@qmc.qkernel
def _tuple_returns_swapped_types() -> tuple[qmc.Bit, qmc.Qubit]:
    """Return tuple elements behind intentionally reversed annotations.

    Returns:
        tuple[qmc.Bit, qmc.Qubit]: Declared tuple types intentionally
        contradicted by the traced values.
    """
    return qmc.qubit("q"), qmc.bit(False)  # type: ignore[return-value]


@qmc.qkernel
def _returns_nested_python_tuple() -> tuple[tuple[qmc.UInt, qmc.UInt], qmc.Float]:
    """Return an unsupported nested Python tuple.

    Returns:
        tuple[tuple[qmc.UInt, qmc.UInt], qmc.Float]: Nested result that cannot
        be represented by the flat Python-tuple output ABI.
    """
    return (qmc.uint(1), qmc.uint(2)), qmc.float_(3.0)


@qmc.qkernel
def _qamomile_tuple_returns_swapped_types(
    value: qmc.Tuple[qmc.UInt, qmc.Bit],
) -> qmc.Tuple[qmc.Bit, qmc.UInt]:
    """Return a structural tuple behind reversed element annotations.

    Args:
        value (qmc.Tuple[qmc.UInt, qmc.Bit]): Structural tuple whose element
            types intentionally contradict the return annotation.

    Returns:
        qmc.Tuple[qmc.Bit, qmc.UInt]: Intentionally incorrect declaration.
    """
    return value  # type: ignore[return-value]


@qmc.qkernel
def _dict_returns_swapped_types(
    value: qmc.Dict[qmc.UInt, qmc.Bit],
) -> qmc.Dict[qmc.Bit, qmc.UInt]:
    """Return a Dict behind reversed key and value annotations.

    Args:
        value (qmc.Dict[qmc.UInt, qmc.Bit]): Dict whose types intentionally
            contradict the return annotation.

    Returns:
        qmc.Dict[qmc.Bit, qmc.UInt]: Intentionally incorrect declaration.
    """
    return value  # type: ignore[return-value]


@qmc.qkernel
def _bare_tuple_return_annotation(
    value: qmc.Tuple[qmc.UInt, qmc.Bit],
) -> qmc.Tuple:
    """Return a typed Tuple behind a bare structural annotation.

    Args:
        value (qmc.Tuple[qmc.UInt, qmc.Bit]): Typed structural tuple.

    Returns:
        qmc.Tuple: Intentionally incomplete return annotation.
    """
    return value


@qmc.qkernel
def _bare_dict_return_annotation(
    value: qmc.Dict[qmc.UInt, qmc.Bit],
) -> qmc.Dict:
    """Return a typed Dict behind a bare structural annotation.

    Args:
        value (qmc.Dict[qmc.UInt, qmc.Bit]): Typed structural dictionary.

    Returns:
        qmc.Dict: Intentionally incomplete return annotation.
    """
    return value


@qmc.qkernel
def _valid_bit() -> qmc.Bit:
    """Measure one qubit into a correctly annotated Bit.

    Returns:
        qmc.Bit: Classical measurement result.
    """
    return qmc.measure(qmc.qubit("q"))


@qmc.qkernel
def _valid_bit_vector() -> qmc.Vector[qmc.Bit]:
    """Measure a qubit register into a correctly annotated Bit vector.

    Returns:
        qmc.Vector[qmc.Bit]: Classical measurement results.
    """
    return qmc.measure(qmc.qubit_array(2, "q"))


@qmc.qkernel
def _valid_python_bool() -> bool:
    """Return a Bit through its supported Python bool annotation.

    Returns:
        bool: Frontend Bit represented by the Python-compatible annotation.
    """
    return qmc.bit(False)  # type: ignore[return-value]


@qmc.qkernel
def _valid_forward_reference() -> "qmc.Bit":
    """Return a Bit through an explicitly deferred annotation.

    Returns:
        qmc.Bit: Classical constant with a string-form annotation.
    """
    return qmc.bit(False)


_STABLE_RETURN_ALIAS = qmc.Bit
_RETURN_ANNOTATION_RESOLUTION_CALLS = 0


def _next_return_annotation() -> Any:
    """Return a different handle type after the first annotation evaluation."""
    global _RETURN_ANNOTATION_RESOLUTION_CALLS

    _RETURN_ANNOTATION_RESOLUTION_CALLS += 1
    if _RETURN_ANNOTATION_RESOLUTION_CALLS == 1:
        return qmc.Bit
    return qmc.UInt


@qmc.qkernel
def _single_resolution_return_annotation() -> _next_return_annotation():
    """Return a Bit through a side-effectful annotation expression.

    Returns:
        _next_return_annotation(): Bit selected by the first evaluation.
    """
    return qmc.bit(False)


@qmc.qkernel
def _stable_return_alias() -> _STABLE_RETURN_ALIAS:
    """Return a Bit through an alias resolved during decoration.

    Returns:
        _STABLE_RETURN_ALIAS: Classical constant with a frozen return alias.
    """
    return qmc.bit(False)


@qmc.qkernel
def _late_build_return_alias() -> _LATE_BUILD_RETURN_ALIAS:
    """Return a Bit through an alias defined after decoration.

    Returns:
        _LATE_BUILD_RETURN_ALIAS: Classical constant resolved on first use.
    """
    return qmc.bit(False)


_LATE_BUILD_RETURN_ALIAS = qmc.Bit


@qmc.qkernel
def _call_late_build_return_alias() -> qmc.Bit:
    """Invoke the kernel whose late return alias resolves during build.

    Returns:
        qmc.Bit: Result reconstructed from the frozen callee contract.
    """
    return _late_build_return_alias()


@qmc.qkernel
def _late_block_return_alias() -> _LATE_BLOCK_RETURN_ALIAS:
    """Return a Bit through a block-first late alias.

    Returns:
        _LATE_BLOCK_RETURN_ALIAS: Classical constant resolved on first use.
    """
    return qmc.bit(False)


_LATE_BLOCK_RETURN_ALIAS = qmc.Bit


@qmc.qkernel
def _late_tuple_return_alias() -> _LATE_TUPLE_RETURN_ALIAS:
    """Return two values through a late Python-tuple alias.

    Returns:
        _LATE_TUPLE_RETURN_ALIAS: Bit and UInt output slots resolved together.
    """
    return qmc.bit(False), qmc.uint(1)


_LATE_TUPLE_RETURN_ALIAS = tuple[qmc.Bit, qmc.UInt]


@qmc.qkernel
def _late_parameterized_return_alias(
    value: qmc.UInt,
) -> _LATE_PARAMETERIZED_RETURN_ALIAS:
    """Return an annotated input through a late return alias.

    Args:
        value (qmc.UInt): Input whose annotation must resolve independently.

    Returns:
        _LATE_PARAMETERIZED_RETURN_ALIAS: The unchanged UInt input.
    """
    return value


_LATE_PARAMETERIZED_RETURN_ALIAS = qmc.UInt


@qmc.qkernel
def _call_late_parameterized_return_alias(value: qmc.UInt) -> qmc.UInt:
    """Invoke a late-return kernel that also has an annotated input.

    Args:
        value (qmc.UInt): Value forwarded to the nested kernel.

    Returns:
        qmc.UInt: Result reconstructed from the nested call.
    """
    return _late_parameterized_return_alias(value)


_STABLE_INPUT_ALIAS = qmc.UInt


@qmc.qkernel
def _stable_input_alias(value: _STABLE_INPUT_ALIAS) -> qmc.UInt:
    """Return an input whose alias is frozen during decoration.

    Args:
        value (_STABLE_INPUT_ALIAS): UInt input declared through an alias.

    Returns:
        qmc.UInt: The unchanged UInt input.
    """
    return value


@qmc.qkernel
def _late_input_alias(value: _LATE_INPUT_ALIAS) -> qmc.UInt:
    """Return an input whose alias is defined after decoration.

    Args:
        value (_LATE_INPUT_ALIAS): UInt input resolved on first use.

    Returns:
        qmc.UInt: The unchanged UInt input.
    """
    return value


_LATE_INPUT_ALIAS = qmc.UInt


@qmc.qkernel
def _shared_late_interface_alias(
    value: _SHARED_LATE_INTERFACE_ALIAS,
) -> _SHARED_LATE_INTERFACE_ALIAS:
    """Use one late alias for both sides of the qkernel interface.

    Args:
        value (_SHARED_LATE_INTERFACE_ALIAS): Lazily resolved UInt input.

    Returns:
        _SHARED_LATE_INTERFACE_ALIAS: The unchanged UInt input.
    """
    return value


_SHARED_LATE_INTERFACE_ALIAS = qmc.UInt


@qmc.qkernel
def _late_quantum_rebind_alias(
    value: _LATE_QUANTUM_REBIND_ALIAS,
) -> qmc.Bit:
    """Rebind an input whose late alias resolves to Qubit.

    Args:
        value (_LATE_QUANTUM_REBIND_ALIAS): Lazily resolved Qubit input.

    Returns:
        qmc.Bit: Classical value that illegally replaces the quantum input.
    """
    value = qmc.bit(False)  # type: ignore[assignment]
    return value  # type: ignore[return-value]


_LATE_QUANTUM_REBIND_ALIAS = qmc.Qubit


@qmc.qkernel
def _late_quantum_rebind_setter_before_access(
    value: _LATE_QUANTUM_REBIND_SETTER_ALIAS,
) -> qmc.Bit:
    """Rebind a deferred Qubit input used to probe ABI setter validation.

    Args:
        value (_LATE_QUANTUM_REBIND_SETTER_ALIAS): Lazily resolved Qubit input.

    Returns:
        qmc.Bit: Classical value that illegally replaces the quantum input.
    """
    value = qmc.bit(False)  # type: ignore[assignment]
    return value  # type: ignore[return-value]


_LATE_QUANTUM_REBIND_SETTER_ALIAS = qmc.Qubit


@qmc.qkernel
def _call_late_input_alias(value: qmc.UInt) -> qmc.UInt:
    """Invoke a kernel whose input alias resolves lazily.

    Args:
        value (qmc.UInt): Value forwarded to the nested kernel.

    Returns:
        qmc.UInt: Result returned by the nested kernel.
    """
    return _late_input_alias(value)


@qmc.qkernel
def _valid_nested_unbound_dict(
    value: qmc.Tuple[qmc.Dict[qmc.UInt, qmc.Bit], qmc.UInt],
) -> qmc.Tuple[qmc.Dict[qmc.UInt, qmc.Bit], qmc.UInt]:
    """Return a structural tuple containing an unbound Dict.

    Args:
        value (qmc.Tuple[qmc.Dict[qmc.UInt, qmc.Bit], qmc.UInt]): Tuple whose
            Dict IR value intentionally carries wildcard key/value types.

    Returns:
        qmc.Tuple[qmc.Dict[qmc.UInt, qmc.Bit], qmc.UInt]: Unchanged input.
    """
    return value


@qmc.qkernel
def _valid_qfixed() -> qmc.QFixed:
    """Return a fixed-point quantum register.

    Returns:
        qmc.QFixed: Fixed-point alias of a two-qubit register.
    """
    return qmc.cast(qmc.qubit_array(2, "q"), qmc.QFixed, int_bits=1)


@qmc.qkernel
def _valid_void() -> None:
    """Trace a kernel with no explicit return value."""
    q = qmc.qubit("q")
    q = qmc.h(q)


@qmc.qkernel
def _native_scalar_results() -> tuple[bool, int, float]:
    """Return frontend handles through Python-native scalar annotations.

    Returns:
        tuple[bool, int, float]: Bit, UInt, and Float frontend values.
    """
    return (  # type: ignore[return-value]
        qmc.bit(False),
        qmc.uint(1),
        qmc.float_(2.0),
    )


@qmc.qkernel
def _call_native_scalar_results() -> tuple[qmc.Bit, qmc.UInt, qmc.Float]:
    """Invoke a kernel whose results use native scalar aliases.

    Returns:
        tuple[qmc.Bit, qmc.UInt, qmc.Float]: Canonical frontend handles
        reconstructed by the invocation path.
    """
    bit, count, value = _native_scalar_results()
    return bit, count, value  # type: ignore[return-value]


@qmc.qkernel
def _singleton_tuple_result() -> tuple[qmc.Bit]:
    """Return one Bit in a Python tuple.

    Returns:
        tuple[qmc.Bit]: One-element Python result tuple.
    """
    return (qmc.bit(False),)


@qmc.qkernel
def _empty_tuple_result() -> tuple[()]:
    """Return an empty Python tuple.

    Returns:
        tuple[()]: Empty Python result tuple.
    """
    return ()


@qmc.qkernel
def _forward_singleton_tuple_result() -> tuple[qmc.Bit]:
    """Forward a one-element Python tuple from a nested qkernel.

    Returns:
        tuple[qmc.Bit]: Nested one-element Python result tuple.
    """
    return _singleton_tuple_result()


@qmc.qkernel
def _unpack_singleton_tuple_result() -> qmc.Bit:
    """Unpack a one-element Python tuple returned by a nested qkernel.

    Returns:
        qmc.Bit: The tuple's only element.
    """
    (value,) = _singleton_tuple_result()
    return value


MISMATCH_CASES = [
    pytest.param(
        _scalar_returns_array,
        r"declares a scalar.*returned Vector",
        id="scalar-returns-array",
    ),
    pytest.param(
        _array_returns_scalar,
        r"declares an array.*returned Qubit",
        id="array-returns-scalar",
    ),
    pytest.param(
        _bit_returns_qubit,
        r"declares Bit.*BitType.*returned Qubit.*QubitType",
        id="bit-returns-qubit",
    ),
    pytest.param(
        _bit_vector_returns_qubit_vector,
        r"declares Vector\[Bit\].*BitType.*returned Vector.*QubitType",
        id="bit-vector-returns-qubit-vector",
    ),
    pytest.param(
        _matrix_returns_vector,
        r"declares Matrix\[Bit\].*returned Vector.*different array rank",
        id="matrix-returns-vector",
    ),
    pytest.param(
        _tuple_returns_swapped_types,
        r"return\[0\].*declares Bit.*returned Qubit",
        id="tuple-returns-swapped-types",
    ),
    pytest.param(
        _returns_nested_python_tuple,
        r"return\[0\].*nested Python tuple return.*not supported",
        id="nested-python-tuple",
    ),
    pytest.param(
        _qamomile_tuple_returns_swapped_types,
        r"return\[0\].*declares Bit.*returned UInt",
        id="qamomile-tuple-returns-swapped-types",
    ),
    pytest.param(
        _dict_returns_swapped_types,
        r"return\.key.*declares Bit.*carries UInt",
        id="dict-returns-swapped-types",
    ),
    pytest.param(
        _bare_tuple_return_annotation,
        r"Tuple annotation must declare element types",
        id="bare-tuple-annotation",
    ),
    pytest.param(
        _bare_dict_return_annotation,
        r"Dict annotation must declare key and value types",
        id="bare-dict-annotation",
    ),
]


@pytest.mark.parametrize("trace_mode", ["build", "block"])
@pytest.mark.parametrize(("kernel", "message"), MISMATCH_CASES)
def test_return_annotation_mismatch_is_rejected(
    kernel: Any,
    message: str,
    trace_mode: str,
) -> None:
    """Both tracing entry points reject every return contract mismatch."""
    with pytest.raises(TypeError, match=message):
        if trace_mode == "build":
            kernel.build()
        else:
            _ = kernel.block


@pytest.mark.parametrize("trace_mode", ["build", "block"])
@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(_valid_bit, id="bit"),
        pytest.param(_valid_bit_vector, id="bit-vector"),
        pytest.param(_valid_python_bool, id="python-bool"),
        pytest.param(_valid_forward_reference, id="forward-reference"),
        pytest.param(_valid_nested_unbound_dict, id="nested-unbound-dict"),
        pytest.param(_valid_qfixed, id="qfixed"),
    ],
)
def test_matching_return_annotation_is_accepted(kernel: Any, trace_mode: str) -> None:
    """Both tracing entry points preserve supported matching return types."""
    if trace_mode == "build":
        block = kernel.build()
    else:
        block = kernel.block

    assert len(block.output_values) == 1


def test_void_build_remains_supported() -> None:
    """The top-level build path preserves its existing void-kernel contract."""
    assert _valid_void.build().output_values == []


def test_nested_call_wraps_native_scalar_return_aliases() -> None:
    """Nested calls canonicalize Python scalar aliases to frontend handles."""
    block = _call_native_scalar_results.build()
    assert [value.type.label() for value in block.output_values] == [
        "BitType",
        "UIntType",
        "FloatType",
    ]


def test_nested_call_preserves_singleton_python_tuple_shape() -> None:
    """Nested invocation retains a Python tuple even with one output slot."""
    forwarded = _forward_singleton_tuple_result.build()
    unpacked = _unpack_singleton_tuple_result.build()

    assert forwarded.output_values[0].type.label() == "BitType"
    assert unpacked.output_values[0].type.label() == "BitType"


@pytest.mark.parametrize("trace_mode", ["build", "block"])
def test_empty_python_tuple_return_remains_supported(trace_mode: str) -> None:
    """Both tracing modes preserve the existing empty-tuple return ABI."""
    if trace_mode == "build":
        block = _empty_tuple_result.build()
    else:
        block = _empty_tuple_result.block

    assert block.output_values == []


def test_resolved_return_alias_is_frozen_across_entrypoints() -> None:
    """A global rebind cannot change a contract resolved at decoration."""
    global _STABLE_RETURN_ALIAS

    original_alias = _STABLE_RETURN_ALIAS
    original_annotation = _stable_return_alias.func.__annotations__["return"]
    _STABLE_RETURN_ALIAS = qmc.UInt
    _stable_return_alias.func.__annotations__["return"] = qmc.UInt
    try:
        traced = _stable_return_alias.build()
        hierarchical = _stable_return_alias.block
    finally:
        _STABLE_RETURN_ALIAS = original_alias
        _stable_return_alias.func.__annotations__["return"] = original_annotation

    assert traced.output_values[0].type.label() == "BitType"
    assert hierarchical.output_values[0].type.label() == "BitType"
    assert _stable_return_alias.return_type is qmc.Bit
    assert _stable_return_alias.output_types == [qmc.Bit]

    copied_output_types = _stable_return_alias.output_types
    copied_output_types[0] = qmc.UInt
    assert _stable_return_alias.output_types == [qmc.Bit]


def test_return_annotation_is_evaluated_once_during_decoration() -> None:
    """Decoration freezes the first successful annotation evaluation."""
    assert _RETURN_ANNOTATION_RESOLUTION_CALLS == 1

    traced = _single_resolution_return_annotation.build()
    hierarchical = _single_resolution_return_annotation.block

    assert traced.output_values[0].type.label() == "BitType"
    assert hierarchical.output_values[0].type.label() == "BitType"
    assert _single_resolution_return_annotation.return_type is qmc.Bit
    assert _single_resolution_return_annotation.output_types == [qmc.Bit]
    assert _RETURN_ANNOTATION_RESOLUTION_CALLS == 1


def test_resolved_inputs_can_be_read_while_return_alias_is_missing() -> None:
    """Input introspection does not require an unrelated late return alias."""
    late_alias = globals().pop("_LATE_BUILD_RETURN_ALIAS")
    try:
        assert _late_build_return_alias.input_types == {}
    finally:
        globals()["_LATE_BUILD_RETURN_ALIAS"] = late_alias


def test_late_return_alias_is_frozen_after_build_first_resolution() -> None:
    """Build resolves a late annotation once for later block construction."""
    global _LATE_BUILD_RETURN_ALIAS

    traced = _late_build_return_alias.build()
    original_alias = _LATE_BUILD_RETURN_ALIAS
    _LATE_BUILD_RETURN_ALIAS = qmc.UInt
    try:
        hierarchical = _late_build_return_alias.block
        caller = _call_late_build_return_alias.build()
    finally:
        _LATE_BUILD_RETURN_ALIAS = original_alias

    assert traced.output_values[0].type.label() == "BitType"
    assert hierarchical.output_values[0].type.label() == "BitType"
    assert caller.output_values[0].type.label() == "BitType"
    assert _late_build_return_alias.return_type is qmc.Bit
    assert _late_build_return_alias.output_types == [qmc.Bit]


def test_late_return_alias_is_frozen_after_block_first_resolution() -> None:
    """Block construction resolves a late annotation for later builds."""
    global _LATE_BLOCK_RETURN_ALIAS

    hierarchical = _late_block_return_alias.block
    original_alias = _LATE_BLOCK_RETURN_ALIAS
    _LATE_BLOCK_RETURN_ALIAS = qmc.UInt
    try:
        traced = _late_block_return_alias.build()
    finally:
        _LATE_BLOCK_RETURN_ALIAS = original_alias

    assert hierarchical.output_values[0].type.label() == "BitType"
    assert traced.output_values[0].type.label() == "BitType"
    assert _late_block_return_alias.return_type is qmc.Bit
    assert _late_block_return_alias.output_types == [qmc.Bit]


def test_late_tuple_alias_updates_complete_and_flattened_return_types() -> None:
    """Late tuple resolution freezes both the full and flattened contracts."""
    block = _late_tuple_return_alias.build()

    assert _late_tuple_return_alias.return_type == tuple[qmc.Bit, qmc.UInt]
    assert _late_tuple_return_alias.output_types == [qmc.Bit, qmc.UInt]
    assert [value.type.label() for value in block.output_values] == [
        "BitType",
        "UIntType",
    ]


def test_late_return_alias_does_not_defer_resolvable_input_types() -> None:
    """One unresolved return hint cannot poison independent input hints."""
    assert _late_parameterized_return_alias.input_types == {"value": qmc.UInt}

    traced = _late_parameterized_return_alias.build(value=1)
    hierarchical = _late_parameterized_return_alias.block
    caller = _call_late_parameterized_return_alias.build(value=1)

    assert traced.output_values[0].type.label() == "UIntType"
    assert hierarchical.output_values[0].type.label() == "UIntType"
    assert caller.output_values[0].type.label() == "UIntType"


def test_resolved_return_can_be_read_while_input_alias_is_missing() -> None:
    """Return introspection does not require an unrelated late input alias."""
    late_alias = globals().pop("_LATE_INPUT_ALIAS")
    try:
        assert _late_input_alias.return_type is qmc.UInt
    finally:
        globals()["_LATE_INPUT_ALIAS"] = late_alias


def test_late_input_alias_is_resolved_once_across_entrypoints() -> None:
    """A deferred input annotation resolves before any ABI consumer runs."""
    assert _late_input_alias.input_types == {"value": qmc.UInt}

    traced = _late_input_alias.build(value=1)
    hierarchical = _late_input_alias.block
    caller = _call_late_input_alias.build(value=1)

    assert traced.input_values[0].type.label() == "UIntType"
    assert hierarchical.input_values[0].type.label() == "UIntType"
    assert traced.param_slots[0].type.label() == "UIntType"
    assert hierarchical.param_slots[0].type.label() == "UIntType"
    assert caller.output_values[0].type.label() == "UIntType"


def test_shared_late_alias_freezes_the_complete_interface_atomically() -> None:
    """First interface access resolves shared input and return aliases together."""
    global _SHARED_LATE_INTERFACE_ALIAS

    assert _shared_late_interface_alias.return_type is qmc.UInt
    original_alias = _SHARED_LATE_INTERFACE_ALIAS
    _SHARED_LATE_INTERFACE_ALIAS = qmc.Float
    try:
        assert _shared_late_interface_alias.input_types == {"value": qmc.UInt}
        block = _shared_late_interface_alias.build(value=1)
    finally:
        _SHARED_LATE_INTERFACE_ALIAS = original_alias

    assert block.input_values[0].type.label() == "UIntType"
    assert block.output_values[0].type.label() == "UIntType"


def test_late_quantum_rebind_error_cannot_be_bypassed_by_alias_rebinding() -> None:
    """A failed semantic check retains the first resolved quantum input type."""
    global _LATE_QUANTUM_REBIND_ALIAS

    with pytest.raises(QubitRebindError):
        _ = _late_quantum_rebind_alias.input_types

    with pytest.raises(QubitRebindError):
        _late_quantum_rebind_alias.input_types = {"value": qmc.UInt}

    original_alias = _LATE_QUANTUM_REBIND_ALIAS
    _LATE_QUANTUM_REBIND_ALIAS = qmc.UInt
    try:
        with pytest.raises(QubitRebindError):
            _late_quantum_rebind_alias.build(value=1)
    finally:
        _LATE_QUANTUM_REBIND_ALIAS = original_alias


def test_input_type_setter_resolves_source_annotations_before_replacement() -> None:
    """ABI replacement cannot hide a deferred quantum rebind violation."""
    with pytest.raises(QubitRebindError):
        _late_quantum_rebind_setter_before_access.input_types = {"value": qmc.UInt}

    with pytest.raises(QubitRebindError):
        _late_quantum_rebind_setter_before_access.build(value=1)


def test_resolved_input_alias_is_frozen_across_entrypoints() -> None:
    """A global rebind cannot split build and block input contracts."""
    global _STABLE_INPUT_ALIAS

    original_alias = _STABLE_INPUT_ALIAS
    original_annotation = _stable_input_alias.func.__annotations__["value"]
    _STABLE_INPUT_ALIAS = qmc.Float
    _stable_input_alias.func.__annotations__["value"] = qmc.Float
    try:
        traced = _stable_input_alias.build(value=1)
        hierarchical = _stable_input_alias.block
    finally:
        _STABLE_INPUT_ALIAS = original_alias
        _stable_input_alias.func.__annotations__["value"] = original_annotation

    assert traced.input_values[0].type.label() == "UIntType"
    assert hierarchical.input_values[0].type.label() == "UIntType"
    assert traced.param_slots[0].type.label() == "UIntType"
    assert hierarchical.param_slots[0].type.label() == "UIntType"
    assert traced.output_values[0].type.label() == "UIntType"
    assert hierarchical.output_values[0].type.label() == "UIntType"

    copied_input_types = _stable_input_alias.input_types
    copied_input_types["value"] = qmc.Float
    assert _stable_input_alias.input_types == {"value": qmc.UInt}
