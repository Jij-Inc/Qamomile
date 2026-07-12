"""End-to-end regressions for scalar branch values feeding RegionArgs."""

from __future__ import annotations

from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle.primitives import Bit, UInt
from qamomile.circuit.frontend.operation.control_flow import emit_if
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.operation.control_flow import ForItemsOperation
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import DependencyError


@qmc.qkernel
def _plain_uint_runtime_branch(prepare: qmc.UInt) -> qmc.UInt:
    """Select plain UInt constants with a measured runtime predicate.

    Args:
        prepare (qmc.UInt): One prepares a measured one; zero prepares zero.

    Returns:
        qmc.UInt: One on the true branch and two on the false branch.
    """
    predicate = qmc.qubit("predicate")
    if prepare == 1:
        predicate = qmc.x(predicate)
    measured = qmc.measure(predicate)
    value = 0
    for _i in qmc.range(2):
        if measured:
            value = 1
        else:
            value = 2
    return value


@qmc.qkernel
def _plain_float_runtime_branch(prepare: qmc.UInt) -> qmc.Float:
    """Select plain Float constants with a measured runtime predicate.

    Args:
        prepare (qmc.UInt): One prepares a measured one; zero prepares zero.

    Returns:
        qmc.Float: 1.25 on the true branch and 2.5 on the false branch.
    """
    predicate = qmc.qubit("predicate")
    if prepare == 1:
        predicate = qmc.x(predicate)
    measured = qmc.measure(predicate)
    value = 0.0
    for _i in qmc.range(2):
        if measured:
            value = 1.25
        else:
            value = 2.5
    return value


@qmc.qkernel
def _plain_bit_runtime_branch(prepare: qmc.UInt) -> qmc.Bit:
    """Select plain bool constants with a measured runtime predicate.

    Args:
        prepare (qmc.UInt): One prepares a measured one; zero prepares zero.

    Returns:
        qmc.Bit: The measured predicate copied through plain bool branches.
    """
    predicate = qmc.qubit("predicate")
    if prepare == 1:
        predicate = qmc.x(predicate)
    measured = qmc.measure(predicate)
    value = False
    for _i in qmc.range(2):
        if measured:
            value = True
        else:
            value = False
    return value


@qmc.qkernel
def _plain_and_handle_runtime_branch(prepare: qmc.UInt) -> qmc.UInt:
    """Merge one plain UInt and one explicit UInt handle.

    Args:
        prepare (qmc.UInt): One prepares a measured one; zero prepares zero.

    Returns:
        qmc.UInt: One from the plain branch or two from the handle branch.
    """
    predicate = qmc.qubit("predicate")
    if prepare == 1:
        predicate = qmc.x(predicate)
    measured = qmc.measure(predicate)
    value = qmc.uint(0)
    for _i in qmc.range(2):
        if measured:
            value = 1
        else:
            value = qmc.uint(2)
    return value


@qmc.qkernel
def _plain_uint_compile_time_branch(flag: qmc.UInt) -> qmc.UInt:
    """Select plain UInt constants with a bound compile-time predicate.

    Args:
        flag (qmc.UInt): One selects the true branch; zero selects false.

    Returns:
        qmc.UInt: One on the true branch and two on the false branch.
    """
    qmc.measure(qmc.qubit("segment_anchor"))
    value = 0
    for _i in qmc.range(2):
        if flag == 1:
            value = 1
        else:
            value = 2
    return value


@qmc.qkernel
def _plain_float_quantum_escape() -> qmc.Bit:
    """Feed a runtime-selected plain Float into later quantum work.

    Returns:
        qmc.Bit: Measurement of the conditionally rotated target.
    """
    measured = qmc.measure(qmc.qubit("predicate"))
    angle = 0.0
    for _i in qmc.range(1):
        if measured:
            angle = 3.141592653589793
        else:
            angle = 0.0
    target = qmc.qubit("target")
    target = qmc.rx(target, angle)
    return qmc.measure(target)


@qmc.qkernel
def _uint_item_live_out(
    items: qmc.Dict[qmc.UInt, qmc.UInt],
) -> qmc.UInt:
    """Return the final UInt value of a nonempty items loop.

    Args:
        items (qmc.Dict[qmc.UInt, qmc.UInt]): Ordered UInt mapping.

    Returns:
        qmc.UInt: The last mapping value.
    """
    qmc.measure(qmc.qubit("segment_anchor"))
    value = qmc.uint(0)
    for _key, item_value in qmc.items(items):
        value = item_value
    return value


@qmc.qkernel
def _float_item_live_out(
    items: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Float:
    """Return the final Float value of a nonempty items loop.

    Args:
        items (qmc.Dict[qmc.UInt, qmc.Float]): Ordered Float mapping.

    Returns:
        qmc.Float: The last mapping value.
    """
    qmc.measure(qmc.qubit("segment_anchor"))
    value = qmc.float_(0.0)
    for _key, item_value in qmc.items(items):
        value = item_value
    return value


@qmc.qkernel
def _vector_key_type(
    items: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.UInt:
    """Read a Vector[UInt] item key through its declared element type.

    Args:
        items (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Vector-key mapping.

    Returns:
        qmc.UInt: First component of the final key.
    """
    qmc.measure(qmc.qubit("segment_anchor"))
    value = qmc.uint(0)
    for key, _coefficient in qmc.items(items):
        value = key[0]
    return value


@qmc.qkernel
def _float_item_key_live_out(
    items: qmc.Dict[qmc.Float, qmc.UInt],
) -> qmc.Float:
    """Return the final Float key of a nonempty items loop.

    Args:
        items (qmc.Dict[qmc.Float, qmc.UInt]): Float-keyed mapping.

    Returns:
        qmc.Float: The last mapping key.
    """
    qmc.measure(qmc.qubit("segment_anchor"))
    key_out = qmc.float_(0.0)
    for key, _item_value in qmc.items(items):
        key_out = key
    return key_out


@qmc.qkernel
def _bit_item_key_controls_target(
    items: qmc.Dict[qmc.Bit, qmc.UInt],
) -> qmc.Bit:
    """Use each Bit key as a quantum branch condition.

    Args:
        items (qmc.Dict[qmc.Bit, qmc.UInt]): Bit-keyed mapping.

    Returns:
        qmc.Bit: Target measurement after applying key-controlled X gates.
    """
    target = qmc.qubit("target")
    for key, _item_value in qmc.items(items):
        if key:
            target = qmc.x(target)
    return qmc.measure(target)


def _sample_single(kernel: Any, bindings: dict[str, Any]) -> Any:
    """Transpile and sample one deterministic qkernel result.

    Args:
        kernel (Any): Qkernel to transpile.
        bindings (dict[str, Any]): Compile-time bindings.

    Returns:
        Any: The sole deterministic output value.

    Raises:
        AssertionError: If sampling produces more than one result value.
    """
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(kernel, bindings=bindings)
    sampled = executable.sample(transpiler.executor(), shots=20).result()
    assert len(sampled.results) == 1
    return sampled.results[0][0]


@pytest.mark.parametrize(
    ("kernel", "expected_false", "expected_true"),
    [
        (_plain_uint_runtime_branch, 2, 1),
        (_plain_float_runtime_branch, 2.5, 1.25),
        (_plain_bit_runtime_branch, False, True),
        (_plain_and_handle_runtime_branch, 2, 1),
    ],
)
def test_plain_scalar_runtime_branch_follows_measurement(
    kernel: Any,
    expected_false: Any,
    expected_true: Any,
) -> None:
    """Plain scalar branch yields are selected at runtime, never trace time."""
    assert _sample_single(kernel, {"prepare": 0}) == expected_false
    assert _sample_single(kernel, {"prepare": 1}) == expected_true


@pytest.mark.parametrize(("flag", "expected"), [(0, 2), (1, 1)])
def test_plain_scalar_compile_time_branch_selects_bound_side(
    flag: int,
    expected: int,
) -> None:
    """Bound compile-time conditions select the matching plain scalar."""
    assert _sample_single(_plain_uint_compile_time_branch, {"flag": flag}) == expected


def test_equal_plain_scalars_pass_through_unpromoted() -> None:
    """Equal plain scalars stay plain Python values across the if.

    Both branches yielding the exact same typed scalar cannot diverge at
    runtime, so no merge is represented and the caller keeps a genuine
    Python value — trace-time interop (list indexing, builtin range)
    must keep working for values that merely live across an if.
    Divergent plain scalars are covered by the promotion tests above.
    """
    condition = Bit(value=Value(type=BitType(), name="condition"))
    tracer = Tracer()

    with trace(tracer):
        merged = emit_if(
            lambda: condition,
            lambda: 7,
            lambda: 7,
            [],
        )

    assert merged == 7
    assert isinstance(merged, int)
    assert not isinstance(merged, UInt)


def test_equal_value_different_type_scalars_still_promote() -> None:
    """`True` vs `1` compare equal but differ in type: not a passthrough.

    The equal-scalar passthrough requires the exact same Python type, so
    this pair still goes through promotion, where the Bit/UInt type
    mismatch is rejected exactly like any divergent-type branch pair.
    """
    condition = Bit(value=Value(type=BitType(), name="condition"))

    with (
        trace(Tracer()),
        pytest.raises(TypeError, match="Type mismatch in if-else branches"),
    ):
        emit_if(
            lambda: condition,
            lambda: True,
            lambda: 1,
            [],
        )


def test_identical_opaque_branch_value_is_safe_passthrough() -> None:
    """One opaque object reused by both branches needs no runtime phi."""
    condition = Bit(value=Value(type=BitType(), name="condition"))
    sentinel = object()

    with trace(Tracer()):
        merged = emit_if(
            lambda: condition,
            lambda: sentinel,
            lambda: sentinel,
            [],
        )

    assert merged is sentinel


def test_divergent_opaque_branch_values_are_rejected() -> None:
    """Different opaque objects cannot form an unrepresented runtime phi."""
    condition = Bit(value=Value(type=BitType(), name="condition"))

    with (
        trace(Tracer()),
        pytest.raises(
            TypeError,
            match="have no Qamomile scalar/handle representation",
        ),
    ):
        emit_if(
            lambda: condition,
            lambda: object(),
            lambda: object(),
            [],
        )


def test_runtime_plain_scalar_cannot_escape_back_into_quantum_work() -> None:
    """A measurement-selected scalar feeding a gate fails at dependency analysis."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    with pytest.raises(DependencyError, match="depends on measurement result"):
        QiskitTranspiler().transpile(_plain_float_quantum_escape, bindings={})


@pytest.mark.parametrize(
    ("kernel", "bindings", "expected", "expected_type"),
    [
        (_uint_item_live_out, {"items": {0: 2, 1: 3}}, 3, UIntType),
        (_float_item_live_out, {"items": {0: 1.25, 1: 2.5}}, 2.5, FloatType),
        (_vector_key_type, {"items": {(3, 4): 1.0}}, 3, UIntType),
    ],
)
def test_for_items_uses_declared_key_and_value_types(
    kernel: Any,
    bindings: dict[str, Any],
    expected: Any,
    expected_type: type[Any],
) -> None:
    """ForItems placeholders and live-out values preserve Dict annotations."""
    block = kernel.build(**bindings)
    loop = next(op for op in block.operations if isinstance(op, ForItemsOperation))
    if kernel is _vector_key_type:
        assert loop.key_is_vector
        assert isinstance(loop.key_var_values[0].type, UIntType)
    else:
        assert isinstance(loop.value_var_value.type, expected_type)
    assert isinstance(block.output_values[0].type, expected_type)
    assert _sample_single(kernel, bindings) == expected


@pytest.mark.parametrize(
    ("kernel", "bindings", "expected", "expected_type"),
    [
        (_float_item_key_live_out, {"items": {1.25: 3}}, 1.25, FloatType),
        (_bit_item_key_controls_target, {"items": {True: 3}}, 1, BitType),
    ],
)
def test_for_items_uses_declared_scalar_key_type(
    kernel: Any,
    bindings: dict[str, Any],
    expected: Any,
    expected_type: type[Any],
) -> None:
    """Scalar item keys retain their Float or Bit Dict annotation."""
    block = kernel.build(**bindings)
    loop = next(op for op in block.operations if isinstance(op, ForItemsOperation))
    assert loop.key_var_values is not None
    assert isinstance(loop.key_var_values[0].type, expected_type)
    assert isinstance(block.output_values[0].type, expected_type)
    assert _sample_single(kernel, bindings) == expected


def test_bit_item_value_placeholder_is_typed_bit() -> None:
    """Dict[UInt, Bit] keeps a Bit item placeholder for branch conditions."""

    @qmc.qkernel
    def kernel(items: qmc.Dict[qmc.UInt, qmc.Bit]) -> qmc.Bit:
        """Apply X for every true item value and measure the target.

        Args:
            items (qmc.Dict[qmc.UInt, qmc.Bit]): Boolean item mapping.

        Returns:
            qmc.Bit: Target measurement.
        """
        target = qmc.qubit("target")
        for _key, enabled in qmc.items(items):
            if enabled:
                target = qmc.x(target)
        return qmc.measure(target)

    block = kernel.build(items={0: True})
    loop = next(op for op in block.operations if isinstance(op, ForItemsOperation))
    assert isinstance(loop.value_var_value.type, BitType)
    assert _sample_single(kernel, {"items": {0: True}}) == 1


_APPLY_X_BEFORE_ROTATION = True


def test_plain_scalar_across_if_keeps_python_interop() -> None:
    """A plain int living across an if still drives Python-level indexing.

    ``n`` is not yielded differently by any branch — it merely lives
    across the if. The equal-scalar passthrough must hand the code after
    the if a genuine Python ``int`` so ``angles[n]`` (Python list
    indexing) keeps working; a promoted symbolic handle raises
    ``TypeError`` at trace time. The x + rx(0.0) body pins the executed
    value deterministically.
    """

    @qmc.qkernel
    def kernel(dummy: qmc.UInt) -> qmc.Bit:
        angles = [0.1, 0.0, 0.3]
        n = 1
        q = qmc.qubit("q")
        if _APPLY_X_BEFORE_ROTATION:
            q = qmc.x(q)
        q = qmc.rx(q, angles[n])
        return qmc.measure(q)

    assert _sample_single(kernel, {"dummy": 0}) == 1
