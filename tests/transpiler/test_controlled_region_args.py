"""Cross-backend regressions for RegionArgs inside controlled blocks."""

from __future__ import annotations

import math
from typing import Any

import pytest

import qamomile.circuit as qmc

BACKENDS = [
    pytest.param("qiskit", id="qiskit"),
    pytest.param("quri_parts", marks=pytest.mark.quri_parts, id="quri_parts"),
    pytest.param("cudaq", marks=pytest.mark.cudaq, id="cudaq"),
]


def _make_transpiler(backend: str) -> Any:
    """Build one installed backend transpiler or skip its test.

    Args:
        backend (str): One of ``qiskit``, ``quri_parts``, or ``cudaq``.

    Returns:
        Any: Backend transpiler when its optional SDK is installed.
    """
    if backend == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()
    if backend == "quri_parts":
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        return QuriPartsTranspiler()
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


@qmc.qkernel
def _carried_index_body(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Flip successive targets and consume the final RegionArg result.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(0)
    for _iteration in qmc.range(2):
        targets[index] = qmc.x(targets[index])
        index = index + 1
    targets[index - 1] = qmc.x(targets[index - 1])
    return targets


@qmc.qkernel
def _nested_carried_index_body(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Carry an index through nested statically replayed loops.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(0)
    for _outer in qmc.range(2):
        for _inner in qmc.range(1):
            targets[index] = qmc.x(targets[index])
            index = index + 1
    return targets


@qmc.qkernel
def _inverse_source_with_carried_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Flip successive targets using a carry equal to the loop induction.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(0)
    for _iteration in qmc.range(2):
        targets[index] = qmc.x(targets[index])
        index = index + 1
    return targets


@qmc.qkernel
def _inverse_with_carried_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply the inverse of a loop whose quantum index is carried.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    return qmc.inverse(_inverse_source_with_carried_index)(targets)


@qmc.qkernel
def _inverse_source_with_noncommuting_carry(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply an ordered CX chain selected by a carried index.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Three-qubit target register.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after CX(0, 1) then CX(1, 2).
    """
    index = qmc.uint(0)
    for _iteration in qmc.range(2):
        targets[index], targets[index + 1] = qmc.cx(
            targets[index],
            targets[index + 1],
        )
        index = index + 1
    return targets


@qmc.qkernel
def _inverse_noncommuting_roundtrip(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Round-trip a noncommuting carried CX sequence from state |100>.

    The source maps |100> to |111>. Applying CX(1, 2) then CX(0, 1)
    restores |100>, while incorrectly replaying the forward order produces
    |101>, so the measured tuple detects an iteration-order regression.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Three-qubit target register in |000>.

    Returns:
        qmc.Vector[qmc.Qubit]: Register restored to |100> only when the
            inverse replays loop iterations in reverse order.
    """
    targets[0] = qmc.x(targets[0])
    targets = _inverse_source_with_noncommuting_carry(targets)
    targets = qmc.inverse(_inverse_source_with_noncommuting_carry)(targets)
    return targets


@qmc.qkernel
def _inverse_with_nested_carried_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invert the existing nested additive carried-index body.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    return qmc.inverse(_nested_carried_index_body)(targets)


@qmc.qkernel
def _inverse_with_post_loop_carried_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invert a body that consumes a final RegionArg after its loop.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    return qmc.inverse(_carried_index_body)(targets)


@qmc.qkernel
def _inverse_source_with_post_nested_carried_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Use a nested RegionArg result before the next outer yield.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(0)
    for _outer in qmc.range(4, 0, -2):
        for _inner in qmc.range(1, 6, 2):
            targets[index] = qmc.x(targets[index])
            index = index + 1
        targets[index] = qmc.x(targets[index])
        index = index + 1
    return targets


@qmc.qkernel
def _inverse_with_post_nested_carried_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invert nested negative-step loops with a post-inner carry use.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    return qmc.inverse(_inverse_source_with_post_nested_carried_index)(targets)


@qmc.qkernel
def _inverse_source_with_offset_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Flip targets one and two using an offset additive carry.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(1)
    for _iteration in qmc.range(2):
        targets[index] = qmc.x(targets[index])
        index = index + 1
    return targets


@qmc.qkernel
def _inverse_with_offset_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply the inverse of a loop with an offset carried index.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    return qmc.inverse(_inverse_source_with_offset_index)(targets)


@qmc.qkernel
def _inverse_source_with_affine_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Advance a carry while the range induction advances by two.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(0)
    for _iteration in qmc.range(0, 4, 2):
        targets[index] = qmc.x(targets[index])
        index = index + 1
    return targets


@qmc.qkernel
def _inverse_with_affine_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply the inverse of a constant affine carried-index loop.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    return qmc.inverse(_inverse_source_with_affine_index)(targets)


@qmc.qkernel
def _inverse_source_with_derived_index_init(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Initialize a carried index from an earlier classical result.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(0)
    index = index + 1
    for _iteration in qmc.range(2):
        targets[index] = qmc.x(targets[index])
        index = index + 1
    return targets


@qmc.qkernel
def _inverse_with_derived_index_init(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invert a loop whose carry initializer is a classical result.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    return qmc.inverse(_inverse_source_with_derived_index_init)(targets)


@qmc.qkernel
def _inverse_source_with_later_carried_bound(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Drive a later loop bound from a previous loop's final carry.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    count = qmc.uint(0)
    for _count_iteration in qmc.range(3):
        count = count + 1
    for _gate_iteration in qmc.range(count):
        targets[0] = qmc.x(targets[0])
    return targets


@qmc.qkernel
def _inverse_with_later_carried_bound(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invert sequential loops joined by a carried static bound.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    return qmc.inverse(_inverse_source_with_later_carried_bound)(targets)


@qmc.qkernel
def _inverse_source_with_zero_trip_nested_recurrence(
    targets: qmc.Vector[qmc.Qubit],
    inner_count: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Place a non-additive carry update inside a possibly empty loop.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.
        inner_count (qmc.UInt): Static trip count for the nested loop.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(0)
    for _outer in qmc.range(2):
        for _inner in qmc.range(inner_count):
            index = index * 2
        targets[index] = qmc.x(targets[index])
        index = index + 1
    return targets


@qmc.qkernel
def _inverse_with_zero_trip_nested_recurrence(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invert after binding an unsupported nested recurrence to zero trips.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    return qmc.inverse(_inverse_source_with_zero_trip_nested_recurrence)(
        targets,
        qmc.uint(0),
    )


@qmc.qkernel
def _inverse_source_with_identity_float_angle(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Keep a Float angle unchanged across two RX iterations.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to rotate.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated target register.
    """
    angle = qmc.float_(math.pi / 2)
    for _iteration in qmc.range(2):
        targets[0] = qmc.rx(targets[0], angle)
        angle = angle + 0.0
    return targets


@qmc.qkernel
def _inverse_with_identity_float_angle(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply the inverse of a loop with an identity Float carry.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to rotate.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated target register.
    """
    return qmc.inverse(_inverse_source_with_identity_float_angle)(targets)


@qmc.qkernel
def _inverse_source_with_derived_identity_float(
    targets: qmc.Vector[qmc.Qubit],
    theta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Use a derived gate angle while carrying Float identity directly.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to rotate.
        theta (qmc.Float): Incoming base rotation angle.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated target register.
    """
    angle = theta
    for _iteration in qmc.range(2):
        shifted = angle + 0.25
        targets[0] = qmc.rx(targets[0], shifted)
        angle = angle
    return targets


@qmc.qkernel
def _inverse_with_derived_identity_float(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invert a true identity Float carry with a derived gate expression.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to rotate.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated target register.
    """
    theta = qmc.float_(math.pi / 2 - 0.25)
    return qmc.inverse(_inverse_source_with_derived_identity_float)(targets, theta)


@qmc.qkernel
def _inverse_source_with_derived_zero_delta_float(
    targets: qmc.Vector[qmc.Qubit],
    theta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Use a derived gate angle off a separate zero-delta yield path.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to rotate.
        theta (qmc.Float): Incoming base rotation angle.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated target register.
    """
    angle = theta
    for _iteration in qmc.range(2):
        shifted = angle + 0.25
        targets[0] = qmc.rx(targets[0], shifted)
        angle = angle + 0.0
    return targets


@qmc.qkernel
def _inverse_with_derived_zero_delta_float(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invert a zero-delta Float yield with a derived gate expression.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to rotate.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated target register.
    """
    theta = qmc.float_(math.pi / 2 - 0.25)
    return qmc.inverse(_inverse_source_with_derived_zero_delta_float)(targets, theta)


@qmc.qkernel
def _inverse_source_with_cancelled_float_recurrence(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Use an algebraically zero but IEEE-sensitive Float recurrence.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to rotate.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated target register.
    """
    angle = qmc.float_(1e16)
    for _iteration in qmc.range(2):
        targets[0] = qmc.rx(targets[0], angle)
        angle = angle + 1.0
        angle = angle - 1.0
    return targets


@qmc.qkernel
def _inverse_with_cancelled_float_recurrence() -> qmc.Vector[qmc.Qubit]:
    """Attempt to invert a reassociated Float recurrence.

    Returns:
        qmc.Vector[qmc.Qubit]: Target register if construction succeeds.
    """
    targets = qmc.qubit_array(1, "targets")
    return qmc.inverse(_inverse_source_with_cancelled_float_recurrence)(targets)


@qmc.qkernel
def _inverse_source_with_rounding_sensitive_float(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Use a Float carry whose additions cannot be reassociated.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to rotate.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated target register.
    """
    angle = qmc.float_(1e16)
    for _iteration in qmc.range(3):
        targets[0] = qmc.rx(targets[0], angle)
        angle = angle + 1.0
    return targets


@qmc.qkernel
def _inverse_with_rounding_sensitive_float(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Attempt to invert a loop with a nonzero Float recurrence.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to rotate.

    Returns:
        qmc.Vector[qmc.Qubit]: Rotated target register if construction succeeds.
    """
    return qmc.inverse(_inverse_source_with_rounding_sensitive_float)(targets)


@qmc.qkernel
def _controlled_inverse_with_rounding_sensitive_float() -> qmc.Vector[qmc.Qubit]:
    """Attempt to control an inverse with a nonzero Float recurrence.

    Returns:
        qmc.Vector[qmc.Qubit]: Target register if construction succeeds.
    """
    control = qmc.x(qmc.qubit("control"))
    targets = qmc.qubit_array(1, "targets")
    control, targets = qmc.control(_inverse_with_rounding_sensitive_float)(
        control,
        targets,
    )
    return targets


@qmc.qkernel
def _inverse_source_with_nonadditive_index(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Use a non-additive carry that inverse fallback cannot represent.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(1)
    for _iteration in qmc.range(2):
        targets[index] = qmc.x(targets[index])
        index = index * 2
    return targets


@qmc.qkernel
def _inverse_with_nonadditive_index() -> qmc.Vector[qmc.Qubit]:
    """Attempt to invert a loop with a non-additive carried index.

    Returns:
        qmc.Vector[qmc.Qubit]: Target register if construction succeeds.
    """
    targets = qmc.qubit_array(4, "targets")
    return qmc.inverse(_inverse_source_with_nonadditive_index)(targets)


@qmc.qkernel
def _inverse_source_with_coupled_index_first(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Update an index from a sibling identity carry listed afterward.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(0)
    delta = qmc.uint(2)
    for _iteration in qmc.range(2):
        targets[index] = qmc.x(targets[index])
        index = index + delta
        delta = delta
    return targets


@qmc.qkernel
def _inverse_source_with_coupled_delta_first(
    targets: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Update an index from a sibling identity carry listed beforehand.

    Args:
        targets (qmc.Vector[qmc.Qubit]): Target register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target register.
    """
    index = qmc.uint(0)
    delta = qmc.uint(2)
    for _iteration in qmc.range(2):
        delta = delta
        targets[index] = qmc.x(targets[index])
        index = index + delta
    return targets


@qmc.qkernel
def _inverse_with_coupled_index_first() -> qmc.Vector[qmc.Qubit]:
    """Attempt to invert the index-first coupled-carry source.

    Returns:
        qmc.Vector[qmc.Qubit]: Target register if construction succeeds.
    """
    targets = qmc.qubit_array(4, "targets")
    return qmc.inverse(_inverse_source_with_coupled_index_first)(targets)


@qmc.qkernel
def _inverse_with_coupled_delta_first() -> qmc.Vector[qmc.Qubit]:
    """Attempt to invert the delta-first coupled-carry source.

    Returns:
        qmc.Vector[qmc.Qubit]: Target register if construction succeeds.
    """
    targets = qmc.qubit_array(4, "targets")
    return qmc.inverse(_inverse_source_with_coupled_delta_first)(targets)


def _controlled_sample_kernel(body: Any) -> Any:
    """Build a two-target sample kernel under an enabled control.

    Args:
        body (Any): Qkernel that consumes and returns a target vector.

    Returns:
        Any: Qkernel applying ``body`` to two target qubits.
    """
    return _controlled_sample_kernel_with_width(body, 2)


def _direct_sample_kernel(body: Any, target_count: int) -> Any:
    """Build a sample kernel applying ``body`` without an outer control.

    Args:
        body (Any): Qkernel that consumes and returns a target vector.
        target_count (int): Number of target qubits to allocate.

    Returns:
        Any: Module-independent qkernel ready for backend transpilation.
    """

    @qmc.qkernel
    def kernel() -> qmc.Vector[qmc.Bit]:
        """Apply the supplied body and measure every target.

        Returns:
            qmc.Vector[qmc.Bit]: Measured target bits.
        """
        targets = qmc.qubit_array(target_count, "targets")
        targets = body(targets)
        return qmc.measure(targets)

    return kernel


def _controlled_sample_kernel_with_width(body: Any, target_count: int) -> Any:
    """Build a controlled sample kernel with a configurable target width.

    Args:
        body (Any): Qkernel that consumes and returns a target vector.
        target_count (int): Number of target qubits to allocate.

    Returns:
        Any: Qkernel applying ``body`` under an enabled control.
    """
    controlled = qmc.control(body)

    @qmc.qkernel
    def kernel() -> qmc.Vector[qmc.Bit]:
        """Enable the control, apply the body, and measure its targets.

        Returns:
            qmc.Vector[qmc.Bit]: Measured target bits.
        """
        control = qmc.x(qmc.qubit("control"))
        targets = qmc.qubit_array(target_count, "targets")
        control, targets = controlled(control, targets)
        return qmc.measure(targets)

    return kernel


def _sample(backend: str, body: Any) -> tuple[int, ...]:
    """Execute one deterministic controlled body on ``backend``.

    Args:
        backend (str): Backend key accepted by ``_make_transpiler``.
        body (Any): Two-target qkernel to apply under control.

    Returns:
        tuple[int, ...]: Deterministic measured target tuple.
    """
    transpiler = _make_transpiler(backend)
    executable = transpiler.transpile(_controlled_sample_kernel(body))
    sampled = executable.sample(transpiler.executor(), shots=16).result()
    assert len(sampled.results) == 1
    value, count = sampled.results[0]
    assert count == 16
    return value


def _sample_inverse(
    backend: str,
    body: Any,
    target_count: int,
    controlled: bool,
) -> tuple[int, ...]:
    """Execute one direct or controlled inverse regression.

    Args:
        backend (str): Backend key accepted by ``_make_transpiler``.
        body (Any): Inverse qkernel to execute.
        target_count (int): Number of measured target qubits.
        controlled (bool): Whether to wrap ``body`` in an enabled control.

    Returns:
        tuple[int, ...]: Deterministic measured target tuple.
    """
    transpiler = _make_transpiler(backend)
    kernel = (
        _controlled_sample_kernel_with_width(body, target_count)
        if controlled
        else _direct_sample_kernel(body, target_count)
    )
    executable = transpiler.transpile(kernel)
    sampled = executable.sample(transpiler.executor(), shots=16).result()
    assert len(sampled.results) == 1
    value, count = sampled.results[0]
    assert count == 16
    return value


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_carried_index_body, (1, 0)),
        (_nested_carried_index_body, (1, 1)),
    ],
)
def test_controlled_region_args_execute_across_backends(
    backend: str,
    body: Any,
    expected: tuple[int, ...],
) -> None:
    """Controlled loop carries advance on every nested static iteration."""
    assert _sample(backend, body) == expected


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("controlled", [False, True], ids=["direct", "controlled"])
@pytest.mark.parametrize(
    ("body", "target_count", "expected"),
    [
        (_inverse_with_carried_index, 2, (1, 1)),
        (_inverse_noncommuting_roundtrip, 3, (1, 0, 0)),
        (_inverse_with_nested_carried_index, 2, (1, 1)),
        (_inverse_with_post_loop_carried_index, 2, (1, 0)),
        (_inverse_with_post_nested_carried_index, 8, (1,) * 8),
        (_inverse_with_offset_index, 3, (0, 1, 1)),
        (_inverse_with_affine_index, 2, (1, 1)),
        (_inverse_with_derived_index_init, 3, (0, 1, 1)),
        (_inverse_with_later_carried_bound, 1, (1,)),
        (_inverse_with_zero_trip_nested_recurrence, 2, (1, 1)),
        (_inverse_with_identity_float_angle, 1, (1,)),
        (_inverse_with_derived_identity_float, 1, (1,)),
        (_inverse_with_derived_zero_delta_float, 1, (1,)),
    ],
    ids=[
        "canonical",
        "noncommuting-order-roundtrip",
        "nested",
        "post-loop-result",
        "post-nested-result-negative-step",
        "offset",
        "affine-stride",
        "derived-init",
        "later-carried-bound",
        "zero-trip-nested",
        "float-identity",
        "float-identity-derived-gate",
        "float-zero-delta-derived-gate",
    ],
)
def test_inverse_additive_region_args_execute_across_backends(
    backend: str,
    controlled: bool,
    body: Any,
    target_count: int,
    expected: tuple[int, ...],
) -> None:
    """Direct and controlled inverse preserve additive carry snapshots."""
    assert _sample_inverse(backend, body, target_count, controlled) == expected


def test_inverse_rejects_nonadditive_region_arg_during_construction() -> None:
    """Unsupported recurrences fail before backend-dependent emission."""
    with pytest.raises(
        NotImplementedError,
        match="constant additive recurrence",
    ):
        _ = _inverse_with_nonadditive_index.block


@pytest.mark.parametrize(
    "wrapper",
    [
        pytest.param(_inverse_with_coupled_index_first, id="index-first"),
        pytest.param(_inverse_with_coupled_delta_first, id="delta-first"),
    ],
)
def test_inverse_rejects_coupled_region_args_independent_of_order(
    wrapper: Any,
) -> None:
    """Sibling carry dependencies reject for either RegionArg order."""
    with pytest.raises(NotImplementedError, match="Coupled carries"):
        _ = wrapper.block


@pytest.mark.parametrize(
    "wrapper",
    [
        pytest.param(
            _inverse_with_rounding_sensitive_float,
            id="inverse",
        ),
        pytest.param(
            _controlled_inverse_with_rounding_sensitive_float,
            id="controlled-inverse",
        ),
    ],
)
def test_inverse_rejects_nonzero_float_region_arg_during_construction(
    wrapper: Any,
) -> None:
    """IEEE-sensitive Float recurrences fail before backend emission."""
    with pytest.raises(
        NotImplementedError,
        match="Float values only with an identity recurrence",
    ):
        _ = wrapper.block


def test_inverse_rejects_cancelled_float_region_arg_during_construction() -> None:
    """Nonzero intermediates on the yielded path remain unsupported."""
    with pytest.raises(
        NotImplementedError,
        match="Float values only with an identity recurrence",
    ):
        _ = _inverse_with_cancelled_float_recurrence.block
