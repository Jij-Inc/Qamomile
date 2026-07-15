"""Tests for direct Qamomile semantic IR to HUGR lowering."""

from __future__ import annotations

import copy
import dataclasses
import math
from types import SimpleNamespace

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.stdlib import amplitude_encoding

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from hugr import ops
from hugr.package import Package
from hugr.std.float import FloatVal

from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.transpiler.errors import EmitError, ValidationError
from qamomile.hugr import HugrTranspiler


def _hugr_operation_names(package: Package) -> list[str]:
    """Return canonical names for operations that expose one."""
    names: list[str] = []
    for _, data in package.modules[0].nodes():
        name = getattr(data.op, "name", None)
        if callable(name):
            names.append(str(name()))
    return names


def _hugr_float_constants(package: Package) -> list[float]:
    """Return concrete floating-point constants in graph order.

    Returns:
        list[float]: Literal floating-point values loaded into the graph.
    """
    return [
        data.op.val.v
        for _, data in package.modules[0].nodes()
        if isinstance(data.op, ops.Const) and isinstance(data.op.val, FloatVal)
    ]


@qmc.qkernel
def _hugr_bell(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Build a parameterized Bell-like program for HUGR tests."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.ry(left, theta)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_helper(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply a reusable Hadamard operation."""
    return qmc.h(qubit)


@qmc.qkernel
def _hugr_identity(qubit: qmc.Qubit) -> qmc.Qubit:
    """Return one qubit unchanged for global-phase tests."""
    return qubit


@qmc.qkernel
def _hugr_identity_vector(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Return a quantum vector unchanged for region-liveness tests."""
    return qubits


@qmc.qkernel
def _hugr_standalone_global_phase(theta: qmc.Float) -> qmc.Bit:
    """Apply an unobservable entrypoint global phase."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.global_phase(_hugr_identity, theta)(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_phased_helper(qubit: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply a parameterized global phase inside a reusable helper."""
    return qmc.global_phase(_hugr_identity, theta)(qubit)


@qmc.qkernel
def _hugr_p_helper(qubit: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply a phase gate whose scalar factor matters under control."""
    return qmc.p(qubit, theta)


@qmc.qkernel
def _hugr_controlled_global_phase(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Control a reusable helper containing global phase."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_phased_helper)(control, target, theta)
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_control_call_global_phase(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Attach a target-global phase directly to a controlled call."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_identity)(
        control,
        target,
        global_phase=theta,
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_powered_control_call_global_phase(
    theta: qmc.Float,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Apply a call-site phase together with a static controlled power."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_identity)(
        control,
        target,
        power=3,
        global_phase=theta,
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_two_control_call_global_phase(
    theta: qmc.Float,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Apply a call-site phase on a two-control active subspace."""
    controls = qmc.qubit_array(2, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(_hugr_identity, num_controls=2)(
        controls,
        target,
        global_phase=theta,
    )
    return qmc.measure(controls), qmc.measure(target)


@qmc.qkernel
def _hugr_three_control_global_phase(
    theta: qmc.Float,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Control a phased reusable body with three coherent controls."""
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(_hugr_phased_helper, num_controls=3)(
        controls,
        target,
        theta,
    )
    return qmc.measure(controls), qmc.measure(target)


@qmc.qkernel
def _hugr_four_control_powered_call_global_phase(
    theta: qmc.Float,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Repeat a four-control call-site phase with a static power."""
    controls = qmc.qubit_array(4, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(_hugr_identity, num_controls=4)(
        controls,
        target,
        power=2,
        global_phase=theta,
    )
    return qmc.measure(controls), qmc.measure(target)


@qmc.qkernel
def _hugr_dynamic_power_control_call_global_phase(
    theta: qmc.Float,
    power: qmc.UInt,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Expose a runtime controlled power outside the current HUGR profile."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_identity)(
        control,
        target,
        power=power,
        global_phase=theta,
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_inverse_global_phase(theta: qmc.Float) -> qmc.Bit:
    """Invert a reusable helper containing global phase."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.inverse(_hugr_phased_helper)(qubit, theta)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_static_for_phased_helper(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Apply phased X operations in a statically bounded loop."""
    for _ in qmc.range(2):
        qubit = qmc.global_phase(_hugr_identity, theta)(qubit)
        qubit = qmc.x(qubit)
    return qubit


@qmc.qkernel
def _hugr_controlled_static_for_global_phase(
    theta: qmc.Float,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Control a reusable static loop containing global phase."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_static_for_phased_helper)(
        control,
        target,
        theta,
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_descending_static_for_phased_helper(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Apply phased X operations over a negative-step range."""
    for _ in qmc.range(5, 0, -2):
        qubit = qmc.global_phase(_hugr_identity, theta)(qubit)
        qubit = qmc.x(qubit)
    return qubit


@qmc.qkernel
def _hugr_controlled_descending_static_for(
    theta: qmc.Float,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Control a reusable negative-step static loop."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_descending_static_for_phased_helper)(
        control,
        target,
        theta,
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_inverse_descending_static_for(theta: qmc.Float) -> qmc.Bit:
    """Invert a reusable negative-step static loop."""
    target = qmc.qubit("target")
    target = qmc.inverse(_hugr_descending_static_for_phased_helper)(target, theta)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_nested_static_for_phased_helper(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Apply phased S operations in nested static loops."""
    for _ in qmc.range(2):
        for _ in qmc.range(3):
            qubit = qmc.global_phase(_hugr_identity, theta)(qubit)
            qubit = qmc.s(qubit)
    return qubit


@qmc.qkernel
def _hugr_nested_static_phase_only_helper(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Apply only phased identities in nested static loops."""
    for _ in qmc.range(2):
        for _ in qmc.range(2):
            qubit = qmc.global_phase(_hugr_identity, theta)(qubit)
    return qubit


@qmc.qkernel
def _hugr_three_control_nested_static_phase(
    theta: qmc.Float,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Control a nested static loop with three coherent controls."""
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(
        _hugr_nested_static_phase_only_helper,
        num_controls=3,
    )(controls, target, theta)
    return qmc.measure(controls), qmc.measure(target)


@qmc.qkernel
def _hugr_inverse_nested_static_for_global_phase(theta: qmc.Float) -> qmc.Bit:
    """Invert a reusable nested static loop containing global phase."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.inverse(_hugr_nested_static_for_phased_helper)(qubit, theta)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_dependent_nested_static_for_phased_helper(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Apply phased X operations under an outer-index-dependent bound."""
    for outer in qmc.range(3):
        for _ in qmc.range(outer + 1):
            qubit = qmc.global_phase(_hugr_identity, theta)(qubit)
            qubit = qmc.x(qubit)
    return qubit


@qmc.qkernel
def _hugr_controlled_dependent_nested_static_for(
    theta: qmc.Float,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Control nested static loops whose inner bound uses the outer index."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_dependent_nested_static_for_phased_helper)(
        control,
        target,
        theta,
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_static_vector_for_helper(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply X to two statically indexed vector elements."""
    for index in qmc.range(2):
        qubits[index] = qmc.x(qubits[index])
    return qubits


@qmc.qkernel
def _hugr_controlled_static_vector_for() -> tuple[qmc.Bit, qmc.Vector[qmc.Bit]]:
    """Control a static loop over vector elements."""
    control = qmc.qubit("control")
    targets = qmc.qubit_array(2, "targets")
    control, targets = qmc.control(_hugr_static_vector_for_helper)(
        control,
        targets,
    )
    return qmc.measure(control), qmc.measure(targets)


@qmc.qkernel
def _hugr_repeated_static_array_element_helper(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Reuse one statically indexed element across controlled iterations."""
    qubits[0] = qmc.x(qubits[0])
    for _ in qmc.range(2):
        qubits[0] = qmc.z(qubits[0])
    return qubits


@qmc.qkernel
def _hugr_controlled_repeated_static_array_element() -> tuple[
    qmc.Bit, qmc.Vector[qmc.Bit]
]:
    """Control repeated operations on one array element."""
    control = qmc.qubit("control")
    targets = qmc.qubit_array(2, "targets")
    control, targets = qmc.control(_hugr_repeated_static_array_element_helper)(
        control,
        targets,
    )
    return qmc.measure(control), qmc.measure(targets)


@qmc.qkernel
def _hugr_inverse_static_vector_for() -> qmc.Vector[qmc.Bit]:
    """Invert a static loop over vector elements."""
    targets = qmc.qubit_array(2, "targets")
    targets = qmc.inverse(_hugr_static_vector_for_helper)(targets)
    return qmc.measure(targets)


@qmc.qkernel
def _hugr_dynamic_for_helper(
    qubit: qmc.Qubit,
    count: qmc.UInt,
) -> qmc.Qubit:
    """Apply a phased X in a runtime-bounded loop."""
    for _ in qmc.range(count):
        qubit = qmc.global_phase(_hugr_identity, 0.125)(qubit)
        qubit = qmc.x(qubit)
    return qubit


@qmc.qkernel
def _hugr_controlled_zero_trip_for() -> tuple[qmc.Bit, qmc.Bit]:
    """Control a loop specialized to an empty range at its call site."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_dynamic_for_helper)(control, target, 0)
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_inverse_zero_trip_for() -> qmc.Bit:
    """Invert a loop specialized to an empty range at its call site."""
    target = qmc.qubit("target")
    target = qmc.inverse(_hugr_dynamic_for_helper)(target, 0)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_dynamic_for_with_tail_helper(
    qubit: qmc.Qubit,
    count: qmc.UInt,
) -> qmc.Qubit:
    """Apply a tail gate after a potentially empty transformed loop."""
    for _ in qmc.range(count):
        qubit = qmc.global_phase(_hugr_identity, 0.125)(qubit)
        qubit = qmc.x(qubit)
    return qmc.z(qubit)


@qmc.qkernel
def _hugr_controlled_zero_trip_for_with_tail() -> tuple[qmc.Bit, qmc.Bit]:
    """Control a tail gate after a loop specialized to an empty range."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_dynamic_for_with_tail_helper)(
        control,
        target,
        0,
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_inverse_zero_trip_for_with_tail() -> qmc.Bit:
    """Invert a tail gate after a loop specialized to an empty range."""
    target = qmc.qubit("target")
    target = qmc.inverse(_hugr_dynamic_for_with_tail_helper)(target, 0)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_controlled_dynamic_for(count: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
    """Control a reusable runtime-bounded loop."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_dynamic_for_helper)(
        control,
        target,
        count,
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_classical_carry_for_helper(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Update a classical angle across static loop iterations."""
    angle = theta
    for _ in qmc.range(2):
        angle = angle + theta
        qubit = qmc.rz(qubit, angle)
    return qubit


@qmc.qkernel
def _hugr_controlled_classical_carry_for(
    theta: qmc.Float,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Control a static loop with a classical carried value."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_classical_carry_for_helper)(
        control,
        target,
        theta,
    )
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_controlled_p(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Control a reusable P body without dropping its scalar factor."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_p_helper)(control, target, theta)
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_controlled_phase_in_if_with_dead_target() -> qmc.Bit:
    """Drop an unchanged phase-only target after a runtime conditional."""
    predicate = qmc.x(qmc.qubit("predicate"))
    take_branch = qmc.measure(predicate)
    control = qmc.h(qmc.qubit("control"))
    target = qmc.qubit("target")
    if take_branch:
        control, target = qmc.control(_hugr_identity)(
            control,
            target,
            global_phase=math.pi,
        )
    control = qmc.h(control)
    return qmc.measure(control)


@qmc.qkernel
def _hugr_controlled_phase_in_if_with_dead_vector() -> qmc.Bit:
    """Drop an unchanged phase-only vector after a runtime conditional."""
    predicate = qmc.x(qmc.qubit("predicate"))
    take_branch = qmc.measure(predicate)
    control = qmc.h(qmc.qubit("control"))
    targets = qmc.qubit_array(2, "targets")
    if take_branch:
        control, targets = qmc.control(_hugr_identity_vector)(
            control,
            targets,
            global_phase=math.pi,
        )
    control = qmc.h(control)
    return qmc.measure(control)


@qmc.qkernel
def _hugr_controlled_phase_in_static_for(theta: qmc.Float) -> qmc.Bit:
    """Repeat a controlled phase call inside a statically bounded loop."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    for _ in qmc.range(2):
        control, target = qmc.control(_hugr_identity)(
            control,
            target,
            global_phase=theta,
        )
    return qmc.measure(control)


@qmc.qkernel
def _hugr_controlled_phase_in_while() -> qmc.Bit:
    """Carry a controlled phase call through measurement feedback."""
    loop_qubit = qmc.x(qmc.qubit("loop"))
    loop_qubit, run = qmc.project_z(loop_qubit)
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    while run:
        control, target = qmc.control(_hugr_identity)(
            control,
            target,
            global_phase=math.pi,
        )
        loop_qubit = qmc.reset(loop_qubit)
        loop_qubit, run = qmc.project_z(loop_qubit)
    return qmc.measure(control)


@qmc.qkernel
def _hugr_three_control_vector_phase_in_if() -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Carry vector controls from a phased call through a runtime branch."""
    predicate = qmc.x(qmc.qubit("predicate"))
    take_branch = qmc.measure(predicate)
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    if take_branch:
        controls, target = qmc.control(_hugr_identity, num_controls=3)(
            controls,
            target,
            global_phase=math.pi,
        )
    return qmc.measure(controls), qmc.measure(target)


@qmc.qkernel
def _hugr_three_control_vector_phase_in_for(
    count: qmc.UInt,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Carry phased-call controls through element and broadcast gates in a loop."""
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    for _ in qmc.range(count):
        controls, target = qmc.control(_hugr_identity, num_controls=3)(
            controls,
            target,
            global_phase=math.pi / 3,
        )
        controls, target = qmc.control(_hugr_identity, num_controls=3)(
            controls,
            target,
            global_phase=math.pi / 5,
        )
        controls[0] = qmc.x(controls[0])
        controls = qmc.z(controls)
    return qmc.measure(controls), qmc.measure(target)


@qmc.qkernel
def _hugr_three_control_vector_phase_in_while() -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Carry phased-call controls through a while-loop and following CX."""
    loop_qubit = qmc.x(qmc.qubit("loop"))
    loop_qubit, run = qmc.project_z(loop_qubit)
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    while run:
        controls, target = qmc.control(_hugr_identity, num_controls=3)(
            controls,
            target,
            global_phase=math.pi / 3,
        )
        controls, target = qmc.control(_hugr_identity, num_controls=3)(
            controls,
            target,
            global_phase=math.pi / 5,
        )
        controls[0], controls[1] = qmc.cx(controls[0], controls[1])
        loop_qubit = qmc.reset(loop_qubit)
        loop_qubit, run = qmc.project_z(loop_qubit)
    return qmc.measure(controls), qmc.measure(target)


@qmc.qkernel
def _hugr_three_control_vector_phase_in_nested_while_if() -> tuple[
    qmc.Vector[qmc.Bit], qmc.Bit
]:
    """Carry vector controls through nested runtime while and if regions."""
    loop_qubit = qmc.x(qmc.qubit("loop"))
    loop_qubit, run = qmc.project_z(loop_qubit)
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    while run:
        if run:
            controls, target = qmc.control(_hugr_identity, num_controls=3)(
                controls,
                target,
                global_phase=math.pi,
            )
        loop_qubit = qmc.reset(loop_qubit)
        loop_qubit, run = qmc.project_z(loop_qubit)
    return qmc.measure(controls), qmc.measure(target)


@qmc.qkernel
def _hugr_helper_entrypoint() -> qmc.Bit:
    """Call a body-backed helper without forcing circuit-style inlining."""
    qubit = qmc.qubit("qubit")
    qubit = _hugr_helper(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_inline_feedback_helper(
    qubit: qmc.Qubit,
    trigger: qmc.Bit,
) -> tuple[qmc.Qubit, qmc.Bit]:
    """Update a callable while condition from a fresh measurement."""
    while trigger:
        qubit = qmc.reset(qubit)
        qubit, trigger = qmc.project_z(qubit)
    return qubit, trigger


@qmc.qkernel
def _hugr_measured_inline_feedback_call() -> qmc.Bit:
    """Pass a measurement-backed condition into an inline-policy helper."""
    qubit = qmc.x(qmc.qubit("qubit"))
    qubit, trigger = qmc.project_z(qubit)
    qubit, _ = _hugr_inline_feedback_helper(qubit, trigger)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_constant_inline_feedback_call() -> qmc.Bit:
    """Pass an invalid constant condition into an inline-policy helper."""
    qubit = qmc.qubit("qubit")
    qubit, _ = _hugr_inline_feedback_helper(qubit, True)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_array_helper(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply a reusable broadcast Hadamard operation."""
    return qmc.h(qubits)


@qmc.qkernel
def _hugr_static_for() -> qmc.Bit:
    """Exercise a statically bounded loop with a carried qubit."""
    qubit = qmc.qubit("qubit")
    for _ in qmc.range(3):
        qubit = qmc.h(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_measurement_if() -> qmc.Bit:
    """Exercise native HUGR conditional dataflow with a carried qubit."""
    qubit = qmc.qubit("qubit")
    qubit, bit = qmc.project_z(qubit)
    if bit:
        qubit = qmc.x(qubit)
    else:
        qubit = qmc.h(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_destructive_branch_array_merge() -> qmc.Bit:
    """Measure different elements of one captured array across branches."""
    qubits = qmc.qubit_array(3, "qubits")
    condition = qmc.measure(qubits[0])
    qubits[2] = qmc.x(qubits[2])
    if condition:
        result = qmc.measure(qubits[1])
    else:
        result = qmc.measure(qubits[2])
    return result


@qmc.qkernel
def _hugr_vector_program() -> qmc.Vector[qmc.Bit]:
    """Exercise fixed-size vector allocation, broadcast, and measurement."""
    qubits = qmc.qubit_array(3, "qubits")
    qubits = qmc.h(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_phase_helper(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply a phase gate used by inverse-call legalization tests."""
    return qmc.s(qubit)


@qmc.qkernel
def _hugr_inverse_program() -> qmc.Bit:
    """Invoke an inverse helper at a HUGR call site."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.inverse(_hugr_phase_helper)(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_x_helper(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply X for controlled-call legalization tests."""
    return qmc.x(qubit)


@qmc.composite_gate(name="shared_display_name")
def _hugr_same_name_h(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply H through the first of two equally named callables."""
    return qmc.h(qubit)


@qmc.composite_gate(name="shared_display_name")
def _hugr_same_name_x(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply X through the second of two equally named callables."""
    return qmc.x(qubit)


@qmc.qkernel
def _hugr_same_name_program() -> tuple[qmc.Bit, qmc.Bit]:
    """Invoke distinct callable bodies that share one display name."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = _hugr_same_name_h(left)
    right = _hugr_same_name_x(right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_controlled_program() -> tuple[qmc.Bit, qmc.Bit]:
    """Invoke a one-control helper at a HUGR call site."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_x_helper)(control, target)
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_state_preparation() -> qmc.Vector[qmc.Bit]:
    """Exercise a semantic state-preparation callable in HUGR.

    Returns:
        qmc.Vector[qmc.Bit]: Prepared-state measurements.
    """
    qubits = qmc.qubit_array(2, "qubits")
    qubits = amplitude_encoding(qubits, [1.0, 2.0, 3.0, 4.0])
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_ripple_carry() -> tuple[qmc.Vector[qmc.Bit], qmc.Bit, qmc.Bit]:
    """Exercise a semantic ripple-carry callable in HUGR.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Bit, qmc.Bit]: Accumulator, carry,
            and overflow measurements.
    """
    left = qmc.qubit_array(2, "left")
    right = qmc.qubit_array(2, "right")
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    _, right, carry, overflow = qmc.ripple_carry_add(left, right, carry, overflow)
    return qmc.measure(right), qmc.measure(carry), qmc.measure(overflow)


@qmc.qkernel
def _hugr_multi_controlled_x() -> qmc.Bit:
    """Exercise semantic MCX at HUGR's current two-control boundary.

    Returns:
        qmc.Bit: Target measurement.
    """
    controls = qmc.x(qmc.qubit_array(2, "controls"))
    target = qmc.qubit("target")
    _, target = qmc.mcx(controls, target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_measurement_while() -> qmc.Bit:
    """Exercise measurement-controlled while with carried quantum state."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.x(qubit)
    qubit, bit = qmc.project_z(qubit)
    while bit:
        if bit:
            qubit = qmc.x(qubit)
        else:
            qubit = qmc.h(qubit)
        qubit = qmc.reset(qubit)
        qubit, bit = qmc.project_z(qubit)
    return bit


@qmc.qkernel
def _hugr_sequential_feedback() -> qmc.Bit:
    """Exercise two sequential measurement-dependent corrections."""
    source = qmc.qubit("source")
    ancilla = qmc.qubit("ancilla")
    target = qmc.qubit("target")
    source = qmc.h(source)
    source, ancilla = qmc.cx(source, ancilla)
    ancilla, target = qmc.cx(ancilla, target)
    first = qmc.measure(source)
    if first:
        target = qmc.z(target)
    second = qmc.measure(ancilla)
    if second:
        target = qmc.x(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_loop_with_feedback() -> qmc.Bit:
    """Exercise a carried qubit through nested loop and conditional regions."""
    qubit = qmc.qubit("qubit")
    for _ in qmc.range(3):
        qubit, bit = qmc.project_z(qubit)
        if bit:
            qubit = qmc.x(qubit)
        else:
            qubit = qmc.h(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_runtime_bounded_for(repetitions: qmc.UInt) -> qmc.Bit:
    """Exercise a runtime-bounded loop without circuit-style unrolling."""
    qubit = qmc.qubit("qubit")
    for _ in qmc.range(repetitions):
        qubit = qmc.h(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_runtime_scalar_carry(repetitions: qmc.UInt) -> qmc.Bit:
    """Thread a scalar RegionArg through a runtime loop and nested if."""
    selector = qmc.measure(qmc.qubit("selector"))
    total = 0.0
    for _ in qmc.range(repetitions):
        if selector:
            total = total + math.pi
        else:
            total = total + 2.0 * math.pi
    target = qmc.qubit("target")
    target = qmc.rx(target, total)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_runtime_uint_carry(repetitions: qmc.UInt) -> qmc.UInt:
    """Accumulate a UInt RegionArg using the runtime loop index."""
    total = qmc.uint(0)
    for index in qmc.range(repetitions):
        total = total + index
    return total


@qmc.qkernel
def _hugr_runtime_uint_conditional_carry(repetitions: qmc.UInt) -> qmc.UInt:
    """Use a UInt loop comparison to guard a carried update."""
    total = qmc.uint(0)
    for index in qmc.range(repetitions):
        if index > 0:
            total = total + index
        else:
            total = total + 0
    return total


@qmc.qkernel
def _hugr_runtime_uint_descending(
    start: qmc.UInt,
    stop: qmc.UInt,
) -> qmc.UInt:
    """Use an unsigned descending runtime range comparison."""
    total = qmc.uint(0)
    for index in qmc.range(start, stop, -1):
        total = total + index
    return total


@qmc.qkernel
def _hugr_runtime_array_carry(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Carry a qubit array through a runtime loop and nested conditional."""
    selector = qmc.measure(qmc.qubit("selector"))
    qubits = qmc.qubit_array(2, "qubits")
    for _ in qmc.range(repetitions):
        if selector:
            qubits = qmc.h(qubits)
        else:
            qubits = qmc.x(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_runtime_array_broadcast(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Carry a broadcast-updated qubit array through a runtime loop."""
    qubits = qmc.qubit_array(2, "qubits")
    for _ in qmc.range(repetitions):
        qubits = qmc.h(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_runtime_array_element(repetitions: qmc.UInt) -> qmc.Bit:
    """Carry one array element through a runtime loop."""
    qubits = qmc.qubit_array(2, "qubits")
    target = qubits[0]
    for _ in qmc.range(repetitions):
        target = qmc.h(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_runtime_scalar_call(repetitions: qmc.UInt) -> qmc.Bit:
    """Carry a scalar qubit returned by a direct call through a runtime loop."""
    target = qmc.qubit("target")
    for _ in qmc.range(repetitions):
        target = _hugr_helper(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_runtime_array_call(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Carry a qubit array returned by a direct call through a runtime loop."""
    qubits = qmc.qubit_array(2, "qubits")
    for _ in qmc.range(repetitions):
        qubits = _hugr_array_helper(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_runtime_quantum_swap(
    repetitions: qmc.UInt,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Expose quantum handle rebinding that needs explicit linear slots."""
    left = qmc.x(qmc.qubit("left"))
    right = qmc.qubit("right")
    for _ in qmc.range(repetitions):
        left, right = right, left
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_runtime_if_qubit(repetitions: qmc.UInt) -> qmc.Bit:
    """Carry a scalar qubit through a runtime loop and nested conditional."""
    selector = qmc.measure(qmc.qubit("selector"))
    target = qmc.qubit("target")
    for _ in qmc.range(repetitions):
        if selector:
            target = qmc.x(target)
        else:
            target = qmc.h(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_while_array_broadcast() -> qmc.Vector[qmc.Bit]:
    """Carry a broadcast-updated qubit array through a measurement while."""
    control = qmc.x(qmc.qubit("control"))
    control, trigger = qmc.project_z(control)
    qubits = qmc.qubit_array(2, "qubits")
    while trigger:
        qubits = qmc.h(qubits)
        control = qmc.reset(control)
        control, trigger = qmc.project_z(control)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_vector_element_if() -> qmc.Bit:
    """Use a measured vector element as a native conditional predicate."""
    bits = qmc.measure(qmc.qubit_array(2, "condition"))
    target = qmc.qubit("target")
    if bits[0]:
        target = qmc.x(target)
    else:
        target = qmc.h(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_vector_element_merge() -> qmc.Bit:
    """Select between two measured array elements in a native conditional."""
    data = qmc.measure(qmc.qubit_array(2, "data"))
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        result = data[0]
    else:
        result = data[1]
    return result


@qmc.qkernel
def _hugr_runtime_quantum_index(
    repetitions: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Expose dynamic indexing of a flattened linear quantum array."""
    qubits = qmc.qubit_array(2, "qubits")
    for index in qmc.range(repetitions):
        qubits[index] = qmc.x(qubits[index])
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_quantum_element_merge() -> qmc.Bit:
    """Expose data-dependent selection between different linear qubits."""
    qubits = qmc.qubit_array(2, "qubits")
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        target = qubits[0]
    else:
        target = qubits[1]
    return qmc.measure(target)


@qmc.qkernel
def _hugr_duplicate_quantum_footprint() -> qmc.Bit:
    """Expose whole-array and element aliases as duplicate merge outputs."""
    qubits = qmc.qubit_array(2, "qubits")
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        qubits[0] = qmc.x(qubits[0])
        target = qubits[0]
    else:
        qubits[0] = qmc.h(qubits[0])
        target = qubits[0]
    return qmc.measure(target)


@qmc.qkernel
def _hugr_local_duplicate_quantum_footprint() -> tuple[qmc.Bit, qmc.Bit]:
    """Expose duplicate aliases of branch-local quantum resources."""
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        qubits = qmc.qubit_array(2, "true_qubits")
        qubits[0] = qmc.x(qubits[0])
        target = qubits[0]
    else:
        qubits = qmc.qubit_array(2, "false_qubits")
        qubits[0] = qmc.h(qubits[0])
        target = qubits[0]
    return qmc.measure(target), qmc.measure(qubits[1])


@qmc.qkernel
def _hugr_conditional_quantum_swap() -> tuple[qmc.Bit, qmc.Bit]:
    """Permute two complete linear resources through a conditional."""
    left = qmc.x(qmc.qubit("left"))
    right = qmc.qubit("right")
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        left, right = right, left
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_branch_local_unused_qubit() -> qmc.Bit:
    """Allocate an unused companion qubit inside each conditional branch."""
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        qubits = qmc.qubit_array(2, "true_qubits")
        target = qubits[0]
    else:
        qubits = qmc.qubit_array(2, "false_qubits")
        target = qubits[0]
    return qmc.measure(target)


@qmc.qkernel
def _hugr_loop_local_unused_qubit(repetitions: qmc.UInt) -> qmc.Bit:
    """Allocate a loop-local scratch qubit that does not escape the body."""
    for _ in qmc.range(repetitions):
        qmc.qubit("scratch")
    return qmc.measure(qmc.qubit("output"))


@qmc.qkernel
def _hugr_disjoint_array_alias_if() -> tuple[qmc.Bit, qmc.Bit]:
    """Carry an element alias and a disjoint sibling through an if."""
    qubits = qmc.qubit_array(2, "qubits")
    target = qmc.h(qubits[0])
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        target = qmc.x(target)
        qubits[1] = qmc.h(qubits[1])
    else:
        target = qmc.z(target)
        qubits[1] = qmc.x(qubits[1])
    return qmc.measure(target), qmc.measure(qubits[1])


@qmc.qkernel
def _hugr_single_array_element_alias_if() -> tuple[qmc.Bit, qmc.Bit]:
    """Carry one element alias while its sibling stays outside the if."""
    qubits = qmc.qubit_array(2, "qubits")
    target = qmc.h(qubits[0])
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        target = qmc.x(target)
    else:
        target = qmc.z(target)
    return qmc.measure(target), qmc.measure(qubits[1])


@qmc.qkernel
def _hugr_disjoint_array_alias_for(
    repetitions: qmc.UInt,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Carry an element alias and a disjoint sibling through a loop."""
    qubits = qmc.qubit_array(2, "qubits")
    target = qmc.h(qubits[0])
    for _ in qmc.range(repetitions):
        target = qmc.x(target)
        qubits[1] = qmc.h(qubits[1])
    return qmc.measure(target), qmc.measure(qubits[1])


@qmc.qkernel
def _hugr_while_scalar_carry() -> qmc.UInt:
    """Expose the unsupported non-condition scalar carry in a while loop."""
    trigger = qmc.measure(qmc.qubit("trigger"))
    total = qmc.uint(0)
    while trigger:
        total = total + 1
        trigger = qmc.measure(qmc.qubit("next_trigger"))
    return total


@qmc.qkernel
def _hugr_register_feedback() -> qmc.Vector[qmc.Bit]:
    """Correct an indexed register element from a measurement predicate."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0], syndrome = qmc.project_z(qubits[0])
    if syndrome:
        qubits[1] = qmc.x(qubits[1])
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_independent_loop_carriers() -> tuple[qmc.Bit, qmc.Bit]:
    """Carry two independent linear qubits through one loop."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    for _ in qmc.range(2):
        left = qmc.h(left)
        right = qmc.x(right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_gate_coverage() -> tuple[qmc.Bit, qmc.Bit]:
    """Exercise composite two-qubit gates and destructive reset lowering."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left, right = qmc.swap(left, right)
    left, right = qmc.rzz(left, right, 0.25)
    left = qmc.p(left, 0.125)
    left, right = qmc.cp(left, right, -0.5)
    left = qmc.reset(left)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_two_control_program() -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
    """Legalize a two-control reusable X call."""
    controls = qmc.qubit_array(2, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(_hugr_x_helper, num_controls=2)(
        controls,
        target,
    )
    return (
        qmc.measure(controls[0]),
        qmc.measure(controls[1]),
        qmc.measure(target),
    )


@qmc.qkernel
def _hugr_pauli_evolution(
    hamiltonian: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Exercise semantic Pauli evolution at the HUGR target boundary."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits = qmc.h(qubits)
    qubits = qmc.pauli_evolve(qubits, hamiltonian, gamma)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_pauli_helper(
    qubits: qmc.Vector[qmc.Qubit],
    hamiltonian: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply Pauli evolution through a transformable helper body."""
    return qmc.pauli_evolve(qubits, hamiltonian, gamma)


@qmc.qkernel
def _hugr_controlled_pauli_evolution(
    hamiltonian: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Exercise a controlled semantic Pauli evolution at the HUGR edge."""
    control = qmc.qubit("control")
    targets = qmc.qubit_array(2, "targets")
    control = qmc.h(control)
    control, targets = qmc.control(_hugr_pauli_helper)(
        control,
        targets,
        hamiltonian=hamiltonian,
        gamma=gamma,
    )
    return qmc.measure(targets)


def _explicit_inverse_zero_trip_for_kernel() -> SimpleNamespace:
    """Keep an empty ForOperation visible at the HUGR inverse boundary.

    The frontend normally folds a literal empty range before HUGR sees it.
    This target-boundary fixture rewrites only the range bounds of an otherwise
    genuine inverse qkernel block so zero-trip inverse lowering is exercised.
    """
    transpiler = HugrTranspiler()
    block = copy.deepcopy(
        transpiler.compiler.to_block(
            _hugr_inverse_descending_static_for,
            parameters=["theta"],
        )
    )
    for position, operation in enumerate(block.operations):
        if not isinstance(operation, InverseBlockOperation):
            continue
        assert operation.source_block is not None
        source = operation.source_block
        loop = next(
            nested for nested in source.operations if isinstance(nested, ForOperation)
        )
        start, stop, step = loop.operands[:3]
        empty_loop = dataclasses.replace(
            loop,
            operands=[
                start.with_const(2),
                stop.with_const(2),
                step.with_const(1),
            ],
        )
        source = dataclasses.replace(
            source,
            operations=[
                empty_loop,
                *(
                    nested
                    for nested in source.operations
                    if not isinstance(nested, ForOperation)
                ),
            ],
        )
        block.operations[position] = dataclasses.replace(
            operation,
            source_block=source,
        )
        return SimpleNamespace(block=block)
    raise AssertionError("Expected an inverse block in the boundary fixture")


def _explicit_controlled_inverse_kernel() -> SimpleNamespace:
    """Keep a controlled inverse primitive at the HUGR target boundary."""
    transpiler = HugrTranspiler()
    block = copy.deepcopy(transpiler.compiler.to_block(_hugr_controlled_program))
    for position, operation in enumerate(block.operations):
        if not isinstance(operation, ControlledUOperation):
            continue
        assert operation.block is not None
        block.operations[position] = InverseBlockOperation(
            operands=list(operation.operands),
            results=list(operation.results),
            num_control_qubits=1,
            num_target_qubits=1,
            custom_name="controlled_x_inverse",
            source_block=operation.block,
            implementation_block=operation.block,
            callable_ref=operation.callable_ref,
            callable_attrs=dict(operation.callable_attrs),
        )
        return SimpleNamespace(block=block)
    raise AssertionError("Expected a controlled call in the boundary fixture")


def _explicit_three_control_inverse_phase_kernel() -> SimpleNamespace:
    """Keep a three-control inverse phase at the HUGR target boundary."""
    transpiler = HugrTranspiler()
    block = copy.deepcopy(
        transpiler.compiler.to_block(
            _hugr_three_control_global_phase,
            parameters=["theta"],
        )
    )
    for position, operation in enumerate(block.operations):
        if not isinstance(operation, ControlledUOperation):
            continue
        assert operation.block is not None
        block.operations[position] = InverseBlockOperation(
            operands=list(operation.operands),
            results=list(operation.results),
            num_control_qubits=len(operation.control_operands),
            num_target_qubits=1,
            custom_name="three_control_phase_inverse",
            source_block=operation.block,
            implementation_block=operation.block,
            callable_ref=operation.callable_ref,
            callable_attrs=dict(operation.callable_attrs),
        )
        return SimpleNamespace(block=block, build=lambda **_: block)
    raise AssertionError("Expected a three-control call in the boundary fixture")


def _explicit_controlled_inverse_pauli_kernel() -> SimpleNamespace:
    """Keep a controlled inverse Pauli body at the HUGR target boundary."""
    hamiltonian = 0.25 * qm_o.X(0) + 0.5 * qm_o.Y(0) * qm_o.Z(1) + 0.75
    bindings = {"hamiltonian": hamiltonian, "gamma": 0.125}
    transpiler = HugrTranspiler()
    block = copy.deepcopy(
        transpiler.compiler.to_block(
            _hugr_controlled_pauli_evolution,
            bindings=bindings,
        )
    )
    for position, operation in enumerate(block.operations):
        if not isinstance(operation, ControlledUOperation):
            continue
        assert operation.block is not None
        block.operations[position] = InverseBlockOperation(
            operands=list(operation.operands),
            results=list(operation.results),
            num_control_qubits=1,
            num_target_qubits=2,
            custom_name="controlled_pauli_inverse",
            source_block=operation.block,
            implementation_block=operation.block,
            callable_ref=operation.callable_ref,
            callable_attrs=dict(operation.callable_attrs),
        )
        return SimpleNamespace(
            block=block,
            build=lambda **_: block,
            bindings=bindings,
        )
    raise AssertionError("Expected a controlled Pauli call in the boundary fixture")


@pytest.mark.hugr
def test_hugr_compiles_bound_quantum_program_and_validates() -> None:
    """A bound quantum program produces a validator-clean HUGR package."""
    compiled = HugrTranspiler().transpile(
        _hugr_bell,
        bindings={"theta": math.pi / 2},
    )

    assert isinstance(compiled.artifact, Package)
    assert compiled.metadata.target == "hugr"
    assert compiled.metadata.pipeline == "program_graph"


@pytest.mark.hugr
def test_hugr_preserves_runtime_parameter_as_public_input() -> None:
    """A runtime angle remains a Float input of the public HUGR function."""
    package = HugrTranspiler().to_hugr(_hugr_bell, parameters=["theta"])
    operations = [data.op for _, data in package.modules[0].nodes()]
    [main] = [op for op in operations if getattr(op, "f_name", None) == "main"]

    assert len(main.inputs) == 1
    assert "float64" in str(main.inputs[0])


@pytest.mark.hugr
def test_hugr_preserves_standalone_phase_without_dropping_abi() -> None:
    """A standalone phase uses TKET's intrinsic and remains a public input."""
    package = HugrTranspiler().to_hugr(
        _hugr_standalone_global_phase,
        parameters=["theta"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]
    [main] = [op for op in operations if getattr(op, "f_name", None) == "main"]

    assert len(main.inputs) == 1
    assert "float64" in str(main.inputs[0])
    assert "tket.global_phase.global_phase" in _hugr_operation_names(package)


@pytest.mark.hugr
@pytest.mark.parametrize(
    ("kernel", "expected_scales"),
    [
        (_hugr_controlled_global_phase, [0.5 / math.pi, 1.0 / math.pi]),
        (_hugr_control_call_global_phase, [0.5 / math.pi, 1.0 / math.pi]),
        (_hugr_inverse_global_phase, [-1.0 / math.pi]),
    ],
)
def test_hugr_preserves_transformed_global_phase(
    kernel: qmc.QKernel,
    expected_scales: list[float],
) -> None:
    """HUGR emits transformed phase semantics instead of dropping them."""
    package = HugrTranspiler().to_hugr(kernel, parameters=["theta"])

    assert "tket.global_phase.global_phase" in _hugr_operation_names(package)
    assert _hugr_float_constants(package) == pytest.approx(
        expected_scales,
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
def test_hugr_controls_static_for_body_with_global_phase() -> None:
    """A controlled static loop repeats its exact relative-phase body."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_controlled_static_for_global_phase,
        parameters=["theta"],
    )

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.global_phase.global_phase") == 2
    assert names.count("tket.quantum.CX") == 2
    assert _hugr_float_constants(package) == pytest.approx(
        [0.5 / math.pi, 1.0 / math.pi] * 2,
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
@pytest.mark.parametrize(
    "kernel",
    [_hugr_controlled_zero_trip_for, _hugr_inverse_zero_trip_for],
)
def test_hugr_preserves_wires_through_zero_trip_transformed_for(
    kernel: qmc.QKernel,
) -> None:
    """A zero-trip transformed loop emits neither its phase nor its gate."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(kernel)

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.global_phase.global_phase") == len(range(0))
    assert names.count("tket.quantum.CX") == len(range(0))
    assert names.count("tket.quantum.X") == len(range(0))


@pytest.mark.hugr
@pytest.mark.parametrize(
    ("kernel", "expected_tail"),
    [
        (_hugr_controlled_zero_trip_for_with_tail, "tket.quantum.CZ"),
        (_hugr_inverse_zero_trip_for_with_tail, "tket.quantum.Z"),
    ],
)
def test_hugr_preserves_tail_after_zero_trip_transformed_for(
    kernel: qmc.QKernel,
    expected_tail: str,
) -> None:
    """A zero-trip body aliases its phantom output before the tail gate."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(kernel)

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count(expected_tail) == 1
    assert names.count("tket.global_phase.global_phase") == 0
    assert names.count("tket.quantum.CX") == 0
    assert names.count("tket.quantum.X") == 0


@pytest.mark.hugr
def test_hugr_inverts_explicit_zero_trip_static_for_at_target_boundary() -> None:
    """An inverse ForOperation with an empty range preserves its input wire."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_explicit_inverse_zero_trip_for_kernel())

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.global_phase.global_phase") == len(range(2, 2))
    assert names.count("tket.quantum.X") == len(range(2, 2))


@pytest.mark.hugr
def test_hugr_controls_negative_step_static_for_exactly() -> None:
    """Controlled negative-step unrolling follows Python range cardinality."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_controlled_descending_static_for,
        parameters=["theta"],
    )

    transpiler.target.validate(package)
    iterations = len(range(5, 0, -2))
    names = _hugr_operation_names(package)
    assert names.count("tket.global_phase.global_phase") == iterations
    assert names.count("tket.quantum.CX") == iterations
    assert _hugr_float_constants(package) == pytest.approx(
        [0.5 / math.pi, 1.0 / math.pi] * iterations,
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
def test_hugr_inverts_negative_step_static_for_exactly() -> None:
    """Inverse negative-step unrolling reverses every Python range iteration."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_inverse_descending_static_for,
        parameters=["theta"],
    )

    transpiler.target.validate(package)
    iterations = len(range(5, 0, -2))
    names = _hugr_operation_names(package)
    assert names.count("tket.global_phase.global_phase") == iterations
    assert names.count("tket.quantum.X") == iterations
    assert _hugr_float_constants(package) == pytest.approx(
        [-1.0 / math.pi] * iterations,
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
def test_hugr_inverts_nested_static_for_body_with_global_phase() -> None:
    """Nested static loops reverse and negate every phased body application."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_inverse_nested_static_for_global_phase,
        parameters=["theta"],
    )

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert _hugr_operation_names(package).count("tket.global_phase.global_phase") == 6
    assert sum("name='Sdg'" in str(operation) for operation in operations) == 6
    assert _hugr_float_constants(package) == pytest.approx(
        [-1.0 / math.pi] * 6,
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
def test_hugr_controls_outer_index_dependent_nested_static_for() -> None:
    """An outer static index resolves the nested bound before control."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_controlled_dependent_nested_static_for,
        parameters=["theta"],
    )

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.global_phase.global_phase") == 6
    assert names.count("tket.quantum.CX") == 6


@pytest.mark.hugr
def test_hugr_controls_static_for_over_vector_elements() -> None:
    """Static element indexing advances each controlled linear target wire."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_controlled_static_vector_for)

    transpiler.target.validate(package)
    assert _hugr_operation_names(package).count("tket.quantum.CX") == 2


@pytest.mark.hugr
def test_hugr_controls_repeated_static_array_element_linearly() -> None:
    """Repeated controlled element access always consumes the latest wire."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_controlled_repeated_static_array_element)

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.quantum.CX") == 1
    assert names.count("tket.quantum.CZ") == 2


@pytest.mark.hugr
def test_hugr_inverts_static_for_over_vector_elements() -> None:
    """Inverse static indexing advances each adjoint linear target wire."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_inverse_static_vector_for)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert sum("name='X'" in str(operation) for operation in operations) == 2


@pytest.mark.hugr
def test_hugr_rejects_dynamic_for_inside_transformed_call() -> None:
    """Runtime bounds fail explicitly instead of emitting one iteration."""
    with pytest.raises(
        EmitError,
        match="bounds must be compile-time constants",
    ) as error:
        HugrTranspiler().to_hugr(
            _hugr_controlled_dynamic_for,
            parameters=["count"],
        )

    assert error.value.operation == "ForOperation"


@pytest.mark.hugr
def test_hugr_rejects_classical_carry_inside_transformed_for() -> None:
    """Classical loop carries fail explicitly at the transformed loop."""
    with pytest.raises(
        EmitError,
        match="do not support loop-carried values.*angle",
    ) as error:
        HugrTranspiler().to_hugr(
            _hugr_controlled_classical_carry_for,
            parameters=["theta"],
        )

    assert error.value.operation == "ForOperation"


@pytest.mark.hugr
def test_hugr_repeats_phase_and_body_for_static_controlled_power() -> None:
    """Static power repeats the complete exact phase realization."""
    package = HugrTranspiler().to_hugr(
        _hugr_powered_control_call_global_phase,
        parameters=["theta"],
    )
    names = _hugr_operation_names(package)

    assert names.count("tket.global_phase.global_phase") == 3
    assert names.count("tket.quantum.Rz") == 3


@pytest.mark.hugr
def test_hugr_preserves_two_control_phase_exactly() -> None:
    """Two controls use the exact CP decomposition including its factor."""
    package = HugrTranspiler().to_hugr(
        _hugr_two_control_call_global_phase,
        parameters=["theta"],
    )
    names = _hugr_operation_names(package)

    assert names.count("tket.global_phase.global_phase") == 1
    assert names.count("tket.quantum.Rz") == 1
    assert names.count("tket.quantum.CRz") == 1
    assert _hugr_float_constants(package) == pytest.approx(
        [0.25 / math.pi, 0.5 / math.pi, 1.0 / math.pi],
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
def test_hugr_preserves_three_control_phase_with_clean_ancillas() -> None:
    """Three controls use an uncomputed and freed conjunction cascade."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_three_control_global_phase,
        parameters=["theta"],
    )

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.quantum.Toffoli") == 2
    assert names.count("tket.quantum.QAlloc") == 5
    assert names.count("tket.quantum.QFree") == 5
    assert names.count("tket.quantum.Rz") == 1
    assert names.count("tket.quantum.CRz") == 1
    assert names.count("tket.global_phase.global_phase") == 1
    assert _hugr_float_constants(package) == pytest.approx(
        [0.25 / math.pi, 0.5 / math.pi, 1.0 / math.pi],
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
def test_hugr_repeats_four_control_phase_cascade_for_power() -> None:
    """Static power repeats each exact four-control phase cascade."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_four_control_powered_call_global_phase,
        parameters=["theta"],
    )

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.quantum.Toffoli") == 8
    assert names.count("tket.quantum.QAlloc") == 9
    assert names.count("tket.quantum.QFree") == 9
    assert names.count("tket.quantum.Rz") == 2
    assert names.count("tket.quantum.CRz") == 2
    assert names.count("tket.global_phase.global_phase") == 2
    assert _hugr_float_constants(package) == pytest.approx(
        [0.25 / math.pi, 0.5 / math.pi, 1.0 / math.pi] * 2,
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
def test_hugr_inverts_three_control_phase_cascade() -> None:
    """Inverse lowering negates the phase and restores every linear wire."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _explicit_three_control_inverse_phase_kernel(),
        parameters=["theta"],
    )

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.quantum.Toffoli") == 2
    assert names.count("tket.quantum.QAlloc") == 5
    assert names.count("tket.quantum.QFree") == 5
    assert names.count("tket.quantum.CRz") == 1
    assert _hugr_float_constants(package) == pytest.approx(
        [-0.25 / math.pi, -0.5 / math.pi, -1.0 / math.pi],
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
def test_hugr_preserves_three_control_phase_in_nested_static_for() -> None:
    """Nested static loops reuse clean phase-conjunction ancillas safely."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_three_control_nested_static_phase,
        parameters=["theta"],
    )

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.quantum.Toffoli") == 8
    assert names.count("tket.quantum.QAlloc") == 8
    assert names.count("tket.quantum.QFree") == 8
    assert names.count("tket.quantum.Rz") == 4
    assert names.count("tket.quantum.CRz") == 4
    assert names.count("tket.global_phase.global_phase") == 4


@pytest.mark.hugr
def test_hugr_controls_phase_gate_with_its_exact_scalar_factor() -> None:
    """A reusable P body becomes exact CP rather than projective CRz only."""
    package = HugrTranspiler().to_hugr(
        _hugr_controlled_p,
        parameters=["theta"],
    )
    names = _hugr_operation_names(package)

    assert names.count("tket.global_phase.global_phase") == 1
    assert names.count("tket.quantum.Rz") == 1
    assert names.count("tket.quantum.CRz") == 1


@pytest.mark.hugr
def test_hugr_frees_dead_phase_target_in_runtime_if() -> None:
    """A captured identity target is consumed when neither branch yields it."""
    package = HugrTranspiler().to_hugr(_hugr_controlled_phase_in_if_with_dead_target)

    assert "tket.global_phase.global_phase" in _hugr_operation_names(package)


@pytest.mark.hugr
def test_hugr_frees_dead_phase_vector_in_runtime_if() -> None:
    """Captured vector elements are consumed once when no branch yields them."""
    package = HugrTranspiler().to_hugr(_hugr_controlled_phase_in_if_with_dead_vector)

    assert "tket.global_phase.global_phase" in _hugr_operation_names(package)


@pytest.mark.hugr
def test_hugr_preserves_controlled_phase_inside_static_for() -> None:
    """Static loop unrolling repeats the complete relative-phase realization."""
    package = HugrTranspiler().to_hugr(
        _hugr_controlled_phase_in_static_for,
        parameters=["theta"],
    )

    assert _hugr_operation_names(package).count("tket.global_phase.global_phase") == 2
    expected = [0.5 / math.pi, 1.0 / math.pi] * 2
    assert _hugr_float_constants(package) == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.hugr
def test_hugr_preserves_controlled_phase_inside_runtime_while() -> None:
    """A TailLoop branch keeps its controlled phase and linear carriers valid."""
    package = HugrTranspiler().to_hugr(_hugr_controlled_phase_in_while)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(type(operation).__name__ == "TailLoop" for operation in operations)
    assert "tket.global_phase.global_phase" in _hugr_operation_names(package)


@pytest.mark.hugr
@pytest.mark.parametrize(
    ("kernel", "parameters", "applications"),
    [
        (_hugr_three_control_vector_phase_in_if, [], 1),
        (_hugr_three_control_vector_phase_in_for, ["count"], 2),
        (_hugr_three_control_vector_phase_in_while, [], 2),
        (_hugr_three_control_vector_phase_in_nested_while_if, [], 1),
    ],
)
def test_hugr_publishes_vector_controls_from_runtime_regions(
    kernel: qmc.QKernel,
    parameters: list[str],
    applications: int,
) -> None:
    """Runtime regions publish a reconstructed control vector for later use."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(kernel, parameters=parameters)

    transpiler.target.validate(package)
    names = _hugr_operation_names(package)
    assert names.count("tket.quantum.Toffoli") == 2 * applications
    assert names.count("tket.global_phase.global_phase") == applications
    assert names.count("tket.quantum.QAlloc") == names.count("tket.quantum.QFree")


@pytest.mark.hugr
def test_hugr_rejects_runtime_controlled_power_explicitly() -> None:
    """A dynamic body power never degrades to one silent application."""
    with pytest.raises(
        EmitError,
        match="compile-time positive integer",
    ) as error:
        HugrTranspiler().to_hugr(
            _hugr_dynamic_power_control_call_global_phase,
            parameters=["theta", "power"],
        )

    assert error.value.operation == "ControlledUOperation"


@pytest.mark.hugr
def test_hugr_preserves_body_backed_helper_as_function_call() -> None:
    """Hierarchical helper semantics lower to HUGR definition and call nodes."""
    package = HugrTranspiler().to_hugr(_hugr_helper_entrypoint)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(
        getattr(op, "f_name", "").endswith("___hugr_helper__v1") for op in operations
    )
    assert any(type(op).__name__ == "Call" for op in operations)


@pytest.mark.hugr
def test_hugr_validates_inline_callable_while_at_its_call_site() -> None:
    """Contextual while validation keeps hierarchical HUGR call emission."""
    package = HugrTranspiler().to_hugr(_hugr_measured_inline_feedback_call)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(type(op).__name__ == "Call" for op in operations)
    assert any(type(op).__name__ == "TailLoop" for op in operations)


@pytest.mark.hugr
def test_hugr_rejects_nonmeasurement_inline_callable_while_call() -> None:
    """Inlining for validation still rejects a constant while condition."""
    with pytest.raises(ValidationError, match="measurement result"):
        HugrTranspiler().to_hugr(_hugr_constant_inline_feedback_call)


@pytest.mark.hugr
def test_hugr_distinguishes_callables_with_the_same_display_name() -> None:
    """Origin-qualified symbols prevent same-name callable miscompilation."""
    package = HugrTranspiler().to_hugr(_hugr_same_name_program)
    operations = [data.op for _, data in package.modules[0].nodes()]
    definitions = [
        op
        for op in operations
        if getattr(op, "f_name", "").endswith("__shared_display_name__v1")
    ]

    assert len(definitions) == 2
    assert any("name='H'" in str(op) for op in operations)
    assert any("name='X'" in str(op) for op in operations)


@pytest.mark.hugr
def test_hugr_package_serialization_round_trip_validates() -> None:
    """Serialized HUGR remains native-validator clean after deserialization."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_helper_entrypoint)
    restored = Package.from_bytes(package.to_bytes())

    transpiler.target.validate(restored)


@pytest.mark.hugr
def test_hugr_lowers_static_for_with_carried_quantum_ssa() -> None:
    """Static loops lower deterministically and remain validator-clean."""
    package = HugrTranspiler().to_hugr(_hugr_static_for)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum("name='H'" in str(op) for op in operations) == 3


@pytest.mark.hugr
def test_hugr_preserves_measurement_if_as_native_conditional() -> None:
    """Measurement-backed branching remains a HUGR Conditional region."""
    package = HugrTranspiler().to_hugr(_hugr_measurement_if)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(type(op).__name__ == "Conditional" for op in operations)


@pytest.mark.hugr
def test_hugr_captures_indexed_register_feedback_as_linear_region_input() -> None:
    """Indexed qubits cross Conditional boundaries without parent-graph edges."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_register_feedback)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert any(type(op).__name__ == "Conditional" for op in operations)


@pytest.mark.hugr
def test_hugr_keeps_independent_loop_carriers_distinct() -> None:
    """Linear-wire alias advancement cannot cross two independent qubits."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_independent_loop_carriers)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert sum("name='H'" in str(op) for op in operations) == 2
    assert sum("name='X'" in str(op) for op in operations) == 2


@pytest.mark.hugr
def test_hugr_validates_composite_gate_and_reset_lowering() -> None:
    """P, CP, and other extended gates produce exact validator-clean HUGR."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_gate_coverage)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert any("name='Reset'" in str(op) for op in operations)
    assert _hugr_operation_names(package).count("tket.global_phase.global_phase") == 2


@pytest.mark.hugr
def test_hugr_validates_two_control_call_legalization() -> None:
    """A two-control X call lowers without duplicating linear controls."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_two_control_program)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert any("name='Toffoli'" in str(op) for op in operations)


@pytest.mark.hugr
@pytest.mark.parametrize(
    "kernel",
    [_hugr_state_preparation, _hugr_ripple_carry, _hugr_multi_controlled_x],
)
def test_hugr_validates_semantic_composite_fallbacks(kernel: qmc.QKernel) -> None:
    """New semantic composites remain validator-clean HUGR callables."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(kernel)

    transpiler.target.validate(package)


@pytest.mark.hugr
def test_hugr_lowers_pauli_evolution_only_at_target_boundary() -> None:
    """A semantic Hamiltonian evolution becomes validator-clean TKET ops."""
    hamiltonian = 0.25 * qm_o.X(0) + 0.5 * qm_o.Y(0) * qm_o.Z(1) + 0.75
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_pauli_evolution,
        bindings={"hamiltonian": hamiltonian},
        parameters=["gamma"],
    )

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert sum("name='Rz'" in str(op) for op in operations) == 2
    assert any("name='CX'" in str(op) for op in operations)
    assert "tket.global_phase.global_phase" in _hugr_operation_names(package)


@pytest.mark.hugr
def test_hugr_lowers_controlled_pauli_evolution_at_target_boundary() -> None:
    """A one-control Pauli evolution uses TKET controlled rotations."""
    hamiltonian = 0.25 * qm_o.X(0) + 0.5 * qm_o.Y(0) * qm_o.Z(1) + 0.75
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_controlled_pauli_evolution,
        bindings={"hamiltonian": hamiltonian, "gamma": 0.125},
    )

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert sum("name='CRz'" in str(op) for op in operations) == 2
    assert sum("name='Rz'" in str(op) for op in operations) == 1
    assert "tket.global_phase.global_phase" in _hugr_operation_names(package)


@pytest.mark.hugr
def test_hugr_lowers_controlled_inverse_pauli_evolution() -> None:
    """Controlled inverse Pauli lowering resolves the body output vector."""
    transpiler = HugrTranspiler()
    kernel = _explicit_controlled_inverse_pauli_kernel()
    package = transpiler.to_hugr(kernel, bindings=kernel.bindings)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert sum("name='CRz'" in str(operation) for operation in operations) == 2
    assert sum("name='Rz'" in str(operation) for operation in operations) == 1


@pytest.mark.hugr
def test_hugr_preserves_tiny_controlled_identity_phase() -> None:
    """A tiny nonzero identity coefficient remains an observable relative phase."""
    hamiltonian = qm_o.Hamiltonian.identity(1e-16)
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_controlled_pauli_evolution,
        bindings={"hamiltonian": hamiltonian, "gamma": 0.125},
    )

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert sum("name='Rz'" in str(op) for op in operations) == 1
    assert not any("name='CRz'" in str(op) for op in operations)
    assert "tket.global_phase.global_phase" in _hugr_operation_names(package)


@pytest.mark.hugr
@pytest.mark.parametrize(
    ("kernel", "hamiltonian"),
    [
        (_hugr_pauli_evolution, qm_o.Hamiltonian.identity(1.0j)),
        (_hugr_controlled_pauli_evolution, (1.0 + 1.0j) * qm_o.X(0)),
    ],
)
def test_hugr_rejects_nonhermitian_pauli_evolution(
    kernel: qmc.QKernel,
    hamiltonian: qm_o.Hamiltonian,
) -> None:
    """HUGR never silently drops imaginary Hamiltonian coefficients."""
    with pytest.raises(EmitError, match="requires a Hermitian Hamiltonian"):
        HugrTranspiler().to_hugr(
            kernel,
            bindings={"hamiltonian": hamiltonian, "gamma": 0.125},
        )


@pytest.mark.hugr
def test_hugr_materializes_fixed_vectors_as_tuple_carriers() -> None:
    """Fixed vectors use HUGR tuples at boundaries and validate natively."""
    package = HugrTranspiler().to_hugr(_hugr_vector_program)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum("name='QAlloc'" in str(op) for op in operations) == 3
    assert any(type(op).__name__ == "MakeTuple" for op in operations)


@pytest.mark.hugr
def test_hugr_legalizes_inverse_calls_at_the_call_site() -> None:
    """Inverse calls inline as adjoint primitives while direct calls remain calls."""
    package = HugrTranspiler().to_hugr(_hugr_inverse_program)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any("name='Sdg'" in str(op) for op in operations)


@pytest.mark.hugr
def test_hugr_legalizes_controlled_x_calls_to_tket_cx() -> None:
    """One-control X helpers lower to the TKET CX extension operation."""
    package = HugrTranspiler().to_hugr(_hugr_controlled_program)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any("name='CX'" in str(op) for op in operations)


@pytest.mark.hugr
def test_hugr_lowers_controlled_inverse_primitive_body() -> None:
    """Controlled inverse lowering resolves the body's output-side wire."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_explicit_controlled_inverse_kernel())

    transpiler.target.validate(package)
    assert _hugr_operation_names(package).count("tket.quantum.CX") == 1


@pytest.mark.hugr
def test_hugr_preserves_measurement_while_as_tail_loop() -> None:
    """Measurement feedback lowers to a validator-clean native TailLoop."""
    package = HugrTranspiler().to_hugr(_hugr_measurement_while)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(type(op).__name__ == "TailLoop" for op in operations)
    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
def test_hugr_preserves_sequential_measurement_feedback_regions() -> None:
    """Sequential feedback corrections remain separate conditional regions."""
    package = HugrTranspiler().to_hugr(_hugr_sequential_feedback)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
def test_hugr_lowers_nested_loop_feedback_with_linear_qubit_carry() -> None:
    """Loop-unrolled conditionals preserve the carried qubit linearly."""
    package = HugrTranspiler().to_hugr(_hugr_loop_with_feedback)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "Conditional" for op in operations) == 3


@pytest.mark.hugr
def test_hugr_preserves_runtime_bounded_for_as_tail_loop() -> None:
    """A runtime range bound remains a compact native HUGR loop."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_bounded_for,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1
    assert sum("name='H'" in str(op) for op in operations) == 1


@pytest.mark.hugr
def test_hugr_threads_region_args_through_runtime_for() -> None:
    """Runtime TailLoop state includes scalar RegionArgs and nested merges."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_scalar_carry,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1
    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
def test_hugr_threads_uint_region_arg_through_runtime_for() -> None:
    """Runtime UInt arithmetic accepts RegionArg and loop-index wires."""
    HugrTranspiler().to_hugr(
        _hugr_runtime_uint_carry,
        parameters=["repetitions"],
    )


@pytest.mark.hugr
def test_hugr_lowers_uint_comparison_inside_runtime_for() -> None:
    """UInt comparisons compose with RegionArgs and nested conditionals."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_uint_conditional_carry,
        parameters=["repetitions"],
    )

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
@pytest.mark.parametrize(
    ("kernel", "parameters", "comparison_name"),
    [
        (_hugr_runtime_uint_carry, ["repetitions"], "ilt_u"),
        (_hugr_runtime_uint_descending, ["start", "stop"], "igt_u"),
    ],
)
def test_hugr_runtime_uint_ranges_use_unsigned_comparisons(
    kernel,
    parameters: list[str],
    comparison_name: str,
) -> None:
    """Runtime UInt bounds retain high-bit unsigned ordering semantics."""
    package = HugrTranspiler().to_hugr(kernel, parameters=parameters)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(comparison_name in str(operation) for operation in operations)


@pytest.mark.hugr
def test_hugr_flattens_array_captures_through_nested_runtime_control() -> None:
    """Runtime TailLoops flatten arrays and advance nested branch merges."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_array_carry,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1
    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
@pytest.mark.parametrize(
    "kernel",
    [
        _hugr_runtime_array_broadcast,
        _hugr_runtime_array_element,
        _hugr_runtime_array_call,
    ],
)
def test_hugr_preserves_array_capture_shapes_through_runtime_for(kernel) -> None:
    """Runtime TailLoops preserve whole-array and array-element carriers."""
    package = HugrTranspiler().to_hugr(kernel, parameters=["repetitions"])
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_advances_direct_call_result_through_runtime_for() -> None:
    """A same-resource direct call advances the captured scalar wire."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_scalar_call,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_rejects_quantum_resource_rebinding_in_runtime_for() -> None:
    """Quantum handle swaps fail before producing an invalid TailLoop."""
    with pytest.raises(EmitError, match="quantum resource rebinding"):
        HugrTranspiler().to_hugr(
            _hugr_runtime_quantum_swap,
            parameters=["repetitions"],
        )


@pytest.mark.hugr
def test_hugr_publishes_nested_scalar_quantum_merge_after_runtime_for() -> None:
    """A nested scalar quantum merge remains visible after a runtime loop."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_if_qubit,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1
    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
def test_hugr_preserves_array_capture_shape_through_while() -> None:
    """A measurement while carries every wire of a broadcast array."""
    package = HugrTranspiler().to_hugr(_hugr_while_array_broadcast)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_resolves_measured_vector_element_if_condition() -> None:
    """Structural measured elements resolve as native HUGR predicates."""
    package = HugrTranspiler().to_hugr(_hugr_vector_element_if)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "Conditional" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_resolves_classical_array_element_branch_yields() -> None:
    """Copyable array elements can cross native conditional outputs."""
    package = HugrTranspiler().to_hugr(_hugr_vector_element_merge)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "Conditional" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_rejects_dynamic_quantum_array_index_with_targeted_error() -> None:
    """Dynamic tuple indexing fails at the explicit HUGR support boundary."""
    with pytest.raises(EmitError, match="dynamic quantum array indexing"):
        HugrTranspiler().to_hugr(
            _hugr_runtime_quantum_index,
            parameters=["repetitions"],
        )


@pytest.mark.hugr
def test_hugr_rejects_data_dependent_quantum_element_selection() -> None:
    """Different captured qubits cannot share one semantic merge port."""
    with pytest.raises(EmitError, match="quantum resource selection"):
        HugrTranspiler().to_hugr(_hugr_quantum_element_merge)


@pytest.mark.hugr
def test_hugr_rejects_duplicate_quantum_merge_footprints() -> None:
    """Overlapping semantic aliases fail before native HUGR validation."""
    with pytest.raises(EmitError, match="same linear quantum resource"):
        HugrTranspiler().to_hugr(_hugr_duplicate_quantum_footprint)


@pytest.mark.hugr
def test_hugr_rejects_duplicate_branch_local_quantum_footprints() -> None:
    """Branch-local root and element aliases cannot duplicate one wire."""
    with pytest.raises(EmitError, match="same linear quantum resource"):
        HugrTranspiler().to_hugr(_hugr_local_duplicate_quantum_footprint)


@pytest.mark.hugr
def test_hugr_allows_complete_conditional_quantum_permutations() -> None:
    """A conditional may permute a complete set of linear merge ports."""
    package = HugrTranspiler().to_hugr(_hugr_conditional_quantum_swap)

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
def test_hugr_frees_unyielded_branch_local_qubits() -> None:
    """Conditional regions free local linear resources that do not escape."""
    package = HugrTranspiler().to_hugr(_hugr_branch_local_unused_qubit)

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
def test_hugr_rejects_consumed_qubit_in_implicit_array_merge() -> None:
    """A stale implicit root merge fails before native HUGR validation."""
    with pytest.raises(EmitError, match="destructively consumed"):
        HugrTranspiler().to_hugr(_hugr_destructive_branch_array_merge)


@pytest.mark.hugr
@pytest.mark.parametrize("runtime", [False, True])
def test_hugr_frees_unyielded_loop_local_qubits(runtime: bool) -> None:
    """Static and runtime loop bodies clean up local linear allocations."""
    kwargs = (
        {"parameters": ["repetitions"]} if runtime else {"bindings": {"repetitions": 2}}
    )
    package = HugrTranspiler().to_hugr(
        _hugr_loop_local_unused_qubit,
        **kwargs,
    )

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
def test_hugr_rejects_overlapping_array_alias_merges_through_if() -> None:
    """Whole-array and element aliases fail before duplicating a HUGR port."""
    with pytest.raises(EmitError, match="same linear quantum resource"):
        HugrTranspiler().to_hugr(_hugr_disjoint_array_alias_if)


@pytest.mark.hugr
def test_hugr_rejects_element_alias_with_implicit_whole_array_merge() -> None:
    """An element alias cannot duplicate its implicit whole-array merge."""
    with pytest.raises(EmitError, match="same linear quantum resource"):
        HugrTranspiler().to_hugr(_hugr_single_array_element_alias_if)


@pytest.mark.hugr
def test_hugr_captures_disjoint_array_aliases_once_through_runtime_for() -> None:
    """A TailLoop carries one root tuple instead of duplicating an element."""
    package = HugrTranspiler().to_hugr(
        _hugr_disjoint_array_alias_for,
        parameters=["repetitions"],
    )

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
def test_hugr_static_zero_trip_publishes_region_arg_initializer() -> None:
    """A zero-trip loop publishes its constant RegionArg initializer."""
    HugrTranspiler().to_hugr(
        _hugr_runtime_scalar_carry,
        bindings={"repetitions": 0},
    )


@pytest.mark.hugr
def test_hugr_rejects_while_scalar_carry_before_lowering() -> None:
    """Residual while scalar carries fail with a targeted diagnostic."""
    with pytest.raises(EmitError, match="loop-carried classical values.*while"):
        HugrTranspiler().to_hugr(_hugr_while_scalar_carry)


@pytest.mark.hugr
def test_hugr_nested_measurement_control_executes_on_selene(tmp_path) -> None:
    """The nested while/if package compiles and terminates on Selene."""
    selene = pytest.importorskip("selene_sim")
    package = HugrTranspiler().to_hugr(_hugr_measurement_while)
    runner = selene.build(
        package,
        name="qamomile_hugr_control_flow",
        build_dir=tmp_path,
    )

    results = list(
        runner.run(
            selene.Quest(),
            n_qubits=1,
            random_seed=1,
            timeout=15.0,
        )
    )

    assert results == []
