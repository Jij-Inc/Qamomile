"""Cross-backend execution tests for frontend quantum patterns.

This module exercises user-facing frontend constructs through the full
``transpile -> sample`` and ``transpile -> run`` paths on every supported
local SDK backend (Qiskit, QURI Parts, CUDA-Q).  Backend emitter tests
already validate low-level gate matrices; these tests instead pin the
combinations users can write in qkernels: native gates, qkernel calls,
broadcasts, controlled calls, composite gates, and Pauli evolution.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.segments import MultipleQuantumSegmentsError

Backend = tuple[str, Any, Any]
SampleMode = Literal["deterministic", "uniform", "bell"]


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def backend(request) -> Backend:
    """Yield ``(name, transpiler, executor)`` for each installed SDK backend."""
    name = request.param
    if name == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "quri_parts":
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        transpiler = CudaqTranspiler()
        return name, transpiler, transpiler.executor()
    raise AssertionError(f"unknown backend {name}")


def _counts(result: Any) -> dict[tuple[int, ...], int]:
    """Convert backend sample results to a bit-tuple count map.

    Args:
        result (Any): Qamomile ``SampleResult``-like object whose
            ``results`` iterable yields ``(bits, count)`` pairs.

    Returns:
        dict[tuple[int, ...], int]: Counts keyed by kernel-order bit
        tuples.
    """
    counts: dict[tuple[int, ...], int] = {}
    for bits, count in result.results:
        bit_tuple = tuple(int(bit) for bit in bits)
        counts[bit_tuple] = counts.get(bit_tuple, 0) + int(count)
    return counts


def _assert_deterministic(
    name: str, counts: dict[tuple[int, ...], int], expected: tuple[int, ...]
) -> None:
    """Assert all sampled shots equal ``expected``.

    Args:
        name (str): Backend name for assertion context.
        counts (dict[tuple[int, ...], int]): Sample counts.
        expected (tuple[int, ...]): Expected deterministic bit tuple.
    """
    assert counts, f"{name}: sampler returned no counts"
    assert set(counts) == {expected}, (
        f"{name}: got counts {counts}, expected {expected}"
    )


def _assert_uniform(
    name: str,
    counts: dict[tuple[int, ...], int],
    expected_support: set[tuple[int, ...]],
    *,
    shots: int,
) -> None:
    """Assert a small uniform distribution has the right support and balance.

    Args:
        name (str): Backend name for assertion context.
        counts (dict[tuple[int, ...], int]): Sample counts.
        expected_support (set[tuple[int, ...]]): Expected support.
        shots (int): Number of requested shots.
    """
    assert set(counts).issubset(expected_support), f"{name}: unexpected counts {counts}"
    assert set(counts) == expected_support, f"{name}: missing support in {counts}"
    expected_freq = 1.0 / len(expected_support)
    for bits in expected_support:
        freq = counts.get(bits, 0) / shots
        assert abs(freq - expected_freq) < 0.15, (
            f"{name}: {bits} frequency {freq:.3f}, expected {expected_freq:.3f}"
        )


def _assert_bell(
    name: str,
    counts: dict[tuple[int, ...], int],
    expected_support: set[tuple[int, ...]],
    *,
    shots: int,
) -> None:
    """Assert Bell-state sampling has only correlated outcomes.

    Args:
        name (str): Backend name for assertion context.
        counts (dict[tuple[int, ...], int]): Sample counts.
        expected_support (set[tuple[int, ...]]): The two correlated
            outcomes expected from the Bell-producing circuit.
        shots (int): Number of requested shots.
    """
    assert len(expected_support) == 2
    assert set(counts).issubset(expected_support), f"{name}: unexpected counts {counts}"
    assert set(counts) == expected_support, f"{name}: missing Bell support in {counts}"
    first = next(iter(expected_support))
    first_freq = counts.get(first, 0) / shots
    assert 0.35 < first_freq < 0.65, f"{name}: Bell split is imbalanced: {counts}"


@qmc.qkernel
def _flip(q: qmc.Qubit) -> qmc.Qubit:
    """Flip a qubit for qkernel-call and controlled-call tests."""
    return qmc.x(q)


@qmc.qkernel
def _hadamard_broadcast(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply a broadcast Hadamard through a helper qkernel."""
    return qmc.h(q)


@qmc.qkernel
def _h_then_full_reslice(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Mutate a view argument and return a full re-slice of it."""
    q = qmc.h(q)
    return q[:]


@qmc.qkernel
def _rx_with_angle(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply RX(theta) through a helper qkernel."""
    return qmc.rx(q, angle=theta)


@qmc.qkernel
def _phase_gate_for_view_qpe(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    """Apply a phase gate for QPE controlled-U regression tests."""
    return qmc.p(q, theta)


@qmc.qkernel
def _qft_layer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply a shape-dependent stdlib composite inside a subkernel."""
    q = qmc.qft(q)
    return q


@qmc.qkernel
def _pauli_x_evolve(
    q: qmc.Vector[qmc.Qubit],
    ham: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a Pauli evolution layer for controlled-evolution tests."""
    return qmc.pauli_evolve(q, ham, gamma)


@qmc.composite_gate(name="bell_pair")
@qmc.qkernel
def _bell_pair(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Create a Bell pair as a custom composite gate."""
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


@qmc.composite_gate(name="ry_composite")
@qmc.qkernel
def _ry_composite(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply ``RY`` as a parameterized custom composite gate."""
    return qmc.ry(q, theta)


@qmc.qkernel
def native_gate_sample() -> qmc.Vector[qmc.Bit]:
    """Exercise every native gate in one deterministic sample kernel."""
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.x(q[0])
    q[1] = qmc.y(q[1])
    q[1] = qmc.z(q[1])
    q[1] = qmc.s(q[1])
    q[1] = qmc.sdg(q[1])
    q[1] = qmc.t(q[1])
    q[1] = qmc.tdg(q[1])
    q[2] = qmc.h(q[2])
    q[2] = qmc.h(q[2])
    q[2] = qmc.rx(q[2], math.pi)
    q[2] = qmc.ry(q[2], math.pi)
    q[2] = qmc.rz(q[2], math.pi / 7)
    q[2] = qmc.p(q[2], math.pi / 5)
    q[0], q[2] = qmc.cx(q[0], q[2])
    q[0], q[1] = qmc.cz(q[0], q[1])
    q[0], q[3] = qmc.swap(q[0], q[3])
    q[1], q[2], q[0] = qmc.ccx(q[1], q[2], q[0])
    q[1], q[2] = qmc.cp(q[1], q[2], math.pi / 3)
    q[2], q[3] = qmc.rzz(q[2], q[3], math.pi / 9)
    return qmc.measure(q)


@qmc.qkernel
def native_gate_run(obs: qmc.Observable) -> qmc.Float:
    """Exercise every native gate and return a deterministic expval."""
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.x(q[0])
    q[1] = qmc.y(q[1])
    q[1] = qmc.z(q[1])
    q[1] = qmc.s(q[1])
    q[1] = qmc.sdg(q[1])
    q[1] = qmc.t(q[1])
    q[1] = qmc.tdg(q[1])
    q[2] = qmc.h(q[2])
    q[2] = qmc.h(q[2])
    q[2] = qmc.rx(q[2], math.pi)
    q[2] = qmc.ry(q[2], math.pi)
    q[2] = qmc.rz(q[2], math.pi / 7)
    q[2] = qmc.p(q[2], math.pi / 5)
    q[0], q[2] = qmc.cx(q[0], q[2])
    q[0], q[1] = qmc.cz(q[0], q[1])
    q[0], q[3] = qmc.swap(q[0], q[3])
    q[1], q[2], q[0] = qmc.ccx(q[1], q[2], q[0])
    q[1], q[2] = qmc.cp(q[1], q[2], math.pi / 3)
    q[2], q[3] = qmc.rzz(q[2], q[3], math.pi / 9)
    return qmc.expval(q, obs)


@qmc.qkernel
def qkernel_broadcast_sample() -> qmc.Vector[qmc.Bit]:
    """Call a helper qkernel that broadcasts over a vector."""
    q = qmc.qubit_array(2, "q")
    q = _hadamard_broadcast(q)
    return qmc.measure(q)


@qmc.qkernel
def qkernel_broadcast_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after a helper-qkernel broadcast."""
    q = qmc.qubit_array(2, "q")
    q = _hadamard_broadcast(q)
    return qmc.expval(q, obs)


@qmc.qkernel
def vector_view_full_reslice_sample() -> qmc.Vector[qmc.Bit]:
    """Sample a sub-kernel that returns a full re-slice of a VectorView."""
    q = qmc.qubit_array(4, "q")
    evens = q[0:4:2]
    evens = _h_then_full_reslice(evens)
    q[0:4:2] = evens
    return qmc.measure(q)


@qmc.qkernel
def vector_view_full_reslice_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after a full-re-slice VectorView sub-kernel return."""
    q = qmc.qubit_array(4, "q")
    evens = q[0:4:2]
    evens = _h_then_full_reslice(evens)
    q[0:4:2] = evens
    return qmc.expval(q, obs)


@qmc.qkernel
def vector_view_value_element_sample(
    values: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Sample after passing ``values[1:3][0]`` to a helper qkernel."""
    q = qmc.qubit_array(1, "q")
    view = values[1:3]
    q[0] = _rx_with_angle(q[0], view[0])
    return qmc.measure(q)


@qmc.qkernel
def vector_view_value_element_run(
    values: qmc.Vector[qmc.Float],
    obs: qmc.Observable,
) -> qmc.Float:
    """Run expval after passing ``values[1:3][0]`` to a helper qkernel."""
    q = qmc.qubit_array(1, "q")
    view = values[1:3]
    q[0] = _rx_with_angle(q[0], view[0])
    return qmc.expval(q, obs)


@qmc.qkernel
def vector_view_qpe_parameter_sample(
    gammas: qmc.Vector[qmc.Float],
) -> qmc.Float:
    """Sample QPE after binding controlled-U theta from a VectorView element."""
    counting = qmc.qubit_array(3, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)
    view = gammas[1:3]
    phase = qmc.qpe(target, counting, _phase_gate_for_view_qpe, theta=view[0])
    return qmc.measure(phase)


@qmc.qkernel
def controlled_qkernel_sample() -> qmc.Vector[qmc.Bit]:
    """Apply a controlled qkernel call with a deterministic output."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_flip = qmc.control(_flip)
    q[0], q[1] = controlled_flip(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def controlled_qkernel_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after a controlled qkernel call."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_flip = qmc.control(_flip)
    q[0], q[1] = controlled_flip(q[0], q[1])
    return qmc.expval(q, obs)


def _apply_controlled_native_gate_suite(
    q: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply every native gate through ``qmc.control`` with fixed output."""
    q[0] = qmc.x(q[0])

    ch = qmc.control(qmc.h)
    q[0], q[1] = ch(q[0], q[1])
    q[0], q[1] = ch(q[0], q[1])

    cx = qmc.control(qmc.x)
    q[0], q[1] = cx(q[0], q[1])

    cy = qmc.control(qmc.y)
    q[0], q[2] = cy(q[0], q[2])

    crx = qmc.control(qmc.rx)
    q[0], q[3] = crx(q[0], q[3], math.pi)

    cry = qmc.control(qmc.ry)
    q[0], q[4] = cry(q[0], q[4], math.pi)

    for gate in (qmc.z, qmc.s, qmc.sdg, qmc.t, qmc.tdg):
        controlled_gate = qmc.control(gate)
        q[0], q[1] = controlled_gate(q[0], q[1])

    cp1 = qmc.control(qmc.p)
    q[0], q[1] = cp1(q[0], q[1], math.pi / 5)

    crz = qmc.control(qmc.rz)
    q[0], q[1] = crz(q[0], q[1], math.pi / 7)

    ccx = qmc.control(qmc.cx)
    q[0], q[1], q[5] = ccx(q[0], q[1], q[5])

    ccz = qmc.control(qmc.cz)
    q[0], q[1], q[5] = ccz(q[0], q[1], q[5])

    cswap = qmc.control(qmc.swap)
    q[0], q[4], q[5] = cswap(q[0], q[4], q[5])

    cccx = qmc.control(qmc.ccx)
    q[0], q[1], q[2], q[3] = cccx(q[0], q[1], q[2], q[3])

    ccp = qmc.control(qmc.cp)
    q[0], q[1], q[2] = ccp(q[0], q[1], q[2], math.pi / 3)

    crzz = qmc.control(qmc.rzz)
    q[0], q[4], q[5] = crzz(q[0], q[4], q[5], math.pi / 9)
    return q


@qmc.qkernel
def controlled_native_gate_sample() -> qmc.Vector[qmc.Bit]:
    """Sample all native gates wrapped by ``qmc.control``."""
    q = qmc.qubit_array(6, "q")
    q = _apply_controlled_native_gate_suite(q)
    return qmc.measure(q)


@qmc.qkernel
def controlled_native_gate_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after all native gates wrapped by ``qmc.control``."""
    q = qmc.qubit_array(6, "q")
    q = _apply_controlled_native_gate_suite(q)
    return qmc.expval(q, obs)


@qmc.qkernel
def controlled_power_sample() -> qmc.Vector[qmc.Bit]:
    """Sample a powered controlled native gate with deterministic output."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_x = qmc.control(qmc.x)
    q[0], q[1] = controlled_x(q[0], q[1], power=2)
    return qmc.measure(q)


@qmc.qkernel
def controlled_power_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after a powered controlled native gate."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_x = qmc.control(qmc.x)
    q[0], q[1] = controlled_x(q[0], q[1], power=2)
    return qmc.expval(q, obs)


@qmc.qkernel
def controlled_native_broadcast_target_sample() -> qmc.Vector[qmc.Bit]:
    """Sample a built-in controlled gate broadcast over a VectorView target."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    controlled_x = qmc.control(qmc.x)
    q[0], targets = controlled_x(q[0], q[1:3])
    q[1:3] = targets
    return qmc.measure(q)


@qmc.qkernel
def controlled_native_broadcast_target_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after a controlled native broadcast target."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    controlled_x = qmc.control(qmc.x)
    q[0], targets = controlled_x(q[0], q[1:3])
    q[1:3] = targets
    return qmc.expval(q, obs)


@qmc.qkernel
def controlled_qkernel_broadcast_sample() -> qmc.Vector[qmc.Bit]:
    """Sample a controlled qkernel that broadcasts over a VectorView."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    controlled_broadcast = qmc.control(_hadamard_broadcast)
    q[0], targets = controlled_broadcast(q[0], q[1:3])
    q[1:3] = targets
    return qmc.measure(q)


@qmc.qkernel
def controlled_qkernel_broadcast_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after a controlled qkernel broadcast."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    controlled_broadcast = qmc.control(_hadamard_broadcast)
    q[0], targets = controlled_broadcast(q[0], q[1:3])
    q[1:3] = targets
    return qmc.expval(q, obs)


@qmc.qkernel
def controlled_stdlib_composite_sample() -> qmc.Vector[qmc.Bit]:
    """Sample controlled qkernel-backed stdlib QFT use."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    controlled_qft = qmc.control(_qft_layer)
    q[0], targets = controlled_qft(q[0], q[1:3])
    q[1:3] = targets
    return qmc.measure(q)


@qmc.qkernel
def controlled_stdlib_composite_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after controlled stdlib QFT use."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    controlled_qft = qmc.control(_qft_layer)
    q[0], targets = controlled_qft(q[0], q[1:3])
    q[1:3] = targets
    return qmc.expval(q, obs)


@qmc.qkernel
def controlled_composite_sample() -> qmc.Vector[qmc.Bit]:
    """Sample a custom composite gate wrapped by ``qmc.control``."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    controlled_bell = qmc.control(_bell_pair)
    q[0], q[1], q[2] = controlled_bell(q[0], q[1], q[2])
    return qmc.measure(q)


@qmc.qkernel
def controlled_composite_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after a controlled custom composite gate."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    controlled_bell = qmc.control(_bell_pair)
    q[0], q[1], q[2] = controlled_bell(q[0], q[1], q[2])
    return qmc.expval(q, obs)


@qmc.qkernel
def controlled_parameterized_composite_sample(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a parameterized custom composite wrapped by ``qmc.control``."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_ry = qmc.control(_ry_composite)
    q[0], q[1] = controlled_ry(q[0], q[1], theta)
    return qmc.measure(q)


@qmc.qkernel
def controlled_parameterized_composite_run(
    theta: qmc.Float,
    obs: qmc.Observable,
) -> qmc.Float:
    """Run expval after a controlled parameterized custom composite."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_ry = qmc.control(_ry_composite)
    q[0], q[1] = controlled_ry(q[0], q[1], theta)
    return qmc.expval(q, obs)


@qmc.qkernel
def vector_view_control_sample() -> qmc.Vector[qmc.Bit]:
    """Sample a native controlled gate with a VectorView control prefix."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[1] = qmc.x(q[1])
    controlled_h = qmc.control(qmc.h, num_controls=2)
    controls, q[2] = controlled_h(q[0:2], q[2])
    q[0:2] = controls
    return qmc.measure(q)


@qmc.qkernel
def vector_view_control_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after a VectorView-control native gate."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[1] = qmc.x(q[1])
    controlled_h = qmc.control(qmc.h, num_controls=2)
    controls, q[2] = controlled_h(q[0:2], q[2])
    q[0:2] = controls
    return qmc.expval(q, obs)


@qmc.qkernel
def symbolic_control_indices_sample(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Sample symbolic-control pool selection with bound indices."""
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.x(q[0])
    q[2] = qmc.x(q[2])
    controlled_x = qmc.control(qmc.x, num_controls=n)
    q[0:3], q[3] = controlled_x(q[0:3], q[3], control_indices=[0, 2])
    return qmc.measure(q)


@qmc.qkernel
def symbolic_control_indices_run(n: qmc.UInt, obs: qmc.Observable) -> qmc.Float:
    """Run expval after symbolic-control pool selection."""
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.x(q[0])
    q[2] = qmc.x(q[2])
    controlled_x = qmc.control(qmc.x, num_controls=n)
    q[0:3], q[3] = controlled_x(q[0:3], q[3], control_indices=[0, 2])
    return qmc.expval(q, obs)


@qmc.qkernel
def bound_control_indices_sample(
    n: qmc.UInt,
    i: qmc.UInt,
    j: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Sample bound ``UInt`` control-index entries."""
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.x(q[0])
    q[2] = qmc.x(q[2])
    controlled_x = qmc.control(qmc.x, num_controls=n)
    q[0:3], q[3] = controlled_x(q[0:3], q[3], control_indices=[i, j])
    return qmc.measure(q)


@qmc.qkernel
def bound_control_indices_run(
    n: qmc.UInt,
    i: qmc.UInt,
    j: qmc.UInt,
    obs: qmc.Observable,
) -> qmc.Float:
    """Run expval after bound ``UInt`` control-index entries."""
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.x(q[0])
    q[2] = qmc.x(q[2])
    controlled_x = qmc.control(qmc.x, num_controls=n)
    q[0:3], q[3] = controlled_x(q[0:3], q[3], control_indices=[i, j])
    return qmc.expval(q, obs)


@qmc.qkernel
def controlled_random_ry_sample(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a randomized-angle controlled ``RY`` gate."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_ry = qmc.control(qmc.ry)
    q[0], q[1] = controlled_ry(q[0], q[1], theta)
    return qmc.measure(q)


@qmc.qkernel
def controlled_random_ry_run(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Run expval after a randomized-angle controlled ``RY`` gate."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_ry = qmc.control(qmc.ry)
    q[0], q[1] = controlled_ry(q[0], q[1], theta)
    return qmc.expval(q, obs)


@qmc.qkernel
def controlled_random_ry_power_sample(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a randomized-angle powered controlled ``RY`` gate."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_ry = qmc.control(qmc.ry)
    q[0], q[1] = controlled_ry(q[0], q[1], theta, power=2)
    return qmc.measure(q)


@qmc.qkernel
def controlled_random_ry_power_run(
    theta: qmc.Float,
    obs: qmc.Observable,
) -> qmc.Float:
    """Run expval after a randomized-angle powered controlled ``RY`` gate."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_ry = qmc.control(qmc.ry)
    q[0], q[1] = controlled_ry(q[0], q[1], theta, power=2)
    return qmc.expval(q, obs)


@qmc.qkernel
def controlled_pauli_evolve_sample(
    ham: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Sample a qkernel-backed Pauli evolution wrapped by ``qmc.control``."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_evolve = qmc.control(_pauli_x_evolve)
    q[0], target = controlled_evolve(q[0], q[1:2], ham=ham, gamma=gamma)
    q[1:2] = target
    return qmc.measure(q)


@qmc.qkernel
def controlled_pauli_evolve_run(
    ham: qmc.Observable,
    gamma: qmc.Float,
    obs: qmc.Observable,
) -> qmc.Float:
    """Run expval after controlled qkernel-backed Pauli evolution."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    controlled_evolve = qmc.control(_pauli_x_evolve)
    q[0], target = controlled_evolve(q[0], q[1:2], ham=ham, gamma=gamma)
    q[1:2] = target
    return qmc.expval(q, obs)


@qmc.qkernel
def composite_gate_sample() -> qmc.Vector[qmc.Bit]:
    """Sample a custom composite Bell-pair gate."""
    q = qmc.qubit_array(2, "q")
    q[0], q[1] = _bell_pair(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def composite_gate_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after a custom composite Bell-pair gate."""
    q = qmc.qubit_array(2, "q")
    q[0], q[1] = _bell_pair(q[0], q[1])
    return qmc.expval(q, obs)


@qmc.qkernel
def stdlib_composite_sample() -> qmc.Vector[qmc.Bit]:
    """Apply QFT then IQFT and sample the restored basis state."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[2] = qmc.x(q[2])
    q = qmc.qft(q)
    q = qmc.iqft(q)
    return qmc.measure(q)


@qmc.qkernel
def stdlib_composite_run(obs: qmc.Observable) -> qmc.Float:
    """Apply QFT then IQFT and run a restored-basis expval."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[2] = qmc.x(q[2])
    q = qmc.qft(q)
    q = qmc.iqft(q)
    return qmc.expval(q, obs)


@qmc.qkernel
def pauli_evolve_sample(ham: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a Pauli evolution frontend operation at identity angle."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q = qmc.pauli_evolve(q, ham, gamma)
    return qmc.measure(q)


@qmc.qkernel
def pauli_evolve_run(
    ham: qmc.Observable, gamma: qmc.Float, obs: qmc.Observable
) -> qmc.Float:
    """Run expval after a Pauli evolution frontend operation."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q = qmc.pauli_evolve(q, ham, gamma)
    return qmc.expval(q, obs)


@qmc.qkernel
def padded_pauli_evolve_sample(
    ham: qmc.Observable, gamma: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Sample evolution under a Hamiltonian narrower than the register.

    The register has 2 qubits but ``ham`` (e.g. ``Z(0)``) acts on only
    1 qubit, so it must be identity-padded onto the untouched qubit
    rather than rejected.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q = qmc.pauli_evolve(q, ham, gamma)
    return qmc.measure(q)


@qmc.qkernel
def padded_pauli_evolve_run(
    ham: qmc.Observable, gamma: qmc.Float, obs: qmc.Observable
) -> qmc.Float:
    """Run expval after evolution under a register-narrower Hamiltonian."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q = qmc.pauli_evolve(q, ham, gamma)
    return qmc.expval(q, obs)


@qmc.qkernel
def sliced_pauli_evolve_sample(
    ham: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Sample Pauli evolution on a non-contiguous VectorView."""
    q = qmc.qubit_array(4, "q")
    evens = q[0:4:2]
    evens = qmc.pauli_evolve(evens, ham, gamma)
    q[0:4:2] = evens
    return qmc.measure(q)


@qmc.qkernel
def sliced_pauli_evolve_run(
    ham: qmc.Observable,
    gamma: qmc.Float,
    obs: qmc.Observable,
) -> qmc.Float:
    """Run expval after Pauli evolution on a non-contiguous VectorView."""
    q = qmc.qubit_array(4, "q")
    evens = q[0:4:2]
    evens = qmc.pauli_evolve(evens, ham, gamma)
    q[0:4:2] = evens
    return qmc.expval(q, obs)


# --- Regression: parameter-expression gate angle must not split segments ---
#
# A gate angle that is an *expression* over a runtime parameter (e.g.
# ``theta=-phase``) lowers to a classical ``BinOp`` that constant folding
# cannot collapse while ``phase`` stays symbolic. When such an op is
# interleaved between quantum operations the segmentation pass used to flush
# the quantum region and start a new one, producing a spurious
# ``MultipleQuantumSegmentsError`` even though the kernel has no
# measurement-dependent control flow. These kernels reproduce that shape
# (a quantum op precedes the negated-angle gate so the ``BinOp`` is genuinely
# interleaved) and check both sample and expval paths across backends.


@qmc.qkernel
def interleaved_param_expr_sample(phase: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample ``rx(q, -phase)`` interleaved after a quantum op."""
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.x(q[0])
    q[0] = qmc.rx(q[0], -phase)
    return qmc.measure(q)


@qmc.qkernel
def interleaved_param_expr_run(phase: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Run expval of ``rx(q, -phase)`` interleaved after a quantum op."""
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.x(q[0])
    q[0] = qmc.rx(q[0], -phase)
    return qmc.expval(q, obs)


# --- Regression: parameter-expression angle computed before quantum init ---
#
# ``angle = -phase`` is computed before ``qmc.qubit_array`` (before any quantum
# op). Its ``BinOp`` lands before the quantum segment, so a position-naive
# absorption (only firing while already inside the quantum segment) would leave
# it stranded in a classical prep segment, where the backend has no gate to bind
# the parameter expression to and silently emits a zero angle. The segmentation
# holds such a leading parameter-expression op and prepends it to the quantum
# segment, so the gate gets the real angle regardless of where it was written.


@qmc.qkernel
def pre_init_param_expr_sample(phase: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample ``rx(q, -phase)`` whose angle is computed before qubit_array."""
    angle = -phase
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.x(q[0])
    q[0] = qmc.rx(q[0], angle)
    return qmc.measure(q)


@qmc.qkernel
def pre_init_param_expr_run(phase: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Run expval of ``rx(q, -phase)`` whose angle is computed before init."""
    angle = -phase
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.x(q[0])
    q[0] = qmc.rx(q[0], angle)
    return qmc.expval(q, obs)


# --- Regression: reported symbolic multi-control phase inside a loop ---
#
# The originally reported case (a QSVT / block-encoding circuit): a symbolic
# multi-control ``qmc.control(qmc.p, num_controls=<symbolic>)`` driven by a
# negated runtime angle inside a ``qmc.range`` loop. With ``num_controls`` =
# ``n - 1`` = 1 here, the operation is equivalent to a single concrete
# ``qmc.cp``; the test asserts the two transpile and execute to the same
# expectation value, which only holds once the spurious segment split is
# fixed.


@qmc.qkernel
def symbolic_mc_phase_run(phase: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Run expval of a symbolic multi-control phase with a negated angle."""
    q = qmc.qubit_array(2, "q")
    for i in qmc.range(2):
        q[i] = qmc.h(q[i])
    num_signal = qmc.uint(2)
    cphase = qmc.control(qmc.p, num_controls=num_signal - 1)
    q[0:1], q[1] = cphase(q[0:1], q[1], theta=-phase)
    return qmc.expval(q, obs)


@qmc.qkernel
def cp_phase_run(phase: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Concrete ``qmc.cp`` equivalent of :func:`symbolic_mc_phase_run`."""
    q = qmc.qubit_array(2, "q")
    for i in qmc.range(2):
        q[i] = qmc.h(q[i])
    q[0], q[1] = qmc.cp(q[0], q[1], -phase)
    return qmc.expval(q, obs)


# --- Regression: a classical output computed after a quantum op is preserved ---
#
# The segment carve-out above only absorbs a non-measurement classical op into
# the quantum segment when its result feeds a quantum gate. A classical
# parameter expression that is instead a *block output* (and feeds no gate)
# must remain a real classical segment so the executor runs it and the
# orchestrator surfaces its value — absorbing it would silently return ``None``.


@qmc.qkernel
def classical_output_after_quantum_run(phase: qmc.Float) -> qmc.Float:
    """Return a parameter expression computed after an unrelated quantum op."""
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.x(q[0])
    return phase * 2.0


# --- Regression: a value feeding both a gate and later classical work ---
#
# ``t = phase * 2`` here feeds a quantum gate (``rx``) *and* a later classical
# computation (``t + 1``, a block output). Absorbing ``t`` into the quantum
# segment would let the later classical segment read a value that never executed
# (the quantum segment runs no classical ops), silently miscompiling. The
# absorbable-set fixpoint refuses to absorb ``t``, so the kernel raises an
# explicit ``MultipleQuantumSegmentsError`` instead of producing a wrong result.


@qmc.qkernel
def value_feeding_gate_and_classical_run(
    phase: qmc.Float,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Float]:
    """Use a parameter expression as both a gate angle and a classical output."""
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.x(q[0])
    angle = phase * 2.0
    q[0] = qmc.rx(q[0], angle)
    return qmc.measure(q), angle + 1.0


@qmc.qkernel
def pre_init_gate_and_classical_run(
    phase: qmc.Float, obs: qmc.Observable
) -> tuple[qmc.Float, qmc.Float]:
    """Use a value as a gate angle and a classical output, computed before init.

    Same dual-use as ``value_feeding_gate_and_classical_run`` but the value is
    built before ``qmc.qubit_array``, so it lands before the quantum segment.
    That produces a legal C->Q->Expval plan (no quantum-segment split), so the
    stranded gate parameter must be caught by an explicit check rather than the
    multiple-segments error.
    """
    angle = phase * 2.0
    out = angle + 1.0
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.rx(q[0], angle)
    return qmc.expval(q, obs), out


# --- Regression: a classical chain split across a control-flow boundary ---
#
# ``base = phase * 2`` is a top-level classical op, but the next link of the
# chain (``angle = base + 1``) lives inside the ``qmc.range`` loop body. The
# absorbable-set analysis seeds nested classical ops as candidates too, so the
# nested ``angle = base + 1`` is itself absorbable (it feeds only the gate), and
# ``base`` — feeding it — stays absorbable rather than being treated as feeding
# a classical sink. Without nested candidates the kernel spuriously raised
# ``MultipleQuantumSegmentsError``. The loop index is used so the loop unrolls
# and the kernel actually executes; both qubits end up in state
# ``rx(2 * phase + 1) |0>``.


@qmc.qkernel
def nested_classical_chain_sample(phase: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a nested classical chain (`base` top-level, `angle` in a loop)."""
    q = qmc.qubit_array(2, "q")
    base = phase * 2.0
    for i in qmc.range(2):
        angle = base + 1.0
        q[i] = qmc.rx(q[i], angle)
    return qmc.measure(q)


@qmc.qkernel
def nested_classical_chain_run(phase: qmc.Float, obs: qmc.Observable) -> qmc.Float:
    """Run expval of a nested classical chain split across a loop boundary."""
    q = qmc.qubit_array(2, "q")
    base = phase * 2.0
    for i in qmc.range(2):
        angle = base + 1.0
        q[i] = qmc.rx(q[i], angle)
    return qmc.expval(q, obs)


# --- Regression: a nested classical op whose result is a block output ---
#
# ``base = phase * 2`` feeds a gate (``rx``) and a nested ``out = base + 1``
# whose result is returned. ``out`` runs inside the loop body, so it cannot be
# surfaced as a block output from the quantum segment. The absorbable-set
# analysis must therefore refuse to absorb ``base`` (the nested ``out`` is a
# classical sink because it is a block output), so the kernel raises an explicit
# ``MultipleQuantumSegmentsError`` instead of silently returning ``None`` for the
# ``out`` field.


@qmc.qkernel
def nested_classical_output_run(
    phase: qmc.Float,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Float]:
    """Return a nested classical value that also feeds a gate angle."""
    q = qmc.qubit_array(1, "q")
    base = phase * 2.0
    for i in qmc.range(1):
        q[0] = qmc.rx(q[0], base)
        out = base + 1.0
    return qmc.measure(q), out


# --- Modulo operator (UInt.__mod__) on alternating loop indices ---
#
# Motivating use case for ``UInt.__mod__`` (backlog BACKLOG-8409): a QSVT-style
# loop that selects between a forward and inverse step on alternating iterations
# via ``i % 2``. ``i % 2`` folds to a concrete value once the loop is unrolled.
#
# The sample kernel uses the natural ``if i % 2 == 0`` form (compile-time
# branch resolved per iteration) to flip the even-indexed qubits, giving the
# deterministic pattern ``(1, 0, 1, 0)``. The run/expval kernel instead folds
# ``i % 2`` into an RX angle (``(i % 2) * pi``) rather than an ``if``: CUDA-Q
# marks any ``if``-containing kernel as RUNNABLE, and ``cudaq.observe()`` (the
# expval path) rejects RUNNABLE artifacts, so the angle form keeps the kernel
# STATIC and exercises ``%`` through the estimator on every backend. RX(pi)
# flips the odd-indexed qubits, so Z0+Z1+Z2+Z3 sums to +1-1+1-1 = 0.


@qmc.qkernel
def uint_mod_alternating_sample() -> qmc.Vector[qmc.Bit]:
    """Flip even-indexed qubits using ``i % 2 == 0`` in an unrolled loop."""
    q = qmc.qubit_array(4, "q")
    for i in qmc.range(4):
        if i % 2 == 0:
            q[i] = qmc.x(q[i])
    return qmc.measure(q)


@qmc.qkernel
def uint_mod_alternating_run(obs: qmc.Observable) -> qmc.Float:
    """Run expval after RX((i % 2) * pi), flipping odd-indexed qubits."""
    q = qmc.qubit_array(4, "q")
    for i in qmc.range(4):
        q[i] = qmc.rx(q[i], (i % 2) * math.pi)
    return qmc.expval(q, obs)


@dataclasses.dataclass(frozen=True)
class FrontendExecutionCase:
    """Describe one frontend pattern with paired sample and run kernels."""

    name: str
    sample_kernel: qmc.QKernel
    run_kernel: qmc.QKernel
    sample_mode: SampleMode
    expected_bits: tuple[int, ...] | None
    expected_support: set[tuple[int, ...]]
    expected_expval: float
    sample_bindings: dict[str, Any] = dataclasses.field(default_factory=dict)
    run_bindings: dict[str, Any] = dataclasses.field(default_factory=dict)
    unsupported_backends: frozenset[str] = dataclasses.field(default_factory=frozenset)


QURI_PARTS_CONTROLLED_FALLBACK_UNSUPPORTED = frozenset({"quri_parts"})


FRONTEND_EXECUTION_CASES = [
    FrontendExecutionCase(
        name="native-gates",
        sample_kernel=native_gate_sample,
        run_kernel=native_gate_run,
        sample_mode="deterministic",
        expected_bits=(1, 1, 1, 1),
        expected_support={(1, 1, 1, 1)},
        expected_expval=-4.0,
        run_bindings={"obs": qm_o.Z(0) + qm_o.Z(1) + qm_o.Z(2) + qm_o.Z(3)},
    ),
    FrontendExecutionCase(
        name="qkernel-broadcast",
        sample_kernel=qkernel_broadcast_sample,
        run_kernel=qkernel_broadcast_run,
        sample_mode="uniform",
        expected_bits=None,
        expected_support={(0, 0), (1, 0), (0, 1), (1, 1)},
        expected_expval=0.0,
        run_bindings={"obs": qm_o.Z(0) + qm_o.Z(1)},
    ),
    FrontendExecutionCase(
        name="vector-view-full-reslice",
        sample_kernel=vector_view_full_reslice_sample,
        run_kernel=vector_view_full_reslice_run,
        sample_mode="uniform",
        expected_bits=None,
        expected_support={(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (1, 0, 1, 0)},
        expected_expval=2.0,
        run_bindings={
            "obs": qm_o.Z(0) + qm_o.Z(1) + qm_o.Z(2) + qm_o.Z(3),
        },
    ),
    FrontendExecutionCase(
        name="vector-view-value-element-helper",
        sample_kernel=vector_view_value_element_sample,
        run_kernel=vector_view_value_element_run,
        sample_mode="deterministic",
        expected_bits=(1,),
        expected_support={(1,)},
        expected_expval=-1.0,
        sample_bindings={"values": np.array([0.0, math.pi, 0.0])},
        run_bindings={
            "values": np.array([0.0, math.pi, 0.0]),
            "obs": qm_o.Z(0),
        },
    ),
    FrontendExecutionCase(
        name="controlled-qkernel",
        sample_kernel=controlled_qkernel_sample,
        run_kernel=controlled_qkernel_run,
        sample_mode="deterministic",
        expected_bits=(1, 1),
        expected_support={(1, 1)},
        expected_expval=-2.0,
        run_bindings={"obs": qm_o.Z(0) + qm_o.Z(1)},
    ),
    FrontendExecutionCase(
        name="controlled-native-gates",
        sample_kernel=controlled_native_gate_sample,
        run_kernel=controlled_native_gate_run,
        sample_mode="deterministic",
        expected_bits=(1, 1, 1, 0, 1, 1),
        expected_support={(1, 1, 1, 0, 1, 1)},
        expected_expval=-4.0,
        run_bindings={
            "obs": qm_o.Z(0) + qm_o.Z(1) + qm_o.Z(2) + qm_o.Z(3) + qm_o.Z(4) + qm_o.Z(5)
        },
    ),
    FrontendExecutionCase(
        name="controlled-power",
        sample_kernel=controlled_power_sample,
        run_kernel=controlled_power_run,
        sample_mode="deterministic",
        expected_bits=(1, 0),
        expected_support={(1, 0)},
        expected_expval=1.0,
        run_bindings={"obs": qm_o.Z(1)},
    ),
    FrontendExecutionCase(
        name="controlled-native-broadcast-target",
        sample_kernel=controlled_native_broadcast_target_sample,
        run_kernel=controlled_native_broadcast_target_run,
        sample_mode="deterministic",
        expected_bits=(1, 1, 1),
        expected_support={(1, 1, 1)},
        expected_expval=-3.0,
        run_bindings={"obs": qm_o.Z(0) + qm_o.Z(1) + qm_o.Z(2)},
    ),
    FrontendExecutionCase(
        name="controlled-qkernel-broadcast",
        sample_kernel=controlled_qkernel_broadcast_sample,
        run_kernel=controlled_qkernel_broadcast_run,
        sample_mode="uniform",
        expected_bits=None,
        expected_support={(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)},
        expected_expval=0.0,
        run_bindings={"obs": qm_o.Z(1) + qm_o.Z(2)},
    ),
    FrontendExecutionCase(
        name="controlled-stdlib-composite",
        sample_kernel=controlled_stdlib_composite_sample,
        run_kernel=controlled_stdlib_composite_run,
        sample_mode="uniform",
        expected_bits=None,
        expected_support={(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)},
        expected_expval=0.0,
        run_bindings={"obs": qm_o.Z(1) + qm_o.Z(2)},
    ),
    FrontendExecutionCase(
        name="controlled-custom-composite",
        sample_kernel=controlled_composite_sample,
        run_kernel=controlled_composite_run,
        sample_mode="bell",
        expected_bits=None,
        expected_support={(1, 0, 0), (1, 1, 1)},
        expected_expval=1.0,
        run_bindings={"obs": qm_o.Z(1) * qm_o.Z(2)},
    ),
    FrontendExecutionCase(
        name="controlled-parameterized-composite",
        sample_kernel=controlled_parameterized_composite_sample,
        run_kernel=controlled_parameterized_composite_run,
        sample_mode="deterministic",
        expected_bits=(1, 1),
        expected_support={(1, 1)},
        expected_expval=-1.0,
        sample_bindings={"theta": math.pi},
        run_bindings={"theta": math.pi, "obs": qm_o.Z(1)},
    ),
    FrontendExecutionCase(
        name="vector-view-control",
        sample_kernel=vector_view_control_sample,
        run_kernel=vector_view_control_run,
        sample_mode="uniform",
        expected_bits=None,
        expected_support={(1, 1, 0), (1, 1, 1)},
        expected_expval=0.0,
        run_bindings={"obs": qm_o.Z(2)},
        unsupported_backends=QURI_PARTS_CONTROLLED_FALLBACK_UNSUPPORTED,
    ),
    FrontendExecutionCase(
        name="symbolic-control-indices",
        sample_kernel=symbolic_control_indices_sample,
        run_kernel=symbolic_control_indices_run,
        sample_mode="deterministic",
        expected_bits=(1, 0, 1, 1),
        expected_support={(1, 0, 1, 1)},
        expected_expval=-2.0,
        sample_bindings={"n": 2},
        run_bindings={"n": 2, "obs": qm_o.Z(0) + qm_o.Z(1) + qm_o.Z(2) + qm_o.Z(3)},
    ),
    FrontendExecutionCase(
        name="bound-control-indices",
        sample_kernel=bound_control_indices_sample,
        run_kernel=bound_control_indices_run,
        sample_mode="deterministic",
        expected_bits=(1, 0, 1, 1),
        expected_support={(1, 0, 1, 1)},
        expected_expval=-2.0,
        sample_bindings={"n": 2, "i": 0, "j": 2},
        run_bindings={
            "n": 2,
            "i": 0,
            "j": 2,
            "obs": qm_o.Z(0) + qm_o.Z(1) + qm_o.Z(2) + qm_o.Z(3),
        },
    ),
    FrontendExecutionCase(
        name="controlled-pauli-evolve",
        sample_kernel=controlled_pauli_evolve_sample,
        run_kernel=controlled_pauli_evolve_run,
        sample_mode="deterministic",
        expected_bits=(1, 1),
        expected_support={(1, 1)},
        expected_expval=-1.0,
        sample_bindings={"ham": qm_o.X(0), "gamma": math.pi / 2},
        run_bindings={
            "ham": qm_o.X(0),
            "gamma": math.pi / 2,
            "obs": qm_o.Z(1),
        },
        unsupported_backends=QURI_PARTS_CONTROLLED_FALLBACK_UNSUPPORTED,
    ),
    FrontendExecutionCase(
        name="sliced-pauli-evolve",
        sample_kernel=sliced_pauli_evolve_sample,
        run_kernel=sliced_pauli_evolve_run,
        sample_mode="deterministic",
        expected_bits=(1, 0, 1, 0),
        expected_support={(1, 0, 1, 0)},
        expected_expval=0.0,
        sample_bindings={"ham": qm_o.X(0) * qm_o.X(1), "gamma": math.pi / 2},
        run_bindings={
            "ham": qm_o.X(0) * qm_o.X(1),
            "gamma": math.pi / 2,
            "obs": qm_o.Z(0) + qm_o.Z(1) + qm_o.Z(2) + qm_o.Z(3),
        },
    ),
    FrontendExecutionCase(
        name="custom-composite",
        sample_kernel=composite_gate_sample,
        run_kernel=composite_gate_run,
        sample_mode="bell",
        expected_bits=None,
        expected_support={(0, 0), (1, 1)},
        expected_expval=1.0,
        run_bindings={"obs": qm_o.Z(0) * qm_o.Z(1)},
    ),
    FrontendExecutionCase(
        name="stdlib-composite",
        sample_kernel=stdlib_composite_sample,
        run_kernel=stdlib_composite_run,
        sample_mode="deterministic",
        expected_bits=(1, 0, 1),
        expected_support={(1, 0, 1)},
        expected_expval=-1.0,
        run_bindings={"obs": qm_o.Z(0) + qm_o.Z(1) + qm_o.Z(2)},
    ),
    FrontendExecutionCase(
        name="uint-mod-alternating",
        sample_kernel=uint_mod_alternating_sample,
        run_kernel=uint_mod_alternating_run,
        sample_mode="deterministic",
        expected_bits=(1, 0, 1, 0),  # sample kernel flips even indices
        expected_support={(1, 0, 1, 0)},
        # run kernel flips ODD indices via RX((i % 2) * pi) -> state (0,1,0,1),
        # so Z0+Z1+Z2+Z3 = +1-1+1-1 = 0.
        expected_expval=0.0,
        run_bindings={"obs": qm_o.Z(0) + qm_o.Z(1) + qm_o.Z(2) + qm_o.Z(3)},
    ),
    FrontendExecutionCase(
        name="pauli-evolve",
        sample_kernel=pauli_evolve_sample,
        run_kernel=pauli_evolve_run,
        sample_mode="deterministic",
        expected_bits=(1, 0),
        expected_support={(1, 0)},
        expected_expval=0.0,
        sample_bindings={"ham": qm_o.Z(0) * qm_o.Z(1), "gamma": 0.0},
        run_bindings={
            "ham": qm_o.Z(0) * qm_o.Z(1),
            "gamma": 0.0,
            "obs": qm_o.Z(0) + qm_o.Z(1),
        },
    ),
    # Regression for #467: a Hamiltonian (Z(0), 1 qubit) narrower than the
    # 2-qubit register must be identity-padded onto the untouched qubit
    # rather than rejected. The non-zero gamma only adds a phase on q[0],
    # so the Z-basis sample and ⟨Z(0)⟩ stay deterministic.
    FrontendExecutionCase(
        name="padded-pauli-evolve",
        sample_kernel=padded_pauli_evolve_sample,
        run_kernel=padded_pauli_evolve_run,
        sample_mode="deterministic",
        expected_bits=(1, 0),
        expected_support={(1, 0)},
        expected_expval=-1.0,
        sample_bindings={"ham": qm_o.Z(0), "gamma": 0.5},
        run_bindings={
            "ham": qm_o.Z(0),
            "gamma": 0.5,
            "obs": qm_o.Z(0),
        },
    ),
]


def _case_id(case: FrontendExecutionCase) -> str:
    """Return a pytest id for a frontend execution case."""
    return case.name


@pytest.mark.parametrize("case", FRONTEND_EXECUTION_CASES, ids=_case_id)
def test_frontend_pattern_sample_execution(
    backend: Backend, case: FrontendExecutionCase
) -> None:
    """Sample frontend patterns through every supported SDK backend."""
    name, transpiler, executor = backend
    shots = 512
    if name in case.unsupported_backends:
        with pytest.raises(EmitError):
            transpiler.transpile(case.sample_kernel, bindings=case.sample_bindings)
        return
    executable = transpiler.transpile(case.sample_kernel, bindings=case.sample_bindings)
    counts = _counts(executable.sample(executor, shots=shots).result())

    if case.sample_mode == "deterministic":
        assert case.expected_bits is not None
        _assert_deterministic(name, counts, case.expected_bits)
    elif case.sample_mode == "uniform":
        _assert_uniform(name, counts, case.expected_support, shots=shots)
    elif case.sample_mode == "bell":
        _assert_bell(name, counts, case.expected_support, shots=shots)
    else:  # pragma: no cover - exhaustive guard for future cases.
        raise AssertionError(f"unknown sample mode {case.sample_mode}")


@pytest.mark.parametrize("case", FRONTEND_EXECUTION_CASES, ids=_case_id)
def test_frontend_pattern_run_execution(
    backend: Backend, case: FrontendExecutionCase
) -> None:
    """Run expval frontend patterns through every supported SDK backend."""
    name, transpiler, executor = backend
    if name in case.unsupported_backends:
        with pytest.raises(EmitError):
            transpiler.transpile(case.run_kernel, bindings=case.run_bindings)
        return
    executable = transpiler.transpile(case.run_kernel, bindings=case.run_bindings)
    got = executable.run(executor).result()

    assert np.isclose(got, case.expected_expval, atol=1e-5), (
        f"{name} {case.name}: got {got}, expected {case.expected_expval}"
    )


def test_vector_view_qpe_controlled_u_parameter_sample(backend: Backend) -> None:
    """Execute QPE with controlled-U theta bound from ``gammas[1:3][0]``."""
    name, transpiler, executor = backend
    shots = 32
    executable = transpiler.transpile(
        vector_view_qpe_parameter_sample,
        bindings={"gammas": np.array([math.pi / 4, math.pi / 2, math.pi])},
    )

    result = executable.sample(executor, shots=shots).result()

    total = 0
    for value, count in result.results:
        assert value == pytest.approx(0.25), (
            f"{name}: expected phase 0.25, got {value} (count={count})"
        )
        total += count
    assert total == shots, f"{name}: expected {shots} shots, got {total}"


def test_controlled_parameterized_composite_runtime_parameter(
    backend: Backend,
) -> None:
    """Execute a controlled custom composite with a runtime angle parameter."""
    name, transpiler, executor = backend

    sample_executable = transpiler.transpile(
        controlled_parameterized_composite_sample,
        parameters=["theta"],
    )
    sample_counts = _counts(
        sample_executable.sample(
            executor,
            shots=128,
            bindings={"theta": math.pi},
        ).result()
    )
    _assert_deterministic(name, sample_counts, (1, 1))

    run_executable = transpiler.transpile(
        controlled_parameterized_composite_run,
        bindings={"obs": qm_o.Z(1)},
        parameters=["theta"],
    )
    got = run_executable.run(executor, bindings={"theta": math.pi}).result()
    assert np.isclose(got, -1.0, atol=1e-5), f"{name}: got {got}, expected -1.0"


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_controlled_random_rotation_sample_and_run(backend: Backend, seed: int) -> None:
    """Check randomized controlled-rotation probabilities and expval."""
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(0.25, math.pi - 0.25))
    shots = 2048

    sample_executable = transpiler.transpile(
        controlled_random_ry_sample, bindings={"theta": theta}
    )
    counts = _counts(sample_executable.sample(executor, shots=shots).result())
    assert set(counts).issubset({(1, 0), (1, 1)}), (
        f"{name}: controlled RY changed the control bit or returned {counts}"
    )
    assert set(counts) == {(1, 0), (1, 1)}, (
        f"{name}: randomized controlled RY missed support in {counts}"
    )
    target_one_freq = counts.get((1, 1), 0) / shots
    expected_target_one_freq = math.sin(theta / 2) ** 2
    assert abs(target_one_freq - expected_target_one_freq) < 0.08, (
        f"{name}: seed={seed} theta={theta} target-one frequency "
        f"{target_one_freq:.3f}, expected {expected_target_one_freq:.3f}"
    )

    run_executable = transpiler.transpile(
        controlled_random_ry_run,
        bindings={"theta": theta, "obs": qm_o.Z(1)},
    )
    got = run_executable.run(executor).result()
    expected = math.cos(theta)
    assert np.isclose(got, expected, atol=1e-5), (
        f"{name}: seed={seed} theta={theta} got {got}, expected {expected}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_powered_controlled_random_rotation_sample_and_run(
    backend: Backend,
    seed: int,
) -> None:
    """Check powered randomized controlled-rotation probabilities and expval."""
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(0.15, (math.pi / 2) - 0.15))
    shots = 2048

    sample_executable = transpiler.transpile(
        controlled_random_ry_power_sample,
        bindings={"theta": theta},
    )
    counts = _counts(sample_executable.sample(executor, shots=shots).result())
    assert set(counts).issubset({(1, 0), (1, 1)}), (
        f"{name}: powered controlled RY changed the control bit or returned {counts}"
    )
    assert set(counts) == {(1, 0), (1, 1)}, (
        f"{name}: powered controlled RY missed support in {counts}"
    )
    target_one_freq = counts.get((1, 1), 0) / shots
    expected_target_one_freq = math.sin(theta) ** 2
    assert abs(target_one_freq - expected_target_one_freq) < 0.08, (
        f"{name}: seed={seed} theta={theta} powered target-one frequency "
        f"{target_one_freq:.3f}, expected {expected_target_one_freq:.3f}"
    )

    run_executable = transpiler.transpile(
        controlled_random_ry_power_run,
        bindings={"theta": theta, "obs": qm_o.Z(1)},
    )
    got = run_executable.run(executor).result()
    expected = math.cos(2 * theta)
    assert np.isclose(got, expected, atol=1e-5), (
        f"{name}: seed={seed} theta={theta} got {got}, expected {expected}"
    )


def test_bound_control_indices_duplicate_rejected(backend: Backend) -> None:
    """Reject ``UInt`` control-index entries that bind to duplicates."""
    _name, transpiler, _executor = backend
    with pytest.raises(EmitError, match="duplicate"):
        transpiler.transpile(
            bound_control_indices_sample,
            bindings={"n": 2, "i": 0, "j": 0},
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_interleaved_param_expr_angle_sample_and_run(
    backend: Backend, seed: int
) -> None:
    """A runtime-parameter angle expression must not split quantum segments.

    Regression for the spurious ``MultipleQuantumSegmentsError`` raised when a
    gate-angle expression (``rx(q, -phase)``) over a runtime parameter is
    interleaved between quantum operations. Checks the negated angle is applied
    with the correct sign on both the sample and expval paths.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    # Include boundary angles (0, pi, 2*pi) alongside a random one.
    angles = [0.0, math.pi, 2 * math.pi, float(rng.uniform(0.1, 2 * math.pi - 0.1))]
    shots = 2048

    sample_executable = transpiler.transpile(
        interleaved_param_expr_sample, parameters=["phase"]
    )
    run_executable = transpiler.transpile(
        interleaved_param_expr_run,
        bindings={"obs": qm_o.Y(0)},
        parameters=["phase"],
    )

    for phase in angles:
        counts = _counts(
            sample_executable.sample(
                executor, shots=shots, bindings={"phase": phase}
            ).result()
        )
        # State is rx(-phase)|1>, so P(measure 1) = cos^2(phase / 2).
        expected_one_freq = math.cos(phase / 2) ** 2
        one_freq = counts.get((1,), 0) / shots
        assert abs(one_freq - expected_one_freq) < 0.06, (
            f"{name}: seed={seed} phase={phase} one-frequency "
            f"{one_freq:.3f}, expected {expected_one_freq:.3f}"
        )

        got = run_executable.run(executor, bindings={"phase": phase}).result()
        # <Y> on rx(-phase)|1> is -sin(phase); a dropped sign would flip it.
        expected_expval = -math.sin(phase)
        assert np.isclose(got, expected_expval, atol=1e-5), (
            f"{name}: seed={seed} phase={phase} got {got}, expected {expected_expval}"
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_pre_init_param_expr_angle_sample_and_run(backend: Backend, seed: int) -> None:
    """A gate-angle expression computed before quantum init must still apply.

    Regression for the silent miscompile where ``angle = -phase`` written before
    ``qmc.qubit_array`` was stranded in a classical prep segment and the gate
    received a zero angle. Checks the negated angle is applied with the correct
    sign on both the sample and expval paths, identical to the interleaved case.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    angles = [0.0, math.pi, 2 * math.pi, float(rng.uniform(0.1, 2 * math.pi - 0.1))]
    shots = 2048

    sample_executable = transpiler.transpile(
        pre_init_param_expr_sample, parameters=["phase"]
    )
    run_executable = transpiler.transpile(
        pre_init_param_expr_run,
        bindings={"obs": qm_o.Y(0)},
        parameters=["phase"],
    )

    for phase in angles:
        counts = _counts(
            sample_executable.sample(
                executor, shots=shots, bindings={"phase": phase}
            ).result()
        )
        # State is rx(-phase)|1>, so P(measure 1) = cos^2(phase / 2).
        expected_one_freq = math.cos(phase / 2) ** 2
        one_freq = counts.get((1,), 0) / shots
        assert abs(one_freq - expected_one_freq) < 0.06, (
            f"{name}: seed={seed} phase={phase} one-frequency "
            f"{one_freq:.3f}, expected {expected_one_freq:.3f}"
        )

        got = run_executable.run(executor, bindings={"phase": phase}).result()
        # <Y> on rx(-phase)|1> is -sin(phase); a zero angle would give 0.
        expected_expval = -math.sin(phase)
        assert np.isclose(got, expected_expval, atol=1e-5), (
            f"{name}: seed={seed} phase={phase} got {got}, expected {expected_expval}"
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_symbolic_multi_control_phase_in_loop_matches_cp(
    backend: Backend, seed: int
) -> None:
    """Reported symbolic multi-control phase in a loop matches its cp equivalent.

    Regression for the originally reported QSVT / block-encoding failure: a
    symbolic-count multi-control phase with a negated runtime angle inside a
    ``qmc.range`` loop must transpile to a single quantum segment and produce
    the same expectation value as the concrete ``qmc.cp`` circuit.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    obs = qm_o.X(0) * qm_o.X(1)
    angles = [0.0, math.pi, 2 * math.pi, float(rng.uniform(0.1, 2 * math.pi - 0.1))]

    symbolic_executable = transpiler.transpile(
        symbolic_mc_phase_run, bindings={"obs": obs}, parameters=["phase"]
    )
    cp_executable = transpiler.transpile(
        cp_phase_run, bindings={"obs": obs}, parameters=["phase"]
    )

    for phase in angles:
        symbolic_value = symbolic_executable.run(
            executor, bindings={"phase": phase}
        ).result()
        cp_value = cp_executable.run(executor, bindings={"phase": phase}).result()
        assert np.isclose(symbolic_value, cp_value, atol=1e-8), (
            f"{name}: seed={seed} phase={phase} symbolic {symbolic_value} != "
            f"cp {cp_value}"
        )


def test_classical_output_after_quantum_op_is_preserved(backend: Backend) -> None:
    """A classical block output computed after a quantum op must be returned.

    Guards the segmentation carve-out: a non-measurement classical op whose
    result is a block output (and feeds no gate) must stay in its own classical
    segment so the executor runs it, rather than being absorbed into the quantum
    segment and silently dropped (returning ``None``).
    """
    name, transpiler, executor = backend
    executable = transpiler.transpile(
        classical_output_after_quantum_run, parameters=["phase"]
    )
    got = executable.run(executor, bindings={"phase": 3.0}).result()
    assert got is not None, f"{name}: classical output was dropped (got None)"
    assert np.isclose(got, 6.0, atol=1e-8), f"{name}: got {got}, expected 6.0"


def test_value_feeding_gate_and_classical_is_not_miscompiled(
    backend: Backend,
) -> None:
    """A value used by both a gate and later classical work is not absorbed.

    Guards the absorbable-set fixpoint: a classical op whose result feeds a
    quantum gate but is also read by a later classical computation must not be
    folded into the quantum segment (which would let the later classical segment
    read a value that never executed). The kernel must raise an explicit
    ``MultipleQuantumSegmentsError`` rather than silently miscompiling.
    """
    _name, transpiler, _executor = backend
    with pytest.raises(MultipleQuantumSegmentsError):
        transpiler.transpile(value_feeding_gate_and_classical_run, parameters=["phase"])


def test_nested_classical_output_is_not_miscompiled(backend: Backend) -> None:
    """A nested classical op that is also a block output is not absorbed.

    Guards against silent data corruption: a top-level value feeding both a gate
    and a nested classical op whose result is returned must not be absorbed just
    because the nested op rides in a quantum loop. The nested op is a block
    output (a classical sink), so the kernel must raise an explicit
    ``MultipleQuantumSegmentsError`` rather than transpiling and returning
    ``None`` for the classical output field.
    """
    _name, transpiler, _executor = backend
    with pytest.raises(MultipleQuantumSegmentsError):
        transpiler.transpile(nested_classical_output_run, parameters=["phase"])


def test_pre_init_gate_and_classical_is_not_miscompiled(backend: Backend) -> None:
    """A dual-use value built before quantum init must not silently miscompile.

    A value used as both a gate angle and a classical output, computed before
    ``qmc.qubit_array``, yields a legal C->Q->Expval plan (no quantum-segment
    split), so it is not caught by the multiple-segments error. The gate would
    otherwise receive a stranded zero parameter, so the transpiler must reject
    it explicitly instead of returning a wrong result.
    """
    _name, transpiler, _executor = backend
    with pytest.raises(MultipleQuantumSegmentsError):
        transpiler.transpile(
            pre_init_gate_and_classical_run,
            bindings={"obs": qm_o.Y(0)},
            parameters=["phase"],
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_nested_classical_chain_in_loop_sample_and_run(
    backend: Backend, seed: int
) -> None:
    """A classical chain split across a loop boundary must not split segments.

    Regression for the absorbable-set analysis covering nested classical ops: a
    top-level parameter expression (``base = phase * 2``) whose next chain link
    (``angle = base + 1``) lives inside a ``qmc.range`` body must still be
    absorbed into the single quantum segment. Both qubits end up in
    ``rx(2 * phase + 1) |0>``, checked on the sample and expval paths.
    """
    name, transpiler, executor = backend
    rng = np.random.default_rng(seed)
    angles = [0.0, math.pi, 2 * math.pi, float(rng.uniform(0.1, 2 * math.pi - 0.1))]
    shots = 2048

    sample_executable = transpiler.transpile(
        nested_classical_chain_sample, parameters=["phase"]
    )
    run_executable = transpiler.transpile(
        nested_classical_chain_run,
        bindings={"obs": qm_o.Y(0) + qm_o.Y(1)},
        parameters=["phase"],
    )

    for phase in angles:
        effective = 2.0 * phase + 1.0
        counts = _counts(
            sample_executable.sample(
                executor, shots=shots, bindings={"phase": phase}
            ).result()
        )
        # Each qubit is rx(2*phase + 1)|0>, so P(qubit == 1) = sin^2(effective/2).
        expected_one_freq = math.sin(effective / 2) ** 2
        for qubit in range(2):
            one_freq = sum(c for bits, c in counts.items() if bits[qubit] == 1) / shots
            assert abs(one_freq - expected_one_freq) < 0.06, (
                f"{name}: seed={seed} phase={phase} qubit={qubit} one-frequency "
                f"{one_freq:.3f}, expected {expected_one_freq:.3f}"
            )

        got = run_executable.run(executor, bindings={"phase": phase}).result()
        # <Y> on rx(theta)|0> is -sin(theta); summed over the two qubits.
        expected_expval = 2.0 * (-math.sin(effective))
        assert np.isclose(got, expected_expval, atol=1e-5), (
            f"{name}: seed={seed} phase={phase} got {got}, expected {expected_expval}"
        )
