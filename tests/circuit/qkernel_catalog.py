from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CompositeGateType,
    InvokeOperation,
)
from qamomile.linalg import PauliLCU, PeriodicShiftLCU


@dataclass(frozen=True)
class QKernelEntry:
    """A catalog entry for a qkernel circuit.

    ``min_params`` contains numeric sweep parameters, while ``fixed_inputs``
    carries structural values such as static block-encoding descriptors that
    remain unchanged across the sweep.
    """

    id: str
    qkernel: QKernel
    description: str
    param_names: tuple[str, ...] = ()
    min_params: dict[str, int] = field(default_factory=dict)
    fixed_inputs: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def minimum_inputs(self) -> dict[str, Any]:
        """Return fixed inputs combined with the minimum sweep parameters.

        Returns:
            dict[str, Any]: Complete minimum input mapping for this entry.
        """
        return {**self.fixed_inputs, **self.min_params}


# ============================================================
# Shared helpers (internal)
# ============================================================


def __emit_oracle(
    *qubits: qmc.Vector[qmc.Qubit] | qmc.Qubit,
    name: str,
    cost: qmc.ResourceEstimate,
) -> None:
    """Emit an opaque oracle invocation directly to the tracer.

    Uses a lightweight marker invocation for catalog-only resource tests where
    the qubit handles intentionally pass through unchanged.

    Accepts any mix of Vector[Qubit] and Qubit arguments.
    """
    tracer = get_current_tracer()
    operands = [q.value for q in qubits]
    ref = CallableRef(namespace="user.oracle", name=name)
    attrs = {
        "kind": "oracle",
        "gate_type": CompositeGateType.CUSTOM.name,
        "num_control_qubits": 0,
        "num_target_qubits": 0,
        "custom_name": name,
    }
    op = InvokeOperation(
        operands=operands,
        results=[],
        target=ref,
        attrs=attrs,
        definition=CallableDef(
            ref=ref,
            opaque_cost=cost,
            attrs=attrs,
        ),
    )
    tracer.add_operation(op)


def _over_oracle(
    *qubits: qmc.Vector[qmc.Qubit] | qmc.Qubit,
    name: str,
    cost: qmc.ResourceEstimate,
) -> tuple[qmc.Vector[qmc.Qubit] | qmc.Qubit, ...]:
    __emit_oracle(*qubits, name=name, cost=cost)
    return qubits


# ============================================================
# Single-qubit gate circuits
# ============================================================


@qmc.qkernel
def single_h() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    return q


@qmc.qkernel
def single_x() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.x(q)
    return q


@qmc.qkernel
def single_y() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.y(q)
    return q


@qmc.qkernel
def single_z() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.z(q)
    return q


@qmc.qkernel
def single_t() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.t(q)
    return q


@qmc.qkernel
def single_tdg() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.tdg(q)
    return q


@qmc.qkernel
def single_s() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.s(q)
    return q


@qmc.qkernel
def single_sdg() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.sdg(q)
    return q


@qmc.qkernel
def single_p() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.p(q, 0.1)
    return q


@qmc.qkernel
def single_rx() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.rx(q, 0.2)
    return q


@qmc.qkernel
def single_ry() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.ry(q, 0.3)
    return q


@qmc.qkernel
def single_rz() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    q = qmc.rz(q, 0.4)
    return q


# ============================================================
# Two-qubit gate circuits
# ============================================================


@qmc.qkernel
def single_cx() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(2, name="q")
    q[0], q[1] = qmc.cx(q[0], q[1])
    return q


@qmc.qkernel
def single_cz() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(2, name="q")
    q[0], q[1] = qmc.cz(q[0], q[1])
    return q


@qmc.qkernel
def single_cp() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(2, name="q")
    q[0], q[1] = qmc.cp(q[0], q[1], 0.5)
    return q


@qmc.qkernel
def single_swap() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(2, name="q")
    q[0], q[1] = qmc.swap(q[0], q[1])
    return q


@qmc.qkernel
def single_rzz() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(2, name="q")
    q[0], q[1] = qmc.rzz(q[0], q[1], 0.6)
    return q


# ============================================================
# Basic circuits
# ============================================================


@qmc.qkernel
def no_operation(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    return qmc.qubit_array(n, name="qs")


@qmc.qkernel
def only_measurements(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(n, name="qs")
    return qmc.measure(qs)


@qmc.qkernel
def simple_for_loop(m: qmc.UInt) -> qmc.Qubit:
    q = qmc.qubit(name="q")
    for _ in qmc.range(m):
        q = qmc.x(q)
    return q


@qmc.qkernel
def all_rx(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    for i in qmc.range(n):
        qs[i] = qmc.rx(qs[i], thetas[i])
    return qs


@qmc.qkernel
def naive_toffoli_decomposition() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(3, name="q")

    q[2] = qmc.h(q[2])
    q[1], q[2] = qmc.cx(q[1], q[2])
    q[2] = qmc.tdg(q[2])
    q[0], q[2] = qmc.cx(q[0], q[2])
    q[2] = qmc.t(q[2])
    q[1], q[2] = qmc.cx(q[1], q[2])
    q[2] = qmc.tdg(q[2])
    q[0], q[2] = qmc.cx(q[0], q[2])
    q[1] = qmc.tdg(q[1])
    q[2] = qmc.t(q[2])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[2] = qmc.h(q[2])
    q[1] = qmc.tdg(q[1])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[0] = qmc.t(q[0])
    q[1] = qmc.s(q[1])

    return q


@qmc.qkernel
def commutated_toffoli_decomposition() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(3, name="q")

    q[2] = qmc.h(q[2])
    q[1], q[2] = qmc.cx(q[1], q[2])
    q[2] = qmc.tdg(q[2])
    q[0], q[2] = qmc.cx(q[0], q[2])
    q[2] = qmc.t(q[2])
    q[1], q[2] = qmc.cx(q[1], q[2])
    q[1] = qmc.tdg(q[1])
    q[2] = qmc.tdg(q[2])
    q[0], q[2] = qmc.cx(q[0], q[2])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[0] = qmc.t(q[0])
    q[1] = qmc.tdg(q[1])
    q[2] = qmc.t(q[2])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1] = qmc.s(q[1])
    q[2] = qmc.h(q[2])

    return q


@qmc.qkernel
def _optimal_toffoli_decomposition(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    q[2] = qmc.h(q[2])
    q[0] = qmc.tdg(q[0])
    q[1] = qmc.t(q[1])
    q[2] = qmc.t(q[2])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[2], q[0] = qmc.cx(q[2], q[0])
    q[0] = qmc.tdg(q[0])
    q[1], q[2] = qmc.cx(q[1], q[2])
    q[1], q[0] = qmc.cx(q[1], q[0])
    q[0] = qmc.tdg(q[0])
    q[1] = qmc.tdg(q[1])
    q[2] = qmc.t(q[2])
    q[2], q[0] = qmc.cx(q[2], q[0])
    q[0] = qmc.s(q[0])
    q[1], q[2] = qmc.cx(q[1], q[2])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[2] = qmc.h(q[2])

    return q


@qmc.qkernel
def optimal_toffoli_decomposition() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(3, name="q")
    q = _optimal_toffoli_decomposition(q)
    return q


@qmc.qkernel
def optimal_toffoli_decomposition_loop(m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(3, name="q")

    q = _optimal_toffoli_decomposition(q)
    for _ in qmc.range(m):
        q = _optimal_toffoli_decomposition(q)
    return q


# ============================================================
# Entanglement
# ============================================================


@qmc.qkernel
def _bell_state(q1: qmc.Qubit, q2: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    q1 = qmc.h(q1)
    q1, q2 = qmc.cx(q1, q2)
    return q1, q2


@qmc.qkernel
def bell_state() -> tuple[qmc.Qubit, qmc.Qubit]:
    q1 = qmc.qubit(name="q1")
    q2 = qmc.qubit(name="q2")
    q1, q2 = _bell_state(q1, q2)
    return q1, q2


@qmc.qkernel
def _linear_entanglement(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    n = qs.shape[0]
    for i in qmc.range(n - 1):
        qs[i], qs[i + 1] = qmc.cx(qs[i], qs[i + 1])

    return qs


@qmc.qkernel
def linear_entanglement(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    qs = _linear_entanglement(qs)
    return qs


@qmc.qkernel
def full_entanglement(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    for i in qmc.range(n):
        for j in qmc.range(i + 1, n):
            qs[i], qs[j] = qmc.cx(qs[i], qs[j])
    return qs


@qmc.qkernel
def ghz_state(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    q = _linear_entanglement(q)
    return q


@qmc.qkernel
def parallel_ghz_state(m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    n = 2**m
    qs = qmc.qubit_array(n, name="qs")
    qs[0] = qmc.h(qs[0])

    for i in qmc.range(m):
        step = n // (2**i)
        for j in qmc.range(0, n, step):
            qs[j], qs[j + step // 2] = qmc.cx(qs[j], qs[j + step // 2])

    return qs


# ============================================================
# QFT / IQFT
# ============================================================


@qmc.qkernel
def _qft(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    n = qs.shape[0]

    for j in qmc.range(n - 1, -1, -1):
        qs[j] = qmc.h(qs[j])

        for k in qmc.range(j - 1, -1, -1):
            angle = math.pi / (2 ** (j - k))
            qs[j], qs[k] = qmc.cp(qs[j], qs[k], angle)

    for j in qmc.range(n // 2):
        qs[j], qs[n - j - 1] = qmc.swap(qs[j], qs[n - j - 1])

    return qs


@qmc.qkernel
def qft(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    qs = _qft(qs)
    return qs


@qmc.qkernel
def _iqft(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    n = qs.shape[0]

    for j in qmc.range(n // 2):
        qs[j], qs[n - j - 1] = qmc.swap(qs[j], qs[n - j - 1])

    for j in qmc.range(n):
        for k in qmc.range(j):
            angle = -math.pi / (2 ** (j - k))
            qs[j], qs[k] = qmc.cp(qs[j], qs[k], angle)
        qs[j] = qmc.h(qs[j])

    return qs


@qmc.qkernel
def iqft(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    qs = _iqft(qs)
    return qs


# ============================================================
# Algorithms — quantum tests / oracle-based
# ============================================================


_controlled_oracle = qmc.Oracle(
    name="controlled_oracle",
    num_qubits=1,
    num_control_qubits=1,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1, two_qubit=1),
        calls=qmc.CallResources(queries_by_name={"controlled_oracle": 1}),
    ),
)


_one_qubit_oracle = qmc.Oracle(
    name="one_qubit_oracle",
    num_qubits=1,
    cost=qmc.ResourceEstimate(
        calls=qmc.CallResources(queries_by_name={"one_qubit_oracle": 1}),
    ),
)


_two_qubit_oracle = qmc.Oracle(
    name="two_qubit_oracle",
    num_qubits=2,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(two_qubit=1),
        calls=qmc.CallResources(queries_by_name={"two_qubit_oracle": 1}),
    ),
)


@qmc.qkernel
def hadamard_test() -> qmc.Bit:
    q = qmc.qubit(name="q")
    psi = qmc.qubit(name="psi")

    q = qmc.h(q)
    q, psi = _controlled_oracle(psi, controls=(q,))
    q = qmc.h(q)

    return qmc.measure(q)


@qmc.qkernel
def swap_test() -> qmc.Bit:
    q = qmc.qubit(name="q")
    psi = qmc.qubit(name="psi")
    phi = qmc.qubit(name="phi")

    q = qmc.h(q)
    psi, phi = qmc.cx(psi, phi)

    q, phi, psi = qmc.ccx(q, phi, psi)

    psi, phi = qmc.cx(psi, phi)
    q = qmc.h(q)

    return qmc.measure(q)


@qmc.qkernel
def simplest_oracle() -> qmc.Qubit:
    q = qmc.qubit(name="q")
    (q,) = _one_qubit_oracle(q)
    return q


@qmc.qkernel
def deutsch() -> qmc.Bit:
    qs = qmc.qubit_array(2, name="qs")
    qs[1] = qmc.x(qs[1])

    qs = qmc.h(qs)

    (qs[0], qs[1]) = _two_qubit_oracle(qs[0], qs[1])

    qs[0] = qmc.h(qs[0])

    return qmc.measure(qs[0])


@qmc.qkernel
def deutsch_jozsa(n: qmc.UInt) -> qmc.Bit:
    qs = qmc.qubit_array(n + 1, name="qs")
    targets = qs[0:n]
    ancilla = qs[n]

    targets = qmc.h(targets)
    ancilla = qmc.x(ancilla)
    ancilla = qmc.h(ancilla)

    (targets, ancilla) = _over_oracle(
        targets,
        ancilla,
        name="deutsch_jozsa_oracle",
        cost=qmc.ResourceEstimate(
            calls=qmc.CallResources(queries_by_name={"deutsch_jozsa_oracle": 1}),
        ),
    )  # type: ignore

    targets = qmc.h(targets)  # type: ignore

    return qmc.measure(targets)  # type: ignore


@qmc.qkernel
def _simon(
    qs1: qmc.Vector[qmc.Qubit], qs2: qmc.Vector[qmc.Qubit]
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    qs1 = qmc.h(qs1)
    qs1, qs2 = _over_oracle(
        qs1,
        qs2,
        name="simon_oracle",
        cost=qmc.ResourceEstimate(
            calls=qmc.CallResources(queries_by_name={"simon_oracle": 1}),
        ),
    )  # type: ignore
    qs1 = qmc.h(qs1)
    return qs1, qs2


@qmc.qkernel
def simon(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(2 * n, name="qs")  # type: ignore
    qs1 = qs[0:n]
    qs2 = qs[n : 2 * n]
    qs1, qs2 = _simon(qs1, qs2)
    return qmc.measure(qs1)


@qmc.qkernel
def teleportation() -> qmc.Qubit:
    psi = qmc.qubit(name="psi")
    ancilla = qmc.qubit(name="ancilla")
    target = qmc.qubit(name="target")

    # Prepare state for teleporations.
    psi = qmc.x(psi)

    # Create Bell pair between ancilla and target.
    ancilla, target = _bell_state(ancilla, target)

    # Bell measurement on psi and ancilla.
    psi, ancilla = qmc.cx(psi, ancilla)
    psi = qmc.h(psi)
    m_psi = qmc.measure(psi)
    m_ancilla = qmc.measure(ancilla)

    if m_ancilla:
        target = qmc.x(target)
    if m_psi:
        target = qmc.z(target)

    return target


# ============================================================
# QPE
# ============================================================


@qmc.qkernel
def _phase(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    return qmc.p(q, theta)


@qmc.qkernel
def phase_gate_qpe(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    target = qmc.qubit(name="target")
    controlled_u = qmc.control(_phase)

    qs = qmc.h(qs)

    for k in qmc.range(n):
        qs[k], target = controlled_u(qs[k], target, power=2**k, theta=theta)

    qs = _iqft(qs)

    return qs


_controlled_u = qmc.Oracle(
    name="controlled_u",
    num_qubits=1,
    num_control_qubits=1,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1, two_qubit=1),
        calls=qmc.CallResources(queries_by_name={"controlled_u": 1}),
    ),
)


@qmc.qkernel
def opaque_oracle_qpe(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    target = qmc.qubit(name="target")

    qs = qmc.h(qs)

    for k in qmc.range(n):
        for _rep in qmc.range(2**k):
            qs[k], target = _controlled_u(target, controls=(qs[k],))

    qs = _iqft(qs)

    return qs


# ============================================================
# Variational / optimization
# ============================================================


@qmc.qkernel
def hardware_efficient_ansatz(
    n: qmc.UInt,
    thetas: qmc.Matrix[qmc.Float],
    phis: qmc.Matrix[qmc.Float],
    num_layers: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    for i in qmc.range(num_layers - 1):
        for j in qmc.range(n):
            qs[j] = qmc.ry(qs[j], thetas[i, j])
            qs[j] = qmc.rz(qs[j], phis[i, j])

        qs = _linear_entanglement(qs)

    for j in qmc.range(n):
        qs[j] = qmc.ry(qs[j], thetas[num_layers - 1, j])
        qs[j] = qmc.rz(qs[j], phis[num_layers - 1, j])
    return qs


@qmc.qkernel
def qaoa_state_umbiguous(
    n: qmc.UInt,
    num_layers: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")

    qs = qmc.h(qs)

    for layer in qmc.range(num_layers):
        # Ising layer
        for (i, j), Jij in quad.items():
            qs[i], qs[j] = qmc.rzz(qs[i], qs[j], angle=Jij * gammas[layer])
        for i, hi in linear.items():
            qs[i] = qmc.rz(qs[i], angle=hi * gammas[layer])
        # X mixer layer
        for i in qmc.range(n):
            qs[i] = qmc.rx(qs[i], angle=2.0 * betas[layer])

    return qs


# ============================================================
# Multi-controlled gates
# ============================================================


@qmc.qkernel
def _network_decomposition_controlled_z(
    qs: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    n = qs.shape[0]
    ladder_qubits = qs[0 : n - 1]
    target_qubit = qs[n - 1]
    num_ancillas = n - 2
    ancillas = qmc.qubit_array(num_ancillas, name="ancillas")

    ladder_qubits[0], ladder_qubits[1], ancillas[0] = qmc.ccx(
        ladder_qubits[0], ladder_qubits[1], ancillas[0]
    )
    for i in qmc.range(0, num_ancillas - 1):
        ladder_qubits[i + 2], ancillas[i], ancillas[i + 1] = qmc.ccx(
            ladder_qubits[i + 2], ancillas[i], ancillas[i + 1]
        )

    target_qubit = qmc.h(target_qubit)
    ancillas[num_ancillas - 1], target_qubit = qmc.cx(
        ancillas[num_ancillas - 1], target_qubit
    )
    target_qubit = qmc.h(target_qubit)

    for i in qmc.range(num_ancillas - 2, -1, -1):
        ladder_qubits[i + 2], ancillas[i], ancillas[i + 1] = qmc.ccx(
            ladder_qubits[i + 2], ancillas[i], ancillas[i + 1]
        )
    ladder_qubits[0], ladder_qubits[1], ancillas[0] = qmc.ccx(
        ladder_qubits[0], ladder_qubits[1], ancillas[0]
    )

    qs[0 : n - 1] = ladder_qubits
    qs[n - 1] = target_qubit
    return qs


@qmc.qkernel
def network_decomposition_controlled_z(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    qs = _network_decomposition_controlled_z(qs)
    return qs


@qmc.qkernel
def _naive_multi_controlled_z(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    n = qs.shape[0]
    multi_controlled_z = qmc.control(qmc.z, num_controls=n - 1)
    qs[0 : n - 1], qs[n - 1] = multi_controlled_z(qs[0 : n - 1], qs[n - 1])
    return qs


@qmc.qkernel
def naive_multi_controlled_z(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    qs = _naive_multi_controlled_z(qs)
    return qs


# ============================================================
# Grover
# ============================================================


@qmc.qkernel
def _grover_operator_network_decomposition(
    qs: qmc.Vector[qmc.Qubit], q: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    # Call the oracle.
    (qs, q) = _over_oracle(
        qs,
        q,
        name="grover_oracle",
        cost=qmc.ResourceEstimate(
            calls=qmc.CallResources(queries_by_name={"grover_oracle": 1}),
        ),
    )  # type: ignore
    # Apply the diffusion operator,
    # which can be implemented as H + X + multi-controlled Z + X + H.
    qs = qmc.h(qs)
    qs = qmc.x(qs)
    qs = _network_decomposition_controlled_z(qs)
    qs = qmc.x(qs)
    qs = qmc.h(qs)

    return qs, q


@qmc.qkernel
def _grover_network_decomposition(
    qs: qmc.Vector[qmc.Qubit], q: qmc.Qubit, n_iters: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    # Initialise all the qubits.
    qs = qmc.h(qs)
    q = qmc.x(q)
    q = qmc.h(q)

    # Apply the diffusion operator (inversion about the mean).
    for _ in qmc.range(n_iters):
        qs, q = _grover_operator_network_decomposition(qs, q)

    return qs


@qmc.qkernel
def grover_network_decomposition(n: qmc.UInt, n_iters: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(n, name="qs")
    q = qmc.qubit(name="q")
    qs = _grover_network_decomposition(qs, q, n_iters)
    bits = qmc.measure(qs)
    return bits


@qmc.qkernel
def _grover_naive_multi_controlled_z(
    qs: qmc.Vector[qmc.Qubit], q: qmc.Qubit, n_iters: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    # Initialise all the qubits.
    qs = qmc.h(qs)
    q = qmc.x(q)
    q = qmc.h(q)

    # Apply the diffusion operator (inversion about the mean).
    for _ in qmc.range(n_iters):
        # Call the oracle.
        (qs, q) = _over_oracle(
            qs,
            q,
            name="grover_oracle",
            cost=qmc.ResourceEstimate(
                calls=qmc.CallResources(queries_by_name={"grover_oracle": 1}),
            ),
        )  # type: ignore
        # Apply the diffusion operator,
        # which can be implemented as H + X + multi-controlled Z + X + H.
        qs = qmc.h(qs)
        qs = qmc.x(qs)
        qs = _naive_multi_controlled_z(qs)
        qs = qmc.x(qs)
        qs = qmc.h(qs)

    return qs


@qmc.qkernel
def grover_naive_multi_controlled_z(
    n: qmc.UInt, n_iters: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(n, name="qs")
    q = qmc.qubit(name="q")
    qs = _grover_naive_multi_controlled_z(qs, q, n_iters)
    bits = qmc.measure(qs)
    return bits


@qmc.qkernel
def quantum_counting(
    n: qmc.UInt, m: qmc.UInt
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    qs1 = qmc.qubit_array(n, name="qs1")
    qs2 = qmc.qubit_array(m, name="qs2")
    q = qmc.qubit(name="q")
    controlled_grover = qmc.control(
        _grover_operator_network_decomposition, num_controls=1
    )

    qs1 = qmc.h(qs1)
    qs2 = qmc.h(qs2)
    q = qmc.h(q)

    for t in qmc.range(n):
        qs1[n - 1 - t], qs2, q = controlled_grover(
            qs1[n - 1 - t],
            qs2,
            q,
            power=2**t,
        )

    qs1 = _iqft(qs1)

    return qs1, qs2, q  # type: ignore


# ============================================================
# Arithmetic
# ============================================================


@qmc.qkernel
def _maj(
    a: qmc.Qubit, b: qmc.Qubit, c: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """(a, b, c) = (MAJ(a, b, c)= ab + bc + ca, a + b, a + c)"""
    a, b = qmc.cx(a, b)
    a, c = qmc.cx(a, c)
    b, c, a = qmc.ccx(b, c, a)
    return a, b, c


@qmc.qkernel
def maj() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    a = qmc.qubit(name="a")
    b = qmc.qubit(name="b")
    c = qmc.qubit(name="c")
    a, b, c = _maj(a, b, c)
    return a, b, c


@qmc.qkernel
def maj_loop(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    a = qmc.qubit_array(n, name="a")
    b = qmc.qubit_array(n, name="b")
    ancilla = qmc.qubit(name="ancilla")

    a[0], b[0], ancilla = _maj(a[0], b[0], ancilla)
    for i in qmc.range(1, n):
        a[i], b[i], a[i - 1] = _maj(a[i], b[i], a[i - 1])

    return a, b


@qmc.qkernel
def _uma_2_cnot(
    a: qmc.Qubit, b: qmc.Qubit, c: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """UnMajority and Add"""
    b, c, a = qmc.ccx(b, c, a)
    a, c = qmc.cx(a, c)
    c, b = qmc.cx(c, b)
    return a, b, c


@qmc.qkernel
def uma_2_cnot() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    a = qmc.qubit(name="a")
    b = qmc.qubit(name="b")
    c = qmc.qubit(name="c")
    a, b, c = _uma_2_cnot(a, b, c)
    return a, b, c


@qmc.qkernel
def uma_2_cnot_loop(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    a = qmc.qubit_array(n, name="a")
    b = qmc.qubit_array(n, name="b")
    ancilla = qmc.qubit(name="ancilla")

    for i in qmc.range(n - 1, 0, -1):
        a[i], b[i], a[i - 1] = _uma_2_cnot(a[i], b[i], a[i - 1])
    a[0], b[0], ancilla = _uma_2_cnot(a[0], b[0], ancilla)

    return a, b


@qmc.qkernel
def _uma_3_cnot(
    a: qmc.Qubit, b: qmc.Qubit, c: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    b = qmc.x(b)
    c, b = qmc.cx(c, b)
    b, c, a = qmc.ccx(b, c, a)
    b = qmc.x(b)
    a, c = qmc.cx(a, c)
    a, b = qmc.cx(a, b)
    return a, b, c


@qmc.qkernel
def uma_3_cnot() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    a = qmc.qubit(name="a")
    b = qmc.qubit(name="b")
    c = qmc.qubit(name="c")
    a, b, c = _uma_3_cnot(a, b, c)
    return a, b, c


@qmc.qkernel
def uma_3_cnot_loop(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    a = qmc.qubit_array(n, name="a")
    b = qmc.qubit_array(n, name="b")
    ancilla = qmc.qubit(name="ancilla")

    for i in qmc.range(n - 1, 0, -1):
        a[i], b[i], a[i - 1] = _uma_3_cnot(a[i], b[i], a[i - 1])
    a[0], b[0], ancilla = _uma_3_cnot(a[0], b[0], ancilla)

    return a, b


@qmc.qkernel
def _simple_ripple_carry_adder_2_cnot(
    a: qmc.Vector[qmc.Qubit], b: qmc.Vector[qmc.Qubit], z: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    n = a.shape[0]
    ancilla = qmc.qubit(name="ancilla")

    a[0], b[0], ancilla = _maj(a[0], b[0], ancilla)
    for i in qmc.range(1, n):
        a[i], b[i], a[i - 1] = _maj(a[i], b[i], a[i - 1])

    a[n - 1], z = qmc.cx(a[n - 1], z)

    for i in qmc.range(n - 1, 0, -1):
        a[i], b[i], a[i - 1] = _uma_2_cnot(a[i], b[i], a[i - 1])
    a[0], b[0], ancilla = _uma_2_cnot(a[0], b[0], ancilla)

    return a, b, z


@qmc.qkernel
def simple_ripple_carry_adder_2_cnot(
    n: qmc.UInt,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    a = qmc.qubit_array(n, name="a")
    b = qmc.qubit_array(n, name="b")
    z = qmc.qubit(name="z")
    a, b, z = _simple_ripple_carry_adder_2_cnot(a, b, z)
    return a, b, z


@qmc.qkernel
def _simple_ripple_carry_adder_3_cnot(
    a: qmc.Vector[qmc.Qubit], b: qmc.Vector[qmc.Qubit], z: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    n = a.shape[0]
    ancilla = qmc.qubit(name="ancilla")

    a[0], b[0], ancilla = _maj(a[0], b[0], ancilla)
    for i in qmc.range(1, n):
        a[i], b[i], a[i - 1] = _maj(a[i], b[i], a[i - 1])

    a[n - 1], z = qmc.cx(a[n - 1], z)

    for i in qmc.range(n - 1, 0, -1):
        a[i], b[i], a[i - 1] = _uma_3_cnot(a[i], b[i], a[i - 1])
    a[0], b[0], ancilla = _uma_3_cnot(a[0], b[0], ancilla)

    return a, b, z


@qmc.qkernel
def simple_ripple_carry_adder_3_cnot(
    n: qmc.UInt,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    a = qmc.qubit_array(n, name="a")
    b = qmc.qubit_array(n, name="b")
    z = qmc.qubit(name="z")
    a, b, z = _simple_ripple_carry_adder_3_cnot(a, b, z)
    return a, b, z


@qmc.qkernel
def _draper_inplace_qc_adder(
    qs: qmc.Vector[qmc.Qubit], num: qmc.UInt, factor: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    n = qs.shape[0]

    # Transform qs to Fourier basis.
    qs = _qft(qs)

    # Add num * factor by applying phase rotations in Fourier basis.
    for i in qmc.range(n):
        angle = factor * (num * math.pi) / (2**i)
        qs[i] = qmc.p(qs[i], angle)

    # Transform back from Fourier basis.
    qs = _iqft(qs)

    return qs


@qmc.qkernel
def draper_inplace_qc_adder(
    n: qmc.UInt, num: qmc.UInt, factor: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    qs = _draper_inplace_qc_adder(qs, num, factor)
    return qs


@qmc.qkernel
def _ttk_adder(
    a: qmc.Vector[qmc.Qubit], b: qmc.Vector[qmc.Qubit], z: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    n = a.shape[0]

    for i in qmc.range(1, n):
        a[i], b[i] = qmc.cx(a[i], b[i])

    a[n - 1], z = qmc.cx(a[n - 1], z)
    for i in qmc.range(n - 2, 0, -1):
        a[i], a[i + 1] = qmc.cx(a[i], a[i + 1])

    for i in qmc.range(n - 1):
        b[i], a[i], a[i + 1] = qmc.ccx(b[i], a[i], a[i + 1])
    b[n - 1], a[n - 1], z = qmc.ccx(b[n - 1], a[n - 1], z)

    for i in qmc.range(n - 1, 0, -1):
        a[i], b[i] = qmc.cx(a[i], b[i])
        b[i - 1], a[i - 1], a[i] = qmc.ccx(b[i - 1], a[i - 1], a[i])

    for i in qmc.range(1, n - 1):
        a[i], a[i + 1] = qmc.cx(a[i], a[i + 1])

    for i in qmc.range(n):
        a[i], b[i] = qmc.cx(a[i], b[i])

    return a, b, z


@qmc.qkernel
def ttk_adder(
    n: qmc.UInt,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    a = qmc.qubit_array(n, name="a")
    b = qmc.qubit_array(n, name="b")
    z = qmc.qubit(name="z")
    a, b, z = _ttk_adder(a, b, z)
    return a, b, z


@qmc.qkernel
def _cdkm_adder(
    a: qmc.Vector[qmc.Qubit], b: qmc.Vector[qmc.Qubit], z: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    x = qmc.qubit(name="x")
    n = a.shape[0]

    for i in qmc.range(1, n, 1):
        a[i], b[i] = qmc.cx(a[i], b[i])

    a[1], x = qmc.cx(a[1], x)

    a[0], b[0], x = qmc.ccx(a[0], b[0], x)
    a[1], a[2] = qmc.cx(a[1], a[2])

    x, b[1], a[1] = qmc.ccx(x, b[1], a[1])
    a[3], a[2] = qmc.cx(a[3], a[2])

    for i in qmc.range(2, n - 2, 1):
        a[i - 1], b[i], a[i] = qmc.ccx(a[i - 1], b[i], a[i])
        a[i + 2], a[i + 1] = qmc.cx(a[i + 2], a[i + 1])

    a[n - 3], b[n - 2], a[n - 2] = qmc.ccx(a[n - 3], b[n - 2], a[n - 2])
    a[n - 1], z = qmc.cx(a[n - 1], z)

    a[n - 2], b[n - 1], z = qmc.ccx(a[n - 2], b[n - 1], z)
    for i in qmc.range(1, n - 1, 1):
        b[i] = qmc.x(b[i])

    x, b[1] = qmc.cx(x, b[1])
    for i in qmc.range(2, n, 1):
        a[i - 1], b[i] = qmc.cx(a[i - 1], b[i])

    a[n - 3], b[n - 2], a[n - 2] = qmc.ccx(a[n - 3], b[n - 2], a[n - 2])

    for i in qmc.range(n - 3, 1, -1):
        a[i - 1], b[i], a[i] = qmc.ccx(a[i - 1], b[i], a[i])
        a[i + 2], a[i + 1] = qmc.cx(a[i + 2], a[i + 1])
        b[i + 1] = qmc.x(b[i + 1])

    x, b[1], a[1] = qmc.ccx(x, b[1], a[1])
    a[3], a[2] = qmc.cx(a[3], a[2])
    b[2] = qmc.x(b[2])

    a[0], b[0], x = qmc.ccx(a[0], b[0], x)
    a[2], a[1] = qmc.cx(a[2], a[1])
    b[1] = qmc.x(b[1])

    a[1], x = qmc.cx(a[1], x)

    for i in qmc.range(0, n, 1):
        a[i], b[i] = qmc.cx(a[i], b[i])

    return a, b, z


@qmc.qkernel
def cdkm_adder(
    n: qmc.UInt,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
    a = qmc.qubit_array(n, name="a")
    b = qmc.qubit_array(n, name="b")
    z = qmc.qubit(name="z")
    a, b, z = _cdkm_adder(a, b, z)
    return a, b, z


# ============================================================
# Block encoding / QSVT resource consumers
# ============================================================


@qmc.qkernel
def block_encoding_consumer(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    r"""Allocate and apply one exact block encoding.

    Let ``a`` and ``n`` be the signal and system widths. In the notation of
    Gilyén et al. (arXiv:1806.01838, Definition 43), the descriptor implements

    ``B = (<0|**a tensor I) U (|0>**a tensor I) = A / alpha``,

    where ``alpha = encoding.normalization``. Qamomile's current LCU
    descriptors are exact, so the paper's block-encoding error ``epsilon`` is
    zero. This consumer allocates both registers before invoking ``U``;
    consequently its logical width is ``a + n``.

    Args:
        encoding (qmc.LCUBlockEncoding): Static exact block-encoding
            descriptor whose all-zero signal block is consumed.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Signal and system
            registers after one application of the block-encoding unitary.
    """
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    return encoding.unitary(signal, system)


@qmc.qkernel
def qsvt_consumer(
    encoding: qmc.LCUBlockEncoding,
    phases: qmc.Vector[qmc.Float],
    phase_count: qmc.UInt,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    r"""Allocate and apply one QSVT sequence in projector-phase convention.

    Write ``Pi = |0><0|**a tensor I`` and
    ``R(phi) = exp(i phi (2 Pi - I))``. For ``phase_count = d + 1``, Qamomile
    chronologically applies ``R(phi_0), U, R(phi_1), U dagger, ...,
    R(phi_d)``. Thus the circuit makes exactly ``d`` block-encoding or inverse
    queries and ``d + 1`` projector rotations.

    If ``B = W Sigma V dagger`` is the normalized encoded block, Gilyén et al.
    (arXiv:1806.01838, Definitions 15-17) show that projection back onto the
    signal-zero subspace gives ``W P(Sigma) V dagger`` for odd ``P`` and
    ``V P(Sigma) V dagger`` for even ``P``. In particular, the Qamomile phases
    ``(0, -pi/2, pi/2)`` realize ``T_2(x) = 2 x**2 - 1``, so the expected even
    transform is ``2 B dagger B - I``. For the non-Hermitian catalog canary
    ``B = |0><1|``, this is ``-Z`` rather than the incorrect matrix polynomial
    ``B**2 = 0``.

    One Qamomile projector rotation contains ``2a`` X gates, two MCX gates,
    and one RZ gate, hence logical cost ``2a + 3``. If one encoding query costs
    ``C_U``, the complete logical estimate is therefore
    ``d C_U + (d + 1)(2a + 3)`` gates and ``n + a + 1`` qubits; the final one
    is the reusable projector auxiliary. Martyn et al. (arXiv:2105.02859,
    Section II and Appendix A) explain why phases synthesized in the different
    ``Wx`` convention require conversion before being used here.

    Args:
        encoding (qmc.LCUBlockEncoding): Static exact block encoding of
            ``A / alpha``.
        phases (qmc.Vector[qmc.Float]): Projector-rotation phases in sequence
            order.
        phase_count (qmc.UInt): Positive number ``d + 1`` of phases to apply.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Signal and system
            registers after the singular-value transformation.
    """
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    return qmc.qsvt(
        signal,
        system,
        phases,
        encoding,
        phase_count=phase_count,
    )


# The catalog values are mathematical canaries, not application benchmarks:
# they span signal widths a=1, 2, 3 and include complex/non-Hermitian blocks.
# Qamomile keeps one pass-through signal qubit for the identity descriptor so
# it remains composable, although Definition 44's abstract trivial encoding
# could use a=0.
_BLOCK_ENCODING_IDENTITY = qmc.identity_block_encoding(num_system_qubits=2)
# PauliLCU.from_matrix encodes B = |0><1| with alpha=1. Its zero singular
# value makes the even T_2 result -Z and detects an accidental B**2 transform.
_BLOCK_ENCODING_PAULI = qmc.pauli_lcu_block_encoding(
    PauliLCU.from_matrix(
        np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
        atol=1e-12,
    )
)
# The diagonal Ising operator is A = 0.5 I + Z_0 - 0.25 i Z_1, with
# alpha = 0.5 + 1 + 0.25 = 1.75 and a two-qubit selector for three terms.
_BLOCK_ENCODING_ISING_Z = qmc.ising_z_block_encoding(
    {(): 0.5, (0,): 1.0, (1,): -0.25j},
    num_system_qubits=2,
)
# This is the periodic second-difference operator A = S^-1 - 2I + S. The LCU
# normalization is alpha = |1| + |-2| + |1| = 4.
_BLOCK_ENCODING_PERIODIC_SHIFT = qmc.periodic_shift_lcu_block_encoding(
    PeriodicShiftLCU.from_coefficients(
        {-1: 1.0, 0: -2.0, 1: 1.0},
        register_sizes=(2,),
    )
)
# Lemma 52 of arXiv:1806.01838 gives the recursive normalization
# alpha = sum_j |c_j| alpha_j = 0.75 + 0.5 * 1.75 = 1.625. The outer selector
# adds one signal qubit to the two-qubit Ising child, giving a=3.
_BLOCK_ENCODING_RECURSIVE_LCU = qmc.lcu_block_encoding(
    (
        qmc.LCUBlockEncodingTerm(0.75, _BLOCK_ENCODING_IDENTITY),
        qmc.LCUBlockEncodingTerm(-0.5j, _BLOCK_ENCODING_ISING_Z),
    )
)

_BLOCK_ENCODING_CATALOG_CASES = (
    ("identity", _BLOCK_ENCODING_IDENTITY),
    ("pauli", _BLOCK_ENCODING_PAULI),
    ("ising_z", _BLOCK_ENCODING_ISING_Z),
    ("periodic_shift", _BLOCK_ENCODING_PERIODIC_SHIFT),
    ("recursive_lcu", _BLOCK_ENCODING_RECURSIVE_LCU),
)


# ============================================================
# Catalog
# ============================================================

n = sp.Symbol("n", integer=True, positive=True)
m = sp.Symbol("m", integer=True, positive=True)
n_iters = sp.Symbol("n_iters", integer=True, positive=True)
num_layers = sp.Symbol("num_layers", integer=True, positive=True)

QKERNEL_CATALOG: list[QKernelEntry] = [
    # --- Single-qubit gate entries ---
    QKernelEntry(
        id="single_h",
        qkernel=single_h,
        description="Single H gate",
        tags=("concrete", "clifford", "single_gate"),
    ),
    QKernelEntry(
        id="single_x",
        qkernel=single_x,
        description="Single X gate",
        tags=("concrete", "clifford", "single_gate"),
    ),
    QKernelEntry(
        id="single_y",
        qkernel=single_y,
        description="Single Y gate",
        tags=("concrete", "clifford", "single_gate"),
    ),
    QKernelEntry(
        id="single_z",
        qkernel=single_z,
        description="Single Z gate",
        tags=("concrete", "clifford", "single_gate"),
    ),
    QKernelEntry(
        id="single_t",
        qkernel=single_t,
        description="Single T gate",
        tags=("concrete", "single_gate"),
    ),
    QKernelEntry(
        id="single_tdg",
        qkernel=single_tdg,
        description="Single T^dagger gate",
        tags=("concrete", "single_gate"),
    ),
    QKernelEntry(
        id="single_s",
        qkernel=single_s,
        description="Single S gate",
        tags=("concrete", "clifford", "single_gate"),
    ),
    QKernelEntry(
        id="single_sdg",
        qkernel=single_sdg,
        description="Single S^dagger gate",
        tags=("concrete", "clifford", "single_gate"),
    ),
    QKernelEntry(
        id="single_p",
        qkernel=single_p,
        description="Single P (phase) gate",
        tags=("concrete", "rotation", "single_gate"),
    ),
    QKernelEntry(
        id="single_rx",
        qkernel=single_rx,
        description="Single RX gate",
        tags=("concrete", "rotation", "single_gate"),
    ),
    QKernelEntry(
        id="single_ry",
        qkernel=single_ry,
        description="Single RY gate",
        tags=("concrete", "rotation", "single_gate"),
    ),
    QKernelEntry(
        id="single_rz",
        qkernel=single_rz,
        description="Single RZ gate",
        tags=("concrete", "rotation", "single_gate"),
    ),
    # --- Two-qubit gate entries ---
    QKernelEntry(
        id="single_cx",
        qkernel=single_cx,
        description="Single CX (CNOT) gate",
        tags=("concrete", "clifford", "single_gate"),
    ),
    QKernelEntry(
        id="single_cz",
        qkernel=single_cz,
        description="Single CZ gate",
        tags=("concrete", "clifford", "single_gate"),
    ),
    QKernelEntry(
        id="single_cp",
        qkernel=single_cp,
        description="Single CP (controlled-phase) gate",
        tags=("concrete", "rotation", "single_gate"),
    ),
    QKernelEntry(
        id="single_swap",
        qkernel=single_swap,
        description="Single SWAP gate",
        tags=("concrete", "clifford", "single_gate"),
    ),
    QKernelEntry(
        id="single_rzz",
        qkernel=single_rzz,
        description="Single RZZ gate",
        tags=("concrete", "rotation", "single_gate"),
    ),
    # --- Basic circuits ---
    QKernelEntry(
        id="no_operation",
        qkernel=no_operation,
        description="No operation",
        param_names=("n",),
        min_params={"n": 1},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="only_measurements",
        qkernel=only_measurements,
        description="Only measurements",
        param_names=("n",),
        min_params={"n": 1},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="simple_for_loop",
        qkernel=simple_for_loop,
        description="Simply for loop with parametric m iterations applying X gate on a single qubit",
        param_names=("m",),
        min_params={"m": 2},
        tags=("clifford", "parametric"),
    ),
    QKernelEntry(
        id="naive_toffoli_decomposition",
        qkernel=naive_toffoli_decomposition,
        description="Naive Toffoli decomposition",
    ),
    QKernelEntry(
        id="commutated_toffoli_decomposition",
        qkernel=commutated_toffoli_decomposition,
        description="Commutated Toffoli decomposition",
    ),
    QKernelEntry(
        id="optimal_toffoli_decomposition",
        qkernel=optimal_toffoli_decomposition,
        description="Optimal Toffoli decomposition",
    ),
    QKernelEntry(
        id="optimal_toffoli_decomposition_loop",
        qkernel=optimal_toffoli_decomposition_loop,
        description="Optimal Toffoli decomposition with loop",
        min_params={"m": 0},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="all_rx",
        qkernel=all_rx,
        description="Apply RX with parametric angles to each of n qubits",
        param_names=("n", "thetas"),
        min_params={"n": 1},
        tags=("parametric",),
    ),
    # --- Entanglement ---
    QKernelEntry(
        id="bell_state",
        qkernel=bell_state,
        description="Bell state: H + CX on 2 qubits",
        tags=("concrete", "clifford"),
    ),
    QKernelEntry(
        id="linear_entanglement",
        qkernel=linear_entanglement,
        description="Linear entanglement: apply CX between adjacent pairs of qubits in a vector",
        param_names=("n",),
        min_params={"n": 2},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="full_entanglement",
        qkernel=full_entanglement,
        description="Full entanglement: apply CX between every pair of qubits in a vector",
        param_names=("n",),
        min_params={"n": 2},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="ghz_state",
        qkernel=ghz_state,
        description="GHZ state with parametric n qubits",
        param_names=("n",),
        min_params={"n": 2},
        tags=("parametric", "clifford"),
    ),
    QKernelEntry(
        id="parallel_ghz_state",
        qkernel=parallel_ghz_state,
        description="GHZ state parallely prepared with parametric 2**m qubits",
        param_names=("m",),
        min_params={"m": 1},
        tags=("parametric", "clifford"),
    ),
    # --- QFT / IQFT ---
    QKernelEntry(
        id="qft",
        qkernel=qft,
        description="QFT with parametric n qubits",
        param_names=("n",),
        min_params={"n": 2},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="iqft",
        qkernel=iqft,
        description="IQFT with parametric n qubits",
        param_names=("n",),
        min_params={"n": 2},
        tags=("parametric",),
    ),
    # --- Algorithms — quantum tests / oracle-based ---
    QKernelEntry(
        id="hadamard_test",
        qkernel=hadamard_test,
        description="Hadamard test with an opaque controlled oracle",
        tags=("oracle",),
    ),
    QKernelEntry(
        id="swap_test",
        qkernel=swap_test,
        description="Swap test",
        tags=("concrete",),
    ),
    QKernelEntry(
        id="simplest_oracle",
        qkernel=simplest_oracle,
        description="Simplest oracle with 1 query to a 1-qubit oracle",
        tags=("oracle",),
    ),
    QKernelEntry(
        id="deutsch",
        qkernel=deutsch,
        description="Deutsch's algorithm",
        tags=("oracle",),
    ),
    QKernelEntry(
        id="deutsch_jozsa",
        qkernel=deutsch_jozsa,
        description="Deutsch-Jozsa algorithm",
        param_names=("n",),
        min_params={"n": 1},
        tags=("oracle",),
    ),
    QKernelEntry(
        id="simon",
        qkernel=simon,
        description="Simon's algorithm",
        param_names=("n",),
        min_params={"n": 1},
        tags=("oracle",),
    ),
    QKernelEntry(
        id="teleportation",
        qkernel=teleportation,
        description="Quantum Teleportation with X",
    ),
    # --- QPE ---
    QKernelEntry(
        id="phase_gate_qpe",
        qkernel=phase_gate_qpe,
        description="QPE with parametric n qubits for phase operator",
        param_names=("n",),
        min_params={"n": 1},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="opaque_oracle_qpe",
        qkernel=opaque_oracle_qpe,
        description="QPE with opaque oracle (controlled-U as black-box)",
        param_names=("n",),
        min_params={"n": 1},
        tags=("parametric", "oracle"),
    ),
    # --- Variational / optimization ---
    QKernelEntry(
        id="hardware_efficient_ansatz",
        qkernel=hardware_efficient_ansatz,
        description="Hardware-efficient ansatz with layers of RX and CX gates.",
        param_names=("n", "num_layers"),
        min_params={"n": 2, "num_layers": 1},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="qaoa_state_umbiguous",
        qkernel=qaoa_state_umbiguous,
        description="QAOA state preparation with parametric n qubits and p layers, and parametric Ising",
        param_names=("n",),
        min_params={"n": 3, "num_layers": 1},
        tags=("parametric",),
    ),
    # --- Multi-controlled gates ---
    QKernelEntry(
        id="network_decomposition_controlled_z",
        qkernel=network_decomposition_controlled_z,
        description="Network decomposition of multi-controlled Z gate with parametric n qubits",
        param_names=("n",),
        min_params={"n": 3},
        tags=("oracle",),
    ),
    QKernelEntry(
        id="naive_multi_controlled_z",
        qkernel=naive_multi_controlled_z,
        description="Naive multi-controlled Z",
        param_names=("n",),
        min_params={"n": 2},
        tags=("oracle",),
    ),
    # --- Grover ---
    QKernelEntry(
        id="grover_network_decomposition",
        qkernel=grover_network_decomposition,
        description="Grover's algorithm with V-chain multi-controlled Z",
        param_names=("n", "n_iters"),
        min_params={"n": 3, "n_iters": 1},
        tags=("oracle",),
    ),
    QKernelEntry(
        id="grover_naive_multi_controlled_z",
        qkernel=grover_naive_multi_controlled_z,
        description="Grover's algorithm with naive multi-controlled Z",
        param_names=("n", "n_iters"),
        min_params={"n": 2, "n_iters": 1},
        tags=("oracle",),
    ),
    QKernelEntry(
        id="quantum_counting",
        qkernel=quantum_counting,
        description="Quantum counting algorithm",
        param_names=("n", "m"),
        min_params={
            "m": 3,
        },
        tags=("oracle",),
    ),
    # --- Block encoding / QSVT ---
    *[
        QKernelEntry(
            id=f"block_encoding_{name}",
            qkernel=block_encoding_consumer,
            description=f"Resource consumer for the {name} block encoding",
            fixed_inputs={"encoding": encoding},
            tags=("block_encoding", "resource_estimation"),
        )
        for name, encoding in _BLOCK_ENCODING_CATALOG_CASES
    ],
    *[
        QKernelEntry(
            id=f"qsvt_{name}",
            qkernel=qsvt_consumer,
            description=f"QSVT resource consumer for the {name} block encoding",
            param_names=("phase_count",),
            min_params={"phase_count": 1},
            fixed_inputs={"encoding": encoding},
            tags=("qsvt", "resource_estimation"),
        )
        for name, encoding in _BLOCK_ENCODING_CATALOG_CASES
    ],
    # --- Arithmetic ---
    QKernelEntry(
        id="maj",
        qkernel=maj,
        description="MAJ gate used in quantum ripple-carry adders",
        tags=("arithmetic",),
    ),
    QKernelEntry(
        id="maj_loop",
        qkernel=maj_loop,
        description="MAJ gate with loops used in quantum ripple-carry adders",
        param_names=("n",),
        min_params={"n": 2},
        tags=("arithmetic",),
    ),
    QKernelEntry(
        id="uma_2_cnot",
        qkernel=uma_2_cnot,
        description="UMA gate with 2 CNOTs used in quantum ripple-carry adders",
        tags=("arithmetic",),
    ),
    QKernelEntry(
        id="uma_2_cnot_loop",
        qkernel=uma_2_cnot_loop,
        description="UMA gate with 2 CNOTs with loops used in quantum ripple-carry adders",
        param_names=("n",),
        min_params={"n": 2},
        tags=("arithmetic",),
    ),
    QKernelEntry(
        id="uma_3_cnot",
        qkernel=uma_3_cnot,
        description="UMA gate with 3 CNOTs used in quantum ripple-carry adders",
        tags=("arithmetic",),
    ),
    QKernelEntry(
        id="uma_3_cnot_loop",
        qkernel=uma_3_cnot_loop,
        description="UMA gate with 3 CNOTs with loops used in quantum ripple-carry adders",
        param_names=("n",),
        min_params={"n": 2},
        tags=("arithmetic",),
    ),
    QKernelEntry(
        id="simple_ripple_carry_adder_2_cnot",
        qkernel=simple_ripple_carry_adder_2_cnot,
        description="Simple ripple-carry adder using MAJ and UMA with 2 CNOTs",
        param_names=("n",),
        min_params={"n": 2},
        tags=("arithmetic",),
    ),
    QKernelEntry(
        id="simple_ripple_carry_adder_3_cnot",
        qkernel=simple_ripple_carry_adder_3_cnot,
        description="Simple ripple-carry adder using MAJ and UMA with 3 CNOTs",
        param_names=("n",),
        min_params={"n": 2},
        tags=("arithmetic",),
    ),
    QKernelEntry(
        id="draper_inplace_qc_adder",
        qkernel=draper_inplace_qc_adder,
        description="Draper's in-place quantum carry-lookahead adder with parametric n qubits and factor",
        param_names=("n",),
        min_params={"n": 2},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="ttk_adder",
        qkernel=ttk_adder,
        description="Takahashi-Tani-Kunihiro (TTK) adder with parametric n qubits",
        param_names=("n",),
        min_params={"n": 2},
        tags=("parametric",),
    ),
    QKernelEntry(
        id="cdkm_adder",
        qkernel=cdkm_adder,
        description="Cuccaro-Draper-Kutin-Moulton (CDKM) ripple-carry adder",
        param_names=("n",),
        min_params={"n": 4},
        tags=("arithmetic",),
    ),
]

QKERNEL_BY_ID: dict[str, QKernelEntry] = {e.id: e for e in QKERNEL_CATALOG}


# ============================================================
# Catalog query helpers
# ============================================================


def concrete_entries() -> list[QKernelEntry]:
    return [e for e in QKERNEL_CATALOG if not e.param_names]


def parametric_entries() -> list[QKernelEntry]:
    return [e for e in QKERNEL_CATALOG if e.param_names]


def entries_with_tag(tag: str) -> list[QKernelEntry]:
    return [e for e in QKERNEL_CATALOG if tag in e.tags]


_OFFSETS = [0, 1, 2, 5, 10, 100]


def concrete_values_for(entry: QKernelEntry) -> list[dict[str, int]]:
    """Generate concrete substitution dicts as cartesian product of per-param offsets."""
    names = list(entry.min_params.keys())
    per_param_values = [
        [entry.min_params[name] + offset for offset in _OFFSETS] for name in names
    ]
    return [dict(zip(names, combo)) for combo in itertools.product(*per_param_values)]
