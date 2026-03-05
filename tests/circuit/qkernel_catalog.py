from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
    ResourceMetadata,
)


@dataclass(frozen=True)
class QKernelEntry:
    """A catalog entry for a qkernel circuit."""

    id: str
    qkernel: QKernel
    description: str
    param_names: tuple[str, ...] = ()
    min_params: dict[str, int] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


# ============================================================
# Shared helpers (internal)
# ============================================================


@qmc.qkernel
def _all_h(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply H gate to all qubits in a vector."""
    n = qs.shape[0]
    for i in qmc.range(n):
        qs[i] = qmc.h(qs[i])
    return qs


@qmc.qkernel
def _all_x(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply X gate to all qubits in a vector."""
    n = qs.shape[0]
    for i in qmc.range(n):
        qs[i] = qmc.x(qs[i])
    return qs


def __emit_oracle(
    *qubits: qmc.Vector[qmc.Qubit] | qmc.Qubit,
    name: str,
    resource_metadata: ResourceMetadata,
) -> None:
    """Emit a stub oracle CompositeGateOperation directly to the tracer.

    Uses the QPE-style factory pattern (see qpe.py:99-110) to create a
    CompositeGateOperation, bypassing _StubCompositeGate.__call__ which
    requires individual Qubit unpacking.

    Accepts any mix of Vector[Qubit] and Qubit arguments.
    """
    tracer = get_current_tracer()
    operands = [q.value for q in qubits]
    op = CompositeGateOperation(
        operands=operands,
        results=[],
        gate_type=CompositeGateType.CUSTOM,
        custom_name=name,
        num_target_qubits=0,
        has_implementation=False,
        resource_metadata=resource_metadata,
    )
    tracer.add_operation(op)


def _over_oracle(
    *qubits: qmc.Vector[qmc.Qubit] | qmc.Qubit,
    name: str,
    resource_metadata: ResourceMetadata,
) -> tuple[qmc.Vector[qmc.Qubit] | qmc.Qubit, ...]:
    __emit_oracle(*qubits, name=name, resource_metadata=resource_metadata)
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


@qmc.composite_gate(
    stub=True,
    name="controlled_oracle",
    num_qubits=1,
    num_controls=1,
    resource_metadata=ResourceMetadata(
        query_complexity=1,
        total_gates=1,
        two_qubit_gates=1,
    ),
)
def _controlled_oracle():
    pass


@qmc.composite_gate(
    stub=True,
    name="one_qubit_oracle",
    num_qubits=1,
    resource_metadata=ResourceMetadata(query_complexity=1),
)
def _one_qubit_oracle():
    pass


@qmc.composite_gate(
    stub=True,
    name="two_qubit_oracle",
    num_qubits=2,
    resource_metadata=ResourceMetadata(query_complexity=1, two_qubit_gates=1),
)
def _two_qubit_oracle():
    pass


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

    qs = _all_h(qs)

    (qs[0], qs[1]) = _two_qubit_oracle(qs[0], qs[1])

    qs[0] = qmc.h(qs[0])

    return qmc.measure(qs[0])


@qmc.qkernel
def deutsch_jozsa(n: qmc.UInt) -> qmc.Bit:
    qs = qmc.qubit_array(n, name="qs")
    target = qmc.qubit(name="target")

    qs = _all_h(qs)
    target = qmc.x(target)
    target = qmc.h(target)

    (qs, target) = _over_oracle(
        qs,
        target,
        name="deutsch_jozsa_oracle",
        resource_metadata=ResourceMetadata(query_complexity=1),
    )  # type: ignore

    qs = _all_h(qs)  # type: ignore

    return qmc.measure(qs)  # type: ignore


@qmc.qkernel
def _simon(
    qs1: qmc.Vector[qmc.Qubit], qs2: qmc.Vector[qmc.Qubit]
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    qs1 = _all_h(qs1)
    qs1, qs2 = _over_oracle(
        qs1,
        qs2,
        name="simon_oracle",
        resource_metadata=ResourceMetadata(query_complexity=1),
    )  # type: ignore
    qs1 = _all_h(qs1)
    return qs1, qs2


@qmc.qkernel
def simon(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    qs1 = qmc.qubit_array(n, name="qs1")
    qs2 = qmc.qubit_array(n, name="qs2")
    qs1, qs2 = _simon(qs1, qs2)
    bits = qmc.measure(qs1)
    return bits


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
    controlled_u = qmc.controlled(_phase)

    qs = _all_h(qs)

    for k in qmc.range(n):
        qs[k], target = controlled_u(qs[k], target, power=2**k, theta=theta)

    qs = _iqft(qs)

    return qs


@qmc.composite_gate(
    stub=True,
    name="controlled_u",
    num_qubits=1,
    num_controls=1,
    resource_metadata=ResourceMetadata(
        query_complexity=1,
        total_gates=1,
        two_qubit_gates=1,
    ),
)
def _controlled_u():
    pass


@qmc.qkernel
def stub_oracle_qpe(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    target = qmc.qubit(name="target")

    qs = _all_h(qs)

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

    qs = _all_h(qs)

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
    qs: qmc.Vector[qmc.Qubit], target_qubit: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    n = qs.shape[0]
    num_ancillas = n - 1
    ancillas = qmc.qubit_array(num_ancillas, name="ancillas")

    qs[0], qs[1], ancillas[0] = qmc.ccx(qs[0], qs[1], ancillas[0])
    for i in qmc.range(0, num_ancillas - 1):
        qs[i + 2], ancillas[i], ancillas[i + 1] = qmc.ccx(
            qs[i + 2], ancillas[i], ancillas[i + 1]
        )

    target_qubit = qmc.h(target_qubit)
    ancillas[num_ancillas - 1], target_qubit = qmc.cx(
        ancillas[num_ancillas - 1], target_qubit
    )
    target_qubit = qmc.h(target_qubit)

    for i in qmc.range(num_ancillas - 2, -1, -1):
        qs[i + 2], ancillas[i], ancillas[i + 1] = qmc.ccx(
            qs[i + 2], ancillas[i], ancillas[i + 1]
        )
    qs[0], qs[1], ancillas[0] = qmc.ccx(qs[0], qs[1], ancillas[0])

    return qs, target_qubit


@qmc.qkernel
def network_decomposition_controlled_z(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    qubit = qmc.qubit(name="target")
    qs, qubit = _network_decomposition_controlled_z(qs, qubit)
    return qs


@qmc.qkernel
def __z(q: qmc.Qubit) -> qmc.Qubit:
    return qmc.z(q)


@qmc.qkernel
def _naive_multi_controlled_z(
    qs: qmc.Vector[qmc.Qubit], target_q: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    n = qs.shape[0]
    multi_controlled_z = qmc.controlled(__z, num_controls=n)
    qs, target_q = multi_controlled_z(qs, target_q)
    return qs, target_q


@qmc.qkernel
def naive_multi_controlled_z(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    qs = qmc.qubit_array(n, name="qs")
    target_q = qmc.qubit(name="target")
    qs, target_q = _naive_multi_controlled_z(qs, target_q)
    return qs, target_q


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
        resource_metadata=ResourceMetadata(query_complexity=1),
    )  # type: ignore
    # Apply the diffusion operator,
    # which can be implemented as H + X + multi-controlled Z + X + H.
    qs = _all_h(qs)
    qs = _all_x(qs)
    unpacked_qs, target_q = qmc.unpack_qubits(
        qs, num_unpacked=2, num_elements=[qs.shape[0] - 1, 1]
    )
    unpacked_qs, target_q = _network_decomposition_controlled_z(unpacked_qs, target_q)
    qs = qmc.pack_qubits(unpacked_qs, target_q)
    qs = _all_x(qs)
    qs = _all_h(qs)

    return qs, q


@qmc.qkernel
def _grover_network_decomposition(
    qs: qmc.Vector[qmc.Qubit], q: qmc.Qubit, n_iters: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    # Initialise all the qubits.
    qs = _all_h(qs)
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
    qs = _all_h(qs)
    q = qmc.x(q)
    q = qmc.h(q)

    # Apply the diffusion operator (inversion about the mean).
    for _ in qmc.range(n_iters):
        # Call the oracle.
        (qs, q) = _over_oracle(
            qs,
            q,
            name="grover_oracle",
            resource_metadata=ResourceMetadata(query_complexity=1),
        )  # type: ignore
        # Apply the diffusion operator,
        # which can be implemented as H + X + multi-controlled Z + X + H.
        qs = _all_h(qs)
        qs = _all_x(qs)
        unpacked_qs, target_q = qmc.unpack_qubits(
            qs, num_unpacked=2, num_elements=[qs.shape[0] - 1, 1]
        )
        unpacked_qs, target_q = _naive_multi_controlled_z(unpacked_qs, target_q)
        qs = qmc.pack_qubits(unpacked_qs, target_q)
        qs = _all_x(qs)
        qs = _all_h(qs)

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
    controlled_grover = qmc.controlled(
        _grover_operator_network_decomposition, num_controls=1
    )

    qs1 = _all_h(qs1)
    qs2 = _all_h(qs2)
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
        param_names=("n", "m"),
        min_params={"n": 1, "m": 2},
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
        description="Hadamard test with a stub controlled oracle",
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
        id="stub_oracle_qpe",
        qkernel=stub_oracle_qpe,
        description="QPE with stub oracle (controlled-U as black-box)",
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
