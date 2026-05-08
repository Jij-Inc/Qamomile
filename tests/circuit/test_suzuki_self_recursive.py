"""Numerical convergence check for the self-recursive Suzuki–Trotter
``@qkernel``.  Verifies that the transpiler's unroll loop produces a
circuit whose fidelity error scales as the expected textbook order
(S2: ``dt^4``, S4: ``dt^8``) for a single-qubit Rabi Hamiltonian.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile as qk_transpile
from qiskit_aer import AerSimulator
from scipy.linalg import expm

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.qiskit import QiskitTranspiler

OMEGA = 1.2
OMEGA_DRIVE = 0.8
T = 1.5

HZ = 0.5 * OMEGA * qm_o.Z(0)
HX = 0.5 * OMEGA_DRIVE * qm_o.X(0)
HS = [HZ, HX]


@qmc.qkernel
def s2_step(
    q: qmc.Vector[qmc.Qubit], Hs: qmc.Vector[qmc.Observable], dt: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    q = qmc.pauli_evolve(q, Hs[0], 0.5 * dt)
    q = qmc.pauli_evolve(q, Hs[1], dt)
    q = qmc.pauli_evolve(q, Hs[0], 0.5 * dt)
    return q


@qmc.qkernel
def suzuki_trotter(
    order: qmc.UInt,
    q: qmc.Vector[qmc.Qubit],
    Hs: qmc.Vector[qmc.Observable],
    dt: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    if order == 2:
        q = s2_step(q, Hs, dt)
    else:
        p = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
        w = 1.0 - 4.0 * p
        q = suzuki_trotter(order - 2, q, Hs, p * dt)
        q = suzuki_trotter(order - 2, q, Hs, p * dt)
        q = suzuki_trotter(order - 2, q, Hs, w * dt)
        q = suzuki_trotter(order - 2, q, Hs, p * dt)
        q = suzuki_trotter(order - 2, q, Hs, p * dt)
    return q


@qmc.qkernel
def rabi_suzuki(
    order: qmc.UInt,
    Hs: qmc.Vector[qmc.Observable],
    dt: qmc.Float,
    n_steps: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = suzuki_trotter(order, q, Hs, dt)
    return qmc.measure(q)


def _exact_state():
    X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
    H_mat = 0.5 * OMEGA * Z_mat + 0.5 * OMEGA_DRIVE * X_mat
    return expm(-1j * T * H_mat) @ np.array([1.0, 0.0], dtype=complex)


def _statevector(circuit):
    stripped = QuantumCircuit(*circuit.qregs)
    for instr in circuit.data:
        if instr.operation.name not in ("measure", "save_statevector"):
            stripped.append(instr)
    stripped = qk_transpile(
        stripped,
        basis_gates=["u", "cx", "rx", "ry", "rz", "h", "p", "sx", "x", "y", "z"],
    )
    stripped.save_statevector()
    sim = AerSimulator(method="statevector")
    return np.asarray(sim.run(stripped).result().get_statevector())


@pytest.mark.parametrize("order,expected_slope", [(2, 4.0), (4, 8.0)])
def test_suzuki_recursive_convergence_slope(order, expected_slope):
    sv_exact = _exact_state()
    tr = QiskitTranspiler()
    dts = []
    errs = []
    for N in (4, 8, 16):
        exe = tr.transpile(
            rabi_suzuki,
            bindings={"order": order, "Hs": HS, "dt": T / N, "n_steps": N},
        )
        sv = _statevector(exe.compiled_quantum[0].circuit)
        # ``abs(vdot)`` can drift slightly above 1.0 from float rounding,
        # which would make ``err`` negative and ``log(err)`` NaN.
        overlap = np.clip(abs(np.vdot(sv_exact, sv)), 0.0, 1.0)
        err = max(1.0 - overlap, np.finfo(float).tiny)
        dts.append(T / N)
        errs.append(err)
    # Fit all three points.  Platform-specific BLAS / transpile decisions
    # (notably macOS ARM) can nudge accumulated gate-level rounding error
    # by a few ULPs per step; with ~125 gates per S4 step that is enough
    # to move the measured slope by a noticeable fraction.  The
    # tolerance is wide enough to absorb that drift while still failing
    # loudly if the convergence order collapses.
    dts_arr = np.asarray(dts)
    errs_arr = np.asarray(errs)
    slope = np.polyfit(np.log(dts_arr), np.log(errs_arr), 1)[0]
    assert abs(slope - expected_slope) < 1.5, (
        f"order={order}: slope={slope:.2f}, expected {expected_slope}"
    )
