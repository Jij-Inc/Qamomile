r"""
This module implements the Quantum Approximate Optimization Algorithm (QAOA) converter
for the Qamomile framework :cite:`farhi2014quantum`.
The parameterized state :math:`|\\vec{\\beta},\\vec{\gamma}\\rangle` of :math:`p`-layer QAOA is defined as:

.. math::
    |\\vec{\\beta},\\vec{\gamma}\\rangle = U(\\vec{\\beta},\\vec{\gamma})|0\\rangle^{\otimes n} = e^{-i\\beta_{p-1} H_M}e^{-i\gamma_{p-1} H_P} \cdots e^{-i\\beta_0 H_M}e^{-i\gamma_0 H_P} H^{\otimes n}|0\\rangle^{\otimes n}

where :math:`H_P` is the cost Hamiltonian, :math:`H_M` is the mixer Hamiltonian and :math:`\gamma_l` and :math:`\\beta_l` are the variational parameters.
The 2 :math:`p` variational parameters are optimized classically to minimize the expectation value :math:`\langle \\vec{\\beta},\\vec{\gamma}|H_P|\\vec{\\beta},\\vec{\gamma}\\rangle`.

This module provides functionality to convert optimization problems which written by `jijmodeling`
into QAOA circuits (:math:`U(\\vec{\\beta},\\vec{\gamma})`), construct cost Hamiltonians (:math:`H_P`), and decode quantum computation results.

The `QAOAConverter` class extends the `QuantumConverter` base class, specializing in
QAOA-specific operations such as ansatz circuit generation and result decoding.


Key Features:
- Generation of QAOA ansatz circuits
- Construction of cost Hamiltonians for QAOA
- Decoding of quantum computation results into classical optimization solutions


Note: This module requires `jijmodeling` for problem representation.

.. bibliography::
    :filter: docname in docnames

"""

import ommx.v1

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import QuantumExecutor
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.converters.converter import QuantumConverter
from qamomile.core.converters.utils import is_close_zero


@qmc.qkernel
def ising_cost(
    q: qmc.Vector[qmc.Qubit],
    edges: qmc.Matrix[qmc.UInt],  # (|E|, 2) matrix
    Jij: qmc.Vector[qmc.Float],  # (|E|,) vector
    hi: qmc.Vector[qmc.Float],  # (|V|,) vector,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    num_e = edges.shape[0]
    for e in qmc.range(num_e):
        i = edges[e, 0]
        j = edges[e, 1]
        angle = 2 * Jij[e] * gamma
        q[i], q[j] = qmc.rzz(q[i], q[j], angle)
    n = hi.shape[0]
    for i in qmc.range(n):
        angle = 2 * hi[i] * gamma
        q[i] = qmc.rz(q[i], angle)
    return q


@qmc.qkernel
def x_mixer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    n = q.shape[0]
    for i in qmc.range(n):
        angle = 2 * beta
        q[i] = qmc.rx(q[i], angle)
    return q


@qmc.qkernel
def qaoa_state(
    edges: qmc.Matrix[qmc.UInt],  # (|E|, 2) matrix
    Jij: qmc.Vector[qmc.Float],  # (|E|,) vector
    hi: qmc.Vector[qmc.Float],  # (|V|,) vector,
    p: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],  # (p,) vector
    betas: qmc.Vector[qmc.Float],  # (p,) vector
) -> qmc.Vector[qmc.Qubit]:
    n = hi.shape[0]
    q = qmc.qubit_array(n, name="qaoa_state")
    for iter in qmc.range(p):
        gamma = gammas[iter]
        beta = betas[iter]
        q = ising_cost(q, edges, Jij, hi, gamma)
        q = x_mixer(q, beta)
    return q



