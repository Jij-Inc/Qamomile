# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, encoding, resource-estimation]
# ---
#
# # Quantum Singular Value Transformation
#
# Quantum singular value transformation (QSVT) {cite:p}`10.1145/3313276.3316366`
# applies a polynomial to the singular values of a matrix embedded in a larger
# unitary. In this article, we construct a block encoding of a non-Hermitian
# $2\times2$ matrix, apply the degree-two Chebyshev polynomial with `qmc.qsvt`,
# and verify the transformed matrix and logical resource estimate.

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit,visualization]"

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Operator

import qamomile.circuit as qmc
from qamomile.linalg import PauliLCU
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Background
#
# ### Quantum signal processing (QSP)
#
# Quantum signal processing (QSP) {cite:p}`10.1103/PhysRevLett.118.010501`
# transforms a scalar value $x\in[-1,1]$ encoded in a single-qubit unitary by a
# polynomial. QSP is constructed as an alternating product of signal unitaries
# and phase rotations. In one common convention, the signal unitary is
#
# $$
# W(x)=e^{i\arccos(x)X}
# =\begin{pmatrix}
# x & i\sqrt{1-x^2}\\
# i\sqrt{1-x^2} & x
# \end{pmatrix}.
# $$
#
# Between signal unitaries, QSP applies controllable $Z$-axis phase rotations
#
# $$
# S(\phi)=e^{i\phi Z}
# =\operatorname{diag}(e^{i\phi},e^{-i\phi}).
# $$
#
# A phase sequence $\Phi=(\phi_0,\phi_1,\ldots,\phi_d)$ defines the alternating
# product
#
# $$
# U_\Phi(x)
# =S(\phi_0)\prod_{j=1}^{d}\left(W(x)S(\phi_j)\right).
# $$
#
# The upper-left entry $\langle0|U_\Phi(x)|0\rangle=P_\Phi(x)$ is a polynomial
# in $x$. A real target polynomial $p(x)$ can be represented as
# $\operatorname{Re}[P_\Phi(x)]$ when it satisfies the following conditions:
#
# - Its degree is at most $d$.
# - Its parity matches $d$: $p(-x)=(-1)^d p(x)$.
# - It is bounded on the signal interval: $|p(x)|\leq1$ for $x\in[-1,1]$.
#
# QSP design first approximates a target function $p(x)$ by such a polynomial
# and then classically synthesizes the corresponding phases. A degree-$d$
# transformation uses $d$ calls to the signal unitary, so the polynomial degree
# determines the query complexity.
#
# QSVT lifts this scalar polynomial transformation to each singular-value
# subspace of a block-encoded matrix.

# %% [markdown]
# ### Block encoding
#
# Before extending QSP to the singular-value subspaces of a matrix, we first
# explain how quantum computation handles non-unitary matrices. Block encoding
# embeds a given (non-unitary) matrix as part of a larger unitary matrix.
#
# Let $A$ act on an $n$-qubit system register. An $(\alpha,a)$ block encoding
# is a unitary $U$ acting on an additional $a$-qubit signal register such that
#
# $$
# (\langle 0|^{\otimes a}\otimes I)U
# (|0\rangle^{\otimes a}\otimes I)=\frac{A}{\alpha}.
# $$
#
# Here, $\alpha$ is a normalization factor satisfying
# $\alpha\geq\lVert A\rVert_2$. The encoded block is therefore $A/\alpha$, so
# QSVT acts on singular values in the interval $[0,1]$. The value of $\alpha$
# determines the scaling between the polynomial implemented by the circuit and
# the function ultimately applied to $A$.
#
# Block encoding converts a generally non-unitary matrix operation into part
# of a unitary evolution. Projecting the signal register onto
# $|0\rangle^{\otimes a}$ before and after $U$ selects the encoded matrix block.
# This interface lets QSVT manipulate the singular values without requiring
# the matrix itself to be unitary or Hermitian.
#
# Common ways to construct a block encoding include:
#
# - **Sparse-access oracles**: for a sparse matrix, coherent oracles provide the
#   locations and values of its nonzero entries. These oracles can be combined
#   into a unitary whose projected block is the normalized sparse matrix.
# - **Linear combination of unitaries (LCU)**: when
#   $A=\sum_j c_j U_j$ is expressed as a weighted sum of efficiently
#   implementable unitaries, an auxiliary register selects $U_j$ with an
#   amplitude determined by $c_j$. Projecting that register produces
#   $A/\sum_j|c_j|$.
#
# Block encodings can also be constructed from state-preparation procedures or
# data stored in quantum read-only memory.

# %% [markdown]
# ## Algorithm
#
# Let $U$ be an $(\alpha,a)$ block encoding of a matrix with singular value
# decomposition
#
# $$
# A=W\Sigma V^\dagger,
# $$
#
# and define the projector onto the signal subspace as
#
# $$
# \Pi=|0\rangle\!\langle0|^{\otimes a}\otimes I.
# $$
#
# A phase-modulated reflection about this subspace can be written as
#
# $$
# R_\Pi(\phi)=\exp\!\left(i\phi(2\Pi-I)\right).
# $$
#
# QSVT alternates applications of $U$ and $U^\dagger$ with these phase
# rotations. For a phase sequence
#
# $$
# \Phi=(\phi_0,\phi_1,\ldots,\phi_d),
# $$
#
# the resulting product acts independently on the two-dimensional invariant
# subspace associated with each singular value $\sigma_j/\alpha$. Choosing the
# phases appropriately makes the projected response equal to a polynomial
# $P(\sigma_j/\alpha)$.
#
# The parity of $P$ determines how the left and right singular-vector spaces
# are connected. With the singular value decomposition above, the singular
# value transform is
#
# $$
# P^{(\mathrm{SV})}(A/\alpha)=
# \begin{cases}
# W P(\Sigma/\alpha)V^\dagger, & P\text{ odd},\\
# V P(\Sigma/\alpha)V^\dagger, & P\text{ even}.
# \end{cases}
# $$
#
# A QSVT algorithm therefore consists of the following steps:
#
# 1. Choose a normalized block encoding of the input matrix.
# 2. Approximate the target scalar function by a bounded polynomial with the
#    required parity.
# 3. Classically synthesize a phase sequence for that polynomial.
# 4. Apply the alternating sequence of block encodings, their adjoints, and
#    projector phase rotations.
# 5. Project onto the signal subspace to obtain the desired singular-value
#    transformation.
#
# Exact phase offsets and product ordering depend on the adopted QSP/QSVT
# convention, so a phase sequence must be used with the convention for which it
# was synthesized.

# %% [markdown]
# ## Qamomile implementation
#
# ### Problem setup
#
# Consider the following non-Hermitian matrix:

# %%
matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

# %% [markdown]
# In the computational basis, this matrix maps $|1\rangle$ to $|0\rangle$ and
# maps $|0\rangle$ to the zero vector. It is neither unitary nor Hermitian, so
# it cannot be applied directly as a quantum gate. QSVT instead works with a
# unitary whose projected block is the normalized matrix $A/\alpha$. The small
# matrix keeps that projection easy to inspect while exercising the same
# block-encoding and QSVT interfaces used for larger matrices.

# %% [markdown]
# ### Constructing the block encoding
#
# LCU represents a matrix as
#
# $$
# A=\sum_j c_j U_j,
# \qquad
# \alpha=\sum_j|c_j|,
# $$
#
# where each $U_j$ is unitary. PREPARE creates the selector state
#
# $$
# \operatorname{PREPARE}|0\rangle
# =\sum_j\sqrt{\frac{|c_j|}{\alpha}}|j\rangle,
# $$
#
# and SELECT applies $e^{i\arg(c_j)}U_j$ conditioned on $|j\rangle$. The
# sequence PREPARE--SELECT--UNPREPARE then has $A/\alpha$ as its all-zero
# selector block.
#
# Here, `PauliLCU.from_matrix` chooses Pauli words for the unitaries $U_j$ and
# computes their complex coefficients. `qmc.pauli_lcu_block_encoding` builds
# the corresponding LCU block encoding. The returned descriptor records the
# normalization and the widths of the signal and system registers; Qamomile
# uses this metadata when the block encoding is inserted into a quantum
# kernel.

# %%
pauli_lcu = PauliLCU.from_matrix(matrix)
block_encoding = qmc.pauli_lcu_block_encoding(pauli_lcu)

print("Pauli terms:", pauli_lcu.num_terms)
print("normalization:", block_encoding.normalization)
print("signal qubits:", block_encoding.num_signal_qubits)
print("system qubits:", block_encoding.num_system_qubits)

assert pauli_lcu.num_terms == 2
assert np.isclose(block_encoding.normalization, 1.0)
assert block_encoding.num_signal_qubits == 1
assert block_encoding.num_system_qubits == 1

# %% [markdown]
# For this matrix, the Pauli expansion is
# $A=(X+iY)/2$. Therefore, the LCU has two terms and normalization
# $\alpha=|1/2|+|i/2|=1$. One signal qubit selects between those two terms,
# while one system qubit carries the state on which $A$ acts. The assertions
# above make these structural properties explicit before the descriptor is
# passed to QSVT.

# %% [markdown]
# ### Defining the quantum kernels
#
# The descriptor is a compile-time input to the quantum kernel. Allocating the
# registers from the descriptor metadata keeps the same quantum kernel reusable
# for other `LCUBlockEncoding` producers and register widths. The phase vector
# is also supplied at compile time in this tutorial, so Qamomile can specialize
# the alternating QSVT sequence before emitting a backend circuit.

# %%
@qmc.qkernel
def singular_value_transform(
    encoding: qmc.LCUBlockEncoding,
    phases: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply QSVT and measure the public registers.

    Args:
        encoding (qmc.LCUBlockEncoding): Exact LCU block encoding.
        phases (qmc.Vector[qmc.Float]): Projector phases in radians.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured signal and
            system registers.
    """
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = qmc.qsvt(signal, system, phases, encoding)
    return qmc.measure(signal), qmc.measure(system)

# %% [markdown]
# `signal` identifies the projected block of the encoding, and `system` holds
# the state transformed by the encoded matrix. `qmc.qsvt` applies the
# projector rotations and alternating block-encoding queries, introducing the
# clean auxiliary qubit needed by the projector rotations internally. The
# kernel measures only the public signal and system registers, so the same
# definition can be executed by a backend after the circuit-level validation
# below.

# %% [markdown]
# ### Constructing the QSVT circuit
#
# Bind the three projector phases at transpile time and construct the QSVT
# circuit. A sequence of three phases produces a degree-two transformation. In
# the convention used by `qmc.qsvt`, the phase sequence below implements
# $T_2(x)=2x^2-1$ through two block-encoding queries and three projector
# rotations. First, draw the quantum kernel directly with Qamomile's circuit
# visualizer.

# %%
qsvt_phases = [0.0, -math.pi / 2.0, math.pi / 2.0]
singular_value_transform.draw(
    encoding=block_encoding,
    phases=qsvt_phases,
    fold_loops=False,
)

# %%
transpiler = QiskitTranspiler()
qsvt_circuit = transpiler.to_circuit(
    singular_value_transform,
    bindings={
        "encoding": block_encoding,
        "phases": qsvt_phases,
    },
).remove_final_measurements(inplace=False)

# %% [markdown]
# The Qiskit conversion is used for the numerical check in the next section.
# Final measurements are removed only from `qsvt_circuit` so that its unitary
# can be inspected; they remain part of the original quantum kernel definition.

# %% [markdown]
# ## Result
#
# Extract the all-zero signal block of the complete QSVT unitary and compare it
# with $2A^\dagger A-I$.

# %%
def projected_signal_block(
    unitary: np.ndarray,
    signal_width: int,
    system_width: int,
) -> np.ndarray:
    """Extract the all-zero-signal block from a unitary matrix.

    Args:
        unitary (np.ndarray): Full circuit unitary.
        signal_width (int): Number of low-order signal qubits.
        system_width (int): Number of system qubits.

    Returns:
        np.ndarray: Projected matrix acting on the system register.
    """
    system_dimension = 1 << system_width
    projected_indices = np.arange(system_dimension) << signal_width
    return unitary[np.ix_(projected_indices, projected_indices)]


qsvt_unitary = np.asarray(Operator(qsvt_circuit).data)
actual_transform = projected_signal_block(
    qsvt_unitary,
    block_encoding.num_signal_qubits,
    block_encoding.num_system_qubits,
)
normalized_matrix = matrix / block_encoding.normalization
expected_transform = (
    2.0 * normalized_matrix.conj().T @ normalized_matrix
    - np.eye(1 << block_encoding.num_system_qubits)
)

np.testing.assert_allclose(
    actual_transform,
    expected_transform,
    rtol=0.0,
    atol=1e-8,
)
print("QSVT projected block:\n", np.round(actual_transform, decimals=8))

# %% [markdown]
# The QSVT implementation uses one clean auxiliary qubit for the projector
# rotations. It must return to zero, so a projected input has no amplitude in
# the auxiliary-one subspace.

# %%
logical_dimension = 1 << (
    block_encoding.num_signal_qubits + block_encoding.num_system_qubits
)
projected_input_indices = (
    np.arange(1 << block_encoding.num_system_qubits)
    << block_encoding.num_signal_qubits
)
auxiliary_one_indices = np.arange(logical_dimension) + logical_dimension

np.testing.assert_allclose(
    qsvt_unitary[np.ix_(auxiliary_one_indices, projected_input_indices)],
    0.0,
    rtol=0.0,
    atol=1e-8,
)

# %% [markdown]
# The two singular values of $A$ are $1$ and $0$. The polynomial maps them to
# $1$ and $-1$, respectively, exactly as observed in the projected block.

# %%
polynomial_x = np.linspace(-1.0, 1.0, 400)
polynomial_y = 2.0 * polynomial_x**2 - 1.0
singular_values = np.linalg.svd(normalized_matrix, compute_uv=False)
transformed_singular_values = 2.0 * singular_values**2 - 1.0

fig, axis = plt.subplots(figsize=(7, 4))
axis.plot(polynomial_x, polynomial_y, label=r"$T_2(x)=2x^2-1$")
axis.scatter(
    singular_values,
    transformed_singular_values,
    color="tab:red",
    zorder=3,
    label="singular values of $A/\\alpha$",
)
axis.set_xlabel("x")
axis.set_ylabel("$T_2(x)$")
axis.set_title("Degree-two singular-value transformation")
axis.grid(alpha=0.3)
axis.legend()
plt.show()

# %% [markdown]
# ## Resource estimation
#
# ### Query complexity
#
# The QSVT circuit construction by Gilyén et al.
# {cite:p}`10.1145/3313276.3316366` (Section 3.2 and Figure 1) uses $U$ and
# $U^\dagger$ a total of $d$ times and applies $d+1$ projector phase rotations
# for a degree-$d$ transformation. The query complexity to the block encoding
# is therefore $\Theta(d)$. The required degree depends on the target function,
# approximation interval, allowed error, and normalization $\alpha$.
#
# If $U$ and $U^\dagger$ have the same cost $G_U$, and one projector phase
# rotation costs $G_\Pi(a)$, the gate cost excluding measurement and other
# post-processing can be organized as
#
# $$
# G_{\mathrm{QSVT}}\simeq dG_U+(d+1)G_\Pi(a).
# $$
#
# The first term counts block-encoding calls, and the second counts projector
# phase rotations. This is a cost breakdown of the cited circuit construction,
# not a universal backend gate count: $G_U$ depends on whether the block
# encoding uses LCU, sparse oracles, or another construction, while $G_\Pi(a)$
# depends on the decomposition of multi-controlled gates.
#
# ### Logical resources for this example
#
# Qamomile's resource estimator can specialize the same quantum kernel with the
# block encoding and phase vector. The estimate is expressed in Qamomile's
# target-neutral logical gate basis.

# %%
estimate = singular_value_transform.estimate_resources(
    inputs={
        "encoding": block_encoding,
        "phases": qsvt_phases,
    }
)

print("logical qubits:", estimate.qubits)
print("logical gates:", estimate.gates.total)
print("logical depth:", estimate.depth.depth)
print("estimate quality:", estimate.quality.value)

expected_qubits = (
    block_encoding.num_signal_qubits
    + block_encoding.num_system_qubits
    + 1
)
assert estimate.qubits == expected_qubits
assert estimate.gates.total == 25
assert estimate.depth.depth == 27
assert estimate.quality.value == "upper_bound"

# %% [markdown]
# The three logical qubits are one signal qubit, one system qubit, and one
# reusable projector auxiliary. Here $d=2$, so the theoretical count gives one
# call to $U$, one call to $U^\dagger$, and three projector rotations. The
# logical gate count and depth printed above expand those operations using the
# selected Pauli LCU block encoding and Qamomile's target-neutral gate basis.
# This provides a comparison on a common abstraction level.
#
# A backend transpiler may further decompose multi-controlled gates and
# rotations according to its gate set, connectivity, and synthesis settings.
# Such a count is useful for a specified target, but it should not be compared
# directly with the block-encoding query count without accounting for these
# implementation choices.

# %% [markdown]
# ## Summary
#
# In this article, we:
#
# - constructed a Pauli LCU block encoding of a non-Hermitian matrix;
# - applied the degree-two QSVT phase sequence and confirmed the even
#   singular-value transform $2(A/\alpha)^\dagger(A/\alpha)-I$; and
# - derived the degree-dependent block-encoding query count and estimated the
#   target-neutral logical width, gate count, and depth of the same Qamomile
#   quantum kernel.
#
# Qamomile implements the QSVT circuit for a caller-supplied projector-phase
# sequence. Phase synthesis, approximation-domain selection, and conversion
# from other QSP conventions remain separate classical preprocessing steps.
