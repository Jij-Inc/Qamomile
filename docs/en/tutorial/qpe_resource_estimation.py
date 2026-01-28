# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # Resource Estimation for Quantum Phase Estimation
#
# This tutorial demonstrates Qamomile's algebraic resource estimation capabilities using Quantum Phase Estimation (QPE) as an example. We'll show how to estimate quantum resources using symbolic expressions with SymPy.
#
# **Prerequisites:** Basic understanding of quantum circuits (Hadamard, CNOT, controlled operations)
#

# %%
import math
import qamomile.circuit as qmc
from qamomile.circuit.estimator import estimate_resources
from qamomile.circuit.estimator.algorithmic import estimate_qpe
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## Section 1: Introduction to Algebraic Resource Estimation
#
# ### Why Algebraic Resource Estimation?
#
# Traditional resource estimation requires implementing and analyzing specific circuits. Qamomile's algebraic approach allows you to:
#
# 1. **Estimate resources symbolically** without fixing parameter values
# 2. **Explore design space** by varying parameters algebraically
# 3. **Compare algorithms** using theoretical complexity formulas
# 4. **Plan hardware requirements** with parametric analysis
#
# ### Two Approaches in Qamomile
#
# 1. **Circuit-based**: Pass a `@qkernel` function to `estimate_resources()` - analyzes actual circuit IR
# 2. **Algorithmic**: Use theoretical formulas like `estimate_qpe()` - based on complexity analysis from research papers
#
# Let's start with a simple example to understand the API.

# %% [markdown]
# ### Warm-up: Bell State Estimation
#
# First, let's estimate resources for a Bell state to understand how `estimate_resources()` works.

# %%
@qmc.qkernel
def bell_state() -> qmc.Vector[qmc.Qubit]:
    """Create a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2"""
    q = qmc.qubit_array(2, name="q")
    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    return q

# Simply pass the qkernel's block to estimate_resources
est = estimate_resources(bell_state.block)

print("Bell State Resource Estimate:")
print(f"  Qubits: {est.qubits}")
print(f"  Total gates: {est.gates.total}")
print(f"  Single-qubit gates: {est.gates.single_qubit}")
print(f"  Two-qubit gates: {est.gates.two_qubit}")
print(f"  Clifford gates: {est.gates.clifford_gates}")
print(f"  Circuit depth: {est.depth.total_depth}")

# %% [markdown]
# As expected: 2 qubits, 2 gates (H + CX), both Clifford, depth 2.
#
# ### Parametric Estimation with Symbolic Variables
#
# Now let's try a parametric circuit - a GHZ state with variable size.

# %%
@qmc.qkernel
def ghz_state(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Create n-qubit GHZ state |0...0⟩ + |1...1⟩"""
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    for i in qmc.range(n - 1):
        q[i], q[i+1] = qmc.cx(q[i], q[i+1])
    return q

# Estimate with symbolic parameter n
est_ghz = estimate_resources(ghz_state.block)

print("\nGHZ State Resource Estimate (symbolic):")
print(f"  Qubits: {est_ghz.qubits}")
print(f"  Total gates: {est_ghz.gates.total}")
print(f"  Two-qubit gates: {est_ghz.gates.two_qubit}")
print(f"  Circuit depth: {est_ghz.depth.total_depth}")

# %% [markdown]
# Notice that the results contain the symbol `n`! This is the power of algebraic estimation.
#
# We can substitute concrete values to get specific estimates:

# %%
# Substitute n=10
est_ghz_10 = est_ghz.substitute(n=10)

print("\nGHZ State with n=10:")
print(f"  Qubits: {est_ghz_10.qubits}")
print(f"  Total gates: {est_ghz_10.gates.total}")
print(f"  Two-qubit gates: {est_ghz_10.gates.two_qubit}")

# Substitute n=100
est_ghz_100 = est_ghz.substitute(n=100)

print("\nGHZ State with n=100:")
print(f"  Qubits: {est_ghz_100.qubits}")
print(f"  Total gates: {est_ghz_100.gates.total}")
print(f"  Two-qubit gates: {est_ghz_100.gates.two_qubit}")

# %% [markdown]
# Now let's apply this to a more complex algorithm: Quantum Phase Estimation.

# %% [markdown]
# ## Section 2: Implementing QPE from Basic Components
#
# Quantum Phase Estimation (QPE) estimates the phase (eigenvalue) of a unitary operator. Given a unitary $U$ with eigenstate $|\psi\rangle$ such that:
#
# $$U|\psi\rangle = e^{2\pi i \theta}|\psi\rangle$$
#
# QPE estimates the phase $\theta$ to $m$ bits of precision.
#
# ### Algorithm Overview
#
# 1. Prepare $m$ counting qubits in superposition
# 2. Apply controlled-$U^{2^k}$ operations
# 3. Apply inverse QFT (IQFT)
# 4. Measure to get phase estimate
#
# Let's implement each component from scratch.

# %% [markdown]
# ### Step 1: Inverse Quantum Fourier Transform (IQFT)
#
# IQFT is the final step of QPE that converts the phase information into measurable basis states.

# %%
@qmc.qkernel
def iqft(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Inverse Quantum Fourier Transform"""
    n = qubits.shape[0]
    # Swap qubits (reverse order)
    for j in qmc.range(n // 2):
        qubits[j], qubits[n - j - 1] = qmc.swap(qubits[j], qubits[n - j - 1])
    # Apply inverse QFT gates
    for j in qmc.range(n):
        for k in qmc.range(j):
            angle = -math.pi / (2 ** (j - k))
            qubits[j], qubits[k] = qmc.cp(qubits[j], qubits[k], theta=angle)
        qubits[j] = qmc.h(qubits[j])
    return qubits

# Let's estimate IQFT resources with symbolic n
@qmc.qkernel
def iqft_n(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """IQFT with n qubits"""
    qubits = qmc.qubit_array(n, name="q")
    return iqft(qubits)

est_iqft = estimate_resources(iqft_n.block)
print("IQFT Resource Estimate (symbolic n):")
print(f"  Qubits: {est_iqft.qubits}")
print(f"  Total gates: {est_iqft.gates.total}")
print(f"  Two-qubit gates: {est_iqft.gates.two_qubit}")
print(f"  Depth: {est_iqft.depth.total_depth}")

# %% [markdown]
# IQFT requires $O(n^2)$ gates due to the nested loops.

# %% [markdown]
# ### Step 2: Define the Target Unitary
#
# For this tutorial, we use a simple phase gate as the target unitary:
#
# $$P(\theta)|1\rangle = e^{i\theta}|1\rangle$$
#
# This has eigenvalue $e^{i\theta}$ for eigenstate $|1\rangle$.

# %%
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: float, iter: int) -> qmc.Qubit:
    """Phase gate: P(θ)|1⟩ = e^{iθ}|1⟩"""
    for i in qmc.range(iter):
        q = qmc.p(q, theta)
    return q

# %% [markdown]
# ### Step 3: Implement QPE from Scratch
#
# Now we implement the full QPE algorithm using basic gates.

# %%
@qmc.qkernel
def qpe_manual(theta: float, m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """QPE with m-bit precision, implemented from basic gates"""
    # Allocate qubits
    counting = qmc.qubit_array(m, name="counting")
    target = qmc.qubit(name="target")

    # Prepare target in eigenstate |1⟩
    target = qmc.x(target)

    # Step 1: Prepare counting qubits in superposition
    for i in qmc.range(m):
        counting[i] = qmc.h(counting[i])

    # Step 2: Apply controlled-U^(2^k) operations
    # For phase gate, U^k just multiplies the phase by k
    controlled_phase = qmc.controlled(phase_gate)
    for i in qmc.range(m):
        # Apply U^(2^i) = P(2^i * theta)
        iterations = 2 ** i
        counting[i], target = controlled_phase(counting[i], target, theta=theta, iter=iterations)

    # Step 3: Apply IQFT
    counting = iqft(counting)

    # Step 4: Measure
    bits = qmc.measure(counting)
    return bits


est_qpe_manual = estimate_resources(qpe_manual.block)
print("\nManual QPE Resource Estimate (symbolic m):")
print(f"  Qubits: {est_qpe_manual.qubits}")
print(f"  Total gates: {est_qpe_manual.gates.total}")
print(f"  Two-qubit gates: {est_qpe_manual.gates.two_qubit}")
print(f"  Depth: {est_qpe_manual.depth.total_depth}")


# %% [markdown]
# ### Observations
#
# - **Qubits**: $m + 1$ (m counting qubits + 1 target qubit) - scales linearly
# - **Gates**: Grows rapidly with precision due to $2^m$ controlled operations
# - **Depth**: Also increases significantly with precision

# %% [markdown]
# ## Section 3: Using Qamomile's Built-in QPE
#
# Qamomile provides a built-in `qmc.qpe()` function for convenience. Let's compare with our manual implementation.

# %%

@qmc.qkernel                                                                                                                                                   
def simple_phase_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:                                                                                                
    """qmc.qpe()用のシンプルな位相ゲート: P(θ)|1⟩ = e^{iθ}|1⟩                                                                                                  
                                                                                                                                                               
    注意: このバージョンは位相ゲートを1回だけ適用します。                                                                                                      
    qmc.qpe()と一緒に使用すると、繰り返し（2^k回）はpowerパラメータで処理されます。                                                                            
    """                                                                                                                                                        
    return qmc.p(q, theta) 

@qmc.qkernel
def qpe_builtin(theta: float, n: qmc.UInt) -> qmc.Float:
    """QPE using Qamomile's built-in qpe function with 8-bit precision"""
    counting = qmc.qubit_array(n, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)  # Prepare eigenstate

    # qmc.qpe() handles the controlled operations and IQFT internally
    phase = qmc.qpe(target, counting, simple_phase_gate, theta=theta)
    return qmc.measure(phase)

est_builtin = estimate_resources(qpe_builtin.block)
est_builtin = est_builtin.simplify()

print("\nBuilt-in QPE (m=8) Resource Estimate:")
print(f"  Qubits: {est_builtin.qubits}")
print(f"  Total gates: {est_builtin.gates.total}")
print(f"  Two-qubit gates: {est_builtin.gates.two_qubit}")
print(f"  Depth: {est_builtin.depth.total_depth}")

# %%
