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
# # Quantum Phase Estimation (QPE) Tutorial
#
# This tutorial explains how to implement the Quantum Phase Estimation (QPE) algorithm using Qamomile.
#
# ## Implementing Quantum Phase Estimation from Scratch
# First, let's implement QPE using Qamomile's basic quantum gates.
#
# ### Inverse Quantum Fourier Transform (IQFT)
#
# The Inverse Quantum Fourier Transform is an important part of the QPE algorithm. Below is the implementation of IQFT.
#

# %%
import math
import qamomile.circuit as qmc


@qmc.qkernel
def iqft(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Inverse Quantum Fourier Transform (IQFT) on a vector of qubits."""
    n = qubits.shape[0]
    for j in qmc.range(n // 2):
        qubits[j], qubits[n - j - 1] = qmc.swap(qubits[j], qubits[n - j - 1])
    for j in qmc.range(n):
        for k in qmc.range(j):
            angle = -math.pi / (2 ** (j - k))
            qubits[j], qubits[k] = qmc.cp(qubits[j], qubits[k], angle)
        qubits[j] = qmc.h(qubits[j])
    return qubits



# %% [markdown]
# ### Defining the Phase Gate
# In this tutorial, we use the Phase Gate as the target for QPE. The Phase Gate is defined as follows:
# $$P(\theta)|1\rangle = e^{i\theta}|1\rangle$$
# Here, $|1\rangle$ is an eigenstate, and $e^{i\theta}$ is the corresponding eigenvalue.
# We will estimate this eigenvalue using QPE.

# %%
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: float, iter: int) -> qmc.Qubit:
    """Phase gate: P(θ)|1⟩ = e^{iθ}|1⟩"""
    for _ in qmc.range(iter):
        q = qmc.p(q, theta)
    return q

# %%
# QPE Implementation
@qmc.qkernel
def qpe(phase: float) -> qmc.Vector[qmc.Bit]:
    phase_register = qmc.qubit_array(3, name="phase_reg")
    target = qmc.qubit(name="target")

    target = qmc.x(target)  # |0⟩ → |1⟩

    controlled_phase_gate = qmc.controlled(phase_gate)

    # Superposition preparation
    n = phase_register.shape[0]
    for i in qmc.range(n):
        phase_register[i] = qmc.h(phase_register[i])

    # Apply QPE algorithm
    # controlled() API: (control, target, **params) -> (control_out, target_out)
    # Apply phase gate 2^i times for control qubit i
    for i in qmc.range(3):
        phase_register[i], target = controlled_phase_gate(phase_register[i], target, theta=phase, iter=2**i)
    iqft(phase_register)

    bits = qmc.measure(phase_register)

    return bits

# %% [markdown]
# ### Running QPE with Different Quantum SDKs
#
# Qamomile supports multiple quantum SDKs. Select your preferred backend:
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: sdk
#
# ```python
# from qamomile.qiskit import QiskitTranspiler
#
# transpiler = QiskitTranspiler()
# executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})
#
# job = executable.sample(transpiler.executor(), shots=1024)
# sample_result = job.result()
# ```
#
# :::
# :::{tab-item} Quri-Parts
# :sync: sdk
#
# ```python
# from qamomile.quri_parts import QuriPartsCircuitTranspiler
#
# transpiler = QuriPartsCircuitTranspiler()
# executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})
#
# # Requires quri-parts-qulacs for simulation
# job = executable.sample(transpiler.executor(), shots=1024)
# sample_result = job.result()
# ```
#
# :::
# :::{tab-item} PennyLane
# :sync: sdk
#
# ```python
# from qamomile.pennylane import PennylaneTranspiler
#
# transpiler = PennylaneTranspiler()
# executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})
#
# job = executable.sample(transpiler.executor(), shots=1024)
# sample_result = job.result()
# ```
#
# :::
# :::{tab-item} CUDA-Q
# :sync: sdk
#
# ```{note}
# CUDA-Q is only available on Linux systems with NVIDIA GPUs.
# ```
#
# ```python
# from qamomile.cudaq import CudaqTranspiler
#
# transpiler = CudaqTranspiler()
# executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})
#
# job = executable.sample(transpiler.executor(), shots=1024)
# sample_result = job.result()
# ```
#
# :::
# ::::
#
# The following code executes QPE using Qiskit (the primary example):

# %%
from qamomile.qiskit import QiskitTranspiler


transpiler = QiskitTranspiler()
executable = transpiler.transpile(qpe, bindings={"phase": math.pi / 2})

job = executable.sample(transpiler.executor(), shots=1024)
sample_result = job.result()

# Decode results
num_bits = 3
for bits, count in sample_result.results:
    phase_estimate = sum(bit * (1 / (2 ** (i + 1))) for i, bit in enumerate(reversed(bits)))
    print(f"Measured bits: {bits}, Count: {count}, Estimated phase: {phase_estimate:.4f}")


# %% [markdown]
# We have successfully implemented and run QPE. The Executor configured in Qiskit's Transpiler uses Qiskit-Aer simulator by default, but you can implement your own Executor and pass it to Qamomile's Transpiler to run on other backends.
# Let's check what Qiskit quantum circuit was generated.

# %%
qiskit_circuit = executable.get_first_circuit()
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# As you can see, the circuit we implemented is generated as a Qiskit quantum circuit.
# Next, let's implement a similar QPE using the qpe() function provided by Qamomile.

# %% [markdown]
# ## Quantum Phase Estimation Using Qamomile's qpe() Function
# Using the predefined qpe() function allows for a more concise QPE implementation.
#
# **Important**: `qmc.qpe()` automatically performs `U^(2^k)` iterations internally,
# so the unitary should be defined for **a single application only**.

# %%
# Simple phase gate for qmc.qpe() (single application only)
@qmc.qkernel
def p_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    """Simple phase gate: P(θ)|1⟩ = e^{iθ}|1⟩"""
    return qmc.p(q, theta)

@qmc.qkernel
def qpe_3bit(phase: float) -> qmc.Float:
    q_phase = qmc.qubit_array(3, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)  # |0⟩ → |1⟩
    # Use p_gate (qmc.qpe() internally repeats 2^k times)
    phase_q: qmc.QFixed = qmc.qpe(target, q_phase, p_gate, theta=phase)
    return qmc.measure(phase_q)

# %% [markdown]
# Simply prepare a register to store the phase, initialize the target state, and call the qpe() function to implement QPE.
# The measurement result is returned as QFixed type, so we use the measure() function to convert it to Float type. The measure function automatically performs decoding based on the type passed to it.
#
# ### Running Simulation with Qiskit
# Let's run on the Qiskit simulator and check the results as before.

# %%
transpiler = QiskitTranspiler()
test_phase = math.pi / 2  # θ = π/2, expected output ≈ 0.25 (since θ/(2π) = 0.25)
executable = transpiler.transpile(qpe_3bit, bindings={"phase": test_phase})

executor = transpiler.executor()
job = executable.sample(executor)
result = job.result()
for value, count in result.results:
    print(f"Measured value: {value}, Count: {count}")

# %% [markdown]
# We have confirmed that we can implement and run QPE using Qamomile's qpe() function. In this way, Qamomile uses QFixed type to handle quantum fixed-point numbers, which simplifies the implementation of quantum algorithms.
# Also, in this case, simply using the measure() function to convert QFixed to Float automatically performs the decoding.
# Let's also check the quantum circuit generated when using the qpe function.

# %%
qiskit_circuit = executable.get_first_circuit()
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# This is also generated as a Qiskit quantum circuit.
# In Qamomile, if the backend supports certain operations, the quantum circuit is generated to use those operations directly whenever possible.
# For example, since IQFT is natively supported in Qiskit, the IQFT part within QPE is also generated directly as an IQFT gate.
