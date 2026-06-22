# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, primitive, resource-estimation]
# ---
#
# # Quantum Phase Estimation as an FTQC Building Block
#
# Quantum phase estimation (QPE) is the control-flow pattern behind many
# fault-tolerant algorithms: prepare an eigenstate, apply controlled powers of
# a unitary, and decode the phase with an inverse QFT. This notebook shows the
# smallest useful Qamomile implementation, checks the measured phase, and links
# the same precision choice to a symbolic resource estimate.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import math

import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.estimator.algorithmic import estimate_qpe
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Background
#
# If $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$, QPE estimates the binary
# digits of $\phi$. The example below uses the one-qubit phase gate
# $P(\theta)|1\rangle = e^{i\theta}|1\rangle$. Setting $\theta=\pi/2$ gives
# $\phi=\theta/(2\pi)=1/4$, exactly representable with three fractional bits.
#
# The target qubit is prepared in $|1\rangle$, an eigenstate of the phase gate.
# The counting register is measured as a `QFixed` value, so Qamomile decodes
# the measured bit string into a `Float`.

# %%
theta = math.pi / 2
expected_phase = sp.Rational(1, 4)
counting_qubits = 3

assert expected_phase == sp.Rational(1, 4)

# %% [markdown]
# ## Implementation
#
# `qmc.qpe` takes the target eigenstate, a counting register, and a Qamomile
# unitary kernel. The controlled powers are generated inside the standard
# library helper. Qamomile keeps the inverse QFT as a composite operation so
# backends can emit it natively or decompose it later.

# %%
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    q = qmc.p(q, theta)
    return q


@qmc.qkernel
def estimate_phase(theta: qmc.Float) -> qmc.Float:
    counting = qmc.qubit_array(counting_qubits, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)

    phase = qmc.qpe(target, counting, phase_gate, theta=theta)
    return qmc.measure(phase)

# %%
block = estimate_phase.build(theta=theta)
composite_types = [
    op.gate_type
    for op in block.operations
    if isinstance(op, CompositeGateOperation)
]

print([gate_type.value for gate_type in composite_types])
assert CompositeGateType.IQFT in composite_types

# %% [markdown]
# ## Result
#
# The phase is exactly representable by the three-qubit counting register, so
# the simulator returns one decoded value.

# %%
transpiler = QiskitTranspiler()
executable = transpiler.transpile(estimate_phase, bindings={"theta": theta})
result = executable.sample(transpiler.executor(), shots=256).result()

print(result.results)
assert len(result.results) == 1
measured_phase, count = result.results[0]
assert measured_phase == sp.Float(expected_phase)
assert count == 256

# %% [markdown]
# ## Resource Estimate
#
# Circuit execution shows the small exact example. For FTQC planning, the more
# important question is how the cost scales with system size and precision.
# `estimate_qpe` records that relationship symbolically, without committing to
# a backend circuit decomposition.

# %%
resource_estimate = estimate_qpe(
    n_system=1,
    precision=counting_qubits,
    hamiltonian_norm=1,
    method="qubitization",
)

print("qubits:", resource_estimate.qubits)
print("total gates:", resource_estimate.gates.total)

assert resource_estimate.qubits == 4
assert sp.simplify(resource_estimate.gates.total - 8) == 0

# %% [markdown]
# ## Summary
#
# In this notebook, we:
#
# - Built a minimal QPE kernel with `qmc.qpe` and measured a `QFixed` phase.
# - Verified that the inverse QFT stays as an abstract composite operation in
#   the IR before backend emission.
# - Connected the same precision parameter to a symbolic QPE resource estimate.
