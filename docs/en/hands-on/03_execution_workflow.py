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
# tags: [tutorial]
# ---
#
# # Execution Workflow
#
# This notebook uses a small VQE (Variational Quantum Eigensolver) example to walk through the workflow of converting a qkernel to a supported quantum SDK/IR and executing it on the target. We use Qiskit as the target below.
#
# Given a target Hamiltonian $H$, VQE repeatedly executes a parameterized quantum circuit $\ket{\psi(\theta)}$ and updates its parameters using the execution results. The algorithm seeks to lower the Hamiltonian's energy (expectation value) $\bra{\psi(\theta)}H\ket{\psi(\theta)}$ by finding suitable parameters $\theta$. See [Peruzzo et al. (2014)](https://doi.org/10.1038/ncomms5213).
#
# This notebook focuses on how to use qkernels to compute expectation values and measure bit strings, without going into parameter optimization methods.

# %%
# Install Qamomile and the packages used in this notebook.
# # !pip install "qamomile[qiskit,visualization]" matplotlib scipy

# %%
from __future__ import annotations

import math

import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

import qamomile as qm
from qamomile.qiskit import bind_circuit, load_circuit

# %% [markdown]
# ## 1. Choose an Execution Endpoint
#
# We compile a qkernel for an execution target. We use the Qiskit Aer simulator as the endpoint for this example and set a seed when running it for reproducibility.

# %%
endpoint = qm.qiskit.Aer()


# %% [markdown]
# ## 2. Define a Parameterized qkernel
#
# First, we define a parameterized qkernel. We prepare an $n$-qubit state and apply $R_Y(\theta_i)$ to each qubit:
#
# $$
# \ket{\psi_n(\theta)}
# = \bigotimes_{i} R_Y(\theta_i)\ket{0}
# = \bigotimes_{i}\left( \cos\frac{\theta_i}{2}\ket{0} +\sin\frac{\theta_i}{2}\ket{1} \right)
# $$
#
# Similarly, we define a qkernel that measures this quantum state. We use the `measure()` function to perform the measurement.


# %%
@qm.qkernel
def parameterized_state(
    n: qm.Const[qm.Int],
    thetas: qm.Param[qm.Vector[qm.Real, n]],  # noqa: F821
) -> qm.Q[qm.Bits[n]]:  # noqa: F821
    """Prepare n qubits with a separate rotation angle for each qubit without measuring them."""
    q = qm.qubits(n)
    for i in range(n):
        q = qm.ry(q, thetas[i], target=i)
    return q


@qm.qkernel
def sample_state(
    n: qm.Const[qm.Int],
    thetas: qm.Param[qm.Vector[qm.Real, n]],  # noqa: F821
) -> qm.Shot[qm.Bits[n]]:  # noqa: F821
    """Measure every qubit of the variational state in the computational basis."""
    q = parameterized_state(n=n, thetas=thetas)
    return qm.measure(q)


# %% [markdown]
# We compile the `sample_state` qkernel into an object that can be executed with Qiskit. An executable entry point takes classical inputs and returns classical outputs.
#
# For simplicity, we set $n = 1$:
#
# $$
# \ket{\psi_1} = \cos\frac{\theta}{2}\ket{0} +\sin\frac{\theta}{2}\ket{1}
# $$
#
# We fix the number of qubits `n` with `qm.specialize` before compilation. The annotation `qm.Param` keeps `thetas` as a runtime parameter so we can update its values during execution.

# %%
sample_program = qm.specialize(sample_state, n=1)
sample_executable = qm.compile(sample_program, to=endpoint.target)
qiskit_circuit = load_circuit(sample_executable)
print(f"Qubits: {qiskit_circuit.num_qubits}")
print(
    "Runtime parameters:", [slot.name for slot in sample_executable.parameter_abi.slots]
)
qiskit_circuit.draw()

# %% [markdown]
# Now we execute the compiled Qiskit quantum circuit. We use `qm.run()` to measure a bit string from the qubits. We supply `thetas` through `parameters` at execution time. Here, we execute the circuit with an initial angle of $\pi / 2$.
#
# `qm.run()` returns a record whose `.result` contains the measurement results.

# %%
shots = 1024
initial_thetas = [math.pi / 2]

initial_parameters = {"thetas": initial_thetas}
sample_run = qm.run(
    sample_executable,
    on=endpoint,
    shots=shots,
    seed=42,
    parameters=initial_parameters,
)

print(type(sample_run))

# %%
initial_sample = sample_run.result
initial_probabilities = {
    bits: count / initial_sample.shots for bits, count in initial_sample.counts.items()
}

print("Initial counts:", dict(initial_sample.counts))
print("Initial probabilities:", initial_probabilities)
assert sum(initial_sample.counts.values()) == shots

# %% [markdown]
# The result obtained from `qm.run()` has a `counts` mapping from bit strings to counts. Dividing each count by the number of shots gives its empirical probability.

# %% [markdown]
# ## 3. Evaluate the Energy
#
# Next, we update the parameters of the quantum state we defined to lower the expectation value of a given Hamiltonian. To compute the exact expectation on a local simulator, we take a copy of the compiled circuit without its final measurements.

# %%
state_circuit = qiskit_circuit.remove_final_measurements(inplace=False)

# %% [markdown]
# The Hamiltonian is a sum of Pauli operators. We use Qiskit's `SparsePauliOp` to represent it for the statevector calculation.

# %%
hamiltonian = SparsePauliOp.from_list([("Z", -1.0), ("X", -1.0)])
print(f"Hamiltonian: {hamiltonian}")

# %% [markdown]
# We bind the same runtime parameters to the state-preparation circuit and evaluate the Hamiltonian against its simulated statevector.


# %%
def run_state(theta_values):
    """Evaluate the Hamiltonian on the shared variational state."""
    invocation = qm.Invocation.create(parameters={"thetas": tuple(theta_values)})
    circuit = bind_circuit(state_circuit, sample_executable, invocation)
    return float(Statevector(circuit).expectation_value(hamiltonian).real)


# %%
initial_energy = run_state(initial_thetas)
print(f"Initial energy: {initial_energy:+.6f}")


# %% [markdown]
# ## 4. Update Parameters Using the Compiled Circuit
#
# Here, we repeatedly call `run_state` while updating the parameters to lower the expectation value. For every evaluation, we bind the new parameters to the same compiled circuit. We use SciPy's `minimize` function to update the parameters.

# %%
history = []


def energy(theta_values):
    """Bind one trial angle and return the energy to the optimizer."""
    value = run_state(theta_values)
    history.append((theta_values, value))
    return value


optimized = minimize(
    energy,
    x0=initial_thetas,
    method="COBYLA",
    options={"maxiter": 50, "rhobeg": 0.5, "tol": 1e-6},
)
optimized_parameters = {"thetas": tuple(map(float, optimized.x))}

print(f"Optimized thetas: {optimized_parameters['thetas']}")
print(f"Energy: {initial_energy:+.6f} -> {optimized.fun:+.6f}")
print("Energy evaluations:", optimized.nfev)
print("Optimizer status:", optimized.message)
assert math.isclose(optimized.fun, -math.sqrt(2), abs_tol=1e-6)

# %% [markdown]
# For this circuit, $\langle Z\rangle=\cos\theta$ and $\langle X\rangle=\sin\theta$, so we obtain analytically
#
# $$
# \bra{\psi_1(\theta)} H \ket{\psi_1(\theta)}
# =-\cos\theta-\sin\theta,\qquad \theta_\star=\pi/4\pmod{2\pi},\qquad E_\star=-\sqrt2
# $$
#
# This confirms that `minimize` has found the optimal parameter.

# %% [markdown]
# ## 5. Compare the Optimization Process and Results
#
# Finally, we examine how the energy changes during optimization and compare the probability distribution using the optimized parameters with the distribution using the initial values. Since `sample_state` has already been compiled, we can call `qm.run()` using the object we already have.

# %%
optimized_sample = qm.run(
    sample_executable,
    on=endpoint,
    shots=shots,
    seed=42,
    parameters=optimized_parameters,
).result
optimized_probabilities = {
    bits: count / optimized_sample.shots
    for bits, count in optimized_sample.counts.items()
}
print("Optimized counts:", dict(optimized_sample.counts))
print("Optimized probabilities:", optimized_probabilities)

# %% [markdown]
# Using the parameter $\theta_* = \pi / 4$ we obtained, we have analytically
#
# $$
# \ket{\psi_1(\pi / 4)} = \cos\frac{\pi}{8}\ket{0} +\sin\frac{\pi}{8}\ket{1}
# $$
#
# We can see that the samples appear to be correct.
#
# We plot these below.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(
    range(1, len(history) + 1),
    [value for _, value in history],
    marker=".",
    color="#2673B8",
    label="Evaluated energy",
)
axes[0].axhline(-math.sqrt(2), color="#555555", linestyle="--", label="Exact minimum")
axes[0].set(xlabel="Energy evaluation", ylabel="Energy", title="Variational loop")
axes[0].legend()

axes[1].bar(
    [-0.18, 0.82],
    [initial_probabilities.get((bit,), 0.0) for bit in (0, 1)],
    width=0.36,
    label="Before",
    color="#2673B8",
)
axes[1].bar(
    [0.18, 1.18],
    [optimized_probabilities.get((bit,), 0.0) for bit in (0, 1)],
    width=0.36,
    label="After",
    color="#E58B38",
)
axes[1].set(
    xticks=[0, 1],
    xticklabels=["0", "1"],
    ylim=(0, 1),
    xlabel="Measurement outcome",
    ylabel="Measured probability",
    title=f"Before and after ({shots} shots each)",
)
axes[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# In this notebook, we learned about compilation and execution through a small VQE example. We used `Const` and `qm.specialize` to fix the circuit structure and `Param` to leave angles adjustable at runtime. We also computed exact expectation values with a Qiskit statevector and measured qubits with `qm.measure()` and `qm.run()`.
