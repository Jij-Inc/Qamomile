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
# # Quantum Teleportation
#
# In this notebook, we use quantum teleportation to learn a minimal set of Qamomile language features.
#
# - Define a quantum program with `@qm.qkernel`
# - Reuse quantum operations by calling one qkernel from another
# - Measure a single-qubit state of type `Q[Bit]` to obtain a classical `Bit` on each shot
# - Use a measurement result in a Python `if` statement to write a dynamic quantum circuit
# - Compile for Qiskit and run the circuit with `qm.run()`
#
# Quantum teleportation is a protocol for transferring the state of a single qubit.
#
# Suppose Alice and Bob share the Bell pair $\frac{|00\rangle+|11\rangle}{\sqrt2}$, and Alice wants to transfer an unknown state $|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$.
#
# Alice performs operations on the state to be transferred and on her half of the Bell pair. She then measures her two qubits and sends the two classical measurement bits to Bob.
# Based on the outcome $(m_0,m_1)$ received from Alice, Bob applies $Z^{m_0}X^{m_1}$ to recover $|\psi\rangle$.
#
# Let us use Qamomile to reproduce quantum teleportation as a quantum circuit.

# %%
# Install Qamomile and the packages used in this notebook.
# # !pip install "qamomile[qiskit]"

# %%
import qamomile as qm

# %% [markdown]
# ## 1. Basic Structure of a Qamomile Program
#
# In Qamomile, the part that constructs a quantum circuit is separate from the part that runs the circuit on a quantum SDK.
#
# - `qamomile` is an API for writing quantum SDK-independent quantum programs with qubits, gates, measurements, and related operations. In this notebook, we use the short name `qm`.
# - A Python function decorated with `@qm.qkernel` becomes a **quantum kernel**, or qkernel, that represents a quantum computation. Qamomile analyzes the function body and converts it into a quantum SDK-independent intermediate representation.
# - `qm.compile` converts the quantum kernel into a form that the selected target can execute.
#
# Qamomile types such as `qm.Q[qm.Bit]` and `qm.Shot[qm.Bit]` are used as annotations for the arguments and return values of a quantum kernel.
# Although a quantum kernel looks like an ordinary Python function, its body describes how to construct a quantum circuit.
# A `Q[Bit]` represents a single-qubit state. Measuring it gives `Shot[Bit]`, a classical bit with value 0 or 1 on each shot.

# %% [markdown]
# ## 2. Create a Bell Pair
#
# First prepare $H|0\rangle=|+\rangle$ on the first qubit, and then apply a CX gate:
#
# $$
# |00\rangle\longmapsto
# \frac{|00\rangle+|11\rangle}{\sqrt2}.
# $$
#
# If we measure either qubit by itself, 0 and 1 occur at random. However, the two measurement results always agree.
# We assume that Alice holds the first qubit of this Bell pair and Bob holds the second.
#
# The following code gives a small example that creates a Bell pair.
# A Qamomile quantum value is a **handle** that identifies a quantum state for use in circuit operations.
# A gate consumes the input handle and returns an updated handle for the same state.
# Because the consumed handle cannot be used in subsequent operations, capture the return value and update the variable, as in `alice = qm.h(alice)`.
# This follows Qamomile’s **linear ownership** rule: each quantum handle is consumed once.
#
# `qm.prepare(0)` initializes each qubit in $|0\rangle$. We apply `qm.h` to Alice's qubit, then use `qm.concat(alice, bob)` to combine the two qubits into a `Q[Bits[2]]` state.
# `qm.cx(pair, control=0, target=1)` uses Alice's qubit as the control and Bob's qubit as the target.
#
# `bell_pair` is a state-preparation qkernel that returns a two-qubit state before measurement.
# By placing the measurements in a separate qkernel, `measure_bell_pair`, we can reuse the prepared Bell pair later in the teleportation circuit.
# `qm.measure(state)` measures the qubits in the computational basis and returns a tuple of bits per shot.
# After measurement, quantum gates such as `H` and `CX` cannot be applied to the measured state.
# The return annotation of `measure_bell_pair`, `qm.Shot[qm.Bits[2]]`, shows that each shot produces two measured bits.


# %%
@qm.qkernel
def bell_pair() -> qm.Q[qm.Bits[2]]:
    """Prepare and return a Bell pair without measuring it."""
    alice = qm.prepare(0)
    bob = qm.prepare(0)
    alice = qm.h(alice)
    pair = qm.concat(alice, bob)
    return qm.cx(pair, control=0, target=1)


@qm.qkernel
def measure_bell_pair() -> qm.Shot[qm.Bits[2]]:
    """Prepare a Bell pair and measure both qubits."""
    return qm.measure(bell_pair())


# %% [markdown]
# ### Run a Quantum Kernel on Qiskit
#
# `measure_bell_pair` is a quantum SDK-independent quantum program, so defining it does not run it.
# Execution has three steps:
#
# 1. Instantiate the Qiskit Aer endpoint with `qm.qiskit.Aer()`.
# 2. Use `qm.compile(qm.specialize(measure_bell_pair), to=endpoint.target)` to compile the qkernel for that endpoint.
# 3. Use `qm.run(artifact, on=endpoint, shots=...)` to run the circuit the specified number of times.
# Because compilation and execution are separate, we can compile the same quantum kernel for different quantum SDKs. We can also run a compiled parameterized circuit repeatedly with different parameter values.
# We reuse the endpoint for the remaining circuit.
#
# `shots=1024` specifies the number of circuit executions, and `.result` retrieves the completed result.
# Each item in `bell_result.counts` maps an outcome of the form `(bit0, bit1)` to the number of times that outcome occurred.
# In this example, only `00` and `11` appear, and the sum of all counts is `1024`.

# %%
shots = 1024
endpoint = qm.qiskit.Aer()

bell_executable = qm.compile(qm.specialize(measure_bell_pair), to=endpoint.target)
bell_result = qm.run(bell_executable, on=endpoint, shots=shots, seed=42).result

print(dict(bell_result.counts))
assert sum(bell_result.counts.values()) == shots
assert all(a == b for a, b in bell_result.counts)


# %% [markdown]
# ## 3. Use a Measurement Result in an `if` Statement
#
# We now consider teleporting $|+\rangle=H|0\rangle=(|0\rangle+|1\rangle)/\sqrt2$ to Bob.
#
# Alice applies `CX(message, alice)` and `H(message)` to her two qubits, `message` and `alice`.
# These operations map the Bell basis to the computational basis. The two measurements that follow are therefore equivalent to a Bell measurement.
#
# `m_message` and `m_alice` are classical `Bit` values that store the measurement results.
# These `if` statements are not ordinary Python branches evaluated once on the host.
# Instead, they are compiled into dynamic control that runs on the target quantum SDK according to the result of each shot.
# In other words, the body of `if m_alice:` does not select a branch in advance in Python. It applies `X` only in shots where the measurement result obtained during circuit execution is 1.
# A circuit with this type of mid-circuit branching is called a **dynamic quantum circuit**.
# To run such a circuit on hardware, the target device must support mid-circuit measurements and classically controlled gates.
#
# In `teleport_plus_state`, we call `bell_pair()` from the previous section and reuse the Bell-pair preparation.
# A qkernel can call another qkernel just like an ordinary Python function can call another function.
# Finally, we apply a Hadamard gate to `bob` to map the X basis to the Z basis. If the teleported state is $|+\rangle$, the measurement result is always 0.


# %%
@qm.qkernel
def teleport_plus_state() -> tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bit], qm.Shot[qm.Bit]]:
    """Teleport a plus state and measure Bob in the X basis."""
    message = qm.h(qm.prepare(0))
    state = qm.concat(message, bell_pair())

    state = qm.cx(state, control=0, target=1)
    state = qm.h(state, target=0)
    alice_qubits, bob = qm.split_at(state, 2)
    alice_bits = qm.measure(alice_qubits)
    m_message, m_alice = alice_bits[0], alice_bits[1]

    if m_alice:
        bob = qm.x(bob)
    if m_message:
        bob = qm.z(bob)

    bob_bits = qm.measure(qm.h(bob))
    return m_message, m_alice, bob_bits[0]


# %% [markdown]
# `teleport_plus_state` calls `bell_pair()` and reuses Bell-pair preparation as a qkernel.
# Because it returns three measured bits, every outcome from `qm.run()` also has three entries: `(m_message, m_alice, bob)`.

# %%
teleport_executable = qm.compile(qm.specialize(teleport_plus_state), to=endpoint.target)
teleport_result = qm.run(teleport_executable, on=endpoint, shots=shots, seed=42).result

for outcome, count in teleport_result.counts.items():
    print(f"Bell measurement={outcome[:2]}, Bob(X basis)={outcome[2]}: {count}")

assert sum(teleport_result.counts.values()) == shots
assert all(bob == 0 for _, _, bob in teleport_result.counts)


# %% [markdown]
# Bob's measurement result is always 0, while Alice's measurement results are random.
# The information in the teleported quantum state therefore does not appear in the two classical bits alone. Bob can recover the state only by combining the shared Bell pair with the conditional corrections.

# %% [markdown]
# ## Summary
#
# - Use `@qm.qkernel` to describe quantum operations.
# - Gates return updated qubit handles, so reassign their return values to variables.
# - `measure()` turns a `Q[Bit]` state into `Shot[Bit]`, with one classical bit per shot.
# - A Python `if` conditioned on a measured `Bit` is converted into dynamic control.
# - Reuse quantum operations by calling one qkernel from another.
# - Run a quantum kernel that returns measurement values with `qm.run()`.
