# ---
# jupyter:
#   jupytext:
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
# # Superposition and Entanglement
#
# Previous tutorials introduced the type system and quantum gates.
# This tutorial shifts the focus from *gate mechanics* to the *concepts* that
# make quantum computing fundamentally different from classical computing:
# **superposition**, **interference**, and **entanglement**.
#
# ## What We Will Learn
# - Superposition: the quantum coin flip
# - Phase and the $|+\rangle$ / $|-\rangle$ states
# - Quantum interference: how phases control outcomes
# - Entanglement with CNOT and Bell states
# - GHZ states: multi-qubit entanglement

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. Superposition: The Quantum Coin Flip
#
# A classical bit is always either 0 or 1.
# A quantum bit (qubit) can exist in a **superposition** of both states
# at the same time.
#
# The Hadamard gate (H) transforms a definite $|0\rangle$ state into:
#
# $$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} \equiv |+\rangle$$
#
# When we measure this state, we get 0 or 1 with equal probability.
# But crucially, before measurement the qubit is *not* secretly 0 or 1 --
# it genuinely holds both possibilities simultaneously.
# This is what distinguishes a "quantum coin flip" from a classical random coin.


# %%
@qmc.qkernel
def superposition() -> qmc.Bit:
    """Create a superposition state with the H gate."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    return qmc.measure(q)


superposition.draw()

# %%
exec_super = transpiler.transpile(superposition)
result_super = exec_super.sample(transpiler.executor(), shots=1000).result()

print("=== Superposition Measurement Results ===")
for value, count in result_super.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### Interpreting the Results
#
# The measurement gives roughly 50% `0` and 50% `1`.
# Each individual run is random, but the statistics are predictable.
#
# Key insight: until measured, the qubit "simultaneously" holds both states.
# At the moment of measurement it "collapses" to one definite value.
# Which outcome appears is governed by the probability amplitudes --
# numbers that encode not just probabilities but also *phases*,
# which will become important in the next section.

# %% [markdown]
# ## 2. Phase and the $|+\rangle$ / $|-\rangle$ States
#
# The superposition state we just created is called $|+\rangle$.
# There is a second superposition state, $|-\rangle$, which differs by a
# sign:
#
# - $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$ -- from $H|0\rangle$
# - $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$ -- from $H|1\rangle$
#
# Both states give the same measurement statistics: 50% chance of 0 and
# 50% chance of 1.  Yet they are *physically distinct* quantum states.
#
# The difference is in the **relative phase** -- the minus sign in $|-\rangle$.
# This sign is invisible to a single measurement, but it becomes
# detectable through interference, as we will see shortly.
#
# Let's create $|-\rangle$ by first flipping $|0\rangle$ to $|1\rangle$
# with the X gate, then applying H.


# %%
@qmc.qkernel
def minus_state() -> qmc.Bit:
    """Create |-> = H|1>."""
    q = qmc.qubit(name="q")
    q = qmc.x(q)  # |0> -> |1>
    q = qmc.h(q)  # |1> -> |->
    return qmc.measure(q)


minus_state.draw()

# %%
exec_plus = transpiler.transpile(superposition)
exec_minus = transpiler.transpile(minus_state)

result_plus = exec_plus.sample(transpiler.executor(), shots=1000).result()
result_minus = exec_minus.sample(transpiler.executor(), shots=1000).result()

print("=== |+> State (from section 1) ===")
for value, count in result_plus.results:
    print(f"  {value}: {count}")

print("\n=== |-> State ===")
for value, count in result_minus.results:
    print(f"  {value}: {count}")

print("\nBoth are approximately 50/50, but they are different quantum states!")

# %% [markdown]
# ### Revealing the Difference
#
# If $|+\rangle$ and $|-\rangle$ give identical measurement results, how can we
# tell them apart?  Apply the Hadamard gate **again**.  Since $H^2 = I$:
#
# - $H|+\rangle = H \cdot H|0\rangle = |0\rangle$ -- always measures **0**
# - $H|-\rangle = H \cdot H|1\rangle = |1\rangle$ -- always measures **1**
#
# The same gate (H) applied to two states that "look the same" in direct
# measurement produces completely opposite, deterministic outcomes.


# %%
@qmc.qkernel
def reveal_plus() -> qmc.Bit:
    """H|+> = |0>: always measures 0."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)  # |0> -> |+>
    q = qmc.h(q)  # |+> -> |0>
    return qmc.measure(q)


reveal_plus.draw()


# %%
@qmc.qkernel
def reveal_minus() -> qmc.Bit:
    """H|-> = |1>: always measures 1."""
    q = qmc.qubit(name="q")
    q = qmc.x(q)  # |0> -> |1>
    q = qmc.h(q)  # |1> -> |->
    q = qmc.h(q)  # |-> -> |1>
    return qmc.measure(q)


reveal_minus.draw()

# %%
exec_rp = transpiler.transpile(reveal_plus)
exec_rm = transpiler.transpile(reveal_minus)

result_rp = exec_rp.sample(transpiler.executor(), shots=1000).result()
result_rm = exec_rm.sample(transpiler.executor(), shots=1000).result()

print("=== Reveal |+> with H ===")
for value, count in result_rp.results:
    print(f"  {value}: {count}")

print("\n=== Reveal |-> with H ===")
for value, count in result_rm.results:
    print(f"  {value}: {count}")

# %% [markdown]
# Even though $|+\rangle$ and $|-\rangle$ give identical 50/50 results when
# measured directly, they respond differently to the *same* gate (H).
# The relative phase -- that minus sign -- is a real, physical property
# of the quantum state.
#
# The second H gate causes the amplitudes to add (for $|+\rangle$) or
# cancel (for $|-\rangle$) in different ways.  This is **interference** --
# the mechanism we explore in detail next.

# %% [markdown]
# ## 3. Quantum Interference
#
# Interference is the mechanism that makes quantum algorithms powerful.
# Probability amplitudes are complex numbers that can reinforce
# (**constructive interference**) or cancel (**destructive interference**).
#
# The reveal experiment above is already an example: the double-Hadamard
# circuit $H \cdot H = I$ works because amplitudes interfere.  Let's trace
# through it step by step:
#
# 1. Start with $|0\rangle$.
# 2. First H creates $\frac{|0\rangle + |1\rangle}{\sqrt{2}}$.
# 3. Second H acts on each component:
#    - $H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$
#    - $H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$
# 4. Adding them: the $|1\rangle$ amplitudes cancel ($+1/2$ and $-1/2$),
#    while the $|0\rangle$ amplitudes reinforce ($+1/2$ and $+1/2$).
# 5. Result: $|0\rangle$ with certainty.
#
# The cancellation of the $|1\rangle$ component is destructive interference,
# and the reinforcement of $|0\rangle$ is constructive interference.
# This is the core mechanism behind quantum algorithms:
# arrange the phases so that wrong answers cancel out
# and correct answers reinforce.

# %% [markdown]
# ### Phase-Controlled Interference
#
# The double-Hadamard is a special case of a more general pattern.
# If we insert a **phase rotation** (RZ) between the two H gates,
# we can smoothly tune the interference from fully constructive to
# fully destructive.
#
# The circuit $H \to RZ(\theta) \to H$ produces:
#
# - $\theta = 0$ : same as $H \cdot H = I$ -- always **0**
# - $\theta = \pi$ : full phase flip -- always **1**
# - $\theta = \pi/2$ : partial phase -- back to 50/50


# %%
@qmc.qkernel
def phase_interference(theta: qmc.Float) -> qmc.Bit:
    """H -> RZ(theta) -> H: phase controls the outcome."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    q = qmc.rz(q, theta)
    q = qmc.h(q)
    return qmc.measure(q)


phase_interference.draw()

# %%
for label, angle in [("0", 0.0), ("pi", math.pi), ("pi/2", math.pi / 2)]:
    exec_pi = transpiler.transpile(phase_interference, bindings={"theta": angle})
    result_pi = exec_pi.sample(transpiler.executor(), shots=1000).result()

    print(f"theta = {label}:")
    for value, count in result_pi.results:
        percentage = count / 1000 * 100
        print(f"  {value}: {count} ({percentage:.1f}%)")
    print()

# %% [markdown]
# The RZ gate rotates the phase without changing the measurement
# probabilities of a single qubit.  But sandwiched between two H gates,
# the phase directly controls which outcome the interference favours.
#
# This is the template for quantum algorithms: **encode information in
# phases, then use interference to extract it**.

# %% [markdown]
# ## 4. Entanglement with CNOT
#
# Superposition makes a single qubit richer than a classical bit.
# **Entanglement** creates correlations between multiple qubits
# that have no classical counterpart.
#
# The recipe is simple: put one qubit in superposition, then
# apply a CNOT gate.
#
# - **CNOT** (Controlled-NOT): flips the target qubit when the control is $|1\rangle$.
#
# When the control qubit is in superposition, the CNOT spreads that
# superposition across both qubits, creating entanglement.
#
# The resulting state is the **Bell state** $|\Phi^+\rangle$:
#
# $$|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$
#
# This state has a remarkable property: measuring one qubit instantly
# determines the other.  If we get 0 on the first qubit, the second
# is guaranteed to be 0.  If we get 1, the second is guaranteed to be 1.
# Yet neither qubit has a definite value before measurement.


# %%
@qmc.qkernel
def bell_state() -> tuple[qmc.Bit, qmc.Bit]:
    """Create Bell state |Phi+> = (|00> + |11>)/sqrt(2)."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    # Step 1: Put q0 in superposition
    q0 = qmc.h(q0)

    # Step 2: Entangle with CNOT
    q0, q1 = qmc.cx(q0, q1)

    return qmc.measure(q0), qmc.measure(q1)


bell_state.draw()

# %%
exec_bell = transpiler.transpile(bell_state)
result_bell = exec_bell.sample(transpiler.executor(), shots=1000).result()

print("=== Bell State Measurement Results ===")
for value, count in result_bell.results:
    percentage = count / 1000 * 100
    print(f"  Result: {value}, Count: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### Characteristics of Entanglement
#
# Only `(0, 0)` and `(1, 1)` appear -- never `(0, 1)` or `(1, 0)`.
#
# This is the signature of entanglement:
#
# - The two qubits are **perfectly correlated**.
# - If one measures to 0, the other is definitely 0.
# - If one measures to 1, the other is definitely 1.
# - But which pair appears is completely random (50/50).
#
# This correlation cannot be explained by any classical mechanism
# where each qubit secretly carries a predetermined value.
# It is a genuinely quantum phenomenon.

# %% [markdown]
# ## 5. The Four Bell States
#
# The Bell state $|\Phi^+\rangle$ is one of four maximally entangled
# two-qubit states called the **Bell basis**.
# Together they form a complete basis for two-qubit entanglement.
#
# | Name | Circuit | Formula | Outcomes |
# |------|---------|---------|----------|
# | Phi+ | H, CX | (\|00> + \|11>) / sqrt(2) | Same: (0,0) or (1,1) |
# | Phi- | H, CX, P(pi) | (\|00> - \|11>) / sqrt(2) | Same: (0,0) or (1,1) |
# | Psi+ | H, CX, X | (\|01> + \|10>) / sqrt(2) | Opposite: (0,1) or (1,0) |
# | Psi- | H, CX, X, P(pi) | (\|01> - \|10>) / sqrt(2) | Opposite: (0,1) or (1,0) |
#
# The Phi states produce same-valued pairs; the Psi states produce
# opposite-valued pairs.  The +/- sign is a phase difference that
# does not affect the measurement outcome probabilities directly, but
# matters for interference and quantum information protocols.
#
# For the phase flip we use the P gate with $\theta = \pi$,
# which is exactly the Z gate: $P(\pi) = \begin{pmatrix}1&0\\0&-1\end{pmatrix}$.


# %%
@qmc.qkernel
def bell_phi_plus() -> tuple[qmc.Bit, qmc.Bit]:
    """Bell state |Phi+> = (|00> + |11>)/sqrt(2)."""
    q0, q1 = qmc.qubit(name="q0"), qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)


bell_phi_plus.draw()


# %%
@qmc.qkernel
def bell_phi_minus() -> tuple[qmc.Bit, qmc.Bit]:
    """Bell state |Phi-> = (|00> - |11>)/sqrt(2)."""
    q0, q1 = qmc.qubit(name="q0"), qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    q0 = qmc.p(q0, math.pi)  # Z gate: phase flip
    return qmc.measure(q0), qmc.measure(q1)


bell_phi_minus.draw()


# %%
@qmc.qkernel
def bell_psi_plus() -> tuple[qmc.Bit, qmc.Bit]:
    """Bell state |Psi+> = (|01> + |10>)/sqrt(2)."""
    q0, q1 = qmc.qubit(name="q0"), qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    q1 = qmc.x(q1)  # Flip target
    return qmc.measure(q0), qmc.measure(q1)


bell_psi_plus.draw()


# %%
@qmc.qkernel
def bell_psi_minus() -> tuple[qmc.Bit, qmc.Bit]:
    """Bell state |Psi-> = (|01> - |10>)/sqrt(2)."""
    q0, q1 = qmc.qubit(name="q0"), qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    q1 = qmc.x(q1)
    q0 = qmc.p(q0, math.pi)  # Z gate: phase flip
    return qmc.measure(q0), qmc.measure(q1)


bell_psi_minus.draw()

# %%
bell_states = [
    ("Phi+", bell_phi_plus),
    ("Phi-", bell_phi_minus),
    ("Psi+", bell_psi_plus),
    ("Psi-", bell_psi_minus),
]

print("=== The Four Bell States ===\n")

for name, circuit in bell_states:
    exec_b = transpiler.transpile(circuit)
    result_b = exec_b.sample(transpiler.executor(), shots=1000).result()

    print(f"|{name}>:")
    for value, count in result_b.results:
        percentage = count / 1000 * 100
        print(f"  {value}: {count} ({percentage:.1f}%)")
    print()

# %% [markdown]
# ### Observations
#
# - **Phi+** and **Phi-** both produce (0,0) and (1,1).
#   They differ only by a relative phase, invisible in these measurements.
# - **Psi+** and **Psi-** both produce (0,1) and (1,0).
#   Again, the phase difference is not visible here.
#
# The phase becomes relevant in more advanced protocols such as
# quantum teleportation and superdense coding, where interference
# between Bell states is used.

# %% [markdown]
# ## 6. GHZ States: Multi-Qubit Entanglement
#
# Entanglement is not limited to two qubits.
# The **GHZ state** (Greenberger-Horne-Zeilinger state) generalizes
# the Bell state to N qubits:
#
# $$|GHZ_N\rangle = \frac{|00\ldots0\rangle + |11\ldots1\rangle}{\sqrt{2}}$$
#
# All N qubits are entangled: measuring any one of them as 0 forces
# all the others to be 0, and similarly for 1.
#
# The construction follows the same pattern as the Bell state:
# put the first qubit in superposition, then chain CNOT gates
# to spread the entanglement to every other qubit.
#
# Here we use `qmc.qubit_array()` and `qmc.range()` (introduced in
# tutorial 02) to write a general N-qubit GHZ circuit.


# %%
@qmc.qkernel
def ghz_state(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Create an N-qubit GHZ state."""
    qubits = qmc.qubit_array(n, name="q")

    # Put the first qubit in superposition
    qubits[0] = qmc.h(qubits[0])

    # Chain CNOT gates to spread entanglement
    for i in qmc.range(n - 1):
        qubits[i], qubits[i + 1] = qmc.cx(qubits[i], qubits[i + 1])

    return qmc.measure(qubits)


ghz_state.draw(n=4, fold_loops=False)

# %%
# 3-qubit GHZ state
exec_ghz3 = transpiler.transpile(ghz_state, bindings={"n": 3})
result_ghz3 = exec_ghz3.sample(transpiler.executor(), shots=1000).result()

print("=== 3-Qubit GHZ State ===")
print("|GHZ> = (|000> + |111>)/sqrt(2)\n")
for value, count in result_ghz3.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %%
# 5-qubit GHZ state
exec_ghz5 = transpiler.transpile(ghz_state, bindings={"n": 5})
result_ghz5 = exec_ghz5.sample(transpiler.executor(), shots=1000).result()

print("=== 5-Qubit GHZ State ===")
print("|GHZ> = (|00000> + |11111>)/sqrt(2)\n")
for value, count in result_ghz5.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### Characteristics of GHZ States
#
# Regardless of the number of qubits, only two outcomes ever appear:
# all zeros or all ones.  Every qubit is perfectly correlated with
# every other qubit.
#
# GHZ states are used in:
# - Tests of quantum nonlocality (Bell inequality violations for N > 2)
# - Quantum error correction
# - Quantum secret sharing protocols

# %% [markdown]
# ## 7. Summary
#
# This tutorial explored three concepts that underpin all of quantum computing:
#
# ### Superposition
# - The H gate puts a qubit into a state that is simultaneously
#   $|0\rangle$ and $|1\rangle$.
# - Measurement collapses the state to one outcome, with probabilities
#   determined by the amplitudes.
#
# ### Interference
# - Probability amplitudes can reinforce (constructive) or cancel
#   (destructive).
# - The double-Hadamard experiment ($H \cdot H = I$) demonstrates
#   this: randomness is introduced and then perfectly undone.
# - Quantum algorithms work by arranging interference so that
#   correct answers are amplified and wrong answers are suppressed.
#
# ### Entanglement
# - The combination H + CNOT creates entangled states where
#   qubits are perfectly correlated.
# - The four **Bell states** are the fundamental two-qubit entangled states.
# - **GHZ states** extend entanglement to N qubits.
#
# These three phenomena -- superposition, interference, and entanglement --
# are the ingredients that give quantum algorithms their power.
# In the next tutorial, we will explore Qamomile's
# [standard library](05_stdlib.ipynb) -- ready-made building blocks like
# QFT and QPE that rely on exactly these principles.

# %% [markdown]
# ## What We Learned
#
# - **Superposition: the quantum coin flip** -- The H gate puts a qubit into an equal mix of $|0\rangle$ and $|1\rangle$, giving 50/50 measurement outcomes.
# - **Phase and the $|+\rangle$ / $|-\rangle$ states** -- Both look identical when measured in the computational basis, but their opposite phases lead to different interference behaviour.
# - **Quantum interference: how phases control outcomes** -- Amplitudes can reinforce or cancel; the double-Hadamard experiment ($H \cdot H = I$) shows randomness introduced and perfectly undone.
# - **Entanglement with CNOT and Bell states** -- H + CNOT creates correlations that cannot exist classically; the four Bell states form the fundamental basis for two-qubit entanglement.
# - **GHZ states: multi-qubit entanglement** -- GHZ extends perfect all-or-nothing correlation to $N$ qubits using a chain of CNOT gates.
