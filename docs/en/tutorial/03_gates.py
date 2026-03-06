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
# # Quantum Gates in Qamomile
#
# A reference for the basic quantum gates available in Qamomile.
#
# ## What We Will Learn
# - All single-qubit gates available in Qamomile
# - All multi-qubit gates available in Qamomile
# - The gate return value pattern (affine types in action)

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. Quick Recap
#
# In the previous tutorial (`02_type_system`), we learned how to create qubits
# with `qmc.qubit()` and `qmc.qubit_array()`. We use those here without
# further explanation. Every qubit starts in $|0\rangle$.

# %% [markdown]
# ---
# ## 2. Single-Qubit Gates
#
# Qamomile provides six single-qubit gates -- this is the **complete** list:
#
# | Gate | Syntax | Parameters |
# |------|--------|------------|
# | H | `qmc.h(q)` | none |
# | X | `qmc.x(q)` | none |
# | P | `qmc.p(q, theta)` | angle |
# | RX | `qmc.rx(q, angle)` | angle |
# | RY | `qmc.ry(q, angle)` | angle |
# | RZ | `qmc.rz(q, angle)` | angle |

# %% [markdown]
# ### 2.1 H Gate (Hadamard)
#
# $$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix},
# \quad H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$


# %%
@qmc.qkernel
def h_gate_demo() -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    return qmc.measure(q)


h_gate_demo.draw()

# %%
result_h = (
    transpiler.transpile(h_gate_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== H Gate ===")
for value, count in result_h.results:
    print(f"  {value}: {count} ({count / 10:.1f}%)")

# %% [markdown]
# Approximately 50/50, confirming superposition.

# %% [markdown]
# ### 2.2 X Gate (NOT)
#
# $$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},
# \quad X|0\rangle = |1\rangle$$


# %%
@qmc.qkernel
def x_gate_demo() -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.x(q)
    return qmc.measure(q)


x_gate_demo.draw()

# %%
result_x = (
    transpiler.transpile(x_gate_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== X Gate ===")
for value, count in result_x.results:
    print(f"  {value}: {count}")

# %% [markdown]
# Always produces $|1\rangle$ from $|0\rangle$.

# %% [markdown]
# ### 2.3 P Gate (Phase)
#
# $$P(\theta) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix}$$
#
# Phase alone does not change measurement probabilities. To see its effect,
# sandwich it between H gates: H-P($\pi$)-H is equivalent to X.


# %%
@qmc.qkernel
def p_gate_demo() -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    q = qmc.p(q, math.pi)
    q = qmc.h(q)
    return qmc.measure(q)


p_gate_demo.draw()

# %%
result_p = (
    transpiler.transpile(p_gate_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== P Gate: H-P(pi)-H = X ===")
for value, count in result_p.results:
    print(f"  {value}: {count}")

# %% [markdown]
# Always 1 -- the phase causes destructive interference on $|0\rangle$.

# %% [markdown]
# ### 2.4 RX Gate (X-axis Rotation)
#
# $$RX(\theta) = \exp\!\bigl(-i\,\tfrac{\theta}{2}\,X\bigr)
#   = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
#     -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$


# %%
@qmc.qkernel
def rx_demo(theta: qmc.Float) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.rx(q, theta)
    return qmc.measure(q)


rx_demo.draw()

# %%
print("=== RX Gate at Different Angles ===\n")
for angle, name in zip(
    [0, math.pi / 4, math.pi / 2, math.pi], ["0", "pi/4", "pi/2", "pi"]
):
    result = (
        transpiler.transpile(rx_demo, bindings={"theta": angle})
        .sample(transpiler.executor(), shots=1000)
        .result()
    )
    print(f"RX({name}):")
    for value, count in result.results:
        print(f"  {value}: {count} ({count / 10:.1f}%)")
    print()

# %% [markdown]
# - **RX(0)**: No rotation -- always 0.
# - **RX(pi/4)**: Small rotation -- 1 begins to appear.
# - **RX(pi/2)**: Approximately 50/50, similar to H.
# - **RX(pi)**: Always 1, equivalent to X (up to global phase).

# %% [markdown]
# ### 2.5 RY Gate (Y-axis Rotation)
#
# $$RY(\theta) = \exp\!\bigl(-i\,\tfrac{\theta}{2}\,Y\bigr)
#   = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
#     \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$
#
# Unlike RX, RY produces purely real amplitudes.


# %%
@qmc.qkernel
def ry_demo(theta: qmc.Float) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


ry_demo.draw()

# %%
result_ry = (
    transpiler.transpile(ry_demo, bindings={"theta": math.pi / 2})
    .sample(transpiler.executor(), shots=1000)
    .result()
)
print("=== RY(pi/2) ===")
for value, count in result_ry.results:
    print(f"  {value}: {count} ({count / 10:.1f}%)")

# %% [markdown]
# Approximately 50/50, similar to H.

# %% [markdown]
# ### 2.6 RZ Gate (Z-axis Rotation)
#
# $$RZ(\theta) = \exp\!\bigl(-i\,\tfrac{\theta}{2}\,Z\bigr)
#   = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$
#
# Like P, RZ only affects phase. From $|0\rangle$ alone it has no visible
# effect, so we sandwich it in H gates.


# %%
@qmc.qkernel
def rz_demo() -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    q = qmc.rz(q, math.pi)
    q = qmc.h(q)
    return qmc.measure(q)


rz_demo.draw()

# %%
result_rz = (
    transpiler.transpile(rz_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== RZ(pi) sandwiched between H gates ===")
for value, count in result_rz.results:
    print(f"  {value}: {count}")

# %% [markdown]
# Always 1. The Z-rotation flips the relative phase inside the superposition,
# and the second H converts that phase difference into a bit flip.

# %% [markdown]
# ---
# ## 3. Multi-Qubit Gates
#
# Qamomile provides five two-qubit gates -- this is the **complete** list:
#
# | Gate | Syntax | Parameters |
# |------|--------|------------|
# | CX (CNOT) | `qmc.cx(control, target)` | none |
# | CZ | `qmc.cz(control, target)` | none |
# | SWAP | `qmc.swap(q0, q1)` | none |
# | CP | `qmc.cp(control, target, theta)` | angle |
# | RZZ | `qmc.rzz(q0, q1, angle)` | angle |
#
# Every two-qubit gate returns **both** qubits (see Section 4).
#
# > **Qubit ordering recap** (see [01_introduction](01_introduction.ipynb)):
# > Tuple results follow array order `(q[0], q[1], ...)`.
# > Ket notation is big-endian: $|q_n \cdots q_1 q_0\rangle$.
# > For example, a result `(1, 0)` for `(q0, q1)` corresponds to $|01\rangle$.

# %% [markdown]
# ### 3.1 CX Gate (CNOT)
#
# Flips the target when the control is $|1\rangle$.
#
# $|10\rangle \to |11\rangle,\quad |11\rangle \to |10\rangle$
# ($|00\rangle$ and $|01\rangle$ unchanged.)


# %%
@qmc.qkernel
def cx_demo() -> tuple[qmc.Bit, qmc.Bit]:
    ctrl = qmc.qubit(name="ctrl")
    tgt = qmc.qubit(name="tgt")
    ctrl = qmc.x(ctrl)
    ctrl, tgt = qmc.cx(ctrl, tgt)
    return qmc.measure(ctrl), qmc.measure(tgt)


cx_demo.draw()

# %%
result_cx = (
    transpiler.transpile(cx_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== CX Gate (control=1) ===")
for value, count in result_cx.results:
    print(f"  {value}: {count}")

# %% [markdown]
# Control is $|1\rangle$, so target flips.
# The result $(1, 1)$ means `(ctrl, tgt) = (1, 1)` — in ket notation $|11\rangle$.

# %% [markdown]
# ### 3.2 CZ Gate (Controlled-Z)
#
# $CZ|11\rangle = -|11\rangle$; all other basis states unchanged.
# The gate is symmetric in its two qubits.
#
# CZ only changes the phase, which is invisible in measurement probabilities.
# A well-known identity converts CZ into CX by sandwiching H gates
# **on the target only**: $(I \otimes H) \cdot CZ \cdot (I \otimes H) = CX$.
# We demonstrate this below.


# %%
@qmc.qkernel
def cz_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.x(q0)  # control = |1>
    q1 = qmc.h(q1)  # H on target only
    q0, q1 = qmc.cz(q0, q1)
    q1 = qmc.h(q1)  # H on target only
    return qmc.measure(q0), qmc.measure(q1)


cz_demo.draw()

# %%
result_cz = (
    transpiler.transpile(cz_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== CZ Gate: (I⊗H)·CZ·(I⊗H) = CX ===")
for value, count in result_cz.results:
    print(f"  {value}: {count}")

# %% [markdown]
# With H on the target qubit only, CZ behaves exactly like CX.
# The control is $|1\rangle$, so the target flips — the result $(1, 1)$ means
# `(q0, q1) = (1, 1)` (ket $|11\rangle$), matching the CX demo above.

# %% [markdown]
# ### 3.3 SWAP Gate
#
# Exchanges the quantum states: $\text{SWAP}|a,b\rangle = |b,a\rangle$.


# %%
@qmc.qkernel
def swap_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.x(q0)  # q0=|1>, q1=|0>
    q0, q1 = qmc.swap(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)


swap_demo.draw()

# %%
result_swap = (
    transpiler.transpile(swap_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== SWAP Gate (q0=|1>, q1=|0> before) ===")
for value, count in result_swap.results:
    print(f"  {value}: {count}")

# %% [markdown]
# Before SWAP: `q0=|1>, q1=|0>`. After SWAP: `q0=|0>, q1=|1>`.
# The result $(0, 1)$ means `(q0, q1) = (0, 1)` — in ket notation $|10\rangle$.

# %% [markdown]
# ### 3.4 CP Gate (Controlled Phase)
#
# $CP(\theta)|11\rangle = e^{i\theta}|11\rangle$; other basis states unchanged.
# Central to the Quantum Fourier Transform (QFT).


# %%
@qmc.qkernel
def cp_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.x(q0)  # control = |1>
    q1 = qmc.h(q1)  # H on target only
    q0, q1 = qmc.cp(q0, q1, math.pi)
    q1 = qmc.h(q1)  # H on target only
    return qmc.measure(q0), qmc.measure(q1)


cp_demo.draw()

# %%
result_cp = (
    transpiler.transpile(cp_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== CP(pi): same pattern as CZ ===")
for value, count in result_cp.results:
    print(f"  {value}: {count}")

# %% [markdown]
# CP($\pi$) is identical to CZ, so the result matches the CZ demo:
# `(q0, q1) = (1, 1)` (ket $|11\rangle$) every time. With smaller angles
# (e.g. $\pi/4$), it becomes a partial phase rotation — the building block of QFT.

# %% [markdown]
# ### 3.5 RZZ Gate
#
# $$RZZ(\theta) = \exp\!\bigl(-i\,\tfrac{\theta}{2}\,Z \otimes Z\bigr)$$
#
# Important in QAOA, where it encodes Ising interaction terms.


# %%
@qmc.qkernel
def rzz_demo(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    q0, q1 = qmc.rzz(q0, q1, theta)
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    return qmc.measure(q0), qmc.measure(q1)


rzz_demo.draw()

# %%
result_rzz = (
    transpiler.transpile(rzz_demo, bindings={"theta": math.pi / 2})
    .sample(transpiler.executor(), shots=1000)
    .result()
)
print("=== RZZ(pi/2) with H-RZZ-H ===")
for value, count in result_rzz.results:
    print(f"  {value}: {count} ({count / 10:.1f}%)")

# %% [markdown]
# RZZ introduces correlations between qubits — the dominant outcomes are
# $(0, 0)$ and $(1, 1)$, i.e. `(q0, q1)` agree (in ket notation
# $|00\rangle$ and $|11\rangle$). We will see this gate again when we
# cover QAOA.

# %% [markdown]
# ---
# ## 4. Gate Return Value Pattern
#
# Every gate returns the qubits it consumed. This is a consequence of the
# **affine type system**: once a qubit enters a gate, the old handle is
# invalidated and a new handle is returned.
#
# ### Single-qubit gates: return one qubit
# ```python
# q = qmc.h(q)
# q = qmc.rx(q, theta)
# ```
#
# ### Two-qubit gates: return both qubits
# ```python
# q0, q1 = qmc.cx(q0, q1)
# q0, q1 = qmc.swap(q0, q1)
# ```
#
# ### With qubit arrays: assign back to the index
# ```python
# qubits[i] = qmc.h(qubits[i])
# qubits[i], qubits[j] = qmc.cx(qubits[i], qubits[j])
# ```
#
# ### Angle parameters are NOT returned
# ```python
# q = qmc.rx(q, theta)              # returns Qubit, not (Qubit, Float)
# q0, q1 = qmc.rzz(q0, q1, theta)  # returns (Qubit, Qubit)
# ```
#
# Ignoring the return value or capturing only one qubit from a two-qubit gate
# will cause an error at build time.

# %% [markdown]
# Here is a concrete example combining several gate types in one circuit.


# %%
@qmc.qkernel
def return_value_demo() -> qmc.Vector[qmc.Bit]:
    qubits = qmc.qubit_array(3, name="q")

    # Single-qubit: assign back to the same slot
    qubits[0] = qmc.h(qubits[0])
    qubits[1] = qmc.rx(qubits[1], math.pi / 4)

    # Two-qubit: unpack both return values
    qubits[0], qubits[1] = qmc.cx(qubits[0], qubits[1])
    qubits[1], qubits[2] = qmc.swap(qubits[1], qubits[2])

    return qmc.measure(qubits)


return_value_demo.draw()

# %%
result_rv = (
    transpiler.transpile(return_value_demo)
    .sample(transpiler.executor(), shots=1000)
    .result()
)
print("=== Return Value Pattern Demo ===")
for value, count in result_rv.results:
    print(f"  {value}: {count}")

# %% [markdown]
# **Interpreting the result:**
#
# 1. H puts `q[0]` in superposition: $\frac{|0\rangle + |1\rangle}{\sqrt{2}}$
# 2. RX($\pi/4$) slightly rotates `q[1]` away from $|0\rangle$
# 3. CX entangles `q[0]` and `q[1]` -- when `q[0]` is $|1\rangle$,
#    `q[1]` gets flipped
# 4. SWAP exchanges `q[1]` and `q[2]`, moving the entangled state to `q[2]`
#
# The dominant outcomes should be `(0, 0, 0)` and `(1, 0, 1)` (tuple order:
# `(q[0], q[1], q[2])`; in ket notation $|000\rangle$ and $|101\rangle$),
# reflecting the entanglement between `q[0]` and the qubit that was swapped
# into position `q[2]`. The small RX rotation on `q[1]` introduces minor
# contributions from other bitstrings.

# %% [markdown]
# ---
# ## 5. Summary Tables
#
# ### All Single-Qubit Gates
#
# | Gate | Qamomile Syntax | Mathematical Definition |
# |------|-----------------|------------------------|
# | H (Hadamard) | `q = qmc.h(q)` | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ |
# | X (NOT) | `q = qmc.x(q)` | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ |
# | P (Phase) | `q = qmc.p(q, theta)` | $\begin{pmatrix}1&0\\0&e^{i\theta}\end{pmatrix}$ |
# | RX | `q = qmc.rx(q, angle)` | $\exp(-i\,\frac{\text{angle}}{2}\,X)$ |
# | RY | `q = qmc.ry(q, angle)` | $\exp(-i\,\frac{\text{angle}}{2}\,Y)$ |
# | RZ | `q = qmc.rz(q, angle)` | $\exp(-i\,\frac{\text{angle}}{2}\,Z)$ |
#
# ### All Multi-Qubit Gates
#
# | Gate | Qamomile Syntax | Description |
# |------|-----------------|-------------|
# | CX (CNOT) | `q0, q1 = qmc.cx(q0, q1)` | Flips target when control is $\|1\rangle$ |
# | CZ | `q0, q1 = qmc.cz(q0, q1)` | Applies Z to target when control is $\|1\rangle$ |
# | SWAP | `q0, q1 = qmc.swap(q0, q1)` | Exchanges the states of two qubits |
# | CP | `q0, q1 = qmc.cp(q0, q1, theta)` | Controlled phase rotation by $\theta$ |
# | RZZ | `q0, q1 = qmc.rzz(q0, q1, angle)` | $\exp(-i\,\frac{\text{angle}}{2}\,Z \otimes Z)$ |
#
# ### Key Rule
#
# Every gate returns the qubits it consumed. Always capture the return value.
#
# In the next tutorial, we will use these gates to build our first quantum
# algorithm.

# %% [markdown]
# ## What We Learned
#
# - **All single-qubit gates available in Qamomile** — H, X, P, RX, RY, and RZ, each taking a qubit (and optionally an angle) and returning the transformed qubit.
# - **All multi-qubit gates available in Qamomile** — CX (CNOT), CZ, SWAP, CP, and RZZ, each consuming and returning all involved qubits as a tuple.
# - **The gate return value pattern (affine types in action)** — Every gate returns the qubits it consumed; always capture the return value to satisfy the affine type system.
