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
# tags: [algorithm, encoding, primitive]
# ---
#
# # Möttönen Amplitude Encoding
#
# **Amplitude encoding** is the operation that, given a unit-norm
# complex vector $a \in \mathbb{C}^{2^n}$, prepares the $n$-qubit state
#
# $$
# |\psi\rangle \;=\; \sum_{i=0}^{2^n - 1} a_i \, |i\rangle
# $$
#
# starting from $|0\rangle^{\otimes n}$. It is the entry door for any
# algorithm that consumes classical data as a quantum state — including
# HHL-style linear-system solvers, kernel methods, and many quantum
# simulation protocols. Qamomile ships a backend-portable implementation
# under `qamomile.circuit.algorithm.state_preparation`, based on the
# uniformly controlled rotation construction of Möttönen, Vartiainen,
# Bergholm and Salomaa (arXiv:quant-ph/0407010).
#
# The construction has two stages:
#
# 1. A cascade of $n$ **uniformly controlled $R_y$** gates that
#    distributes the magnitude $|a_i|$ across the basis states. This
#    stage alone is sufficient for real (signed) amplitude vectors.
# 2. A second cascade of **uniformly controlled $R_z$** gates that
#    restores the relative phases. Only emitted when the input has a
#    non-zero imaginary part.
#
# Each uniformly controlled rotation is decomposed into elementary
# `RY` / `RZ` and `CNOT` gates using the Gray-code recipe of
# Möttönen-Vartiainen. The total cost is
#
# | Stage | Real input | Complex input |
# |---|---:|---:|
# | $R_y$ rotations | $2^n - 1$ | $2^n - 1$ |
# | $R_z$ rotations | $0$ | $2^n - 1$ |
# | `CNOT` | $2^n - 2$ | $2 (2^n - 2)$ |
#
# This tutorial walks through the public API surface, demonstrates the
# three input modes (concrete sequence, compile-time-bound
# `Vector[Float]`, runtime-parametric angles), and shows where each
# mode shines.

# %%
import numpy as np
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Statevector

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    amplitude_encoding_from_angles,
    compute_mottonen_amplitude_encoding_ry_angles,
    compute_mottonen_amplitude_encoding_rz_angles,
)
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
executor = transpiler.executor()


def fidelity(prepared: np.ndarray, target: np.ndarray) -> float:
    """Phase-invariant fidelity ``|<prepared|target>|^2``."""
    return float(np.abs(np.vdot(prepared, target)) ** 2)


def normalize(amps: list[float] | list[complex]) -> np.ndarray:
    """Unit-norm copy of *amps* (complex dtype if any element is complex)."""
    if any(isinstance(x, complex) for x in amps):
        arr = np.asarray(amps, dtype=complex)
    else:
        arr = np.asarray(amps, dtype=float)
    return arr / np.linalg.norm(arr)


# %% [markdown]
# ## 1. The simplest call — concrete real amplitudes
#
# `amplitude_encoding(qubits, amplitudes)` is the everyday entry point.
# It accepts a Python sequence or NumPy array, normalises it
# automatically, and prepares the corresponding state.
#
# As a first sanity check we encode the (un-normalised) vector
# $a = (1, 2, 3, 4)$ on a 2-qubit register and read back the simulator's
# statevector.

# %%
amps_real = [1.0, 2.0, 3.0, 4.0]


@qmc.qkernel
def prepare_real() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps_real)
    return qmc.measure(q)


qc = transpiler.to_circuit(prepare_real)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
expected = normalize(amps_real)
print(f"prepared      = {np.round(sv, 4)}")
print(f"target (norm) = {np.round(expected, 4)}")
print(f"fidelity      = {fidelity(sv, expected):.6f}")

# %% [markdown]
# Negative real amplitudes flow through the magnitude stage naturally —
# the leaf-level $R_y$ angle is taken as a signed `arctan2`, so the sign
# is captured without an extra phase stage. The state $a = (1, -1, 1, -1)$
# is therefore prepared with `RY` and `CNOT` only.

# %%
amps_signed = [1.0, -1.0, 1.0, -1.0]


@qmc.qkernel
def prepare_signed() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps_signed)
    return qmc.measure(q)


qc = transpiler.to_circuit(prepare_signed)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
expected = normalize(amps_signed)
print(f"fidelity (signed) = {fidelity(sv, expected):.6f}")

# %% [markdown]
# ## 2. Complex amplitudes
#
# The same API accepts complex inputs. When at least one entry has a
# non-zero imaginary part, the implementation switches to the two-stage
# (Ry + Rz) construction automatically. A complex vector with
# identically zero imaginary part is silently coerced to the cheaper
# real path.
#
# We encode $a = (1, 1+i, 1-i, 2i)$ — a generic complex 2-qubit state —
# and verify the resulting amplitudes match (up to global phase).

# %%
amps_complex = [1 + 0j, 1 + 1j, 1 - 1j, 0 + 2j]


@qmc.qkernel
def prepare_complex() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps_complex)
    return qmc.measure(q)


qc = transpiler.to_circuit(prepare_complex)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
expected = normalize(amps_complex)
print(f"fidelity (complex) = {fidelity(sv, expected):.6f}")

# %% [markdown]
# ## 3. Inspecting the gate budget
#
# `MottonenAmplitudeEncoding` exposes a `_resources()` method that
# returns the per-gate counts predicted by the Gray-walk decomposition.
# The single-stage path (real input) and the two-stage path (complex
# input) are reported separately.

# %%
gate_real = MottonenAmplitudeEncoding(amps_real)
gate_complex = MottonenAmplitudeEncoding(amps_complex)

for label, gate in (("real", gate_real), ("complex", gate_complex)):
    meta = gate._resources().custom_metadata
    print(
        f"{label:7s}: RY={meta['num_ry_gates']:>3d}  RZ={meta['num_rz_gates']:>3d}"
        f"  CNOT={meta['num_cnot_gates']:>3d}  complex_input={meta['complex_input']}"
    )

# %% [markdown]
# Both counts grow as $O(2^n)$: amplitude encoding is intrinsically
# expensive for many qubits. In practice this construction is most
# useful at the small register sizes one encounters as a building block
# inside larger algorithms (HHL with 4–8 logical qubits in the
# input register, QSCI with a sampled subspace, error-correction
# warm-starts, etc.), not as a stand-alone preparation for hundreds of
# qubits.

# %% [markdown]
# ## 4. Three input modes
#
# `amplitude_encoding` accepts the amplitudes in three different forms;
# `amplitude_encoding_from_angles` adds a fourth mode that exposes
# pre-computed angles so the same compiled circuit can be re-bound at
# runtime. Picking the right one depends on **when** the values are
# known and **how** you want to feed them into a kernel.
#
# | Mode | Input lives in | When values are needed | Re-binding |
# |---|---|---|---|
# | A. Closure | Python (outer scope) | Trace time | Recompile per amplitude |
# | B. `bindings={"amps": ...}` on `Vector[Float]` | Kernel parameter | Trace time (extracted from binding metadata) | Recompile per amplitude |
# | C. `amplitude_encoding_from_angles` + `bindings` | Kernel parameter | Trace time | Recompile per angle vector |
# | D. `amplitude_encoding_from_angles` + `parameters` | Kernel parameter | **Runtime** | Re-bind without recompilation |
#
# Modes A–C are different ways of expressing "the amplitudes are known
# at compile time"; D is the only mode that lets you reuse a compiled
# circuit across different amplitudes inside an inner loop.
#
# We illustrate each mode below.

# %% [markdown]
# ### Mode A — Closure (the calls we already saw)
#
# This is what every example above used: the amplitudes live as a
# Python list in the outer scope and the kernel closes over them.
# Best when the kernel and the amplitudes live near each other in code.

# %% [markdown]
# ### Mode B — `Vector[Float]` parameter, bound at compile time
#
# When you would rather expose the amplitudes as a kernel parameter
# (for documentation, for sweeping over different vectors, or to keep
# the kernel definition free of magic numbers), declare the parameter
# as `Vector[Float]` and pass the values via `bindings={...}`. The
# implementation reads the bound concrete data out of the handle's
# `array_runtime_metadata` at trace time, so the angle computation
# still runs classically and the IR carries a single
# `MottonenAmplitudeEncoding` composite gate.

# %%
@qmc.qkernel
def prepare_via_binding(amps: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps)
    return qmc.measure(q)


qc = transpiler.to_circuit(
    prepare_via_binding, bindings={"amps": [1.0, 2.0, 3.0, 4.0]}
)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
print(f"fidelity (mode B) = {fidelity(sv, normalize([1.0, 2.0, 3.0, 4.0])):.6f}")

# %% [markdown]
# Trying to leave the parameter symbolic with `parameters=["amps"]` is
# rejected with a directing error — the angle computation
# (`atan2(|a_1|, |a_0|)` and friends) needs concrete numbers and
# therefore fundamentally cannot be deferred to runtime. The error
# points at `amplitude_encoding_from_angles` for the runtime case.

# %%
try:
    transpiler.transpile(prepare_via_binding, parameters=["amps"])
except ValueError as exc:
    print(f"ValueError: {exc}")

# %% [markdown]
# ### Mode C — Pre-computed angles, bound at compile time
#
# The classical part of Möttönen's construction is two helpers:
# `compute_mottonen_amplitude_encoding_ry_angles(amps)` and
# `compute_mottonen_amplitude_encoding_rz_angles(amps)`. They return
# Gray-walk-ordered $R_y$ and $R_z$ angles of length $2^n - 1$ each
# (the $R_z$ array is identically zero for real input).
#
# `amplitude_encoding_from_angles` is the companion function that
# accepts those angles instead of amplitudes. Used with
# `bindings={...}`, it behaves much like Mode B — but **without**
# wrapping in the `MottonenAmplitudeEncoding` composite gate. The IR
# carries the elementary `RY` / `RZ` / `CNOT` gates directly, which is
# what unlocks the runtime mode below.

# %%
ry_angles = compute_mottonen_amplitude_encoding_ry_angles(amps_complex).tolist()
rz_angles = compute_mottonen_amplitude_encoding_rz_angles(amps_complex).tolist()


@qmc.qkernel
def prepare_from_angles(
    ry_a: qmc.Vector[qmc.Float], rz_a: qmc.Vector[qmc.Float]
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding_from_angles(q, ry_a, rz_a)
    return qmc.measure(q)


qc = transpiler.to_circuit(
    prepare_from_angles, bindings={"ry_a": ry_angles, "rz_a": rz_angles}
)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
print(f"fidelity (mode C) = {fidelity(sv, normalize(amps_complex)):.6f}")

# %% [markdown]
# ### Mode D — Runtime-parametric angles
#
# The reason `amplitude_encoding_from_angles` exists at all is to
# enable this mode. By transpiling with `parameters=[...]` we keep the
# angle vectors symbolic in the emitted backend circuit, so the **same
# compiled circuit** can be re-bound to many different amplitude
# vectors via `executable.sample(bindings={...})`. This is the right
# pattern for hybrid loops (e.g. classical optimisation over the
# amplitudes), where recompilation per iteration would otherwise
# dominate the wall-clock budget.

# %%
exe = transpiler.transpile(prepare_from_angles, parameters=["ry_a", "rz_a"])
n_runtime_params = len(exe.compiled_quantum[0].circuit.parameters)
print(f"runtime parameters in compiled circuit: {n_runtime_params}")

shots = 8192
for trial_amps in (
    [1.0, 0.0, 0.0, 1.0],
    [3.0, 4.0, 0.0, 0.0],
    [1 + 0j, 1j, -1 + 0j, -1j],
):
    ry = compute_mottonen_amplitude_encoding_ry_angles(trial_amps).tolist()
    rz = compute_mottonen_amplitude_encoding_rz_angles(trial_amps).tolist()
    counts = (
        exe.sample(executor, shots=shots, bindings={"ry_a": ry, "rz_a": rz})
        .result()
        .results
    )
    observed = np.zeros(4)
    for bits, c in counts:
        idx = sum(int(b) << i for i, b in enumerate(bits))
        observed[idx] = c / shots
    expected_probs = np.abs(normalize(trial_amps)) ** 2
    print(
        f"amps={str(trial_amps):<48s}  "
        f"max|p_obs - p_exp| = {np.max(np.abs(observed - expected_probs)):.4f}"
    )

# %% [markdown]
# All three iterations sample from the same compiled circuit; only the
# runtime bindings change. The maximum per-bin deviation stays within
# a few standard deviations of the multinomial shot noise.

# %% [markdown]
# ## 5. Estimating an observable on the encoded state
#
# `amplitude_encoding` is a building block — most users plug it into a
# larger kernel. The simplest such use case is computing
# $\langle \psi | H | \psi \rangle$ for some Hamiltonian $H$ on the
# prepared state. The kernel becomes a single `expval`, and the
# observable can be passed in as a runtime binding.
#
# As a small analytic check, the encoded state for $a = (1, 2, 3, 4)$
# (little-endian, qubit $0$ = LSB) gives
#
# $$
#   \langle Z_0 \rangle
#   = (p_{00} + p_{10}) - (p_{01} + p_{11})
#   = \frac{1 + 9 - 4 - 16}{30}
#   = -\tfrac{1}{3},
# $$
#
# which we now reproduce with the estimator path.

# %%
@qmc.qkernel
def expval_kernel(H: qmc.Observable) -> qmc.Float:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, [1.0, 2.0, 3.0, 4.0])
    return qmc.expval(q, H)


H = qm_o.Z(0) + 0.0 * qm_o.Z(1)  # pad to 2-qubit width
exe_expval = transpiler.transpile(expval_kernel, bindings={"H": H})
result = exe_expval.run(executor).result()
print(f"<Z_0> = {float(result):+.6f}   (analytic: {-1/3:+.6f})")

# %% [markdown]
# ## When to use which
#
# - **Concrete amplitudes known in Python** → call `amplitude_encoding`
#   directly with the list/array. Mode A.
# - **Amplitudes need to be a kernel parameter for clarity** → declare
#   it as `Vector[Float]` and bind via `bindings={...}`. Mode B.
# - **Hybrid optimisation loop, recompilation cost matters** → call
#   `amplitude_encoding_from_angles` with `parameters=[...]` and
#   pre-compute the angles inside the loop. Mode D.
# - **Resource estimation** → use `MottonenAmplitudeEncoding._resources()`
#   to get the closed-form gate counts before transpilation.
#
# The exponential gate growth ($O(2^n)$ rotations and CNOTs) is the
# fundamental cost of Möttönen's construction. For larger registers,
# specialised constructions (e.g. low-rank approximations of the input,
# or backend-native primitives such as Qiskit's `StatePreparation`) are
# usually preferred — Qamomile's CompositeGate machinery leaves room to
# dispatch to such native paths in the future via a
# `CompositeGateEmitter`, but the elementary decomposition above is
# always available as the portable fallback.
