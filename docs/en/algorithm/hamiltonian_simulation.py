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
# tags: [algorithm, primitive, simulation]
# ---
#
# # Hamiltonian Simulation with Suzuki–Trotter (Rabi oscillation)
#
# Simulating the time evolution $e^{-iHt}$ of a quantum system is one of the
# canonical applications of a quantum computer. When the Hamiltonian splits
# into non-commuting pieces $H = A + B$ with $[A, B] \neq 0$, the naive
# factorisation $e^{-i(A+B)t} = e^{-iAt}\,e^{-iBt}$ is wrong; the standard fix
# is **Trotter–Suzuki product formulas**, which interleave short evolutions of
# each piece. The error decreases as we take smaller steps (Lie–Trotter,
# first order), symmetrise the step (Strang, second order), or nest the
# symmetric step recursively via Suzuki's construction (any even order).
#
# This article builds these approximations end-to-end in Qamomile on a
# one-qubit Rabi Hamiltonian so the Trotter error is measurable. We start
# with $S_1$ and $S_2$, then use the standard-library `suzuki_trotter`
# factory for higher orders. Rust derives a bounded schedule from static
# formula options; the generated qkernel retains structured loops. Finally, we
# verify the textbook convergence rates ($S_k$'s fidelity error scales as
# $\Delta t^{2k}$) against the exact 2x2 propagator.

# %% [markdown]
# ## Installation
#
# Install the Qamomile distribution:
#
# ```bash
# pip install qamomile
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import math

import numpy as np

import qamomile as qm
from qamomile.stdlib import suzuki_trotter

# %% [markdown]
# ## The Rabi Hamiltonian
#
# A single two-level system driven on resonance is governed by
#
# $$ H = \underbrace{\tfrac{\omega}{2} Z}_{H_z} + \underbrace{\tfrac{\Omega}{2} X}_{H_x}. $$
#
# Because $[Z, X] \neq 0$, splitting $H$ into $H_z$ and $H_x$ introduces a
# Trotter error that is easy to see even on a single qubit. Starting from
# $|0\rangle$, the excitation probability oscillates as
# $P_{|1\rangle}(t) = (\Omega/E)^2 \sin^2(Et/2)$ with $E = \sqrt{\omega^2 + \Omega^2}$.
#
# In Qamomile we build $H_z$ and $H_x$ as two separate `Observable`s and pack
# them into a Python list. The generated qkernel accepts a fixed-size
# `Const[Vector[Observable, ...]]`, bound during specialization. QCore retains
# abstract Pauli evolution and structured loops; target lowering determines
# how to realize them as SDK operations.


# %% [markdown]
# ### Verifying $[H_z, H_x] \neq 0$
#
# Trotter approximations only matter because $H_z$ and $H_x$ do not commute.
# `qamomile.observable.commutator(a, b)` computes the commutator
# $[a, b] = a b - b a$ on Hamiltonians directly. Internally it iterates over
# the Pauli-string pairs once and uses the qubit-parity rule — two Pauli
# strings anticommute iff the number of qubits on which they carry different
# non-identity Paulis is odd — to drop every commuting pair before any
# product is built. This is cheaper than expanding `Hz * Hx - Hx * Hz` and
# cancelling, and the result is a fully simplified `Hamiltonian` that we can
# inspect or compare against an analytic value.
#
# For the Rabi Hamiltonian the textbook value is
#
# $$ [H_z, H_x] \;=\; \tfrac{\omega \Omega}{4}\,[Z, X] \;=\; i\,\tfrac{\omega \Omega}{2}\, Y, $$
#
# which `commutator` reproduces exactly:

# %%
omega = 1.2
drive = 0.8
evolution_time = 1.5

drift_hamiltonian = 0.5 * omega * qm.Z(0)
drive_hamiltonian = 0.5 * drive * qm.X(0)
hamiltonians = (drift_hamiltonian, drive_hamiltonian)

commutator = qm.commutator(*hamiltonians)
expected_commutator = 0.5j * omega * drive * qm.Y(0)

assert commutator == expected_commutator
print("[H_z, H_x] =", commutator)

# %% [markdown]
# `Hamiltonian` values preserve the logical-qubit convention and Pauli
# coefficients independently of any SDK. The same values can therefore enter
# a Qamomile program before a target is chosen.

# %% [markdown]
# ## Exact reference state
#
# A 2x2 matrix exponential gives the exact state $|\psi(T)\rangle = e^{-iHT}|0\rangle$,
# which each Trotter approximation is judged against via the **fidelity error**
# $1 - |\langle\psi_\text{exact}|\psi_\text{trotter}\rangle|$.

# %%
pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
matrix = 0.5 * omega * pauli_z + 0.5 * drive * pauli_x
initial_state = np.array([1.0, 0.0], dtype=np.complex128)


def matrix_evolution(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    """Return the exact dense exponential of one Hermitian matrix."""

    values, vectors = np.linalg.eigh(hamiltonian)
    return vectors @ np.diag(np.exp(-1j * time * values)) @ vectors.conj().T


exact_state = matrix_evolution(matrix, evolution_time) @ initial_state

# %% [markdown]
# ## $S_1$: First-order Suzuki–Trotter decomposition (Lie–Trotter)
#
# The simplest split is
#
# $$ S_1(\Delta t) = e^{-i H_x \Delta t}\, e^{-i H_z \Delta t}, $$
#
# applied $N$ times for a total evolution time $T = N \Delta t$. Per step the
# local error is $O(\Delta t^2)$; integrated over $N = T/\Delta t$ steps the
# global state-norm error is $O(\Delta t)$.
#
# The factory below generates first- and second-order kernels with four
# time slices. `steps` is a static factory option; the total duration remains
# a `Param` value. Changing the duration reuses the formula, while changing
# the slice count requires constructing a different formula.


# %% [markdown]
# ## $S_2$: Second-order Suzuki–Trotter decomposition (Strang splitting)
#
# Symmetrising the step around the middle term cancels the leading error:
#
# $$ S_2(\Delta t) = e^{-i H_z \Delta t/2}\, e^{-i H_x \Delta t}\, e^{-i H_z \Delta t/2}. $$
#
# The local error drops to $O(\Delta t^3)$ and the global state-norm error to
# $O(\Delta t^2)$. The step kernel is just three `pauli_evolve` calls.


# %%
first_order = suzuki_trotter(term_count=2, order=1, steps=4)
second_order = suzuki_trotter(term_count=2, order=2, steps=4)


@qm.qkernel
def compare_orders(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    first = first_order(qm.qubits(1), hamiltonians, time)
    second = second_order(qm.qubits(1), hamiltonians, time)
    return first, second


# %%
order_program = qm.specialize(compare_orders, hamiltonians=hamiltonians)
order_study = qm.resources.logical(order_program)
order_metric = qm.resources.logical_operation_count("quantum.pauli_evolution")

assert order_study.metric(order_metric).numeric_enclosure() == (20, 20)

# %% [markdown]
# Four first-order slices contain eight term evolutions, while four
# second-order slices contain twelve. Returning both owned states is valid
# because each allocation has one independent owner; callers can compose or
# measure them separately.

# %% [markdown]
# ## Higher-order Suzuki–Trotter decomposition: the fractal recursion
#
# Masuo Suzuki showed that an arbitrary even-order Trotter approximation can
# be built **recursively** from $S_2$ by nesting five rescaled copies at each
# level:
#
# $$ S_{2k}(\Delta t) = S_{2k-2}(p_k \Delta t)^2 \, S_{2k-2}\bigl((1 - 4 p_k)\Delta t\bigr) \, S_{2k-2}(p_k \Delta t)^2, $$
#
# with the level-specific coefficient
#
# $$ p_k = \frac{1}{4 - 4^{1/(2k-1)}}. $$
#
# $p_k$ is chosen so that the $(2k-1)$-th-order error of the lower formula
# cancels, leaving a local error of $O(\Delta t^{2k+1})$ per step — the
# coefficient **must** be recomputed at every level. Concretely:
#
# - $k=2$ (4th order): $p_2 = 1/(4 - 4^{1/3}) \approx 0.4145$,
# - $k=3$ (6th order): $p_3 = 1/(4 - 4^{1/5}) \approx 0.3731$,
# - $k=4$ (8th order): $p_4 = 1/(4 - 4^{1/7}) \approx 0.3596$.
#
# Reusing $p_2$ at every level leaves a non-zero $(2k-1)$-th-order error term,
# so the resulting formula is no better than $S_4$ — a classic trap when
# implementing Suzuki-Trotter by hand.

# %% [markdown]
# ## Choose a bounded fourth-order formula
#
# `suzuki_trotter` takes static topology options at ordinary Python factory
# time. Here we choose two Hamiltonian terms, fourth order, and four equal time
# slices. Rust derives and bounds the eleven-entry symmetric schedule; the
# returned value is an ordinary typed Qamomile function.
#
# The schedule remains composition, not a special Hamiltonian-simulation
# opcode. QCore keeps fixed loops around one abstract Pauli-evolution operation,
# and a target decides how to realize that operation only during lowering.

# %%
evolve = suzuki_trotter(term_count=2, order=4, steps=4)


@qm.qkernel
def rabi_samples(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[1]]:
    """Evolve and measure one driven two-level system.

    Args:
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Ordered drift and
            drive terms.
        time (qm.Param[qm.Real]): Total evolution time.

    Returns:
        qm.Shot[qm.Bits[1]]: One sampled excitation bit.
    """

    state = evolve(qm.qubits(1), hamiltonians, time)
    return qm.measure(state)


# %% [markdown]
# The Hamiltonian split is `Const` because it selects static program data.
# Evolution time is `Param`, so one compiled artifact can be launched with
# different times without rebuilding its structure.

# %% [markdown]
# ## Estimate the program before choosing hardware
#
# Specialization produces the verified target-independent `Program`.
# Logical resource analysis reads that same image and composes the loop trip
# counts with the Rust-owned product schedule.

# %%
program = qm.specialize(rabi_samples, hamiltonians=hamiltonians)
study = qm.resources.logical(program)
pauli_evolutions = qm.resources.logical_operation_count("quantum.pauli_evolution")

assert "quantum.pauli_evolution" in program.inspect()
assert "quantum.hamiltonian_simulation" not in program.inspect()
assert study.metric(pauli_evolutions).numeric_enclosure() == (44, 44)
print("logical Pauli evolutions:", study.metric(pauli_evolutions))

# %% [markdown]
# The estimate is exact: eleven schedule entries are invoked in each of four
# slices. Resource analysis does not expand a statevector, transpile an SDK
# circuit, or mistake an unavailable decomposition for zero work.

# %% [markdown]
# ### Reproduce product-formula resource scaling
#
# Section 11.1 of [](https://doi.org/10.48550/arXiv.2310.03011) constructs a
# product formula by repeating $r$ short-time slices and obtains its gate cost
# by multiplying the work per slice by $r$. In the Pauli-access model it uses
# no additional ancilla qubits. For this two-term second-order formula, every
# slice contains exactly three abstract Pauli evolutions. We expose the repeat
# count as a `Const` argument so the body is analyzed once and each reported
# row is an algebraic substitution.


# %%
second_order_slice = suzuki_trotter(term_count=2, order=2, steps=1)


@qm.qkernel
def surveyed_product_formula(
    terms: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
    steps: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    """Repeat one second-order Suzuki slice a symbolic number of times."""

    state = qm.qubits(1)
    for _ in range(steps):
        state = second_order_slice(state, terms, time / steps)
    return state


product_formula_study = qm.resources.logical(surveyed_product_formula)


def product_formula_resource_row(steps: int) -> tuple[int, int]:
    """Return Pauli-evolution and allocated-qubit counts for one step count."""

    resource_study = product_formula_study.evaluate(steps=steps)
    evolutions = resource_study.metric(pauli_evolutions).numeric_enclosure()
    allocated = resource_study.metric(
        qm.resources.LOGICAL_ALLOCATED_QUBITS
    ).numeric_enclosure()
    assert evolutions[0] == evolutions[1]
    assert allocated[0] == allocated[1]
    return evolutions[0], allocated[0]


product_formula_scaling = {
    steps: product_formula_resource_row(steps) for steps in (1, 2, 4, 8, 16)
}
for steps, (evolutions, allocated) in product_formula_scaling.items():
    assert evolutions == 3 * steps
    assert allocated == 1

print("second-order product formula (r: evolutions, qubits):", product_formula_scaling)

# %% [markdown]
# The exact finite count is $3r$, while the register remains one qubit. This
# reproduces the survey's “cost per slice times $r$” algebra and its no-ancilla
# statement. It does not claim that one abstract `pauli_evolution` equals one
# fault-tolerant elementary gate; that later realization is target dependent.

# %% [markdown]
# ## Convergence sweep
#
# We now sweep the number of Trotter steps $N$ and plot the fidelity error
# against the step size $\Delta t = T / N$ on a log-log axis. The expected
# slopes are:
#
# | Formula | Local error | Global norm error | Fidelity error ($1 -$ overlap) |
# |---------|-------------|-------------------|---------------------------------|
# | $S_1$   | $O(\Delta t^2)$ | $O(\Delta t)$   | $O(\Delta t^2)$  |
# | $S_2$   | $O(\Delta t^3)$ | $O(\Delta t^2)$ | $O(\Delta t^4)$  |
# | $S_4$   | $O(\Delta t^5)$ | $O(\Delta t^4)$ | $O(\Delta t^8)$  |
# | $S_6$   | $O(\Delta t^7)$ | $O(\Delta t^6)$ | $O(\Delta t^{12})$ |
#
# The fidelity error is the *square* of the state-norm error (to leading
# order): $1 - |\langle a | b \rangle| \approx \tfrac{1}{2}\lVert a - b \rVert^2$
# when the two vectors are close, so the plot below shows slopes of $2, 4, 8$
# rather than $1, 2, 4$.


# %% [markdown]
# The lines on the plot have slopes $\approx 2, 4, 8$, matching the
# fidelity-error orders in the table above. $S_4$ hits the float64 floor
# already at $N = 16$; $S_6$ is at the floor across the entire sweep, so its
# line appears flat (the expected $\Delta t^{12}$ slope is not resolvable on
# this 1-qubit problem in double precision).

# %%
drift_matrix = 0.5 * omega * pauli_z
drive_matrix = 0.5 * drive * pauli_x


def suzuki_step(order: int, duration: float) -> np.ndarray:
    """Return one dense Suzuki step for the two-term tutorial model."""

    if order == 1:
        return matrix_evolution(drive_matrix, duration) @ matrix_evolution(
            drift_matrix, duration
        )
    if order == 2:
        half_drift = matrix_evolution(drift_matrix, duration / 2.0)
        return half_drift @ matrix_evolution(drive_matrix, duration) @ half_drift
    coefficient = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    outer = suzuki_step(order - 2, coefficient * duration)
    middle = suzuki_step(order - 2, (1.0 - 4.0 * coefficient) * duration)
    return outer @ outer @ middle @ outer @ outer


step_counts = np.array([2, 4, 8, 16], dtype=np.int64)
step_sizes = evolution_time / step_counts
errors: dict[int, list[float]] = {1: [], 2: [], 4: []}

for steps in step_counts:
    for order in errors:
        step = suzuki_step(order, evolution_time / int(steps))
        approximation = np.linalg.matrix_power(step, int(steps)) @ initial_state
        errors[order].append(1.0 - abs(np.vdot(exact_state, approximation)))

slopes = {
    order: np.polyfit(
        np.log(step_sizes[:3] if order == 4 else step_sizes),
        np.log(values[:3] if order == 4 else values),
        1,
    )[0]
    for order, values in errors.items()
}

assert 1.7 < slopes[1] < 2.3
assert 3.7 < slopes[2] < 4.3
assert 7.0 < slopes[4] < 9.0
print("fidelity-error slopes:", slopes)

# %% [markdown]
# The measured slopes recover the expected $\Delta t^2$, $\Delta t^4$, and
# $\Delta t^8$ fidelity-error behavior. In an application, changing `steps`
# changes program structure, so build the desired bounded factory explicitly;
# runtime `time` values can still reuse the resulting artifact.

# %% [markdown]
# ## Compile and execute the same program
#
# `Reference` provides both a target and a local endpoint. Compilation seals
# the program digest, runtime parameter ABI, result ABI, and resource evidence
# in an immutable artifact. Execution consumes that artifact rather than
# tracing the Python function again.

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference)
record = qm.run(
    artifact,
    on=reference,
    parameters={"time": evolution_time},
    shots=128,
    seed=2,
)

assert artifact.source_digest.hex() == program.semantic_digest
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("time",)
assert record.result.shots == 128
assert sum(record.result.counts.values()) == 128

# %% [markdown]
# For this one-qubit model the exact excitation probability is
#
# $$
# P_1(t)=\left(\frac{\Omega}{E}\right)^2
# \sin^2\left(\frac{Et}{2}\right),
# \qquad E=\sqrt{\omega^2+\Omega^2}.
# $$
#
# The product formula is already much more accurate than the sampling
# uncertainty at this scale, so the sampled probability should lie within a
# conservative four-standard-deviation interval plus a small discretization
# allowance.

# %%
energy = math.hypot(omega, drive)
expected_probability = (drive / energy) ** 2 * math.sin(
    energy * evolution_time / 2.0
) ** 2
observed_probability = record.result.probability((1,))
sampling_bound = (
    4.0
    * math.sqrt(
        expected_probability * (1.0 - expected_probability) / record.result.shots
    )
    + 0.02
)

assert abs(observed_probability - expected_probability) <= sampling_bound
print(f"expected P(1): {expected_probability:.3f}")
print(f"sampled  P(1): {observed_probability:.3f}")

# %% [markdown]
# ## Summary
#
# - **Model**: a single-qubit Rabi Hamiltonian $H = H_z + H_x$ whose non-zero
#   commutator makes Trotter error measurable.
# - **`Const[Vector[Observable, ...]]` + `pauli_evolve`**: Hamiltonians are
#   specialized while evolution and loop structure remain abstract in QCore.
# - **Suzuki–Trotter fractal**: $S_{2k}$ nests five rescaled copies of
#   $S_{2k-2}$ with the level-specific coefficient $p_k = 1/(4 - 4^{1/(2k-1)})$.
# - **Bounded construction**: `suzuki_trotter` takes static term count, order,
#   and slice count; Rust derives the schedule used by the generated qkernel.
# - **Convergence**: the host matrix calculation checks fidelity-error slopes
#   of $2, 4, 8$. It is a numerical reference, separate from sampled execution.
