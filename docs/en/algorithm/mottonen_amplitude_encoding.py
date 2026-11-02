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
# ---
# tags: [algorithm, encoding, primitive]
# ---
#
# # Möttönen Amplitude Encoding
#
# This notebook prepares real and complex amplitude vectors with Qamomile's
# explicit Möttönen construction, verifies the resulting states and gate counts,
# and shows when to choose the method-specific API instead of generic amplitude
# encoding.
#
# ## Background
#
# **Amplitude encoding** prepares an $n$-qubit state from a unit-norm complex
# vector $a \in \mathbb{C}^{2^n}$:
#
# $$
# |0\rangle^{\otimes n}
# \longmapsto
# |\psi\rangle
# = \sum_{i=0}^{2^n - 1} a_i |i\rangle.
# $$
#
# Möttönen amplitude encoding is one algorithm that realizes this
# state-preparation map. The current stdlib exposes one portable
# `amplitude_encoding(...)` implementation, and that name means the explicit
# Möttönen/Gray-walk construction described here. Target adapters do not
# silently replace it with an unrelated native state-preparation method.
#
# The explicit implementation follows the uniformly controlled rotation
# construction of Möttönen, Vartiainen, Bergholm, and Salomaa
# {cite:p}`10.48550/arXiv.quant-ph/0407010`. The paper treats the more general
# arbitrary-state transformation $|a\rangle \to |b\rangle$; Qamomile implements
# the state-preparation half with the input fixed to $|0\rangle^{\otimes n}$.
#
# :::{note}
# If the input qubits are not in $|0\rangle^{\otimes n}$, neither API
# generally guarantees the output state.
# :::

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

import qamomile as qm
from qamomile.stdlib import amplitude_angles, amplitude_encoding_from_angles

# %% [markdown]
# ## Problem Settings
#
# We use two qubits and compare the prepared statevector with three normalized
# targets:
#
# - a positive real vector $(1, 2, 3, 4)$,
# - a signed real vector $(1, -1, 1, -1)$, and
# - a complex vector $(1, 1+i, 1-i, 2i)$.
#
# State fidelity is phase-invariant, so it verifies the intended physical state
# even if an engine chooses a different global phase.

# %%
positive_amplitudes = (1.0, 2.0, 3.0, 4.0)
signed_amplitudes = (1.0, -1.0, 1.0, -1.0)

# %% [markdown]
# ## Algorithm
#
# The Möttönen state-preparation construction has two cascades:
#
# 1. Uniformly controlled $R_y$ rotations distribute the magnitudes $|a_i|$
#    across computational-basis states. For real vectors, signed rotation angles
#    also encode negative amplitudes.
# 2. Uniformly controlled $R_z$ rotations restore relative phases. Qamomile
#    omits this cascade when every amplitude has zero imaginary part.
#
# Each uniformly controlled rotation is decomposed into elementary rotations and
# `CNOT` gates with the Gray-code construction. For Qamomile's
# $|0\rangle^{\otimes n} \to |\psi\rangle$ specialization, the resulting counts
# are:
#
# | Input | Rotations | `CNOT` gates |
# |---|---:|---:|
# | Real | $2^n - 1$ | $2^n - 2$ |
# | Complex | $2(2^n - 1)$ | $2(2^n - 2)$ |
#
# A $k$-controlled uniformly controlled rotation costs $2^k$ elementary
# rotations and $2^k$ `CNOT` gates. Summing the stages for
# $k = 0, 1, \ldots, n-1$ gives the formulas above; the uncontrolled stage is
# `CNOT`-free. The larger counts quoted in the paper's abstract apply to the
# full arbitrary-input transformation $|a\rangle \to |b\rangle$, rather than the
# state-preparation half implemented here. Qamomile also retains the plain
# per-stage decomposition instead of applying inter-cascade `CNOT`
# cancellation.
#
# ### Derive reusable coordinates
#
# We prepare the two-qubit complex vector
#
# $$
# (1,\;1+i,\;1-i,\;2i).
# $$
#
# Its squared norm is nine, so the computational-basis probabilities are
# $(1,2,2,4)/9$. `amplitude_angles` accepts the unnormalized vector and returns
# an immutable value containing three $R_y$ angles, three $R_z$ angles, and one
# retained global phase.

# %%
amplitudes = (
    1.0 + 0.0j,
    1.0 + 1.0j,
    1.0 - 1.0j,
    0.0 + 2.0j,
)
coordinates = amplitude_angles(amplitudes)

assert coordinates.width == 2
assert len(coordinates.ry) == 3
assert len(coordinates.rz) == 3
assert all(math.isfinite(value) for value in coordinates.ry)
assert all(math.isfinite(value) for value in coordinates.rz)
assert math.isfinite(coordinates.phase)
print(coordinates)

# %% [markdown]
# Coordinate synthesis is host-side preprocessing over finite classical data.
# It crosses one required Rust boundary and does not create a Python numerical
# fallback. Invalid length, nonfinite input, the zero vector, or an excessive
# width fails before QCore construction.

# %% [markdown]
# ## Implementation
#
# ### Compile once with runtime-rebindable angles
#
# `amplitude_encoding_from_angles(2)` fixes only the two-qubit topology. The
# coordinates remain aggregate runtime parameters, so one artifact can prepare
# many vectors without recompilation. Their shapes are part of the function
# type and the generated launch ABI.

# %%
encode = amplitude_encoding_from_angles(2)


@qm.qkernel
def prepared_state(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],
    phase: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[2]]:
    """Prepare one two-qubit state from Möttönen coordinates.

    Args:
        ry_angles (qm.Param[qm.Vector[qm.Real, 3]]): Magnitude-cascade angles.
        rz_angles (qm.Param[qm.Vector[qm.Real, 3]]): Relative-phase angles.
        phase (qm.Param[qm.Real]): Retained global phase.

    Returns:
        qm.Q[qm.Bits[2]]: Prepared two-qubit state.
    """

    return encode(qm.qubits(2), ry_angles, rz_angles, phase)


@qm.qkernel
def prepared_samples(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],
    phase: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    """Measure the reusable two-qubit preparation.

    Args:
        ry_angles (qm.Param[qm.Vector[qm.Real, 3]]): Magnitude-cascade angles.
        rz_angles (qm.Param[qm.Vector[qm.Real, 3]]): Relative-phase angles.
        phase (qm.Param[qm.Real]): Retained global phase.

    Returns:
        qm.Shot[qm.Bits[2]]: Sampled computational-basis state.
    """

    return qm.measure(prepared_state(ry_angles, rz_angles, phase))


# %% [markdown]
# For a vector fixed at compile time, `amplitude_encoding(amplitudes)` returns a
# closed unary transform instead. Both forms use the same Rust coordinates and
# ordinary gate vocabulary; choose the runtime form only when artifact reuse is
# useful.

# %% [markdown]
# ### Build kernels for resource checks
#
# The verified program contains no amplitude-preparation opcode. Resource
# analysis composes its ordinary rotations and controlled operations directly.

# %%
program = qm.specialize(prepared_samples)
study = qm.resources.logical(program)
inspection = program.inspect()

assert "quantum.amplitude" not in inspection
assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
    6,
    6,
)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cx")
).numeric_enclosure() == (4, 4)

# %% [markdown]
# A general $n$-qubit complex preparation uses $2(2^n-1)$ rotations and
# $2(2^n-2)$ `CX` gates in this decomposition. The exponential cost is visible
# in the estimate rather than hidden behind a backend-native instruction.

# %% [markdown]
# ## Result
#
# ### State magnitudes
#
# Compilation seals the three aggregate parameter slots. `bindings()` produces
# exactly the matching launch mapping, including the global phase.

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference)
parameters = coordinates.bindings()
record = qm.run(
    artifact,
    on=reference,
    parameters=parameters,
    shots=128,
    seed=2,
)

assert tuple(slot.name for slot in artifact.parameter_abi.slots) == (
    "ry_angles",
    "rz_angles",
    "phase",
)
assert record.result.shots == 128
assert sum(record.result.counts.values()) == 128

expected_probabilities = {
    (0, 0): 1.0 / 9.0,
    (1, 0): 2.0 / 9.0,
    (0, 1): 2.0 / 9.0,
    (1, 1): 4.0 / 9.0,
}
for bits, expected in expected_probabilities.items():
    sampling_bound = 5.0 * math.sqrt(expected * (1.0 - expected) / record.result.shots)
    assert abs(record.result.probability(bits) - expected) <= sampling_bound

observed_probabilities = {
    bits: record.result.probability(bits) for bits in expected_probabilities
}
print("sampled probabilities:", observed_probabilities)

# %% [markdown]
# ### Runtime rebinding
#
# The same artifact can prepare all three targets because only the six
# rotation angles and global phase change. Structure, result ABI, and target
# payload stay fixed.

# %%
for trial_amplitudes in (
    positive_amplitudes,
    signed_amplitudes,
    amplitudes,
):
    trial_coordinates = amplitude_angles(trial_amplitudes)
    trial_record = qm.run(
        artifact,
        on=reference,
        parameters=trial_coordinates.bindings(),
        shots=128,
        seed=3,
    )
    norm = sum(abs(value) ** 2 for value in trial_amplitudes)
    for index, value in enumerate(trial_amplitudes):
        bits = (index & 1, (index >> 1) & 1)
        expected = abs(value) ** 2 / norm
        bound = 5.0 * math.sqrt(expected * (1.0 - expected) / trial_record.result.shots)
        assert abs(trial_record.result.probability(bits) - expected) <= bound

# %% [markdown]
# Sampling checks magnitudes. Sign and complex phase require an interference
# witness, which is why the next section evaluates observables as well.

# %% [markdown]
# ## Verify relative phase
#
# Computational-basis probabilities cannot distinguish complex phases. The
# observable $Y_0X_1$ has exact expectation $8/9$ for this vector, so it gives a
# phase-sensitive witness using the same `prepared_state` implementation.

# %%
phase_witness = qm.expectation(
    prepared_state,
    qm.Y(0) * qm.X(1),
    on=reference,
    parameters=parameters,
    shots=64,
    seed=2,
)
expected_phase_witness = 8.0 / 9.0
phase_sampling_bound = 5.0 * math.sqrt(
    (1.0 - expected_phase_witness**2) / phase_witness.shots
)

assert abs(phase_witness.value - expected_phase_witness) <= phase_sampling_bound
print(f"<Y0 X1>: {phase_witness.value:.3f}")

# %%
positive_coordinates = amplitude_angles(positive_amplitudes)
signed_coordinates = amplitude_angles(signed_amplitudes)
positive_witness = qm.expectation(
    prepared_state,
    qm.Z(0),
    on=reference,
    parameters=positive_coordinates.bindings(),
    shots=64,
    seed=4,
)
signed_witness = qm.expectation(
    prepared_state,
    qm.X(0),
    on=reference,
    parameters=signed_coordinates.bindings(),
    shots=64,
    seed=4,
)

assert abs(positive_witness.value + 1.0 / 3.0) <= 5.0 * math.sqrt(
    (1.0 - (1.0 / 3.0) ** 2) / positive_witness.shots
)
assert abs(signed_witness.value + 1.0) <= 0.1

# %% [markdown]
# For $(1,2,3,4)$, $\langle Z_0\rangle=-1/3$. For
# $(1,-1,1,-1)$, $\langle X_0\rangle=-1$, which proves that the negative signs
# survive even though computational-basis probabilities cannot see them.

# %% [markdown]
# ## Fixed amplitudes versus reusable angles
#
# `amplitude_encoding(amplitudes)` computes and closes the coordinates at
# Python factory time. Use it when the target vector is part of program
# structure. `amplitude_encoding_from_angles(width)` fixes only the topology and
# leaves typed angle vectors in the runtime ABI, as this page does. Both are the
# same Möttönen synthesis and therefore have the same gate-count formulas.

# %% [markdown]
# ## Summary
#
# - Rust owns normalization, exact-phase coordinates, Gray-walk topology, and
#   width budgets.
# - Aggregate `Param[Vector[Real, 3]]` annotations make the reusable ABI typed.
# - The program is ordinary `RY`, `RZ`, `CX`, and global-phase composition.
# - The same source supports exact logical resources, compilation, sampling,
#   and phase-sensitive expectation values.
# - One artifact is rebound across positive, signed-real, and complex targets.
# - All executable evidence stays at two qubits; wider scaling is symbolic.
#
# Next, [Multidimensional QFT](multidimensional_qft.ipynb) composes state
# preparation with independent Fourier transforms along two logical axes.
