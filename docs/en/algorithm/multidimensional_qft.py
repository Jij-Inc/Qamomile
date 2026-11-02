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
# tags: [algorithm, chemistry, simulation]
# ---
#
# # Multidimensional QFT for Estimating Nanosheet Material Properties
#
# The quantum Fourier transform (QFT) is widely used as a core primitive in quantum algorithms, and many applications have been proposed.
# Conventional QFT-based constructions, however, are limited when input data is not naturally defined on power-of-two grid sizes.
# This article summarizes an implementation of a multidimensional QFT for inputs with arbitrary grid sizes, following the method proposed in [](https://doi.org/10.1039/d4cp04399e).
# Through this multidimensional QFT implementation, you can learn how to use Qamomile for this kind of workflow.
#
# ## Background
#
# ### Problem: Characterizing nanosheet materials
#
# Nanosheet materials have attracted attention for a wide range of chemical applications.
# Their physicochemical properties can change dramatically with thickness.
# Therefore, thickness-controlled synthesis and layer-number identification are crucial for controlling the diverse properties of nanosheet materials.
# Transmission electron microscopy (TEM) is a promising way to identify the number of layers.
# TEM observations provide both transmission images and selected area electron diffraction (SAED) patterns.
# This has made computational simulation of SAED patterns for nanosheet materials a promising approach.
#
# One established approach is to compute the discrete Fourier transform with the classical fast Fourier transform (FFT).
# FFT is very fast, with known complexity $\mathcal{O} (N \log_2 N)$, where $N$ is the number of grid points.
# If the number of grid points is represented as a power of two, $N = 2^n$, the complexity becomes $\mathcal{O} (n 2^n)$, which becomes difficult as the exponent $n$ grows.
# As a promising alternative, the quantum Fourier transform (QFT) has recently attracted attention in quantum computing.
# The standard exact QFT unitary on $n$ qubits can be implemented with $\mathcal{O} (n^2)$ gates, which motivates its use as a quantum Fourier-transform primitive. State preparation and readout costs must still be considered separately.
# The paper therefore simulated approximate SAED patterns of $\mathrm{CeO_2}$ nanosheets with QFT, aiming to explore the applicability of multidimensional QFT and characterize usable input data in more detail.
#
# ### Multidimensional QFT
#
# QFT is well known as a core technique in Shor's algorithm and quantum phase estimation.
# A relevant extension of QFT is described in [](https://doi.org/10.48550/arXiv.2301.13835).
# That work derived an efficient circuit for multidimensional QFT based on known QFT circuits.
#
# ```{figure} assets/multidimensional_qft_01.png
# :width: 100%
# :alt: quantum circuit to calculate the d-dimensional FFT from Pfeffer (2023)
#
# Quantum circuit diagram for implementing a $d$-dimensional Fourier transform, from [](https://doi.org/10.48550/arXiv.2301.13835).
# ```
#
# The ${\mathrm{initialize}}(\vert v \rangle)$ block in the figure represents a subroutine that initializes the quantum state according to the input data.
# Several state-preparation methods are available; the implementation below uses the [Möttönen state-preparation routine for amplitude encoding](mottonen_amplitude_encoding).
# However, the method in the figure assumes that each dimension size satisfies $N_i = 2^{n_i}$.
# The method proposed in [](https://doi.org/10.1039/d4cp04399e) removes this restriction.
# Qamomile includes a standard implementation of one-dimensional QFT.
# By using Qamomile's features, multidimensional QFT can also be implemented easily.
#

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
from __future__ import annotations

import numpy as np

import qamomile as qm
from qamomile.stdlib import amplitude_encoding, qft

# %% [markdown]
# ## Encode a small two-dimensional input
#
# We use a $4\times4$ grid with little-endian flat index
#
# $$
# j=x+4y.
# $$
#
# The input is uniform along the $x$ axis and localized at $y=1$. Its
# normalized state is therefore
#
# $$
# \frac{1}{2}\sum_{x=0}^{3}|x\rangle|1\rangle.
# $$
#
# The $x$-axis QFT must produce reciprocal coordinate zero. The localized
# $y$ coordinate transforms to equal support over all four reciprocal values.

# %%
side = 4
input_amplitudes = tuple(
    1.0 if side <= index < 2 * side else 0.0 for index in range(side * side)
)
prepare = amplitude_encoding(input_amplitudes)

assert len(input_amplitudes) == 16
assert sum(value * value for value in input_amplitudes) == 4.0

# %% [markdown]
# `amplitude_encoding` normalizes the finite vector and closes Rust-derived
# exact-phase coordinates into an ordinary unitary Qamomile function. The input
# does not become an SDK state-preparation object.

# %% [markdown]
# ## Algorithm: QFT for multidimensional inputs with arbitrary periodicity
#
# Real-space crystal and nanosheet structures are not generally sampled on power-of-two grids.
# Their lattice periodicities can correspond to arbitrary integer grid counts.
# The method in [](https://doi.org/10.1039/d4cp04399e) therefore proposed a preprocessing technique for encoding crystal periodicity directly into a quantum state.
#
# ### Domain truncation and zero padding
#
# Real crystal-structure data is not always defined on a power-of-two grid.
# If the number of sampled points along a dimension is larger than $2^p$, we can make QFT applicable by extracting only the first $2^p$ points.
# If the number of sampled points along a dimension is smaller than $2^p$, we can also make QFT applicable by adding zeros before and after the data.
#
# ### Applying a flat-top window function
#
# Truncation and zero padding can introduce spectral leakage, producing sidelobes around the main spectral peaks.
# These sidelobes are artificial spectral components introduced by the finite-domain treatment.
# To suppress them, we introduce a weighting function, or window function, that smoothly approaches zero at both ends of the truncated or zero-padded input interval.
# The paper uses the following flat-top window function.
#
# $$
# w_5(n) = \sum_{k=0}^4 (-1)^k a_k \cos \left( \frac{2\pi k n}{N}\right)
# $$
#
# Here $(a_0, a_1, a_2, a_3, a_4) = (0.1881, 0.36923, 0.28702, 0.13077, 0.02488)$, $N$ is the total number of points in the dimension being considered, and $n$ is the point position $(n=0, 1, \dots, N-1)$.
# In two dimensions, we multiply the input by a product such as $w_5(x) w_5(y)$.


# %%
def pad_center(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-pad a two-dimensional array to one power-of-two shape."""

    padding = []
    for size, target in zip(array.shape, target_shape, strict=True):
        total = target - size
        padding.append(((total + 1) // 2, total // 2))
    return np.pad(array, tuple(padding))


def flat_top(size: int) -> np.ndarray:
    """Return the five-term flat-top window used by the workflow."""

    coefficients = np.array([0.1881, 0.36923, 0.28702, 0.13077, 0.02488])
    positions = np.arange(size)
    return sum(
        (-1) ** order * coefficient * np.cos(2.0 * np.pi * order * positions / size)
        for order, coefficient in enumerate(coefficients)
    )


source_side = 6
x, y = np.meshgrid(
    np.arange(source_side),
    np.arange(source_side),
    indexing="ij",
)
periodic_data = np.cos(np.pi * x / 2.0) * np.cos(np.pi * y / 2.0)
padded_data = pad_center(periodic_data, (8, 8))
window_2d = flat_top(8)[:, None] * flat_top(8)[None, :]
windowed_data = padded_data * window_2d

assert periodic_data.shape == (6, 6)
assert padded_data.shape == (8, 8)
assert windowed_data.shape == (8, 8)

# %% [markdown]
# ## Implementation with Qamomile
#
# ### Factor, transform, and rejoin the axes
#
# `split_at(state, 2)` consumes `Q[Bits[4]]` and returns two owned `Q[Bits[2]]`
# values in source order. Each component is consumed by the same `qft` call.
# `concat` then consumes both results and reconstructs `Q[Bits[4]]`.


# %%
@qm.qkernel
def two_axis_qft_state() -> qm.Q[qm.Bits[4]]:
    """Prepare the grid and apply one QFT to each logical axis.

    Returns:
        qm.Q[qm.Bits[4]]: Reciprocal-grid state with two two-bit axes.
    """

    state = prepare(qm.qubits(4))
    x_axis, y_axis = qm.split_at(state, 2)
    x_axis = qft(x_axis, 2)
    y_axis = qft(y_axis, 2)
    return qm.concat(x_axis, y_axis)


@qm.qkernel
def two_axis_qft_samples() -> qm.Shot[qm.Bits[4]]:
    """Measure the two-dimensional reciprocal coordinate.

    Returns:
        qm.Shot[qm.Bits[4]]: Two little-endian two-bit coordinates.
    """

    return qm.measure(two_axis_qft_state())


# %% [markdown]
# No slice aliases or overlapping views exist. A missing return, duplicated
# component, reversed width equation, or second use of an owned state is a type
# or ownership error before target lowering.

# %% [markdown]
# ## Inspect and estimate the same program
#
# The specialized QCore program retains one `qft` definition and two call
# edges. Structural split and concatenation operations become checked zero-cost
# reshapes; there is no multidimensional-QFT opcode or backend algorithm case.

# %%
program = qm.specialize(two_axis_qft_samples)
inspection = program.inspect()
qft_symbols = {
    function.symbol
    for function in program.functions
    if function.symbol.rsplit(".", 1)[-1] == "qft"
}

assert len(qft_symbols) == 1
assert inspection.count(f"callee='{next(iter(qft_symbols))}'") == 2
assert "quantum.multidimensional_qft" not in inspection

study = qm.resources.logical(program)
assert study.metric(
    qm.resources.logical_operation_count("quantum.h")
).numeric_enclosure() == (4, 4)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cp")
).numeric_enclosure() == (2, 2)
assert study.metric(
    qm.resources.logical_operation_count("quantum.swap")
).numeric_enclosure() == (2, 2)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# The exact QFT contribution is derived from the same call graph that will be
# lowered for execution. State-preparation work remains visible separately; no
# backend-native box can cause the logical estimate to report zero.

# %% [markdown]
# ### Reproduce the survey's QFT scaling
#
# Section 12 of [](https://doi.org/10.48550/arXiv.2310.03011) states that a
# textbook QFT on $N=2^n$ amplitudes uses $O(\log N)$ qubits and
# $O(\log^2 N)$ quantum gates. The circuit in that section consists of $n$
# Hadamards, $n(n-1)/2$ controlled phase shifts, and $\lfloor n/2\rfloor$
# final swaps. We can check those finite coefficients directly instead of
# fitting a slope to five aggregate totals. The width remains symbolic while
# the compiler analyzes the QFT body once; `evaluate` only substitutes the
# widths displayed in the table below.


# %%
@qm.qkernel
def surveyed_qft(width: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[width]]:
    """Build the textbook QFT family used by the resource sweep."""

    return qm.measure(qft(qm.qubits(width), width))


qft_scaling_study = qm.resources.logical(surveyed_qft)


def qft_resource_row(width: int) -> tuple[int, int, int, int]:
    """Return exact textbook-QFT resource counts for one register width."""

    qft_study = qft_scaling_study.evaluate(width=width)
    hadamards = qft_study.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure()
    controlled_phases = qft_study.metric(
        qm.resources.logical_operation_count("quantum.cp")
    ).numeric_enclosure()
    swaps = qft_study.metric(
        qm.resources.logical_operation_count("quantum.swap")
    ).numeric_enclosure()
    peak = qft_study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure()
    assert hadamards[0] == hadamards[1]
    assert controlled_phases[0] == controlled_phases[1]
    assert swaps[0] == swaps[1]
    assert peak[0] == peak[1]
    return hadamards[0], controlled_phases[0], swaps[0], peak[0]


qft_scaling = {width: qft_resource_row(width) for width in range(2, 9)}
for width, (hadamards, controlled_phases, swaps, peak) in qft_scaling.items():
    assert hadamards == width
    assert controlled_phases == width * (width - 1) // 2
    assert swaps == width // 2
    assert peak == width

print("QFT scaling (n: H, CP, SWAP, peak qubits):", qft_scaling)

# %% [markdown]
# Qamomile reproduces the finite textbook circuit exactly. Therefore the
# dominant controlled-phase count is $\Theta(n^2)=\Theta(\log^2 N)$ and the
# peak width is $n=\log_2N$, matching the survey. This is a logical-circuit
# comparison: amplitude loading and tomography are deliberately outside the
# QFT body, just as the survey's caveat requires.

# %% [markdown]
# ## Compile and execute
#
# `ResultABI` preserves the four logical bits in source order. The first two
# bits are the little-endian $x$ coordinate and the final two are the
# little-endian $y$ coordinate, regardless of a backend's native bit order.

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference)
record = qm.run(
    artifact,
    on=reference,
    shots=32,
    seed=2,
)

observed_y = {
    sum(bit << index for index, bit in enumerate(bits[2:]))
    for bits in record.result.counts
}

assert record.result.shots == 32
assert sum(record.result.counts.values()) == 32
assert all(bits[:2] == (0, 0) for bits in record.result.counts)
assert observed_y == set(range(side))
print("reciprocal-grid samples:", record.result.counts)

# %% [markdown]
# Every sample has $k_x=0$, as required by the uniform $x$ input. All four
# $k_y$ values appear because a basis-localized $y$ input has uniform Fourier
# magnitude. These structural facts expose an axis swap or result-bit reversal
# without a large statistical comparison.

# %% [markdown]
# ## Results
#
# ### Comparison with a Classical Method: Fast Fourier Transform (FFT)
#
# We now run the same composition on the padded 6×6 periodic input. The helper
# below closes one normalized amplitude vector, factors six qubits into two
# three-qubit axes, and applies the reusable QFT to both. A classical inverse
# FFT uses the same positive-exponent convention as this QFT; magnitudes are
# independent of the remaining phase convention.


# %%
def make_grid_qft(amplitudes: tuple[float, ...]) -> qm.QKernel:
    """Build one fixed six-qubit two-axis QFT program."""

    prepare_grid = amplitude_encoding(amplitudes)

    @qm.qkernel
    def grid_qft() -> qm.Shot[qm.Bits[6]]:
        state = prepare_grid(qm.qubits(6))
        x_axis, y_axis = qm.split_at(state, 3)
        x_axis = qft(x_axis, 3)
        y_axis = qft(y_axis, 3)
        return qm.measure(qm.concat(x_axis, y_axis))

    return grid_qft


def classical_probabilities(array: np.ndarray) -> np.ndarray:
    """Return normalized two-dimensional inverse-FFT probabilities."""

    normalized = array / np.linalg.norm(array)
    transformed = np.fft.ifft2(normalized) * np.sqrt(array.size)
    return np.abs(transformed) ** 2


def sampled_grid(array: np.ndarray, seed: int) -> np.ndarray:
    """Execute the QFT program and return an 8×8 sampled distribution."""

    kernel = make_grid_qft(tuple(array.reshape(-1)))
    result = qm.run(
        qm.specialize(kernel),
        on=qm.Reference(),
        shots=512,
        seed=seed,
    ).result
    probabilities = np.zeros((8, 8))
    for bits, count in result.counts.items():
        kx = sum(bit << index for index, bit in enumerate(bits[:3]))
        ky = sum(bit << index for index, bit in enumerate(bits[3:]))
        probabilities[kx, ky] += count / result.shots
    return probabilities


padded_classical = classical_probabilities(padded_data)
padded_quantum = sampled_grid(padded_data, seed=5)
windowed_classical = classical_probabilities(windowed_data)
windowed_quantum = sampled_grid(windowed_data, seed=7)

padded_distance = np.abs(padded_quantum - padded_classical).sum()
windowed_distance = np.abs(windowed_quantum - windowed_classical).sum()
print("distribution distances:", padded_distance, windowed_distance)
assert padded_distance < 0.5
assert windowed_distance < 0.5
print("padded peak:", np.unravel_index(padded_quantum.argmax(), (8, 8)))
print("windowed peak:", np.unravel_index(windowed_quantum.argmax(), (8, 8)))

# %% [markdown]
# The sampled QFT distributions agree with the classical FFT references within
# finite-shot variation. Applying the window broadens the main lobe but
# localizes boundary-induced sidelobes, reproducing the preprocessing tradeoff
# from the nanosheet workflow.

# %% [markdown]
# ## Summary
#
# This article introduced a paper that developed preprocessing techniques for input data whose size is not a power of two, and showed how to implement QFT with Qamomile.
# The key points from this page are summarized below.
#
# * Qamomile includes a standard feature for implementing QFT.
# * If the input size is not a power of two, QFT can be applied by truncating or zero-padding the domain.
# * Artifacts caused by domain truncation or zero padding can be suppressed with a window function.
