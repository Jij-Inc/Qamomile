# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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

# %%
# From the repository root, install this Qamomile checkout and Matplotlib.
# # pip install -e . matplotlib

# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import qamomile as qm
from qamomile.stdlib import amplitude_encoding, qft

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 1 if docs_test_mode else 2**14

# %% [markdown]
# ## Background
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

# %% [markdown]
# ## Problem Settings
#
# First, create input data.
# Here we use the following hypothetical periodic data.

# %%
Nx = 6
Ny = 6
xs = np.arange(Nx)
ys = np.arange(Ny)
X, Y = np.meshgrid(xs, ys, indexing="ij")
f = np.cos(np.pi * X / 2) * np.cos(np.pi * Y / 2)

# %% [markdown]
# The created input data is two-dimensional data with simple periodicity.

# %%
im1 = plt.pcolormesh(f.T, cmap="viridis")
cbar = plt.colorbar(im1)
cbar.set_label("f", fontsize=14)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()


# %% [markdown]
# ## Algorithm
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
# The preprocessing proposed in [](https://doi.org/10.1039/d4cp04399e) adapts the input to a power-of-two grid before the QFT.
# Qamomile includes a standard implementation of one-dimensional QFT.
# By using Qamomile's features, multidimensional QFT can also be implemented easily.
#
# Real-space crystal and nanosheet structures are not generally sampled on power-of-two grids.
# Their lattice periodicities can correspond to arbitrary integer grid counts.
# The method in [](https://doi.org/10.1039/d4cp04399e) therefore proposed a preprocessing technique for encoding crystal periodicity directly into a quantum state.
#
# Real crystal-structure data is not always defined on a power-of-two grid.
# If the number of sampled points along a dimension is larger than $2^p$, we can make QFT applicable by extracting only the first $2^p$ points.
# If the number of sampled points along a dimension is smaller than $2^p$, we can also make QFT applicable by adding zeros before and after the data.
#
# Truncation can introduce spectral leakage, producing sidelobes around the main spectral peaks.
# These sidelobes are artificial spectral components introduced by the finite-domain treatment.
# Zero padding samples this spectrum more finely.
# To suppress them, we introduce a weighting function, or window function, that smoothly approaches zero at both ends of the truncated or zero-padded input interval.
# The paper uses the following flat-top window function.
#
# $$
# w_5(n) = \sum_{k=0}^4 (-1)^k a_k \cos \left( \frac{2\pi k n}{N}\right)
# $$
#
# Here $(a_0, a_1, a_2, a_3, a_4) = (0.1881, 0.36923, 0.28702, 0.13077, 0.02488)$, $N$ is the total number of points in the dimension being considered, and $n$ is the point position $(n=0, 1, \dots, N-1)$.
# In two dimensions, we multiply the input by a product such as $w_5(x) w_5(y)$.

# %% [markdown]
# ## Implementation
#
# Let us implement a quantum circuit for multidimensional QFT with Qamomile.
#
# Because this data size is not a power of two, the implemented QFT cannot be applied to it as is.
# We therefore apply zero padding to extend the size to a power of two.
# Here we use three qubits for each coordinate register, one for the $x$ direction and one for the $y$ direction.
# In other words, we zero-pad the data so that each dimension has $2^3 = 8$ grid points.

# %%
def crop_and_pad(
    array_2d: np.ndarray, num_x_target: int, num_y_target: int
) -> np.ndarray:
    array_2d = array_2d[:num_x_target, :num_y_target]
    num_x, num_y = array_2d.shape
    pad_width_x = num_x_target - num_x
    pad_width_y = num_y_target - num_y
    pad_x_left = int(np.ceil(pad_width_x / 2.0))
    pad_x_right = int(np.floor(pad_width_x / 2.0))
    pad_y_left = int(np.ceil(pad_width_y / 2.0))
    pad_y_right = int(np.floor(pad_width_y / 2.0))
    array_2d_padding = np.pad(
        array_2d, ((pad_x_left, pad_x_right), (pad_y_left, pad_y_right))
    )
    return array_2d_padding


Nqx = 3
Nqy = 3
Nx_target = 2**Nqx
Ny_target = 2**Nqy
f_padding = crop_and_pad(f, Nx_target, Ny_target)

# %% [markdown]
# Let us inspect the zero-padded data.

# %%
im2 = plt.pcolor(f_padding.T, cmap="viridis")
cbar = plt.colorbar(im2)
cbar.set_label("f", fontsize=14)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()


# %% [markdown]
# Next, implement a window function to suppress side lobes.

# %%
def w5(N: int) -> np.ndarray:
    a = np.array([0.1881, 0.36923, 0.28702, 0.13077, 0.02488])
    n = np.arange(N)
    results = [(-1) ** k * a[k] * np.cos(2 * np.pi * k * n / N) for k in range(5)]
    return sum(results)


wx = w5(Nx_target)
wy = w5(Ny_target)
w2d = wx[:, None] * wy[None, :]

# %% [markdown]
# Let us plot the overall shape of the window function.

# %%
im3 = plt.pcolor(w2d.T, cmap="plasma")
cbar = plt.colorbar(im3)
cbar.set_label("Flat-top window function", fontsize=14)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.show()


# %% [markdown]
# The plot shows that the window gives the greatest weight to the central region and small weights near the edges.
# Now apply this window function to the zero-padded data and run the two-dimensional QFT.

# %% [markdown]
# Use Qamomile's `qft` function to implement multidimensional QFT.

# %%
def make_multidimensional_qft(inputs: np.ndarray, nqx: int, nqy: int) -> qm.QKernel:
    if nqx < 1 or nqy < 1 or inputs.shape != (2**nqx, 2**nqy):
        raise ValueError("The input shape must match the two nonempty qubit registers.")
    width = nqx + nqy
    prepare = amplitude_encoding(tuple(inputs.ravel(order="F")))

    @qm.qkernel
    def qft_for_multidimension() -> qm.Shot[qm.Bits[width]]:
        state = prepare(qm.qubits(width))
        x_axis, y_axis = qm.split_at(state, nqx)
        x_axis = qft(x_axis, nqx)
        y_axis = qft(y_axis, nqy)
        return qm.measure(qm.concat(x_axis, y_axis))

    return qft_for_multidimension

# %% [markdown]
# We can now apply QFT.

# %% [markdown]
# Compile the QFT circuit implemented above for Qamomile's built-in simulator, `qm.Reference()`.
# Flatten the input with $x$ varying fastest so that the low-order qubits encode $x$.

# %%
reference = qm.Reference()
program = qm.specialize(make_multidimensional_qft(f_padding, Nqx, Nqy))
exe = qm.compile(program, to=reference)

# %% [markdown]
# ## Result
#
# Run the compiled program and map the sampled bitstrings to probabilities on the two-dimensional reciprocal-space grid. Here $k_x$ and $k_y$ are the frequency indices along $x$ and $y$.

# %%
def compute_prob(result: qm.SampleResult, nqx: int, nqy: int) -> np.ndarray:
    prob = np.zeros((2**nqx, 2**nqy))
    for bits, count in result.counts.items():
        kx = sum(bits[i] << i for i in range(nqx))
        ky = sum(bits[nqx + i] << i for i in range(nqy))
        prob[kx, ky] += count / result.shots
    return prob


result1 = qm.run(exe, on=reference, shots=sample_shots, seed=5).result
prob = compute_prob(result1, Nqx, Nqy)

# %% [markdown]
# For comparison, compute the result obtained by applying the classical FFT to the same input.

# %%
def compute_classical_fft(f: np.ndarray) -> np.ndarray:
    Nx_target, Ny_target = f.shape
    f_normalized = f / np.linalg.norm(f)
    fft = np.fft.ifft2(f_normalized) * np.sqrt(Nx_target * Ny_target)
    prob = np.abs(fft) ** 2
    return prob


prob_classical = compute_classical_fft(f_padding)
assert np.allclose(prob, prob_classical, atol=3 / np.sqrt(sample_shots))

# %% [markdown]
# Visualize the results.

# %%
vmin = min(prob_classical.min(), prob.min())
vmax = max(prob_classical.max(), prob.max())
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im4 = axes[0].pcolor(prob_classical.T, vmin=vmin, vmax=vmax, cmap="viridis")
axes[0].set_title("Classical")
axes[0].set_xlabel("kx")
axes[0].set_ylabel("ky")
im5 = axes[1].pcolor(prob.T, vmin=vmin, vmax=vmax, cmap="viridis")
axes[1].set_title("Quantum")
axes[1].set_xlabel("kx")
print("Classical peak value: {}".format(prob_classical.max()))
print("Quantum peak value: {}".format(prob.max()))
fig.colorbar(im5, ax=axes, fraction=0.046, pad=0.04)
plt.show()

# %% [markdown]
# Both the FFT reference and the QFT sampling result show similar sidelobe structure.

# %% [markdown]
# Repeat the same calculation with the input after applying the window function.

# %%
fw = w2d * f_padding
program = qm.specialize(make_multidimensional_qft(fw, Nqx, Nqy))
exe = qm.compile(program, to=reference)
result2 = qm.run(exe, on=reference, shots=sample_shots, seed=7).result
prob2 = compute_prob(result2, Nqx, Nqy)

# %% [markdown]
# For comparison, also apply the classical FFT to the input with the window function applied.

# %%
prob_classical2 = compute_classical_fft(fw)
assert np.allclose(prob2, prob_classical2, atol=3 / np.sqrt(sample_shots))

# %% [markdown]
# Visualize these two results.

# %%
vmin = min(prob_classical2.min(), prob2.min())
vmax = max(prob_classical2.max(), prob2.max())
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im6 = axes[0].pcolor(prob_classical2.T, vmin=vmin, vmax=vmax, cmap="viridis")
axes[0].set_title("Classical")
axes[0].set_xlabel("kx")
axes[0].set_ylabel("ky")
im7 = axes[1].pcolor(prob2.T, vmin=vmin, vmax=vmax, cmap="viridis")
axes[1].set_title("Quantum")
axes[1].set_xlabel("kx")
print("Classical peak value: {}".format(prob_classical2.max()))
print("Quantum peak value: {}".format(prob2.max()))
fig.colorbar(im7, ax=axes, fraction=0.046, pad=0.04)
plt.show()

# %% [markdown]
# In this example, applying the window lowers the peak probability and spreads it over neighboring frequency bins.

# %% [markdown]
# ## Summary
#
# This article introduced a paper that developed preprocessing techniques for input data whose size is not a power of two, and showed how to implement QFT with Qamomile.
# The key points from this page are summarized below.
#
# * Qamomile includes a standard feature for implementing QFT.
# * If the input size is not a power of two, QFT can be applied by truncating or zero-padding the domain.
# * Sidelobes caused by domain truncation can be suppressed with a window function.
