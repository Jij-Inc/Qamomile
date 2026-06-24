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
# # Introduction to Quantum Fourier transform (QFT)
#
# The Quantum Fourier Transform (QFT) is the quantum analogue of the discrete Fourier transform. It appears as a core subroutine in quantum phase estimation, order finding, and many algorithms that need to move information between a computational-basis description and a phase description {cite:p}`10.48550/arXiv.quant-ph/0201067`.
#
# This page introduces the classical Fourier transform viewpoint, explains the QFT circuit at a high level, and then implements a four-qubit frequency-estimation example. By the end, you will have a Qamomile qkernel that draws the QFT circuit, executes it on a local Qiskit simulator, estimates the dominant frequency from a histogram, and uses resource estimation to inspect the symbolic scaling of the standard decomposition.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.qft import qft
from qamomile.qiskit import QiskitTranspiler
from qiskit_aer import AerSimulator

transpiler = QiskitTranspiler()

# %% [markdown]
# ## Background: Fourier Transform
#
# A Fourier transform rewrites data in terms of frequencies. In continuous settings, the input is a function over time or space and the output tells us how much of each frequency is present. In finite computation, the version we usually meet is the **Discrete Fourier Transform** (DFT).
#
# For a vector $x = (x_0, x_1, \ldots, x_{N-1})$, the unit-normalized DFT convention used by QFT is
#
# $$
# y_k = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} x_j e^{2\pi i jk / N},
# \qquad k = 0, 1, \ldots, N-1.
# $$
#
# The output index $k$ labels a frequency component. The complex phase factor $e^{2\pi i jk / N}$ is what makes each input position contribute with a different phase to each output frequency.

# %% [markdown]
# ## Algorithm
#
# QFT applies the same mathematical transform to quantum amplitudes. If $N = 2^n$, then on computational basis states it acts as
#
# $$
# \mathrm{QFT}_N \lvert x \rangle =
# \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi i xk/N}\lvert k \rangle.
# $$
#
# The important difference from a classical DFT is that the vector being transformed is the quantum state vector. Measuring immediately after QFT only reveals samples from the output probabilities; the phase information is still there, but it is accessed through later interference, as in phase estimation.
#
# The standard QFT circuit uses Hadamard gates, controlled phase rotations, and final swaps. For an $n$-qubit register, the exact decomposition uses $O(n^2)$ two-qubit rotation structure rather than a dense $2^n \times 2^n$ matrix. Use the binary-fraction notation
#
# $$
# [0.x_jx_{j+1}\ldots x_n] =
# \sum_{m=j}^{n} x_m 2^{-(m-j+1)}.
# $$
#
# ### Step 1: Select one target qubit
#
# Work across the register one target at a time. On the last qubit, a Hadamard gate creates the first Fourier-basis factor:
#
# $$
# \lvert x_n\rangle
# \xrightarrow{H}
# \frac{1}{\sqrt{2}}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_n]}\lvert 1\rangle\right).
# $$
#
# ### Step 2: Add controlled phase rotations
#
# Controlled phase rotations from earlier qubits add the remaining bits of the binary fraction. For a target qubit $x_j$, the Hadamard plus controlled rotations produce
#
# $$
# \lvert x_j\rangle
# \longmapsto
# \frac{1}{\sqrt{2}}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_jx_{j+1}\ldots x_n]}\lvert 1\rangle\right).
# $$
#
# The rotation angle gets smaller as the distance between the control and target positions grows:
#
# $$
# \theta = \frac{\pi}{2^d},
# $$
#
# where $d$ is the distance between the control and target positions.
#
# ### Step 3: Repeat across the register
#
# Repeating the Hadamard-plus-controlled-phase pattern gives the tensor-product form of QFT:
#
# $$
# \mathrm{QFT}\lvert x_1x_2\ldots x_n\rangle =
# \frac{1}{\sqrt{2^n}}
# \bigotimes_{j=n}^{1}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_jx_{j+1}\ldots x_n]}\lvert 1\rangle\right).
# $$
#
# For the four-qubit case, this is
#
# $$
# \mathrm{QFT}\lvert x_1x_2x_3x_4\rangle =
# \frac{1}{\sqrt{16}}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_4]}\lvert 1\rangle\right)
# \otimes
# \left(\lvert 0\rangle + e^{2\pi i[0.x_3x_4]}\lvert 1\rangle\right)
# \otimes
# \left(\lvert 0\rangle + e^{2\pi i[0.x_2x_3x_4]}\lvert 1\rangle\right)
# \otimes
# \left(\lvert 0\rangle + e^{2\pi i[0.x_1x_2x_3x_4]}\lvert 1\rangle\right).
# $$
#
# ### Step 4: Reverse the output order
#
# The textbook QFT decomposition naturally produces the qubits in reversed order. A final layer of swaps restores the register order. Some algorithms omit these swaps and track the reversed order classically.
#
# ```{figure} assets/qft_circuit.png
# :alt: Standard QFT circuit
# :width: 720px
#
# QFT quantum circuit for $n=4$.
# ```

# %% [markdown]
# ## Implementation: `qft` function
#
# The `qamomile.circuit.stdlib.qft.qft` function contains the full process described above: Hadamard gates, controlled phase rotations, and final swaps.
#
# ### Problem Example
#
# We use $N=16$ samples, so the quantum register has four qubits. For this example, $n=4$, $N=16$, $\omega = e^{2\pi i/16}$, and
#
# $$
# F_{16}\lvert x \rangle =
# \frac{1}{\sqrt{16}}\sum_{k=0}^{15}\omega^{xk}\lvert k\rangle.
# $$
#
# We prepare a quantum state whose amplitude at basis index $j$ is $s_j$:
#
# $$
# \lvert \psi_f\rangle =
# \sum_{j=0}^{N-1} s_j \lvert j\rangle,
# \qquad
# s_j = \frac{1}{\sqrt{N}} e^{-2\pi i f j/N},
# \qquad f=5,\quad j=0,1,\ldots,N-1.
# $$
#
# In other words, the function values $f(j)=e^{-2\pi i f j/N}$ are embedded into the amplitudes of $\lvert \psi_f\rangle$ with the normalization factor $1/\sqrt{N}$. Applying QFT to this state moves the frequency component of $f(j)$ into the output amplitudes. With the DFT convention from the background section, all spectral weight should appear at frequency index $k=5$.

# %%
num_qubits = 4
dimension = 2**num_qubits
frequency = 5
positions = np.arange(dimension)

signal = np.exp(-2j * np.pi * frequency * positions / dimension) / np.sqrt(dimension)
spectrum = np.fft.ifft(signal, norm="ortho")
expected_spectrum = np.zeros(dimension, dtype=complex)
expected_spectrum[frequency] = 1.0

print(np.round(np.abs(spectrum), 3))
assert np.allclose(spectrum, expected_spectrum)

# %% [markdown]
# ### Quantum Kernel with `qft`
#
# Qamomile provides QFT as a standard-library composite gate. The convenience function `qamomile.circuit.stdlib.qft.qft` accepts a `Vector[Qubit]`, applies the QFT to the whole register, and returns the transformed vector.
#
# The state preparation below creates the phase ramp directly on the amplitudes. We treat `q[0]` as the least significant bit of the sample index $j$, so the phase added to each qubit doubles from left to right.

# %%
@qmc.qkernel
def qft_frequency_estimator() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(num_qubits, name="q")

    # Start from a uniform superposition over all sample indices.
    q = qmc.h(q)

    # Encode the phase ramp exp(-2 pi i f j / N) into the amplitudes.
    q[0] = qmc.p(q[0], -2 * math.pi * frequency / dimension)
    q[1] = qmc.p(q[1], -2 * math.pi * frequency * 2 / dimension)
    q[2] = qmc.p(q[2], -2 * math.pi * frequency * 4 / dimension)
    q[3] = qmc.p(q[3], -2 * math.pi * frequency * 8 / dimension)

    # Apply QFT and measure the frequency index.
    q = qft(q)
    return qmc.measure(q)


# %%
qft_frequency_estimator.draw()

# %% [markdown]
# The compact diagram keeps QFT as one composite operation. If you want to see how a backend receives the circuit, transpile it. Qiskit can emit a native `QFTGate`, followed by measurements here.

# %%
qiskit_circuit = transpiler.to_circuit(qft_frequency_estimator)
print(qiskit_circuit.draw())

# %% [markdown]
# ### Execution Result
#
# Now execute the qkernel locally and plot a histogram of the measured frequency indices. The helper converts a measured bit tuple to an integer with `q[0]` as the least significant bit, matching the state preparation above.

# %%
backend = AerSimulator(seed_simulator=1234, max_parallel_threads=1)
shots = 512
executable = transpiler.transpile(qft_frequency_estimator)
result = executable.sample(transpiler.executor(backend), shots=shots).result()

probabilities = np.zeros(dimension)
for outcome, count in result.results:
    frequency_index = sum(bit << idx for idx, bit in enumerate(outcome))
    probabilities[frequency_index] = count / shots

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(range(dimension), probabilities)
ax.set_xlabel("frequency index")
ax.set_ylabel("probability")
ax.set_xticks(range(dimension))
ax.set_ylim(0, 1.05)
plt.show()

estimated_frequency = int(np.argmax(probabilities))
print(f"estimated frequency: {estimated_frequency}")

assert result.shots == shots
assert sum(count for _, count in result.results) == shots
assert estimated_frequency == frequency
assert probabilities[frequency] > 0.95
assert all(
    isinstance(outcome, tuple) and len(outcome) == num_qubits
    for outcome, _ in result.results
)

# %% [markdown]
# ## Resource Estimation
#
# The standard exact QFT decomposition uses:
#
# - $n$ Hadamard gates
# - $\frac{n(n - 1)}{2}$ controlled phase rotations
# - $\left\lfloor n / 2 \right\rfloor$ swaps
#
# Therefore the total gate count is $n + \frac{n(n - 1)}{2} + \left\lfloor n / 2 \right\rfloor$, which scales as $O(n^2)$.
#
# Now let's check this with Qamomile's resource estimation feature.

# %%
@qmc.qkernel
def qft_scaling(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(n, name="q")

    for j in qmc.range(n - 1, -1, -1):
        q[j] = qmc.h(q[j])

        for k in qmc.range(j - 1, -1, -1):
            angle = math.pi / (2 ** (j - k))
            q[j], q[k] = qmc.cp(q[j], q[k], angle)

    for j in qmc.range(n // 2):
        q[j], q[n - j - 1] = qmc.swap(q[j], q[n - j - 1])

    return q


# %%
symbolic_estimate = qft_scaling.estimate_resources().simplify()
print("qubits:", symbolic_estimate.qubits)
print("total gates:", symbolic_estimate.gates.total)
print("single-qubit gates:", symbolic_estimate.gates.single_qubit)
print("two-qubit gates:", symbolic_estimate.gates.two_qubit)
print("rotation gates:", symbolic_estimate.gates.rotation_gates)
print("Clifford gates:", symbolic_estimate.gates.clifford_gates)

assert "n" in str(symbolic_estimate.gates.total)
assert "j" not in str(symbolic_estimate.gates.total)
assert "k" not in str(symbolic_estimate.gates.total)

estimate_8 = symbolic_estimate.substitute(n=8)
assert estimate_8.qubits == 8
assert estimate_8.gates.total == 40
assert estimate_8.gates.single_qubit == 8
assert estimate_8.gates.two_qubit == 32
assert estimate_8.gates.rotation_gates == 28
assert estimate_8.gates.clifford_gates == 12

# %% [markdown]
# A dense classical DFT on a length-$N$ vector uses $O(N^2)$ arithmetic operations, and the fast Fourier transform reduces this to $O(N\log N)$. If the Fourier transform dimension is $N = 2^n$, the exact QFT uses $O(n^2)=O((\log N)^2)$ gates. As a unitary transformation, QFT is therefore exponentially smaller in $N$ than the dense classical DFT, and still polylogarithmic compared with the FFT. This does not mean that QFT reads out all $N$ Fourier coefficients efficiently: the transform happens coherently in the quantum state, and a measurement still returns only samples. The scaling benefit matters when later quantum operations use the phase information without materializing the full vector classically.

# %% [markdown]
# ## Summary
#
# In this notebook, we connected the classical DFT viewpoint to QFT, implemented a four-qubit frequency-estimation example, sampled the output, and checked the symbolic resource scaling.
#
# - The DFT rewrites a finite vector into frequency components; QFT applies the same transform to quantum amplitudes.
# - The four-qubit example prepares a single-frequency phase ramp, applies QFT, and estimates the dominant frequency from the sampled histogram.
# - In Qamomile, `qamomile.circuit.stdlib.qft.qft` applies QFT directly to a `Vector[Qubit]`.
# - `draw()`, backend execution, and `estimate_resources()` give you the circuit view, sampled behavior, and symbolic scaling of the same QFT structure.
