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
# The Quantum Fourier Transform (QFT) is the quantum analogue of the discrete Fourier transform. It appears as a core subroutine in quantum phase estimation, order finding, and many algorithms that need to move information between a computational-basis description and a phase description.
#
# This page introduces the classical Fourier transform viewpoint, explains the QFT circuit at a high level, and then implements the three-qubit $F_8$ example from the [Wikipedia QFT example](https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Example), which follows the standard presentation in Nielsen and Chuang's *Quantum Computation and Quantum Information* {cite:p}`10.1017/CBO9780511976667`. By the end, you will have a Qamomile qkernel that draws the QFT circuit, executes it on a local Qiskit simulator, and reports the resource estimate for the standard decomposition.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
from itertools import product

import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.qft import qft
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## Fourier Transform
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

# %%
num_qubits = 3
dimension = 2**num_qubits
basis_index = 5
omega = np.exp(2j * np.pi / dimension)

signal = np.zeros(dimension, dtype=complex)
signal[basis_index] = 1.0
dft = np.fft.ifft(signal, norm="ortho")
expected_dft = np.array(
    [omega ** (basis_index * k) / np.sqrt(dimension) for k in range(dimension)]
)

print(np.round(dft, 3))
assert np.allclose(dft, expected_dft)
assert np.allclose(np.abs(dft), 1 / np.sqrt(dimension))

# %% [markdown]
# This is the classical Fourier transform of the basis vector whose only nonzero entry is at index $5$. Its output has equal magnitude in all eight components, with phases set by powers of the eighth root of unity $\omega = e^{2\pi i/8}$.

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
# For the three-qubit example used here, $n=3$, $N=8$, $\omega = e^{2\pi i/8}$, and
#
# $$
# F_8\lvert x \rangle =
# \frac{1}{\sqrt{8}}\sum_{k=0}^{7}\omega^{xk}\lvert k\rangle.
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
# For the three-qubit case, this is
#
# $$
# \mathrm{QFT}\lvert x_1x_2x_3\rangle =
# \frac{1}{\sqrt{8}}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_3]}\lvert 1\rangle\right)
# \otimes
# \left(\lvert 0\rangle + e^{2\pi i[0.x_2x_3]}\lvert 1\rangle\right)
# \otimes
# \left(\lvert 0\rangle + e^{2\pi i[0.x_1x_2x_3]}\lvert 1\rangle\right).
# $$
#
# ### Step 4: Reverse the output order
#
# The textbook QFT decomposition naturally produces the qubits in reversed order. A final layer of swaps restores the register order. Some algorithms omit these swaps and track the reversed order classically, but Qamomile's standard `qft` function includes them.

# %% [markdown]
# ## Implementation: `qft` function
#
# Qamomile provides QFT as a standard-library composite gate. The convenience function `qamomile.circuit.stdlib.qft.qft` accepts a `Vector[Qubit]`, applies the QFT to the whole register, and returns the transformed vector.
#
# We prepare the three-qubit basis state $\lvert 101\rangle$, which is index $5$, and apply QFT. This is the $F_8\lvert 5\rangle$ instance of the example above. Direct measurement samples uniformly over the 8 basis states because each output amplitude has magnitude $1/\sqrt{8}$.

# %%
@qmc.qkernel
def qft_on_five() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")
    q[0] = qmc.x(q[0])
    q[2] = qmc.x(q[2])
    q = qft(q)
    return qmc.measure(q)


# %%
qft_on_five.draw()

# %% [markdown]
# The compact diagram keeps QFT as one composite operation. If you want to see how a backend receives the circuit, transpile it. Qiskit can emit a native `QFTGate`, followed by measurements here.

# %%
qiskit_circuit = transpiler.to_circuit(qft_on_five)
print(qiskit_circuit.draw())

# %% [markdown]
# Now execute the qkernel locally. The result is stochastic because we sample a quantum state, but the expected probability is $1/8$ for each 3-bit outcome.

# %%
try:
    from qiskit_aer import AerSimulator

    backend = AerSimulator(seed_simulator=1234, max_parallel_threads=1)
except ImportError:
    from qiskit.providers.basic_provider import BasicSimulator

    backend = BasicSimulator()
    backend.set_options(seed_simulator=1234)

shots = 512
result = (
    transpiler.transpile(qft_on_five)
    .sample(transpiler.executor(backend), shots=shots)
    .result()
)
counts = dict(result.results)
expected_outcomes = set(product([0, 1], repeat=3))

for outcome in sorted(expected_outcomes):
    probability = counts.get(outcome, 0) / shots
    print(f"{outcome}: {probability:.3f}")

assert result.shots == shots
assert sum(counts.values()) == shots
assert set(counts).issubset(expected_outcomes)
assert all(isinstance(outcome, tuple) and len(outcome) == 3 for outcome in counts)

# %% [markdown]
# Finally, ask Qamomile for a resource estimate. For this kernel, the standard 3-qubit QFT contributes 3 Hadamard gates, 3 controlled phase rotations, and 1 swap. The two extra X gates prepare $\lvert 101\rangle$.

# %%
estimate = qft_on_five.estimate_resources().simplify()
print("qubits:", estimate.qubits)
print("total gates:", estimate.gates.total)
print("single-qubit gates:", estimate.gates.single_qubit)
print("two-qubit gates:", estimate.gates.two_qubit)
print("rotation gates:", estimate.gates.rotation_gates)
print("Clifford gates:", estimate.gates.clifford_gates)

assert estimate.qubits == 3
assert estimate.gates.total == 9
assert estimate.gates.single_qubit == 5
assert estimate.gates.two_qubit == 4
assert estimate.gates.rotation_gates == 3
assert estimate.gates.clifford_gates == 6

# %% [markdown]
# ## Summary
#
# - The DFT rewrites a finite vector into frequency components; QFT applies the same transform to quantum amplitudes.
# - The three-qubit $F_8$ example maps $\lvert 5\rangle = \lvert 101\rangle$ to an equal-magnitude phase pattern over all 8 computational-basis states.
# - In Qamomile, `qamomile.circuit.stdlib.qft.qft` applies QFT directly to a `Vector[Qubit]`.
# - `draw()`, backend execution, and `estimate_resources()` give you the circuit view, sampled behavior, and gate-cost view of the same qkernel.
