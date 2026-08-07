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
# # Quantum Fourier Transform (QFT)
#
# The Quantum Fourier Transform (QFT) is the quantum version of the discrete Fourier transform. It is an important subroutine in quantum phase estimation, Shor's algorithm {cite:p}`10.1109/SFCS.1994.365700`, and related algorithms that use phases encoded in quantum amplitudes {cite:p}`10.48550/arXiv.quant-ph/0201067`.
#
# This notebook starts with the classical Fourier transform, explains the QFT circuit, and implements a four-qubit frequency-estimation example with Qamomile. It compares a from-scratch implementation with the built-in `qft` function, estimates the dominant frequency from sampled results, and examines the required gate count.

# %%
# Install the latest Qamomile and the optional packages used in this notebook.
# # !pip install "qamomile[qiskit,visualization]"

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## Background: Fourier Transform
#
# A Fourier transform represents data in terms of frequency components and reveals how much of each frequency is present. For finite vectors, the version we usually use is the **Discrete Fourier Transform** (DFT).
#
# For a vector $x = (x_0, x_1, \ldots, x_{N-1})$, this notebook uses the following normalized DFT:
#
# $$
# y_k = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} x_j e^{2\pi i jk / N},
# \qquad k = 0, 1, \ldots, N-1.
# $$
#
# The output index $k$ labels a frequency component. The angular frequency corresponding to this index is $2\pi k/N$, and the phase factor $e^{2\pi i jk / N}$ makes each input position contribute with a different phase.

# %% [markdown]
# ## Algorithm
#
# QFT applies the same transform as DFT to the amplitudes of a quantum state. If $N = 2^n$, its action on the computational basis state $\lvert x\rangle$ corresponding to an integer $x$ is
#
# $$
# \mathrm{QFT}_N \lvert x \rangle =
# \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi i xk/N}\lvert k \rangle.
# $$
#
# For a general quantum state $\lvert\psi\rangle = \sum_{j=0}^{N-1} a_j\lvert j\rangle$, QFT returns the state obtained by linearly combining the action on each computational basis state $\lvert j\rangle$.
#
# A classical DFT returns the full output vector. In contrast, QFT transforms the amplitudes of a quantum state and returns the transformed quantum state. Therefore, a measurement immediately after QFT gives only a computational-basis outcome sampled according to the transformed probability distribution. However, when QFT is used as a subroutine, as in phase estimation, the transformed phase information can be used directly.
#
# The standard QFT circuit uses Hadamard gates, controlled phase rotations, and final swaps. For an $n$-qubit array, the exact circuit uses $O(n^2)$ gates. It is useful to write the phases as binary fractions:
#
# $$
# [0.x_jx_{j+1}\ldots x_n] =
# \frac{x_j}{2}
# + \frac{x_{j+1}}{2^2}
# + \cdots
# + \frac{x_n}{2^{n-j+1}}
# =
# \sum_{m=j}^{n} \frac{x_m}{2^{m-j+1}}.
# $$
#
# ### Step 1: Select one target qubit
#
# Process one target qubit at a time. On the last qubit, a Hadamard gate creates the first factor in the QFT output:
#
# If $x_n=0$, then $H\lvert0\rangle = (\lvert0\rangle + \lvert1\rangle)/\sqrt{2}$. If $x_n=1$, then $H\lvert1\rangle = (\lvert0\rangle - \lvert1\rangle)/\sqrt{2}$, which can be written in the same form using $e^{2\pi i[0.1]} = e^{\pi i} = -1$.
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
# Controlled phase rotations from the remaining qubits complete the binary fraction. For a target qubit $x_j$, the Hadamard gate and controlled rotations produce
#
# $$
# \lvert x_j\rangle
# \longmapsto
# \frac{1}{\sqrt{2}}
# \left(\lvert 0\rangle + e^{2\pi i[0.x_jx_{j+1}\ldots x_n]}\lvert 1\rangle\right).
# $$
#
# These phase rotations are commonly written as $R_k$:
#
# $$
# R_k =
# \begin{pmatrix}
# 1 & 0 \\
# 0 & e^{2\pi i / 2^k}
# \end{pmatrix}.
# $$
#
# A controlled-$R_k$ applies this rotation to the target only when the control qubit is $\lvert 1\rangle$. If the control and target positions are distance $d$ apart, the QFT circuit uses controlled-$R_{d+1}$. Its rotation angle is
#
# $$
# \theta = \frac{2\pi}{2^{d+1}} = \frac{\pi}{2^d}.
# $$
#
# ### Step 3: Repeat across all qubits
#
# Repeating this pattern gives the following product form of QFT:
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
# The standard QFT circuit naturally puts the output qubits in reverse order. A final layer of swaps restores the original qubit order. Some algorithms omit these swaps and track the reversed order in classical code.
#
# ```{figure} assets/qft_circuit.png
# :alt: Standard QFT circuit
# :width: 720px
#
# QFT quantum circuit for $n=4$.
# ```

# %% [markdown]
# ## Qamomile Implementation
#
# We first implement the QFT circuit from scratch with Qamomile gates and then replace that implementation with the built-in `qmc.qft` function.
#
# ### Problem setting
#
# We use $N=16$ samples, so four qubits are enough. In this example, we prepare a quantum state whose amplitude for the computational basis state $\lvert j\rangle$ is $s_j$. Here, $n=4$, $N=16$, and $\omega = e^{2\pi i/16}$.
#
# $$
# \lvert \psi_f\rangle =
# \sum_{j=0}^{N-1} s_j \lvert j\rangle,
# \qquad
# s_j = \frac{1}{\sqrt{N}} e^{-2\pi i f j/N}
# = \frac{1}{\sqrt{N}}\omega^{-fj},
# \qquad f=5,\quad j=0,1,\ldots,N-1.
# $$
#
# In other words, the values of $e^{-2\pi i f j/N}$ are placed in the amplitudes of $\lvert \psi_f\rangle$, with the normalization factor $1/\sqrt{N}$. Applying QFT sums the contributions from each computational basis state $\lvert j\rangle$, leaving only the frequency index $f$.
#
# $$
# \mathrm{QFT}_{16}\lvert \psi_f \rangle =
# \frac{1}{16}\sum_{j=0}^{15}\sum_{k=0}^{15}\omega^{j(k-f)}\lvert k\rangle
# = \lvert f\rangle.
# $$
#
# Therefore, when $f=5$, the output should be concentrated at frequency index $k=5$. Let's verify this by performing a classical DFT. We compute it using NumPy's `np.fft.ifft`.

# %%
EXAMPLE_INPUTS = {"num_qubits": 4, "frequency": 5}
dimension = 2 ** EXAMPLE_INPUTS["num_qubits"]
positions = np.arange(dimension)

signal = np.exp(
    -2j * np.pi * EXAMPLE_INPUTS["frequency"] * positions / dimension
) / np.sqrt(dimension)
spectrum = np.fft.ifft(signal, norm="ortho")
expected_spectrum = np.zeros(dimension, dtype=complex)
expected_spectrum[EXAMPLE_INPUTS["frequency"]] = 1.0

print(np.round(np.abs(spectrum), 3))
assert np.allclose(spectrum, expected_spectrum, rtol=1e-10, atol=1e-10)

# %% [markdown]
# ### From-scratch implementation
#
# The following qkernel directly implements the four circuit steps described above. Qamomile writes a controlled phase rotation as `qmc.cp(control, target, angle)`. The outer loop selects target qubits from right to left, and the inner loop applies the required rotations from the qubits to their left. The final loop restores the qubit order with swaps.


# %%
@qmc.qkernel
def qft_from_scratch(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    num_qubits = qubits.shape[0]
    for offset in qmc.range(num_qubits):
        target = num_qubits - 1 - offset
        qubits[target] = qmc.h(qubits[target])
        for delta in qmc.range(target):
            control = target - 1 - delta
            angle = math.pi / (2 ** (target - control))
            qubits[control], qubits[target] = qmc.cp(
                qubits[control], qubits[target], angle
            )
    for index in qmc.range(num_qubits // 2):
        mirror = num_qubits - index - 1
        qubits[index], qubits[mirror] = qmc.swap(qubits[index], qubits[mirror])
    return qubits


# %%
qft_from_scratch.draw(qubits=4)

# %% [markdown]
# The state-preparation qkernel below encodes $e^{-2\pi i f j/N}$ in the amplitudes. It treats `qubits[0]` as the least significant bit of the sample index $j$, so the phase angle doubles for each successive qubit. Both frequency estimators accept the qubit count and frequency as explicit qkernel inputs; the concrete example values are supplied as compile-time bindings.


# %%
@qmc.qkernel
def prepare_frequency_state(
    qubits: qmc.Vector[qmc.Qubit], frequency: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    num_qubits = qubits.shape[0]
    dimension = 2**num_qubits
    qubits = qmc.h(qubits)
    for index in qmc.range(num_qubits):
        angle = -2 * math.pi * frequency * (2**index) / dimension
        qubits[index] = qmc.p(qubits[index], angle)
    return qubits


@qmc.qkernel
def qft_frequency_estimator_from_scratch(
    num_qubits: qmc.UInt, frequency: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    qubits = qmc.qubit_array(num_qubits, name="qubits")
    qubits = prepare_frequency_state(qubits, frequency)
    qubits = qft_from_scratch(qubits)
    return qmc.measure(qubits)


# %% [markdown]
# ### Built-in `qft`
#
# Qamomile provides the same transform as the built-in qkernel `qmc.qft`. It accepts a `Vector[Qubit]`, applies the Hadamard gates, controlled phase rotations, and swaps, and returns the transformed qubit array. The frequency estimator only needs to replace the call to `qft_from_scratch` with `qmc.qft`.


# %%
@qmc.qkernel
def qft_frequency_estimator_with_stdlib(
    num_qubits: qmc.UInt, frequency: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    qubits = qmc.qubit_array(num_qubits, name="qubits")
    qubits = prepare_frequency_state(qubits, frequency)
    qubits = qmc.qft(qubits)
    return qmc.measure(qubits)


# %%
qft_frequency_estimator_with_stdlib.draw(**EXAMPLE_INPUTS)

# %% [markdown]
# `draw()` shows the built-in QFT as one operation. To inspect the circuit emitted for Qiskit, convert the qkernel with the same compile-time bindings.

# %%
qiskit_circuit = transpiler.to_circuit(
    qft_frequency_estimator_with_stdlib,
    bindings=EXAMPLE_INPUTS,
)
print(qiskit_circuit.draw())

# %% [markdown]
# ### Execution result
#
# We execute both qkernels with identical bindings and compare their measured frequency distributions. The conversion below treats `qubits[0]` as the least significant bit, matching the state preparation above.

# %%
backend = AerSimulator(seed_simulator=42, max_parallel_threads=1)
shots = 512
results = {}
for implementation, kernel in {
    "from scratch": qft_frequency_estimator_from_scratch,
    "built-in qft": qft_frequency_estimator_with_stdlib,
}.items():
    executable = transpiler.transpile(kernel, bindings=EXAMPLE_INPUTS)
    results[implementation] = executable.sample(
        transpiler.executor(backend), shots=shots
    ).result()

probabilities_by_implementation = {}
for implementation, result in results.items():
    probabilities = np.zeros(dimension)
    for outcome, count in result.results:
        frequency_index = sum(bit << index for index, bit in enumerate(outcome))
        probabilities[frequency_index] = count / shots
    probabilities_by_implementation[implementation] = probabilities

fig, ax = plt.subplots(figsize=(7, 3))
indices = np.arange(dimension)
bar_width = 0.4
ax.bar(
    indices - bar_width / 2,
    probabilities_by_implementation["from scratch"],
    width=bar_width,
    color="#2696EB",
    label="from scratch",
)
ax.bar(
    indices + bar_width / 2,
    probabilities_by_implementation["built-in qft"],
    width=bar_width,
    color="#FF6B6B",
    label="built-in qft",
)
ax.set_xlabel("frequency index")
ax.set_ylabel("probability")
ax.set_xticks(indices)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.legend()
plt.show()

for implementation, probabilities in probabilities_by_implementation.items():
    estimated_frequency = int(np.argmax(probabilities))
    print(f"{implementation}: estimated frequency = {estimated_frequency}")
    assert estimated_frequency == EXAMPLE_INPUTS["frequency"]
    assert probabilities[EXAMPLE_INPUTS["frequency"]] > 0.95

for result in results.values():
    assert result.shots == shots
    assert sum(count for _, count in result.results) == shots
assert all(
    isinstance(outcome, tuple) and len(outcome) == EXAMPLE_INPUTS["num_qubits"]
    for result in results.values()
    for outcome, _ in result.results
)
assert np.allclose(
    probabilities_by_implementation["from scratch"],
    probabilities_by_implementation["built-in qft"],
    rtol=0.0,
    atol=0.0,
)

# %% [markdown]
# ## Resource Estimation
#
# The exact QFT circuit uses:
#
# - $n$ Hadamard gates
# - $\frac{n(n - 1)}{2}$ controlled phase rotations
# - $\left\lfloor n / 2 \right\rfloor$ swaps
#
# Therefore the total gate count is $n + \frac{n(n - 1)}{2} + \left\lfloor n / 2 \right\rfloor$, so it scales as $O(n^2)$.
#
# The qubit count determines the circuit structure, so the resource-estimation qkernel accepts it as an explicit input. Qamomile can then specialize one symbolic estimate for each concrete size through `inputs` without constructing a separate qkernel factory.


# %%
@qmc.qkernel
def qft_resource_kernel(num_qubits: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    qubits = qmc.qubit_array(num_qubits, name="qubits")
    qubits = qmc.qft(qubits)
    return qubits


# %%
estimate_4 = qft_resource_kernel.estimate_resources(inputs={"num_qubits": 4}).simplify()
print("qubits:", estimate_4.qubits)
print("total gates:", estimate_4.gates.total)
print("single-qubit gates:", estimate_4.gates.single_qubit)
print("two-qubit gates:", estimate_4.gates.two_qubit)
print("rotation gates:", estimate_4.gates.rotation_gates)
print("Clifford gates:", estimate_4.gates.clifford_gates)

assert estimate_4.qubits == 4
assert estimate_4.gates.total == 12
assert estimate_4.gates.single_qubit == 4
assert estimate_4.gates.two_qubit == 8
assert estimate_4.gates.rotation_gates == 6
assert estimate_4.gates.clifford_gates == 6

# %% [markdown]
# A direct classical DFT on a length-$N$ vector uses $O(N^2)$ arithmetic operations, and the fast Fourier transform (FFT) uses $O(N\log N)$. If $N = 2^n$, the exact QFT circuit uses $O(n^2)=O((\log N)^2)$ gates. Compared with the direct classical DFT, this is exponentially smaller in $N$. The caveat is important: measuring the state does not give all $N$ Fourier coefficients. QFT is useful when later quantum steps can use the transformed amplitudes without reading out the whole vector.
#
# The next plot compares the `qmc.qft` gate count returned by `.estimate_resources()` with the exact formula above. It also includes a scaled $O(N\log N)$ reference for the classical FFT, normalized to the QFT count at $n=3$, to show how the two growth rates differ.

# %%
qft_qubit_counts = np.arange(3, 10)
qft_total_gates = []

for num_qubits in qft_qubit_counts:
    estimate_n = qft_resource_kernel.estimate_resources(
        inputs={"num_qubits": int(num_qubits)}
    ).simplify()
    qft_total_gates.append(int(estimate_n.gates.total))

theoretical_qft_gate_counts = [
    num_qubits + num_qubits * (num_qubits - 1) // 2 + num_qubits // 2
    for num_qubits in qft_qubit_counts
]

dimension_counts = 2**qft_qubit_counts
nlogn_reference = dimension_counts * qft_qubit_counts
nlogn_reference = nlogn_reference / nlogn_reference[0] * qft_total_gates[0]

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(
    qft_qubit_counts,
    qft_total_gates,
    marker="o",
    color="#2696EB",
    label="Qamomile qft",
)
ax.plot(
    qft_qubit_counts,
    theoretical_qft_gate_counts,
    linestyle="--",
    color="#FF6B6B",
    label="exact QFT formula",
)
ax.plot(
    qft_qubit_counts,
    nlogn_reference,
    linestyle="--",
    color="#4ECDC4",
    label=r"FFT $O(N\log N)$ (scaled)",
)
ax.set_xlabel(r"number of qubits $n$")
ax.set_ylabel("total gates")
ax.set_yscale("log")
ax.set_xticks(qft_qubit_counts)
ax.grid(alpha=0.3)
ax.legend()
plt.show()

assert qft_total_gates == theoretical_qft_gate_counts
assert len(theoretical_qft_gate_counts) == len(qft_total_gates)
assert len(nlogn_reference) == len(qft_total_gates)

# %% [markdown]
# ## Summary
#
# In this notebook, we learned:
#
# - QFT applies the DFT to quantum amplitudes and returns a transformed quantum state; measuring that state yields samples rather than the full vector of Fourier coefficients.
# - A QFT circuit can be constructed from Hadamard gates, controlled phase rotations, and swaps, while Qamomile's built-in `qmc.qft` expresses the same transformation in one call.
# - The exact $n$-qubit circuit uses $n + n(n-1)/2 + \lfloor n/2\rfloor=O(n^2)$ gates, and Qamomile can evaluate this scaling directly with `.estimate_resources()`.
