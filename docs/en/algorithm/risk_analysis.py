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
#     display_name: qamomile (3.11.16.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, finance, simulation]
# ---
#
# # Risk Analysis with Quantum Computers
#
# Value at Risk (VaR) and Conditional Value at Risk (CVaR) are important measures for quantifying financial risk.
# Classical Monte Carlo simulation has traditionally been used to compute these measures, but its computational cost can become prohibitively large.
# Quantum computing methods have been proposed as a way to reduce this computational burden and enable faster calculations.
# In this article, we present an example implementation in Qamomile based on [Woerner & Egger (2019)](https://www.nature.com/articles/s41534-019-0130-6), which proposed a quantum algorithm for efficiently computing VaR and CVaR.
#

# %%
# Install the latest Qamomile through pip! 
# # !pip install qamomile

# %%
import numpy as np
import matplotlib.pyplot as plt
import qamomile.circuit as qmc
from qamomile.circuit.stdlib import iqft, ripple_carry_add
from qamomile.qiskit import QiskitTranspiler
from scipy.stats import norm


# %% [markdown]
# ## Background
#
# ### Problem: Computing VaR and CVaR
#
# Risk management plays a central role in financial systems.
# In particular, VaR represents the maximum loss amount that is expected to remain within a given probability over a specified period, while CVaR represents the expected asset value below VaR. Both are important risk measures.  
# Classical Monte Carlo simulation is a standard approach for computing VaR and CVaR.
# However, for a sample size $M$, its error is known to scale as $\mathcal{O}(M^{-1/2})$, and this slow error decay becomes a computational bottleneck.
# Variance reduction methods and Quasi-Monte Carlo (QMC) methods are examples of approaches for improving this performance.
# Variance reduction decreases constant factors without changing the asymptotic scaling, but it requires problem-specific structural design and therefore lacks generality.
# QMC improves the asymptotic behavior, but its effectiveness is limited to low-dimensional problems, making it less suitable for realistic applications.  
# Quantum Amplitude Estimation (QAE) is a quantum algorithm for estimating an unknown parameter, and its error is known to converge as $\mathcal{O}(M^{-1})$.
# Compared with classical methods such as Monte Carlo, this provides a quadratic quantum speedup.
# [Woerner & Egger (2019)](https://www.nature.com/articles/s41534-019-0130-6) proposed extending QAE to the computation of variance, VaR, and CVaR for probability distributions.
# They also discussed shallow-circuit settings, with a focus on possible NISQ implementations.
#
# ### Related Work
#
# An important study showing that QAE can accelerate Monte Carlo computation is [Montanaro (2015)](https://royalsocietypublishing.org/rspa/article/471/2181/20150301/57575/Quantum-speedup-of-Monte-Carlo-methodsQuantum).
# This work theoretically demonstrated, in a general setting, that QAE can speed up Monte Carlo methods.
# It also showed that, when combined with quantum walks, QAE can accelerate classical algorithms for computing partition functions using multistage Markov-chain Monte Carlo methods.
# Another relevant study is [Rebentrost et al. (2018)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.022321), which proposed a quantum algorithm for Monte Carlo pricing of financial derivatives.
# The paper presented theoretical analysis and numerical simulations showing a quadratic quantum speedup in the number of steps required to obtain a pricing estimate by using QAE.
# However, [Rebentrost et al. (2018)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.022321) focused on expectation-value calculations and on problem settings where QAE can be applied relatively directly, specifically the calculation of fair values of derivatives.
# Computing VaR and CVaR is fundamentally nonlinear, so QAE cannot be applied to these quantities as directly.
# [Woerner & Egger (2019)](https://www.nature.com/articles/s41534-019-0130-6) therefore extended QAE to the more practically relevant task of computing VaR and CVaR.
#
# ## Algorithm
#
# ### Encoding a Probability Distribution into a Quantum State
#
# First, let us represent the distribution of a random variable $X$ as a quantum state.
# Using $n$ qubits, we discretize the range of the random variable as $\{0, 1, \dots, N-1\} \ (N=2^n)$.
# We prepare an operator $\mathcal{R}$ that represents the random variable $X$ as the following quantum state:
#
# $$
# \mathcal{R} \vert 0 \rangle_n 
# = \vert \psi \rangle_n
# = \sum_{i=0}^{N-1} \sqrt{p_i} \vert i \rangle_n \tag{1}
# $$
#
# Here, $p_i$ is the probability of measuring $\vert i \rangle_n$, with $\sum_i p_i = 1$.
#
# ### Objective Operator $F$
#
# Next, consider a function $f(i) \in [0, 1]$ and define an operator $F$ that encodes this function into the amplitude of an ancilla qubit:
#
# $$
# F \vert i \rangle_n \vert 0 \rangle 
# = \vert i \rangle_n \left( \sqrt{1-f(i)} \vert 0 \rangle + \sqrt{f(i)} \vert 1 \rangle \right) \tag{2}
# $$
#
# Applying this operator $F$ to Eq. (1) gives
#
# $$
# F \vert \psi \rangle_n \vert 0 \rangle 
# = \sum_{i=0}^{N-1} \sqrt{1-f(i)} \sqrt{p_i} \vert i \rangle_n \vert 0 \rangle + \sum_{i=0}^{N-1} \sqrt{f(i)} \sqrt{p_i} \vert i \rangle_n \vert 1 \rangle \tag{3}
# $$
#
# The probability of measuring the ancilla qubit in $\vert 1 \rangle$ is then
#
# $$
# P_1 
# = \sum_{i=0}^{N-1} f(i) p_i \tag{4}
# $$
#
# Equation (4) corresponds to the expectation value $\mathbb{E}[f(X)]$ of $f(i)$, and the statistic that can be computed depends on the choice of $f(X)$.
# The following table shows the correspondence between the target statistic and the function $f(i)$.
#
# | Statistic | $f(i)$ |
# |-|-|
# | $\mathbb{E}[X]$ | $i/(N-1)$ |
# | $\mathbb{E}[X^2]$ | $i^2/(N-1)^2$ |
#
# [Woerner & Egger (2019)](https://www.nature.com/articles/s41534-019-0130-6) uses QAE to estimate the coefficient $P_1$ associated with the $\vert 1 \rangle$ component.  
# Directly implementing $f(i)$ as $F$ in a quantum circuit would require many ancillary qubits.
# Therefore, [Woerner & Egger (2019)](https://www.nature.com/articles/s41534-019-0130-6) introduced the following approximation.
# Consider an operator that uses a degree-$k$ polynomial $\zeta(x)$ and performs
#
# $$
# \vert x \rangle_n \vert 0 \rangle \longrightarrow
# \vert x \rangle_n (\cos \zeta(x) \vert 0 \rangle + \sin \zeta(x) \vert 1 \rangle) \tag{5}
# $$
#
# This is a multi-controlled Y rotation and can be implemented using $\mathcal{O}(n^{k+1})$ gates and $\mathcal{O}(n)$ ancilla qubits.
# After this operation, the probability of measuring the auxiliary qubit in $\vert 1 \rangle$ is $\sin^2 \zeta(x)$.  
# Using the approximation $\sin^2(y + \pi/4) \approx y + 1/2$ for small $\vert y \vert$, and setting $y = f(i) \in [0,1]$, we obtain
#
# $$
# c\!\left(y - \frac{1}{2}\right) + \frac{1}{2}
# \approx \sin^2\!\left(c\,\zeta(y) + \frac{\pi}{4}\right)
# \implies
# \zeta(y) \approx \frac{1}{c}\!\left\{\arcsin\sqrt{c\!\left(y-\frac{1}{2}\right)
# +\frac{1}{2}} - \frac{\pi}{4}\right\} \tag{6}
# $$
#
# We then use the Taylor expansion of this expression around $y = 1/2$ to order $2u+1$.
# Because $\zeta(y)$ is odd with respect to $y=1/2$, all even-order terms vanish, allowing an efficient implementation.
# Under this approximation, the total error when QAE is performed is
#
# $$
# \varepsilon = \mathcal{O}\!\left(M^{-\frac{2u+2}{2u+3}}\right) \tag{7}
# $$
#
# Even for $u=0$ (the minimum circuit depth), the error scales as $\mathcal{O}(M^{-2/3})$, which converges faster than the classical Monte Carlo rate of $\mathcal{O}(M^{-1/2})$.
# As $u$ increases, the scaling approaches the optimal $\mathcal{O}(M^{-1})$ rate.
#
# ### Computing VaR and CVaR
#
# Let us now extend the above construction to VaR and CVaR.
# For a given confidence level $\alpha \in [0, 1]$, $\mathrm{VaR}_\alpha (X)$ is the smallest $x \in \{0, 1, \dots, N-1\}$ satisfying $P[X \leq x] \geq 1-\alpha$, i.e., the $1-\alpha$ quantile of the loss distribution.
# To compute it, consider $f_\ell (i) = \mathbf{1}(i \leq \ell)$.
# Here, $\mathbf{1}(i \leq \ell)$ is an indicator function that returns 1 if $i \leq \ell$ and 0 otherwise.
# Applying the corresponding operator $F_\ell$ to $\vert \psi \rangle_n \vert 0 \rangle$ gives
#
# $$
# F_\ell \vert \psi \rangle_n \vert 0 \rangle 
# = \sum_{i=0}^{N-1} \sqrt{1-f_\ell(i)} \sqrt{p_i} \vert i \rangle_n \vert 0 \rangle + \sum_{i=0}^{N-1} \sqrt{f_\ell(i)} \sqrt{p_i} \vert i \rangle_n \vert 1 \rangle 
# = \sum_{i=\ell+1}^{N-1} \sqrt{p_i} \vert i \rangle_n \vert 0 \rangle + \sum_{i=0}^\ell \sqrt{p_i} \vert i \rangle_n \vert 1 \rangle \tag{8}
# $$
#
# Thus, the probability of measuring the ancilla qubit in $\vert 1 \rangle$ is $\sum_{i=0}^\ell p_i = P[X \leq \ell]$.
# As described above, QAE is used to estimate $a_\ell = \mathbf{E}[f_\ell (X)] = P[X \leq \ell]$.  
# The value of $\ell$ can be found by binary search.
# Using binary search, the smallest index $\ell_\alpha$ satisfying $P[X \leq \ell_\alpha] \geq 1-\alpha$ can be found in at most $\mathcal{O}(\log 2^n) = \mathcal{O}(n)$ steps.  
# Next, let us consider the computation of CVaR.
# Assume that $\ell_\alpha = \mathrm{VaR}_\alpha (X)$ has already been obtained from the VaR calculation.
# Then CVaR can be written as
#
# $$
# \mathrm{CVaR}_\alpha (X) 
# = \mathbb{E} [X \vert X \leq \ell_\alpha] 
# = \frac{\mathbb{E}[X \cdot \mathbf{1} [X \leq \ell_\alpha]]}{P[X \leq \ell_\alpha]} \tag{9}
# $$
#
# The denominator in this expression is available from the VaR calculation.
# The numerator can be written as
#
# $$
# \mathbb{E} [X \cdot \mathbf{1} [X \leq \ell_\alpha]] 
# = \sum_{i=0}^{N-1} i \mathbf{1} [i \leq \ell_\alpha] p_i 
# = \sum_{i=0}^{\ell_\alpha} i p_i \tag{10}
# $$
#
# As noted above, the expectation value $\mathbb{E}[f(X)]$ can be estimated using QAE.
# However, QAE estimates quantities in the interval $[0,1]$, i.e., probabilities.
# Because Eq. (10) can generally exceed $[0,1]$, we normalize $0 \leq i \leq \ell_\alpha$ by dividing by $\ell_\alpha$:
#
# $$
# \mathbb{E} \left[ \frac{X}{\ell_\alpha} \cdot \mathbf{1} [X \leq \ell_\alpha] \right] 
# = \sum_{i=0}^{\ell_\alpha} \frac{i}{\ell_\alpha} p_i \in [0, 1] \tag{11}
# $$
#
# Comparing Eq. (4) and Eq. (11), the corresponding function is
#
# $$
# f(i) 
# = \frac{i}{\ell_\alpha} \mathbf{1} [i \leq \ell_\alpha] \tag{12}
# $$
#
# QAE is used to estimate $\mathbb{E}[f(X)]$, after which multiplying by $\ell_\alpha / P[X \leq \ell_\alpha]$ yields the CVaR.
#

# %% [markdown]
# ## Implementation with Qamomile
#
# Now let us implement the method proposed in [Woerner & Egger (2019)](https://www.nature.com/articles/s41534-019-0130-6) using Qamomile.
#
# ### Classical Preprocessing
#
# We discretize a normal distribution into $N = 2^n$ points and prepare the amplitudes $\sqrt{p_i}$ required for amplitude encoding.
# For the minimum-depth case in [Woerner & Egger (2019)](https://www.nature.com/articles/s41534-019-0130-6), $u=0$ gives $\zeta(y) \simeq y - \frac{1}{2}$.
# Keeping only the leading terms, the estimation error is approximately
#
# $$
# \epsilon (c) 
# \simeq \frac{\pi}{Mc} + \frac{c^2}{6} \tag{13}
# $$
#
# We use the value $c \simeq \left( \frac{3\pi}{M}\right)^{1/3}$ that minimizes this expression, while restricting $c \leq 1$.
#

# %%
# ===================================================
# Step 0: Classical preprocessing
# ===================================================

def make_normal_amplitudes(n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    N = 2 ** n
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, N)
    probs = norm.pdf(x, mu, sigma)
    probs /= probs.sum()
    return np.sqrt(probs)


def choose_u0_scaling_c(m: int) -> float:
    """Choose the scaling parameter c used in the Woerner--Egger u=0 approximation."""
    M = 2 ** m
    return float(min(1.0, (3.0 * np.pi / M) ** (1.0 / 3.0)))


# %% [markdown]
# ### Kernel Composition
#
# To repeatedly apply the QAE Grover operator $Q$, we define helper functions that compose Qamomile kernels in sequence.
# In this implementation, $A$ acts not only on the distribution register `q`, but also on the comparator's constant register `const`, the ripple-carry workspace qubits `carry` and `overflow`, the `flag` qubit that stores the CVaR condition, and the `anc` qubit used by QAE to identify the "good state." Because all workspace qubits are returned to $\lvert0\rangle$ at the end of the objective, $A^\dagger$ can also be constructed using Qamomile's `qmc.inverse`.
#

# %%
# ===================================================
# Step 1: Kernel composition utilities
# ===================================================

def merge_system_kernels(left, right):
    @qmc.qkernel
    def merged(
        q: qmc.Vector[qmc.Qubit],
        const: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        anc: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        q, const, carry, overflow, flag, anc = left(
            q, const, carry, overflow, flag, anc
        )
        q, const, carry, overflow, flag, anc = right(
            q, const, carry, overflow, flag, anc
        )
        return q, const, carry, overflow, flag, anc

    return merged


@qmc.qkernel
def _identity_const(
    const: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    return const


def make_const_x_kernel(bit: int):
    @qmc.qkernel
    def x_const(
        const: qmc.Vector[qmc.Qubit],
    ) -> qmc.Vector[qmc.Qubit]:
        const[bit] = qmc.x(const[bit])
        return const

    return x_const


def merge_const_kernels(left, right):
    @qmc.qkernel
    def merged_const(
        const: qmc.Vector[qmc.Qubit],
    ) -> qmc.Vector[qmc.Qubit]:
        const = left(const)
        const = right(const)
        return const

    return merged_const


def make_prepare_constant_kernel(n: int, value: int):
    """Construct |0...0> -> |value> using only X gates (little-endian)."""
    one_bits = [bit for bit in range(n) if (value >> bit) & 1]
    if not one_bits:
        return _identity_const

    kernel = make_const_x_kernel(one_bits[0])
    for bit in one_bits[1:]:
        kernel = merge_const_kernels(kernel, make_const_x_kernel(bit))
    return kernel



# %% [markdown]
# ### Implementing the Objective Operator $F$
#
# Using Qamomile's `ripple_carry_add`, we construct a comparator that determines whether $i \leq \ell$.
# For VaR, the comparison result is written directly to the ancilla qubit so that QAE can estimate $P[X \leq \ell]$.
# For CVaR, the comparison result is stored in `flag`, and controlled $R_y$ rotations based on the $u = 0$ Taylor approximation from [Woerner & Egger (2019)](https://www.nature.com/articles/s41534-019-0130-6) are applied only to the tail region.
#

# %%
# ===================================================
# Step 2: Ripple-carry comparator and Woerner--Egger-style efficient F
# ===================================================

def make_comparator_arithmetic(n: int, l: int):
    """Build the constant preparation, addition, and inverse addition used by the comparator."""
    offset = 2 ** n - 1 - l
    prepare_const = make_prepare_constant_kernel(n, offset)

    @qmc.qkernel
    def add_offset(
        const: qmc.Vector[qmc.Qubit],
        q: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
    ]:
        const, q, carry, overflow = ripple_carry_add(
            const, q, carry, overflow
        )
        return const, q, carry, overflow

    add_offset_dag = qmc.inverse(add_offset)
    return prepare_const, add_offset, add_offset_dag


def make_var_objective_kernel(n: int, l: int):
    """Apply anc ^= 1[q <= l] and return all workspace qubits to |0>."""
    prepare_const, add_offset, add_offset_dag = make_comparator_arithmetic(n, l)

    @qmc.qkernel
    def objective(
        q: qmc.Vector[qmc.Qubit],
        const: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        anc: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        const = prepare_const(const)
        const, q, carry, overflow = add_offset(
            const, q, carry, overflow
        )

        # overflow = 0 <=> q <= l
        anc = qmc.x(anc)
        overflow, anc = qmc.cx(overflow, anc)

        # Uncompute the arithmetic workspace
        const, q, carry, overflow = add_offset_dag(
            const, q, carry, overflow
        )
        const = prepare_const(const)
        return q, const, carry, overflow, flag, anc

    return objective


def make_leq_flag_kernel(n: int, l: int):
    """Apply flag ^= 1[q <= l] and restore all arithmetic workspace qubits."""
    prepare_const, add_offset, add_offset_dag = make_comparator_arithmetic(n, l)

    @qmc.qkernel
    def comparator(
        q: qmc.Vector[qmc.Qubit],
        const: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        anc: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        const = prepare_const(const)
        const, q, carry, overflow = add_offset(
            const, q, carry, overflow
        )

        flag = qmc.x(flag)
        overflow, flag = qmc.cx(overflow, flag)

        const, q, carry, overflow = add_offset_dag(
            const, q, carry, overflow
        )
        const = prepare_const(const)
        return q, const, carry, overflow, flag, anc

    return comparator


def make_cvar_objective_kernel(n: int, l_alpha: int, c: float):
    """CVaR objective using the Woerner--Egger u=0 Taylor approximation."""
    if l_alpha <= 0:
        raise ValueError("l_alpha must be positive for the CVaR objective kernel")

    comparator = make_leq_flag_kernel(n, l_alpha)
    cry = qmc.control(qmc.ry)
    ccry = qmc.control(qmc.ry, num_controls=2)

    base_angle = float(np.pi / 2.0 - c)
    bit_angles = [float(2.0 * c * (2 ** bit) / l_alpha) for bit in range(n)]

    @qmc.qkernel
    def base_rotation(
        q: qmc.Vector[qmc.Qubit],
        const: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        anc: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        flag, anc = cry(flag, anc, angle=base_angle)
        return q, const, carry, overflow, flag, anc

    def make_bit_rotation(bit: int, angle: float):
        @qmc.qkernel
        def bit_rotation(
            q: qmc.Vector[qmc.Qubit],
            const: qmc.Vector[qmc.Qubit],
            carry: qmc.Qubit,
            overflow: qmc.Qubit,
            flag: qmc.Qubit,
            anc: qmc.Qubit,
        ) -> tuple[
            qmc.Vector[qmc.Qubit],
            qmc.Vector[qmc.Qubit],
            qmc.Qubit,
            qmc.Qubit,
            qmc.Qubit,
            qmc.Qubit,
        ]:
            flag, q[bit], anc = ccry(flag, q[bit], anc, angle=angle)
            return q, const, carry, overflow, flag, anc

        return bit_rotation

    rotation_kernel = base_rotation
    for bit, angle in enumerate(bit_angles):
        rotation_kernel = merge_system_kernels(
            rotation_kernel,
            make_bit_rotation(bit, angle),
        )

    @qmc.qkernel
    def objective(
        q: qmc.Vector[qmc.Qubit],
        const: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        anc: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        # flag = 1[q <= l_alpha]
        q, const, carry, overflow, flag, anc = comparator(
            q, const, carry, overflow, flag, anc
        )

        # Apply the efficient F only to the tail region
        q, const, carry, overflow, flag, anc = rotation_kernel(
            q, const, carry, overflow, flag, anc
        )

        # Reset flag to |0>
        q, const, carry, overflow, flag, anc = comparator(
            q, const, carry, overflow, flag, anc
        )
        return q, const, carry, overflow, flag, anc

    return objective



# %% [markdown]
# ### Implementing Quantum Amplitude Estimation
#
# Using the operators $F$ constructed above for VaR and CVaR, we now build QAE itself.
# We combine amplitude encoding of the probability distribution with $F$ to construct the state-preparation operator $\mathcal{A}$, and from it build the Grover operator $\mathcal{Q} = \mathcal{A} S_0 \mathcal{A}^\dagger S_\chi$ used in QAE.
#

# %%
# ===================================================
# Step 3: Fully QAE
# ===================================================

def make_a_kernel(n: int, amplitudes: np.ndarray, objective_kernel):
    """Kernel for A = F · R."""

    @qmc.qkernel
    def a_kernel(
        q: qmc.Vector[qmc.Qubit],
        const: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        anc: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        q = qmc.amplitude_encoding(q, amplitudes)
        q, const, carry, overflow, flag, anc = objective_kernel(
            q, const, carry, overflow, flag, anc
        )
        return q, const, carry, overflow, flag, anc

    return a_kernel


def make_grover_q_kernel(n: int, a_ker, a_dag):
    """Grover reflection operator Q = A · S0 · A† · Schi."""

    @qmc.qkernel
    def s_chi(
        q: qmc.Vector[qmc.Qubit],
        const: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        anc: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        anc = qmc.x(anc)
        anc = qmc.z(anc)
        anc = qmc.x(anc)
        return q, const, carry, overflow, flag, anc

    # controls = q(n) + const(n) + carry + overflow + flag = 2n+3
    mcz_all = qmc.control(qmc.z, num_controls=2 * n + 3)

    @qmc.qkernel
    def s_0(
        q: qmc.Vector[qmc.Qubit],
        const: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        anc: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        q = qmc.x(q)
        const = qmc.x(const)
        carry = qmc.x(carry)
        overflow = qmc.x(overflow)
        flag = qmc.x(flag)
        anc = qmc.x(anc)

        q, const, carry, overflow, flag, anc = mcz_all(
            q, const, carry, overflow, flag, anc
        )

        q = qmc.x(q)
        const = qmc.x(const)
        carry = qmc.x(carry)
        overflow = qmc.x(overflow)
        flag = qmc.x(flag)
        anc = qmc.x(anc)
        return q, const, carry, overflow, flag, anc

    @qmc.qkernel
    def q_kernel(
        q: qmc.Vector[qmc.Qubit],
        const: qmc.Vector[qmc.Qubit],
        carry: qmc.Qubit,
        overflow: qmc.Qubit,
        flag: qmc.Qubit,
        anc: qmc.Qubit,
    ) -> tuple[
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        q, const, carry, overflow, flag, anc = s_chi(
            q, const, carry, overflow, flag, anc
        )
        q, const, carry, overflow, flag, anc = a_dag(
            q, const, carry, overflow, flag, anc
        )
        q, const, carry, overflow, flag, anc = s_0(
            q, const, carry, overflow, flag, anc
        )
        q, const, carry, overflow, flag, anc = a_ker(
            q, const, carry, overflow, flag, anc
        )
        return q, const, carry, overflow, flag, anc

    return q_kernel


def make_q_power_kernel(q_kernel, power: int):
    if power == 1:
        return q_kernel

    result = q_kernel
    for _ in range(power - 1):
        result = merge_system_kernels(result, q_kernel)
    return result


def make_qae_kernel(
    n: int,
    m: int,
    amplitudes: np.ndarray,
    objective_kernel,
):
    a_ker = make_a_kernel(n, amplitudes, objective_kernel)
    a_dag = qmc.inverse(a_ker)
    q_ker = make_grover_q_kernel(n, a_ker, a_dag)

    def merge_qpe_steps(left, right):
        @qmc.qkernel
        def merged_step(
            sv: qmc.Vector[qmc.Qubit],
            q: qmc.Vector[qmc.Qubit],
            const: qmc.Vector[qmc.Qubit],
            carry: qmc.Qubit,
            overflow: qmc.Qubit,
            flag: qmc.Qubit,
            anc: qmc.Qubit,
        ) -> tuple[
            qmc.Vector[qmc.Qubit],
            qmc.Vector[qmc.Qubit],
            qmc.Vector[qmc.Qubit],
            qmc.Qubit,
            qmc.Qubit,
            qmc.Qubit,
            qmc.Qubit,
        ]:
            sv, q, const, carry, overflow, flag, anc = left(
                sv, q, const, carry, overflow, flag, anc
            )
            sv, q, const, carry, overflow, flag, anc = right(
                sv, q, const, carry, overflow, flag, anc
            )
            return sv, q, const, carry, overflow, flag, anc

        return merged_step

    def build_qpe_kernel(step_list: list):
        steps = list(step_list)
        while len(steps) > 1:
            next_steps = []
            for i in range(0, len(steps), 2):
                if i + 1 < len(steps):
                    next_steps.append(merge_qpe_steps(steps[i], steps[i + 1]))
                else:
                    next_steps.append(steps[i])
            steps = next_steps
        return steps[0]

    def make_qpe_step(k: int):
        cqk = qmc.control(make_q_power_kernel(q_ker, 2 ** k))
        sk = k

        @qmc.qkernel
        def qpe_step(
            sv: qmc.Vector[qmc.Qubit],
            q: qmc.Vector[qmc.Qubit],
            const: qmc.Vector[qmc.Qubit],
            carry: qmc.Qubit,
            overflow: qmc.Qubit,
            flag: qmc.Qubit,
            anc: qmc.Qubit,
        ) -> tuple[
            qmc.Vector[qmc.Qubit],
            qmc.Vector[qmc.Qubit],
            qmc.Vector[qmc.Qubit],
            qmc.Qubit,
            qmc.Qubit,
            qmc.Qubit,
            qmc.Qubit,
        ]:
            sv[sk] = qmc.h(sv[sk])
            sv[sk], q, const, carry, overflow, flag, anc = cqk(
                sv[sk], q, const, carry, overflow, flag, anc
            )
            return sv, q, const, carry, overflow, flag, anc

        return qpe_step

    step_kernels = [make_qpe_step(k) for k in range(m)]
    qpe_kernel = build_qpe_kernel(step_kernels)

    @qmc.qkernel
    def qae_kernel() -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, name="q")
        const = qmc.qubit_array(n, name="const")
        carry = qmc.qubit(name="carry")
        overflow = qmc.qubit(name="overflow")
        flag = qmc.qubit(name="flag")
        anc = qmc.qubit(name="anc")
        sv = qmc.qubit_array(m, name="sv")

        q, const, carry, overflow, flag, anc = a_ker(
            q, const, carry, overflow, flag, anc
        )
        sv, q, const, carry, overflow, flag, anc = qpe_kernel(
            sv, q, const, carry, overflow, flag, anc
        )
        sv = iqft(sv)
        return qmc.measure(sv)

    return qae_kernel


def estimate_amplitude_qae(
    n: int,
    m: int,
    amplitudes: np.ndarray,
    objective_kernel,
    shots: int = 4096,
) -> float:
    M = 2 ** m
    transpiler = QiskitTranspiler()
    kernel = make_qae_kernel(n, m, amplitudes, objective_kernel)
    exe = transpiler.transpile(kernel)
    result = exe.sample(transpiler.executor(), shots=shots).result()

    counts: dict[int, int] = {}
    for outcome, count in result.results:
        if isinstance(outcome, (int, np.integer)):
            y = int(outcome)
        elif isinstance(outcome, tuple):
            flat = []
            for b in outcome:
                if isinstance(b, tuple):
                    flat.extend(int(x) for x in b)
                else:
                    flat.append(int(b))
            y = sum(bit * (2 ** k) for k, bit in enumerate(flat))
        else:
            y = int(outcome)
        counts[y] = counts.get(y, 0) + count

    y_star = max(counts, key=counts.get)
    if y_star > M // 2:
        y_star = M - y_star

    return float(np.sin(y_star * np.pi / M) ** 2)



# %% [markdown]
# ### Computing VaR and CVaR
#
# We compute $\mathrm{VaR}_\alpha$ by combining QAE-based estimation of $P[X \leq \ell]$ with binary search.
# At each step, QAE estimates the cumulative probability at the midpoint $\ell_\mathrm{mid}$, and the search interval is narrowed by comparing that estimate with $1-\alpha$.
# For CVaR, we apply the $u = 0$ objective $F$ only to the region below the estimated VaR and then perform QAE again.
#

# %%
# ===================================================
# Step 4: VaR computation
# ===================================================

def compute_var(
    alpha: float,
    n: int,
    m: int,
    amplitudes: np.ndarray,
    mu: float,
    sigma: float,
    shots: int = 4096,
) -> tuple[int, float, float]:
    """Use QAE to find the lower (1-alpha) quantile, treating alpha as the confidence level."""
    N = 2 ** n
    tail_prob = 1.0 - alpha
    x_vals = np.linspace(mu - 3 * sigma, mu + 3 * sigma, N)

    l_low, l_high = 0, N - 1
    step = 0

    print(
        f"  Starting VaR computation (confidence={alpha:.1%}, tail={tail_prob:.1%}, "
        f"n={n}, m={m}, N={N}, M={2**m})"
    )

    while l_low < l_high:
        step += 1
        l_mid = (l_low + l_high) // 2
        objective = make_var_objective_kernel(n, l_mid)
        prob = estimate_amplitude_qae(
            n, m, amplitudes, objective, shots=shots
        )

        print(
            f"    Step {step}: l_mid={l_mid} (x={x_vals[l_mid]:.3f}), "
            f"P[X<=l_mid]~{prob:.4f}"
        )

        if prob >= tail_prob:
            l_high = l_mid
        else:
            l_low = l_mid + 1

    var_alpha_index = l_low
    var_alpha_x = x_vals[var_alpha_index]

    objective = make_var_objective_kernel(n, var_alpha_index)
    prob_var = estimate_amplitude_qae(
        n, m, amplitudes, objective, shots=shots
    )

    print(
        f"  -> VaR_{alpha:.0%} = index {var_alpha_index} "
        f"(x={var_alpha_x:.4f}), P[X<=VaR]~{prob_var:.4f}"
    )
    return var_alpha_index, var_alpha_x, prob_var


# ===================================================
# Step 5: CVaR computation and validity checks
# ===================================================

def validate_cvar_estimate(
    truncated_normalized_mean: float,
    conditional_mean_index: float,
    cvar_x: float,
    prob_var: float,
    var_alpha_index: int,
    var_alpha_x: float,
    x_min: float,
    atol: float = 1e-9,
) -> tuple[bool, list[str]]:
    """Check whether the reconstructed lower-tail CVaR satisfies the required mathematical conditions."""
    reasons: list[str] = []

    values = [
        truncated_normalized_mean,
        conditional_mean_index,
        cvar_x,
        prob_var,
    ]
    if not all(np.isfinite(v) for v in values):
        reasons.append("non-finite value detected")
        return False, reasons

    # In the tail, 0 <= i/l <= 1, so
    # E[(i/l) 1[i<=l]] must lie in [0, P(i<=l)].
    if not (-atol <= truncated_normalized_mean <= prob_var + atol):
        reasons.append("truncated normalized mean is outside [0, P(tail)]")

    # The conditional mean index must lie within the tail index range [0, l_alpha].
    if not (-atol <= conditional_mean_index <= var_alpha_index + atol):
        reasons.append(
            f"conditional mean index is outside [0, {var_alpha_index}]"
        )

    # Lower-tail CVaR must be no smaller than the minimum value and no greater than VaR.
    if not (x_min - atol <= cvar_x <= var_alpha_x + atol):
        reasons.append(
            "lower-tail CVaR is outside "
            f"[{x_min:.6f}, VaR={var_alpha_x:.6f}]"
        )

    return len(reasons) == 0, reasons


def classical_cvar_at_index(
    var_idx: int,
    amplitudes: np.ndarray,
    mu: float,
    sigma: float,
) -> float:
    """Compute the classical CVaR for the same discrete distribution and VaR index."""
    probs = np.asarray(amplitudes, dtype=float) ** 2
    N = len(probs)
    x_vals = np.linspace(mu - 3 * sigma, mu + 3 * sigma, N)

    tail_probability = float(np.sum(probs[: var_idx + 1]))
    if tail_probability <= 0.0:
        return float("nan")

    return float(
        np.dot(x_vals[: var_idx + 1], probs[: var_idx + 1])
        / tail_probability
    )


def compute_cvar(
    alpha: float,
    var_alpha_index: int,
    var_alpha_x: float,
    prob_var: float,
    n: int,
    m: int,
    amplitudes: np.ndarray,
    mu: float,
    sigma: float,
    shots: int = 4096,
) -> float:
    """Estimate lower-tail CVaR using the Woerner--Egger u=0 efficient F.

    If the reconstructed conditional mean violates the mathematical constraints of the tail region,
    the finite-m QAE precision is treated as insufficient and np.nan is returned.
    """
    del alpha  # Kept as an argument to make the function definition explicit

    N = 2 ** n
    x_vals = np.linspace(mu - 3 * sigma, mu + 3 * sigma, N)
    x_min = x_vals[0]
    dx = x_vals[1] - x_vals[0]

    if prob_var <= 0.0 or not np.isfinite(prob_var):
        print("  [CVaR invalid] P[X<=VaR] is non-positive or non-finite.")
        return float("nan")

    # If l_alpha = 0, the selected region contains only index 0.
    if var_alpha_index == 0:
        return float(x_min)

    c = choose_u0_scaling_c(m)
    objective = make_cvar_objective_kernel(n, var_alpha_index, c)
    amplitude = estimate_amplitude_qae(
        n, m, amplitudes, objective, shots=shots
    )

    if not np.isfinite(amplitude):
        print("  [CVaR invalid] QAE amplitude is not finite.")
        return float("nan")

    # u=0:
    # amplitude ~= c * E[(i/l) 1[i<=l]] + (1-c)/2 * P[i<=l]
    truncated_normalized_mean = (
        amplitude - 0.5 * (1.0 - c) * prob_var
    ) / c

    conditional_mean_index = (
        var_alpha_index * truncated_normalized_mean / prob_var
    )
    cvar_x = x_min + dx * conditional_mean_index

    print(
        f"  CVaR efficient F: c={c:.4f}, QAE amplitude~{amplitude:.4f}, "
        f"E[index | tail]~{conditional_mean_index:.4f}"
    )

    cvar_valid, reasons = validate_cvar_estimate(
        truncated_normalized_mean=truncated_normalized_mean,
        conditional_mean_index=conditional_mean_index,
        cvar_x=cvar_x,
        prob_var=prob_var,
        var_alpha_index=var_alpha_index,
        var_alpha_x=var_alpha_x,
        x_min=x_min,
    )

    if not cvar_valid:
        print("  [CVaR invalid] finite-m QAE precision is insufficient.")
        for reason in reasons:
            print(f"    - {reason}")
        print(f"    raw CVaR estimate = {cvar_x:.4f}")
        print("    -> This CVaR estimate will be discarded.")
        return float("nan")

    return float(cvar_x)



# %% [markdown]
# ### Implementing the Visualization
#
# To make the results easier to interpret, we implement a plotting function that visualizes the input distribution together with the estimated VaR and CVaR.
#

# %%
# ===================================================
# Step 6: Visualization (accuracy comparison for different m)
# ===================================================

def plot_results_vs_m(
    results_by_m: dict,
    n: int,
    alpha: float,
    mu: float,
    sigma: float,
):
    m_list = sorted(results_by_m.keys())
    N = 2 ** n
    tail_prob = 1.0 - alpha

    x_vals = np.linspace(mu - 3 * sigma, mu + 3 * sigma, N)
    probs = norm.pdf(x_vals, mu, sigma)
    probs /= probs.sum()

    theory_var = norm.ppf(tail_prob, mu, sigma)

    n_cols = len(m_list)
    fig = plt.figure(figsize=(4 * n_cols, 6))
    gs = plt.GridSpec(1, n_cols, wspace=0.35)

    for col, m in enumerate(m_list):
        ax = fig.add_subplot(gs[0, col])
        res = results_by_m[m]

        var_idx = res["var_alpha_index"]
        var_x = res["var_alpha_x"]
        cvar_val = res["cvar_alpha"]
        cvar_valid = res["cvar_valid"]
        cvar_reference = res["cvar_reference"]

        bar_width = (x_vals[1] - x_vals[0]) * 0.85
        ax.bar(
            x_vals[: var_idx + 1],
            probs[: var_idx + 1],
            width=bar_width,
            color="#D85A30",
            alpha=0.6,
            label="CVaR region",
        )
        ax.bar(
            x_vals[var_idx + 1 :],
            probs[var_idx + 1 :],
            width=bar_width,
            color="#1f77b4",
            alpha=0.6,
        )

        ax.axvline(
            x=var_x,
            color="#D85A30",
            linestyle="--",
            linewidth=1.8,
            label=f"QAE VaR = {var_x:.2f}",
        )

        # Plot the classical CVaR for the same discrete distribution and VaR index as a reference line.
        if np.isfinite(cvar_reference):
            ax.axvline(
                x=cvar_reference,
                color="gray",
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
                label=f"Discrete CVaR = {cvar_reference:.2f}",
            )

        # Plot only QAE CVaR estimates that satisfy the mathematical validity conditions.
        if cvar_valid and np.isfinite(cvar_val):
            ax.axvline(
                x=cvar_val,
                color="#7F77DD",
                linestyle=":",
                linewidth=1.8,
                label=f"QAE CVaR = {cvar_val:.2f}",
            )

        ax.axvline(
            x=theory_var,
            color="gray",
            linestyle="-.",
            linewidth=1.0,
            alpha=0.7,
            label=f"Continuous VaR = {theory_var:.2f}",
        )

        status_lines = []
        if res.get("qae_resolution_warning", False):
            status_lines.append("QAE resolution: coarse")
        if not cvar_valid:
            status_lines.append("CVaR invalid - not plotted")

        if status_lines:
            ax.text(
                0.98,
                0.97,
                "\n".join(status_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                bbox={"boxstyle": "round", "alpha": 0.15},
            )

        ax.set_title(f"$m={m}$ ($M={2**m}$)", fontsize=11)
        ax.set_xlabel("Portfolio value $X$", fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
        ax.set_xlim(x_vals[0] - 0.2, x_vals[-1] + 0.2)

    plt.suptitle(
        f"Quantum Risk Analysis — Woerner & Egger (2019)\n"
        f"Fully QAE: fixed $n={n}$ ($N={N}$), varied $m$ \n"
        f"(confidence $\\alpha={alpha}$, tail $1-\\alpha={tail_prob:.3f}$)",
        fontsize=12,
        y=1.02,
    )
    plt.show()



# %%
# ===================================================
# Numerical sanity checks
# ===================================================

def run_numerical_sanity_checks():
    print("Running numerical sanity checks...")

    # ------------------------------------------------
    # Check 1: the probabilities used for amplitude encoding are normalized
    # ------------------------------------------------
    test_amplitudes = np.sqrt(np.array([0.5, 0.5], dtype=float))
    np.testing.assert_allclose(
        np.sum(test_amplitudes ** 2),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    # ------------------------------------------------
    # Check 2: a small QAE instance with analytical result P[X<=0] = 0.5
    # n=1, p(0)=p(1)=0.5, l=0。
    # With m=2 (M=4), a=0.5 is represented exactly as a QAE candidate.
    # ------------------------------------------------
    test_objective = make_var_objective_kernel(n=1, l=0)
    test_qae = estimate_amplitude_qae(
        n=1,
        m=2,
        amplitudes=test_amplitudes,
        objective_kernel=test_objective,
        shots=1024,
    )
    np.testing.assert_allclose(
        test_qae,
        0.5,
        rtol=0.0,
        atol=1e-12,
    )

    # ------------------------------------------------
    # Check 3: do not reject a valid CVaR example
    # ------------------------------------------------
    valid, reasons = validate_cvar_estimate(
        truncated_normalized_mean=0.10,
        conditional_mean_index=1.50,
        cvar_x=-2.40,
        prob_var=0.20,
        var_alpha_index=3,
        var_alpha_x=-1.80,
        x_min=-3.00,
    )
    assert valid, f"A valid CVaR example was rejected: {reasons}"

    # ------------------------------------------------
    # Check 4: reliably reject an invalid CVaR example
    # Intentionally construct E[index | tail] > l_alpha and CVaR > VaR.
    # ------------------------------------------------
    invalid, reasons = validate_cvar_estimate(
        truncated_normalized_mean=0.10,
        conditional_mean_index=5.00,
        cvar_x=-1.00,
        prob_var=0.08,
        var_alpha_index=4,
        var_alpha_x=-1.40,
        x_min=-3.00,
    )
    assert not invalid, "An invalid CVaR example was not rejected."
    assert len(reasons) > 0

    print("All numerical sanity checks passed.")


run_numerical_sanity_checks()



# %% [markdown]
# ## Results
#
# Finally, we write the main execution cell that runs the implementation developed above.
# Let us execute it and inspect the results.
#

# %%
# ===================================================
# Main execution
# ===================================================

alpha = 0.95     # Confidence level; lower-tail probability is 1-alpha = 0.05
shots = 4096
mu = 0.0
sigma = 1.0
n = 4
m_list = [1, 3, 5]

tail_prob = 1.0 - alpha

print("=" * 60)
print("Quantum Risk Analysis — Fully QAE (Woerner--Egger u=0 F)")
print(f"n={n} fixed (N={2**n}), m={m_list}")
print(
    f"confidence alpha={alpha:.2f} ({alpha:.0%}), "
    f"lower-tail probability={tail_prob:.2f} ({tail_prob:.0%})"
)
print(f"mu={mu}, sigma={sigma}")
print("=" * 60)

# Continuous normal reference
z = norm.ppf(tail_prob)
theory_var = norm.ppf(tail_prob, mu, sigma)
theory_cvar = mu - sigma * norm.pdf(z) / tail_prob
print(
    f"\nContinuous normal: VaR_{alpha:.0%}={theory_var:.4f}, "
    f"CVaR_{alpha:.0%}={theory_cvar:.4f}\n"
)

# Distribution preparation
amplitudes = make_normal_amplitudes(n, mu=mu, sigma=sigma)

# Also verify the normalization of the production amplitude-encoding input with an assertion.
np.testing.assert_allclose(
    np.sum(amplitudes ** 2),
    1.0,
    rtol=0.0,
    atol=1e-12,
)

results_by_m = {}

for m in m_list:
    print(f"\n{'=' * 45}")
    print(f"m={m} (M={2**m})")
    print(f"{'=' * 45}")

    # Smallest positive amplitude candidate representable by Fully QAE.
    # If this is larger than the target tail probability, probabilities around 5%
    # are likely to be too coarsely resolved, so emit an explicit warning.
    M = 2 ** m
    min_positive_amplitude = float(np.sin(np.pi / M) ** 2)
    resolution_warning = min_positive_amplitude > tail_prob

    if resolution_warning:
        print(
            "  [QAE resolution warning] "
            f"smallest positive amplitude={min_positive_amplitude:.4f} "
            f"> target tail probability={tail_prob:.4f}"
        )
        print(
            "    -> probabilities near the target tail are "
            "too coarsely resolved for this m."
        )

    # VaR
    var_idx, var_x, prob_var = compute_var(
        alpha,
        n,
        m,
        amplitudes,
        mu=mu,
        sigma=sigma,
        shots=shots,
    )

    # CVaR
    cvar = compute_cvar(
        alpha,
        var_idx,
        var_x,
        prob_var,
        n,
        m,
        amplitudes,
        mu=mu,
        sigma=sigma,
        shots=shots,
    )
    cvar_valid = bool(np.isfinite(cvar))

    # Compute the classical CVaR for the same discrete distribution and VaR index as a reference.
    cvar_reference = classical_cvar_at_index(
        var_idx,
        amplitudes,
        mu,
        sigma,
    )

    print("\n--- result ---")
    print(
        f"VaR_{alpha:.0%}  = {var_x:.4f}  "
        f"(continuous normal {theory_var:.4f})"
    )

    if cvar_valid:
        cvar_abs_error = abs(cvar - cvar_reference)
        print(f"CVaR_{alpha:.0%} = {cvar:.4f}")
        print(
            f"  discrete classical reference at same VaR = "
            f"{cvar_reference:.4f}"
        )
        print(f"  absolute error = {cvar_abs_error:.4f}")
    else:
        cvar_abs_error = float("nan")
        print(f"CVaR_{alpha:.0%} = unavailable")
        print(
            "  insufficient QAE precision; invalid estimate discarded"
        )
        print(
            f"  discrete classical reference at same VaR = "
            f"{cvar_reference:.4f}"
        )

    results_by_m[m] = {
        "var_alpha_index": var_idx,
        "var_alpha_x": var_x,
        "prob_var": prob_var,
        "cvar_alpha": cvar,
        "cvar_valid": cvar_valid,
        "cvar_reference": cvar_reference,
        "cvar_abs_error": cvar_abs_error,
        "qae_resolution_warning": resolution_warning,
        "qae_min_positive_amplitude": min_positive_amplitude,
    }


# ===================================================
# Regression assertions for the main experiment
# ===================================================

# Prevent the notebook from succeeding if every CVaR silently becomes NaN.
# Regression guard.
assert any(
    result["cvar_valid"] for result in results_by_m.values()
), (
    "No valid CVaR estimate was obtained. "
    "Check QAE precision and CVaR reconstruction."
)

# Verify the valid/invalid storage convention and the required condition for lower-tail CVaR.
for m, result in results_by_m.items():
    if result["cvar_valid"]:
        assert np.isfinite(result["cvar_alpha"])
        assert result["cvar_alpha"] <= result["var_alpha_x"] + 1e-9, (
            f"m={m}: valid CVaR exceeds VaR."
        )
    else:
        assert np.isnan(result["cvar_alpha"]), (
            f"m={m}: invalid CVaR must be NaN."
        )

print("\nMain-result validation passed.")

plot_results_vs_m(
    results_by_m,
    n=n,
    alpha=alpha,
    mu=mu,
    sigma=sigma,
)

# %% [markdown]
# The figure shows the discretized normal distribution used as input, the VaR and CVaR estimated by QAE, and the corresponding classically computed VaR/CVaR reference values as vertical lines.
# The orange bars indicate the lower-tail region at or below VaR, while the blue bars indicate the region above VaR.
# $m$ is the number of sampling qubits in the QPE subroutine used inside QAE, and $M=2^m$ determines the resolution of the phase and, consequently, the amplitude estimate.
# In these results, the VaR estimate improves substantially for $m=3,5$ compared with $m=1$.
# However, because finite-$m$ QAE can estimate only a discrete set of amplitudes and those estimates are used inside binary search, the VaR error does not necessarily decrease monotonically as $m$ increases.
# CVaR, on the other hand, is a conditional expectation over the region below VaR, and this implementation uses the $u=0$ Taylor-approximated objective $F$.  
# Therefore, the CVaR error includes not only finite-precision QAE error, but also the approximation error in $F$ and the error in the VaR estimate.
# In particular, when the tail probability is small, QAE error can be amplified when reconstructing the conditional expectation, so these results do not show a monotonic improvement in CVaR as $m$ increases.
# For $m=5$, the CVaR estimate is especially inaccurate and does not satisfy $\mathrm{CVaR} \leq \mathrm{VaR}$, so the corresponding CVaR line is not displayed.
# In addition, because the distribution supplied to the quantum circuit is discretized into $N=2^n$ points, there is a discretization error due to finite $n$ in addition to the QAE estimation error.
# Therefore, achieving higher accuracy requires improving both $m$, which determines the QAE resolution, and $n$, which determines the discretization resolution of the probability distribution.
#

# %% [markdown]
# ## Summary
#
# In this document, we implemented the QAE-based VaR and CVaR calculations from [Woerner & Egger (2019)](https://www.nature.com/articles/s41534-019-0130-6) using Qamomile.
# The key points are summarized below.
#
# * We use Qamomile's `ripple_carry_add` (a function that reversibly adds integers stored in two quantum registers) to construct a comparator for $i \leq \ell$, and use QAE to estimate $P[X \leq \ell]$ from the comparison result.
# * QAE is constructed directly in Qamomile by combining the state-preparation operator $\mathcal{A}$, the Grover operator $\mathcal{Q} = \mathcal{A} S_0 \mathcal{A}^\dagger S_\chi$, and the inverse quantum Fourier transform.
# * In the numerical experiments, we set the QPE phase-estimation register to $m = 1, 3, 5$ and investigated the resulting differences in QAE resolution. The VaR estimate improves substantially from the coarsest case $m=1$, although the improvement is not necessarily monotonic.
# * CVaR is affected not only by QAE error, but also by the Taylor-expansion approximation and the VaR estimation error, and no clear improvement was observed over the range tested here.
# * The final error contains both the estimation error associated with finite $m$ in QAE and the discretization error arising from the number of distribution qubits $n$.
#

# %% [markdown]
#
