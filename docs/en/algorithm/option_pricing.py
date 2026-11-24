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
# tags: [algorithm, finance, simulation]
# ---
#
# # Option Pricing with Quantum Amplitude Estimation
#
# Quantum amplitude estimation on gate-based quantum computers has been proposed for a wide range of applications. Option pricing in financial engineering is one of its representative applications. Classical Monte Carlo methods have traditionally been used for option pricing and require $\mathcal{O}(\epsilon^{-2})$ oracle calls to reach a target precision $\epsilon$. In contrast, quantum amplitude estimation requires $\mathcal{O}(\epsilon^{-1})$ calls, providing a quadratic speedup. This article explains and implements a quantum amplitude estimation approach to option and option-portfolio pricing {cite:p}`10.22331/q-2020-07-06-291`. Use this option-pricing example to learn how to work with Qamomile.

# %%
# Install Qamomile with Qiskit, SciPy and Matplotlib from PyPI.
# #!pip install "qamomile[qiskit]" scipy matplotlib

# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import qamomile as qm
from qamomile.qiskit import Aer
from qamomile.stdlib import amplitude_encoding, uniformly_controlled_ry

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Background
#
# First, let us define the terms used in this article.
#
# * Asset: a financial instrument, such as a stock, that underlies an option.
# * Option: a right to buy or sell an asset at a specified price on a future date (the maturity date).
# * Payoff: the value of an option at maturity.
#
# ### The option-pricing problem
#
# Financial risk calculations are in high demand. Monte Carlo (MC) methods are central to these calculations and are widely used for estimating value at risk (VaR) and pricing over-the-counter derivatives. Their drawback is that the error decays as $\varepsilon = \mathcal{O} (M^{-1/2})$. Here, $M$ is the number of samples. When the target probability distribution can be prepared as a quantum state, quantum amplitude estimation (QAE) can estimate statistics such as expected values with error $\mathcal{O} (M^{-1})$. This provides a quadratic speedup over classical MC.
#
# ### Prior work
#
# A theoretical framework for quantum algorithms for Monte Carlo pricing was presented in {cite:p}`10.1103/PhysRevA.98.022321`. The paper showed theoretically that its pricing algorithm achieves a quadratic speedup over classical Monte Carlo methods. It discussed both European and Asian options and demonstrated the method’s effectiveness in numerical simulations for European options. However, its contribution was limited to a theoretical framework and numerical simulations; it did not demonstrate execution on quantum hardware.
#
# Another important work is {cite:p}`10.1038/s41534-019-0130-6`. It applied QAE to value at risk (VaR) and conditional value at risk (CVaR). It improved the payoff calculation of the Monte Carlo pricing framework, substantially reducing the required number of qubits and gates {cite:p}`10.1103/PhysRevA.98.022321`. An implementation then extended that payoff-calculation method to option pricing {cite:p}`10.22331/q-2020-07-06-291`.
#
# ## Algorithm: Option pricing with QAE
#
# ### Calculating the payoff
#
# QAE repeats the Grover operator $\mathcal{Q}$ $k$ times, causing the measurement probability to follow the known form $\sin^2((2k+1)\theta_a)$. This makes it possible to estimate $\theta_a$ accurately with relatively few measurements. Let $S_i$ be an asset price and $p_i$ its probability. Suppose that an operator $\mathcal{A}$ prepares the state
#
# $$
# \mathcal{A} \vert 0 \rangle_{n+1}
# = \sum_{i=0}^{2^n - 1} \sqrt{1-f(S_i)} \sqrt{p_i} \vert S_i \rangle \vert 0 \rangle + \sum_{i=0}^{2^n - 1} \sqrt{f(S_i)} \sqrt{p_i} \vert S_i \rangle \vert 1 \rangle \tag{1}
# $$
#
# Then the probability of observing the payoff qubit in $\vert 1 \rangle$ is $a = \mathbb{E}[f(S)]$, allowing QAE to calculate the expected payoff. A binary payoff representation in an auxiliary register increases the number of qubits and circuit depth {cite:p}`10.1103/PhysRevA.98.022321`. The lightweight $R_y$ method avoids this overhead {cite:p}`10.22331/q-2020-07-06-291,10.1038/s41534-019-0130-6`.
#
# ### Applying the method to an expected value and the parameter $c$
#
# To expand around $\pi/4$, where the derivative of $\sin^2$ is maximal, define the target amplitude angle as
#
# $$
# \tilde{f}(i) = 2 \frac{f(i) - f_\mathrm{min}}{f_\mathrm{max} - f_\mathrm{min}} - 1, \qquad
# \alpha(i) = c\tilde{f}(i) + \frac{\pi}{4} \tag{2}
# $$
#
# where the scaling parameter $c$ satisfies $0 < c \leq 1$. The probability that the payoff qubit is $\vert 1 \rangle$ is then
#
# $$
# P_1 \approx \frac{1}{2} + c\,\mathbb{E}[\tilde{f}] + \mathcal{O}(c^3) \tag{3}
# $$
#
# Estimating $P_1$ lets us calculate $\mathbb{E}[f(S)]$ from the known values of $c$, $f_\mathrm{min}$, and $f_\mathrm{max}$. If the estimation error of $P_1$ is $\delta P_1$, then $\mathbb{E}[\tilde{f}] \approx (P_1 - 1/2)/c$, so the estimation error is amplified to $\delta P_1/c$. Reducing $c$ increases this amplification, while increasing $c$ increases the approximation bias $\mathcal{O}(c^2)$. This tradeoff gives a convergence rate of $\mathcal{O}(M^{-2/3})$. Although this does not reach a full quadratic speedup, it still outperforms the classical MC rate of $\mathcal{O}(M^{-1/2})$. The implementation below uses lightweight maximum-likelihood amplitude estimation (MLAE), which does not require phase estimation.
#
# ### Applying the method to a call option
#
# The payoff of a call option, $f_C(S_T) = \max(0, S_T - K)$, is a piecewise-linear function with a kink at $K$. This implementation uses uniformly controlled $R_y$ rotations (UCR-Y) to apply the target amplitude angle at each grid point $S_i$:
#
# $$
# \alpha(S_i) = \frac{\pi}{4} - c + \frac{2c\,\max(S_i - K,\,0)}{S_\mathrm{max} - K} \tag{4}
# $$
#
# This directly applies the target amplitude angle. For $S < K$ the angle is the constant $\pi/4 - c$, and for $S \geq K$ it increases linearly, so this piecewise-linear structure can be encoded exactly.

# %% [markdown]
# ## Implementation with Qamomile
#
# ### Preparing the distribution
#
# Let us now examine the option-pricing implementation {cite:p}`10.22331/q-2020-07-06-291`. First, we prepare a discretized log-normal distribution for the asset price at maturity $S_T$ under geometric Brownian motion.

# %%
n_qubits = 4
S0, K, r, sigma, T = 2.0, 1.9, 0.05, 0.4, 40 / 365
if not (sigma > 0 and T > 0):
    raise ValueError("Use positive values for sigma and T in this example.")

mu  = (r - 0.5 * sigma**2) * T
std = sigma * np.sqrt(T)
S_min = float(S0 * np.exp(mu - 3 * std))
S_max = float(S0 * np.exp(mu + 3 * std))
S_grid = np.linspace(S_min, S_max, 2**n_qubits)

def lognpdf(S):
    return (np.exp(-(np.log(S / S0) - mu)**2 / (2 * std**2))
            / (S * std * np.sqrt(2 * np.pi)))

probs = lognpdf(S_grid)
probs /= probs.sum()
amps  = np.sqrt(probs)   # Amplitudes sqrt(p_i) for Mottönen encoding

c     = 0.25             # Scaling parameter
if not 0 < c <= 1:
    raise ValueError("Choose c such that 0 < c <= 1.")

F_max = float(S_max - K)
if F_max <= 0:
    raise ValueError("The payoff is zero at every grid point. Choose K < S_max in this example.")

# ── Target amplitude angles ──
payoff       = np.maximum(S_grid - K, 0.0)
alpha_target = np.pi / 4 - c + 2 * c * payoff / F_max   # shape: (2^n,)

# ── Reference values ──
# exact_E_f: exact expected payoff from numerical integration over the discrete grid
exact_E_f  = float(np.sum(probs * payoff))
exact_fair = np.exp(-r * T) * exact_E_f

# Exact P1: sum_i p_i sin^2(alpha_i), without using the linear approximation
P1_exact  = float(np.sum(probs * np.sin(alpha_target)**2))

# Approximate P1
E_ftilde  = 2 * exact_E_f / F_max - 1
P1_approx = c * E_ftilde + 0.5

print(f"Expected payoff (numerical integration): {exact_E_f:.6f}")
print(f"Fair value (reference):                 {exact_fair:.6f}")
print(f"Exact P1:                               {P1_exact:.6f}")
print(f"Approximate P1 (linear approximation):  {P1_approx:.6f}")
print(f"P1 approximation error:                 {abs(P1_exact - P1_approx):.6f}")


# %% [markdown]
# ### Implementing the payoff operator
#
# As shown in the algorithm section, the call-option payoff $f(S) = \max(S-K, 0)$ is piecewise linear with a kink at $K$. We therefore need to apply a different target amplitude angle at every grid point:
#
# $$
# \alpha(S_i) = \frac{\pi}{4} - c + \frac{2c\,\max(S_i - K,\, 0)}{S_\mathrm{max} - K}
# $$
#
# UCR-Y can apply an independent amplitude angle $\alpha(S_i)$ to each basis state $\vert i \rangle$, which lets us encode the piecewise-linear structure around $K$ exactly. The two-price example below shows how UCR-Y decomposes into $R_y$ and CNOT gates. For the full price grid, we use Qamomile’s `uniformly_controlled_ry`.

# %%
# Ry rotates by twice the desired amplitude angle at each price point.
price_rotation_angles = (2 * alpha_target).tolist()

# With just the lowest and highest prices, one price qubit controls both CNOTs.
# All four gates target the payoff qubit.
low_price_angle, high_price_angle = price_rotation_angles[0], price_rotation_angles[-1]
first_ry = (low_price_angle + high_price_angle) / 2
second_ry = (low_price_angle - high_price_angle) / 2
print(f"Ry({first_ry:.4f}) → CNOT → Ry({second_ry:.4f}) → CNOT")

# On the full grid, the price qubits are the controls and the final qubit is the target.
apply_payoff = uniformly_controlled_ry(price_rotation_angles)
print(f"Number of UCR-Y gates: {2 * len(price_rotation_angles)} (Ry + CNOT)")


# %% [markdown]
# ### Implementing distribution loading
#
# Next, let us implement the operation that loads the prepared asset-price data and its Hermitian adjoint. Qamomile provides the amplitude-encoding function, `amplitude_encoding`, as part of its standard library.

# %%
# P_X loads the distribution. Ak applies its adjoint with qm.adjoint.
load_distribution = amplitude_encoding(tuple(amps))


# %% [markdown]
# Combining distribution loading with UCR-Y gives a circuit that encodes the call-option payoff into the amplitude of the payoff qubit. Run `num_shots` measurements to estimate the probability $P_1$ that the payoff qubit is $\vert 1 \rangle$. We then calculate the expected payoff from the approximation in equation (3), $P_1 \approx \frac{1}{2} + c\mathbb{E}[\tilde{f}]$. Showing that the measured $P_1$ is close to the theoretical value verifies that the payoff circuit works correctly.

# %%
@qm.qkernel
def payoff_call() -> qm.Shot[qm.Bits[1]]:
    q_S = qm.qubits(n_qubits)
    q_p = qm.qubits(1)

    # (1) Load the distribution: P_X
    q_S = load_distribution(q_S)

    # (2) UCR-Y payoff circuit: F
    #     |i>|0> -> |i>(cos alpha(i)|0> + sin alpha(i)|1>)
    #     alpha(i) = pi/4 - c + 2c*max(S_i-K, 0) / (S_max-K)
    #     S < K: constant angle pi/4 - c; S >= K: linear increase
    state = apply_payoff(qm.concat(q_S, q_p))
    q_S, q_p = qm.split_at(state, n_qubits)
    _ = qm.measure(q_S)
    return qm.measure(q_p)


num_shots = 1 if docs_test_mode else 4096
simulator = Aer()
exe_k0 = qm.compile(payoff_call, to=simulator.target)
res_k0 = qm.run(exe_k0, on=simulator, shots=num_shots).result
assert sum(res_k0.counts.values()) == num_shots
ones_k0 = res_k0.counts.get((1,), 0)
P1_k0 = ones_k0 / num_shots

print(f"\n[Qamomile k=0]")
print(f"Measured P1:             {P1_k0:.4f}")
print(f"Exact P1 (reference):    {P1_exact:.4f}")
print(f"Approximate P1:          {P1_approx:.4f}")
print("-> The payoff circuit works correctly when the measurement is close to the exact value.")


# %% [markdown]
# ### Implementing quantum amplitude estimation
#
# Let us construct a circuit that applies the Grover operator $\mathcal{Q} = \mathcal{A} \mathcal{S}_0 \mathcal{A}^\dagger \mathcal{S}_\chi$ $k$ times for amplitude estimation. For the state after $k$ applications, we calculate the probability that the payoff qubit is $\vert 1 \rangle$: $P_1^{(k)} = \sin^2 ((2k+1) \theta_\alpha)$.

# %%
@qm.qkernel
def flip_target(q_p: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(q_p)


@qm.qkernel
def Ak(k: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    q_S = qm.qubits(n_qubits)
    q_p = qm.qubits(1)

    # ── A = F · P_X ──
    q_S = load_distribution(q_S)
    state = apply_payoff(qm.concat(q_S, q_p))

    # ── Apply Q k times ──
    for _ in range(k):

        # S_chi: apply phase -1 when the payoff qubit is |1>
        state = qm.z(state, target=n_qubits)

        # A dagger = P_X dagger · F dagger
        state = qm.adjoint(apply_payoff, state)
        q_S, q_p = qm.split_at(state, n_qubits)
        q_S = qm.adjoint(load_distribution, q_S)

        # S_0: apply phase -1 to |0...0>
        # Invert with X, apply MCZ (= H · MCX · H), then restore with X.
        # Controls: all of q_S; target: q_p. No ancilla is required.
        for i in range(n_qubits):
            q_S = qm.x(q_S, target=i)
        q_p = qm.x(q_p)
        q_p = qm.h(q_p)
        q_S, q_p = qm.control(flip_target, q_S, q_p)
        q_p = qm.h(q_p)
        for i in range(n_qubits):
            q_S = qm.x(q_S, target=i)
        q_p = qm.x(q_p)

        # A
        q_S = load_distribution(q_S)
        state = apply_payoff(qm.concat(q_S, q_p))

    q_S, q_p = qm.split_at(state, n_qubits)
    _ = qm.measure(q_S)
    return qm.measure(q_p)

k_list = [1] if docs_test_mode else [0, 1, 2, 4, 8, 16]
shots_per_k = 1 if docs_test_mode else 2048

h_list, N_list = [], []

# Use the exact value, rather than the linear approximation, for the P1 theory.
theta_a_theory = np.arcsin(np.sqrt(max(0.0, min(1.0, P1_exact))))
print()
for k in k_list:
    program = qm.specialize(Ak, k=k)
    exe = qm.compile(program, to=simulator.target)
    result = qm.run(exe, on=simulator, shots=shots_per_k).result
    assert sum(result.counts.values()) == shots_per_k
    ones = result.counts.get((1,), 0)
    h_list.append(ones)
    N_list.append(shots_per_k)
    P1_k  = ones / shots_per_k
    P1_th = np.sin((2*k + 1) * theta_a_theory)**2
    print(f"k={k:2d}: measured P1={P1_k:.4f}, theoretical (exact) P1={P1_th:.4f}")


# %% [markdown]
# ## Results
#
# Based on the measurements at several $k$ values (binomial data), let us perform maximum-likelihood estimation with the model $P_1^{(k)} = \sin^2 ((2k+1) \theta_\alpha)$. This estimates the amplitude angle $\theta_\alpha$, from which we calculate $P_1 = \sin^2 \theta_\alpha$. We then calculate the expected payoff $\mathbb{E}[f(S)]$ from $P_1$ and discount it at the risk-free rate to obtain the option's fair value. Finally, we compare it with a reference value from an exact calculation.

# %%
# ============================================================
# 4. Maximum-likelihood estimation (MLE)
# ============================================================
def neg_log_lik(theta: float) -> float:
    """
    Negative log likelihood:
      -sum_k [h_k log sin^2((2k+1)theta) + (N_k - h_k) log cos^2((2k+1)theta)]
    """
    ll = 0.0
    for k, h, N in zip(k_list, h_list, N_list):
        s2 = float(np.clip(np.sin((2*k + 1) * theta)**2, 1e-12, 1 - 1e-12))
        ll += h * np.log(s2) + (N - h) * np.log(1 - s2)
    return -ll

opt       = minimize_scalar(neg_log_lik, bounds=(0, np.pi / 2), method='bounded')
theta_hat = opt.x
a_hat     = np.sin(theta_hat)**2

# Recover E[f(S)] from P1 with the equation (3) linear approximation;
# its approximation bias is O(c^2).
# E[f_tilde] approx (P1 - 1/2) / c, E[f] = F_max/2 * (E[f_tilde] + 1)
E_ftilde_hat = (a_hat - 0.5) / c
E_f_hat      = (E_ftilde_hat + 1) * F_max / 2
fair_hat     = np.exp(-r * T) * E_f_hat

print(f"\n===== Estimation results =====")
print(f"Estimated P1:          {a_hat:.6f}  "
      f"(exact {P1_exact:.6f}, approximation {P1_approx:.6f})")
print(f"Estimated E[f(S)]:     {E_f_hat:.6f}  (reference {exact_E_f:.6f})")
print(f"Estimated fair value:  {fair_hat:.6f}  (reference {exact_fair:.6f})")
print(f"Absolute error:        {abs(fair_hat - exact_fair):.6f}")

# %% [markdown]
# The value `E[f(S)]` estimated from the quantum circuit differs from the reference value because it uses the approximation
#
# $$
# \mathbb{E}[f] = \frac{F_\mathrm{max}}{2} \left( \frac{P_1 - \frac{1}{2}}{c} + 1 \right)
# $$
#
# where $F_\mathrm{max} = S_\mathrm{max} - K$. Let us quantify how much of the difference between the estimated $\mathbb{E}[\hat{f}]$ and the reference value comes from this approximation.
# %%
total_error = E_f_hat - exact_E_f
approximation_error = F_max / 2 * (P1_exact - P1_approx) / c
statistical_error = total_error - approximation_error
error_magnitude_sum = abs(approximation_error) + abs(statistical_error)
approximation_share = abs(approximation_error) / error_magnitude_sum

print(f"Approximation error: {approximation_error:.6f}")
print(f"Statistical error:   {statistical_error:.6f}")
print(f"Approximation share of total error magnitude: {approximation_share:.1%}")

# %% [markdown]
# In this example, most of the error comes from the approximation. The remaining error is statistical and can be reduced by increasing the number of measurements in the quantum amplitude-estimation step. However, increasing the number of measurements also increases execution time.
#
# Finally, let us plot the log-likelihood function $\log \mathcal{L} (\sin^2 \theta)$ and $P_1^{(k)}$. The left plot shows the log-likelihood function, the maximum-likelihood estimate, and the theoretical value from the exact calculation. The right plot overlays the measured values with the sine-squared curve $P_1^{(k)} = \sin^2 ((2k+1) \hat{\theta})$ obtained by substituting the estimated $\hat{\theta}$.

# %%
# ============================================================
# 5. Visualization
# ============================================================
theta_grid = np.linspace(1e-4, np.pi / 2 - 1e-4, 400)
ll_grid    = np.array([-neg_log_lik(t) for t in theta_grid])
a_grid     = np.sin(theta_grid)**2

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(a_grid, ll_grid, color='steelblue')
axes[0].axvline(a_hat,     color='red',   ls='--',
                label=f'estimate  {a_hat:.4f}')
axes[0].axvline(P1_exact,  color='navy',  ls=':',
                label=f'exact P1  {P1_exact:.4f}')
axes[0].axvline(P1_approx, color='green', ls='-.',
                label=f'approx P1 {P1_approx:.4f}')
axes[0].set_xlabel('a = sin²(θ)')
axes[0].set_ylabel('log-likelihood')
axes[0].set_title('MLAE likelihood')
axes[0].legend()

k_smooth = np.linspace(0, max(k_list), 500)
P1_sine_curve = np.sin((2 * k_smooth + 1) * theta_hat)**2
axes[1].plot(k_smooth, P1_sine_curve, '-', color='steelblue',
             label=rf'$\sin^2((2k+1)\hat{{\theta}})$ ($\hat{{\theta}}={theta_hat:.3f}$)')
axes[1].plot(k_list, [h / N for h, N in zip(h_list, N_list)],
             'o', color='red', zorder=5, label='measured value')
axes[1].set_xlabel('k  (Grover iterations)')
axes[1].set_ylabel('P₁ estimation')
axes[1].set_title('P₁^(k) = sin²((2k+1)θ̂)')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# MLAE estimates the region where the log likelihood is maximized.

# %% [markdown]
# ## Summary
#
# This article showed how to implement the option-pricing algorithm {cite:p}`10.22331/q-2020-07-06-291` with Qamomile. The key points are as follows.
#
# * We encoded a log-normal distribution into a quantum state with Mottönen encoding and used uniformly controlled rotations to encode the payoff of a call option into the payoff-qubit amplitude. Use Qamomile's `amplitude_encoding` to load the state.
# * We measured circuits with the Grover operator $\mathcal{Q}$ applied $k$ times for several values of $k$, then estimated the amplitude angle by maximum likelihood to calculate the expected payoff.
