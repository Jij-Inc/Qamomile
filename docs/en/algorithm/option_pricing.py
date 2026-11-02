# ---
# jupyter:
#   jupytext:
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
# Quantum amplitude estimation on gate-based quantum computers has been proposed for a wide range of applications.
# Option pricing in financial engineering is one of its representative applications.
# Classical Monte Carlo methods have traditionally been used for option pricing and require $\mathcal{O}(\epsilon^{-2})$ oracle calls to reach a target precision $\epsilon$.
# In contrast, quantum amplitude estimation requires $\mathcal{O}(\epsilon^{-1})$ calls, providing a quadratic speedup.
# This article explains and implements a quantum amplitude estimation approach to option and option-portfolio pricing {cite:p}`10.22331/q-2020-07-06-291`. Use this option-pricing example to learn how to work with Qamomile.
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
import math
import os

import numpy as np

import qamomile as qm
from qamomile.stdlib import (
    amplitude_angles,
    amplitude_encoding,
    amplitude_encoding_from_angles,
    uniformly_controlled_angles,
    uniformly_controlled_ry,
    uniformly_controlled_ry_from_angles,
)

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
# Financial risk calculations are in high demand.
# Monte Carlo (MC) methods are central to these calculations and are widely used for estimating value at risk (VaR) and pricing over-the-counter derivatives.
# Their drawback is that the error decays as $\varepsilon = \mathcal{O} (M^{-1/2})$.
# Here, $M$ is the number of samples.
# When the target probability distribution can be prepared as a quantum state, quantum amplitude estimation (QAE) can estimate statistics such as expected values with error $\mathcal{O} (M^{-1})$.
# This provides a quadratic speedup over classical MC.
#
# ### Prior work
#
# A theoretical framework for quantum algorithms for Monte Carlo pricing was presented in {cite:p}`10.1103/PhysRevA.98.022321`.
# The paper theoretically showed a quadratic speedup over classical pricing calculations.
# It also demonstrated numerical simulations for European and Asian options.
# However, its contribution was limited to a theoretical framework and numerical simulations; it did not demonstrate execution on quantum hardware.
# Another important work is {cite:p}`10.1038/s41534-019-0130-6`.
# Rather than pricing, it applied QAE to value at risk (VaR) and conditional value at risk (CVaR).
# It improved the payoff calculation of the Monte Carlo pricing framework, substantially reducing the required number of qubits and gates {cite:p}`10.1103/PhysRevA.98.022321`.
# An implementation then extended that payoff-calculation method to option pricing {cite:p}`10.22331/q-2020-07-06-291`.
#
# ## Algorithm: Option pricing with QAE
#
# ### Calculating the payoff
#
# QAE repeats the Grover operator $\mathcal{Q}$ $k$ times, causing the measurement probability to follow the known form
# $\sin^2((2k+1)\theta_a)$.
# This makes it possible to estimate $\theta_a$ accurately with relatively few measurements.
# Let $S_i$ be an asset price and $p_i$ its probability.
# Suppose that an operator $\mathcal{A}$ prepares the state
#
# $$
# \mathcal{A} \vert 0 \rangle_{n+1}
# = \sum_{i=0}^{2^n - 1} \sqrt{1-f(S_i)} \sqrt{p_i} \vert S_i \rangle \vert 0 \rangle + \sum_{i=0}^{2^n - 1} \sqrt{f(S_i)} \sqrt{p_i} \vert S_i \rangle \vert 1 \rangle \tag{1}
# $$
#
# Then the probability of observing the payoff qubit in $\vert 1 \rangle$ is $a = \mathbb{E}[f(S)]$, allowing QAE to calculate the expected payoff.
# A binary payoff representation in an auxiliary register increases the number of qubits and circuit depth {cite:p}`10.1103/PhysRevA.98.022321`.
# The lightweight $R_y$ method avoids this overhead {cite:p}`10.22331/q-2020-07-06-291,10.1038/s41534-019-0130-6`.
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
# where $c \in [0, 1]$ is a scaling parameter.
# The probability that the payoff qubit is $\vert 1 \rangle$ is then
#
# $$
# P_1 \approx \frac{1}{2} + c\,\mathbb{E}[\tilde{f}] + \mathcal{O}(c^3) \tag{3}
# $$
#
# Estimating $P_1$ lets us calculate $\mathbb{E}[f(S)]$ from the known values of $c$, $f_\mathrm{min}$, and $f_\mathrm{max}$.
# If the estimation error of $P_1$ is $\delta P_1$, then $\mathbb{E}[\tilde{f}] \approx (P_1 - 1/2)/c$, so the estimation error is amplified to $\delta P_1/c$.
# Reducing $c$ increases this amplification, while increasing $c$ increases the approximation bias $\mathcal{O}(c^2)$.
# This tradeoff gives a convergence rate of $\mathcal{O}(M^{-2/3})$.
# Although this does not reach a full quadratic speedup, it still outperforms the classical MC rate of $\mathcal{O}(M^{-1/2})$.
# The implementation below uses lightweight maximum-likelihood amplitude estimation (MLAE), which does not require phase estimation.
#
# ### Applying the method to a call option
#
# The payoff of a call option, $f_C(S_T) = \max(0, S_T - K)$, is a piecewise-linear function with a kink at $K$.
# This implementation uses uniformly controlled $R_y$ rotations (UCR-Y) to apply the target amplitude angle at each grid point $S_i$:
#
# $$
# \alpha(S_i) = \frac{\pi}{4} - c + \frac{2c\,\max(S_i - K,\,0)}{S_\mathrm{max} - K} \tag{4}
# $$
#
# This directly applies the target amplitude angle.
# For $S < K$ the angle is the constant $\pi/4 - c$, and for $S \geq K$ it increases linearly, so this piecewise-linear structure can be encoded exactly.
#

# %% [markdown]
# ## Implementation with Qamomile
#
# ### Preparing the distribution and discretizing the option
#
# Consider a European call with strike $K=100$. We retain two possible prices
# at maturity: $80$ with risk-neutral probability $0.75$, and $120$ with
# probability $0.25$. Dividing the cash payoff $\max(S_T-K,0)$ by its maximum
# value $20$ gives the normalized payoff table $(0,1)$.
#
# One address qubit represents the two prices and one payoff qubit represents
# the normalized expected payoff. This two-qubit witness is enough to expose
# state loading, payoff multiplexing, adjoint composition, amplification,
# runtime parameter binding, result decoding, and resource estimation. Increasing
# the grid changes the coordinate vectors, not the programming model.

# %%
asset_prices = (80.0, 120.0)
probabilities = (0.75, 0.25)
strike = 100.0
risk_free_rate = 0.05
maturity = 1.0

cash_payoffs = tuple(max(price - strike, 0.0) for price in asset_prices)
payoff_scale = max(cash_payoffs)
normalized_payoffs = tuple(payoff / payoff_scale for payoff in cash_payoffs)
expected_cash_payoff = sum(
    probability * payoff
    for probability, payoff in zip(probabilities, cash_payoffs, strict=True)
)
exact_price = math.exp(-risk_free_rate * maturity) * expected_cash_payoff

assert probabilities == (0.75, 0.25)
assert normalized_payoffs == (0.0, 1.0)
assert expected_cash_payoff == 5.0

# %% [markdown]
# The standard-library coordinate transforms run in Rust. `amplitude_angles`
# normalizes the square-root probability amplitudes and preserves their phase.
# `uniformly_controlled_angles` converts the desired branch rotations to the
# cyclic Gray-walk coordinates used by the reusable payoff multiplexer.
#
# The resulting scalars are runtime `Param` values. The circuit can therefore be
# compiled once and rebound to another distribution or payoff table of the same
# shape without changing its typed body.

# %%
distribution = amplitude_angles(
    tuple(math.sqrt(probability) for probability in probabilities)
)
payoff_angles = uniformly_controlled_angles(
    tuple(2.0 * math.asin(math.sqrt(payoff)) for payoff in normalized_payoffs)
)

prepare_distribution = amplitude_encoding_from_angles(1)
encode_payoff = uniformly_controlled_ry_from_angles(1)

parameters = {
    "distribution_ry": distribution.ry,
    "distribution_rz": distribution.rz,
    "distribution_phase": distribution.phase,
    "payoff_angles": payoff_angles,
}

# %% [markdown]
# ### Implementing the payoff and distribution loading
#
# `option_state` consumes a two-component owned state. `split_at` separates the
# asset address from the payoff qubit, the distribution unitary prepares the
# address, and `concat` rejoins the components before the payoff multiplexer.
# Every width and runtime vector length is part of the function type.


# %%
@qm.qkernel
def option_state(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
    distribution_ry: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_rz: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_phase: qm.Param[qm.Real],
    payoff_angles: qm.Param[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare a two-point call-option payoff amplitude.

    Args:
        state (qm.Q[qm.Bits[2]]): Asset-address and payoff qubits.
        distribution_ry (qm.Param[qm.Vector[qm.Real, 1]]): Runtime magnitude
            coordinate for the risk-neutral distribution.
        distribution_rz (qm.Param[qm.Vector[qm.Real, 1]]): Runtime relative
            phase coordinate for the distribution.
        distribution_phase (qm.Param[qm.Real]): Runtime global phase.
        payoff_angles (qm.Param[qm.Vector[qm.Real, 2]]): Runtime Gray-walk
            coordinates for the normalized payoff.

    Returns:
        qm.Q[qm.Bits[2]]: State whose final component encodes expected payoff.
    """

    asset, payoff = qm.split_at(state, 1)
    asset = prepare_distribution(
        asset,
        distribution_ry,
        distribution_rz,
        distribution_phase,
    )
    return encode_payoff(qm.concat(asset, payoff), payoff_angles)


@qm.qkernel
def option_samples(
    distribution_ry: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_rz: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_phase: qm.Param[qm.Real],
    payoff_angles: qm.Param[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
    iterations: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Sample the option amplitude after fixed Grover iterations.

    Args:
        distribution_ry (qm.Param[qm.Vector[qm.Real, 1]]): Distribution
            magnitude coordinate.
        distribution_rz (qm.Param[qm.Vector[qm.Real, 1]]): Distribution phase
            coordinate.
        distribution_phase (qm.Param[qm.Real]): Distribution global phase.
        payoff_angles (qm.Param[qm.Vector[qm.Real, 2]]): Payoff multiplexer
            coordinates.
        iterations (qm.Const[qm.Int]): Compile-time Grover iteration count.

    Returns:
        qm.Shot[qm.Bits[2]]: Asset-address and payoff measurement.
    """

    state = option_state(
        qm.qubits(2),
        distribution_ry,
        distribution_rz,
        distribution_phase,
        payoff_angles,
    )
    for _ in range(iterations):
        state = qm.z(state, target=1)
        state = qm.adjoint(
            option_state,
            state,
            distribution_ry,
            distribution_rz,
            distribution_phase,
            payoff_angles,
        )
        state = qm.x(state, target=0)
        state = qm.x(state, target=1)
        state = qm.cz(state, control=0, target=1)
        state = qm.x(state, target=0)
        state = qm.x(state, target=1)
        state = option_state(
            state,
            distribution_ry,
            distribution_rz,
            distribution_phase,
            payoff_angles,
        )
    return qm.measure(state)


# %% [markdown]
# `iterations` is a `Const`: it changes the circuit structure and is fixed when
# specializing. The distribution and payoff coordinates are `Param`: they stay
# in the artifact invocation ABI. Mixing these two stages, duplicating `state`,
# or forgetting the successor returned by an operation is rejected before
# backend lowering.

# %% [markdown]
# ### Inspecting and estimating the amplified program
#
# The one-iteration program is ordinary composition of preparation, phase
# marking, an automatically derived adjoint, reflection, and preparation again.
# There is no finance-specific compiler instruction or backend path.

# %%
baseline_program = qm.specialize(option_samples, iterations=0)
amplified_program = qm.specialize(option_samples, iterations=1)
inspection = amplified_program.inspect()
study = qm.resources.logical(amplified_program)

assert "qcore.adjoint_call" in inspection
assert "quantum.option_pricing" not in inspection
assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
    12,
    12,
)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cx")
).numeric_enclosure() == (4, 4)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cz")
).numeric_enclosure() == (1, 1)
assert study.metric(qm.resources.LOGICAL_TWO_QUBIT_GATES).numeric_enclosure() == (
    7,
    7,
)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# These counts come from the same specialized call graph that will execute.
# The adjoint is not a copied inverse gate list, and the estimator does not use
# a separately maintained formula for option pricing.

# %% [markdown]
# ### Reproduce amplitude-estimation work scaling
#
# Section 14.2 of [](https://doi.org/10.48550/arXiv.2310.03011) contrasts
# $O(1/\epsilon^2)$ incoherent classical samples with $O(1/\epsilon)$ coherent
# walk-operator applications. This tutorial uses likelihood-based amplitude
# estimation rather than the textbook QPE circuit, but every additional
# Grover power still contributes one preparation inverse, two reflections, and
# one preparation. The resource algebra can verify the finite affine cost from
# one symbolic analysis; the rows below only substitute iteration counts.

# %%
amplitude_scaling_study = qm.resources.logical(option_samples)
amplitude_studies = tuple(
    amplitude_scaling_study.evaluate(iterations=iterations) for iterations in range(5)
)
amplitude_scaling = tuple(
    (
        resource_study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure()[
            0
        ],
        resource_study.metric(
            qm.resources.logical_operation_count("quantum.cx")
        ).numeric_enclosure()[0],
        resource_study.metric(
            qm.resources.logical_operation_count("quantum.cz")
        ).numeric_enclosure()[0],
        resource_study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure()[0],
    )
    for resource_study in amplitude_studies
)

for iterations, (rotations, cnot_gates, controlled_z, work) in enumerate(
    amplitude_scaling
):
    assert rotations == 4 + 8 * iterations
    assert cnot_gates == 2 + 2 * iterations
    assert controlled_z == iterations
    assert work == 9 + 20 * iterations

print("amplitude-estimation scaling (k: rotations, CX, CZ, work):", amplitude_scaling)

# %% [markdown]
# All four finite metrics are affine in the Grover power. Consequently a
# schedule with $M=\Theta(1/\epsilon)$ coherent walk uses has
# $\Theta(1/\epsilon)$ logical work for this fixed preparation, matching the
# survey's quantum side. This circuit study does not itself prove the
# statistical success bound or include classical likelihood-optimization cost.

# %% [markdown]
# ### Implementing quantum amplitude estimation
#
# We compile both members of the same program family for the Reference backend.
# The baseline estimates $a=\sin^2\theta$; one Grover iteration estimates
# $\sin^2(3\theta)$. For the exact value $a=1/4$, the latter is one, giving a
# sharp structural check of the amplification body.

# %%
reference = qm.Reference()
baseline_artifact = qm.compile(baseline_program, to=reference)
amplified_artifact = qm.compile(amplified_program, to=reference)

baseline = qm.run(
    baseline_artifact,
    on=reference,
    parameters=parameters,
    shots=128,
    seed=2,
).result
amplified = qm.run(
    amplified_artifact,
    on=reference,
    parameters=parameters,
    shots=4,
    seed=2,
).result

baseline_successes = sum(
    count for bits, count in baseline.counts.items() if bits[1] == 1
)
amplified_successes = sum(
    count for bits, count in amplified.counts.items() if bits[1] == 1
)
observations = (
    (0, baseline_successes, baseline.shots),
    (1, amplified_successes, amplified.shots),
)

assert baseline.shots == 128
assert amplified.counts == {(1, 1): 4}

# %% [markdown]
# ## Results
#
# ### Estimate and discount on the host
#
# Maximum-likelihood estimation is ordinary classical work. Keeping it in
# Python makes the quantum/classical boundary explicit and avoids encoding a
# numerical optimizer as compiler semantics. For observations $(h_k,N_k)$ we
# maximize
#
# $$
# \prod_k p_k(\theta)^{h_k}(1-p_k(\theta))^{N_k-h_k},\qquad
# p_k(\theta)=\sin^2((2k+1)\theta).
# $$


# %%
def negative_log_likelihood(theta: float) -> float:
    """Return the binomial negative log likelihood for all Grover powers.

    Args:
        theta (float): Candidate amplitude angle in the open first quadrant.

    Returns:
        float: Negative log likelihood for the measured observations.
    """

    total = 0.0
    for iterations, successes, shots in observations:
        probability = math.sin((2 * iterations + 1) * theta) ** 2
        probability = min(max(probability, 1e-12), 1.0 - 1e-12)
        total -= successes * math.log(probability)
        total -= (shots - successes) * math.log(1.0 - probability)
    return total


theta_candidates = tuple(index * math.pi / (2 * 4096) for index in range(1, 4096))
theta_hat = min(theta_candidates, key=negative_log_likelihood)
amplitude_hat = math.sin(theta_hat) ** 2
payoff_hat = payoff_scale * amplitude_hat
price_hat = math.exp(-risk_free_rate * maturity) * payoff_hat

sampling_error = 6.0 * math.sqrt(0.25 * 0.75 / baseline.shots)
assert abs(amplitude_hat - 0.25) <= sampling_error

print("measured Grover data:", observations)
print(f"estimated normalized payoff: {amplitude_hat:.6f}")
print(f"estimated option price:       {price_hat:.6f}")
print(f"exact discretized price:      {exact_price:.6f}")

# %% [markdown]
# The final multiplication by the cash-payoff scale and the discount factor
# $e^{-rT}$ stays outside the quantum program. A richer grid replaces the two
# runtime coordinate vectors and uses more address qubits, while the typed
# prepare-amplify-estimate-discount decomposition remains the same.

# %% [markdown]
# ### Restore the original 16-point log-normal option
#
# The compact witness above makes every circuit component easy to inspect. We
# now run the original reader-facing financial model as well: four address
# qubits hold a 16-point discretized log-normal terminal-price distribution,
# the strike is 1.9, and the payoff angle uses the original $c=0.25$ linearized
# construction. The same prepare, Grover-amplify, maximum-likelihood, and
# discount workflow applies without changing APIs.

# %%
FULL_ADDRESS_QUBITS = 4
FULL_TOTAL_QUBITS = 5
spot = 2.0
full_strike = 1.9
full_rate = 0.05
volatility = 0.4
full_maturity = 40.0 / 365.0
drift = (full_rate - 0.5 * volatility**2) * full_maturity
deviation = volatility * math.sqrt(full_maturity)
minimum_price = spot * math.exp(drift - 3.0 * deviation)
maximum_price = spot * math.exp(drift + 3.0 * deviation)
price_grid = np.linspace(minimum_price, maximum_price, 2**FULL_ADDRESS_QUBITS)
full_probabilities = np.exp(
    -((np.log(price_grid / spot) - drift) ** 2) / (2.0 * deviation**2)
) / (price_grid * deviation * math.sqrt(2.0 * math.pi))
full_probabilities /= full_probabilities.sum()
full_payoffs = np.maximum(price_grid - full_strike, 0.0)
full_payoff_scale = maximum_price - full_strike
scale_c = 0.25
target_amplitudes = (
    math.pi / 4.0 - scale_c + 2.0 * scale_c * full_payoffs / full_payoff_scale
)
full_exact_payoff = float(np.sum(full_probabilities * full_payoffs))
full_exact_price = math.exp(-full_rate * full_maturity) * full_exact_payoff

prepare_full_distribution = amplitude_encoding(
    tuple(math.sqrt(float(value)) for value in full_probabilities)
)
encode_full_payoff = uniformly_controlled_ry(
    tuple(2.0 * float(value) for value in target_amplitudes)
)


@qm.qkernel
def flip_target(target: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
    """Flip one target used by the multi-controlled zero reflection."""

    return qm.x(target)


@qm.qkernel
def full_option_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Prepare the original 16-point distribution and scaled payoff."""

    address, payoff = qm.split_at(state, 4)
    address = prepare_full_distribution(address)
    return encode_full_payoff(qm.concat(address, payoff))


@qm.qkernel
def full_option_samples(
    iterations: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the full option model after fixed Grover iterations."""

    state = full_option_state(qm.qubits(5))
    for _ in range(iterations):
        state = qm.z(state, target=4)
        state = qm.adjoint(full_option_state, state)
        for index in range(5):
            state = qm.x(state, target=index)
        controls, target = qm.split_at(state, 4)
        target = qm.h(target)
        controls, target = qm.control(flip_target, controls, target)
        target = qm.h(target)
        state = qm.concat(controls, target)
        for index in range(5):
            state = qm.x(state, target=index)
        state = full_option_state(state)
    return qm.measure(state)


full_shots = 32 if docs_test_mode else 256
full_observations: list[tuple[int, int, int]] = []
for power in (0, 1):
    full_program = qm.specialize(full_option_samples, iterations=power)
    full_artifact = qm.compile(full_program, to=reference)
    full_result = qm.run(
        full_artifact,
        on=reference,
        shots=full_shots,
        seed=20 + power,
    ).result
    successes = sum(
        count
        for bits, count in full_result.counts.items()
        if bits[FULL_TOTAL_QUBITS - 1] == 1
    )
    full_observations.append((power, successes, full_result.shots))


def full_negative_log_likelihood(theta: float) -> float:
    """Return the likelihood objective for the full option observations."""

    total = 0.0
    for power, successes, shots in full_observations:
        probability = math.sin((2 * power + 1) * theta) ** 2
        probability = min(max(probability, 1e-12), 1.0 - 1e-12)
        total -= successes * math.log(probability)
        total -= (shots - successes) * math.log(1.0 - probability)
    return total


full_theta_hat = min(theta_candidates, key=full_negative_log_likelihood)
full_probability_hat = math.sin(full_theta_hat) ** 2
scaled_payoff_hat = (full_probability_hat - 0.5) / scale_c
full_payoff_hat = full_payoff_scale * (scaled_payoff_hat + 1.0) / 2.0
full_price_hat = math.exp(-full_rate * full_maturity) * full_payoff_hat

print("16-point Grover data:", full_observations)
print(f"16-point estimated price: {full_price_hat:.6f}")
print(f"16-point reference price: {full_exact_price:.6f}")
assert abs(full_price_hat - full_exact_price) < (0.5 if docs_test_mode else 0.2)

# %% [markdown]
# ## Summary
#
# This article showed how to implement the option-pricing algorithm {cite:p}`10.22331/q-2020-07-06-291` with Qamomile.
# The key points are as follows.
#
# * We encoded a log-normal distribution into a quantum state with Mottönen encoding and used uniformly controlled rotations to encode the payoff of a call option into the payoff-qubit amplitude. Use Qamomile's `amplitude_encoding` to load the state.
# * We measured circuits with the Grover operator $\mathcal{Q}$ applied $k$ times for several values of $k$, then estimated the amplitude angle by maximum likelihood to calculate the expected payoff.
