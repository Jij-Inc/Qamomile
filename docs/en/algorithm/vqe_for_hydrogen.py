# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# tags: [algorithm, chemistry, variational]
# ---
#
# # VQE for the Hydrogen Molecule: Energy Estimation and Variational Optimization
#
# The variational quantum eigensolver (VQE) combines a parameterized quantum circuit with classical optimization to approximate the ground-state energy of a Hamiltonian.
# Here, we use the hydrogen molecule $(\mathrm{H}_2)$ as an example, construct its Hamiltonian with OpenFermion and PySCF, and estimate its energy using a Qamomile state-preparation kernel and `Estimator`.
# We first optimize at a single interatomic distance, then explore the energy curve as the distance changes.

# %%
# Install the latest Qamomile and chemistry extras through pip!
# # !pip install "qamomile[visualization]" openfermion pyscf openfermionpyscf

# %%
from __future__ import annotations

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import openfermion.chem as of_chem
import openfermion.transforms as of_trans
import openfermionpyscf as of_pyscf
from scipy.optimize import minimize

import qamomile as qm
from qamomile.measurement import Estimator

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"


# %% [markdown]
# ## Problem Settings
#
# ### Constructing the Hydrogen Molecule Hamiltonian
#
# We fix the positions of the nuclei and calculate the electronic state using the STO-3G basis.
# For $\mathrm{H}_2$ in STO-3G, each of the two spatial orbitals has two spin states, giving four spin orbitals.
# The Jordan–Wigner transformation represents the occupation of each spin orbital with one qubit, so this example requires four qubits without qubit reduction.\
# The steps are to calculate the molecular integrals, construct the fermionic operator, and apply the Jordan–Wigner transformation.\
# In `MolecularData`, `multiplicity=1` specifies a spin singlet, and `charge=0` specifies zero net charge.
# The $\mathrm{H}_2$ molecule is a two-electron system.
# We begin by fixing the interatomic distance at $0.977$ Å.\
# We also calculate the full configuration interaction (FCI) energy with PySCF and save it for comparison.
# FCI provides a reference solution under the chosen conditions, including the basis and electron number.
# It does not eliminate the error associated with the finite STO-3G basis.
# Throughout this example, distances are in angstroms (Å) and energies are in Hartree.

# %%
def make_hydrogen_molecule(bond_length: float):
    """Generate one STO-3G hydrogen molecule and its qubit Hamiltonian.

    Args:
        bond_length: Distance between the hydrogen atoms in angstroms.

    Returns:
        A pair containing the Jordan–Wigner operator and molecular data.
    """

    geometry = [["H", [0, 0, 0]], ["H", [0, 0, bond_length]]]
    molecule = of_chem.MolecularData(geometry, "sto-3g", 1, 0)
    molecule = of_pyscf.run_pyscf(molecule, run_scf=True, run_fci=True)
    fermionic = of_trans.get_fermion_operator(molecule.get_molecular_hamiltonian())
    return of_trans.jordan_wigner(fermionic), molecule


jordan_wigner_hamiltonian, molecule = make_hydrogen_molecule(0.977)
N_QUBITS = molecule.n_qubits
N_ELECTRONS = molecule.n_electrons

assert N_QUBITS == 4
assert N_ELECTRONS == 2
assert molecule.fci_energy is not None
print(f"FCI reference energy: {molecule.fci_energy:+.12f}")

# %% [markdown]
# ## Algorithm
#
# ### The Quantity Minimized by VQE
#
# Let $|\psi(\boldsymbol{\theta})\rangle$ be the trial state prepared by the quantum circuit and $H$ the qubit Hamiltonian. The quantity we want to minimize is
#
# $$
# E(\boldsymbol{\theta})
# =
# \langle\psi(\boldsymbol{\theta})|H|\psi(\boldsymbol{\theta})\rangle
# $$
#
# Here, $\boldsymbol{\theta}$ is the collection of angles supplied to the rotation gates.\
# The quantum computation prepares and measures the state, while the classical computation estimates the energy and updates the angles.
# Repeating these steps searches for parameters that yield a lower energy.
# In this example, we use `qm.Reference()`, Qamomile's built-in local simulator, as the execution endpoint.
# The true expectation value of a normalized state is no lower than the lowest eigenvalue of the Hamiltonian in the search space containing that state.
# However, estimates obtained from a finite number of measurements have statistical fluctuations.
# When comparing against the molecular FCI reference, we must also check conditions such as the electron number.

# %% [markdown]
# ## Implementation
#
# ### Converting to a Qamomile Hamiltonian
#
# We convert the OpenFermion `QubitOperator` into a Qamomile `Hamiltonian` representing the same operator.
# A qubit Hamiltonian can generally be written as
#
# $$
# H=c_0 I+\sum_{\ell}c_\ell P_\ell
# $$
#
# where $P_\ell$ is a product of Pauli operators, $c_\ell$ is its coefficient, and $c_0 I$ is a constant term proportional to the identity operator.
# The following `openfermion_to_qamomile` function preserves the qubit indices and coefficients of each term.
# We retain the constant term so that we can compare energies with FCI.
# We display `len(hamiltonian.terms)` to check the number of non-identity Pauli terms.
# The constant term is stored separately, so this count differs from the OpenFermion term count when the identity term is included.

# %%
def openfermion_to_qamomile(operator, num_qubits: int) -> qm.Hamiltonian:
    """Convert an OpenFermion QubitOperator into a Qamomile Hamiltonian.

    Args:
        operator: Jordan–Wigner-transformed OpenFermion operator.
        num_qubits: Width retained even when the highest mode is identity.

    Returns:
        The equivalent immutable Qamomile Hamiltonian.
    """

    paulis = {"X": qm.X, "Y": qm.Y, "Z": qm.Z}
    result = qm.Hamiltonian.zero(num_qubits=num_qubits)
    for factors, coefficient in operator.terms.items():
        assert abs(coefficient.imag) < 1e-12
        if not factors:
            result = result + coefficient.real
            continue
        word = qm.Hamiltonian.identity(num_qubits=num_qubits)
        for index, label in factors:
            word = word * paulis[label](index)
        result = result + coefficient.real * word
    return result


hamiltonian = openfermion_to_qamomile(
    jordan_wigner_hamiltonian,
    N_QUBITS,
)

assert hamiltonian.num_qubits == N_QUBITS
assert hamiltonian.is_hermitian
assert qm.Hamiltonian.from_portable(hamiltonian.to_portable()) == hamiltonian
print(f"Hamiltonian terms: {len(hamiltonian.terms)}")

# %% [markdown]
# ### Creating the VQE Ansatz
#
# An ansatz is a circuit that prepares trial states as its parameters vary.
# Here, we construct a hardware-efficient ansatz from RY gates, RZ gates, and CX gates between neighboring qubits.\
# We repeat “RY on all qubits -> RZ on all qubits -> a CX chain” four times, then add a final RY and RZ layer.
# Each rotation gate has an independent angle, giving
#
# $$
# (4+1)\times2\times4=40
# $$
#
# parameters.
# We declare `thetas` as `Param` so that the angles can change between executions while the circuit structure stays fixed.\
# The following `hydrogen_vqe_state` is a state-preparation kernel that returns a quantum register without measuring it.
# Later, we pass this definition to `Estimator`, which combines it with the required measurements.

# %%
REPETITIONS = 1 if docs_test_mode else 4
NUM_PARAMETERS = (REPETITIONS + 1) * 2 * N_QUBITS


@qm.qkernel
def hydrogen_vqe_state(
    thetas: qm.Param[qm.Vector[qm.Real, NUM_PARAMETERS]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Prepare the four-qubit hardware-efficient hydrogen trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, NUM_PARAMETERS]]): Runtime SU(2) angles.

    Returns:
        qm.Q[qm.Bits[4]]: Prepared variational state.
    """

    state = qm.qubits(4)
    for repetition in range(REPETITIONS):
        base = repetition * 8
        for target in range(4):
            state = qm.ry(state, thetas[base + target], target=target)
        for target in range(4):
            state = qm.rz(state, thetas[base + 4 + target], target=target)
        for control in range(3):
            state = qm.cx(state, control=control, target=control + 1)
    for target in range(4):
        state = qm.ry(state, thetas[8 * REPETITIONS + target], target=target)
    for target in range(4):
        state = qm.rz(state, thetas[8 * REPETITIONS + 4 + target], target=target)
    return state


# %% [markdown]
# :::{note} This Circuit Does Not Preserve Electron Number
#
# The circuit starts from $|0000\rangle$, and the RY and CX gates can prepare states with different occupation numbers.
# Reading `N_ELECTRONS = 2` or specifying `charge=0` in the molecular calculation does not restrict the variational circuit to two-electron states.
#
# In this example, FCI serves as a reference for the target two-electron molecule.
# To claim that the true expectation value for arbitrary parameters is no lower than this FCI value, we must either match the electron-number and spin conditions of the search space or verify that the FCI value equals the lowest energy over the full search space.
#
# A more rigorous implementation for finding the molecular ground state needs to consider an ansatz or constraints that preserve symmetries such as electron number.
# Starting from a Hartree–Fock state alone does not prevent subsequent gates from changing the electron number.
# :::
#
# ### Estimating Energy with `Estimator`
#
# We pass the state-preparation kernel and Hamiltonian to `Estimator`.
# It estimates $E(\boldsymbol{\theta})$ by measuring the expectation value of each Pauli term, multiplying by its coefficient, and summing the contributions.\
# The `grouping="qwc"` setting groups terms that can use the same measurement basis on each qubit.
# For example, $Z_0$ and $Z_0Z_1$ can be measured together, whereas $Z_0$ and $X_0$ use different bases.
# Even when terms commute, this method measures them separately if they require different measurement bases on any qubit.\
# The qubit positions in the state-preparation return value must correspond to the indices in `qm.X(i)`, `qm.Z(i)`, and similar operators.
# Whenever we change the measurement basis, we prepare a fresh state and measure it.

# %%
endpoint = qm.Reference()
estimator = Estimator(hydrogen_vqe_state, hamiltonian, grouping="qwc")
assert NUM_PARAMETERS == (16 if docs_test_mode else 40)


# %% [markdown]
# #### Total Shot Count and Standard Error
#
# The next function returns the energy estimate. The `shots` argument is the total number of shots across all measurement groups in one call to `estimate`. The default `allocation="uniform"` divides this total equally among the groups, giving one extra shot to each of the first groups if there is a remainder. The identity contribution is a constant and is added without measurement.
#
# Pauli terms in the same group share that group's measurement results. For example, if $Z_0$ and $Z_0Z_1$ are measured together for 1,024 shots, both expectations are estimated from the same 1,024 results. The group's shot count is not divided further among its terms.
#
# This example uses five measurement groups. The 5,120 total shots per objective evaluation give each group 1,024 shots, and the 40,960 total shots for the final measurement after optimization give each group 8,192 shots.
#
# To make results reproducible when run in the same environment and order, we advance the random seed for each call, using different pseudorandom streams during optimization and final measurement.
#
# The energy estimate is `estimate.value`, and its standard error is `estimate.standard_error`.
# The standard error describes measurement fluctuations under the assumption of independent shots.
# It does not include incomplete optimization, limited circuit expressiveness, basis-set approximation, or systematic errors on a physical device.
# With too few measurements per group, the standard error may not be estimable.\
# If you substantially reduce `SHOTS`, check that the measurement groups still receive enough samples.

# %%
SHOTS = 10 if docs_test_mode else 5120
FINAL_SHOTS = 10 if docs_test_mode else 40960
measurement_seeds = itertools.count(42)


def estimate_energy(values, estimator, *, shots=SHOTS):
    """Evaluate the prepared estimator with fresh random streams."""
    estimate = estimator.estimate(
        on=endpoint,
        parameters={"thetas": tuple(float(value) for value in values)},
        shots=shots,
        seed=next(measurement_seeds),
    )
    assert estimate.shots == shots
    assert np.isfinite(estimate.value)
    assert estimate.standard_error is not None
    assert np.isfinite(estimate.standard_error)
    return estimate


# %% [markdown]
# ## Result
#
# ### Running Classical Optimization and Reevaluating the Final Parameters
#
# We use SciPy's `minimize` with COBYLA to search for angles that reduce the measured energy estimate.
# COBYLA is a derivative-free optimization method and does not require finite-difference gradient calculations.
# However, being derivative-free does not guarantee convergence to a global optimum for an objective with shot noise.\
# For this method, `maxiter` specifies the **maximum number of objective evaluations**, rather than the maximum number of iterations.
# We allow up to 500 evaluations for each optimization.
# We choose the initial values from a uniform distribution with a fixed random seed.\
# The lowest energy observed during optimization can be biased by selecting a measurement that happened to fluctuate downward.
# Therefore, instead of reporting `optimization.fun` directly, we measure the returned `optimization.x` again with fresh shots.\
# We use `success` and `message` to check why the optimizer stopped.
# Even `success=True` does not mean that the result agrees with FCI or that a global optimum has been found.

# %%
cost_history = []


def cost_function(values) -> float:
    """Record the sampled energy used by the classical optimizer."""
    estimate = estimate_energy(values, estimator)
    cost_history.append(estimate.value)
    return estimate.value


rng = np.random.default_rng(42)
initial_parameters = rng.uniform(0.0, np.pi, NUM_PARAMETERS)
max_evaluations = NUM_PARAMETERS + 2 if docs_test_mode else 500
optimizer_options = {"maxiter": max_evaluations, "rhobeg": 0.5}
if docs_test_mode:
    optimizer_options["f_target"] = float("inf")

optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options=optimizer_options,
)
if docs_test_mode:
    assert optimization.nfev == 1
final_estimate = estimate_energy(optimization.x, estimator, shots=FINAL_SHOTS)
fci_reference = float(molecule.fci_energy)
print(
    f"VQE energy: {final_estimate.value:+.6f} +/- {final_estimate.standard_error:.6f} (SE)"
)
print(f"FCI energy: {fci_reference:+.6f}")
print(f"Final measurement shots: {final_estimate.shots}")
assert len(optimization.x) == NUM_PARAMETERS
assert len(cost_history) == optimization.nfev
print("Optimizer success:", optimization.success)
print("Optimizer message:", optimization.message)
print("Objective evaluations:", optimization.nfev)
print(f"Difference from FCI: {final_estimate.value - fci_reference:+.6f} Hartree")

# %% [markdown]
# ### Inspecting the Optimization History
#
# We plot the progress of the energy search with objective evaluations ordered along the horizontal axis.
# The final point is an independent measurement at the selected parameters, not an additional optimization step.\
# The error bars represent one standard error (1 SE).
# They are not intervals guaranteed to contain the molecule's true ground-state energy.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
evaluation_numbers = np.arange(1, len(cost_history) + 1)
ax.plot(evaluation_numbers, cost_history, label="Sampled VQE objective")
ax.errorbar(
    [len(cost_history) + 1],
    [final_estimate.value],
    yerr=[final_estimate.standard_error],
    fmt="o",
    capsize=4,
    label="Fresh final estimate (1 SE)",
)
ax.axhline(fci_reference, linestyle="dashed", label="FCI reference (2 electrons)")
ax.set_xlabel("Objective evaluation / final remeasurement")
ax.set_ylabel("Energy (Hartree)")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### When a Difference from FCI Remains
#
# If the difference between the final estimate and FCI is much larger than the standard error, measurement fluctuations alone do not explain it.
# In that case, check factors such as the evaluation budget, initial values, circuit structure, and electron-number and spin conditions.
# It is difficult to identify the dominant factor from a single run.\
# Increasing the number of shots in the final measurement gives a more precise energy estimate at the same parameters.
# It does not improve the selected state itself to a lower-energy state.\
# We must also distinguish the true expectation value, to which the variational principle applies, from a finite-shot estimate.
# Even relative to the ground-state energy in the same search space, an estimate can fall below it because of statistical fluctuations.
# In this example, the electron-number conditions of FCI and the variational circuit do not automatically match either.
#
# ### Exploring the Energy Curve as the Interatomic Distance Changes
#
# For each interatomic distance $R$, we recalculate the Hamiltonian and FCI reference, then optimize using an `Estimator` for that Hamiltonian.
# We use 15 points from $0.2$ Å to $1.5$ Å.\
# At the first point, we use the parameters obtained in the previous section. At each subsequent point, we use the parameters from the preceding distance as the initial values.
# This approach is called a warm start, but it can also carry over incomplete optimization from the previous point.
# The result may therefore depend on the order in which we explore the distances.\
# We explicitly set `grouping="qwc"` again to use the same grouping as before.
# At each point, we reevaluate the final energy with fresh shots.\
# This calculation compares energies at fixed distances.
# It is not a geometry optimization that treats the interatomic distance as a continuous optimization variable.
# The resulting curve reflects the finite basis and limited evaluation budget.

# %%
n_points = 1 if docs_test_mode else 15
bond_lengths = np.linspace(0.2, 1.5, n_points)
vqe_energies = []
vqe_errors = []
fci_energies = []
optimization_success = []
objective_evaluations = []
starting_parameters = optimization.x.copy()

for bond_length in bond_lengths:
    transformed, bond_molecule = make_hydrogen_molecule(float(bond_length))
    bond_hamiltonian = openfermion_to_qamomile(transformed, bond_molecule.n_qubits)
    bond_estimator = Estimator(hydrogen_vqe_state, bond_hamiltonian, grouping="qwc")
    bond_fci = bond_molecule.fci_energy
    assert bond_fci is not None
    bond_result = minimize(
        lambda values: estimate_energy(values, bond_estimator).value,
        starting_parameters,
        method="COBYLA",
        options=optimizer_options,
    )
    starting_parameters = bond_result.x.copy()
    optimization_success.append(bool(bond_result.success))
    objective_evaluations.append(int(bond_result.nfev))
    measured = estimate_energy(
        starting_parameters,
        bond_estimator,
        shots=FINAL_SHOTS,
    )
    vqe_energies.append(measured.value)
    vqe_errors.append(measured.standard_error)
    fci_energies.append(float(bond_fci))
    print(
        f"distance={bond_length:.3f}, VQE={measured.value:+.6f} "
        f"+/- {measured.standard_error:.6f} (SE), FCI={bond_fci:+.6f}, "
        f"success={bond_result.success}, nfev={bond_result.nfev}"
    )

assert len(vqe_energies) == len(vqe_errors) == len(fci_energies) == n_points
print("Optimizer successes:", sum(optimization_success), "/", n_points)

# %% [markdown]
# Let's visualize the results.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(
    bond_lengths,
    vqe_energies,
    yerr=vqe_errors,
    fmt="-o",
    capsize=3,
    label="Sampled VQE (1 SE)",
)
ax.plot(bond_lengths, fci_energies, "--", label="FCI reference (2 electrons)")
ax.set_xlabel("H-H distance (angstrom)")
ax.set_ylabel("Energy (Hartree)")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - We constructed the Hamiltonian with OpenFermion and PySCF and passed the four-qubit operator obtained by the Jordan–Wigner transformation to Qamomile.
# - We combined a parameterized state-preparation kernel with `Estimator` and optimized finite-shot energy estimates using COBYLA.
# - When assessing the results, we need to distinguish the basis and electron-number conditions used for the FCI comparison, the optimizer's stopping conditions, and the standard error of the final measurement.
# - We reported the final results as energy estimates with error bars from fresh measurements. Aggregation across shots takes place outside the qkernel.
