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
# ---
# tags: [algorithm, chemistry, variational]
# ---
#
# # Variational Quantum Eigensolver (VQE) for the Hydrogen Molecule
#
# This tutorial demonstrates how to implement the Variational Quantum Eigensolver (VQE) algorithm to find the ground state energy of the hydrogen molecule (H₂). We use [OpenFermion](https://quantumai.google/openfermion) for generating molecular Hamiltonians.
#
# The workflow is as follows:
# 1. Convert the molecular Hamiltonian to qubit operators
# 2. Create a parametrized quantum circuit (ansatz)
# 3. Implement VQE optimization
# 4. Analyze the energy landscape across different atomic distances
#
# We show how to solve quantum chemistry problems using quantum computing, focusing on finding the minimum-energy structure of the H₂ molecule.

# %% [markdown]
# ## Installation
#
# ```bash
# pip install "qamomile[qiskit,visualization]" openfermion pyscf openfermionpyscf
# ```

# %%
# Install the latest Qamomile and chemistry extras through pip!
# # !pip install "qamomile[qiskit,visualization]" openfermion pyscf openfermionpyscf

# %%
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
from qamomile.qiskit import Aer

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Creating the Hamiltonian of the Hydrogen Molecule
#
# H₂ in the minimal STO-3G basis has two spatial orbitals, hence four spin
# orbitals and four qubits after Jordan–Wigner encoding. PySCF supplies both the
# molecular integrals and the full-configuration-interaction (FCI) reference
# energy used to check the variational result.


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
# ## Converting to a Qamomile Hamiltonian
#
# In this section, we convert the OpenFermion Hamiltonian to the Qamomile format. After applying the Jordan–Wigner transformation to convert fermionic operators to qubit operators, we use custom conversion functions to create a Hamiltonian representation compatible with Qamomile.


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
# ## Creating the VQE Ansatz
#
# The hardware-efficient SU(2) ansatz matches the original tutorial: four
# repetitions of an RY layer, an RZ layer, and a linear CX chain, followed by a
# final RY/RZ pair. For four qubits this gives
# $(4 + 1)\times 2\times 4=40$ runtime parameters. One state-returning program
# is reused by wrappers that explicitly select a measurement basis and
# return one classical sample per execution.

# %%
REPETITIONS = 4
NUM_PARAMETERS = (REPETITIONS + 1) * 2 * N_QUBITS


@qm.qkernel
def hydrogen_vqe_state(
    thetas: qm.Param[qm.Vector[qm.Real, 40]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Prepare the four-qubit hardware-efficient hydrogen trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, 40]]): Runtime SU(2) angles.

    Returns:
        qm.Q[qm.Bits[4]]: Prepared variational state.
    """

    state = qm.qubits(4)
    for repetition in range(4):
        base = repetition * 8
        for target in range(4):
            state = qm.ry(state, thetas[base + target], target=target)
        for target in range(4):
            state = qm.rz(state, thetas[base + 4 + target], target=target)
        for control in range(3):
            state = qm.cx(state, control=control, target=control + 1)
    for target in range(4):
        state = qm.ry(state, thetas[32 + target], target=target)
    for target in range(4):
        state = qm.rz(state, thetas[36 + target], target=target)
    return state


# %% [markdown]
# ## Prepare the Energy Estimator
#
# `Estimator` accepts the state-preparation definition and the Hamiltonian. Its
# constructor validates the request without generating measurement circuits.
# `estimate` selects a compatible backend estimator or builds measured programs.
# Qubit-wise-commuting (QWC) terms share a local measurement basis.
# The returned quantum-bit positions identify the Hamiltonian's qubit indices.
# Each generated execution entry returns classical measurement results; the
# quantum state never leaves an execution. Runtime angles reuse compiled programs.
#
# %%
endpoint = Aer()
estimator = Estimator(hydrogen_vqe_state, hamiltonian, grouping="qwc")
assert NUM_PARAMETERS == 40


# %% [markdown]
# ## Estimate Energy from Finite Shots
#
# `estimate` schedules the measurements and combines their statistics. `shots`
# is the **total budget**, distributed uniformly across measurement groups.
# Within-group covariance is retained; different groups use separate executions
# and seed streams. The identity term needs no device shots.
#
# Standard errors quantify sampling uncertainty under independent, identically
# distributed shots, not hardware bias or optimization error. One shot cannot
# determine a sample variance; its standard error would be `None`.

# %%
SHOTS = 640 if docs_test_mode else 5120
FINAL_SHOTS = 2560 if docs_test_mode else 40960
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
# ## Optimize and Measure the Selected Parameters Again
#
# COBYLA receives the sampled energy and does not require a finite-difference
# gradient. The objective is noisy. We therefore use fresh, higher-shot
# measurements for the final reported energy, rather than reporting the smallest
# noisy objective encountered by the optimizer. The reader-facing budget is
# 200 objective evaluations per geometry; the test budget checks execution only.
# Error bars quantify sampling uncertainty at the selected parameters. They do
# not include incomplete optimization or ansatz error, so they need not cover FCI.
#
# The true variational expectation is bounded below by the ground-state energy.
# A finite-shot estimate can fall below that reference. We display its estimated
# standard error instead of asserting an exact variational bound on each sample
# mean. The error bars show one standard error, not a guaranteed confidence bound.

# %%
cost_history = []


def cost_function(values) -> float:
    """Record the sampled energy used by the classical optimizer."""
    estimate = estimate_energy(values, estimator)
    cost_history.append(estimate.value)
    return estimate.value


rng = np.random.default_rng(42)
initial_parameters = rng.uniform(0.0, np.pi, NUM_PARAMETERS)
max_iterations = 42 if docs_test_mode else 200
optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": max_iterations, "rhobeg": 0.5},
)
final_estimate = estimate_energy(optimization.x, estimator, shots=FINAL_SHOTS)
fci_reference = float(molecule.fci_energy)
print(
    f"VQE energy: {final_estimate.value:+.6f} +/- {final_estimate.standard_error:.6f} (SE)"
)
print(f"FCI energy: {fci_reference:+.6f}")
print(f"Final measurement shots: {final_estimate.shots}")
assert len(optimization.x) == NUM_PARAMETERS
assert len(cost_history) == optimization.nfev

# %%
plt.plot(cost_history, label="Sampled VQE objective")
plt.errorbar(
    [len(cost_history) - 1],
    [final_estimate.value],
    yerr=[final_estimate.standard_error],
    fmt="o",
    capsize=4,
    label="Fresh final estimate (1 SE)",
)
plt.axhline(fci_reference, linestyle="dashed", color="black", label="FCI reference")
plt.xlabel("Energy evaluation")
plt.ylabel("Energy (Hartree)")
plt.legend()
plt.show()

# %% [markdown]
# ## Changing the Distance Between Atoms
#
# Recompute the Hamiltonian and create an estimator at each bond length. Each
# estimator reuses its compiled measurement programs throughout optimization.
# Each optimized point is evaluated again with fresh shots. Documentation tests
# use one geometry; the reader-facing notebook uses 15 bond lengths.

# %%
n_points = 1 if docs_test_mode else 15
bond_lengths = np.linspace(0.2, 1.5, n_points)
vqe_energies = []
vqe_errors = []
fci_energies = []
starting_parameters = optimization.x

for bond_length in bond_lengths:
    transformed, bond_molecule = make_hydrogen_molecule(float(bond_length))
    bond_hamiltonian = openfermion_to_qamomile(transformed, bond_molecule.n_qubits)
    bond_estimator = Estimator(hydrogen_vqe_state, bond_hamiltonian)
    bond_fci = bond_molecule.fci_energy
    assert bond_fci is not None
    bond_result = minimize(
        lambda values: estimate_energy(values, bond_estimator).value,
        starting_parameters,
        method="COBYLA",
        options={"maxiter": max_iterations, "rhobeg": 0.5},
    )
    starting_parameters = bond_result.x
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
        f"+/- {measured.standard_error:.6f} (SE), FCI={bond_fci:+.6f}"
    )

assert len(vqe_energies) == len(vqe_errors) == len(fci_energies) == n_points

# %%
plt.errorbar(
    bond_lengths,
    vqe_energies,
    yerr=vqe_errors,
    fmt="-o",
    capsize=3,
    label="Sampled VQE (1 SE)",
)
plt.plot(bond_lengths, fci_energies, "--", label="FCI reference")
plt.xlabel("H–H distance (Å)")
plt.ylabel("Energy (Hartree)")
plt.legend()
plt.show()

# %% [markdown]
# ## Summary
#
# - OpenFermion/PySCF provides the four-qubit Hamiltonian and FCI reference.
# - A state-preparation qkernel is composed with measurement circuits by `Estimator`.
# - Ordinary compiled sampling programs are reused across parameter updates.
# - Classical parity scores, within-group covariance, and independent-group
#   uncertainty are computed by `qamomile.statistics`.
# - Fresh final measurements report energy estimates with sampling error bars;
#   cross-shot reduction stays outside the qkernel language.
