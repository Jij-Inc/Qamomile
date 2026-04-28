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
# title: VQE for the Hydrogen Molecule
# tags: [vqe, variational, chemistry, ground-state, openfermion, intermediate]
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

# %%
# Required packages can be installed with the following command
# # !pip install openfermion pyscf openfermionpyscf

# %%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import openfermion.chem as of_chem
import openfermion.transforms as of_trans
import openfermionpyscf as of_pyscf
from qiskit_aer.primitives import EstimatorV2
from scipy.optimize import minimize

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import cx_entangling_layer, ry_layer, rz_layer
from qamomile.qiskit import QiskitTranspiler
from qamomile.qiskit.transpiler import QiskitExecutor

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Creating the Hamiltonian of the Hydrogen Molecule

# %%
basis = "sto-3g"
multiplicity = 1
charge = 0
distance = 0.977
geometry = [["H", [0, 0, 0]], ["H", [0, 0, distance]]]
description = "tmp"
molecule = of_chem.MolecularData(geometry, basis, multiplicity, charge, description)
molecule = of_pyscf.run_pyscf(molecule, run_scf=True, run_fci=True)
n_qubit = molecule.n_qubits
n_electron = molecule.n_electrons
fermionic_hamiltonian = of_trans.get_fermion_operator(molecule.get_molecular_hamiltonian())
jw_hamiltonian = of_trans.jordan_wigner(fermionic_hamiltonian)


# %% [markdown]
# ## Converting to a Qamomile Hamiltonian
#
# In this section, we convert the OpenFermion Hamiltonian to the Qamomile format. After applying the Jordan–Wigner transformation to convert fermionic operators to qubit operators, we use custom conversion functions to create a Hamiltonian representation compatible with Qamomile.

# %%
def operator_to_qamomile(operators: tuple[tuple[int, str], ...]) -> qm_o.Hamiltonian:
    pauli = {"X": qm_o.X, "Y": qm_o.Y, "Z": qm_o.Z}
    H = qm_o.Hamiltonian()
    H.constant = 1.0
    for ope in operators:
        H *= pauli[ope[1]](ope[0])
    return H

def openfermion_to_qamomile(of_h) -> qm_o.Hamiltonian:
    H = qm_o.Hamiltonian()
    for k, v in of_h.terms.items():
        if len(k) == 0:
            H.constant += v
        else:
            H += operator_to_qamomile(k) * v
    return H

hamiltonian = openfermion_to_qamomile(jw_hamiltonian)


# %% [markdown]
# ## Creating the VQE Ansatz
#
# In this section, we create an EfficientSU2 ansatz for the VQE algorithm using the `@qkernel` decorator. An ansatz is a parametrized quantum circuit that prepares a trial wavefunction. We build it by combining `ry_layer`, `rz_layer`, and a linear CX entangling layer, and finally compute the expectation value of the Hamiltonian using `expval`.

# %%
@qmc.qkernel
def vqe_ansatz(
    n: qmc.UInt,
    reps: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
    H: qmc.Observable,
) -> qmc.Float:
    q = qmc.qubit_array(n, name="q")
    for r in qmc.range(reps):
        base = r * 2 * n
        q = ry_layer(q, thetas, base)
        q = rz_layer(q, thetas, base + n)
        q = cx_entangling_layer(q)
    # Final rotation layer
    final_base = reps * 2 * n
    q = ry_layer(q, thetas, final_base)
    q = rz_layer(q, thetas, final_base + n)
    return qmc.expval(q, H)


# %% [markdown]
# ## Running VQE with Qiskit
#
# In this section, we transpile the VQE kernel to an executable object using `QiskitTranspiler`. The default executor runs this object and returns the expectation value, which the defined qkernel computes using `expval`. Thus, the user only needs to implement the optimisation loop.

# %%
transpiler = QiskitTranspiler()
reps = 4

executable = transpiler.transpile(
    vqe_ansatz,
    bindings={"n": n_qubit, "reps": reps, "H": hamiltonian},
    parameters=["thetas"],
)

# Transpiled quantum circuit
executable.quantum_circuit.draw("mpl")

# %%
cost_history = []
executor = QiskitExecutor(estimator=EstimatorV2())


def cost_fn(param_values):
    job = executable.run(executor, bindings={"thetas": list(param_values)})
    return job.result()


def cost_callback(param_values):
    cost_history.append(cost_fn(param_values))


num_params = len(executable.parameter_names)
rng = np.random.default_rng(42)
initial_params = rng.uniform(0, np.pi, num_params)

# Run VQE optimization
maxiter = 1 if docs_test_mode else 50
warnings.filterwarnings("ignore", message="Maximum number of iterations")
result = minimize(
    cost_fn,
    initial_params,
    method="BFGS",
    options={"disp": True, "maxiter": maxiter, "gtol": 1e-6},
    callback=cost_callback,
)
print(result)

# %%
plt.plot(cost_history)
plt.plot(
    range(len(cost_history)),
    [molecule.fci_energy] * len(cost_history),
    linestyle="dashed",
    color="black",
    label="Exact Solution",
)
plt.legend()
plt.show()


# %% [markdown]
# ## Changing the Distance Between Atoms

# %%
def hydrogen_molecule(bond_length):
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    geometry = [["H", [0, 0, 0]], ["H", [0, 0, bond_length]]]
    description = "tmp"
    molecule = of_chem.MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = of_pyscf.run_pyscf(molecule, run_scf=True, run_fci=True)
    fermionic_hamiltonian = of_trans.get_fermion_operator(
        molecule.get_molecular_hamiltonian()
    )
    jw_hamiltonian = of_trans.jordan_wigner(fermionic_hamiltonian)
    return openfermion_to_qamomile(jw_hamiltonian), molecule.fci_energy

n_points = 3 if docs_test_mode else 15
bond_lengths = np.linspace(0.2, 1.5, n_points)
energies = []
for bond_length in bond_lengths:
    hamiltonian, fci_energy = hydrogen_molecule(bond_length)

    executable = transpiler.transpile(
        vqe_ansatz,
        bindings={"n": hamiltonian.num_qubits, "reps": reps, "H": hamiltonian},
        parameters=["thetas"],
    )

    num_params = len(executable.parameter_names)
    initial_params = rng.uniform(0, np.pi, num_params)
    result = minimize(
        cost_fn,
        initial_params,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": 1e-6},
    )

    energies.append(result.fun)

    print("distance: ", bond_length, "energy: ", result.fun, "fci_energy: ", fci_energy)

# %%
plt.plot(bond_lengths, energies, "-o")
plt.xlabel("Distance")
plt.ylabel("Energy")
plt.show()
