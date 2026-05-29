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

# %%
# Install the latest Qamomile through pip!
# (Google Colab) Pick the line that matches your chosen Transpiler tab
# below and remove the leading "# " from it to run.
# # !pip install qamomile openfermion pyscf openfermionpyscf                  # Qiskit (default)
# # !pip install "qamomile[quri_parts]" openfermion pyscf openfermionpyscf    # QURI Parts
# # !pip install "qamomile[cudaq-cu12]" openfermion pyscf openfermionpyscf    # CUDA-Q on a CUDA 12.x toolchain (use qamomile[cudaq-cu13] on CUDA 13.x). Linux / macOS-arm64 / WSL2 only.

# %% [markdown]
# This article uses Qiskit by default. Qamomile transpiles the same
# `@qkernel` to multiple quantum SDKs, so you can follow it with another
# SDK by swapping the import shown below — the rest of the article code
# is identical regardless of the SDK you pick. On Colab, uncomment the
# matching `pip install` line in the cell above first.
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# from qamomile.qiskit import QiskitTranspiler
#
# transpiler = QiskitTranspiler()
# ```
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# from qamomile.quri_parts import QuriPartsTranspiler
#
# transpiler = QuriPartsTranspiler()
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# Use `qamomile[cudaq-cu12]` for a CUDA 12.x toolchain or
# `qamomile[cudaq-cu13]` for a CUDA 13.x toolchain — pick the one that
# matches your installed CUDA Toolkit. CUDA-Q is supported on Linux,
# macOS arm64, and Windows-via-WSL2 only.
#
# ```python
# from qamomile.cudaq import CudaqTranspiler
#
# transpiler = CudaqTranspiler()
# ```
# :::
# ::::

# %%
# Transpiler — by default this article uses Qiskit. If you picked a
# different tab above (QURI Parts / CUDA-Q), copy the two lines from
# that tab into this cell in place of the two below, and make sure the
# matching pip install line further up has been uncommented.
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import openfermion.chem as of_chem
import openfermion.transforms as of_trans
import openfermionpyscf as of_pyscf
from scipy.optimize import minimize

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import cx_entangling_layer, ry_layer, rz_layer

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
# H2 in the STO-3G basis has 2 spatial orbitals -> 4 spin orbitals -> 4 qubits,
# and is a 2-electron molecule.
assert n_qubit == 4
assert n_electron == 2
fermionic_hamiltonian = of_trans.get_fermion_operator(
    molecule.get_molecular_hamiltonian()
)
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
assert hamiltonian.num_qubits == n_qubit


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
# ## Running VQE
#
# In this section, we transpile the VQE kernel to an executable object using the `transpiler` constructed at the top of this article. The default executor runs the executable and returns the expectation value that the qkernel computes via `expval`; only the optimisation loop is left for us to implement. The Transpiler / Executor tab block already in this article controls which SDK actually carries out that emit + execute pair — swap tabs to switch SDKs.

# %%
reps = 4

executable = transpiler.transpile(
    vqe_ansatz,
    bindings={"n": n_qubit, "reps": reps, "H": hamiltonian},
    parameters=["thetas"],
)

# Transpiled quantum circuit (SDK-native object; `.draw("mpl")` is
# Qiskit-specific, so we just print the type for SDK portability — use
# the SDK's own drawing API for an actual diagram).
print(type(executable.quantum_circuit).__name__)

# %% [markdown]
# We need an executor that can compute expectation values (the
# `expval` inside `vqe_ansatz` returns a parametric `Float` that the
# optimisation loop drives to its minimum). How to wire that depends
# on the SDK you picked at the top — copy the matching snippet from
# the tab block below into the executor cell further down.
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# from qiskit_aer.primitives import EstimatorV2
# from qamomile.qiskit.transpiler import QiskitExecutor
#
# executor = QiskitExecutor(estimator=EstimatorV2())
# ```
#
# `EstimatorV2` is Qiskit's current-generation primitive; it computes
# expectation values in one shot rather than going through sampling.
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# from qamomile.quri_parts import QuriPartsExecutor
#
# # QuriPartsExecutor lazily constructs a parametric estimator backed
# # by qulacs when one isn't supplied. That default is fine for the
# # VQE loop below.
# executor = QuriPartsExecutor()
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# ```python
# from qamomile.cudaq import CudaqExecutor
#
# # CudaqExecutor computes expectation values via cudaq.observe under
# # the hood — no explicit estimator wiring is needed.
# executor = CudaqExecutor()
# ```
# :::
# ::::

# %%
# Executor — by default this article uses Qiskit's EstimatorV2 for
# expectation values. If you picked a different tab above, copy that
# tab's snippet over the lines below (and make sure the matching pip
# install line at the top of this article is uncommented).
from qiskit_aer.primitives import EstimatorV2

from qamomile.qiskit.transpiler import QiskitExecutor

executor = QiskitExecutor(estimator=EstimatorV2())

# %%
cost_history = []


def cost_fn(param_values):
    job = executable.run(executor, bindings={"thetas": list(param_values)})
    return job.result()


def cost_callback(param_values):
    cost_history.append(cost_fn(param_values))


num_params = len(executable.parameter_names)
rng = np.random.default_rng(42)
initial_params = rng.uniform(0, np.pi, num_params)
# Each rep emits one RY layer (n parameters) and one RZ layer (n parameters);
# the final pre-CX layer adds one more RY + RZ pair. So the parameter count is
# (reps + 1) * 2 * n_qubit.
assert num_params == (reps + 1) * 2 * n_qubit
assert initial_params.shape == (num_params,)

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
# Variational principle: any trial energy is an upper bound on the FCI
# ground-state energy, regardless of how short the BFGS budget is.
assert result.fun >= molecule.fci_energy - 1e-9
assert len(result.x) == num_params

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
assert bond_lengths.shape == (n_points,)
energies = []
for bond_length in bond_lengths:
    hamiltonian, fci_energy = hydrogen_molecule(bond_length)
    # H2 remains a 4-qubit problem regardless of bond length.
    assert hamiltonian.num_qubits == 4

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
    # Variational principle holds at every bond length.
    assert result.fun >= fci_energy - 1e-9

    print("distance: ", bond_length, "energy: ", result.fun, "fci_energy: ", fci_energy)

assert len(energies) == n_points

# %%
plt.plot(bond_lengths, energies, "-o")
plt.xlabel("Distance")
plt.ylabel("Energy")
plt.show()
