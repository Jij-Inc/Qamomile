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
# tags: [algorithm, chemistry, variational]
# ---
#
# # Calculating the Hydrogen Molecule Ground-State Energy with ADAPT-VQE
#
# In this tutorial, we implement ADAPT-VQE with Qamomile to calculate the ground-state energy curve of $\mathrm{H}_2$ in the STO-3G basis at eight bond lengths and compare it with full configuration interaction (FCI).
# The variational quantum eigensolver (VQE) depends on the wavefunction ansatz chosen in advance. ADAPT-VQE, proposed by {cite:t}`grimsley2019`, instead constructs an ansatz by adding operators one at a time according to the molecule being studied. This can reduce the number of parameters and the circuit depth.

# %% [markdown]
# ## Installation
#
# This example uses Qamomile with Qiskit and Qiskit Aer for exact expectation values, OpenFermion for fermionic operators, and PySCF with OpenFermion-PySCF for molecular calculations. NumPy, SciPy, and Matplotlib provide numerical arrays, optimization, and plotting; they are installed with the packages below.
# In Google Colab, uncomment the following line to install the dependencies.

# %%
# # !pip install "qamomile[qiskit]" openfermion pyscf openfermionpyscf

# %%
from __future__ import annotations

from itertools import combinations, product
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from openfermion import (
    FermionOperator,
    commutator,
    hermitian_conjugated,
    jordan_wigner,
    normal_ordered,
)
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator
from openfermionpyscf import run_pyscf
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2
from scipy.optimize import minimize

import qamomile as qm
import qamomile.observable as qmo
from qamomile.qiskit import Qiskit, to_circuit

# %% [markdown]
# ## Background
#
# ### Quantum Chemistry Calculations and VQE
#
# When calculating molecular states, the space needed to represent the electronic state grows exponentially with the number of orbitals.
# This limits quantum chemistry calculations on classical computers, and quantum computing is expected to overcome these limitations.
# Achieving this requires efficient quantum algorithms as well as enough qubits with low gate and readout error rates and coherence times long enough to execute the circuit.
# VQE performs state preparation and measurements on a quantum computer, while updating parameters classically.
# Although it requires many measurements, its shallow circuits make it suitable for execution on NISQ devices.
#
# ### Previous Work: Adaptive Coupled Cluster Theory (`@CC`)
#
# Before explaining ADAPT-VQE, let us briefly introduce an important earlier classical method proposed by {cite:t}`lyakh2010`.
# To calculate a molecule's electronic state, we start from the state $\vert \Phi_\mathrm{HF} \rangle$ obtained using the Hartree-Fock method.
# Coupled cluster (CC) theory applies the following operator to this state:
#
# ```{math}
# :label: eq-adapt-cc
# \vert \Psi_\mathrm{CC} \rangle
# = e^T \vert \Phi_\mathrm{HF} \rangle
# ```
#
# Here,
#
# ```{math}
# :label: eq-adapt-cluster
# T
# = T_1 + T_2 + \cdots
# = \sum_{i, a} t_{ia} a_a^\dagger a_i + \frac{1}{4} \sum_{i, j, a, b} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i + \cdots
# ```
#
# is the excitation operator, where $T_1$ represents single-electron excitations, $T_2$ represents two-electron excitations, and so on.
# The operators $a_p^\dagger, a_p$ create and annihilate an electron in orbital $p$, respectively.
# Coupled cluster theory can thus be viewed as incorporating excitation operators through an exponential.
# The method is called CCS when only single excitations are included, CCSD when excitations up to doubles are included, CCSDT when excitations up to triples are included, and so on.
# Conventional CC methods improve accuracy by increasing the excitation rank treated independently, following the sequence CCS -> CCSD -> CCSDT.
# However, this approach includes every term at each rank, even those that contribute little, so it requires handling a large number of amplitudes.
# The `@CC` method instead adds only excitations considered important.
# If $S$ denotes the set of selected excitations, the calculation can be defined by
#
# ```{math}
# :label: eq-adapt-selected-cluster
# T_S
# = \sum_{\mu \in S} t_\mu \tau_\mu, \quad
# \vert \Psi_S \rangle
# = e^{T_S} \vert \Phi_\mathrm{HF} \rangle
# ```
#
# This specifies the model to be calculated.
# For example, one could use
#
# ```{math}
# :label: eq-adapt-selected-ranks
# T_S
# = T_1 + T_2 + \sum_{\mu \in S_3} t_\mu \tau_\mu + \sum_{\nu \in S_4} t_\nu \tau_\nu
# ```
#
# as the cluster operator.
# In this case, both the set $S$, which determines which amplitudes are included, and the values of the selected amplitudes $t_\mu$ must be adjusted.
# The `@CC` method uses an importance function to select the cluster amplitudes needed to describe the target electronic state.
# Its aim is to construct a computational model suited to the problem, instead of including all excitation operators uniformly.
#
# ## How ADAPT-VQE Works
#
# Let us now take a closer look at ADAPT-VQE.
# As in `@CC`, the starting point is the Hartree-Fock state $\vert \Phi_\mathrm{HF} \rangle$.
# Recall that CCSD expresses the state as $\vert \Psi_\mathrm{CCSD} \rangle = e^{T_1 + T_2} \vert \Phi_\mathrm{HF} \rangle$.
# To prepare this state with a quantum circuit, we need to consider unitary CCSD (UCCSD).
# We therefore define
#
# ```{math}
# :label: eq-adapt-ucc-generator
# K
# = T_1 - T_1^\dagger + T_2 - T_2^\dagger
# = A_1 + A_2 \quad (A_1 \equiv T_1 - T_1^\dagger, A_2 \equiv T_2 - T_2^\dagger)
# ```
#
# and use it to construct the state through a unitary transformation:
#
# ```{math}
# :label: eq-adapt-ucc-state
# \lvert \Psi_\mathrm{UCCSD} \rangle
# = e^K \vert \Phi_\mathrm{HF} \rangle
# ```
#
# This gives the UCCSD state.
# Indeed, $K^\dagger = - K$, so $e^K$ is unitary because $(e^K)^\dagger e^K = e^{-K} e^K = I$.
# Conventional single and double excitations distinguish between occupied orbitals $i, j$ and unoccupied orbitals $a, b$ in the Hartree-Fock state.
# The generalized operators introduced in ADAPT-VQE, however, do not restrict orbital combinations according to whether the orbitals are occupied or unoccupied in the Hartree-Fock state.
# This allows the candidate set to include operators that give zero when applied directly to the reference Hartree-Fock state.
# The UCCSD state introduced above can be written as $\vert \Psi_\mathrm{UCCSD} \rangle = e^{\sum_\mu \theta_\mu A_\mu} \vert \Phi_\mathrm{HF} \rangle$.
# Here, $\theta_\mu$ is a parameter that determines the weight of each operator.
# Starting from this idea, ADAPT-VQE selects individual unitary transformations one at a time and constructs the state as their ordered product.
# Specifically,
#
# ```{math}
# :label: eq-adapt-ansatz
# \vert \Psi_\mathrm{ADAPT} \rangle
# = e^{\theta_L A_{\mu_L}} e^{\theta_{L-1} A_{\mu_{L-1}}} \cdots e^{\theta_2 A_{\mu_2}} e^{\theta_1 A_{\mu_1}} \vert \Phi_\mathrm{HF} \rangle
# ```
#
# constructs the ansatz in a manner resembling a Trotter decomposition.
# Note, however, that unlike the usual Trotter decomposition $e^{A + B} = \lim_{r \rightarrow \infty} (e^{A/r} e^{B/r})^r$, the same coefficients are not used throughout.
#
# ### Constructing the Operator Pool and Selecting Operators
#
# As described above, the basic building blocks combine single and double electron excitations with their Hermitian conjugates, which represent de-excitations.
# For a single excitation, the operator is
#
# ```{math}
# :label: eq-adapt-single
# A_{pq}
# =  a_p^\dagger a_q - a_q^\dagger a_p
# ```
#
# This combines the excitation and de-excitation terms.
# For a double excitation, we use
#
# ```{math}
# :label: eq-adapt-double
# B_{pqrs}
# = a_p^\dagger a_q^\dagger a_r a_s, \qquad
# A_{pqrs}
# = B_{pqrs} - B_{pqrs}^\dagger
# ```
#
# which has the same structure.
# ADAPT-VQE uses the operators in Eqs. {eq}`eq-adapt-single` and {eq}`eq-adapt-double` to prepare a pool of candidates $\mathcal{P} = \{A_1, A_2, \dots, A_\mu, \dots, A_M\}$ for constructing the circuit.
# For example, suppose that $A_3$ is selected first, followed by $A_8$, and finally $A_2$.
# The resulting state is $\vert \Psi (\boldsymbol{\theta}) \rangle = e^{\theta_3 A_2} e^{\theta_2 A_8} e^{\theta_1 A_3} \vert \Phi_\mathrm{HF} \rangle$.
# Selected operators remain in the pool and may be selected again in later iterations.
# To determine which operator to apply next, let $\vert \Psi^{(k)} \rangle$ denote the state obtained after adding $k$ operators and optimizing their parameters.
# The state obtained by appending a candidate operator $A_\mu$ and its energy are
#
# ```{math}
# :label: eq-adapt-candidate
# \vert \Psi (\alpha) \rangle
# = e^{\alpha A_\mu} \vert \Psi^{(k)} \rangle, \qquad
# E_\mu (\alpha)
# = \langle \Psi^{(k)} \vert e^{-\alpha A_\mu} H e^{\alpha A_\mu} \vert \Psi^{(k)} \rangle
# ```
#
# respectively.
# We now calculate the gradient at $\alpha = 0$, before the appended operator changes the state.
# This gives
#
# ```{math}
# :label: eq-adapt-gradient
# g_\mu
# = \left. \frac{dE_\mu (\alpha)}{d\alpha} \right\vert_{\alpha = 0}
# = \left. \langle \Psi^{(k)} \vert (-A_\mu e^{-\alpha A_\mu} H e^{\alpha A_\mu} + e^{-\alpha A_\mu} H A_\mu e^{\alpha A_\mu}) \vert \Psi^{(k)} \rangle \right\vert_{\alpha = 0}
# = \langle \Psi^{(k)} \vert (-A_\mu H + H A_\mu) \vert \Psi^{(k)} \rangle
# = \langle \Psi^{(k)} \vert [H, A_\mu] \vert \Psi^{(k)} \rangle
# ```
#
# Using these gradients $g_\mu$, we select the operator $A_{\mu^\ast}$ with the largest absolute gradient.
# Even when the gradient is positive, the corresponding parameter $\theta_{k+1}$ can be made negative, so selection depends only on the magnitude of the gradient.
# This criterion measures how readily the energy can improve near the current state $\vert \Psi^{(k)} \rangle$.
# It does not exhaustively search for the operator that would produce the largest energy reduction after it is added and all parameters are optimized.
# We append the selected operator to obtain the ansatz $\vert \Psi^{(k+1)} (\boldsymbol{\theta}) \rangle = e^{\theta_{k+1} A_{\mu^\ast}} e^{\theta_{k} A_{\mu_k}} e^{\theta_{k-1} A_{\mu_{k-1}}} \cdots e^{\theta_{1} A_{\mu_{1}}} \vert \Phi_\mathrm{HF} \rangle$, then run conventional VQE to solve
#
# ```{math}
# :label: eq-adapt-optimize
# \min_{\theta_1, \theta_2, \dots, \theta_{k+1}} \langle \Psi^{(k+1)} (\boldsymbol{\theta}) \vert H \vert \Psi^{(k+1)} (\boldsymbol{\theta}) \rangle
# ```
#
# for the minimum energy.
# All parameters $\theta_{k+1}, \theta_k, \dots, \theta_1$ associated with the operators added so far are optimized.
# For the VQE calculation at step $k + 1$, we initialize the parameters as $\boldsymbol{\theta}_\mathrm{initial} = (\boldsymbol{\theta}^\ast, 0)$, using the optimized parameters $\boldsymbol{\theta}^\ast$ from step $k$.
# The initial value of $\theta_{k+1}$, the parameter for the operator $A_{\mu^\ast}$ newly added at step $k+1$, is zero.
# Since $e^{0 A_{\mu^\ast}} = I$, the state immediately after adding the operator is identical to the optimized state from step $k$.
# Initializing $\theta_{k+1}$ to zero therefore naturally embeds the previous solution in the new ansatz.

# %% [markdown]
# ## Implementation with Qamomile
#
# Let us now implement ADAPT-VQE with Qamomile.
#
# ### Setting Up the Molecule
#
# We will calculate the energy of the $\mathrm{H}_2$ molecule.
# Specify the internuclear distances in angstrom (Å) and report the energies in hartree (Ha).
# We use STO-3G, a minimal orbital basis: STO means Slater-type orbital, and 3G means that each such orbital is approximated by a combination of three Gaussian functions. For $\mathrm{H}_2$, this supplies two spatial orbitals, each with two spin states, giving four spin orbitals and four qubits under the Jordan-Wigner transformation.

# %%
# Internuclear distances R [Å]. Run the cells in order; adjust this array as needed.
bond_lengths = np.array([0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00])
basis = "sto-3g"

gradient_threshold = 1e-3  # Threshold for the normalized operator pool
max_adapt_steps = 6  # Maximum number of operators to add at each distance
max_vqe_iterations = 120  # Maximum BFGS iterations after each operator addition

assert bond_lengths.ndim == 1 and len(bond_lengths) >= 2
assert np.all(bond_lengths > 0)
assert np.all(np.diff(bond_lengths) > 0), (
    "Distances must be strictly increasing with no duplicates"
)
print("Distances to evaluate [Å]:", bond_lengths.tolist())
print("Number of points:", len(bond_lengths))

# %% [markdown]
# ### Determining the Number of Qubits
#
# In the STO-3G basis, $\mathrm{H}_2$ has two spatial orbitals, four spin orbitals, and two electrons.
# As long as we use the same basis, the number of qubits and electrons does not change with the distance. The generalized excitation pool and its generators depend on these orbital labels rather than the distance, so we construct them once and reuse them throughout the scan.
# We therefore create a molecular model at the smallest distance specified above and determine the required number of qubits.
# We also calculate the FCI reference energy. FCI diagonalizes the electronic Hamiltonian over all electron configurations allowed by the chosen orbital basis and symmetry sector. Agreement with FCI therefore validates this STO-3G model, rather than the complete-basis molecular problem.

# %%
molecule_dir = TemporaryDirectory()


def calculate_h2(distance):
    """Compute the molecular integrals and reference energies at one distance."""
    geometry = [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, distance]]]
    molecule = MolecularData(
        geometry,
        basis,
        multiplicity=1,
        charge=0,
        description=str(distance),
        data_directory=molecule_dir.name,
    )
    return run_pyscf(molecule, run_scf=True, run_fci=True)


first_R = float(bond_lengths[0])
first_molecule = calculate_h2(first_R)

n_qubits = first_molecule.n_qubits
n_electrons = first_molecule.n_electrons
n_spatial = n_qubits // 2

assert n_qubits == 4 and n_electrons == 2
print(f"Qubits: {n_qubits}, Electrons: {n_electrons}, Spatial orbitals: {n_spatial}")
print(
    f"R={first_R:.2f} Å: HF={first_molecule.hf_energy:.10f} Ha, "
    f"FCI={first_molecule.fci_energy:.10f} Ha"
)

# %% [markdown]
# ### Enumerating Generalized Single and Double Excitations
#
# We consider generalized combinations of spatial orbitals, rather than restricting excitations to transitions from occupied to unoccupied Hartree-Fock orbitals.
# Here, we construct a simplified pool of fermionic operators for $\mathrm{H}_2$.

# %%
raw_pool = []

# Spin-complemented generalized single excitations.
for p, q in combinations(range(n_spatial), 2):
    T = FermionOperator(((2 * q, 1), (2 * p, 0))) + FermionOperator(
        ((2 * q + 1, 1), (2 * p + 1, 0))
    )
    raw_pool.append((f"Single excitation({p}->{q})", T))

# Opposite-spin double excitations and their spin-exchanged counterparts.
for p, q, r, s in product(range(n_spatial), repeat=4):
    T = FermionOperator(
        ((2 * r, 1), (2 * s + 1, 1), (2 * q + 1, 0), (2 * p, 0))
    ) + FermionOperator(((2 * r + 1, 1), (2 * s, 1), (2 * q, 0), (2 * p + 1, 0)))
    raw_pool.append((f"Double excitation_ab({p},{q}->{r},{s})", T))

# Same-spin double excitations and their spin-exchanged counterparts.
for p, q in combinations(range(n_spatial), 2):
    for r, s in combinations(range(n_spatial), 2):
        T = FermionOperator(
            ((2 * r, 1), (2 * s, 1), (2 * q, 0), (2 * p, 0))
        ) + FermionOperator(
            ((2 * r + 1, 1), (2 * s + 1, 1), (2 * q + 1, 0), (2 * p + 1, 0))
        )
        raw_pool.append((f"Double excitation_same({p},{q}->{r},{s})", T))

print("Number of candidates before deduplication:", len(raw_pool))

# %% [markdown]
# ### Anti-Hermitian Operators, Normalization, and Deduplication
#
# From each excitation operator $T_\mu$ constructed above, we form $A_\mu = T_\mu - T_\mu^\dagger$.
# We then normalize its coefficient norm to one.
# Since $A_\mu$ and $- A_\mu$ can be implemented by reversing the sign of the same parameter, we remove these duplicates.

# %%
pool = []
pool_labels = []
seen = set()

for label, T in raw_pool:
    A = normal_ordered(T - hermitian_conjugated(T))
    A.compress()
    norm = np.sqrt(sum(abs(c) ** 2 for c in A.terms.values()))
    if norm < 1e-12:
        continue

    A = A / norm
    first_coeff = sorted(A.terms.items())[0][1]
    if first_coeff.real < 0:
        A = -A

    key = tuple(
        (term, round(float(coeff.real), 10), round(float(coeff.imag), 10))
        for term, coeff in sorted(A.terms.items())
    )
    if key in seen:
        continue
    seen.add(key)
    pool.append(A)
    pool_labels.append(label)

print("Operator pool size after deduplication:", len(pool))
for j, label in enumerate(pool_labels):
    print(f"{j:2d}: {label}")
assert len(pool) > 0

# %% [markdown]
# ### Converting to Observables
#
# We apply the Jordan-Wigner transformation to the generator $G_\mu = i A_\mu$ that appears in the exponent.
# Using the transformed generator, we implement $e^{-i\theta G_\mu} \ (=e^{\theta A_\mu})$.


# %%
def to_qamomile_observable(op, width):
    """Convert a real-coefficient OpenFermion Pauli sum to an observable."""
    pauli_factories = {"X": qmo.X, "Y": qmo.Y, "Z": qmo.Z}
    obs = qmo.Hamiltonian(num_qubits=width)
    for term, coeff in op.terms.items():
        if abs(coeff.imag) > 1e-8:
            raise ValueError(f"The observable has a complex coefficient: {coeff}")
        value = float(coeff.real)
        if abs(value) < 1e-12:
            continue
        if len(term) == 0:
            obs += qmo.Hamiltonian.identity(value, num_qubits=width)
        else:
            pauli_product = qmo.Hamiltonian.identity(num_qubits=width)
            for qubit_index, symbol in term:
                pauli_product *= pauli_factories[symbol](qubit_index)
            obs += value * pauli_product
    return obs


# Apply the Jordan-Wigner transformation to G_mu = i*A_mu for the generalized H2 excitations.
generators = [to_qamomile_observable(jordan_wigner(1j * A), n_qubits) for A in pool]

M = len(pool)
assert len(generators) == M and M > 0
print(f"Operator pool size: {M}")


# %% [markdown]
# ### Creating a Qamomile Quantum Kernel
#
# Let us create a quantum kernel that prepares the Hartree-Fock state $\vert \Phi_\mathrm{HF} \rangle$, applies the selected generators $G_\mu$ in sequence, and returns the prepared state. The `adapt_vqe_expectation` function evaluates the expectation value of a specified observable using the exact Aer estimator.
# Passing $H$ as `observable` gives the energy.
# Passing $[H, A_\mu]$ instead gives the gradient used for operator selection.


# %%
@qm.qkernel
def adapt_vqe_kernel(
    width: qm.Const[qm.Int],
    occupied: qm.Const[qm.Int],
    depth: qm.Const[qm.Int],
    count: qm.Const[qm.Int],
    operators: qm.Const[qm.Vector[qm.Observable, count]],  # noqa: F821
    thetas: qm.Param[qm.Vector[qm.Real, count]],  # noqa: F821
) -> qm.Q[qm.Bits[width]]:  # noqa: F821
    q = qm.qubits(width)
    for i in range(occupied):
        q = qm.x(q, target=i)  # Occupy the Hartree-Fock orbitals
    for i in range(depth):
        q = qm.pauli_evolve(q, operators[i], thetas[i])
    return q


def adapt_vqe_expectation(program, thetas, observable):
    circuit = to_circuit(
        program, target=target, parameters={"thetas": tuple(float(x) for x in thetas)}
    )
    width = circuit.num_qubits
    if observable.num_qubits > width:
        raise ValueError(
            f"Observable needs {observable.num_qubits} qubits, circuit has {width}"
        )
    rows = [("I" * width, observable.constant)]
    for word, coefficient in observable.terms:
        label = ["I"] * width
        for factor in word.factors:
            label[width - factor.index - 1] = factor.pauli.value
        rows.append(("".join(label), coefficient))
    operator = SparsePauliOp.from_list(rows)
    return float(executor.run([(circuit, operator)]).result()[0].data.evs)


# %% [markdown]
# ## Execution and Results
#
# We are now ready to run ADAPT-VQE at each distance.
# For each distance `R`, we generate the $\mathrm{H}_2$ Hamiltonian and calculate the Hartree-Fock state.
# We also calculate the FCI energy for comparison.
# Using the same operator pool $\{A_\mu\}$ throughout, we construct $[H(R), A_\mu]$ for each distance.
# At each step, we evaluate $g_\mu = \langle [H(R), A_\mu]\rangle$ and append the operator with the largest $\vert g_\mu \vert$.
# After each addition, we optimize all variational parameters with BFGS and record the final energy, gradient norm, number of operators, and selected operators for each distance.
#
# :::{note}
# This implementation decomposes Qamomile's Pauli time evolution into elementary gates. `pauli_evolve` requires commuting Pauli terms. For molecules such as $\mathrm{LiH}$ or $\mathrm{H}_4$, generators with noncommuting terms require an explicit product formula, which can introduce approximation errors.
# :::

# %% [markdown]
# ### 1. Build the Energy and Gradient Observables
#
# Following Eq. {eq}`eq-adapt-gradient`, construct the molecular Hamiltonian $H$ and each commutator $[H, A_\mu]$. Reusing the generator conversion function handles identity terms and Pauli products in the same way.


# %%
def molecular_observables(molecule):
    """Build the energy and commutator observables for one molecular model."""
    assert molecule.n_qubits == n_qubits and molecule.n_electrons == n_electrons
    fermionic_H = get_fermion_operator(molecule.get_molecular_hamiltonian())
    H = to_qamomile_observable(jordan_wigner(fermionic_H), n_qubits)
    gradient_observables = [
        to_qamomile_observable(
            jordan_wigner(normal_ordered(commutator(fermionic_H, A))), n_qubits
        )
        for A in pool
    ]
    assert len(gradient_observables) == M
    return H, gradient_observables


# %% [markdown]
# ### 2. Grow the Ansatz at One Distance
#
# Prepare the state in Eq. {eq}`eq-adapt-ansatz` and select an operator using the gradients in Eq. {eq}`eq-adapt-gradient`. Then reoptimize all parameters with BFGS as in Eq. {eq}`eq-adapt-optimize`. Each call starts from the Hartree-Fock state and stops when the gradient norm falls below the threshold or the operator cap is reached.

# %%
target = Qiskit(use_native_pauli_evolution=False)
executor = EstimatorV2()


def run_adapt_vqe(R, molecule):
    """Optimize one distance and return its results, labels, and energy trace."""
    hf_energy = float(molecule.hf_energy)
    fci_energy = float(molecule.fci_energy)
    H, gradient_observables = molecular_observables(molecule)
    selected = []
    theta = np.array([], dtype=float)
    energy_history = [hf_energy]
    gradient_norm_history = []
    converged = False
    vqe_warning_count = 0

    # Use one extra iteration to check the gradient norm after the last operator addition.
    for step in range(max_adapt_steps + 1):
        current_operators = (
            [generators[j] for j in selected] if selected else [qmo.Z(0)]
        )
        program = qm.specialize(
            adapt_vqe_kernel,
            width=n_qubits,
            occupied=n_electrons,
            depth=len(selected),
            count=len(current_operators),
            operators=tuple(current_operators),
        )

        # Evaluate the energy gradients for every operator in the pool
        gradients = np.zeros(M)
        for j in range(M):
            gradients[j] = adapt_vqe_expectation(
                program, theta if selected else [0.0], gradient_observables[j]
            )

        gradient_norm = float(np.linalg.norm(gradients))
        gradient_norm_history.append(gradient_norm)
        if gradient_norm < gradient_threshold:
            converged = True
            print(f"  Converged: ||g||={gradient_norm:.3e}, Operators={len(selected)}")
            break

        if step == max_adapt_steps:
            print(
                f"  Reached the maximum number of operators: ||g||={gradient_norm:.3e}"
            )
            break

        # Append the operator with the largest gradient magnitude and reoptimize all theta
        best = int(np.argmax(np.abs(gradients)))
        selected.append(best)  # Keep it in the pool so it can be selected again
        theta = np.append(theta, 0.0)

        energy_program = qm.specialize(
            adapt_vqe_kernel,
            width=n_qubits,
            occupied=n_electrons,
            depth=len(selected),
            count=len(selected),
            operators=tuple(generators[j] for j in selected),
        )
        result = minimize(
            lambda x: adapt_vqe_expectation(energy_program, x, H),
            theta,
            method="BFGS",
            options={"maxiter": max_vqe_iterations, "gtol": 1e-6},
        )
        if not np.isfinite(result.fun):
            raise RuntimeError(f"R={R:.3f} Å: The VQE energy is not finite")

        theta = np.asarray(result.x, dtype=float)
        energy_history.append(float(result.fun))
        if not result.success:
            vqe_warning_count += 1
            print(f"  BFGS warning (step={step + 1}): {result.message}")
        print(f"  Added {step + 1}: {pool_labels[best]} / E={result.fun:.10f} Ha")

    # Store the results for this distance
    adapt_energy = float(energy_history[-1])
    absolute_error_kcal_mol = abs(adapt_energy - fci_energy) * 627.509474
    record = {
        "R": R,
        "HF": hf_energy,
        "FCI": fci_energy,
        "ADAPT": adapt_energy,
        "log10_absolute_error": (
            np.log10(absolute_error_kcal_mol)
            if absolute_error_kcal_mol > 0
            else -np.inf
        ),
        "n_operators": len(selected),
        "gradient_norm": gradient_norm_history[-1],
        "gradient_converged": converged,
        "vqe_warnings": vqe_warning_count,
    }
    return record, [pool_labels[j] for j in selected], energy_history.copy()


# %% [markdown]
# ### 3. Scan All Requested Distances
#
# Build the molecular model at each distance, run the preceding function, and store the energy, convergence status, and selected operators. Reuse the molecular model already calculated at the first distance.

# %%
scan_results = []
selected_labels_by_distance = []
energy_traces_by_distance = []

for distance_index, distance in enumerate(bond_lengths):
    R = float(distance)
    print(f"\n[{distance_index + 1}/{len(bond_lengths)}] R = {R:.3f} Å")

    if distance_index == 0:
        molecule = first_molecule
    else:
        molecule = calculate_h2(R)

    record, labels, energy_history = run_adapt_vqe(R, molecule)
    scan_results.append(record)
    selected_labels_by_distance.append(labels)
    energy_traces_by_distance.append(energy_history)
    print(
        f"  Final: HF={record['HF']:.8f}, ADAPT={record['ADAPT']:.8f}, "
        f"FCI={record['FCI']:.8f} Ha, "
        f"log10(absolute error / (1 kcal/mol))={record['log10_absolute_error']:.4g}"
    )

print(f"\nCompleted calculations for all {len(scan_results)} points.")

# %% [markdown]
# ### 4. Validate the Results
#
# Check that every requested distance was evaluated and that every gradient norm fell below the convergence threshold. Also require ADAPT-VQE and FCI energies to agree within an absolute tolerance of $10^{-8}$ Ha. This tolerance checks the numerical calculation for this ideal simulator and STO-3G model.

# %%
R_values = np.array([row["R"] for row in scan_results])
E_hf = np.array([row["HF"] for row in scan_results])
E_adapt = np.array([row["ADAPT"] for row in scan_results])
E_fci = np.array([row["FCI"] for row in scan_results])

assert len(scan_results) == len(bond_lengths), "Some requested distances are missing"
np.testing.assert_array_equal(R_values, bond_lengths)
assert all(row["gradient_converged"] for row in scan_results), "ADAPT did not converge"
assert all(0 <= row["gradient_norm"] < gradient_threshold for row in scan_results)
assert np.all(np.isfinite(E_adapt)) and np.all(np.isfinite(E_fci))
fci_tolerance = 1e-8  # Absolute energy tolerance in Ha for this exact simulation.
np.testing.assert_allclose(E_adapt, E_fci, rtol=0, atol=fci_tolerance, equal_nan=False)

# %% [markdown]
# ### Inspecting and Plotting the Results
#
# Finally, we plot the energy in Ha against the internuclear distance in Å.

# %%
table_header = (
    f"{'R[Å]':>4}  {'HF[Ha]':>12}  {'ADAPT[Ha]':>12}  {'FCI[Ha]':>12}  "
    f"{'log10(absolute error / (1 kcal/mol))':>36}  {'Nop':>3}  Converged"
)
print(table_header)
print("-" * len(table_header))
for result in scan_results:
    print(
        f"{result['R']:4.2f}  {result['HF']:12.9f}  {result['ADAPT']:12.9f}  "
        f"{result['FCI']:12.9f}  {result['log10_absolute_error']:36.5f}  "
        f"{result['n_operators']:3d}  {result['gradient_converged']}"
    )

for record, labels in zip(scan_results, selected_labels_by_distance):
    print(
        f"R={record['R']:.2f} Å: ",
        " -> ".join(labels) if labels else "No operators added",
    )

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(R_values, E_hf, "s--", label="Hartree-Fock")
ax.plot(R_values, E_adapt, "o-", label="ADAPT-VQE")
ax.plot(R_values, E_fci, "^-", label="FCI")
ax.set_xlabel("H-H distance (Angstrom)")
ax.set_ylabel("Total energy (Hartree)")
ax.set_title("H2 potential energy curve (STO-3G)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# This tutorial explained the method proposed by {cite:t}`grimsley2019` and showed how to implement it with Qamomile.
# The key points are:
#
# * ADAPT-VQE is a hybrid quantum–classical algorithm that selects operators using energy gradients and progressively grows the ansatz.
# * In this implementation, the pool consists of generalized single and double fermionic excitation operators. The gradient for each candidate can be evaluated as the expectation value of its commutator with the Hamiltonian.
# * Qamomile lets us express parameterized state preparation in a quantum kernel and expectation-value evaluation with the exact Aer estimator. The same kernel can be used for both energy evaluation and operator selection.
# * The $\mathrm{H}_2$ energy curve calculated with ADAPT-VQE agrees with the FCI results.
# * These results are a small-scale validation using an ideal simulator. For more complex molecules or execution on real hardware, we need to consider the operator pool size, the cost of measuring gradients, circuit depth, and the effects of noise.

# %% [markdown]
#
