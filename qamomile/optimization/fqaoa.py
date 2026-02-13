"""
This module implements the Fermionic QAOA (FQAOA) converter for the Qamomile framework :cite:`yoshioka2023fermionic`.
FQAOA translates the Hamiltonians into the representation of fermion systems,
and the equality constraint is naturally incorporated as a constant number of particles condition.

The parameterized state :math:`|\\vec{\\beta},\\vec{\\gamma}\\rangle` of :math:`p`-layer QAOA is defined as:

.. math::
    |\\vec{\\beta},\\vec{\\gamma}\\rangle = U(\\vec{\\beta},\\vec{\\gamma})|0\\rangle^{\\otimes n} = e^{-i\\beta_{p-1} H_M}e^{-i\\gamma_{p-1} H_P} \\cdots e^{-i\\beta_0 H_M}e^{-i\\gamma_0 H_P} U_{init}|0\\rangle^{\\otimes n}

where :math:`H_P` is the cost Hamiltonian, :math:`H_M` is the mixer Hamiltonian and :math:`\\gamma_l` and :math:`\\beta_l` are the variational parameters.
The :math:`2p` variational parameters are optimized classically to minimize the expectation value :math:`\\langle \\vec{\\beta},\\vec{\\gamma}|H_P|\\vec{\\beta},\\vec{\\gamma}\\rangle`.
:math:`U_{init}` prepares the initial state using Givens rotation gates :cite:`jiang2018quantum`.

This module provides functionality to convert optimization problems which written by `jijmodeling`
into FQAOA circuits (:math:`U(\\vec{\\beta}, \\vec{\\gamma})`), construct cost Hamiltonians (:math:`H_P`), and decode quantum computation results.

The `FQAOAConverter` class extends the `MathematicalProblemConverter` base class, specializing in
FQAOA-specific operations such as ansatz circuit generation and result decoding.


Key Features:
        - Generation of FQAOA ansatz circuits
        - Construction of cost Hamiltonians for QAOA
        - Decoding of quantum computation results into classical optimization solutions

Note:
        This module requires `jijmodeling` and `ommx` for problem representation.

.. bibliography::
    :filter: docname in docnames

"""

import numpy as np
import typing as typ
import ommx.v1

import qamomile.circuit as qm_c
import qamomile.observable as qm_o
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.algorithm.fqaoa import (
    initial_occupations,
    givens_rotations,
    hopping_gate,
    mixer_layer,
    cost_layer,
    fqaoa_layers,
)
from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.utils import is_close_zero
from qamomile.optimization.converters.converter import MathematicalProblemConverter


class FQAOAConverter(MathematicalProblemConverter):
    """
    FQAOA (Fermionic Quantum Approximate Optimization Algorithm) converter class.

    This class provides methods to convert optimization problems into FQAOA circuits,
    construct cost Hamiltonians, and decode quantum computation results.

    Examples:

    .. code::

        from qamomile.optimization.fqaoa import FQAOAConverter

        # Initialize with a compiled optimization problem instance
        fqaoa_converter = FQAOAConverter(compiled_instance, num_fermions=4)

        # Generate QAOA circuit and cost Hamiltonian
        p = 2  # Number of QAOA layers
        fqaoa_kernel = fqaoa_converter.get_fqaoa_ansatz(p)
        cost_hamiltonian = fqaoa_converter.get_cost_hamiltonian()

    """

    def __init__(
        self,
        instance: ommx.v1.Instance,
        num_fermions: int,
        normalize_ising: typ.Optional[typ.Literal["abs_max", "rms"]] = None,
    ):
        """
        Initialize the FQAOAConverter.

        This method initializes the converter with the compiled instance of the optimization problem.

        Args:
            instance: ommx.v1.Instance.
            num_fermions (int): Number of fermions. This means the constraint :math:`M = \\sum_{l,d} x_{l,d}`.
            normalize_ising (Literal["abs_max", "rms"] | None): The normalization method for the Ising Hamiltonian. \
                Available options:
                - "abs_max": Normalize by absolute maximum value
                - "rms": Normalize by root mean square
                Defaults to None.

        """
        if instance.objective.degree() > 2:
            raise ValueError("FQAOAConverter supports only QUBO instances.")

        self.num_fermions = num_fermions
        self.normalize_ising = normalize_ising

        # FQAOA uses uniform_penalty_weight=0.0 (constraints handled by fermion number conservation)
        qubo, constant = instance.to_qubo(uniform_penalty_weight=0.0)
        binary_model = BinaryModel.from_qubo(qubo, constant)

        # Store the original instance for cyclic_mapping and index labeling
        self._original_instance = instance

        # Pass the BinaryModel to the base class (which converts to SPIN internally)
        super().__init__(binary_model)

    def __post_init__(self) -> None:
        last_var = self._original_instance.decision_variables[-1]
        n, d = last_var.subscripts
        self.num_integers, self.num_bits = n + 1, d + 1
        self.var_map = self.cyclic_mapping()
        self.num_qubits = self.spin_model.num_bits

        # Apply normalization if requested
        if isinstance(self.normalize_ising, str):
            if self.normalize_ising == "abs_max":
                self.spin_model.normalize_by_abs_max(replace=True)
            elif self.normalize_ising == "rms":
                self.spin_model.normalize_by_rms(replace=True)
            else:
                raise ValueError(
                    f"Invalid value for normalize_ising: {self.normalize_ising}"
                )

    def cyclic_mapping(self) -> dict[tuple[int, int], int]:
        """
        Get variable maps between decision variable indices :math:`(l,d)` and qubit index :math:`i`.

        Return:
                        dict[tuple[int, int], int] : A variable map for ring driver.
        """
        cyclic_var_map = {}
        for var in self._original_instance.decision_variables:
            # l = pos[0], d = pos[1]
            pos = var.subscripts
            cyclic_var_map[tuple(pos)] = pos[0] + self.num_integers * pos[1]

        return cyclic_var_map

    def get_fermi_orbital(self) -> np.ndarray:
        """
        Compute the single-particle wave functions of the occupied spin orbitals.

        Return:
                        numpy.ndarray: A 2D numpy array of shape (num_fermions, num_qubits)

        """
        orbital = np.zeros((self.num_fermions, self.num_qubits))

        if self.num_fermions % 2 == 0:  # num_fermion is even
            for i in range(self.num_qubits):
                for k in range(int(self.num_fermions / 2)):
                    angle = 2.0 * np.pi * (k + 0.5) * (i + 1) / self.num_qubits
                    orbital[k, i] = np.sqrt(2.0 / self.num_qubits) * np.sin(angle)
                    orbital[int(self.num_fermions - 1 - k), i] = np.sqrt(
                        2.0 / self.num_qubits
                    ) * np.cos(angle)
        else:  # num_fermion is odd
            for i in range(self.num_qubits):
                orbital[0, i] = np.sqrt(1.0 / self.num_qubits)
                for k in range(int(self.num_fermions / 2)):
                    angle = 2.0 * np.pi * (k + 1) * (i + 1) / self.num_qubits
                    orbital[k, i] = np.sqrt(2.0 / self.num_qubits) * np.sin(angle)
                    orbital[int(self.num_fermions - 1 - k), i] = np.sqrt(
                        2.0 / self.num_qubits
                    ) * np.cos(angle)

        return orbital

    def _givens_decomposition(self, fermi_orbital: np.ndarray) -> list[list]:
        m, n = fermi_orbital.shape
        matrix = fermi_orbital.copy()

        # left unitary
        for j in reversed(range(n - m, n)):
            for i in range(m - n + j):
                # givens rotation matrix
                sin_ = -matrix[i, j] / np.sqrt(
                    matrix[i, j] ** 2 + matrix[i + 1, j] ** 2
                )
                cos_ = matrix[i + 1, j] / np.sqrt(
                    matrix[i, j] ** 2 + matrix[i + 1, j] ** 2
                )

                # rotate
                row_1 = matrix[i].copy()
                row_2 = matrix[i + 1].copy()
                matrix[i] = cos_ * row_1 + sin_ * row_2
                matrix[i + 1] = -sin_ * row_1 + cos_ * row_2

        # right unitary
        givens_angles = []
        for i in range(m):
            for j in reversed(range(i, i + n - m)):
                # givens rotation matrix
                cos_ = matrix[i, j] / np.sqrt(matrix[i, j] ** 2 + matrix[i, j + 1] ** 2)
                sin_ = -matrix[i, j + 1] / np.sqrt(
                    matrix[i, j] ** 2 + matrix[i, j + 1] ** 2
                )
                if sin_ >= 0:
                    angle = np.arccos(cos_)
                else:
                    angle = -np.arccos(cos_)

                givens_angles.append([(j, j + 1), angle])

                # rotate
                col_1 = matrix[:, j].copy()
                col_2 = matrix[:, j + 1].copy()
                matrix[:, j] = np.cos(angle) * col_1 - np.sin(angle) * col_2
                matrix[:, j + 1] = np.sin(angle) * col_1 + np.cos(angle) * col_2

        return givens_angles

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Construct the Ising cost Hamiltonian from the spin model.

        Builds a Pauli-Z Hamiltonian from the spin model's linear and
        quadratic terms.

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian.
        """
        hamiltonian = qm_o.Hamiltonian()

        for i, hi in self.spin_model.linear.items():
            if not is_close_zero(hi):
                hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, i),), hi)

        for (i, j), Jij in self.spin_model.quad.items():
            if not is_close_zero(Jij):
                hamiltonian.add_term(
                    (
                        qm_o.PauliOperator(qm_o.Pauli.Z, i),
                        qm_o.PauliOperator(qm_o.Pauli.Z, j),
                    ),
                    Jij,
                )

        hamiltonian.constant = self.spin_model.constant
        return hamiltonian

    def get_init_state(self) -> qm_c.QKernel:
        """Generate the initial state preparation kernel for FQAOA."""
        unitary_rows = self.get_fermi_orbital()
        givens_rotation_data = self._givens_decomposition(unitary_rows)
        num_fermions = self.num_fermions

        @qm_c.qkernel
        def init_state(q: qm_c.Vector[qm_c.Qubit]) -> qm_c.Vector[qm_c.Qubit]:
            q = initial_occupations(q, num_fermions)
            q = givens_rotations(q, givens_rotation_data)
            return q

        return init_state

    def get_mixer_ansatz(self, hopping: float = 1.0) -> qm_c.QKernel:
        """Generate the fermionic mixer ansatz kernel (:math:`e^{-\\beta H_d}`)."""
        num_qubits = self.num_qubits

        @qm_c.qkernel
        def mixer(
            q: qm_c.Vector[qm_c.Qubit],
            beta: qm_c.Float,
        ) -> qm_c.Vector[qm_c.Qubit]:
            return mixer_layer(q, beta, hopping, num_qubits)

        return mixer

    def get_cost_ansatz(self) -> qm_c.QKernel:
        """Generate the cost ansatz kernel (:math:`e^{-\\gamma H_P}`)."""
        linear = self.spin_model.linear
        quad = self.spin_model.quad

        @qm_c.qkernel
        def cost(
            q: qm_c.Vector[qm_c.Qubit],
            gamma: qm_c.Float,
        ) -> qm_c.Vector[qm_c.Qubit]:
            return cost_layer(q, gamma, linear, quad)

        return cost

    def get_fqaoa_ansatz(self, p: int, hopping: float = 1.0) -> qm_c.QKernel:
        """Generate the FQAOA ansatz kernel."""
        num_qubits = self.num_qubits
        num_fermions = self.num_fermions
        unitary_rows = self.get_fermi_orbital()
        givens_rotation_data = self._givens_decomposition(unitary_rows)
        linear = self.spin_model.linear
        quad = self.spin_model.quad

        @qm_c.qkernel
        def fqaoa_state(
            betas: qm_c.Vector[qm_c.Float],
            gammas: qm_c.Vector[qm_c.Float],
        ) -> qm_c.Vector[qm_c.Qubit]:
            q = qm_c.qubit_array(num_qubits, name="q")
            q = initial_occupations(q, num_fermions)
            q = givens_rotations(q, givens_rotation_data)
            q = fqaoa_layers(q, betas, gammas, p, linear, quad, hopping, num_qubits)
            return q

        return fqaoa_state

    def transpile(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        hopping: float = 1.0,
    ) -> ExecutableProgram:
        """Compile FQAOA ansatz into an executable program with measurements."""
        num_qubits = self.num_qubits
        num_fermions = self.num_fermions
        unitary_rows = self.get_fermi_orbital()
        givens_rotation_data = self._givens_decomposition(unitary_rows)

        # Filter near-zero coefficients to keep loops compact
        linear = {
            i: hi for i, hi in self.spin_model.linear.items() if not is_close_zero(hi)
        }
        quad = {
            ij: Jij
            for ij, Jij in self.spin_model.quad.items()
            if not is_close_zero(Jij)
        }

        last_qubit = num_qubits - 1

        @qm_c.qkernel
        def fqaoa_sampling(
            betas: qm_c.Vector[qm_c.Float],
            gammas: qm_c.Vector[qm_c.Float],
            linear: qm_c.Dict[qm_c.UInt, qm_c.Float],
            quad: qm_c.Dict[qm_c.Tuple[qm_c.UInt, qm_c.UInt], qm_c.Float],
        ) -> qm_c.Vector[qm_c.Bit]:
            q = qm_c.qubit_array(num_qubits, name="q")

            # Initial occupations
            for i in qm_c.range(num_fermions):
                q[i] = qm_c.x(q[i])

            # Givens rotations (precomputed constants)
            q = givens_rotations(q, givens_rotation_data)

            for layer in qm_c.range(p):
                # Cost layer
                for i, hi in qm_c.items(linear):
                    q[i] = qm_c.rz(q[i], 2 * hi * gammas[layer])
                for (i, j), Jij in qm_c.items(quad):
                    q[i], q[j] = qm_c.rzz(q[i], q[j], 2 * Jij * gammas[layer])

                # Mixer layer
                for i in qm_c.range(0, last_qubit, 2):
                    q = hopping_gate(q, i, i + 1, betas[layer], hopping)
                for i in qm_c.range(1, last_qubit, 2):
                    q = hopping_gate(q, i, i + 1, betas[layer], hopping)
                # Boundary hop (wrap in control flow to avoid segment splits)
                for _ in qm_c.range(1):
                    q = hopping_gate(q, 0, last_qubit, betas[layer], hopping)

            return qm_c.measure(q)

        return transpiler.transpile(
            fqaoa_sampling,
            bindings={"linear": linear, "quad": quad},
            parameters=["betas", "gammas"],
        )
