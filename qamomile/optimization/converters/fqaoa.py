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

The `QAOAConverter` class extends the `QuantumConverter` base class, specializing in
FQAOA-specific operations such as ansatz circuit generation and result decoding.


Key Features:
        - Generation of FQAOA ansatz circuits
        - Construction of cost Hamiltonians for QAOA
        - Decoding of quantum computation results into classical optimization solutions

Note:
        This module requires `jijmodeling` and `jijmodeling_transpiler` for problem representation.

.. bibliography::
    :filter: docname in docnames

"""

import copy
import numpy as np
import typing as typ
import ommx.v1

import qamomile.circuit as qm_c
import qamomile.observable as qm_o
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.optimization.ising_model import IsingModel
from qamomile.optimization.utils import is_close_zero
from qamomile.optimization.converter import MathematicalProblemConverter


@qm_c.qkernel
def _ry_gate(q: qm_c.Qubit, theta: qm_c.Float) -> qm_c.Qubit:
    return qm_c.ry(q, theta)


_controlled_ry = qm_c.controlled(_ry_gate)


def _apply_initial_occupations(
    q: qm_c.Vector[qm_c.Qubit],
    num_fermions: int,
) -> qm_c.Vector[qm_c.Qubit]:
    for i in range(num_fermions):
        q[i] = qm_c.x(q[i])
    return q


def _apply_givens_rotation(
    q: qm_c.Vector[qm_c.Qubit],
    i: int,
    j: int,
    theta: float,
) -> qm_c.Vector[qm_c.Qubit]:
    q[j], q[i] = qm_c.cx(q[j], q[i])
    q[i], q[j] = _controlled_ry(q[i], q[j], theta=-2.0 * theta)
    q[j], q[i] = qm_c.cx(q[j], q[i])
    return q


def _apply_givens_rotations(
    q: qm_c.Vector[qm_c.Qubit],
    givens_rotations: list[tuple[tuple[int, int], float]] | list[list],
) -> qm_c.Vector[qm_c.Qubit]:
    for (i, j), theta in givens_rotations:
        q = _apply_givens_rotation(q, i, j, theta)
    return q


def _apply_hopping_gate(
    q: qm_c.Vector[qm_c.Qubit],
    i: int,
    j: int,
    beta: qm_c.Float,
    hopping: float,
) -> qm_c.Vector[qm_c.Qubit]:
    q[i] = qm_c.rx(q[i], -0.5 * np.pi)
    q[j] = qm_c.rx(q[j], 0.5 * np.pi)
    q[i], q[j] = qm_c.cx(q[i], q[j])
    q[i] = qm_c.rx(q[i], -1.0 * beta * hopping)
    q[j] = qm_c.rz(q[j], beta * hopping)
    q[i], q[j] = qm_c.cx(q[i], q[j])
    q[i] = qm_c.rx(q[i], 0.5 * np.pi)
    q[j] = qm_c.rx(q[j], -0.5 * np.pi)
    return q


def _apply_mixer_layer(
    q: qm_c.Vector[qm_c.Qubit],
    beta: qm_c.Float,
    hopping: float,
    num_qubits: int,
) -> qm_c.Vector[qm_c.Qubit]:
    for i in range(0, num_qubits - 1, 2):
        q = _apply_hopping_gate(q, i, i + 1, beta, hopping)
    for i in range(1, num_qubits - 1, 2):
        q = _apply_hopping_gate(q, i, i + 1, beta, hopping)
    q = _apply_hopping_gate(q, 0, num_qubits - 1, beta, hopping)
    return q


def _apply_cost_layer(
    q: qm_c.Vector[qm_c.Qubit],
    gamma: qm_c.Float,
    linear: dict[int, float],
    quad: dict[tuple[int, int], float],
) -> qm_c.Vector[qm_c.Qubit]:
    for i, hi in linear.items():
        if not is_close_zero(hi):
            q[i] = qm_c.rz(q[i], 2 * hi * gamma)

    for (i, j), Jij in quad.items():
        if not is_close_zero(Jij):
            q[i], q[j] = qm_c.rzz(q[i], q[j], 2 * Jij * gamma)

    return q


def _apply_fqaoa_layers(
    q: qm_c.Vector[qm_c.Qubit],
    betas: qm_c.Vector[qm_c.Float],
    gammas: qm_c.Vector[qm_c.Float],
    p: int,
    linear: dict[int, float],
    quad: dict[tuple[int, int], float],
    hopping: float,
    num_qubits: int,
) -> qm_c.Vector[qm_c.Qubit]:
    for layer in range(p):
        q = _apply_cost_layer(q, gammas[layer], linear, quad)
        q = _apply_mixer_layer(q, betas[layer], hopping, num_qubits)
    return q


class FQAOAConverter(MathematicalProblemConverter):
    """
    FQAOA (Fermionic Quantum Approximate Optimization Algorithm) converter class.

    This class provides methods to convert optimization problems into FQAOA circuits,
    construct cost Hamiltonians, and decode quantum computation results.

    Examples:

    .. code::

        from qamomile.optimization.converters.fqaoa import FQAOAConverter

        # Initialize with a compiled optimization problem instance
        fqaoa_converter = FQAOAConverter(compiled_instance, num_fermion=4)

        # Generate QAOA circuit and cost Hamiltonian
        p = 2  # Number of QAOA layers
        fqaoa_kernel = fqaoa_converter.get_fqaoa_ansatz(p)
        cost_hamiltonian = fqaoa_converter.get_cost_hamiltonian()

    """

    def __init__(
        self,
        instance: ommx.v1.Instance,
        num_fermions: int,
        normalize_model: bool = False,
        normalize_ising: typ.Optional[typ.Literal["abs_max", "rms"]] = None,
    ):
        """
        Initialize the FQAOAConverter.

        This method initializes the converter with the compiled instance of the optimization problem.

        Args:
            compiled_instance: ommx.v1.Instance.
            num_fermions (int): Number of fermions. This means the constraint :math:`M = \\sum_{l,d} x_{l,d}`.
            normalize_model (bool): The objective function and the constraints are normalized using the maximum absolute value of the coefficients contained in each.\
                Defaults to False
            normalize_ising (Literal["abs_max", "rms"] | None): The normalization method for the Ising Hamiltonian. \
                Available options:
                - "abs_max": Normalize by absolute maximum value
                - "rms": Normalize by root mean square
                Defaults to None.

        """
        self.num_fermions = num_fermions
        self.normalize_model = normalize_model
        self.normalize_ising = normalize_ising
        self.int2varlabel: dict[int, str] = {}
        self._ising: typ.Optional[IsingModel] = None

        if isinstance(instance, ommx.v1.Instance) and instance.objective.degree() > 2:
            raise ValueError("FQAOAConverter supports only QUBO instances.")

        super().__init__(instance)

    def __post_init__(self) -> None:
        if self.instance is None:
            raise TypeError("FQAOAConverter requires an ommx.v1.Instance")

        if self.instance.objective.degree() > 2:
            raise ValueError("FQAOAConverter supports only QUBO instances.")

        self.original_instance = self.instance
        last_var = self.original_instance.decision_variables[-1]
        n, d = last_var.subscripts
        self.num_integers, self.num_bits = n + 1, d + 1
        self.var_map = self.cyclic_mapping()
        self.ising = self.fqaoa_get_ising()
        self.num_qubits = self.ising.num_bits

    def fqaoa_instance_to_qubo(self) -> tuple[dict[tuple[int, int], float], float]:
        """
        Convert the instance to QUBO format.

        This method converts the optimization problem instance into a QUBO (Quadratic Unconstrained Binary Optimization)
        representation, which is suitable for quantum computation.

        Returns:
            tuple[dict[int, float], float]: A tuple containing the QUBO dictionary and the constant term.

        """
        instance_copy = copy.deepcopy(self.original_instance)
        qubo, constant = instance_copy.to_qubo(uniform_penalty_weight=0.0)
        return qubo, constant

    def fqaoa_get_ising(self) -> IsingModel:
        """
        Get the Ising model representation of the problem.

        Returns:
            IsingModel: The Ising model representation.

        """
        if self._ising is None:
            self._ising = self.fqaoa_ising_encode()
        return self._ising

    def fqaoa_ising_encode(self) -> IsingModel:
        qubo, constant = self.fqaoa_instance_to_qubo()
        ising = IsingModel.from_qubo(qubo, simplify=False)
        ising.constant += constant

        # normalize
        if isinstance(self.normalize_ising, str):
            if self.normalize_ising == "abs_max":
                ising.normalize_by_abs_max()
            elif self.normalize_ising == "rms":
                ising.normalize_by_rms()
            else:
                raise ValueError(
                    f"Invalid value for normalize_ising: {self.normalize_ising}"
                )

        # index labeling
        for ising_index, qubo_index in ising.index_map.items():
            deci_var = self.original_instance.get_decision_variable_by_id(qubo_index)
            var_name = deci_var.name
            subscripts = tuple(deci_var.subscripts)

            fermionic_index = self.var_map[subscripts]
            self.int2varlabel[fermionic_index] = (
                var_name + "_{" + ",".join(map(str, subscripts)) + "}"
            )

        return ising

    def cyclic_mapping(self) -> dict[tuple[int, int], int]:
        """
        Get variable maps between decision variable indices :math:`(l,d)` and qubit index :math:`i`.

        Return:
                        dict[tuple[int, int], int] : A variable map for ring driver.
        """
        cyclic_var_map = {}
        for var in self.original_instance.decision_variables:
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

    def _givens_decomposition(self, fermi_orbital):
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
    def get_init_state(self) -> qm_c.QKernel:
        """Generate the initial state preparation kernel for FQAOA."""
        unitary_rows = self.get_fermi_orbital()
        givens_rotations = self._givens_decomposition(unitary_rows)
        num_fermions = self.num_fermions

        @qm_c.qkernel
        def init_state(q: qm_c.Vector[qm_c.Qubit]) -> qm_c.Vector[qm_c.Qubit]:
            q = _apply_initial_occupations(q, num_fermions)
            q = _apply_givens_rotations(q, givens_rotations)
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
            return _apply_mixer_layer(q, beta, hopping, num_qubits)

        return mixer

    def get_cost_ansatz(self) -> qm_c.QKernel:
        """Generate the cost ansatz kernel (:math:`e^{-\\gamma H_P}`)."""
        linear = self.ising.linear
        quad = self.ising.quad

        @qm_c.qkernel
        def cost(
            q: qm_c.Vector[qm_c.Qubit],
            gamma: qm_c.Float,
        ) -> qm_c.Vector[qm_c.Qubit]:
            return _apply_cost_layer(q, gamma, linear, quad)

        return cost

    def get_fqaoa_ansatz(self, p: int, hopping: float = 1.0) -> qm_c.QKernel:
        """Generate the FQAOA ansatz kernel."""
        num_qubits = self.num_qubits
        num_fermions = self.num_fermions
        unitary_rows = self.get_fermi_orbital()
        givens_rotations = self._givens_decomposition(unitary_rows)
        linear = self.ising.linear
        quad = self.ising.quad

        @qm_c.qkernel
        def fqaoa_state(
            betas: qm_c.Vector[qm_c.Float],
            gammas: qm_c.Vector[qm_c.Float],
        ) -> qm_c.Vector[qm_c.Qubit]:
            q = qm_c.qubit_array(num_qubits, name="q")
            q = _apply_initial_occupations(q, num_fermions)
            q = _apply_givens_rotations(q, givens_rotations)
            q = _apply_fqaoa_layers(
                q, betas, gammas, p, linear, quad, hopping, num_qubits
            )
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
        givens_rotations = self._givens_decomposition(unitary_rows)

        # Filter near-zero coefficients to keep loops compact
        linear = {
            i: hi for i, hi in self.ising.linear.items() if not is_close_zero(hi)
        }
        quad = {
            ij: Jij for ij, Jij in self.ising.quad.items() if not is_close_zero(Jij)
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
            q = _apply_givens_rotations(q, givens_rotations)

            for layer in range(p):
                # Cost layer
                for i, hi in qm_c.items(linear):
                    q[i] = qm_c.rz(q[i], 2 * hi * gammas[layer])
                for (i, j), Jij in qm_c.items(quad):
                    q[i], q[j] = qm_c.rzz(q[i], q[j], 2 * Jij * gammas[layer])

                # Mixer layer
                for i in qm_c.range(0, last_qubit, 2):
                    q = _apply_hopping_gate(q, i, i + 1, betas[layer], hopping)
                for i in qm_c.range(1, last_qubit, 2):
                    q = _apply_hopping_gate(q, i, i + 1, betas[layer], hopping)
                # Boundary hop (wrap in control flow to avoid segment splits)
                for _ in qm_c.range(1):
                    q = _apply_hopping_gate(q, 0, last_qubit, betas[layer], hopping)

            return qm_c.measure(q)

        return transpiler.transpile(
            fqaoa_sampling,
            bindings={"linear": linear, "quad": quad},
            parameters=["betas", "gammas"],
        )

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """
        Construct the cost Hamiltonian (:math:`H_P`) for FQAOA.

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian.
        """
        hamiltonian = qm_o.Hamiltonian()

        # Add linear terms
        for i, hi in self.ising.linear.items():
            if not is_close_zero(hi):
                hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, i),), hi)

        # Add quadratic terms
        for (i, j), Jij in self.ising.quad.items():
            if not is_close_zero(Jij):
                hamiltonian.add_term(
                    (
                        qm_o.PauliOperator(qm_o.Pauli.Z, i),
                        qm_o.PauliOperator(qm_o.Pauli.Z, j),
                    ),
                    Jij,
                )

        return hamiltonian
