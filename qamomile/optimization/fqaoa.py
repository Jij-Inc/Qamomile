"""FQAOA (Fermionic QAOA) converter.

Translates constrained integer optimization problems into FQAOA circuits.
Integer variables are encoded into binary via unary encoding, and equality
constraints are incorporated as fermion number conservation.
"""

import numpy as np
import ommx.v1

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile._utils import is_close_zero
from qamomile.circuit.algorithm.fqaoa import (
    fqaoa_state,
    hubo_fqaoa_state,
)
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.optimization.binary_model import BinaryModel

from .converter import MathematicalProblemConverter
from .integer_encoding import BinaryIntegerEncoder


class FQAOAConverter(MathematicalProblemConverter):
    """FQAOA (Fermionic Quantum Approximate Optimization Algorithm) converter.

    Accepts an integer optimization problem where all decision variables
    are integer-typed with linear equality constraints.  The converter
    applies unary encoding via
    :class:`~qamomile.optimization.integer_encoding.BinaryIntegerEncoder`
    and derives ``num_fermions`` automatically from the constraint
    structure.

    Example:
        >>> converter = FQAOAConverter(integer_instance)
        >>> hamiltonian = converter.get_cost_hamiltonian()
        >>> executable = converter.transpile(transpiler, p=2)
    """

    def __init__(
        self,
        instance: ommx.v1.Instance,
    ):
        """Initialize the FQAOAConverter.

        Args:
            instance (ommx.v1.Instance): The integer optimization problem.
                All decision variables must be integer-typed with finite
                bounds.  All constraints must be linear equalities.

        Raises:
            ValueError: If the instance contains non-integer decision
                variables, non-equality constraints, or non-linear
                constraints.
        """
        encoder = BinaryIntegerEncoder(instance, encoding="unary")
        encoded_instance, constraint_rhs_total = encoder.encode()
        self.num_fermions = constraint_rhs_total

        self._original_instance = encoded_instance

        working = ommx.v1.Instance.from_bytes(encoded_instance.to_bytes())
        hubo_dict = {
            var_ids: coeff for var_ids, coeff in working.objective.terms.items()
        }
        binary_model = BinaryModel.from_hubo(hubo_dict)

        super().__init__(binary_model)
        self.instance = working

    def __post_init__(self) -> None:
        last_var = self._original_instance.decision_variables[-1]
        n, d = last_var.subscripts
        self.num_integers, self.num_bits = n + 1, d + 1
        self.var_map = self.cyclic_mapping()
        self.num_qubits = self.spin_model.num_bits

    def cyclic_mapping(self) -> dict[tuple[int, int], int]:
        """Build variable map between decision variable indices and qubit index.

        Maps ``(l, d)`` subscript pairs to qubit indices following
        the ring-driver layout ``l + num_integers * d``.

        Returns:
            dict[tuple[int, int], int]: Variable map for the ring driver.
        """
        cyclic_var_map: dict[tuple[int, int], int] = {}
        for var in self._original_instance.decision_variables:
            pos = var.subscripts
            cyclic_var_map[(pos[0], pos[1])] = pos[0] + self.num_integers * pos[1]
        return cyclic_var_map

    def get_fermi_orbital(self) -> np.ndarray:
        """Compute single-particle wave functions of occupied spin orbitals.

        Returns:
            numpy.ndarray: A 2D array of shape ``(num_fermions, num_qubits)``.
        """
        orbital = np.zeros((self.num_fermions, self.num_qubits))

        if self.num_fermions % 2 == 0:
            for i in range(self.num_qubits):
                for k in range(int(self.num_fermions / 2)):
                    angle = 2.0 * np.pi * (k + 0.5) * (i + 1) / self.num_qubits
                    orbital[k, i] = np.sqrt(2.0 / self.num_qubits) * np.sin(angle)
                    orbital[int(self.num_fermions - 1 - k), i] = np.sqrt(
                        2.0 / self.num_qubits
                    ) * np.cos(angle)
        else:
            for i in range(self.num_qubits):
                orbital[0, i] = np.sqrt(1.0 / self.num_qubits)
                for k in range(int(self.num_fermions / 2)):
                    angle = 2.0 * np.pi * (k + 1) * (i + 1) / self.num_qubits
                    orbital[k + 1, i] = np.sqrt(2.0 / self.num_qubits) * np.sin(
                        angle
                    )
                    orbital[int(self.num_fermions - 1 - k), i] = np.sqrt(
                        2.0 / self.num_qubits
                    ) * np.cos(angle)

        return orbital

    def _givens_decomposition(self, fermi_orbital: np.ndarray) -> list[list]:
        """Decompose the fermi orbital into Givens rotation angles.

        Args:
            fermi_orbital (numpy.ndarray): Orbital matrix of shape
                ``(num_fermions, num_qubits)``.

        Returns:
            list[list]: Each entry is ``[(i, j), angle]`` for one Givens
                rotation.
        """
        m, n = fermi_orbital.shape
        matrix = fermi_orbital.copy()

        for j in reversed(range(n - m, n)):
            for i in range(m - n + j):
                sin_ = -matrix[i, j] / np.sqrt(
                    matrix[i, j] ** 2 + matrix[i + 1, j] ** 2
                )
                cos_ = matrix[i + 1, j] / np.sqrt(
                    matrix[i, j] ** 2 + matrix[i + 1, j] ** 2
                )
                row_1 = matrix[i].copy()
                row_2 = matrix[i + 1].copy()
                matrix[i] = cos_ * row_1 + sin_ * row_2
                matrix[i + 1] = -sin_ * row_1 + cos_ * row_2

        givens_angles = []
        for i in range(m):
            for j in reversed(range(i, i + n - m)):
                cos_ = matrix[i, j] / np.sqrt(matrix[i, j] ** 2 + matrix[i, j + 1] ** 2)
                sin_ = -matrix[i, j + 1] / np.sqrt(
                    matrix[i, j] ** 2 + matrix[i, j + 1] ** 2
                )
                if sin_ >= 0:
                    angle = np.arccos(cos_)
                else:
                    angle = -np.arccos(cos_)

                givens_angles.append([(j, j + 1), angle])

                col_1 = matrix[:, j].copy()
                col_2 = matrix[:, j + 1].copy()
                matrix[:, j] = np.cos(angle) * col_1 - np.sin(angle) * col_2
                matrix[:, j + 1] = np.sin(angle) * col_1 + np.cos(angle) * col_2

        return givens_angles

    def _flatten_givens_data(
        self,
        givens_angles: list[list],
    ) -> tuple[np.ndarray, list[float]]:
        """Convert Givens decomposition output to a matrix and a vector.

        Args:
            givens_angles (list[list]): Output from
                :meth:`_givens_decomposition`.

        Returns:
            tuple[numpy.ndarray, list[float]]: ``(givens_ij, givens_theta)``
                where *givens_ij* is a ``(N, 2)`` uint array of qubit
                index pairs and *givens_theta* is a list of rotation
                angles.
        """
        indices_ij: list[list[int]] = []
        angles: list[float] = []
        for (i, j), theta in givens_angles:
            indices_ij.append([i, j])
            angles.append(float(theta))
        return np.array(indices_ij, dtype=np.uint64), angles

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Construct the Ising cost Hamiltonian from the spin model.

        Builds a Pauli-Z Hamiltonian from the spin model's linear, quadratic,
        and higher-order terms.

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

        for indices, coeff in self.spin_model.higher.items():
            if not is_close_zero(coeff):
                pauli_ops = tuple(
                    qm_o.PauliOperator(qm_o.Pauli.Z, idx) for idx in indices
                )
                hamiltonian.add_term(pauli_ops, coeff)

        hamiltonian.constant = self.spin_model.constant
        return hamiltonian

    def transpile(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        hopping: float = 1.0,
    ) -> ExecutableProgram:
        """Transpile the model into an executable FQAOA circuit.

        Dispatches to the quadratic-only fast path when no higher-order terms
        are present, otherwise uses the HUBO path with phase-gadget
        decomposition.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of FQAOA layers.
            hopping (float): Hopping parameter for the fermionic mixer.
                Defaults to 1.0.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """
        if not self.spin_model.higher:
            return self._transpile_quadratic(transpiler, p=p, hopping=hopping)
        return self._transpile_hubo(transpiler, p=p, hopping=hopping)

    def _transpile_quadratic(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        hopping: float,
    ) -> ExecutableProgram:
        """Transpile a quadratic-only model using the standard FQAOA circuit.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of FQAOA layers.
            hopping (float): Hopping parameter for the fermionic mixer.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """
        unitary_rows = self.get_fermi_orbital()
        givens_data = self._givens_decomposition(unitary_rows)
        givens_ij, gtheta = self._flatten_givens_data(givens_data)

        linear = {
            i: hi for i, hi in self.spin_model.linear.items() if not is_close_zero(hi)
        }
        quad = {
            ij: Jij
            for ij, Jij in self.spin_model.quad.items()
            if not is_close_zero(Jij)
        }

        # NOTE: @qkernel is defined inline (not at module level) because
        # transpile() binds instance-specific data at call time.
        @qmc.qkernel
        def fqaoa_sampling(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            n: qmc.UInt,
            num_fermions: qmc.UInt,
            givens_ij: qmc.Matrix[qmc.UInt],
            givens_theta: qmc.Vector[qmc.Float],
            hopping: qmc.Float,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = fqaoa_state(
                p=p,
                quad=quad,
                linear=linear,
                n=n,
                num_fermions=num_fermions,
                givens_ij=givens_ij,
                givens_theta=givens_theta,
                hopping=hopping,
                gammas=gammas,
                betas=betas,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            fqaoa_sampling,
            bindings={
                "p": p,
                "quad": quad,
                "linear": linear,
                "n": self.num_qubits,
                "num_fermions": self.num_fermions,
                "givens_ij": givens_ij,
                "givens_theta": gtheta,
                "hopping": hopping,
            },
            parameters=["gammas", "betas"],
        )

    def _transpile_hubo(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        hopping: float,
    ) -> ExecutableProgram:
        """Transpile a model with higher-order terms using phase-gadget decomposition.

        Decomposes k-body Z-rotation terms into phase gadgets,
        while reusing the standard ``cost_layer`` for quadratic and linear
        terms.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of FQAOA layers.
            hopping (float): Hopping parameter for the fermionic mixer.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """
        unitary_rows = self.get_fermi_orbital()
        givens_data = self._givens_decomposition(unitary_rows)
        givens_ij, gtheta = self._flatten_givens_data(givens_data)

        linear = {
            i: hi for i, hi in self.spin_model.linear.items() if not is_close_zero(hi)
        }
        quad = {
            ij: Jij
            for ij, Jij in self.spin_model.quad.items()
            if not is_close_zero(Jij)
        }

        # NOTE: @qkernel is defined inline (not at module level) because
        # transpile() binds instance-specific data at call time.
        @qmc.qkernel
        def fqaoa_sampling_hubo(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
            n: qmc.UInt,
            num_fermions: qmc.UInt,
            givens_ij: qmc.Matrix[qmc.UInt],
            givens_theta: qmc.Vector[qmc.Float],
            hopping: qmc.Float,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = hubo_fqaoa_state(
                p=p,
                quad=quad,
                linear=linear,
                higher=higher,
                n=n,
                num_fermions=num_fermions,
                givens_ij=givens_ij,
                givens_theta=givens_theta,
                hopping=hopping,
                gammas=gammas,
                betas=betas,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            fqaoa_sampling_hubo,
            bindings={
                "p": p,
                "quad": quad,
                "linear": linear,
                "higher": self.spin_model.higher,
                "n": self.num_qubits,
                "num_fermions": self.num_fermions,
                "givens_ij": givens_ij,
                "givens_theta": gtheta,
                "hopping": hopping,
            },
            parameters=["gammas", "betas"],
        )
