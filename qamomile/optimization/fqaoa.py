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

import typing as typ

import numpy as np
import ommx.v1

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile._utils import is_close_zero
from qamomile.circuit.algorithm.fqaoa import (
    fqaoa_state,
)
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.converter import MathematicalProblemConverter


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

        # Generate cost Hamiltonian and transpile
        cost_hamiltonian = fqaoa_converter.get_cost_hamiltonian()
        executable = fqaoa_converter.transpile(transpiler, p=2)

    """

    def __init__(
        self,
        instance: ommx.v1.Instance,
        num_fermions: int,
        normalize_ising: typ.Optional[typ.Literal["abs_max", "rms"]] = None,
        *,
        uniform_penalty_weight: float | None = None,
        penalty_weights: dict[int, float] | None = None,
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
            uniform_penalty_weight (float | None): Uniform penalty for OMMX
                constraints. The fermion-number constraint is already
                satisfied by construction, while all other constraints remain
                represented in the cost Hamiltonian. ``None`` delegates
                weight selection to OMMX.
            penalty_weights (dict[int, float] | None): Optional per-constraint
                penalty weights keyed by constraint ID.

        """
        if isinstance(num_fermions, bool) or not isinstance(num_fermions, int):
            raise TypeError("num_fermions must be an integer")
        if num_fermions < 0:
            raise ValueError("num_fermions must be nonnegative")
        if instance.objective.degree() > 2:
            raise ValueError("FQAOAConverter supports only QUBO instances.")

        self._validate_particle_number_constraints(instance, num_fermions)

        self.num_fermions = num_fermions
        self.normalize_ising = normalize_ising

        # Deep-copy via bytes round-trip before to_qubo: to_qubo mutates the
        # instance it is called on (absorbs constraints into the objective).
        # Run the mutation on a throwaway copy so the caller's instance is
        # left untouched, and store the copy for evaluate_samples — it still
        # carries original-constraint metadata for feasibility reporting.
        original = ommx.v1.Instance.from_bytes(instance.to_bytes())
        working = ommx.v1.Instance.from_bytes(instance.to_bytes())
        qubo, constant = working.to_qubo(
            uniform_penalty_weight=uniform_penalty_weight,
            penalty_weights=dict(penalty_weights or {}),
        )
        binary_model = BinaryModel.from_qubo(qubo, constant)

        # Keep the caller's original (untouched) instance for cyclic_mapping
        # and index labeling — its decision_variables list is the user-facing
        # one, not the post-qubo working copy.
        self._original_instance = original

        # Pass the BinaryModel to the base class (which converts to SPIN internally)
        super().__init__(binary_model)
        # Base class set self.instance = None because we passed a BinaryModel.
        # Wire in the post-qubo working copy so decode() can round-trip back
        # to ommx.v1.SampleSet (feasibility evaluated against the user's
        # original constraints, which OMMX retains internally).
        self.instance = working
        self.original_instance = original

    @staticmethod
    def _validate_particle_number_constraints(
        instance: ommx.v1.Instance,
        num_fermions: int,
    ) -> None:
        """Reject cardinality constraints inconsistent with the FQAOA sector.

        Args:
            instance (ommx.v1.Instance): Original optimization instance.
            num_fermions (int): Particle-number sector prepared by FQAOA.

        Raises:
            ValueError: If an equality constraint is exactly the sum of all
                decision variables but its right-hand side differs from
                ``num_fermions``.
        """
        variable_ids = {variable.id for variable in instance.decision_variables}
        for constraint in instance.constraints:
            function = constraint.function
            if (
                constraint.equality != ommx.v1.Constraint.EQUAL_TO_ZERO
                or function.degree() > 1
                or set(function.linear_terms) != variable_ids
            ):
                continue
            coefficients = list(function.linear_terms.values())
            if not coefficients:
                continue
            coefficient = coefficients[0]
            if not np.isclose(abs(coefficient), 1.0) or not all(
                np.isclose(item, coefficient) for item in coefficients
            ):
                continue
            cardinality = -function.constant_term / coefficient
            if not np.isclose(cardinality, num_fermions):
                raise ValueError(
                    "num_fermions is inconsistent with cardinality constraint "
                    f"{constraint.id}: expected {cardinality:g}, got "
                    f"{num_fermions}."
                )

    def __post_init__(self) -> None:
        positions: list[tuple[int, int]] = []
        for variable in self._original_instance.decision_variables:
            if len(variable.subscripts) != 2:
                raise ValueError(
                    "FQAOA variables must each have exactly two integer "
                    "subscripts (site, orbital)"
                )
            site, orbital = variable.subscripts
            if site < 0 or orbital < 0:
                raise ValueError("FQAOA variable subscripts must be nonnegative")
            positions.append((site, orbital))
        if not positions:
            raise ValueError("FQAOA requires at least one decision variable")
        self.num_integers = max(site for site, _ in positions) + 1
        self.num_bits = max(orbital for _, orbital in positions) + 1
        expected_positions = {
            (site, orbital)
            for site in range(self.num_integers)
            for orbital in range(self.num_bits)
        }
        if set(positions) != expected_positions or len(positions) != len(
            expected_positions
        ):
            raise ValueError(
                "FQAOA variables must form a unique, dense rectangular "
                "grid of (site, orbital) subscripts"
            )
        self.var_map = self.cyclic_mapping()
        self.num_qubits = self.spin_model.num_bits
        if self.num_fermions > self.num_qubits:
            raise ValueError("num_fermions cannot exceed the number of encoded qubits")

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
        cyclic_var_map: dict[tuple[int, int], int] = {}
        for var in self._original_instance.decision_variables:
            # l = pos[0], d = pos[1]
            pos = var.subscripts
            cyclic_var_map[(pos[0], pos[1])] = pos[0] + self.num_integers * pos[1]

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
                    orbital[k + 1, i] = np.sqrt(2.0 / self.num_qubits) * np.sin(angle)
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

    def _flatten_givens_data(
        self,
        givens_angles: list[list],
    ) -> tuple[np.ndarray, list[float]]:
        """Convert givens decomposition output to a matrix and a vector.

        Returns:
            Tuple of ``(givens_ij, givens_theta)`` where *givens_ij* is a
            ``(N, 2)`` uint array of qubit index pairs and *givens_theta* is
            a list of rotation angles.
        """
        indices_ij: list[list[int]] = []
        angles: list[float] = []
        # ``_givens_decomposition`` records the rotations that eliminate the
        # orbital matrix.  State preparation applies the inverse product, so
        # the recorded rotations must be replayed in reverse order.  Each
        # elementary circuit rotation already has the inverse sign convention.
        for (i, j), theta in reversed(givens_angles):
            indices_ij.append([i, j])
            angles.append(float(theta))
        return np.array(indices_ij, dtype=np.uint64), angles

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

    def transpile(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        hopping: float = 1.0,
    ) -> ExecutableProgram:
        """Compile FQAOA ansatz into an executable program with measurements."""
        unitary_rows = self.get_fermi_orbital()
        givens_data = self._givens_decomposition(unitary_rows)
        givens_ij, gtheta = self._flatten_givens_data(givens_data)

        # Filter near-zero coefficients to keep loops compact
        linear = {
            i: hi for i, hi in self.spin_model.linear.items() if not is_close_zero(hi)
        }
        quad = {
            ij: Jij
            for ij, Jij in self.spin_model.quad.items()
            if not is_close_zero(Jij)
        }

        @qmc.qkernel
        def fqaoa_sampling(
            p: qmc.UInt,
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            num_qubits: qmc.UInt,
            num_fermions: qmc.UInt,
            givens_ij: qmc.Matrix[qmc.UInt],
            givens_theta: qmc.Vector[qmc.Float],
            hopping: qmc.Float,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = fqaoa_state(
                p,
                linear,
                quad,
                num_qubits,
                num_fermions,
                givens_ij,
                givens_theta,
                hopping,
                gammas,
                betas,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            fqaoa_sampling,
            bindings={
                "p": p,
                "linear": linear,
                "quad": quad,
                "num_qubits": self.num_qubits,
                "num_fermions": self.num_fermions,
                "givens_ij": givens_ij,
                "givens_theta": gtheta,
                "hopping": hopping,
            },
            parameters=["gammas", "betas"],
        )
