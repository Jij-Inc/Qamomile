"""FQAOA (Fermionic QAOA) converter.

Translates constrained integer and fixed-Hamming-weight binary optimization
problems into FQAOA circuits. Integer variables are encoded into binary via
unary encoding, while native binary variables are consumed directly when their
linear equality constraint fixes the total fermion number.
"""

from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True)
class _FQAOAProblemData:
    """Store the normalized problem data consumed by FQAOAConverter.

    Args:
        instance (ommx.v1.Instance): Binary OMMX instance used for decoding
            and objective evaluation. Integer inputs are unary-encoded before
            reaching this field; native binary inputs are deep-copied as-is.
        binary_model (BinaryModel): Binary objective model built from
            ``instance``. It includes zero linear terms for every decision
            variable so qubits are allocated even for variables absent from
            the objective.
        num_fermions (int): Fixed particle number derived from the equality
            constraint structure.
        var_id_to_qubit (dict[int, int]): Mapping from OMMX decision variable
            IDs in ``instance`` to the sequential qubit indices used by
            ``binary_model``.
        layout (Literal["unary_integer", "native_binary"]): Origin of the
            normalized binary instance.
    """

    instance: ommx.v1.Instance
    binary_model: BinaryModel
    num_fermions: int
    var_id_to_qubit: dict[int, int]
    layout: Literal["unary_integer", "native_binary"]


def _prepare_fqaoa_problem(instance: ommx.v1.Instance) -> _FQAOAProblemData:
    """Normalize an OMMX instance into the binary form used by FQAOA.

    Args:
        instance (ommx.v1.Instance): Input optimization problem. All decision
            variables must be either integer variables or native binary
            variables.

    Returns:
        _FQAOAProblemData: Binary instance, binary model, fixed fermion count,
        and variable-to-qubit mapping used by :class:`FQAOAConverter`.

    Raises:
        ValueError: If the instance has no variables, mixes variable kinds, or
            uses a variable kind unsupported by FQAOA.
    """
    if not instance.decision_variables:
        raise ValueError("FQAOAConverter requires at least one decision variable.")

    if all(
        dv.kind == ommx.v1.DecisionVariable.INTEGER
        for dv in instance.decision_variables
    ):
        return _prepare_integer_problem(instance)
    if all(
        dv.kind == ommx.v1.DecisionVariable.BINARY for dv in instance.decision_variables
    ):
        return _prepare_native_binary_problem(instance)

    raise ValueError(
        "FQAOAConverter requires all decision variables to be all INTEGER "
        "or all BINARY; mixed or unsupported variable kinds are not supported."
    )


def _prepare_integer_problem(instance: ommx.v1.Instance) -> _FQAOAProblemData:
    """Encode an integer OMMX instance into unary binary FQAOA data.

    Args:
        instance (ommx.v1.Instance): Integer optimization problem with finite
            bounds and linear equality constraints.

    Returns:
        _FQAOAProblemData: Unary-encoded binary problem data.

    Raises:
        ValueError: If integer encoding rejects the instance or produces no
            binary decision variables, or if the encoded constraints are not a
            single full-register cardinality equality.
    """
    encoder = BinaryIntegerEncoder(instance, encoding="unary")
    encoded_instance, _ = encoder.encode()
    working = ommx.v1.Instance.from_bytes(encoded_instance.to_bytes())
    if not working.decision_variables:
        raise ValueError(
            "FQAOAConverter requires at least one binary variable after "
            "integer encoding."
        )

    constraint, linear = _require_single_linear_equality(working)
    variable_ids = {dv.id for dv in working.decision_variables}
    num_fermions = _derive_cardinality_rhs(
        linear, variable_ids, constraint.name or constraint.id
    )
    binary_model = _build_binary_model_from_instance(working)
    _require_minimum_fqaoa_qubits(binary_model)
    return _FQAOAProblemData(
        instance=working,
        binary_model=binary_model,
        num_fermions=num_fermions,
        var_id_to_qubit=_variable_id_to_qubit(binary_model, working),
        layout="unary_integer",
    )


def _prepare_native_binary_problem(instance: ommx.v1.Instance) -> _FQAOAProblemData:
    """Validate and normalize a native binary OMMX instance.

    Args:
        instance (ommx.v1.Instance): Binary optimization problem whose single
            equality constraint fixes the Hamming weight over all binary
            decision variables.

    Returns:
        _FQAOAProblemData: Native-binary problem data.

    Raises:
        ValueError: If constraints are not a single fixed-Hamming-weight
            equality over all binary decision variables.
    """
    working = ommx.v1.Instance.from_bytes(instance.to_bytes())
    num_fermions = _derive_native_binary_num_fermions(working)
    binary_model = _build_binary_model_from_instance(working)
    _require_minimum_fqaoa_qubits(binary_model)
    return _FQAOAProblemData(
        instance=working,
        binary_model=binary_model,
        num_fermions=num_fermions,
        var_id_to_qubit=_variable_id_to_qubit(binary_model, working),
        layout="native_binary",
    )


def _derive_native_binary_num_fermions(instance: ommx.v1.Instance) -> int:
    """Derive the fixed particle number from a binary equality constraint.

    The current FQAOA mixer conserves only the total Hamming weight across the
    full qubit register. Native binary instances are therefore accepted only
    when they contain exactly one linear equality equivalent to
    ``sum_i x_i == k`` over every decision variable.

    Args:
        instance (ommx.v1.Instance): Native binary OMMX instance.

    Returns:
        int: Fixed number of occupied orbitals.

    Raises:
        ValueError: If the constraint structure is not a single linear
            fixed-Hamming-weight equality, or if the derived count is outside
            ``[0, num_variables]``.
    """
    constraint, linear = _require_single_linear_equality(instance)
    variable_ids = {dv.id for dv in instance.decision_variables}
    return _derive_cardinality_rhs(
        linear, variable_ids, constraint.name or constraint.id
    )


def _require_minimum_fqaoa_qubits(binary_model: BinaryModel) -> None:
    """Validate that the FQAOA mixer has distinct hopping endpoints.

    Args:
        binary_model (BinaryModel): Binary model prepared for FQAOA.

    Raises:
        ValueError: If the model contains fewer than two binary variables.
    """
    if binary_model.num_bits < 2:
        raise ValueError(
            "FQAOAConverter requires at least two binary variables because "
            "the fermionic mixer applies hopping gates between distinct qubits."
        )


def _require_single_linear_equality(
    instance: ommx.v1.Instance,
) -> tuple[ommx.v1.Constraint, ommx.v1.Linear]:
    """Return the FQAOA constraint as a linear equality.

    Args:
        instance (ommx.v1.Instance): Binary OMMX instance whose constraints
            should encode a fixed Hamming weight.

    Returns:
        tuple[ommx.v1.Constraint, ommx.v1.Linear]: A tuple containing
        ``constraint`` (the single OMMX constraint) and ``linear`` (the
        constraint function viewed as a linear expression).

    Raises:
        ValueError: If the instance does not contain exactly one constraint,
            or if that constraint is not a linear equality.
    """
    if len(instance.constraints) != 1:
        raise ValueError(
            "FQAOA requires exactly one fixed-Hamming-weight equality constraint."
        )

    constraint = instance.constraints[0]
    name = constraint.name or constraint.id
    if constraint.equality != ommx.v1.Constraint.EQUAL_TO_ZERO:
        raise ValueError(
            f"Constraint '{name}' is not an equality constraint. FQAOA requires "
            "a fixed-Hamming-weight equality."
        )
    if constraint.function.degree() > 1:
        raise ValueError(
            f"Constraint '{name}' is not linear (degree "
            f"{constraint.function.degree()}). FQAOA requires a linear "
            "fixed-Hamming-weight equality."
        )

    linear = constraint.function.as_linear()
    if linear is None:
        raise ValueError(
            f"Constraint '{name}' is not linear. FQAOA requires a linear "
            "fixed-Hamming-weight equality."
        )

    return constraint, linear


def _derive_cardinality_rhs(
    linear: ommx.v1.Linear,
    variable_ids: set[int],
    constraint_name: str | int,
) -> int:
    """Derive the RHS of a full-register cardinality equality.

    Args:
        linear (ommx.v1.Linear): Linear constraint function in normalized
            zero form.
        variable_ids (set[int]): OMMX decision variable IDs that must appear
            in the cardinality constraint.
        constraint_name (str | int): Human-readable constraint identifier used
            in error messages.

    Returns:
        int: Fixed number of occupied orbitals implied by the constraint.

    Raises:
        ValueError: If the linear expression does not include every variable
            exactly once with the same non-zero coefficient, or if the derived
            fermion count is not an integer in the feasible range.
    """
    if not variable_ids:
        raise ValueError("FQAOA requires at least one decision variable.")

    constrained_ids = set(linear.linear_terms)
    if constrained_ids != variable_ids:
        raise ValueError(
            f"Constraint '{constraint_name}' must include all binary decision "
            "variables exactly once so the global FQAOA mixer preserves feasibility."
        )

    coefficients = [
        float(linear.linear_terms[var_id]) for var_id in sorted(variable_ids)
    ]
    first_coeff = coefficients[0]
    if is_close_zero(first_coeff) or any(
        not np.isclose(coeff, first_coeff) for coeff in coefficients
    ):
        raise ValueError(
            f"Constraint '{constraint_name}' must use the same non-zero coefficient for "
            "every binary decision variable."
        )

    num_fermions = -float(linear.constant_term) / first_coeff
    rounded = round(num_fermions)
    if not np.isclose(num_fermions, rounded) or rounded < 0:
        raise ValueError(
            f"Constraint '{constraint_name}' implies a non-integer or negative fermion "
            f"count ({num_fermions})."
        )
    if rounded > len(variable_ids):
        raise ValueError(
            f"Constraint '{constraint_name}' implies {rounded} fermions for "
            f"{len(variable_ids)} binary variables."
        )

    return int(rounded)


def _build_binary_model_from_instance(instance: ommx.v1.Instance) -> BinaryModel:
    """Build a BinaryModel while preserving every decision variable.

    Args:
        instance (ommx.v1.Instance): Binary OMMX instance whose objective is
            represented as HUBO terms.

    Returns:
        BinaryModel: Binary model with sequential internal indices. Variables
        absent from the objective are retained through zero linear terms.
    """
    hubo_dict: dict[tuple[int, ...], float] = {
        (dv.id,): 0.0 for dv in instance.decision_variables
    }
    for var_ids, coeff in instance.objective.terms.items():
        key = tuple(var_ids)
        hubo_dict[key] = hubo_dict.get(key, 0.0) + float(coeff)
    return BinaryModel.from_hubo(hubo_dict)


def _variable_id_to_qubit(
    binary_model: BinaryModel,
    instance: ommx.v1.Instance,
) -> dict[int, int]:
    """Build the OMMX variable-ID to qubit-index mapping.

    Args:
        binary_model (BinaryModel): Binary model used by FQAOA.
        instance (ommx.v1.Instance): Binary OMMX instance whose decision
            variables should all be present in ``binary_model``.

    Returns:
        dict[int, int]: Mapping from OMMX decision variable ID to sequential
        qubit index.

    Raises:
        KeyError: If a decision variable was not retained in the binary model.
    """
    return {
        dv.id: binary_model.index_origin_to_new[dv.id]
        for dv in instance.decision_variables
    }


class FQAOAConverter(MathematicalProblemConverter):
    """FQAOA (Fermionic Quantum Approximate Optimization Algorithm) converter.

    Accepts either an integer optimization problem where all decision variables
    are integer-typed with linear equality constraints, or a native binary
    problem with a single fixed-Hamming-weight equality over all variables.
    Integer inputs are unary-encoded via
    :class:`~qamomile.optimization.integer_encoding.BinaryIntegerEncoder`
    and native binary inputs are consumed directly. In both cases the converter
    derives ``num_fermions`` automatically from the constraint structure.

    Example:
        >>> converter = FQAOAConverter(instance)
        >>> hamiltonian = converter.get_cost_hamiltonian()
        >>> executable = converter.transpile(transpiler, p=2)
    """

    def __init__(
        self,
        instance: ommx.v1.Instance,
    ):
        """Initialize the FQAOAConverter.

        Args:
            instance (ommx.v1.Instance): The optimization problem. All
                decision variables must be integer variables or native binary
                variables. Integer variables require finite bounds and linear
                equality constraints. Native binary variables require exactly
                one linear equality that fixes the Hamming weight across all
                variables.

        Raises:
            ValueError: If the instance contains unsupported variable kinds,
                invalid integer encodings, or native binary constraints that
                are not compatible with the global FQAOA mixer.
        """
        problem_data = _prepare_fqaoa_problem(instance)
        self._problem_data = problem_data
        self._original_instance = problem_data.instance
        self.num_fermions = problem_data.num_fermions

        super().__init__(problem_data.binary_model)
        self.instance = problem_data.instance

    def __post_init__(self) -> None:
        """Populate FQAOA-specific dimensions and variable mappings."""
        self.input_layout = self._problem_data.layout
        self.var_id_to_qubit = self._problem_data.var_id_to_qubit.copy()
        self.num_qubits = self.spin_model.num_bits
        if self.input_layout == "unary_integer":
            positions = [
                tuple(var.subscripts)
                for var in self._original_instance.decision_variables
            ]
            self.num_integers = max(pos[0] for pos in positions) + 1
            self.num_bits = max(pos[1] for pos in positions) + 1
        else:
            self.num_integers = len(self._original_instance.decision_variables)
            self.num_bits = 1
        self.var_map = self.cyclic_mapping()

    def cyclic_mapping(self) -> dict[tuple[int, int], int]:
        """Build variable map between decision variable indices and qubit index.

        For unary-encoded integer inputs, maps ``(l, d)`` subscript pairs to
        the actual qubit indices used by the internal binary model. For native
        binary inputs, maps ``(variable_id, 0)`` pairs to those same qubit
        indices.

        Returns:
            dict[tuple[int, int], int]: Variable map for the ring driver.

        Raises:
            ValueError: If a unary-encoded variable does not carry the expected
            two-dimensional subscript metadata.
        """
        cyclic_var_map: dict[tuple[int, int], int] = {}
        if self.input_layout == "native_binary":
            for var in self._original_instance.decision_variables:
                cyclic_var_map[(var.id, 0)] = self.var_id_to_qubit[var.id]
            return cyclic_var_map

        for var in self._original_instance.decision_variables:
            pos = var.subscripts
            if len(pos) != 2:
                raise ValueError(
                    "Unary-encoded FQAOA variables must have two-dimensional "
                    f"subscripts; variable {var.id} has {pos}."
                )
            cyclic_var_map[(pos[0], pos[1])] = self.var_id_to_qubit[var.id]
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
                    orbital[k + 1, i] = np.sqrt(2.0 / self.num_qubits) * np.sin(angle)
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
