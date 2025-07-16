import jijmodeling as jm
import ommx.v1
import qamomile.core.operator as qm_o


class HamiltonianExpr:
    """Hamiltonian expression class.

    Attributes:
        hamiltonian : Hamiltonian expression.
        name (str) : Name of the Hamiltonian expression.

    .. code::

        N = jm.Placeholder("N")
        Z = PauliExpr.z(shape=(N,))
        i = jm.Element("i", belong_to = (0,N - 1))
        h_expr = HamiltonianExpr(jm.sum(i, Z[i] * Z[i+1]))

    """

    def __init__(self, expr, name: str = ""):
        self.hamiltonian = expr
        self.name = name

    def _repr_latex_(self):
        return self.hamiltonian._repr_latex_()


class HamiltonianBuilder:
    """Hamiltonian builder class.

    Attributes:
        hamiltonian_expr (HamiltonianExpr): Hamiltonian expression.
        instance_data (dict): Instance data.

    Example:
        >>> Z = PauliExpr.z(shape=(3,))
        >>> i = jm.Element("i", belong_to = (0, 3))
        >>> h_expr = HamiltonianExpr(jm.sum(i, Z[i]))
        >>> builder = HamiltonianBuilder(h_expr,{})
        >>> h = builder.build()
        Hamiltonian((Z0,): 1.0, (Z1,): 1.0, (Z2,): 1.0)

    """

    def __init__(self, hamiltonian_expr: HamiltonianExpr, instance_data: dict):
        self.hamiltonian_expr = hamiltonian_expr
        self.instance_data = instance_data

    def make_substituted_hamiltonian(self):
        """Make substituted Hamiltonian.

        This method substitutes the Hamiltonian expression with the given instance data.

        """
        interpreter = jm.Interpreter(self.instance_data)
        problem = jm.Problem("Hamiltonian")
        problem += self.hamiltonian_expr.hamiltonian
        subsituted_expr: ommx.v1.Instance = interpreter.eval_problem(problem)

        self._subsituted_expr: ommx.v1.Instance = subsituted_expr

        var_map = {}
        for dc in subsituted_expr.decision_variables:
            if dc.name not in var_map:
                var_map[dc.name] = {}
            var_map[dc.name][tuple(dc.subscripts)] = dc.id

        self._var_map: dict[str, dict[tuple[int, ...], int]] = var_map

    def _pauli_str_to_pauli_type(self, pauli_str: str) -> qm_o.Pauli:
        """Convert string Pauli name to Qamomile Pauli type.

        Args:
            pauli_str (str): string Pauli name which is "_PauliX", "_PauliY", or "_PauliZ".

        Raises:
            ValueError: If the input string is not "_PauliX", "_PauliY", or "_PauliZ".

        Returns:
            qm_o.Pauli: Qamomile Pauli type.
        """
        if pauli_str == "_PauliX":
            return qm_o.Pauli.X
        elif pauli_str == "_PauliY":
            return qm_o.Pauli.Y
        elif pauli_str == "_PauliZ":
            return qm_o.Pauli.Z
        else:
            raise ValueError("Invalid Pauli string")

    def make_qubit_index_mapper(self):
        """Make qubit index mapper.

        This method creates a mapper from the operator index to the quantum bit index.

        """
        qubit_index_map = {}
        qubit_index = 0
        for pauli_op, indices_map in self._var_map.items():
            for indices, _ in indices_map.items():
                if indices not in qubit_index_map.keys():
                    qubit_index_map[indices] = qubit_index
                    qubit_index += 1

        self._qubit_index_map = qubit_index_map

    def make_reverse_var_map(self):
        """Make mapper from the index to the Pauli operator."""
        reverse_var_map = {}
        self.make_qubit_index_mapper()
        for pauli_op, indices_map in self._var_map.items():
            pauli_type = self._pauli_str_to_pauli_type(pauli_op)
            for indices, var in indices_map.items():
                qubit_index = self._qubit_index_map[indices]
                reverse_var_map[var] = qm_o.PauliOperator(pauli_type, qubit_index)
        self._reverse_var_map: dict[int, qm_o.PauliOperator] = reverse_var_map

    def make_hamiltonian_operator(self) -> qm_o.Hamiltonian:
        """Make Qamomile Hamiltonian operator.

        Returns:
            qm_o.Hamiltonian: Qamomile Hamiltonian operator.
        """
        hamiltonian = qm_o.Hamiltonian()
        hamiltonian.constant = self._subsituted_expr.objective.constant_term

        # Handle linear terms
        for var_id, coeff in self._subsituted_expr.objective.linear_terms.items():
            operators = tuple([self._reverse_var_map[var_id]])
            hamiltonian.add_term(operators, coeff)

        # Handle quadratic terms
        for (var_id1, var_id2), coeff in self._subsituted_expr.objective.quadratic_terms.items():
            operators = tuple([self._reverse_var_map[var_id1], self._reverse_var_map[var_id2]])
            hamiltonian.add_term(operators, coeff)

        return hamiltonian

    def build(self) -> qm_o.Hamiltonian:
        """Build Qamomile Hamiltonian operator.

        Returns:
            qm_o.Hamiltonian: Qamomile Hamiltonian operator.
        """
        self.make_substituted_hamiltonian()
        self.make_reverse_var_map()

        return self.make_hamiltonian_operator()
