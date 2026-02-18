import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.qaoa import (
    apply_phase_gadget,
    ising_cost_circuit,
    qaoa_state,
    superposition_vector,
    x_mixier_circuit,
)
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import ExecutableProgram

from .converter import MathematicalProblemConverter
from .utils import is_close_zero


class QAOAConverter(MathematicalProblemConverter):
    """Converter for Quantum Approximate Optimization Algorithm (QAOA).

    Supports both standard quadratic (QUBO/Ising) models and higher-order
    binary optimization (HUBO) models. When higher-order terms are present,
    automatically uses phase-gadget decomposition for k-body Z-rotations.

    Example:
        >>> model = BinaryModel.from_hubo({(0, 1, 2): 1.0, (0,): -2.0})
        >>> converter = QAOAConverter(model)
        >>> executable = converter.transpile(QiskitTranspiler(), p=2)
    """

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

    def transpile(self, transpiler: Transpiler, *, p: int) -> ExecutableProgram:
        """Transpile the model into an executable QAOA circuit.

        Dispatches to the quadratic-only fast path when no higher-order terms
        are present, otherwise uses the HUBO path with phase-gadget
        decomposition.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of QAOA layers.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """
        if not self.spin_model.higher:
            return self._transpile_quadratic(transpiler, p=p)
        return self._transpile_hubo(transpiler, p=p)

    def _transpile_quadratic(
        self,
        transpiler: Transpiler,
        *,
        p: int,
    ) -> ExecutableProgram:
        """Transpile a quadratic-only model using the standard QAOA circuit.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of QAOA layers.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """

        # NOTE: @qkernel is defined inline (not at module level) because
        # transpile() binds instance-specific data at call time.
        @qmc.qkernel
        def qaoa_sampling(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
            n: qmc.UInt,
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(p, quad, linear, n, gammas, betas)
            return qmc.measure(q)

        return transpiler.transpile(
            qaoa_sampling,
            bindings={
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "n": self.spin_model.num_bits,
                "p": p,
            },
            parameters=["gammas", "betas"],
        )

    def _transpile_hubo(
        self,
        transpiler: Transpiler,
        *,
        p: int,
    ) -> ExecutableProgram:
        """Transpile a model with higher-order terms using phase-gadget decomposition.

        Decomposes k-body Z-rotation terms into CX ladder + RZ primitives
        via ``apply_phase_gadget``, while reusing the standard
        ``ising_cost_circuit`` for quadratic and linear terms.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of QAOA layers.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """
        # NOTE: @qkernel functions below are defined inline (not at module level)
        # because they capture `higher_terms` via the `_apply_higher` closure.
        higher_terms = sorted(self.spin_model.higher.items())

        def _apply_higher(
            q: qmc.Vector[qmc.Qubit], gamma: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            for indices, coeff in higher_terms:
                q = apply_phase_gadget(q, list(indices), coeff * gamma)
            return q

        @qmc.qkernel
        def ising_cost_circuit_hubo(
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            q: qmc.Vector[qmc.Qubit],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            q = ising_cost_circuit(quad, linear, q, gamma)
            q = _apply_higher(q, gamma)
            return q

        @qmc.qkernel
        def qaoa_circuit_hubo(
            p_val: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            q: qmc.Vector[qmc.Qubit],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Qubit]:
            for layer in qmc.range(p_val):
                q = ising_cost_circuit_hubo(quad, linear, q, gammas[layer])
                q = x_mixier_circuit(q, betas[layer])
            return q

        @qmc.qkernel
        def qaoa_sampling_hubo(
            p_val: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
            n: qmc.UInt,
        ) -> qmc.Vector[qmc.Bit]:
            q = superposition_vector(n)
            q = qaoa_circuit_hubo(p_val, quad, linear, q, gammas, betas)
            return qmc.measure(q)

        return transpiler.transpile(
            qaoa_sampling_hubo,
            bindings={
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "n": self.spin_model.num_bits,
                "p_val": p,
            },
            parameters=["gammas", "betas"],
        )
