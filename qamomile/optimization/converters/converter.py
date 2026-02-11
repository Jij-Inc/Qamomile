import ommx.v1

import qamomile.observable as qm_o
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinaryModel, VarType, BinarySampleSet
from qamomile.optimization.utils import is_close_zero


class MathematicalProblemConverter:

    def __init__(
        self,
        instance: ommx.v1.Instance | BinaryModel,
    ) -> None:
        if isinstance(instance, BinaryModel):
            self.instance = None
            self.original_vartype = instance.vartype
            self.spin_model = instance.change_vartype(VarType.SPIN)
        elif isinstance(instance, ommx.v1.Instance):
            self.instance = instance
            self.original_vartype = VarType.BINARY  # OMMX uses BINARY
            qubo, constant = instance.to_qubo()
            self.spin_model = BinaryModel.from_qubo(qubo, constant).change_vartype(VarType.SPIN)
        else:
            raise TypeError("instance must be ommx.v1.Instance or BinaryModel")

        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Construct the cost Hamiltonian from the spin model.

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

    def decode(
        self,
        samples: SampleResult[list[int]],
    ) -> BinarySampleSet:
        """Decode quantum measurement results.

        Returns results in the original vartype (BINARY or SPIN) that was
        provided when constructing the converter.
        """
        # First decode in SPIN domain
        spin_sampleset = self.spin_model.decode_from_sampleresult(samples)

        # If original problem was BINARY, convert back
        if self.original_vartype == VarType.BINARY:
            binary_samples = []
            for spin_sample in spin_sampleset.samples:
                # Convert SPIN (+/-1) to BINARY (0/1): x = (1 - s) / 2
                binary_sample = {
                    idx: (1 - spin_val) // 2
                    for idx, spin_val in spin_sample.items()
                }
                binary_samples.append(binary_sample)

            return BinarySampleSet(
                samples=binary_samples,
                num_occurrences=spin_sampleset.num_occurrences,
                energy=spin_sampleset.energy,
                vartype=VarType.BINARY,
            )
        else:
            # Already in SPIN, return as-is
            return spin_sampleset

