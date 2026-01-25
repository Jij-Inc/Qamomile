import abc

import ommx.v1

import qamomile.observable as qm_o
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinaryModel, VarType, BinarySampleSet


class MathematicalProblemConverter(abc.ABC):

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

    @abc.abstractmethod
    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        raise NotImplementedError()

    # @abc.abstractmethod
    # def transpile(
    #     self, 
    #     transpiler: Transpiler,
    #     *,
    #     p: int
    # ) -> ExecutableProgram:
    #     raise NotImplementedError()

    # @abc.abstractmethod
    # def decode(
    #     self,
    #     samples: SampleResult[list[int]]
    # ) -> BinarySampleSet:
    #     raise NotImplementedError()

    # def decode_to_samples_without_eval(
    #     self,
    #     binary_sampleset: BinarySampleSet,
    # ) -> ommx.v1.Samples:
    #     samples = ommx.v1.Samples({})
    #     sample_id = 0

    #     _samples = binary_sampleset.samples
    #     _num_occurrences = binary_sampleset.num_occurrences
    #     index_map = self.spin_model.index_new_to_origin
    #     for sample, num_occurrences in zip(_samples, _num_occurrences):
    #         entries = {index_map[var]: bit for var, bit in sample.items()}
    #         state = ommx.v1.State(entries)

    #         # Generate sample IDs for each occurrence
    #         ids = []
    #         for _ in range(num_occurrences):
    #             ids.append(sample_id)
    #             sample_id += 1
    #         samples.append(ids, state)
    
    #     return samples

    # def decode_to_sampleset(
    #     self,
    #     result: SampleResult[list[int]],
    # ) -> ommx.v1.SampleSet:
    #     """
    #     """
    #     binary_sampleset = self.decode(result)
    #     samples = self.decode_to_samples_without_eval(binary_sampleset)
    #     sample_set = self.instance.evaluate_samples(samples)
    #     return sample_set

