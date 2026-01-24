import abc

import ommx.v1

from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.job import SampleResult
from .ising_qubo import IsingModel


class MathematicalProblemConverter(abc.ABC):

    def __init__(self, instance: ommx.v1.Instance) -> None:
        self.instance = instance
        qubo, constant = instance.to_qubo()
        self.ising: IsingModel = IsingModel.from_qubo(qubo, constant)

    @abc.abstractmethod
    def transpile(
        self, 
        transpiler: Transpiler,
        *,
        p: int
    ) -> ExecutableProgram:
        raise NotImplementedError()


    @abc.abstractmethod
    def decode(
        self,
        samples: SampleResult[list[int]]
    ) -> ommx.v1.SampleSet:
        raise NotImplementedError()

