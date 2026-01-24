import qamomile.circuit as qmc
import ommx.v1
from qamomile.circuit.algorithm.qaoa import qaoa_state
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import ExecutableProgram, QuantumExecutor

from .converter import MathematicalProblemConverter


class QAOAConverter(MathematicalProblemConverter):

    def transpile(
        self, 
        transpiler: Transpiler,
        *,
        p: int
    ) -> ExecutableProgram:
        
        @qmc.qkernel
        def qaoa_sampling(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
            n: qmc.UInt,
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(
                p,
                quad,
                linear,
                n,
                gammas,
                betas,
            )
            return qmc.measure(q)

        executable = transpiler.transpile(
            qaoa_sampling,
            bindings={
                "linear": self.ising.linear,
                "quad": self.ising.quad,
                "n": self.ising.num_bits,
                "p": p
            },
            parameters=["gammas", "betas"],
        )
        return executable


    def decode(
        self,
        samples: list
    ) -> ommx.v1.Solution:
        pass