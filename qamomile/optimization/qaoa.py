import qamomile.circuit as qmc
import qamomile.core.bitssample as qm_bs
import ommx.v1
from qamomile.circuit.algorithm.qaoa import qaoa_state
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import ExecutableProgram, QuantumExecutor
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.observable.hamiltonian import Hamiltonian

from .binary_model import BinarySampleSet
from .converters.converter import MathematicalProblemConverter

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
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "n": self.spin_model.num_bits,
                "p": p
            },
            parameters=["gammas", "betas"],
        )
        return executable
