import qamomile.circuit as qmc
import qamomile.core.bitssample as qm_bs
import ommx.v1
from qamomile.circuit.algorithm.qaoa import qaoa_state
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import ExecutableProgram, QuantumExecutor
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.observable.hamiltonian import Hamiltonian

from .binary_model import BinarySampleSet
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
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "n": self.spin_model.num_bits,
                "p": p
            },
            parameters=["gammas", "betas"],
        )
        return executable


    def decode(
        self,
        samples: SampleResult[list[int]],
    ) -> BinarySampleSet:
        """Decode quantum measurement results.

        Returns results in the original vartype (BINARY or SPIN) that was
        provided when constructing the converter.
        """
        from .binary_model import VarType

        # First decode in SPIN domain
        spin_sampleset = self.spin_model.decode_from_sampleresult(samples)

        # If original problem was BINARY, convert back
        if self.original_vartype == VarType.BINARY:
            binary_samples = []
            for spin_sample in spin_sampleset.samples:
                # Convert SPIN (±1) to BINARY (0/1): x = (1 - s) / 2
                binary_sample = {
                    idx: (1 - spin_val) // 2
                    for idx, spin_val in spin_sample.items()
                }
                binary_samples.append(binary_sample)

            return BinarySampleSet(
                samples=binary_samples,
                num_occurrences=spin_sampleset.num_occurrences,
                energy=spin_sampleset.energy,
                vartype=VarType.BINARY
            )
        else:
            # Already in SPIN, return as-is
            return spin_sampleset
