import qamomile.circuit as qmc
import qamomile.core.bitssample as qm_bs
import ommx.v1
from qamomile.circuit.algorithm.qaoa import qaoa_state
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import ExecutableProgram, QuantumExecutor
from qamomile.circuit.transpiler.job import SampleResult

from .converter import MathematicalProblemConverter


def _convert_sampleresult_to_bitssampleset(
    result: SampleResult[list[int]]
) -> qm_bs.BitsSampleSet:
    """Convert ExecutableProgram SampleResult to BitsSampleSet.

    This is a thin adapter that bridges the new circuit API with the
    old core converter decoding logic.

    Args:
        result: SampleResult from ExecutableProgram.sample()

    Returns:
        BitsSampleSet compatible with QuantumConverter.decode_bits_to_sampleset()
    """
    bitarrays = [
        qm_bs.BitsSample(bits=bitstring, num_occurrences=count)
        for bitstring, count in result.results
    ]
    return qm_bs.BitsSampleSet(bitarrays)


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
        samples: SampleResult[list[int]],
    ) -> ommx.v1.SampleSet:
        """Decode ExecutableProgram sample results to OMMX SampleSet.

        This method bridges the new circuit API with OMMX by:
        1. Converting SampleResult → BitsSampleSet
        2. Mapping Ising indices to OMMX decision variable IDs
        3. Evaluating samples to compute objective values

        Args:
            samples: SampleResult from ExecutableProgram.sample().result()

        Returns:
            ommx.v1.SampleSet with decoded solutions and objective values

        Example:
            >>> converter = QAOAConverter(instance)
            >>> executable = converter.transpile(transpiler, p=2)
            >>> job = executable.sample(executor, shots=1000, bindings={...})
            >>> result = job.result()
            >>> sampleset = converter.decode(result)
        """
        # Step 1: Convert SampleResult to BitsSampleSet
        bitssampleset = _convert_sampleresult_to_bitssampleset(samples)

        # Step 2: Map Ising indices to OMMX decision variable IDs
        ising = self.ising
        sample_id = 0
        ommx_samples = ommx.v1.Samples(entries=[])

        for bitssample in bitssampleset.bitarrays:
            sample = {}
            for i, bit in enumerate(bitssample.bits):
                # Map Ising index → OMMX decision variable ID
                index = ising.index_from_new[i]
                sample[index] = bit

            state = ommx.v1.State(entries=sample)

            # Handle num_occurrences (multiple identical samples)
            ids = []
            for _ in range(bitssample.num_occurrences):
                ids.append(sample_id)
                sample_id += 1
            ommx_samples.append(sample_ids=ids, state=state)

        # Step 3: Evaluate samples to get SampleSet with energies
        return self.instance.evaluate_samples(ommx_samples)
