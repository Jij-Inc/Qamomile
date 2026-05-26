"""This module implements the Grover Adaptive Search (GAS) algorithm for Combinatorial Polynomial Binary Optimization (CPBO).
The quantum optimization algorithm iteratively applies Grover's search to find the minimum of the function. 
The GAS algorithm is designed to efficiently find the optimal solution by adaptively adjusting the search 
space based on previous iterations' results.
"""


import qamomile.circuit as qmc
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.executable import ExecutableProgram

from qamomile.circuit.algorithm.gas import grover_algorithm

from .converter import MathematicalProblemConverter

class GASConverter(MathematicalProblemConverter):
    """Converter for Grover Adaptive Search (GAS).
    """

    def get_cost_hamiltonian(self) -> None:
        """GAS does not use a cost Hamiltonian in the same way as QAOA, so this method is not implemented."""
        return None

    def transpile(self, transpiler: Transpiler, *, output_bits : int, y: int, num_iterations: int) -> ExecutableProgram:
        """
        Transpile the model into an executable QAOA circuit.

        Dispatches to the quadratic-only fast path when no higher-order terms
        are present, otherwise uses the HUBO path with phase-gadget
        decomposition.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            output_bits (int): The number of output bits to use in the circuit, which determines the size of the function domain.
            y (int): The value of the minimum known so far, used to define the oracle.
            num_iterations (int): The number of iterations to perform in the Grover algorithm.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """
        if not self.spin_model.higher:
            return self._transpile_quadratic(transpiler, output_bits=output_bits, y=y, num_iterations=num_iterations)
        return self._transpile_hubo(transpiler, output_bits=output_bits, y=y, num_iterations=num_iterations)
    
    def _transpile_quadratic(self, transpiler: Transpiler, *, output_bits: int, y: int, num_iterations : int) -> ExecutableProgram:
        """Transpile the model into an executable QAOA circuit using the quadratic-only fast path.

        Args:

            transpiler (Transpiler): Backend transpiler to use.
            output_bits (int): The number of output bits to use in the circuit, which determines the size of the function domain.
            y (int): The value of the minimum known so far, used to define the oracle.
            num_iterations (int): The number of iterations to perform in the Grover algorithm.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """

        @qmc.qkernel
        def measure_grover_algorithm(
            n: qmc.UInt,
            m: qmc.UInt,
            y: qmc.UInt,
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            iters: qmc.UInt = 1
        ) -> qmc.Vector[qmc.Bit]:
            
            q_output, q_input = grover_algorithm(
                n=n,
                m=m,
                y=y,
                linear=linear,
                quad=quad,
                iters=iters,
            )
            return qmc.measure(q_input)
        
        return transpiler.transpile(
            measure_grover_algorithm,
            bindings={
                "n": self.spin_model.num_bits,
                "m": output_bits,
                "y": y,
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "iters": num_iterations,
            }
        )