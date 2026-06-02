"""This module implements the Grover Adaptive Search (GAS) algorithm for Combinatorial Polynomial Binary Optimization (CPBO).

The quantum optimization algorithm iteratively applies Grover's search to find the minimum of the function.
The GAS algorithm is designed to efficiently find the optimal solution by adaptively adjusting the search
space based on previous iterations' results.
"""

import math

import numpy as np

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.gas import (
    grover_algorithm,
    zero_degree_qft_encoding,
)
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.transpiler import Transpiler

from .binary_model import VarType
from .converter import MathematicalProblemConverter


class GASConverter(MathematicalProblemConverter):
    """Converter for Grover Adaptive Search (GAS).

    Internally maintains a BINARY-domain model derived from ``spin_model``
    so that the Grover QFT-arithmetic circuit receives the correct QUBO
    coefficients (binary variables take values in {0, 1}, not ±1).
    """

    def __post_init__(self) -> None:
        """Derive and cache the BINARY model from the parent spin model."""
        self.binary_model = self.spin_model.change_vartype(VarType.BINARY)

    @staticmethod
    def _required_output_bits(binary_model) -> int:
        """Compute the maximum output-register size for the Grover QFT circuit.

        The register holds ``f(x) − y`` in two's complement.  The extremes of
        that quantity are ``±(f_max − f_min)``, so we need
        ``2^(m−1) > f_max − f_min``.

        Args:
            binary_model (BinaryModel): Binary model (QUBO or HUBO) whose
                coefficients define the objective range.

        Returns:
            int: Minimum number of output qubits ``m``.

        """
        all_coeffs = (
            list(binary_model.linear.values())
            + list(binary_model.quad.values())
            + list(binary_model.higher.values())
        )
        f_max = binary_model.constant + sum(c for c in all_coeffs if c > 0)
        f_min = binary_model.constant + sum(c for c in all_coeffs if c < 0)
        half_range = f_max - f_min
        if half_range <= 0:
            return 2
        return max(2, int(math.floor(math.log2(half_range))) + 2)

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian | None:
        """Raise NotImplementedError because GAS is oracle-based and has no cost Hamiltonian.

        GAS marks states via an oracle reflection rather than minimizing the
        expectation value of a cost Hamiltonian. This method always raises so
        that incorrect usage fails immediately rather than propagating a silent
        ``None`` that would cause a confusing error later.

        Returns:
            qm_o.Hamiltonian | None: Never returns; declared for base-class
            compatibility only.

        Raises:
            NotImplementedError: Always. GAS exposes no cost Hamiltonian.

        """
        raise NotImplementedError(
            "GASConverter does not expose a cost Hamiltonian. "
            "GAS is oracle-based and does not use Hamiltonian expectation values."
        )

    def transpile(
        self,
        transpiler: Transpiler,
        *,
        output_bits: int | None = None,
        y: int,
        num_iterations: int,
    ) -> ExecutableProgram:
        """Transpile the model into an executable Grover circuit.

        Dispatches to the quadratic-only path, with build-in function, when no
        higher-order terms are present. Otherwise uses the HUBO path with the
        qkernel factory.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            output_bits (int | None): Number of output qubits for the function output domain.
                arithmetic register.  When ``None`` (default), the minimum
                sufficient size is computed automatically via
                ``_required_output_bits``.  A manual value must satisfy
                ``2**(output_bits-1) > f_max - f_min``.
            y (int): Current best known objective value.  The oracle marks
                all states ``x`` where ``f(x) < y``.  Pass the QUBO objective
                directly — the sign convention is handled internally.
            num_iterations (int): Number of Grover operator applications.

        Returns:
            ExecutableProgram: The compiled circuit program.

        """
        if output_bits is None:
            output_bits = self._required_output_bits(self.binary_model)
        if not self.binary_model.higher:
            return self._transpile_quadratic(
                transpiler, output_bits=output_bits, y=y, num_iterations=num_iterations
            )
        return self._transpile_hubo(
            transpiler, output_bits=output_bits, y=y, num_iterations=num_iterations
        )

    def _transpile_quadratic(
        self,
        transpiler: Transpiler,
        *,
        output_bits: int,
        y: int,
        num_iterations: int,
    ) -> ExecutableProgram:
        """Transpile a QUBO model into an executable Grover circuit.

        Uses the BINARY-domain coefficients so that the QFT phase-encoding
        computes the correct objective value ``f(x) = constant + Σ linear[i]·xᵢ
        + Σ quad[i,j]·xᵢxⱼ`` for binary inputs xᵢ ∈ {0, 1}.

        The circuit encodes ``f(x) − y`` in the output register and the oracle
        marks states where that quantity is negative (two's-complement MSB = 1),
        i.e. states where ``f(x) < y``.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            output_bits (int): Number of output qubits for the QFT arithmetic
                register.
            y (int): Current best known objective value (QUBO scale).
            num_iterations (int): Number of Grover operator applications.

        Returns:
            ExecutableProgram: The compiled circuit program.

        """

        @qmc.qkernel
        def measure_grover_algorithm(
            n: qmc.UInt,
            m: qmc.UInt,
            y: qmc.Float,
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            iters: qmc.UInt = 1,
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

        # The QFT register encodes  y_circuit + Σ linear[i]·xᵢ + Σ quad[i,j]·xᵢxⱼ
        # = (constant − y) + (f(x) − constant)  =  f(x) − y.
        # The oracle fires when this is negative (MSB = 1), i.e. f(x) < y.
        y_circuit = int(self.binary_model.constant - y)

        return transpiler.transpile(
            measure_grover_algorithm,
            bindings={
                "n": self.binary_model.num_bits,
                "m": output_bits,
                "y": y_circuit,
                "linear": self.binary_model.linear,
                "quad": self.binary_model.quad,
                "iters": num_iterations,
            },
        )

    ######################################################################################
    #                   HUBO PATH (degree ≥ 3) using qkernel factory                     #
    ######################################################################################

    def _make_term_encoding(self, ctrl_indices: list[int]):
        """Factory to create kernel that encodes one polynomial term by controlled phase.

        Args:
            ctrl_indices (list[int]): Input-qubit indices used as controls.

        Returns:
            callable: A qkernel with signature ``(q_output, q_input, theta)``.

        """
        degree = len(ctrl_indices)

        if degree == 0:

            @qmc.qkernel
            def constant_encoding(
                q_output: qmc.Vector[qmc.Qubit],
                q_input: qmc.Vector[qmc.Qubit],
                theta: qmc.Float,
            ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
                # theta is already the pre-scaled angle (2π·coef/2^m); apply
                # p-rotations directly to avoid the extra 2π/2^m scaling that
                # qft_encoding would introduce.
                m = q_output.shape[0]
                for i in qmc.range(m):
                    q_output[i] = qmc.p(q_output[i], theta * (2**i))
                return q_output, q_input

            return constant_encoding

        @qmc.qkernel
        def term_encoding(
            q_output: qmc.Vector[qmc.Qubit],
            q_input: qmc.Vector[qmc.Qubit],
            theta: qmc.Float,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            m = q_output.shape[0]
            # num_controls must be a Python int — baked in via closure over `degree`.
            mcp_phase = qmc.control(qmc.p, num_controls=degree)
            # ctrl_indices is a plain Python list resolved at trace time; build
            # controls once outside the loop — it does not depend on i.
            controls = [q_input[ci] for ci in ctrl_indices]
            for i in qmc.range(m):
                mcp_phase(*controls, q_output[i], theta=theta * (2**i))
            return q_output, q_input

        return term_encoding

    def _compose_encoders_baked(
        self,
        encoders: list,
        theta_values: list,
    ):
        """Factory that composes term encoders with baked-in angles into a single kernel.

        Builds the chain iteratively at factory time — no Python recursion — so
        models with many terms do not hit the interpreter's recursion limit. Each
        ``(encoder, theta)`` pair is wrapped in its own helper to ensure correct
        closure capture before being prepended to the growing chain.

        Args:
            encoders (list): Encoder kernels to apply in sequence.
            theta_values (list): Per-encoder phase angles, one per encoder.

        Returns:
            callable: A qkernel with signature ``(q_output, q_input)``.

        """
        steps = list(zip(encoders, theta_values))

        if not steps:

            @qmc.qkernel
            def identity(
                q_output: qmc.Vector[qmc.Qubit],
                q_input: qmc.Vector[qmc.Qubit],
            ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
                return q_output, q_input

            return identity

        def _make_leaf(enc, th):
            @qmc.qkernel
            def leaf(
                q_output: qmc.Vector[qmc.Qubit],
                q_input: qmc.Vector[qmc.Qubit],
            ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
                q_output, q_input = enc(q_output, q_input, th)
                return q_output, q_input

            return leaf

        def _make_step(enc, th, rest):
            @qmc.qkernel
            def step(
                q_output: qmc.Vector[qmc.Qubit],
                q_input: qmc.Vector[qmc.Qubit],
            ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
                q_output, q_input = enc(q_output, q_input, th)
                q_output, q_input = rest(q_output, q_input)
                return q_output, q_input

            return step

        chain = _make_leaf(*steps[-1])
        for enc, th in reversed(steps[:-1]):
            chain = _make_step(enc, th, chain)
        return chain

    def _make_apply_function_preparation_hubo(
        self,
        output_bits: int,
    ):
        """Factory to create the HUBO state-preparation kernel.

        Args:
            output_bits (int): Number of qubits in the QFT output register.

        Returns:
            callable: A qkernel with signature ``(q_output, q_input, y)``.

        """
        # Gather all terms — linear, quadratic, and higher-order — into a
        # single encoder list. The function doesn't need to distinguish them.
        terms_indices = []
        coefficients = []

        for idx, coef in self.binary_model.linear.items():
            terms_indices.append([idx])
            coefficients.append(float(coef))

        for idxs, coef in self.binary_model.quad.items():
            terms_indices.append(list(idxs))
            coefficients.append(float(coef))

        for idxs, coef in self.binary_model.higher.items():
            terms_indices.append(list(idxs))
            coefficients.append(float(coef))

        # Pre-compute phase angles for each term.
        functional_theta = [2 * np.pi * c / (2**output_bits) for c in coefficients]

        encoders = [self._make_term_encoding(idxs) for idxs in terms_indices]
        all_phases = self._compose_encoders_baked(encoders, functional_theta)

        @qmc.qkernel
        def apply_hubo_preparation(
            q_output: qmc.Vector[qmc.Qubit],
            q_input: qmc.Vector[qmc.Qubit],
            y: qmc.Float,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            for i in qmc.range(output_bits):
                q_output[i] = qmc.h(q_output[i])
            for i in qmc.range(self.binary_model.num_bits):
                q_input[i] = qmc.h(q_input[i])

            q_output, q_input = zero_degree_qft_encoding(q_output, q_input, y)
            q_output, q_input = all_phases(q_output, q_input)
            q_output = qmc.iqft(q_output)
            return q_output, q_input

        return apply_hubo_preparation

    def _make_apply_function_preparation_hubo_dagger(
        self,
        output_bits: int,
    ):
        """Factory: returns the Hermitian conjugate of the qkernel from _make_apply_function_preparation_hubo.

        Applies all phase encodings in reverse order with negated angles, replaces the
        final IQFT with a QFT, negates the y offset, and undoes the initial Hadamards.

        Args:
            output_bits (int): Number of output qubits for the QFT register.
                Must match the value used for the forward preparation kernel.

        Returns:
            callable: A qkernel with signature
                ``(q_output: Vector[Qubit], q_input: Vector[Qubit], y: UInt)
                -> tuple[Vector[Qubit], Vector[Qubit]]``.

        """
        terms_indices = []
        coefficients = []

        for idx, coef in self.binary_model.linear.items():
            terms_indices.append([idx])
            coefficients.append(float(coef))

        for idxs, coef in self.binary_model.quad.items():
            terms_indices.append(list(idxs))
            coefficients.append(float(coef))

        for idxs, coef in self.binary_model.higher.items():
            terms_indices.append(list(idxs))
            coefficients.append(float(coef))

        # Negate all angles for the dagger, then reverse order of application.
        functional_theta_dagger = [
            -2 * np.pi * c / (2**output_bits) for c in coefficients
        ]
        encoders = [self._make_term_encoding(idxs) for idxs in terms_indices]
        encoders_rev = list(reversed(encoders))
        thetas_rev = list(reversed(functional_theta_dagger))

        all_phases_dagger = self._compose_encoders_baked(encoders_rev, thetas_rev)

        @qmc.qkernel
        def apply_hubo_preparation_dagger(
            q_output: qmc.Vector[qmc.Qubit],
            q_input: qmc.Vector[qmc.Qubit],
            y: qmc.Float,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            # Reverse of the final iqft
            q_output = qmc.qft(q_output)
            # Reverse all phase encodings with negated angles
            q_output, q_input = all_phases_dagger(q_output, q_input)
            # Reverse the zero-degree phase with negated y
            q_output, q_input = zero_degree_qft_encoding(q_output, q_input, (-1.0) * y)
            # Reverse the initial Hadamards (H is self-adjoint)
            for i in qmc.range(self.binary_model.num_bits):
                q_input[i] = qmc.h(q_input[i])
            for i in qmc.range(output_bits):
                q_output[i] = qmc.h(q_output[i])
            return q_output, q_input

        return apply_hubo_preparation_dagger

    def _transpile_hubo(
        self,
        transpiler: Transpiler,
        *,
        output_bits: int,
        y: int,
        num_iterations: int,
    ) -> ExecutableProgram:
        """Transpile a HUBO model into an executable Grover circuit using the factory methods.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            output_bits (int): Number of output qubits.
            y (int): Current best known objective value.
            num_iterations (int): Number of Grover operator applications.

        Returns:
            ExecutableProgram: The compiled circuit program.

        """
        apply_function_preparation_hubo = self._make_apply_function_preparation_hubo(
            output_bits=output_bits
        )
        apply_function_preparation_hubo_dagger = (
            self._make_apply_function_preparation_hubo_dagger(output_bits=output_bits)
        )

        # Local diffusion that writes the slice back after controlled_Z so the
        # slice borrow is released before the next prep call (which accesses
        # q_input with concrete indices and would collide with a live borrow).
        @qmc.qkernel
        def hubo_diffusion_op(
            q_input: qmc.Vector[qmc.Qubit],
        ) -> qmc.Vector[qmc.Qubit]:
            """Apply the diffusion step for the HUBO Grover operator.

            Args:
                q_input (qmc.Vector[qmc.Qubit]): Input register to reflect around the uniform superposition.

            Returns:
                qmc.Vector[qmc.Qubit]: Updated input register.

            """
            n = q_input.shape[0]
            controlled_z = qmc.control(qmc.z, num_controls=n - 1)
            for i in qmc.range(n):
                q_input[i] = qmc.x(q_input[i])
            controls = q_input[0 : n - 1]
            target = q_input[n - 1]
            controls, target = controlled_z(controls, target)
            q_input[0 : n - 1] = controls  # ReleaseSliceViewOperation — releases borrow
            q_input[n - 1] = target
            for i in qmc.range(n):
                q_input[i] = qmc.x(q_input[i])
            return q_input

        @qmc.qkernel
        def hubo_grover_operator(
            q_output: qmc.Vector[qmc.Qubit],
            q_input: qmc.Vector[qmc.Qubit],
            y: qmc.Float,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            """Apply one Grover iteration for the HUBO GAS oracle.

            Args:
                q_output (qmc.Vector[qmc.Qubit]): Output register for arithmetic encoding.
                q_input (qmc.Vector[qmc.Qubit]): Input register for decision variables.
                y (qmc.Float): Objective threshold offset encoded as a constant term.

            Returns:
                tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated output and input registers.

            """
            m = q_output.shape[0]

            # Oracle
            q = q_output[m - 1]
            q = qmc.z(q)
            q_output[m - 1] = q

            # A_y^dagger
            q_output, q_input = apply_function_preparation_hubo_dagger(
                q_output, q_input, y
            )

            # Diffusion
            q_input = hubo_diffusion_op(q_input)

            # A_y
            q_output, q_input = apply_function_preparation_hubo(q_output, q_input, y)

            return q_output, q_input

        @qmc.qkernel
        def hubo_grover_algorithm(
            n: qmc.UInt,
            m: qmc.UInt,
            y: qmc.Float,
            iters: qmc.UInt = 1,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            """Run repeated Grover iterations for the HUBO GAS circuit.

            Args:
                n (qmc.UInt): Number of input qubits.
                m (qmc.UInt): Number of output qubits.
                y (qmc.Float): Objective threshold offset encoded as a constant term.
                iters (qmc.UInt): Number of Grover iterations.

            Returns:
                tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Output and input qubit registers after all iterations.

            """
            q_output = qmc.qubit_array(m, name="q_output")
            q_input = qmc.qubit_array(n, name="q_input")
            q_output, q_input = apply_function_preparation_hubo(q_output, q_input, y)

            for _ in qmc.range(iters):
                q_output, q_input = hubo_grover_operator(
                    q_output,
                    q_input,
                    y=y,
                )

            return q_output, q_input

        @qmc.qkernel
        def measure_hubo_grover_algorithm(
            n: qmc.UInt,
            m: qmc.UInt,
            y: qmc.Float,
            iters: qmc.UInt = 1,
        ) -> qmc.Vector[qmc.Bit]:
            """Measure the input register after applying the Grover algorithm.

            Args:
                n (qmc.UInt): Number of input (decision-variable) qubits.
                m (qmc.UInt): Number of output (objective-value) qubits.
                y (qmc.Float): Internal circuit threshold: ``binary_model.constant − best_y``.
                    Encodes the shift so the oracle fires on ``f(x) < best_y``.
                iters (qmc.UInt): Number of Grover iterations. Defaults to 1.

            Returns:
                qmc.Vector[qmc.Bit]: Measurement outcomes of the input register.

            """
            q_output, q_input = hubo_grover_algorithm(
                n=n,
                m=m,
                y=y,
                iters=iters,
            )
            return qmc.measure(q_input)

        y_circuit = int(self.binary_model.constant - y)

        return transpiler.transpile(
            measure_hubo_grover_algorithm,
            bindings={
                "n": self.binary_model.num_bits,
                "m": output_bits,
                "y": y_circuit,
                "iters": num_iterations,
            },
        )
