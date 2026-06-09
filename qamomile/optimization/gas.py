"""This module implements the Grover Adaptive Search (GAS) algorithm for Combinatorial Polynomial Binary Optimization (CPBO).

The quantum optimization algorithm iteratively applies Grover's search to find the minimum of the function.
The GAS algorithm is designed to efficiently find the optimal solution by adaptively adjusting the search
space based on previous iterations' results.
"""

import math
import warnings
from decimal import Decimal

import numpy as np

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.gas import (
    grover_algorithm,
    qft_encoding,
    zero_degree_qft_encoding,
)
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.transpiler import Transpiler

from .binary_model import VarType
from .binary_model.model import BinaryModel
from .converter import MathematicalProblemConverter


class GASConverter(MathematicalProblemConverter):
    """Converter for Grover Adaptive Search (GAS).

    Internally maintains a BINARY-domain model derived from ``spin_model``
    so that the Grover QFT-arithmetic circuit receives the correct QUBO
    coefficients (binary variables take values in {0, 1}, not ±1).
    """

    _MAX_EXACT_INT_IN_FLOAT_BITS = 54

    def __post_init__(self) -> None:
        """Derive and cache the BINARY model from the parent spin model."""
        self.binary_model = self.spin_model.change_vartype(VarType.BINARY)
        coeffs = [self.binary_model.constant] + list(self.binary_model.coefficients.values())
        coeffs = GASConverter._align_precision(coeffs)
        precision_aligned_constant = coeffs[0]
        precision_aligned_coefficients = {
            k: v for k, v in zip(self.binary_model.coefficients.keys(), coeffs[1:])
        }
        self.binary_model = BinaryModel.from_hubo(
            hubo=precision_aligned_coefficients,
            constant=precision_aligned_constant,
        )

    @staticmethod
    def _align_precision(values: list[float]) -> list[float]:
        """Round noisy non-integer coefficients to a common user-intended precision.

        Floating-point computation can leave a coefficient that the user meant
        as an exact decimal (e.g. ``1.0``) fakely precise (e.g.
        ``0.9999999999999999`` or ``1.0000000000000002``). Left as-is, such a
        value would make ``_detect_eps`` report a spuriously fine epsilon driven
        by rounding noise rather than by the precision the user actually wrote.

        By default, this infers the *finest meaningful* decimal precision across
        non-integer values while ignoring binary floating-point tails.
        Concretely, for each value it finds the smallest number of decimal
        digits that reproduces the value within ``atol=1e-12``; then all
        non-integer values are rounded to the maximum of those per-value
        precisions. Integer-valued coefficients (within ``atol=1e-12``) are
        considered exact.

        Args:
            values (list[float]): Coefficient values to align, typically the
                model's constant followed by its interaction coefficients.

        Returns:
            list[float]: ``values`` with every non-integer entry rounded to the
                finest meaningful decimal precision inferred from non-integer
                entries. Returned unchanged (aside from float coercion) when
                every value is integer-valued.

        """
        non_integer_values = []
        for v in values:
            fv = float(v)
            if not np.isclose(fv, round(fv), atol=1e-12):
                non_integer_values.append(fv)

        if not non_integer_values:
            return [
                float(round(float(v)))
                if np.isclose(float(v), round(float(v)), atol=1e-12)
                else float(v)
                for v in values
            ]

        max_scan_decimals = 12
        inferred_digits = []
        for fv in non_integer_values:
            inferred = None
            for d in range(max_scan_decimals + 1):
                if np.isclose(fv, round(fv, d), atol=1e-12, rtol=0.0):
                    inferred = d
                    break

            if inferred is None:
                exp = Decimal(str(fv)).as_tuple().exponent
                inferred = -exp if exp < 0 else 0  # type: ignore[operator]

            inferred_digits.append(inferred)

        round_digits = max(inferred_digits)

        aligned = []
        for v in values:
            fv = float(v)
            if np.isclose(fv, round(fv), atol=1e-12):
                # Snap integer-like float artifacts (e.g. 4.000000000000005)
                # to exact integers so downstream precision detection does not
                # keep meaningless binary tails.
                aligned.append(float(round(fv)))
            else:
                aligned.append(round(fv, round_digits))
        return aligned

    @staticmethod
    def _detect_eps(values: list[float]) -> float:
        """Auto-detect the input precision epsilon from a list of floats.

        Uses Python's shortest-decimal guarantee: ``str(float(v))`` always
        produces the shortest decimal string that round-trips back to the same
        float.  The exponent of that decimal representation gives the precision
        of the value as the user wrote it (e.g. ``0.3`` → exponent ``-1`` →
        eps ``0.1``; ``1/3 = 0.3333...`` → exponent ``-16`` → eps ``1e-16``).
        The overall epsilon is the minimum across all values.

        Args:
            values (list[float]): Coefficient values to inspect.

        Returns:
            float: The finest common step size that represents all values.

        """
        eps = 1.0
        for v in values:
            exp = Decimal(str(float(v))).as_tuple().exponent
            if exp < 0:  # type: ignore[operator]
                eps = min(eps, 10.0**exp)  # type: ignore[operator]
        return eps

    @staticmethod
    def _greedy_quantization_parameter(
        binary_model: BinaryModel,
    ) -> int:
        """Compute the quantization precision for approximating a real-valued model.

        Args:
            binary_model (BinaryModel): The binary model whose coefficients are
                quantized. Both ``constant`` and all interaction coefficients are
                considered.

        Returns:
            int: Quantization parameter ``m_q``. Each coefficient ``a`` is
                approximated as ``k = round(a / max_coeff * 2**(m_q - 1))``.
        """
        safety = 2  # Extra safety bits (default 2, the minimum for correctness).
        coeffs = [binary_model.constant] + list(binary_model.coefficients.values())

        max_coeff = float(np.max(np.abs(coeffs)))
        N_terms = len(coeffs)

        epsilon = GASConverter._detect_eps(coeffs)

        p = math.ceil(math.log2(max_coeff / epsilon))
        log_N = math.ceil(math.log2(N_terms)) if N_terms > 1 else 0

        return p + log_N + safety

    @staticmethod
    def approximate_real_valued_model(
        binary_model: BinaryModel, quantization_parameter: int | None = None
    ) -> BinaryModel:
        """Rescale and round all model coefficients to integers for QFT arithmetic.

        Divides every coefficient (including the constant) by the maximum absolute
        value to map them into [-1, 1], then multiplies by ``2^(quantization_parameter - 1)``
        and rounds to the nearest integer. The resulting model has integer coefficients
        that the Grover QFT circuit can encode exactly.

        Args:
            binary_model (BinaryModel): The original binary model with real-valued
                coefficients.
            quantization_parameter (int | None): Number of bits used for the fixed-point
                representation. ``2^(quantization_parameter - 1)`` is the scale factor
                applied after normalisation. When ``None``, the value is chosen
                automatically by ``_greedy_quantization_parameter``.

        Returns:
            BinaryModel: A new binary model whose coefficients are integers that
                approximate the original up to the chosen precision.
        """

        #### Rescaling
        coef_list = [binary_model.constant] + list(binary_model.coefficients.values())
        coef_array = np.asarray(coef_list, dtype=float)
        max_val = np.max(np.abs(coef_array))
        if np.isclose(max_val, 0.0, atol=1e-12):
            return binary_model
        rescaled_coef_list = coef_array / max_val

        #### Approximation

        if quantization_parameter is None:
            quantization_parameter = GASConverter._greedy_quantization_parameter(
                binary_model
            )
            if quantization_parameter > GASConverter._MAX_EXACT_INT_IN_FLOAT_BITS:
                warnings.warn(
                    "Auto-selected quantization_parameter is too large for exact integer "
                    "representation in float-backed binary models. Clamping it to 54 bits "
                    "to preserve integer-valued coefficients.",
                    UserWarning,
                )
                quantization_parameter = GASConverter._MAX_EXACT_INT_IN_FLOAT_BITS

        scale = 2 ** (quantization_parameter - 1)
        frac_list = [int(round(float(a) * scale)) for a in rescaled_coef_list]
        new_constant = frac_list[0]
        new_coef = {
            i: j for i, j in zip(binary_model.coefficients.keys(), frac_list[1:])
        }

        ### Build new model with the rational coefficients
        return BinaryModel.from_hubo(
            hubo=new_coef,
            constant=new_constant,
        )

    @staticmethod
    def _required_output_bits(binary_model: BinaryModel) -> int:
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
        range_span = f_max - f_min
        if range_span <= 0:
            return 2
        return max(2, int(math.floor(math.log2(range_span))) + 2)

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Raise NotImplementedError because GAS is oracle-based and has no cost Hamiltonian.

        GAS marks states via an oracle reflection rather than minimizing the
        expectation value of a cost Hamiltonian. This method always raises so
        that incorrect usage fails immediately rather than propagating a silent
        ``None`` that would cause a confusing error later.

        Returns:
            qm_o.Hamiltonian: Never returns; declared for base-class
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
        approximate_real_coefficients: bool = True,
        quantization_parameter: int | None = None,
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

        # If the model contains real values
        all_values = [self.binary_model.constant] + list(
            self.binary_model.coefficients.values()
        )
        has_non_integer = any(
            not np.isclose(v, round(v), atol=1e-12) for v in all_values
        )
        if has_non_integer and approximate_real_coefficients:
            self.effective_model = self.approximate_real_valued_model(
                self.binary_model, quantization_parameter=quantization_parameter
            )
            warnings.warn(
                "The binary model contains non-integer coefficients. "
                "They have been rescaled and approximated to the nearest integer. "
                "This may affect the accuracy of the results. "
                "Use approximate_real_coefficients=False to disable this approximation "
                "and use the original real coefficients."
            )
        else:
            self.effective_model = self.binary_model

        if output_bits is None:
            output_bits = self._required_output_bits(self.effective_model)

        if not self.effective_model.higher:
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
            iters: qmc.UInt = 1,  # type: ignore[assignment]
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
        y_circuit = self.effective_model.constant - y

        return transpiler.transpile(
            measure_grover_algorithm,
            bindings={
                "n": self.effective_model.num_bits,
                "m": output_bits,
                "y": y_circuit,
                "linear": self.effective_model.linear,
                "quad": self.effective_model.quad,
                "iters": num_iterations,
            },
        )

    ######################################################################################
    #                   HUBO PATH (degree ≥ 3) using qkernel factory                     #
    ######################################################################################

    def _make_term_encoding(self, ctrl_indices: list[int]) -> qmc.QKernel:
        """Factory to create kernel that encodes one polynomial term via qft_encoding.

        Args:
            ctrl_indices (list[int]): Input-qubit indices used as controls.

        Returns:
            qmc.QKernel: A qkernel with signature ``(q_output, q_input, coef)``.

        """
        degree = len(ctrl_indices)

        if degree == 0:
            return zero_degree_qft_encoding

        @qmc.qkernel
        def term_encoding(
            q_output: qmc.Vector[qmc.Qubit],
            q_input: qmc.Vector[qmc.Qubit],
            coef: qmc.Float,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            # num_controls baked in at factory time via closure over `degree`.
            ctrl_qft = qmc.control(qft_encoding, num_controls=degree)
            controls = [q_input[ci] for ci in ctrl_indices]
            result = ctrl_qft(*controls, q_output, coef)
            # result = (ctrl_0, ..., ctrl_{degree-1}, q_output)
            # List-comp instead of a for-statement: the DSL transformer converts
            # `for k in range(...)` statements into for_loop() context managers
            # where k becomes a symbolic UInt, causing ctrl_indices[k] and
            # result[k] (both Python-sequence indexing) to call UInt.__index__()
            # and raise TypeError.  Inside a list comprehension range(degree) is
            # Python's built-in and k stays a concrete int.
            [q_input.__setitem__(ctrl_indices[k], result[k]) for k in range(degree)]  # type: ignore[index]
            q_output = result[degree]  # type: ignore[index]
            return q_output, q_input

        return term_encoding

    def _compose_encoders_baked(
        self,
        encoders: list[qmc.QKernel],
        coef_values: list[float],
    ) -> qmc.QKernel:
        """Factory that composes term encoders with baked-in coefficients into a single kernel.

        Builds the chain iteratively at factory time — no Python recursion — so
        models with many terms do not hit the interpreter's recursion limit. Each
        ``(encoder, coef)`` pair is wrapped in its own helper to ensure correct
        closure capture before being prepended to the growing chain.

        Args:
            encoders (list[qmc.QKernel]): Encoder kernels to apply in sequence.
            coef_values (list[float]): Per-encoder raw coefficients, one per encoder.

        Returns:
            qmc.QKernel: A qkernel with signature ``(q_output, q_input)``.

        """
        if len(encoders) != len(coef_values):
            raise ValueError(
                "encoders and coef_values must have the same length "
                f"(got {len(encoders)} and {len(coef_values)})."
            )

        steps = list(zip(encoders, coef_values))

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
    ) -> qmc.QKernel:
        """Factory to create the HUBO state-preparation kernel.

        Args:
            output_bits (int): Number of qubits in the QFT output register.

        Returns:
            qmc.QKernel: A qkernel with signature ``(q_output, q_input, y)``.

        """
        # Gather all terms — linear, quadratic, and higher-order — into a
        # single encoder list. The function doesn't need to distinguish them.
        terms_indices = []
        coefficients = []

        for idx, coef in self.effective_model.linear.items():
            terms_indices.append([idx])
            coefficients.append(float(coef))

        for idxs, coef in self.effective_model.quad.items():
            terms_indices.append(list(idxs))
            coefficients.append(float(coef))

        for idxs, coef in self.effective_model.higher.items():
            terms_indices.append(list(idxs))
            coefficients.append(float(coef))

        encoders = [self._make_term_encoding(idxs) for idxs in terms_indices]
        all_phases = self._compose_encoders_baked(encoders, coefficients)

        @qmc.qkernel
        def apply_hubo_preparation(
            q_output: qmc.Vector[qmc.Qubit],
            q_input: qmc.Vector[qmc.Qubit],
            y: qmc.Float,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            for i in qmc.range(output_bits):
                q_output[i] = qmc.h(q_output[i])
            for i in qmc.range(self.effective_model.num_bits):
                q_input[i] = qmc.h(q_input[i])

            q_output, q_input = zero_degree_qft_encoding(q_output, q_input, y)
            q_output, q_input = all_phases(q_output, q_input)
            q_output = qmc.iqft(q_output)
            return q_output, q_input

        return apply_hubo_preparation

    def _make_apply_function_preparation_hubo_dagger(
        self,
        output_bits: int,
    ) -> qmc.QKernel:
        """Factory: returns the Hermitian conjugate of the qkernel from _make_apply_function_preparation_hubo.

        Applies all phase encodings in reverse order with negated angles, replaces the
        final IQFT with a QFT, negates the y offset, and undoes the initial Hadamards.

        Args:
            output_bits (int): Number of output qubits for the QFT register.
                Must match the value used for the forward preparation kernel.

        Returns:
            qmc.QKernel: A qkernel with signature
                ``(q_output: Vector[Qubit], q_input: Vector[Qubit], y: Float)
                -> tuple[Vector[Qubit], Vector[Qubit]]``.

        """
        terms_indices = []
        coefficients = []

        for idx, coef in self.effective_model.linear.items():
            terms_indices.append([idx])
            coefficients.append(float(coef))

        for idxs, coef in self.effective_model.quad.items():
            terms_indices.append(list(idxs))
            coefficients.append(float(coef))

        for idxs, coef in self.effective_model.higher.items():
            terms_indices.append(list(idxs))
            coefficients.append(float(coef))

        # Negate all coefficients for the dagger, then reverse order of application.
        encoders = [self._make_term_encoding(idxs) for idxs in terms_indices]
        encoders_rev = list(reversed(encoders))
        coefs_neg_rev = list(reversed([-c for c in coefficients]))

        all_phases_dagger = self._compose_encoders_baked(encoders_rev, coefs_neg_rev)

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
            for i in qmc.range(self.effective_model.num_bits):
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
            controls = q_input[0 : n - 1]  # type: ignore[misc]
            target = q_input[n - 1]  # type: ignore[misc]
            controls, target = controlled_z(controls, target)
            q_input[0 : n - 1] = controls  # type: ignore[misc]  # ReleaseSliceViewOperation — releases borrow
            q_input[n - 1] = target  # type: ignore[misc]
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
            iters: qmc.UInt = 1,  # type: ignore[assignment]
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
            iters: qmc.UInt = 1,  # type: ignore[assignment]
        ) -> qmc.Vector[qmc.Bit]:
            """Measure the input register after applying the Grover algorithm.

            Args:
                n (qmc.UInt): Number of input (decision-variable) qubits.
                m (qmc.UInt): Number of output (objective-value) qubits.
                y (qmc.Float): Internal circuit threshold: ``effective_model.constant − best_y``.
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

        y_circuit =self.effective_model.constant - y

        return transpiler.transpile(
            measure_hubo_grover_algorithm,
            bindings={
                "n": self.effective_model.num_bits,
                "m": output_bits,
                "y": y_circuit,
                "iters": num_iterations,
            },
        )
