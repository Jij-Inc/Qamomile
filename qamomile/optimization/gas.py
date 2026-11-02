# ruff: noqa: F821 -- widths are closed into concrete Qamomile shapes.

"""Grover Adaptive Search composed from ordinary Qamomile qkernels."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from typing import cast

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.compilation import (
    CompilationTarget,
    CompiledProgram,
    ExecutionEndpoint,
    SampleResult,
    compile as compile_program,
    run as run_program,
)
from qamomile.ir import Program
from qamomile.stdlib._primitives import flip
from qamomile.stdlib._structure import (
    adjoint_unary,
    compose_unary,
    lift_indices,
    lift_slice,
)
from qamomile.stdlib.arithmetic import add_constant

from .model import BinaryDomain, BinaryModel, VarType
from .qaoa import OptimizationExperimentIdentity, OptimizationRun

_MAX_EXACT_FLOAT_INTEGER_BITS = 54


def _decision_superposition(*, total_width: int, decision_width: int) -> qm.QKernel:
    """Construct Hadamards over the decision prefix of a joint register.

    Args:
        total_width (int): Width of the decision and output registers together.
        decision_width (int): Positive decision-register prefix width.

    Returns:
        qm.QKernel: Unary joint-register transform.

    Raises:
        ValueError: If the widths do not describe a nonempty proper prefix.
    """

    if decision_width <= 0 or decision_width >= total_width:
        raise ValueError("GAS decision width must be a nonempty register prefix")

    @qm.qkernel
    def prepare(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Prepare a uniform decision-register superposition.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Owned joint register.

        Returns:
            qm.Q[qm.Bits[total_width]]: Joint register with Hadamards applied to
            the decision prefix.
        """

        for index in range(decision_width):
            state = qm.h(state, target=index)
        return state

    return prepare


def _controlled_term_encoding(
    *,
    decision_width: int,
    output_width: int,
    controls: tuple[int, ...],
    coefficient: float,
) -> qm.QKernel:
    """Lift one monomial-controlled modular addition onto a joint register.

    Args:
        decision_width (int): Number of decision bits in the joint prefix.
        output_width (int): Number of objective bits in the joint suffix.
        controls (tuple[int, ...]): Dense decision-bit positions in the
            monomial.
        coefficient (float): Integer-valued coefficient added when every control is
            one.

    Returns:
        qm.QKernel: Unary transform over decision and output registers.

    Raises:
        ValueError: If the control row is empty, repeated, or out of range.
    """

    if not controls:
        raise ValueError("a controlled GAS term must have at least one variable")
    if len(set(controls)) != len(controls) or any(
        index < 0 or index >= decision_width for index in controls
    ):
        raise ValueError("GAS term controls must be unique decision positions")
    if not float(coefficient).is_integer():
        raise ValueError("GAS arithmetic requires integer-valued coefficients")
    addition = add_constant(width=output_width, value=int(coefficient))
    selected_width = len(controls) + output_width
    degree = len(controls)

    @qm.qkernel
    def controlled(
        state: qm.Q[qm.Bits[selected_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[selected_width]]:  # type: ignore[valid-type]
        """Add one objective term under every monomial control.

        Args:
            state (qm.Q[qm.Bits[selected_width]]): Selected controls followed by
                the objective register.

        Returns:
            qm.Q[qm.Bits[selected_width]]: Updated controls and objective register.
        """

        control_state, output = qm.split_at(state, degree)
        control_state, output = qm.control(addition, control_state, output)
        return qm.concat(control_state, output)

    total_width = decision_width + output_width
    output_indices = tuple(range(decision_width, total_width))
    return lift_indices(
        total_width=total_width,
        indices=(*controls, *output_indices),
        transform=controlled,
    )


def _zero_reflection(*, width: int) -> qm.QKernel:
    """Construct the reflection about the all-zero decision state.

    Args:
        width (int): Positive decision-register width.

    Returns:
        qm.QKernel: Unary zero-state reflection.

    Raises:
        ValueError: If ``width`` is not positive.
    """

    if width <= 0:
        raise ValueError("GAS decision width must be positive")
    if width == 1:

        @qm.qkernel
        def single(
            state: qm.Q[qm.Bits[1]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
            """Reflect one decision qubit about zero up to global phase.

            Args:
                state (qm.Q[qm.Bits[1]]): Owned decision qubit.

            Returns:
                qm.Q[qm.Bits[1]]: Reflected decision qubit.
            """

            return qm.z(state)

        return single

    control_width = width - 1

    @qm.qkernel
    def reflect(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Reflect a multi-bit decision register about the zero state.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned decision register.

        Returns:
            qm.Q[qm.Bits[width]]: Reflected decision register.
        """

        for index in range(width):
            state = qm.x(state, target=index)
        controls, target = qm.split_at(state, control_width)
        target = qm.h(target)
        controls, target = qm.control(flip, controls, target)
        target = qm.h(target)
        state = qm.concat(controls, target)
        for index in range(width):
            state = qm.x(state, target=index)
        return state

    return reflect


def _gas_preparation(
    *,
    model: BinaryModel[BinaryDomain],
    threshold: float,
    output_width: int,
) -> qm.QKernel:
    """Construct the reversible quantum dictionary for one fixed model.

    Args:
        model (BinaryModel[BinaryDomain]): Effective integer or real binary
            polynomial encoded by the circuit.
        threshold (float): Threshold in the effective model's scale.
        output_width (int): Two's-complement objective-register width.

    Returns:
        qm.QKernel: Unary preparation over decision and objective registers.

    Raises:
        ValueError: If a model term refers to an unknown variable.
    """

    decision_width = model.num_variables
    total_width = decision_width + output_width
    output_indices = tuple(range(decision_width, total_width))
    position = {variable: index for index, variable in enumerate(model.variables)}
    steps: list[qm.QKernel] = [
        _decision_superposition(
            total_width=total_width,
            decision_width=decision_width,
        ),
    ]
    constant = model.constant - threshold
    if constant != 0.0:
        if not float(constant).is_integer():
            raise ValueError("GAS threshold must map to exact integer arithmetic")
        steps.append(
            lift_indices(
                total_width=total_width,
                indices=output_indices,
                transform=add_constant(width=output_width, value=int(constant)),
            )
        )
    for variables, coefficient in model.coefficients.items():
        try:
            controls = tuple(position[variable] for variable in variables)
        except KeyError as error:
            raise ValueError("GAS term refers to an unknown model variable") from error
        steps.append(
            _controlled_term_encoding(
                decision_width=decision_width,
                output_width=output_width,
                controls=controls,
                coefficient=coefficient,
            )
        )
    return compose_unary(width=total_width, steps=steps)


@qm.qkernel
def _gas_transform(
    preparation: qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]],  # type: ignore[valid-type]
    inverse_preparation: qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]],  # type: ignore[valid-type]
    reflection: qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]],  # type: ignore[valid-type]
    state: qm.Q[qm.Bits[n]],  # type: ignore[valid-type]
    sign_index: qm.Const[qm.Int],  # type: ignore[valid-type]
    iterations: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[n]]:  # type: ignore[valid-type]
    """Apply one fixed-threshold Grover Adaptive Search transform.

    Args:
        preparation (qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]]): Closed reversible
            objective preparation over the joint register.
        inverse_preparation (qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]]): Exact
            inverse of ``preparation``.
        reflection (qm.QKernel[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n]]]): Closed lifted
            zero-state reflection over the same register.
        state (qm.Q[qm.Bits[n]]): Fresh owned joint register.
        sign_index (qm.Const[qm.Int]): Objective sign-bit position.
        iterations (qm.Const[qm.Int]): Nonnegative Grover-operator count.

    Returns:
        qm.Q[qm.Bits[n]]: Joint decision and objective state after amplification.

    Raises:
        ValueError: If a width or iteration count is invalid at specialization.
    """

    state = preparation(state)
    for _ in range(iterations):
        state = qm.z(state, target=sign_index)
        state = inverse_preparation(state)
        state = reflection(state)
        state = preparation(state)
    return state


def _gas_sample_program(
    *,
    model: BinaryModel[BinaryDomain],
    threshold: float,
    iterations: int,
    output_width: int,
) -> Program:
    """Build one closed GAS sampling program.

    Args:
        model (BinaryModel[BinaryDomain]): Effective model encoded by the
            quantum dictionary.
        threshold (float): Threshold in the effective model's scale.
        iterations (int): Nonnegative Grover-operator count.
        output_width (int): Objective-register width.

    Returns:
        Program: Specialized backend-neutral GAS sampling program.

    Raises:
        ValueError: If model or register widths are invalid.
    """

    decision_width = model.num_variables
    if decision_width <= 0:
        raise ValueError("GAS requires at least one model variable")
    total_width = decision_width + output_width
    preparation = _gas_preparation(
        model=model,
        threshold=threshold,
        output_width=output_width,
    )
    reflection = lift_slice(
        total_width=total_width,
        start=0,
        width=decision_width,
        transform=_zero_reflection(width=decision_width),
    )
    inverse_preparation = adjoint_unary(
        width=total_width,
        transform=preparation,
    )

    @qm.qkernel
    def entry() -> qm.Shot[qm.Bits[decision_width]]:  # type: ignore[valid-type]
        """Close the planned transforms at one compiler-visible call site.

        Returns:
            qm.Shot[qm.Bits[decision_width]]: Source-ordered decision bits.
        """

        state = _gas_transform(
            preparation,
            inverse_preparation,
            reflection,
            qm.qubits(total_width),
            total_width - 1,
            iterations,
        )
        decision, objective = qm.split_at(state, decision_width)
        result = qm.measure(decision)
        _ignored_objective = qm.measure(objective)
        return result

    return entry.specialize()


@dataclass(frozen=True, slots=True)
class GASExperiment:
    """Fix one binary objective, threshold, and Grover search topology.

    Args:
        model (BinaryModel[BinaryDomain]): Binary polynomial to minimize.
        threshold (float): Mark assignments whose objective is strictly below
            this value.
        iterations (int): Nonnegative number of Grover iterations.
        output_bits (int | None): Optional objective-register width. ``None``
            derives the smallest safe two's-complement width.
        approximate_real_coefficients (bool): Whether to quantize noninteger
            coefficients before building the arithmetic circuit. ``False``
            preserves the original binary64 coefficients exactly through a
            power-of-two scale and rejects widths above 54 objective bits.
        quantization_parameter (int | None): Optional positive quantization
            precision. ``None`` derives and bounds it automatically.

    Raises:
        TypeError: If an argument has the wrong host type.
        ValueError: If a value is non-finite, a count is invalid, or the
            requested output register is too narrow.
    """

    model: BinaryModel[BinaryDomain]
    threshold: float
    iterations: int = 1
    output_bits: int | None = None
    approximate_real_coefficients: bool = True
    quantization_parameter: int | None = None
    effective_model: BinaryModel[BinaryDomain] = field(init=False)
    quantization_scale: float = field(init=False)
    resolved_output_bits: int = field(init=False)
    program: Program = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration and build one immutable closed program.

        Raises:
            TypeError: If model, flags, or integer controls are ill-typed.
            ValueError: If the model domain, threshold, counts, or width are
                invalid.
        """

        if not isinstance(self.model, BinaryModel):
            raise TypeError("model must be a qamomile.optimization.BinaryModel")
        if self.model.vartype is not VarType.BINARY:
            raise ValueError("GAS requires a BINARY-domain model")
        if self.model.num_variables <= 0:
            raise ValueError("GAS requires at least one model variable")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError("threshold must be a real number")
        threshold = float(self.threshold)
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if type(self.iterations) is not int:
            raise TypeError("iterations must be an integer")
        if self.iterations < 0:
            raise ValueError("iterations must be nonnegative")
        if not isinstance(self.approximate_real_coefficients, bool):
            raise TypeError("approximate_real_coefficients must be a Boolean")
        if self.quantization_parameter is not None and (
            type(self.quantization_parameter) is not int
            or self.quantization_parameter <= 0
        ):
            raise ValueError("quantization_parameter must be a positive integer")
        if self.output_bits is not None and type(self.output_bits) is not int:
            raise TypeError("output_bits must be an integer or None")

        if self.approximate_real_coefficients:
            aligned = _aligned_model(self.model)
            effective = aligned
            scale = 1.0
            if _has_noninteger_coefficients(aligned):
                effective, scale = _quantize_model(
                    aligned,
                    quantization_parameter=self.quantization_parameter,
                )
                warnings.warn(
                    "GAS quantized noninteger coefficients for exact QFT arithmetic; "
                    "inspect effective_model and quantization_scale for the circuit "
                    "values.",
                    UserWarning,
                    stacklevel=2,
                )
            effective_threshold = math.ceil(threshold * scale)
            required = _required_output_bits(effective, effective_threshold)
        else:
            effective, scale, effective_threshold, required = _exact_integer_model(
                self.model, threshold
            )
        resolved = required if self.output_bits is None else self.output_bits
        if (
            not self.approximate_real_coefficients
            and resolved > native.MAX_EXACT_GAS_BITS
        ):
            raise ValueError("exact GAS output_bits exceeds the 54-bit budget")
        if resolved < required:
            minimum, maximum = _objective_bounds(effective)
            raise ValueError(
                f"output_bits={resolved} is too small for f(x) - threshold in "
                f"[{minimum - effective_threshold}, "
                f"{maximum - effective_threshold}]; at least {required} bits "
                "are required"
            )
        program = _gas_sample_program(
            model=effective,
            threshold=effective_threshold,
            iterations=self.iterations,
            output_width=resolved,
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "effective_model", effective)
        object.__setattr__(self, "quantization_scale", scale)
        object.__setattr__(self, "resolved_output_bits", resolved)
        object.__setattr__(self, "program", program)

    @property
    def identity(self) -> OptimizationExperimentIdentity:
        """Return the exact original-problem and closed-program identity.

        Returns:
            OptimizationExperimentIdentity: Stable problem/program digest pair.
        """

        return OptimizationExperimentIdentity(
            "qamomile.gas.v1",
            self.model.semantic_digest,
            self.program.semantic_digest,
        )

    def compile(self, *, to: CompilationTarget) -> CompiledGASExperiment:
        """Compile this fixed experiment for one target.

        Args:
            to (CompilationTarget): Target-format compiler configuration.

        Returns:
            CompiledGASExperiment: Reusable compiled GAS experiment.

        Raises:
            TypeError: If ``to`` is not an artifact target.
        """

        return CompiledGASExperiment(self, compile_program(self.program, to=to))

    def run(
        self,
        *,
        on: ExecutionEndpoint,
        shots: int,
        seed: int = 0,
    ) -> OptimizationRun[BinaryDomain]:
        """Compile, execute, and decode this fixed search experiment.

        Args:
            on (ExecutionEndpoint): Backend-neutral execution endpoint.
            shots (int): Positive number of observations.
            seed (int): Optional deterministic local-endpoint seed.

        Returns:
            OptimizationRun[BinaryDomain]: Execution and decoded distribution.

        Raises:
            TypeError: If the endpoint or launch values are invalid.
            ValueError: If the endpoint cannot run the compiled program.
        """

        return self.compile(to=on.target).run(on=on, shots=shots, seed=seed)


@dataclass(frozen=True, slots=True)
class CompiledGASExperiment:
    """Pair one immutable GAS experiment with its compiled artifact.

    Args:
        experiment (GASExperiment): Source experiment owning the model.
        artifact (CompiledProgram): Compiled target artifact.

    Raises:
        TypeError: If either value has the wrong public type.
        ValueError: If the artifact belongs to a different program.
    """

    experiment: GASExperiment
    artifact: CompiledProgram
    identity: OptimizationExperimentIdentity = field(init=False)

    def __post_init__(self) -> None:
        """Verify the retained artifact belongs to the experiment.

        Raises:
            TypeError: If experiment or artifact has the wrong type.
            ValueError: If the artifact source digest does not match.
        """

        if not isinstance(self.experiment, GASExperiment):
            raise TypeError("compiled GAS experiment requires a GASExperiment")
        if not isinstance(self.artifact, CompiledProgram):
            raise TypeError("compiled GAS experiment requires an CompiledProgram")
        identity = self.experiment.identity
        if self.artifact.source_digest.hex() != identity.program_digest:
            raise ValueError("compiled artifact does not match GAS experiment")
        object.__setattr__(self, "identity", identity)

    def run(
        self,
        *,
        on: ExecutionEndpoint,
        shots: int,
        seed: int = 0,
    ) -> OptimizationRun[BinaryDomain]:
        """Execute and decode the compiled search program.

        Args:
            on (ExecutionEndpoint): Endpoint compatible with the artifact.
            shots (int): Positive number of observations.
            seed (int): Optional deterministic local-endpoint seed.

        Returns:
            OptimizationRun[BinaryDomain]: Execution and original-model
            decoded samples.

        Raises:
            TypeError: If the target result is not a sample distribution.
            ValueError: If endpoint capabilities or launch values are invalid.
        """

        execution = run_program(
            self.artifact,
            on=on,
            shots=shots,
            seed=seed,
        )
        if not isinstance(execution.result, SampleResult):
            raise TypeError("GAS sample program returned a non-sampling result")
        return OptimizationRun(
            identity=self.identity,
            execution=execution,
            samples=self.experiment.model.decode(execution.result),
        )


def _exact_integer_model(
    model: BinaryModel[BinaryDomain], threshold: float
) -> tuple[BinaryModel[BinaryDomain], float, int, int]:
    """Project the checked native exact objective encoding for GAS composition.

    Args:
        model (BinaryModel[BinaryDomain]): Original, unrounded model.
        threshold (float): Finite strict bound on the original objective.

    Returns:
        tuple[BinaryModel[BinaryDomain], float, int, int]: Integer model, exact
        scale, integer threshold, and minimal safe register width.

    Raises:
        TypeError: If native support is unavailable or the projection is malformed.
        ValueError: If exact encoding exceeds scaling or register-width budgets.
    """
    derive = getattr(model._native_model, "exact_integer_encoding", None)
    if not callable(derive):
        raise TypeError("Q7560: native exact integer encoding is unavailable")
    projection = derive(threshold)
    if (
        not isinstance(projection, tuple)
        or len(projection) != 4
        or type(projection[1]) is not float
        or not math.isfinite(projection[1])
        or projection[1] <= 0
        or not isinstance(projection[2], str)
        or type(projection[3]) is not int
        or not 2 <= projection[3] <= native.MAX_EXACT_GAS_BITS
    ):
        raise TypeError("Q7560: malformed exact integer encoding")
    encoded, scale, bound, width = projection
    effective = cast(
        BinaryModel[BinaryDomain], BinaryModel._from_native(VarType.BINARY, encoded)
    )
    return effective, scale, int(bound), width


def _aligned_model(model: BinaryModel[BinaryDomain]) -> BinaryModel[BinaryDomain]:
    """Remove insignificant binary64 tails before optional quantization.

    Args:
        model (BinaryModel[BinaryDomain]): Original immutable binary model.

    Returns:
        BinaryModel[BinaryDomain]: Precision-aligned model with the same
        variable universe.
    """

    values = [model.constant, *model.coefficients.values()]
    noninteger = [
        value
        for value in values
        if not math.isclose(value, round(value), abs_tol=1e-12, rel_tol=0.0)
    ]
    digits = 0
    for value in noninteger:
        inferred = 12
        for candidate in range(13):
            if math.isclose(
                value,
                round(value, candidate),
                abs_tol=1e-12,
                rel_tol=0.0,
            ):
                inferred = candidate
                break
        digits = max(digits, inferred)

    def align(value: float) -> float:
        """Align one finite coefficient to the inferred decimal precision.

        Args:
            value (float): Finite input coefficient.

        Returns:
            float: Integer-snapped or decimal-aligned coefficient.
        """

        if math.isclose(value, round(value), abs_tol=1e-12, rel_tol=0.0):
            return float(round(value))
        return round(value, digits)

    coefficients = {
        variables: align(coefficient)
        for variables, coefficient in model.coefficients.items()
    }
    return BinaryModel.from_hubo(
        coefficients,
        align(model.constant),
        variables=model.variables,
    )


def _has_noninteger_coefficients(model: BinaryModel[BinaryDomain]) -> bool:
    """Return whether any model coefficient is genuinely noninteger.

    Args:
        model (BinaryModel[BinaryDomain]): Precision-aligned binary model.

    Returns:
        bool: ``True`` when quantization can change a coefficient.
    """

    return any(
        not math.isclose(value, round(value), abs_tol=1e-12, rel_tol=0.0)
        for value in (model.constant, *model.coefficients.values())
    )


def _quantize_model(
    model: BinaryModel[BinaryDomain],
    *,
    quantization_parameter: int | None,
) -> tuple[BinaryModel[BinaryDomain], float]:
    """Scale one real model into exactly representable integer coefficients.

    Args:
        model (BinaryModel[BinaryDomain]): Precision-aligned binary model.
        quantization_parameter (int | None): Positive precision or ``None`` for
            bounded automatic selection.

    Returns:
        tuple[BinaryModel[BinaryDomain], float]: Integer model and positive
        scale relating it to the original coefficients.

    Raises:
        ValueError: If an explicit precision is not positive.
    """

    values = [model.constant, *model.coefficients.values()]
    maximum = max(abs(value) for value in values)
    if math.isclose(maximum, 0.0, abs_tol=1e-12, rel_tol=0.0):
        return model, 1.0
    if quantization_parameter is None:
        decimal_exponents = (
            Decimal(str(float(value))).as_tuple().exponent for value in values
        )
        negative_exponents = (
            exponent
            for exponent in decimal_exponents
            if isinstance(exponent, int) and exponent < 0
        )
        epsilon = min(
            (10.0**exponent for exponent in negative_exponents),
            default=1.0,
        )
        precision = math.ceil(math.log2(maximum / epsilon))
        term_bits = math.ceil(math.log2(len(values))) if len(values) > 1 else 0
        quantization_parameter = precision + term_bits + 2
        if quantization_parameter > _MAX_EXACT_FLOAT_INTEGER_BITS:
            warnings.warn(
                "GAS automatic quantization was clamped to 54 bits so its "
                "integer coefficients remain exactly representable.",
                UserWarning,
                stacklevel=3,
            )
            quantization_parameter = _MAX_EXACT_FLOAT_INTEGER_BITS
    elif quantization_parameter <= 0:
        raise ValueError("quantization_parameter must be positive")
    integer_scale = 2 ** (quantization_parameter - 1)
    scale = float(integer_scale / maximum)
    coefficients = {
        variables: float(round(coefficient * scale))
        for variables, coefficient in model.coefficients.items()
    }
    return (
        BinaryModel.from_hubo(
            coefficients,
            float(round(model.constant * scale)),
            variables=model.variables,
        ),
        scale,
    )


def _objective_bounds(model: BinaryModel[BinaryDomain]) -> tuple[float, float]:
    """Return a safe coefficient-wise envelope of the binary objective.

    Args:
        model (BinaryModel[BinaryDomain]): Binary model to bound.

    Returns:
        tuple[float, float]: Inclusive lower and upper objective bounds.
    """

    coefficients = tuple(model.coefficients.values())
    minimum = model.constant + sum(value for value in coefficients if value < 0)
    maximum = model.constant + sum(value for value in coefficients if value > 0)
    return minimum, maximum


def _required_output_bits(
    model: BinaryModel[BinaryDomain],
    threshold: float,
) -> int:
    """Return the smallest safe two's-complement objective width.

    Args:
        model (BinaryModel[BinaryDomain]): Effective binary model.
        threshold (float): Threshold in the effective model's scale.

    Returns:
        int: Width whose sign bit cannot wrap over the objective envelope.
    """

    minimum, maximum = _objective_bounds(model)
    bound = max(abs(minimum - threshold), abs(maximum - threshold))
    if bound <= 0:
        return 2
    return max(2, math.floor(math.log2(bound)) + 2)


__all__ = ["CompiledGASExperiment", "GASExperiment"]
