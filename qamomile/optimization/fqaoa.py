# ruff: noqa: F821 -- dimensions are Qamomile dependent shape binders.

"""Typed fermionic-QAOA experiments over an immutable OMMX problem."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import ommx.v1

import qamomile as qm
from qamomile.compilation import (
    CompilationTarget,
    CompiledProgram,
    ExecutionEndpoint,
    SampleResult,
    compile as compile_program,
    run as run_program,
)
from qamomile.ir import Program
from qamomile.observable import Hamiltonian, _hamiltonian_from_native_header
from qamomile.stdlib.variational import (
    fqaoa_layers,
    fqaoa_state,
    initial_occupations,
)

from .model import BinaryDomain, DecodedSample, DecodedSamples, VarType
from .ommx import OMMXProblem
from .qaoa import OptimizationExperimentIdentity, OptimizationRun, _angles


@qm.qkernel
def _fqaoa_sample_without_givens(
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
    particles: qm.Const[qm.Int],  # type: ignore[valid-type]
    layers: qm.Const[qm.Int],  # type: ignore[valid-type]
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    hopping: qm.Param[qm.Real],  # type: ignore[valid-type]
    gammas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
    betas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[width]]:  # type: ignore[valid-type]
    """Measure an extreme-sector FQAOA state without Givens data.

    Args:
        width (qm.Const[qm.Int]): Positive number of fermionic modes.
        particles (qm.Const[qm.Int]): Occupied leading modes.
        layers (qm.Const[qm.Int]): Positive number of FQAOA layers.
        hamiltonian (qm.Const[qm.Observable]): Closed cost operator.
        hopping (qm.Param[qm.Real]): Runtime hopping coefficient.
        gammas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime cost angles.
        betas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime mixer angles.

    Returns:
        qm.Shot[qm.Bits[width]]: One sampled occupation row.

    Raises:
        ValueError: If a dependent width is not positive at specialization.
    """

    state = initial_occupations(qm.qubits(width), particles)
    state = fqaoa_layers(
        state,
        width,
        layers,
        hamiltonian,
        hopping,
        gammas,
        betas,
    )
    return qm.measure(state)


@qm.qkernel
def _fqaoa_sample(
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
    particles: qm.Const[qm.Int],  # type: ignore[valid-type]
    givens_count: qm.Const[qm.Int],  # type: ignore[valid-type]
    pairs: qm.Const[  # type: ignore[valid-type]
        qm.Vector[tuple[qm.Int, qm.Int], givens_count]
    ],
    givens_angles: qm.Const[  # type: ignore[valid-type]
        qm.Vector[qm.Real, givens_count]
    ],
    layers: qm.Const[qm.Int],  # type: ignore[valid-type]
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    hopping: qm.Param[qm.Real],  # type: ignore[valid-type]
    gammas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
    betas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[width]]:  # type: ignore[valid-type]
    """Measure an FQAOA state with closed topology and preparation.

    Args:
        width (qm.Const[qm.Int]): Positive number of fermionic modes.
        particles (qm.Const[qm.Int]): Occupied-particle count.
        givens_count (qm.Const[qm.Int]): Number of preparation rotations.
        pairs (qm.Const[qm.Vector[tuple[qm.Int, qm.Int], givens_count]]):
            Closed mode pairs for the Givens rotations.
        givens_angles (qm.Const[qm.Vector[qm.Real, givens_count]]): Closed
            preparation angles.
        layers (qm.Const[qm.Int]): Positive number of FQAOA layers.
        hamiltonian (qm.Const[qm.Observable]): Closed cost operator.
        hopping (qm.Param[qm.Real]): Runtime hopping coefficient.
        gammas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime cost angles.
        betas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime mixer angles.

    Returns:
        qm.Shot[qm.Bits[width]]: One sampled occupation row.

    Raises:
        ValueError: If a dependent width is not positive at specialization.
    """

    return qm.measure(
        fqaoa_state(
            width,
            particles,
            givens_count,
            pairs,
            givens_angles,
            layers,
            hamiltonian,
            hopping,
            gammas,
            betas,
        )
    )


@dataclass(frozen=True, slots=True)
class FQAOAExperiment:
    """A fixed fermionic sector and cost topology ready for repeated launches.

    The OMMX decision variables must be binary and form a dense rectangular
    ``(site, orbital)`` subscript grid. The grid defines a particle-preserving
    periodic mode order independently of arbitrary decision-variable IDs.
    Auxiliary penalty bits are rejected because they have no fermionic mode
    geometry and cannot be mixed in the declared particle-number sector.
    """

    problem: OMMXProblem
    particles: int
    layers: int = 1
    num_sites: int = field(init=False)
    num_orbitals: int = field(init=False)
    variable_order: tuple[int, ...] = field(init=False)
    givens_pairs: tuple[tuple[int, int], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    givens_angles: tuple[float, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    hamiltonian: Hamiltonian = field(init=False, repr=False, compare=False)
    program: Program = field(init=False, repr=False, compare=False)
    _native_plan: object = field(init=False, repr=False, compare=False)
    _decode_row_limit: int = field(init=False, repr=False, compare=False)
    _decode_cell_limit: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the fermionic geometry and build one closed program."""

        if not isinstance(self.problem, OMMXProblem):
            raise TypeError("problem must be a qamomile.optimization.OMMXProblem")
        if type(self.particles) is not int:
            raise TypeError("particles must be an integer")
        if type(self.layers) is not int:
            raise TypeError("layers must be an integer")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        instance = self.problem.original_instance
        positions = _variable_positions(instance)
        if len(positions) < 2:
            raise ValueError("FQAOA requires at least two fermionic modes")
        if not 0 <= self.particles <= len(positions):
            raise ValueError("particles must lie between zero and the mode count")
        _validate_particle_constraints(instance, self.particles)
        derive = getattr(self.problem.model._native_model, "fqaoa_plan", None)
        if not callable(derive):
            raise TypeError("Q7590: native FQAOA planning is unavailable")
        plan = derive(
            [format(variable.id, "x") for variable in instance.decision_variables],
            list(positions),
            self.particles,
        )
        (
            num_sites,
            num_orbitals,
            variable_order,
            givens_pairs,
            givens_angles,
            hamiltonian,
            row_limit,
            cell_limit,
        ) = _decode_native_plan(plan, self.problem.model.variables)
        const: dict[str, object] = {
            "width": len(variable_order),
            "particles": self.particles,
            "layers": self.layers,
            "hamiltonian": hamiltonian,
        }
        if givens_pairs:
            const["givens_count"] = len(givens_pairs)
            const["pairs"] = givens_pairs
            const["givens_angles"] = givens_angles
        sampler = _fqaoa_sample if givens_pairs else _fqaoa_sample_without_givens
        program = sampler.specialize(**const)
        object.__setattr__(self, "num_sites", num_sites)
        object.__setattr__(self, "num_orbitals", num_orbitals)
        object.__setattr__(self, "variable_order", variable_order)
        object.__setattr__(self, "givens_pairs", givens_pairs)
        object.__setattr__(self, "givens_angles", givens_angles)
        object.__setattr__(self, "hamiltonian", hamiltonian)
        object.__setattr__(self, "program", program)
        object.__setattr__(self, "_native_plan", plan)
        object.__setattr__(self, "_decode_row_limit", row_limit)
        object.__setattr__(self, "_decode_cell_limit", cell_limit)

    @property
    def num_qubits(self) -> int:
        """Return the number of physical fermionic modes."""

        return len(self.variable_order)

    @property
    def identity(self) -> OptimizationExperimentIdentity:
        """Return the exact problem/program pair defining this experiment."""

        return OptimizationExperimentIdentity(
            "qamomile.fqaoa.v1",
            self.problem.model.semantic_digest,
            self.program.semantic_digest,
        )

    def compile(self, *, to: CompilationTarget) -> CompiledFQAOAExperiment:
        """Compile this fixed experiment once for repeated target launches."""

        return CompiledFQAOAExperiment(self, compile_program(self.program, to=to))

    @classmethod
    def from_instance(
        cls,
        instance: ommx.v1.Instance,
        *,
        particles: int,
        layers: int = 1,
        uniform_penalty_weight: float | None = None,
        penalty_weights: Mapping[int, float] | None = None,
    ) -> FQAOAExperiment:
        """Normalize an OMMX instance and construct its FQAOA experiment."""

        return cls(
            OMMXProblem.from_instance(
                instance,
                uniform_penalty_weight=uniform_penalty_weight,
                penalty_weights=penalty_weights,
            ),
            particles,
            layers,
        )

    def run(
        self,
        *,
        on: ExecutionEndpoint,
        gammas: Sequence[float],
        betas: Sequence[float],
        shots: int,
        hopping: float = 1.0,
        seed: int = 0,
    ) -> OptimizationRun[BinaryDomain]:
        """Compile, execute, and decode one FQAOA coordinate."""

        gamma_values = _angles("gammas", gammas, self.layers)
        beta_values = _angles("betas", betas, self.layers)
        hopping_value = _finite_real("hopping", hopping)
        return self.compile(to=on.target)._run_normalized(
            on=on,
            gammas=gamma_values,
            betas=beta_values,
            shots=shots,
            hopping=hopping_value,
            seed=seed,
        )

    def _decode(self, result: SampleResult) -> DecodedSamples[BinaryDomain]:
        """Map physical mode-order counts back to original decision IDs."""

        if result.width != self.num_qubits:
            raise ValueError(
                f"sample width {result.width} does not match FQAOA width "
                f"{self.num_qubits}"
            )
        rows = tuple(
            (bits, occurrences)
            for bits, occurrences in result.counts.items()
            if occurrences > 0
        )
        if (
            len(rows) > self._decode_row_limit
            or len(rows) * self.num_qubits > self._decode_cell_limit
        ):
            raise ValueError("Q7599: FQAOA decoding exceeds its work budget")
        decode = getattr(self._native_plan, "decode", None)
        if not callable(decode):
            raise TypeError("Q7590: native FQAOA planning is unavailable")
        samples = _decode_native_samples(
            decode(rows),
            self.problem.model.variables,
        )
        try:
            return DecodedSamples[BinaryDomain](VarType.BINARY, samples)
        except (TypeError, ValueError) as error:
            raise TypeError("Q7590: malformed native FQAOA plan") from error

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Reconstruct the private native plan from stable public inputs."""

        return type(self), (self.problem, self.particles, self.layers)


@dataclass(frozen=True, slots=True)
class CompiledFQAOAExperiment:
    """One immutable FQAOA plan compiled for a concrete artifact target."""

    experiment: FQAOAExperiment
    artifact: CompiledProgram
    identity: OptimizationExperimentIdentity = field(init=False)

    def __post_init__(self) -> None:
        """Verify that the target artifact belongs to the retained experiment."""

        if not isinstance(self.experiment, FQAOAExperiment):
            raise TypeError("compiled FQAOA experiment requires an FQAOAExperiment")
        if not isinstance(self.artifact, CompiledProgram):
            raise TypeError("compiled FQAOA experiment requires an CompiledProgram")
        identity = self.experiment.identity
        if self.artifact.source_digest.hex() != identity.program_digest:
            raise ValueError("Q7590: compiled artifact does not match FQAOA experiment")
        object.__setattr__(self, "identity", identity)

    def run(
        self,
        *,
        on: ExecutionEndpoint,
        gammas: Sequence[float],
        betas: Sequence[float],
        shots: int,
        hopping: float = 1.0,
        seed: int = 0,
    ) -> OptimizationRun[BinaryDomain]:
        """Execute one coordinate without recompiling the experiment."""

        return self._run_normalized(
            on=on,
            gammas=_angles("gammas", gammas, self.experiment.layers),
            betas=_angles("betas", betas, self.experiment.layers),
            shots=shots,
            hopping=_finite_real("hopping", hopping),
            seed=seed,
        )

    def _run_normalized(
        self,
        *,
        on: ExecutionEndpoint,
        gammas: tuple[float, ...],
        betas: tuple[float, ...],
        shots: int,
        hopping: float,
        seed: int,
    ) -> OptimizationRun[BinaryDomain]:
        """Execute already validated fixed-shape FQAOA coordinates."""

        execution = run_program(
            self.artifact,
            on=on,
            parameters={
                "hopping": hopping,
                "gammas": gammas,
                "betas": betas,
            },
            shots=shots,
            seed=seed,
        )
        if not isinstance(execution.result, SampleResult):
            raise TypeError("FQAOA sample program returned a non-sampling result")
        return OptimizationRun(
            identity=self.identity,
            execution=execution,
            samples=self.experiment._decode(execution.result),
        )


def _variable_positions(
    instance: ommx.v1.Instance,
) -> tuple[tuple[int, int], ...]:
    """Validate binary two-dimensional OMMX decision-variable coordinates."""

    positions: list[tuple[int, int]] = []
    if not instance.decision_variables:
        raise ValueError("FQAOA requires at least one decision variable")
    for variable in instance.decision_variables:
        if variable.kind != variable.BINARY:
            raise ValueError("FQAOA decision variables must all be binary")
        if len(variable.subscripts) != 2:
            raise ValueError(
                "FQAOA variables must each have exactly two integer subscripts "
                "(site, orbital)"
            )
        site, orbital = variable.subscripts
        if type(site) is not int or type(orbital) is not int:
            raise TypeError("FQAOA variable subscripts must be integers")
        if site < 0 or orbital < 0:
            raise ValueError("FQAOA variable subscripts must be non-negative")
        positions.append((site, orbital))
    return tuple(positions)


def _validate_particle_constraints(
    instance: ommx.v1.Instance,
    particles: int,
) -> None:
    """Reject an explicit total-cardinality constraint for another sector."""

    variable_ids = {variable.id for variable in instance.decision_variables}
    for constraint in instance.constraints:
        function = constraint.function
        if (
            constraint.equality != ommx.v1.Constraint.EQUAL_TO_ZERO
            or function.degree() > 1
            or set(function.linear_terms) != variable_ids
        ):
            continue
        coefficients = tuple(function.linear_terms.values())
        if not coefficients:
            continue
        coefficient = coefficients[0]
        if not math.isclose(abs(coefficient), 1.0) or not all(
            math.isclose(value, coefficient) for value in coefficients
        ):
            continue
        cardinality = -function.constant_term / coefficient
        if not math.isclose(cardinality, particles):
            raise ValueError(
                "particles is inconsistent with cardinality constraint "
                f"{constraint.id}: expected {cardinality:g}, got {particles}"
            )


def _decode_native_plan(
    plan: object,
    variables: tuple[int, ...],
) -> tuple[
    int,
    int,
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    tuple[float, ...],
    Hamiltonian,
    int,
    int,
]:
    """Validate one transient native FQAOA plan projection."""

    num_sites = getattr(plan, "num_sites", None)
    num_orbitals = getattr(plan, "num_orbitals", None)
    raw_order = getattr(plan, "variable_order", None)
    raw_pairs = getattr(plan, "givens_pairs", None)
    raw_angles = getattr(plan, "givens_angles", None)
    raw_hamiltonian = getattr(plan, "hamiltonian", None)
    row_limit = getattr(plan, "max_decode_rows", None)
    cell_limit = getattr(plan, "max_decode_cells", None)
    width = len(variables)
    if (
        type(num_sites) is not int
        or num_sites <= 0
        or type(num_orbitals) is not int
        or num_orbitals <= 0
        or num_sites * num_orbitals != width
        or not isinstance(raw_order, list)
        or any(type(position) is not int for position in raw_order)
        or sorted(raw_order) != list(range(width))
        or not isinstance(raw_pairs, list)
        or not isinstance(raw_angles, list)
        or len(raw_pairs) != len(raw_angles)
        or type(row_limit) is not int
        or row_limit <= 0
        or type(cell_limit) is not int
        or cell_limit <= 0
    ):
        raise TypeError("Q7590: malformed native FQAOA plan")
    pairs: list[tuple[int, int]] = []
    for pair in raw_pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(type(index) is not int for index in pair)
            or pair[1] != pair[0] + 1
            or not 0 <= pair[0] < pair[1] < width
        ):
            raise TypeError("Q7590: malformed native FQAOA plan")
        pairs.append(pair)
    if any(
        type(angle) is not float or not math.isfinite(angle) for angle in raw_angles
    ):
        raise TypeError("Q7590: malformed native FQAOA plan")
    try:
        variable_order = tuple(variables[position] for position in raw_order)
        hamiltonian = _hamiltonian_from_native_header(raw_hamiltonian)
    except (IndexError, TypeError) as error:
        raise TypeError("Q7590: malformed native FQAOA plan") from error
    if hamiltonian.num_qubits != width:
        raise TypeError("Q7590: malformed native FQAOA plan")
    return (
        num_sites,
        num_orbitals,
        variable_order,
        tuple(pairs),
        tuple(raw_angles),
        hamiltonian,
        row_limit,
        cell_limit,
    )


def _decode_native_samples(
    projection: object,
    variables: tuple[int, ...],
) -> tuple[DecodedSample, ...]:
    """Validate one native FQAOA decode projection."""

    if not isinstance(projection, list):
        raise TypeError("Q7590: malformed native FQAOA plan")
    samples: list[DecodedSample] = []
    for row in projection:
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[0], list)
            or len(row[0]) != len(variables)
            or any(type(value) is not int for value in row[0])
            or not isinstance(row[1], float)
            or type(row[2]) is not int
        ):
            raise TypeError("Q7590: malformed native FQAOA plan")
        samples.append(
            DecodedSample._from_native(
                tuple(zip(variables, row[0], strict=True)),
                row[1],
                row[2],
            )
        )
    return tuple(samples)


def _finite_real(name: str, value: object) -> float:
    """Normalize one finite non-Boolean real coordinate."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


__all__ = ["CompiledFQAOAExperiment", "FQAOAExperiment"]
