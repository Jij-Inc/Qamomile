# ruff: noqa: F821 -- dimensions are Qamomile dependent shape binders.

"""Backend-neutral QAOA experiment composition."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import qamomile as qm
from qamomile.compilation import (
    CompilationTarget,
    CompiledProgram,
    ExecutionEndpoint,
    Invocation,
    RunRecord,
    SampleResult,
    compile as compile_program,
    run as run_program,
)
from qamomile.ir import Program
from qamomile.stdlib.variational import qaoa_state, uniform_superposition

from .model import BinaryModel, DecodedSamples, VarType

_Domain = TypeVar("_Domain", bound=VarType)


@qm.qkernel
def _qaoa_initial_sample(
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[width]]:  # type: ignore[valid-type]
    """Sample the uniform initial state without variational coordinates.

    Args:
        width (qm.Const[qm.Int]): Positive number of problem variables.

    Returns:
        qm.Shot[qm.Bits[width]]: One sampled source-ordered bit string.
    """

    return qm.measure(uniform_superposition(qm.qubits(width), width))


@qm.qkernel
def _qaoa_sample(
    width: qm.Const[qm.Int],  # type: ignore[valid-type]
    layers: qm.Const[qm.Int],  # type: ignore[valid-type]
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    gammas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
    betas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[width]]:  # type: ignore[valid-type]
    """Measure one QAOA state through a reusable dependent declaration.

    Args:
        width (qm.Const[qm.Int]): Positive number of problem variables.
        layers (qm.Const[qm.Int]): Positive number of QAOA layers.
        hamiltonian (qm.Const[qm.Observable]): Closed Ising cost operator.
        gammas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime cost angles.
        betas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime mixer angles.

    Returns:
        qm.Shot[qm.Bits[width]]: One sampled source-ordered bit string.

    Raises:
        ValueError: If a dependent width is not positive at specialization.
    """

    return qm.measure(qaoa_state(width, layers, hamiltonian, gammas, betas))


@dataclass(frozen=True, slots=True)
class OptimizationExperimentIdentity:
    """Exact typed identity of an optimization problem and executable plan."""

    kind: str
    problem_digest: str
    program_digest: str

    def __post_init__(self) -> None:
        """Reject empty kinds and malformed Rust-owned component identities."""

        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Q7670: optimization experiment kind must be non-empty")
        for name, value in (
            ("problem", self.problem_digest),
            ("program", self.program_digest),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"Q7670: optimization experiment {name} identity must be a "
                    "SHA-256 digest"
                )


@dataclass(frozen=True, slots=True)
class OptimizationRun(Generic[_Domain]):
    """One identified execution paired with its typed decoded distribution."""

    identity: OptimizationExperimentIdentity
    execution: RunRecord
    samples: DecodedSamples[_Domain]

    def __post_init__(self) -> None:
        """Reject execution records for a different experiment program."""

        if not isinstance(self.identity, OptimizationExperimentIdentity):
            raise TypeError("optimization run identity is invalid")
        if not isinstance(self.execution, RunRecord):
            raise TypeError("optimization run execution must be a RunRecord")
        if not isinstance(self.samples, DecodedSamples):
            raise TypeError("optimization run samples must be DecodedSamples")
        if self.execution.artifact.source_digest.hex() != self.identity.program_digest:
            raise ValueError(
                "Q7671: optimization run artifact does not match its experiment "
                "identity"
            )

    @property
    def invocation(self) -> Invocation:
        """Return the exact normalized coordinates and execution policy."""

        return self.execution.invocation


@dataclass(frozen=True, slots=True)
class QAOAExperiment(Generic[_Domain]):
    """A fixed problem and layer topology ready for repeated QAOA launches.

    The immutable experiment closes only compile-time problem structure. The
    variational coordinates remain the aggregate runtime parameter ABI, so a
    caller may reuse :attr:`program` across optimization iterations and target
    it at any compatible execution endpoint.

    Args:
        model (BinaryModel[_Domain]): Nonempty optimization problem.
        layers (int): Nonnegative number of QAOA layers. Zero samples the
            uniform initial state and requires empty gamma and beta sequences.

    Raises:
        TypeError: If the model or layer count has an invalid type.
        ValueError: If layers is negative or the model has no variables.
    """

    model: BinaryModel[_Domain]
    layers: int = 1
    program: Program = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate topology and build the initial-state or layered Program.

        Raises:
            TypeError: If model is not a BinaryModel or layers is not an integer.
            ValueError: If layers is negative or the model has no variables.
        """

        if not isinstance(self.model, BinaryModel):
            raise TypeError("model must be a qamomile.optimization.BinaryModel")
        if type(self.layers) is not int:
            raise TypeError("layers must be an integer")
        if self.layers < 0:
            raise ValueError("layers must be nonnegative")
        if self.model.num_variables <= 0:
            raise ValueError("QAOA requires at least one model variable")
        program = (
            _qaoa_initial_sample.specialize(width=self.model.num_variables)
            if self.layers == 0
            else _qaoa_sample.specialize(
                **{
                    "width": self.model.num_variables,
                    "layers": self.layers,
                    "hamiltonian": self.model.hamiltonian,
                }
            )
        )
        object.__setattr__(self, "program", program)

    @property
    def identity(self) -> OptimizationExperimentIdentity:
        """Return the exact problem/program pair defining this experiment."""

        return OptimizationExperimentIdentity(
            "qamomile.qaoa.v1",
            self.model.semantic_digest,
            self.program.semantic_digest,
        )

    def compile(self, *, to: CompilationTarget) -> CompiledQAOAExperiment[_Domain]:
        """Compile this fixed experiment once for repeated target launches."""

        return CompiledQAOAExperiment(self, compile_program(self.program, to=to))

    def run(
        self,
        *,
        on: ExecutionEndpoint,
        gammas: Sequence[float],
        betas: Sequence[float],
        shots: int,
        seed: int = 0,
    ) -> OptimizationRun[_Domain]:
        """Compile, execute, and decode one variational coordinate.

        Args:
            on: Backend-neutral execution endpoint.
            gammas: One cost angle per static layer.
            betas: One mixer angle per static layer.
            shots: Number of observations requested from the endpoint.
            seed: Optional endpoint seed.

        Returns:
            The complete compilation record and model-domain sample set.

        Raises:
            TypeError: If an angle vector is not a real numeric sequence.
            ValueError: If an angle vector has the wrong length or is non-finite.
        """

        gamma_values = _angles("gammas", gammas, self.layers)
        beta_values = _angles("betas", betas, self.layers)
        return self.compile(to=on.target)._run_normalized(
            on=on,
            gammas=gamma_values,
            betas=beta_values,
            shots=shots,
            seed=seed,
        )


@dataclass(frozen=True, slots=True)
class CompiledQAOAExperiment(Generic[_Domain]):
    """One immutable QAOA plan compiled for a concrete artifact target."""

    experiment: QAOAExperiment[_Domain]
    artifact: CompiledProgram
    identity: OptimizationExperimentIdentity = field(init=False)

    def __post_init__(self) -> None:
        """Verify that the target artifact belongs to the retained experiment."""

        if not isinstance(self.experiment, QAOAExperiment):
            raise TypeError("compiled QAOA experiment requires a QAOAExperiment")
        if not isinstance(self.artifact, CompiledProgram):
            raise TypeError("compiled QAOA experiment requires an CompiledProgram")
        identity = self.experiment.identity
        if self.artifact.source_digest.hex() != identity.program_digest:
            raise ValueError("Q7670: compiled artifact does not match QAOA experiment")
        object.__setattr__(self, "identity", identity)

    def run(
        self,
        *,
        on: ExecutionEndpoint,
        gammas: Sequence[float],
        betas: Sequence[float],
        shots: int,
        seed: int = 0,
    ) -> OptimizationRun[_Domain]:
        """Execute one coordinate without recompiling the experiment."""

        return self._run_normalized(
            on=on,
            gammas=_angles("gammas", gammas, self.experiment.layers),
            betas=_angles("betas", betas, self.experiment.layers),
            shots=shots,
            seed=seed,
        )

    def _run_normalized(
        self,
        *,
        on: ExecutionEndpoint,
        gammas: tuple[float, ...],
        betas: tuple[float, ...],
        shots: int,
        seed: int,
    ) -> OptimizationRun[_Domain]:
        """Execute coordinates through the compiled program's parameter ABI.

        Args:
            on (ExecutionEndpoint): Local or authorized remote endpoint.
            gammas (tuple[float, ...]): One validated cost angle per layer.
            betas (tuple[float, ...]): One validated mixer angle per layer.
            shots (int): Number of observations.
            seed (int): Endpoint random seed.

        Returns:
            OptimizationRun[_Domain]: Identified execution and decoded samples.

        Raises:
            TypeError: If the program returns a non-sampling result.
        """

        coordinates = {"gammas": gammas, "betas": betas}
        execution = run_program(
            self.artifact,
            on=on,
            parameters={
                slot.name: coordinates[slot.name]
                for slot in self.artifact.parameter_abi.slots
            },
            shots=shots,
            seed=seed,
        )
        if not isinstance(execution.result, SampleResult):
            raise TypeError("QAOA sample program returned a non-sampling result")
        return OptimizationRun(
            identity=self.identity,
            execution=execution,
            samples=self.experiment.model.decode(execution.result),
        )


def _angles(name: str, values: object, layers: int) -> tuple[float, ...]:
    """Validate one fixed-shape finite variational coordinate."""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a real numeric sequence")
    if len(values) != layers:
        raise ValueError(f"{name} must contain exactly {layers} values")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain only real numbers")
        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise ValueError(f"{name} must contain only finite values")
        normalized.append(coordinate)
    return tuple(normalized)


__all__ = [
    "CompiledQAOAExperiment",
    "OptimizationExperimentIdentity",
    "OptimizationRun",
    "QAOAExperiment",
]
