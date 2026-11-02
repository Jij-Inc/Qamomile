"""Immutable eigenstate-filter experiments over Rust-owned optimization models.

Thresholds close ordinary stdlib source into portable programs. Rust owns model
conversion, phase verification, block planning, compilation, success statistics,
and decoded energies. Classical threshold searches remain application code.
"""

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
    RunRecord,
    SampleResult,
)
from qamomile.ir import Program
from qamomile.observable import Hamiltonian
from qamomile.stdlib import BlockEncoding, CertifiedQSVTPolynomial, QSVTPhases
from qamomile.stdlib.pauli import pauli_block_encoding
from qamomile.stdlib.qsvt_eigenstate_filter import (
    qsvt_filter_projector,
    synthesize_sign_filter,
)
from qamomile.stdlib.signals import Signal, SuccessEstimate, estimate_success, sample

from .model import BinaryModel, DecodedSamples, VarType
from .qaoa import OptimizationExperimentIdentity

_Domain = TypeVar("_Domain", bound=VarType)


@dataclass(frozen=True, slots=True)
class QSVTFilterRun(Generic[_Domain]):
    """One identified execution and its conditional problem samples.

    Attributes:
        identity (OptimizationExperimentIdentity): Exact model/program identity.
        execution (RunRecord): Complete compiled execution record.
        success (SuccessEstimate): Native-derived success fraction and interval.
        samples (DecodedSamples | None): Successful model-domain observations,
            or None when no shot succeeded. Failures remain in execution.result.
    """

    identity: OptimizationExperimentIdentity
    execution: RunRecord
    success: SuccessEstimate
    samples: DecodedSamples[_Domain] | None

    def __post_init__(self) -> None:
        """Reject mismatched artifacts and conditional sample counts.

        Raises:
            TypeError: If a supplied value has the wrong public type.
            ValueError: If the artifact, success, or observation count differs.
        """
        if not isinstance(self.identity, OptimizationExperimentIdentity):
            raise TypeError("filter run identity is invalid")
        if not isinstance(self.execution, RunRecord):
            raise TypeError("filter run execution must be a RunRecord")
        if self.execution.artifact.source_digest.hex() != self.identity.program_digest:
            raise ValueError("filter run artifact does not match its experiment")
        if not isinstance(self.execution.result, SampleResult):
            raise TypeError("filter run requires a sampling result")
        if not isinstance(self.success, SuccessEstimate):
            raise TypeError("filter run success must be a SuccessEstimate")
        if self.success != estimate_success(self.execution.result):
            raise ValueError("filter run success does not match its execution")
        if self.samples is not None and not isinstance(self.samples, DecodedSamples):
            raise TypeError("filter run samples must be DecodedSamples or None")
        observations = 0 if self.samples is None else self.samples.total_occurrences
        if observations != self.success.successes:
            raise ValueError("filter run samples do not match its successful shots")

    @property
    def success_probability(self) -> float:
        """Return the fraction of all shots passing the complete filter."""
        return self.success.probability


@dataclass(frozen=True, slots=True)
class QSVTEigenstateFilterExperiment(Generic[_Domain]):
    r"""Close an energy threshold into a portable eigenstate-filter program.

    The positive sign approximation keeps energies below mu through
    ``(I - p((H - mu I) / alpha')) / 2``. Extracting the real polynomial
    component prevents the unselected complex QSP response from contributing
    to success. Use OMMXProblem.model for OMMX problems and evaluate the
    successful samples with that problem separately.

    Args:
        model (BinaryModel): Immutable BINARY or SPIN objective.
        mu (float): Finite threshold in model energy units, including its offset.
        polynomial (CertifiedQSVTPolynomial | None): Explicit Rust-replayed
            certificate, mutually exclusive with phases.
        phases (Sequence[float] | QSVTPhases | None): Projector phases without
            an asserted sign-approximation error.
        degree (int): Odd synthesis degree, default 61; ignored with phase input.
        delta (float): Positive pyqsp transition sharpness, default 20.
        scale (float): Nonzero polynomial scale, default 1.10. Positive keeps
            lower energies and negative keeps higher energies.

    Raises:
        TypeError: If model, threshold, or phase input has the wrong type.
        ValueError: If the model is empty, threshold is nonfinite, phase inputs
            conflict, or polynomial generation or native verification fails.
    """

    model: BinaryModel[_Domain]
    mu: float
    polynomial: CertifiedQSVTPolynomial | None = field(default=None, kw_only=True)
    phases: Sequence[float] | QSVTPhases | None = field(default=None, kw_only=True)
    degree: int = field(default=61, kw_only=True)
    delta: float = field(default=20.0, kw_only=True)
    scale: float = field(default=1.10, kw_only=True)
    energy_offset: float = field(init=False)
    encoding: BlockEncoding = field(init=False, repr=False, compare=False)
    shifted_encoding: BlockEncoding = field(init=False, repr=False, compare=False)
    filter_encoding: BlockEncoding = field(init=False, repr=False, compare=False)
    program: Program = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build source from canonical native models and checked filter data.

        Raises:
            TypeError: If host fields or required native support are invalid.
            ValueError: If threshold arithmetic or the filter contract fails.
            ImportError: If optional default phase synthesis is unavailable.
        """
        if not isinstance(self.model, BinaryModel):
            raise TypeError("model must be a qamomile.optimization.BinaryModel")
        if self.model.num_variables <= 0:
            raise ValueError("eigenstate filtering requires at least one variable")
        if isinstance(self.mu, bool) or not isinstance(self.mu, (int, float)):
            raise TypeError("mu must be a real number")
        mu = float(self.mu)
        if not math.isfinite(mu):
            raise ValueError("mu must be finite")
        if self.phases is not None and self.polynomial is not None:
            raise ValueError("phases and polynomial are mutually exclusive")
        phases = self.phases
        if phases is not None and not isinstance(phases, QSVTPhases):
            phases = QSVTPhases(tuple(phases))
        polynomial = self.polynomial
        if phases is None and polynomial is None:
            polynomial = synthesize_sign_filter(
                degree=self.degree, delta=self.delta, scale=self.scale
            )
        cost = self.model.hamiltonian
        encoding = pauli_block_encoding(
            Hamiltonian(cost.terms, num_qubits=cost.num_qubits)
        )
        shifted = pauli_block_encoding(cost - mu)
        filtered = qsvt_filter_projector(shifted, phases=phases, polynomial=polynomial)
        width = self.model.num_variables

        @qm.qkernel
        def uniform(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]  # noqa: F821
            """Prepare the uniform superposition on the logical model register.

            Args:
                state (qm.Q[qm.Bits[width]]): Owned zero-state model register.

            Returns:
                qm.Q[qm.Bits[width]]: Uniform model state.
            """
            for i in range(width):
                state = qm.h(state, target=i)
            return state

        probe = sample(Signal.from_block_encoding(filtered), source=uniform)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "polynomial", polynomial)
        object.__setattr__(self, "energy_offset", cost.constant.real)
        object.__setattr__(self, "encoding", encoding)
        object.__setattr__(self, "shifted_encoding", shifted)
        object.__setattr__(self, "filter_encoding", filtered)
        object.__setattr__(self, "program", probe.specialize())

    @property
    def normalization(self) -> float:
        """Return the nonconstant cost encoding's spectrum-bound scale."""
        return self.encoding.normalization

    @property
    def num_ancilla_bits(self) -> int:
        """Return the complete filter workspace width for this threshold."""
        return self.filter_encoding.signal_width

    @property
    def identity(self) -> OptimizationExperimentIdentity:
        """Return Rust-owned model and program digests for this experiment."""
        return OptimizationExperimentIdentity(
            "qamomile.qsvt_eigenstate_filter.v1",
            self.model.semantic_digest,
            self.program.semantic_digest,
        )

    def decode(self, result: SampleResult) -> DecodedSamples[_Domain] | None:
        """Decode successful payloads in the original model's variable domain.

        Args:
            result (SampleResult): Typed success-first probe observations.

        Returns:
            DecodedSamples | None: Native-decoded successful assignments, or
            None when the complete filter accepted no observation.

        Raises:
            TypeError: If the result lacks the success-first typed ABI.
            ValueError: If payload width differs from the model width.
        """
        if not isinstance(result, SampleResult):
            raise TypeError("result must be a qamomile.SampleResult")
        if result.width != self.model.num_variables + 1:
            raise ValueError("filter result payload width does not match its model")
        success = estimate_success(result)
        if success.successes == 0:
            return None
        conditional = SampleResult.from_counts(
            backend=result.backend,
            counts={
                bits[1:]: count for bits, count in result.counts.items() if bits[0]
            },
            worker_pid=result.worker_pid,
            expected_width=self.model.num_variables,
        )
        return self.model.decode(conditional)

    def success_probability(self, result: SampleResult) -> float:
        """Estimate the fraction of all shots passing the complete filter.

        Args:
            result (SampleResult): Typed success-first probe observations.

        Returns:
            float: Native-derived observed success fraction.

        Raises:
            TypeError: If result lacks the success-first typed ABI.
            ValueError: If payload width differs from the model width.
        """
        if not isinstance(result, SampleResult):
            raise TypeError("result must be a qamomile.SampleResult")
        if result.width != self.model.num_variables + 1:
            raise ValueError("filter result payload width does not match its model")
        return estimate_success(result).probability

    def compile(
        self, *, to: CompilationTarget
    ) -> CompiledQSVTEigenstateFilterExperiment[_Domain]:
        """Compile this threshold for repeated launches on one target.

        Args:
            to (CompilationTarget): Artifact target for the source program.

        Returns:
            CompiledQSVTEigenstateFilterExperiment: Reusable compiled experiment.
        """
        return CompiledQSVTEigenstateFilterExperiment(
            self, qm.compile(self.program, to=to)
        )

    def run(
        self, *, on: ExecutionEndpoint, shots: int, seed: int = 0
    ) -> QSVTFilterRun[_Domain]:
        """Compile, execute, and decode this threshold.

        Args:
            on (ExecutionEndpoint): Compatible execution endpoint.
            shots (int): Positive observation count.
            seed (int): Endpoint sampling seed, default zero.

        Returns:
            QSVTFilterRun: Full execution and conditional problem samples.
        """
        return self.compile(to=on.target).run(on=on, shots=shots, seed=seed)


@dataclass(frozen=True, slots=True)
class CompiledQSVTEigenstateFilterExperiment(Generic[_Domain]):
    """One immutable threshold experiment compiled for an execution target.

    Args:
        experiment (QSVTEigenstateFilterExperiment): Retained source experiment.
        artifact (CompiledProgram): Artifact for exactly that program.
    """

    experiment: QSVTEigenstateFilterExperiment[_Domain]
    artifact: CompiledProgram

    def __post_init__(self) -> None:
        """Reject an artifact compiled from a different experiment.

        Raises:
            TypeError: If a constructor argument has the wrong type.
            ValueError: If source and artifact identities differ.
        """
        if not isinstance(self.experiment, QSVTEigenstateFilterExperiment):
            raise TypeError("compiled filter requires a QSVTEigenstateFilterExperiment")
        if not isinstance(self.artifact, CompiledProgram):
            raise TypeError("compiled filter requires a CompiledProgram")
        if self.artifact.source_digest.hex() != self.experiment.identity.program_digest:
            raise ValueError("compiled artifact does not match filter experiment")

    @property
    def identity(self) -> OptimizationExperimentIdentity:
        """Return the retained source experiment identity."""
        return self.experiment.identity

    def run(
        self, *, on: ExecutionEndpoint, shots: int, seed: int = 0
    ) -> QSVTFilterRun[_Domain]:
        """Execute the retained artifact and preserve its success denominator.

        Args:
            on (ExecutionEndpoint): Endpoint for the compiled target.
            shots (int): Positive observation count.
            seed (int): Endpoint sampling seed, default zero.

        Returns:
            QSVTFilterRun: Execution, native success estimate, and decoded samples.

        Raises:
            TypeError: If the probe unexpectedly returns a nonsampling result.
        """
        execution = qm.run(self.artifact, on=on, shots=shots, seed=seed)
        if not isinstance(execution.result, SampleResult):
            raise TypeError("eigenstate filter returned a nonsampling result")
        return QSVTFilterRun(
            self.identity,
            execution,
            estimate_success(execution.result),
            self.experiment.decode(execution.result),
        )


__all__ = [
    "CompiledQSVTEigenstateFilterExperiment",
    "QSVTEigenstateFilterExperiment",
    "QSVTFilterRun",
]
