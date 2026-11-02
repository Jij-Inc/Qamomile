"""Implement QURI Parts circuit realizations and sampling endpoints."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

from qamomile.circuit import (
    RESULT_POSTPROCESSING_FORMAT,
    RESULT_POSTPROCESSING_METADATA,
    CircuitProgram,
    circuit_result_postprocessing,
    lower_circuit_target,
)
from qamomile.compilation import (
    CapabilitySnapshot,
    CompiledProgram,
    CompletedJob,
    Invocation,
    Job,
    LoweredProgram,
    RealizationCapability,
    SampleResult,
)
from qamomile.effects import Requirement
from qamomile.errors import ExecutionError
from qamomile.ir import FrozenMapping, Program
from qamomile.local_execution import launch_local_sample
from qamomile.lowering import RealizationOptions

from ._materialization import (
    _decode_native_counts,
    _execute_local as _execute_local,
    _materialize_circuit,
    _validate_block,
    load_circuit,
    load_logical_circuit,
)
from ._profile import FORMAT as FORMAT

_CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.PARAMETERIZED_EXECUTION,
        Requirement.QUANTUM,
        Requirement.MEASUREMENT,
        Requirement.DISCARD,
        Requirement.ROTATION,
    }
)
_REALIZATIONS = frozenset(
    {
        RealizationCapability.RZZ,
        RealizationCapability.MCX,
        RealizationCapability.CONTROLLED_SWAP,
    }
)


@dataclass(frozen=True, slots=True)
class QuriParts:
    """Compile circuit-shaped QCore to a portable logical-circuit payload.

    Args:
        realizations (RealizationOptions): Exact operation-realization overrides.
    """

    realizations: RealizationOptions = RealizationOptions()

    @property
    def name(self) -> str:
        """Return the stable artifact format identifier."""

        return FORMAT

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return QURI Parts target capabilities."""

        return CapabilitySnapshot(
            FORMAT,
            _CAPABILITIES,
            _REALIZATIONS,
            self.realizations,
        )

    @property
    def fingerprint(self) -> str:
        """Return the installed QURI Parts realizations identity."""

        identity = (
            f"quri-parts-logical-v3:{version('quri-parts-circuit')}:"
            f"{version('quri-parts-qulacs')}:{RESULT_POSTPROCESSING_FORMAT}:"
            + ",".join(sorted(item.value for item in _REALIZATIONS))
            + repr(self.realizations.overrides)
        )
        return "quri-parts-" + hashlib.sha256(identity.encode()).hexdigest()[:20]

    @property
    def supports_circuit_result_postprocessing(self) -> bool:
        """Return whether execution evaluates retained result trees."""

        return True

    def lower(self, program: Program) -> LoweredProgram:
        """Project QCore to retained SDK gates and typed angle bindings."""

        return lower_circuit_target(program, self)

    def lower_circuit(self, logical: CircuitProgram) -> LoweredProgram:
        """Retain SDK gates and deferred angle expressions without JSON."""

        _validate_block(logical.body)
        from ._payload import QuriExecutable

        payload = QuriExecutable.from_circuit(logical)
        return LoweredProgram(
            payload=payload,
            parameter_abi=logical.parameter_abi,
            result_abi=logical.result_abi,
            stages=(*logical.realization_stages, "quri-parts-portable-emission"),
            circuit_qubits=logical.width,
            circuit_gates=logical.operation_count,
            emitted_instructions=logical.operation_count,
            metadata=FrozenMapping.from_mapping(
                {
                    "quri_parts_version": version("quri-parts-circuit"),
                    **(
                        {RESULT_POSTPROCESSING_METADATA: plan}
                        if (plan := circuit_result_postprocessing(logical)) is not None
                        else {}
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class Qulacs:
    """Execute QURI Parts artifacts through Qulacs in this process.

    Args:
        realizations (RealizationOptions): Exact choices used when compiling source.
    """

    realizations: RealizationOptions = RealizationOptions()

    @property
    def name(self) -> str:
        """Return the endpoint identity."""

        return "quri-parts.qulacs"

    @property
    def target(self) -> QuriParts:
        """Return the compiler target accepted by this endpoint."""

        return QuriParts(realizations=self.realizations)

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the live endpoint capability snapshot."""

        return CapabilitySnapshot(FORMAT, _CAPABILITIES, seeded_execution=True)

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Execute one artifact with Qulacs in this process.

        Args:
            artifact (CompiledProgram): Sealed QURI Parts artifact.
            invocation (Invocation): Validated parameters, shots, and seed.

        Returns:
            Job: Completed local simulation job.

        Raises:
            ExecutionError: If the artifact format or simulation is invalid.
        """

        return launch_local_sample(
            artifact,
            invocation,
            endpoint=self.name,
            artifact_format=FORMAT,
            job_prefix="quri-parts-qulacs",
            execute=lambda: _execute_local(artifact, invocation),
        )


@dataclass(frozen=True, slots=True)
class QuriPartsSampler:
    """Execute portable artifacts with one live QURI Parts sampler.

    The sampler remains a process-local endpoint dependency and never enters
    the portable artifact. This keeps one compiled artifact reusable across
    QURI Parts simulators and providers without making live SDK objects part of
    QCore persistence.

    Args:
        sampler: A QURI Parts sampler callable accepting ``(circuit, shots)``
            and returning integer basis-state counts.
        label: Stable, nonempty name used to identify results from this
            endpoint.

    Raises:
        TypeError: If ``sampler`` is not callable.
        ValueError: If ``label`` is empty.
    """

    sampler: Any = field(repr=False, compare=False)
    label: str = "custom"

    def __post_init__(self) -> None:
        """Validate the live sampler boundary without retaining it in data."""

        if not callable(self.sampler):
            raise TypeError("sampler must be callable")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a nonempty string")

    @property
    def name(self) -> str:
        """Return the endpoint identity for result provenance."""

        return f"quri-parts.sampler.{self.label}"

    @property
    def target(self) -> QuriParts:
        """Return the compiler target accepted by this sampler endpoint."""

        return QuriParts()

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the conservative static-circuit capability snapshot."""

        return CapabilitySnapshot(FORMAT, _CAPABILITIES)

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Materialize, sample, and normalize one portable artifact.

        Args:
            artifact: QURI Parts logical-circuit artifact to execute.
            invocation: Validated runtime bindings and shot count.

        Returns:
            A completed job containing normalized logical counts.

        Raises:
            ExecutionError: If the artifact is incompatible, a nonzero seed is
                requested, or the sampler returns malformed counts.
        """

        if artifact.format != FORMAT:
            raise ExecutionError(
                f"Q8716: QURI Parts sampler accepts {FORMAT!r}, got {artifact.format!r}"
            )
        if invocation.seed != 0:
            raise ExecutionError(
                "Q8717: live QURI Parts samplers do not provide a portable "
                "seed contract; use seed=0"
            )
        circuit, measurements = _materialize_circuit(artifact, invocation)
        try:
            native = self.sampler(circuit, invocation.shots)
        except Exception as error:
            raise ExecutionError("Q8718: QURI Parts sampler failed") from error
        logical = _decode_native_counts(
            artifact,
            invocation,
            measurements,
            native,
        )
        result = SampleResult.from_counts(
            backend=self.name,
            counts=logical,
            expected_width=artifact.result_abi.width,
        )
        return CompletedJob(f"quri-parts-{uuid.uuid4().hex}", result)


for _public_symbol in (
    QuriParts,
    Qulacs,
    QuriPartsSampler,
    load_circuit,
    load_logical_circuit,
):
    _public_symbol.__module__ = __package__
del _public_symbol


__all__ = [
    "FORMAT",
    "Qulacs",
    "QuriParts",
    "QuriPartsSampler",
    "load_circuit",
    "load_logical_circuit",
]
