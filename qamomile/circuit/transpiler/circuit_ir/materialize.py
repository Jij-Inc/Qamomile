"""Shared materialization boundary for circuit-family backend artifacts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.transpiler.circuit_ir.capability import (
    DEFAULT_POLICY,
    CircuitCapabilities,
    CompilationPolicy,
)
from qamomile.circuit.transpiler.circuit_ir.legalize import (
    legalize_program,
    verify_target_legal,
)
from qamomile.circuit.transpiler.circuit_ir.lowering import lower_circuit_plan
from qamomile.circuit.transpiler.circuit_ir.model import CircuitProgram
from qamomile.circuit.transpiler.circuit_ir.verify import verify_circuit
from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.emit_support import ClbitMap, QubitMap
from qamomile.circuit.transpiler.segments import ProgramPlan

ArtifactT = TypeVar("ArtifactT")


@dataclasses.dataclass(frozen=True)
class MaterializedCircuit(Generic[ArtifactT]):
    """Package a circuit artifact and backend-specific binding metadata.

    Args:
        artifact (Any): Backend-native circuit object.
        parameters (Mapping[str, Any]): Backend parameters keyed by public
            parameter name.
        measurement_qubit_map (Mapping[int, int] | None): Static-measurement
            mapping from classical output slot to physical qubit slot. ``None``
            preserves the lowering-provided mapping; an empty mapping is an
            explicit override.
        parameter_order (tuple[str, ...] | None): Artifact ABI order for
            positional parameters. ``None`` denotes name-based binding.
    """

    artifact: ArtifactT
    parameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    measurement_qubit_map: Mapping[int, int] | None = None
    parameter_order: tuple[str, ...] | None = None


class CircuitMaterializer(Protocol[ArtifactT]):
    """Convert one target-legal circuit program to a backend artifact.

    A materializer owns two things: a declaration of what it accepts
    (:attr:`capabilities`) and a mechanical conversion of programs that
    verification has already proven against that declaration. Realization
    decisions (native intrinsic vs fallback body, decomposition choices)
    belong to legalization, never here.
    """

    @property
    def capabilities(self) -> CircuitCapabilities:
        """Declare what this target accepts in circuit IR.

        Returns:
            CircuitCapabilities: Immutable capability declaration consulted
                by legalization and enforced by target verification.
        """
        ...

    def materialize(self, program: CircuitProgram) -> MaterializedCircuit[ArtifactT]:
        """Materialize one circuit program.

        Args:
            program (CircuitProgram): Target-legal circuit-family program.

        Returns:
            MaterializedCircuit: Artifact plus backend binding metadata.
        """
        ...


class CircuitBackendEmitPass(EmitPass[ArtifactT]):
    """Lower, legalize, verify, and materialize a circuit-family plan.

    The pass runs the three phases in order and never interleaves them:
    shared lowering produces backend-neutral circuit IR, target legalization
    rewrites it under the materializer's declared capabilities and the
    compilation policy, target verification proves the result, and only then
    does the materializer convert it mechanically.

    Args:
        materializer (CircuitMaterializer[ArtifactT]): Backend artifact
            materializer owning the target capability declaration.
        bindings (dict[str, Any] | None): Compile-time bindings. Defaults to
            ``None``.
        parameters (list[str] | None): Runtime parameter names. Defaults to
            ``None``.
        policy (CompilationPolicy | None): Realization preferences. Defaults
            to ``None``, meaning :data:`DEFAULT_POLICY`.
    """

    def __init__(
        self,
        materializer: CircuitMaterializer[ArtifactT],
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
        policy: CompilationPolicy | None = None,
    ) -> None:
        """Initialize a circuit-family lowering and materialization pass.

        Args:
            materializer (CircuitMaterializer[ArtifactT]): Backend artifact
                materializer owning the target capability declaration.
            bindings (dict[str, Any] | None): Compile-time bindings. Defaults
                to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.
            policy (CompilationPolicy | None): Realization preferences.
                Defaults to ``None``, meaning :data:`DEFAULT_POLICY`.
        """
        super().__init__(bindings, parameters)
        self.materializer = materializer
        self.parameter_names = list(parameters or ())
        self.policy = policy if policy is not None else DEFAULT_POLICY

    def run(self, input: ProgramPlan) -> ExecutableProgram[ArtifactT]:
        """Lower, legalize, verify, and materialize every quantum segment.

        Args:
            input (ProgramPlan): Circuit-family execution plan.

        Returns:
            ExecutableProgram[ArtifactT]: Backend-native executable structure.

        Raises:
            TargetCapabilityError: If a legalized segment still requires a
                capability the target does not declare.
            ValueError: If a legalized segment fails structural verification.
        """
        lowered = lower_circuit_plan(
            input,
            bindings=self.bindings,
            parameters=self.parameter_names,
        )
        capabilities = self.materializer.capabilities
        legal_segments = []
        for segment in lowered.compiled_quantum:
            program = legalize_program(segment.circuit, capabilities, self.policy)
            verify_circuit(program)
            verify_target_legal(program, capabilities)
            legal_segments.append(dataclasses.replace(segment, circuit=program))
        legalized = dataclasses.replace(
            lowered,
            compiled_quantum=legal_segments,
        )
        return materialize_executable(legalized, self.materializer)

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[ArtifactT, QubitMap, ClbitMap]:
        """Reject the obsolete direct semantic-emission hook.

        Args:
            operations (list[Operation]): Unused semantic operations.
            bindings (dict[str, Any]): Unused emit bindings.

        Returns:
            tuple[ArtifactT, QubitMap, ClbitMap]: This method never returns.

        Raises:
            RuntimeError: Always, because :meth:`run` owns the new path.
        """
        del operations, bindings
        raise RuntimeError("Circuit backends must materialize CircuitProgram")


def materialize_executable(
    executable: ExecutableProgram[CircuitProgram],
    materializer: CircuitMaterializer[ArtifactT],
) -> ExecutableProgram[ArtifactT]:
    """Materialize every quantum segment while preserving orchestration.

    Args:
        executable (ExecutableProgram[CircuitProgram]): Lowered circuit-family
            execution structure.
        materializer (CircuitMaterializer[ArtifactT]): Backend materializer.

    Returns:
        ExecutableProgram[ArtifactT]: Execution structure containing native
            backend circuits and unchanged ABI, classical, expectation-value,
            mapping, and parameter metadata.
    """
    quantum_segments = []
    for segment in executable.compiled_quantum:
        materialized = materializer.materialize(segment.circuit)
        metadata_names = tuple(
            parameter.name for parameter in segment.parameter_metadata.parameters
        )
        materialized_names = tuple(materialized.parameters)
        if set(materialized_names) != set(metadata_names):
            raise ValueError(
                "Materializer parameter names do not match the compiled ABI: "
                f"materialized={materialized_names}, metadata={metadata_names}"
            )
        if (
            materialized.parameter_order is not None
            and materialized.parameter_order != metadata_names
        ):
            raise ValueError(
                "Materializer positional parameter order does not match the "
                f"compiled ABI: materialized={materialized.parameter_order}, "
                f"metadata={metadata_names}"
            )
        parameter_metadata = dataclasses.replace(
            segment.parameter_metadata,
            parameters=[
                dataclasses.replace(
                    parameter,
                    backend_param=materialized.parameters.get(
                        parameter.name,
                        parameter.backend_param,
                    ),
                )
                for parameter in segment.parameter_metadata.parameters
            ],
        )
        quantum_segments.append(
            CompiledQuantumSegment(
                segment=segment.segment,
                circuit=materialized.artifact,
                qubit_map=segment.qubit_map,
                clbit_map=segment.clbit_map,
                measurement_qubit_map=(
                    segment.measurement_qubit_map
                    if materialized.measurement_qubit_map is None
                    else dict(materialized.measurement_qubit_map)
                ),
                parameter_metadata=parameter_metadata,
            )
        )
    return ExecutableProgram(
        plan=executable.plan,
        compiled_quantum=quantum_segments,
        compiled_classical=executable.compiled_classical,
        compiled_expval=executable.compiled_expval,
        output_refs=executable.output_refs,
        num_output_bits=executable.num_output_bits,
    )
