"""Implement CUDA-Q source emission and local execution."""

from __future__ import annotations

import hashlib
import linecache
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from qamomile.circuit import (
    RESULT_POSTPROCESSING_FORMAT,
    RESULT_POSTPROCESSING_METADATA,
    CircuitProgram,
    circuit_result_postprocessing,
    lower_circuit_target,
    postprocess_circuit_counts,
    scalar_parameter_bindings,
)
from qamomile.compilation import (
    CapabilitySnapshot,
    CompiledProgram,
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

from ._emission import emit_source as _emit_source

FORMAT = "cudaq.python-source.v1"
_CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.PARAMETERIZED_EXECUTION,
        Requirement.QUANTUM,
        Requirement.MEASUREMENT,
        Requirement.MID_CIRCUIT_MEASUREMENT,
        Requirement.DYNAMIC_BRANCH,
        Requirement.RESET,
        Requirement.DISCARD,
        Requirement.ROTATION,
    }
)
_REALIZATIONS = frozenset(
    {
        RealizationCapability.RZZ,
        RealizationCapability.MCX,
        RealizationCapability.CONTROLLED_SWAP,
        RealizationCapability.PAULI_EVOLUTION,
    }
)
_CUDAQ_EXECUTION_LOCK = threading.RLock()
_CUDAQ_EXECUTION_STATE = threading.local()


@dataclass(frozen=True, slots=True)
class Cudaq:
    """Compile circuit-shaped QCore into self-contained CUDA-Q Python source.

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
        """Return the source-artifact capability contract."""

        return CapabilitySnapshot(
            FORMAT, _CAPABILITIES, _REALIZATIONS, self.realizations
        )

    @property
    def fingerprint(self) -> str:
        """Return the versioned CUDA-Q emission identity."""

        import cudaq

        return (
            "cudaq-python-source-"
            + hashlib.sha256(
                (
                    f"{FORMAT}:{cudaq.__version__}:"
                    f"{RESULT_POSTPROCESSING_FORMAT}:"
                    + ",".join(sorted(item.value for item in _REALIZATIONS))
                    + repr(self.realizations.overrides)
                ).encode()
            ).hexdigest()[:20]
        )

    @property
    def supports_circuit_result_postprocessing(self) -> bool:
        """Return whether execution evaluates retained result trees."""

        return True

    def lower(self, program: Program) -> LoweredProgram:
        """Retain a CUDA-Q kernel and its target-neutral ABI facts.

        Args:
            program (Program): Verified specialized QCore program.

        Returns:
            LoweredProgram: Source artifact or native-selected circuit family.
        """

        return lower_circuit_target(program, self)

    def lower_circuit(self, logical: CircuitProgram) -> LoweredProgram:
        """Emit a verified circuit and its helper kernels as one Python module.

        Args:
            logical (CircuitProgram): Verified circuit and execution ABI.

        Returns:
            LoweredProgram: Kernel owner, execution ABI, and circuit report.
        """

        import cudaq

        from ._payload import CudaqExecutable

        source = _emit_source(logical)
        return LoweredProgram(
            payload=CudaqExecutable(source),
            parameter_abi=logical.parameter_abi,
            result_abi=logical.result_abi,
            stages=(
                *logical.realization_stages,
                "cudaq-source-emission",
            ),
            circuit_qubits=logical.width,
            circuit_gates=logical.operation_count,
            emitted_instructions=logical.operation_count,
            metadata=FrozenMapping.from_mapping(
                {
                    "cudaq_version": str(cudaq.__version__),
                    **(
                        {RESULT_POSTPROCESSING_METADATA: plan}
                        if (plan := circuit_result_postprocessing(logical)) is not None
                        else {}
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CudaqLocal:
    """Execute CUDA-Q source artifacts in the calling process.

    Qamomile CUDA-Q launches are serialized while their SDK target is active.
    The prior target is restored afterward; the SDK seed cannot be restored.

    Args:
        runtime_target (str): SDK simulator name, defaulting to qpp-cpu.
        realizations (RealizationOptions): Exact choices used when compiling source.
    """

    runtime_target: str = "qpp-cpu"
    realizations: RealizationOptions = RealizationOptions()

    @property
    def name(self) -> str:
        """Return the endpoint identity."""

        return f"cudaq.{self.runtime_target}"

    @property
    def target(self) -> Cudaq:
        """Return the compiler target accepted by this endpoint."""

        return Cudaq(realizations=self.realizations)

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return this CUDA-Q runtime's immutable capability snapshot."""

        return CapabilitySnapshot(FORMAT, _CAPABILITIES, seeded_execution=True)

    def launch(
        self,
        artifact: CompiledProgram,
        invocation: Invocation,
    ) -> Job:
        """Build and execute the artifact's kernel in the calling process.

        Args:
            artifact (CompiledProgram): Sealed CUDA-Q kernel artifact.
            invocation (Invocation): Validated launch values, shots, and seed.

        Returns:
            Job: Completed local sampling job.

        Raises:
            ExecutionError: If the artifact or SDK launch is invalid.
        """

        return launch_local_sample(
            artifact,
            invocation,
            endpoint=self.name,
            artifact_format=FORMAT,
            job_prefix="cudaq-local",
            execute=lambda: _execute_local(
                artifact, invocation, runtime_target=self.runtime_target
            ),
        )


def _load_source(source: str) -> Any:
    """Define one generated module in a fresh namespace and return its entry.

    CUDA-Q inspects Python source while decorating functions. Register it only
    during module execution; each resulting decorator retains its own source
    and defining namespace, including any helper kernels.

    Args:
        source (str): Complete generated CUDA-Q Python module.

    Returns:
        Any: CUDA-Q entry kernel with module-local helper bindings.

    Raises:
        ExecutionError: If the module cannot define a CUDA-Q entry kernel.
    """
    import cudaq

    filename = "<qamomile-cudaq-" + hashlib.sha256(source.encode()).hexdigest() + ">"
    namespace: dict[str, Any] = {"__name__": "_qamomile_cudaq"}
    # Source inspection and runtime target selection share the same lock.
    with _CUDAQ_EXECUTION_LOCK:
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )
        try:
            exec(compile(source, filename, "exec"), namespace)  # noqa: S102
            kernel = namespace.get("_qamomile_entry")
            if not isinstance(kernel, cudaq.PyKernelDecorator):
                raise ValueError("source must define the CUDA-Q entry kernel")
            return kernel
        except Exception as error:
            raise ExecutionError("Q8504: invalid CUDA-Q source artifact") from error
        finally:
            linecache.cache.pop(filename, None)


def load_kernel(artifact: CompiledProgram) -> Any:
    """Return the retained CUDA-Q kernel, restoring saved source only once.

    Args:
        artifact (CompiledProgram): Inspected CUDA-Q Python source artifact.

    Returns:
        Any: SDK entry kernel for the currently selected target.

    Raises:
        ExecutionError: If the format, UTF-8 source, or entry kernel is invalid.
    """

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8503: expected {FORMAT!r}, got {artifact.format!r}")
    from ._payload import CudaqExecutable

    # Keep SDK -> artifact lock ordering identical for public and launch paths.
    with _CUDAQ_EXECUTION_LOCK:
        return artifact.execution_payload(CudaqExecutable).kernel()


@contextmanager
def _selected_runtime_target(runtime_target: str, seed: int) -> Iterator[None]:
    """Serialize Qamomile CUDA-Q launches and restore the previous SDK target.

    Args:
        runtime_target (str): CUDA-Q target selected for this launch.
        seed (int): Invocation seed passed to the SDK.

    Yields:
        None: Control while the requested target is active and locked.

    Raises:
        ExecutionError: If the launch reenters, target setup fails, or target
            restoration fails.
    """

    import cudaq

    with _CUDAQ_EXECUTION_LOCK:
        if getattr(_CUDAQ_EXECUTION_STATE, "active", False):
            raise ExecutionError("Q8516: reentrant CUDA-Q launch is unsupported")
        _CUDAQ_EXECUTION_STATE.active = True
        previous_target = None
        try:
            try:
                if not isinstance(runtime_target, str):
                    raise ValueError("runtime target must be a string")
                previous_target = cudaq.get_target()
                cudaq.set_target(runtime_target)
                if cudaq.get_target().name != runtime_target:
                    raise RuntimeError("CUDA-Q selected a different target")
                cudaq.set_random_seed(seed)
            except ExecutionError:
                raise
            except Exception as error:
                raise ExecutionError("Q8517: CUDA-Q target setup failed") from error
            yield
        finally:
            try:
                if previous_target is not None:
                    cudaq.set_target(previous_target)
            except Exception as error:
                raise ExecutionError(
                    "Q8519: CUDA-Q target restoration failed"
                ) from error
            finally:
                _CUDAQ_EXECUTION_STATE.active = False


def _execute_local(
    artifact: CompiledProgram,
    invocation: Invocation,
    *,
    runtime_target: str,
) -> SampleResult:
    """Execute and normalize a source-artifact kernel in the calling process.

    Args:
        artifact (CompiledProgram): Sealed CUDA-Q kernel artifact.
        invocation (Invocation): Validated launch values, shots, and seed.
        runtime_target (str): SDK target selected for this launch.

    Returns:
        SampleResult: Counts normalized to the artifact result ABI.

    Raises:
        ExecutionError: If SDK setup, execution, or result decoding fails.
    """

    import cudaq

    with _selected_runtime_target(runtime_target, invocation.seed):
        kernel = load_kernel(artifact)
        arguments = [
            value
            for _, value in scalar_parameter_bindings(
                artifact.parameter_abi,
                invocation.values(),
            )
        ]
        try:
            native = cudaq.run(kernel, *arguments, shots_count=invocation.shots)
        except ExecutionError:
            raise
        except Exception as error:
            raise ExecutionError("Q8518: CUDA-Q execution failed") from error
        native_counts: dict[tuple[int, ...], int] = {}
        for value in native:
            if isinstance(value, bool):
                bits = (int(value),)
            elif isinstance(value, tuple) and all(
                isinstance(item, bool) for item in value
            ):
                bits = tuple(int(item) for item in value)
            else:
                raise ExecutionError(
                    f"Q8512: CUDA-Q returned an unexpected value {value!r}"
                )
            native_counts[bits] = native_counts.get(bits, 0) + 1
        plan = dict(artifact.metadata.entries).get(RESULT_POSTPROCESSING_METADATA)
        if plan is None:
            logical_counts = native_counts
        else:
            if not isinstance(plan, bytes):
                raise ExecutionError("Q8515: invalid CUDA-Q result metadata")
            logical_counts = postprocess_circuit_counts(
                plan,
                native_counts,
                invocation.values(),
                artifact.result_abi,
            )
        return SampleResult.from_counts(
            backend=f"cudaq.{runtime_target}",
            counts=logical_counts,
            expected_width=artifact.result_abi.width,
        )


for _public_symbol in (Cudaq, CudaqLocal, load_kernel):
    _public_symbol.__module__ = __package__
del _public_symbol


__all__ = ["Cudaq", "CudaqLocal", "FORMAT", "load_kernel"]
