"""Execute result-instrumented HUGR packages with local Selene."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType

from qamomile.compilation import (
    CapabilitySnapshot,
    CompiledProgram,
    Invocation,
    Job,
    SampleResult,
)
from qamomile.errors import ExecutionError
from qamomile.local_execution import launch_local_sample
from qamomile.lowering import RealizationOptions

from ._profile import CAPABILITIES, FORMAT
from ._target import Hugr


@dataclass(frozen=True, slots=True)
class Selene:
    """Execute result-instrumented HUGR packages with Selene Quest.

    Args:
        simulator (str): Selene simulator profile. Only ``"quest"`` is
            currently part of Qamomile's portable endpoint contract.
        timeout (float): Nonnegative overall execution timeout in seconds.
        n_processes (int): Positive number of Selene shot-worker processes.
        realizations (RealizationOptions): Exact operation implementations
            selected during HUGR lowering.

    Raises:
        TypeError: If an option has an incompatible type.
        ValueError: If an option is outside the supported range.
    """

    simulator: str = "quest"
    timeout: float = 30.0
    n_processes: int = 1
    realizations: RealizationOptions = RealizationOptions()

    def __post_init__(self) -> None:
        """Validate the immutable local execution policy.

        Raises:
            TypeError: If an option has an incompatible type.
            ValueError: If an option is outside the supported range.
        """

        if not isinstance(self.simulator, str):
            raise TypeError("simulator must be a string")
        if self.simulator != "quest":
            raise ValueError("simulator must be 'quest'")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout < 0.0
        ):
            raise ValueError("timeout must be a nonnegative finite number")
        if isinstance(self.n_processes, bool) or not isinstance(self.n_processes, int):
            raise TypeError("n_processes must be an integer")
        if self.n_processes <= 0:
            raise ValueError("n_processes must be positive")
        if not isinstance(self.realizations, RealizationOptions):
            raise TypeError("realizations must be RealizationOptions")

    @property
    def name(self) -> str:
        """Return the target-scoped endpoint identity.

        Returns:
            str: Stable local Selene endpoint name.
        """

        return f"hugr.selene.{self.simulator}"

    @property
    def target(self) -> Hugr:
        """Return the result-instrumented HUGR target.

        Returns:
            Hugr: HUGR target using this endpoint's realization choices.
        """

        return Hugr(record_results=True, realizations=self.realizations)

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the direct HUGR execution capability snapshot.

        Returns:
            CapabilitySnapshot: Supported HUGR format and requirements.
        """

        return CapabilitySnapshot(
            FORMAT,
            CAPABILITIES,
            realizations=self.realizations,
        )

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Build and execute one closed HUGR artifact through Selene.

        Args:
            artifact (CompiledProgram): Sealed result-instrumented HUGR package.
            invocation (Invocation): Validated shots, seed, and empty parameter
                row after launch materialization.

        Returns:
            Job: Completed local sampling job.

        Raises:
            ExecutionError: If the artifact lacks result records, retains
                parameters, or produces an invalid Selene result stream.
            ImportError: If the optional Selene runtime is unavailable.
        """

        metadata = dict(artifact.metadata.entries)
        if metadata.get("record_results") is not True:
            raise ExecutionError(
                "Q8690: Selene requires a HUGR target with result recording"
            )
        if invocation.parameters.entries:
            raise ExecutionError("Q8691: Selene received unresolved parameters")
        return launch_local_sample(
            artifact,
            invocation,
            endpoint=self.name,
            artifact_format=FORMAT,
            job_prefix="selene",
            execute=lambda: _execute_local(
                artifact,
                invocation,
                timeout=float(self.timeout),
                n_processes=self.n_processes,
            ),
        )


def _execute_local(
    artifact: CompiledProgram,
    invocation: Invocation,
    *,
    timeout: float,
    n_processes: int,
) -> SampleResult:
    """Reuse a built Selene runner with a fresh shot stream for each launch.

    Args:
        artifact (CompiledProgram): Sealed HUGR package and source ResultABI.
        invocation (Invocation): Validated shot count and deterministic seed.
        timeout (float): Overall Selene execution timeout in seconds.
        n_processes (int): Maximum Selene shot-worker process count.

    Returns:
        SampleResult: Source-ordered counts decoded from QSystem result tags.

    Raises:
        ImportError: If ``selene-sim`` is unavailable.
        ExecutionError: If Selene returns malformed or incompatible results.
    """

    selene = _load_selene()

    from ._payload import HugrExecutable

    executable = artifact.execution_payload(HugrExecutable)
    with executable.lock:
        runner = executable.runner(selene)
        native_shots = runner.run_shots(
            selene.Quest(),
            n_qubits=artifact.report.literal_int("qamomile.circuit.virtual_qubits"),
            n_shots=invocation.shots,
            random_seed=invocation.seed,
            timeout=timeout,
            n_processes=n_processes,
        )
        counts = _decode_result_counts(
            native_shots,
            shots=invocation.shots,
            width=artifact.result_abi.width,
        )
    return SampleResult.from_counts(
        backend="hugr.selene.quest",
        counts=counts,
        expected_width=artifact.result_abi.width,
    )


def _load_selene() -> ModuleType:
    """Load the optional Selene runtime at the execution boundary.

    Returns:
        ModuleType: Imported ``selene_sim`` runtime module.

    Raises:
        ImportError: If the HUGR execution extra is unavailable.
    """

    try:
        import selene_sim
    except ImportError as error:
        raise ImportError(
            'Selene execution requires pip install "qamomile[hugr]"'
        ) from error
    return selene_sim


def _decode_result_counts(
    native_shots: Iterable[Iterable[tuple[str, object]]],
    *,
    shots: int,
    width: int,
) -> dict[tuple[int, ...], int]:
    """Decode ordered ``qamomile`` tags from Selene's result stream.

    Args:
        native_shots (Iterable[Iterable[tuple[str, object]]]): Per-shot tagged
            result iterators returned by Selene.
        shots (int): Exact number of shots required by the invocation.
        width (int): Logical result width from the sealed ResultABI.

    Returns:
        dict[tuple[int, ...], int]: Source-ordered bit counts.

    Raises:
        ExecutionError: If the stream has the wrong shot count, tag width, or
            value domain.
    """

    counts: Counter[tuple[int, ...]] = Counter()
    observed_shots = 0
    try:
        for native_shot in native_shots:
            observed_shots += 1
            values = [value for tag, value in native_shot if tag == "qamomile"]
            if len(values) != width:
                raise ExecutionError(
                    "Q8692: Selene result width disagrees with ResultABI"
                )
            bits: list[int] = []
            for value in values:
                if not isinstance(value, (bool, int)) or int(value) not in {0, 1}:
                    raise ExecutionError(
                        "Q8692: Selene result contains a non-bit value"
                    )
                bits.append(int(value))
            counts[tuple(bits)] += 1
    except ExecutionError:
        raise
    except Exception as error:
        raise ExecutionError(f"Q8693: invalid Selene result stream: {error}") from error
    if observed_shots != shots:
        raise ExecutionError(
            f"Q8692: Selene returned {observed_shots} shots; expected {shots}"
        )
    return dict(counts)


__all__ = ["Selene"]
