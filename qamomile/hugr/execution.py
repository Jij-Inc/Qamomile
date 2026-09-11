"""Select local Selene or Nexus Helios for direct HUGR execution."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import tempfile
import threading
from enum import StrEnum
from pathlib import Path
from typing import Any

from qamomile.circuit.transpiler import (
    CompletedExecutionHandle,
    EstimationAccuracy,
    Exact,
    ExecutionCapabilities,
    ExecutionHandle,
    ExecutionReference,
    ShotBased,
    TargetPrecision,
)
from qamomile.hugr._nexus import (
    NexusExecutionOptions,
    NexusTransport as _NexusTransport,
)


class HugrExecutionTarget(StrEnum):
    """Identify a supported HUGR execution destination."""

    SELENE = "selene"
    HELIOS = "helios"


@dataclasses.dataclass(frozen=True)
class SeleneExecutionOptions:
    """Configure local Selene execution.

    Args:
        seed (int | None): Reproducible simulator seed, or provider default.
        n_qubits (int | None): Simulator capacity override. By default use
            an allocation bound expanded through the submitted call graph.
        timeout_seconds (float | None): Maximum simulator execution duration.
        build_dir (Path | None): Root for persistent isolated compilation
            directories. If omitted, the executor owns a temporary directory
            that is removed when the executor is released.

    Raises:
        ValueError: If a numeric option is invalid.
        TypeError: If the seed or capacity is not an integer.
    """

    seed: int | None = None
    n_qubits: int | None = None
    timeout_seconds: float | None = None
    build_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate local options before compiling a program.

        Raises:
            TypeError: If seed or capacity is not an integer.
            ValueError: If capacity or timeout is invalid.
        """
        for name in ("seed", "n_qubits"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
        if self.n_qubits is not None and self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")


def _validate_shots(shots: int) -> None:
    """Reject invalid shot counts before any provider side effects.

    Args:
        shots (int): Requested positive shot count.

    Raises:
        ValueError: If shots is not a positive integer.
    """
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")


def _resolve_estimation_shots(
    estimation: EstimationAccuracy | None, shots: int | None
) -> int:
    """Resolve the common estimation policy and the legacy shot-count argument.

    Args:
        estimation (EstimationAccuracy | None): Per-execution accuracy policy.
        shots (int | None): Legacy shots per Pauli term. Mutually exclusive
            with estimation. Both omitted selects 1024 shots per term.

    Returns:
        int: Positive measurement count per nonidentity Pauli term.

    Raises:
        ValueError: If both arguments are supplied or the shot count is invalid.
        TypeError: If estimation is not a shared accuracy policy.
        NotImplementedError: If exact or target-precision estimation is requested.
    """
    if estimation is not None and shots is not None:
        raise ValueError("Specify either estimation or shots, not both")
    if estimation is not None:
        if isinstance(estimation, ShotBased):
            shots = estimation.shots
        elif isinstance(estimation, (Exact, TargetPrecision)):
            raise NotImplementedError(
                "HUGR execution supports ShotBased estimation; "
                f"{type(estimation).__name__} is unavailable"
            )
        else:
            raise TypeError("estimation must be an EstimationAccuracy policy")
    resolved = 1024 if shots is None else shots
    _validate_shots(resolved)
    return resolved


def _qubit_capacity(package: Any) -> int:
    """Bound live capacity by allocation sites expanded through direct calls.

    Summing both branches overestimates capacity safely. A loop reuses its
    fixed linear carrier, so one traversal counts its simultaneous temporary
    allocations. Recursive and indirect calls require an explicit capacity.

    Args:
        package (Any): Submitted HUGR package with a public main function.

    Returns:
        int: Conservative positive simulator qubit capacity.

    Raises:
        ValueError: If the call graph has recursion or unresolved calls.
    """
    from hugr import ops

    if len(package.modules) != 1:
        raise ValueError("Automatic Selene capacity requires one HUGR module")
    graph = package.modules[0]
    entries = [
        node
        for node, data in graph.nodes()
        if isinstance(data.op, ops.FuncDefn) and data.op.f_name == "main"
    ]
    if len(entries) != 1:
        raise ValueError("Automatic Selene capacity requires one main function")
    active: set[Any] = set()
    cached: dict[Any, int] = {}

    def allocations(node: Any) -> int:
        """Count allocations within one graph subtree and its direct callees.

        Args:
            node (Any): HUGR node being traversed.

        Returns:
            int: Upper bound on allocations made by the subtree.

        Raises:
            ValueError: If a call cannot be bounded statically.
        """
        operation = graph[node].op
        if isinstance(operation, ops.FuncDefn):
            if node in active:
                raise ValueError(
                    "Recursive HUGR requires SeleneExecutionOptions(n_qubits=...)"
                )
            if node in cached:
                return cached[node]
            active.add(node)
            count = sum(allocations(child) for child in graph.children(node))
            active.remove(node)
            cached[node] = count
            return count
        if isinstance(operation, ops.Call):
            linked = list(
                graph.linked_ports(node.inp(len(operation.instantiation.input)))
            )
            if len(linked) != 1 or not isinstance(
                graph[linked[0].node].op, ops.FuncDefn
            ):
                raise ValueError(
                    "External HUGR calls require SeleneExecutionOptions(n_qubits=...)"
                )
            return allocations(linked[0].node)
        if isinstance(operation, ops.CallIndirect):
            raise ValueError(
                "Indirect HUGR calls require SeleneExecutionOptions(n_qubits=...)"
            )
        name = getattr(operation, "name", None)
        own = int(callable(name) and name() == "tket.quantum.QAlloc")
        return own + sum(allocations(child) for child in graph.children(node))

    return max(1, allocations(entries[0]))


class HugrExecutor:
    """Execute HUGR packages on local Selene or Nexus Helios.

    Args:
        target (str | HugrExecutionTarget): Destination, default ``selene``.
        options (SeleneExecutionOptions | NexusExecutionOptions | None):
            Destination-specific settings. Defaults to the destination defaults.

    Raises:
        ValueError: If the destination is unknown.
        TypeError: If options belong to another destination.

    Example:
        >>> executor = HugrExecutor(options=SeleneExecutionOptions(seed=7))
        >>> remote = HugrExecutor("helios", options=NexusExecutionOptions())
    """

    def __init__(
        self,
        target: str | HugrExecutionTarget = HugrExecutionTarget.SELENE,
        *,
        options: SeleneExecutionOptions | NexusExecutionOptions | None = None,
    ) -> None:
        """Initialize the destination and local compilation cache.

        Args:
            target (str | HugrExecutionTarget): Selene or Helios destination.
            options (SeleneExecutionOptions | NexusExecutionOptions | None):
                Destination-specific configuration.

        Raises:
            ValueError: If the destination is unknown.
            TypeError: If options are incompatible with the destination.
        """
        try:
            self.target = HugrExecutionTarget(target)
        except ValueError as error:
            raise ValueError("HUGR target must be 'selene' or 'helios'") from error
        expected = (
            SeleneExecutionOptions
            if self.target is HugrExecutionTarget.SELENE
            else NexusExecutionOptions
        )
        if options is not None and not isinstance(options, expected):
            raise TypeError(f"{self.target.value} requires {expected.__name__}")
        self.options = options or expected()
        self._nexus = (
            _NexusTransport(self.options)
            if isinstance(self.options, NexusExecutionOptions)
            else None
        )
        self._runners: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._namespace: Path | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Describe this destination's actual execution features.

        Returns:
            ExecutionCapabilities: Shot estimation and remote lifecycle support.
        """
        remote = self.target is HugrExecutionTarget.HELIOS
        return ExecutionCapabilities(
            supports_async_sampling=remote,
            supports_async_estimation=remote,
            supports_estimation=True,
            supports_cancellation=remote,
            supports_restoration=remote,
            estimation_accuracy=frozenset({ShotBased}),
        )

    def submit(
        self, package: Any, shots: int = 1024
    ) -> ExecutionHandle[list[dict[str, Any]]]:
        """Submit a zero-argument HUGR entrypoint with recorded outputs.

        Use ``HugrExecutable`` for ABI-aware runtime arguments and typed jobs.

        Args:
            package (Any): HUGR package with a zero-argument ``main`` function.
            shots (int): Number of repetitions, default ``1024``.

        Returns:
            ExecutionHandle[list[dict[str, Any]]]: Tagged values for each shot.

        Raises:
            ValueError: If shots is invalid or output tags repeat.
            ImportError: If destination dependencies are missing.
            RuntimeError: If the simulator returns an incorrect shot count.
            Exception: If provider compilation or submission fails.
        """
        _validate_shots(shots)
        if self._nexus is not None:
            return self._nexus.submit(package, shots)
        import selene_sim

        options = self.options
        assert isinstance(options, SeleneExecutionOptions)
        fingerprint = hashlib.sha256(package.to_bytes()).hexdigest()
        with self._lock:
            runner = self._runners.get(fingerprint)
            if runner is None:
                if self._namespace is None:
                    if options.build_dir is None:
                        self._temporary_directory = tempfile.TemporaryDirectory(
                            prefix="qamomile-hugr-"
                        )
                        self._namespace = Path(self._temporary_directory.name)
                    else:
                        Path(options.build_dir).mkdir(parents=True, exist_ok=True)
                        self._namespace = Path(
                            tempfile.mkdtemp(
                                prefix="qamomile-hugr-", dir=options.build_dir
                            )
                        )
                runner = selene_sim.build(
                    package.to_bytes(),
                    name=f"hugr_{fingerprint}",
                    build_dir=self._namespace / fingerprint,
                )
                self._runners[fingerprint] = runner
            capacity = options.n_qubits
            if capacity is None:
                capacity = _qubit_capacity(package)
            samples = []
            for shot in runner.run_shots(
                selene_sim.Quest(),
                n_qubits=capacity,
                n_shots=shots,
                random_seed=options.seed,
                timeout=options.timeout_seconds,
            ):
                row = {}
                for tag, value in shot:
                    if tag in row:
                        raise ValueError(f"Duplicate HUGR output tag: {tag}")
                    row[tag] = value
                samples.append(row)
        if len(samples) != shots:
            raise RuntimeError(
                f"Selene returned {len(samples)} shots; expected {shots}"
            )
        return CompletedExecutionHandle(samples)

    def retrieve(
        self, reference: ExecutionReference
    ) -> ExecutionHandle[list[dict[str, Any]]]:
        """Restore a remote tagged execution without resubmitting it.

        Args:
            reference (ExecutionReference): Saved Nexus job reference.

        Returns:
            ExecutionHandle[list[dict[str, Any]]]: Restored remote handle.

        Raises:
            ValueError: If the destination is local or reference is invalid.
            Exception: If the provider cannot retrieve the referenced job.
        """
        if self._nexus is None:
            raise ValueError("Local Selene executions cannot be restored")
        return self._nexus.retrieve(reference)
