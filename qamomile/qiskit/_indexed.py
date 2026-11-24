"""Transport native-materialized logical indices through one Qiskit submission."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from qamomile.compilation import CompiledProgram, Invocation, SampleResult
from qamomile.errors import ExecutionError
from qamomile.ir import FrozenMapping

IndexedChildren = tuple[tuple[CompiledProgram, Invocation], ...]
_MAX_INDEXED_PREPARATION_WORK = 16_000_000
_MAX_INDEXED_METADATA_BYTES = 256 * 1024 * 1024


class IndexedMaterializationGuard:
    """Bound retained indexed SDK preparation before a producer stores children.

    Args:
        backend (Any | None): Optional SDK backend whose circuit limit applies.
            None checks only the adapter's preparation and metadata limits.
    """

    def __init__(self, backend: Any | None = None) -> None:
        """Initialize cumulative limits for one ordered SDK submission.

        Args:
            backend (Any | None): SDK backend, or None without a circuit limit.
        """
        self._backend = backend
        self._work = 0
        self._metadata_bytes = 0
        self._active = 0
        self._artifacts: set[int] = set()

    def add(self, child: CompiledProgram, *, copies: int = 1) -> None:
        """Check a child's cumulative cost before retaining or copying it.

        Args:
            child (CompiledProgram): Retained SDK payload and portable ABI facts.
            copies (int): Number of SDK copies prepared from this artifact. Its
                shared ABI and metadata content is charged only once.

        Raises:
            ExecutionError: If preparation units, metadata content bytes, or the
                declared backend circuit count exceed the single-job limits.
        """
        from ._payload import QiskitExecutable

        payload = child.execution_payload(QiskitExecutable)
        work = self._work + copies * payload.indexed_preparation_work
        metadata_bytes = self._metadata_bytes
        if id(child) not in self._artifacts:
            metadata_bytes += (
                len(child.parameter_abi._to_bytes())
                + len(child.result_abi._to_bytes())
                + _metadata_content_bytes(child.metadata)
            )
        active = self._active + copies * int(payload.indexed_has_qubits)
        if work > _MAX_INDEXED_PREPARATION_WORK:
            raise ExecutionError(
                "Q8234: indexed SDK preparation exceeds its "
                f"{_MAX_INDEXED_PREPARATION_WORK}-unit work budget"
            )
        if metadata_bytes > _MAX_INDEXED_METADATA_BYTES:
            raise ExecutionError(
                "Q8234: indexed ABI and metadata content exceeds its "
                f"{_MAX_INDEXED_METADATA_BYTES}-byte preparation budget"
            )
        if self._backend is not None:
            require_batch_capacity(self._backend, active)
        self._work = work
        self._metadata_bytes = metadata_bytes
        self._active = active
        self._artifacts.add(id(child))


def _metadata_content_bytes(value: object) -> int:
    """Count portable metadata content without serializing any SDK circuit.

    Args:
        value (object): A compiler-validated portable metadata value.

    Returns:
        int: Bytes of strings, byte payloads, and scalar numeric content,
        excluding Python container overhead and serialization framing.
    """
    if isinstance(value, FrozenMapping):
        return sum(
            len(key.encode("utf-8")) + _metadata_content_bytes(item)
            for key, item in value.entries
        )
    if isinstance(value, tuple):
        return sum(_metadata_content_bytes(item) for item in value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(1, (value.bit_length() + 8) // 8)
    if isinstance(value, float):
        return 8
    if isinstance(value, complex):
        return 16
    return 0


def require_batch_capacity(backend: Any, count: int) -> None:
    """Reject batches larger than a backend's declared experiment limit.

    Args:
        backend (Any): SDK backend exposing an optional maximum circuit count.
        count (int): Number of circuits in this single submission.

    Raises:
        ExecutionError: If the declared limit is smaller than the submission.
    """
    limit = getattr(backend, "max_circuits", None)
    if type(limit) is int and limit > 0 and count > limit:
        raise ExecutionError(
            f"Q8234: indexed batch has {count} circuits; backend limit is {limit}"
        )


def indexed_circuits(
    children: IndexedChildren, invocation: Invocation, *, backend: Any | None = None
) -> tuple[Any, ...]:
    """Bind already checked child artifacts in logical-index order.

    Args:
        children (IndexedChildren): Native-materialized one-shot child artifacts.
        invocation (Invocation): Original logical invocation.
        backend (Any | None): Optional backend whose capacity is checked before
            allocating circuit copies.

    Returns:
        tuple[Any, ...]: Bound SDK circuits in native-assigned logical order.

    Raises:
        ExecutionError: If cardinality, SDK preparation, retained metadata, or
            backend capacity exceeds the single-submission contract.
    """
    from ._integration import bind_circuit, load_circuit

    require_indexed_cardinality(children, invocation)
    guard = IndexedMaterializationGuard(backend)
    for child, _ in children:
        guard.add(child)
    return tuple(
        bind_circuit(load_circuit(child), child, child_invocation)
        for child, child_invocation in children
    )


def require_indexed_cardinality(
    children: IndexedChildren, invocation: Invocation
) -> None:
    """Check native child coverage before any SDK preparation.

    Args:
        children (IndexedChildren): Ordered native-materialized children.
        invocation (Invocation): Original logical shot count.

    Raises:
        ExecutionError: If the batch lacks exactly one one-shot child per index.
    """
    if len(children) != invocation.shots or any(
        child_invocation.shots != 1 for _, child_invocation in children
    ):
        raise ExecutionError("Q8232: indexed circuit batch requires one child per shot")


def require_result_coverage(result: Any, experiments: int, shots: int) -> None:
    """Validate SDK success and exact experiment/shot coverage before decoding.

    Args:
        result (Any): Native backend result.
        experiments (int): Expected number of ordered result experiments.
        shots (int): Required shot count for every experiment.

    Raises:
        ExecutionError: If result success, experiment count, or shots disagree.
    """
    if getattr(result, "success", None) is not True:
        raise ExecutionError("Q8233: indexed backend execution did not succeed")
    rows = getattr(result, "results", None)
    if not isinstance(rows, Sequence) or len(rows) != experiments:
        raise ExecutionError("Q8233: indexed result count disagrees with submission")
    for row in rows:
        if getattr(row, "success", None) is not True:
            raise ExecutionError("Q8233: indexed backend experiment did not succeed")
        actual = getattr(row, "shots", None)
        if type(actual) is not int or actual != shots:
            policy = "one shot" if shots == 1 else f"exactly {shots} shots"
            raise ExecutionError(f"Q8233: indexed circuit results require {policy}")


def indexed_result(
    artifact: CompiledProgram,
    children: IndexedChildren,
    circuits: Sequence[Any],
    native_counts: Sequence[Mapping[Any, int]],
    *,
    backend: str,
) -> SampleResult:
    """Evaluate each child's native result plan before combining sample counts.

    Args:
        artifact (CompiledProgram): Original invocation's sealed result contract.
        children (IndexedChildren): Child artifacts retaining their native context.
        circuits (Sequence[Any]): Corresponding bound SDK circuits.
        native_counts (Sequence[Mapping]): One-shot native counts per child.
        backend (str): Endpoint identity for the resulting samples.

    Returns:
        SampleResult: Logical sample counts with the original result ABI.

    Raises:
        ExecutionError: If indexed results are missing, contain malformed count
            maps, or do not represent exactly one shot per child.
    """
    from ._integration import decode_counts

    if len(native_counts) != len(children) or len(circuits) != len(children):
        raise ExecutionError("Q8233: indexed result count disagrees with submission")
    counts: Counter[tuple[int, ...]] = Counter()
    for (child, invocation), circuit, raw in zip(children, circuits, native_counts):
        if not isinstance(raw, Mapping) or any(
            type(count) is not int or count < 0 for count in raw.values()
        ):
            raise ExecutionError(
                "Q8233: indexed backend returned malformed sampling counts"
            )
        if sum(raw.values()) != 1:
            raise ExecutionError("Q8233: indexed circuit results require one shot")
        logical = decode_counts(child, raw, invocation.values(), len(circuit.clbits))
        if sum(logical.values()) != 1:
            raise ExecutionError("Q8233: indexed result evaluation changed shot count")
        counts.update(logical)
    return replace(
        SampleResult.from_counts(
            backend=backend,
            counts=counts,
            expected_width=artifact.result_abi.width,
        ),
        result_abi=artifact.result_abi,
    )


def backend_indexed_counts(
    result: Any, circuits: Sequence[Any]
) -> tuple[Mapping[Any, int], ...]:
    """Read per-experiment counts without merging logical coordinates.

    Args:
        result (Any): Native backend result for one ordered circuit batch.
        circuits (Sequence[Any]): Submitted circuits in experiment order.

    Returns:
        tuple[Mapping, ...]: One count map per original logical coordinate.

    Raises:
        ExecutionError: If the backend or count retrieval fails, or an experiment
            is missing, malformed, or not one-shot.
    """
    active = [circuit for circuit in circuits if circuit.num_qubits]
    require_result_coverage(result, len(active), 1)
    try:
        experiment = iter(range(len(active)))
        counts = []
        for circuit in circuits:
            if not circuit.num_qubits:
                counts.append({"": 1})
                continue
            index = next(experiment)
            counts.append(result.get_counts(index) if circuit.clbits else {"": 1})
        return tuple(counts)
    except Exception as error:
        raise ExecutionError(
            "Q8233: indexed backend returned no sampling counts"
        ) from error
