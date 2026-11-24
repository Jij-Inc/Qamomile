"""Transport fixed circuits while retaining compiler-owned shot coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from qamomile.circuit import scalar_parameter_bindings
from qamomile.compilation import CompiledProgram, Invocation, SampleResult
from qamomile.errors import ExecutionError

from ._payload import copy_circuit


@dataclass(frozen=True, slots=True)
class MeasurementFreeShots:
    """Carry completed zero-measurement coordinates without allocating rows.

    Args:
        shots (int): Number of logical coordinates certified by the SDK result,
            or the invocation cardinality when no SDK work is needed. The native
            owner validates the count and requires a measurement-free plan.
    """

    shots: int


def template_inputs(
    artifact: CompiledProgram, invocation: Invocation, circuit: Any | None = None
) -> tuple[Any, tuple[Any, ...], Sequence[Sequence[float]]]:
    """Bind public parameters and obtain compiler-ordered hidden parameter rows.

    Args:
        artifact (CompiledProgram): Artifact retaining one native shot template.
        invocation (Invocation): Validated public bindings and shot policy.
        circuit (Any | None): Prepared SDK template, or None for the retained copy.

    Returns:
        tuple: Public-bound circuit, ordered hidden SDK parameters, and a checked
            native row sequence. Zero-slot rows retain cardinality only.

    Raises:
        ExecutionError: If the owner, row layout, or remaining SDK bindings disagree.
    """
    from ._integration import _free_circuit_parameters
    from ._payload import QiskitExecutable

    payload = artifact.execution_payload(QiskitExecutable)
    owner = payload.shot_template
    if owner is None:
        raise ExecutionError("Q8235: artifact has no native shot template")
    try:
        owner.check_invocation(invocation.shots)
    except ValueError as error:
        raise ExecutionError(str(error)) from error
    if circuit is None:
        circuit = payload.circuit()
    hidden = tuple(
        parameter for _, parameter in sorted(payload.hidden_parameters.items())
    )
    flattened = scalar_parameter_bindings(artifact.parameter_abi, invocation.values())
    by_name = {
        parameter.name: parameter for parameter in _free_circuit_parameters(circuit)
    }
    public = {
        by_name[f"p{index}"]: value
        for index, (_, value) in enumerate(flattened)
        if f"p{index}" in by_name
    }
    if public:
        circuit = copy_circuit(circuit, public)
    if _free_circuit_parameters(circuit) - set(hidden):
        raise ExecutionError("Q8203: missing Qiskit public parameters")
    try:
        rows = owner.bind_rows(invocation.values(), invocation.shots)
    except ValueError as error:
        raise ExecutionError(str(error)) from error
    if len(rows) != invocation.shots or rows.slot_count != len(hidden):
        raise ExecutionError(
            "Q8235: native hidden parameter rows disagree with template"
        )
    return circuit, hidden, cast(Sequence[Sequence[float]], rows)


def template_circuits(
    circuit: Any, hidden: Sequence[Any], rows: Sequence[Sequence[float]]
) -> tuple[Any, ...]:
    """Copy one prepared SDK circuit and bind each native-assigned coordinate.

    Args:
        circuit (Any): Public-bound prepared template.
        hidden (Sequence[Any]): Compiler-ordered hidden SDK parameters.
        rows (Sequence[Sequence[float]]): Native hidden values per logical index.

    Returns:
        tuple[Any, ...]: Fully bound copies in logical-index order.
    """
    free = set(circuit.parameters)
    return tuple(
        copy_circuit(
            circuit,
            {
                parameter: value
                for parameter, value in zip(hidden, row)
                if parameter in free
            },
        )
        for row in rows
    )


def aer_parameter_bindings(
    circuit: Any, hidden: Sequence[Any], rows: Sequence[Sequence[float]]
) -> dict[Any, list[float]]:
    """Transpose retained native slots into one Aer parameter table.

    Args:
        circuit (Any): Public-bound circuit after Aer preparation.
        hidden (Sequence[Any]): Original compiler-ordered hidden SDK parameters.
        rows (Sequence[Sequence[float]]): Checked values in logical-shot order.

    Returns:
        dict[Any, list[float]]: Retained parameter columns, or an empty mapping
        when preparation removed every hidden parameter.
    """
    free = set(circuit.parameters)
    retained = tuple(
        (index, parameter)
        for index, parameter in enumerate(hidden)
        if parameter in free
    )
    bindings: dict[Any, list[float]] = {parameter: [] for _, parameter in retained}
    if retained:
        for row in rows:
            for index, parameter in retained:
                bindings[parameter].append(row[index])
    return bindings


def aer_template_counts(
    result: Any, circuit: Any, shots: int, *, parameterized: bool
) -> tuple[Mapping[Any, int], ...] | MeasurementFreeShots:
    """Read ordered Aer rows without reconstructing samples from a histogram.

    Args:
        result (Any): Native result from exactly one SDK submission.
        circuit (Any): Prepared circuit defining classical measurement width.
        shots (int): Number of logical coordinates in this invocation.
        parameterized (bool): Whether each parameter row is a one-shot experiment.

    Returns:
        tuple[Mapping[Any, int], ...] | MeasurementFreeShots: Ordered measured
        rows, or cardinality-only coverage when the circuit has no measurements.

    Raises:
        ExecutionError: If execution failed, coverage is incomplete, or ordered
            sampling data is missing or malformed.
    """
    from ._indexed import require_result_coverage

    require_result_coverage(
        result, shots if parameterized else 1, 1 if parameterized else shots
    )
    if not circuit.clbits:
        return MeasurementFreeShots(shots)
    try:
        if parameterized:
            return tuple(result.get_counts(index) for index in range(shots))
        memory = result.get_memory(0)
    except Exception as error:
        raise ExecutionError(
            "Q8233: Aer template returned no ordered sampling data"
        ) from error
    if not isinstance(memory, Sequence) or isinstance(memory, (str, bytes)):
        raise ExecutionError("Q8233: Aer template returned malformed shot memory")
    if len(memory) != shots:
        raise ExecutionError("Q8233: Aer shot memory count disagrees with invocation")
    if any(not isinstance(sample, str) for sample in memory):
        raise ExecutionError("Q8233: Aer template returned malformed shot memory")
    return tuple({sample: 1} for sample in memory)


def backend_template_counts(
    result: Any, circuits: Sequence[Any]
) -> tuple[Mapping[Any, int], ...] | MeasurementFreeShots:
    """Validate backend coverage and retain empty measurements as cardinality.

    Args:
        result (Any): Provider result for the submitted one-shot circuit copies.
        circuits (Sequence[Any]): Submitted copies in logical-index order.

    Returns:
        tuple[Mapping[Any, int], ...] | MeasurementFreeShots: Ordered measurement
        maps or compact coverage for a measurement-free template.

    Raises:
        ExecutionError: If provider success, shot counts, or coverage is invalid.
    """
    from ._indexed import backend_indexed_counts, require_result_coverage

    if all(not circuit.clbits for circuit in circuits):
        require_result_coverage(result, sum(bool(c.num_qubits) for c in circuits), 1)
        return MeasurementFreeShots(len(circuits))
    return backend_indexed_counts(result, circuits)


def template_result(
    artifact: CompiledProgram,
    invocation: Invocation,
    native_counts: Sequence[Mapping[Any, int]] | MeasurementFreeShots,
    native_width: int,
    *,
    backend: str,
) -> SampleResult:
    """Evaluate indexed raw measurements in the native template before aggregation.

    Args:
        artifact (CompiledProgram): Native template and sealed logical result ABI.
        invocation (Invocation): Original public parameters and logical shot count.
        native_counts (Sequence[Mapping] | MeasurementFreeShots): Ordered SDK
            measurements or cardinality-only measurement-free coverage.
        native_width (int): SDK classical-bit width before native postprocessing.
        backend (str): Endpoint identity for the resulting samples.

    Returns:
        SampleResult: Native-evaluated counts carrying the original result ABI.

    Raises:
        ExecutionError: If any coordinate is malformed or native evaluation fails.
    """
    from ._payload import QiskitExecutable

    owner = artifact.execution_payload(QiskitExecutable).shot_template
    if owner is None:
        raise ExecutionError("Q8235: artifact has no native shot template")
    try:
        if isinstance(native_counts, MeasurementFreeShots):
            if native_counts.shots != invocation.shots or native_width != 0:
                raise ExecutionError(
                    "Q8233: measurement-free coverage disagrees with invocation"
                )
            evaluated = owner.postprocess_unmeasured(
                invocation.values(), invocation.shots, artifact.result_abi._to_bytes()
            )
        else:
            indexed = _indexed_measurements(
                native_counts, native_width, invocation.shots
            )
            evaluated = owner.postprocess(
                indexed,
                invocation.values(),
                invocation.shots,
                artifact.result_abi._to_bytes(),
            )
    except ValueError as error:
        raise ExecutionError(str(error)) from error
    result = SampleResult.from_counts(
        backend=backend,
        counts={tuple(bits): count for bits, count in evaluated},
        expected_width=artifact.result_abi.width,
    )
    return replace(result, result_abi=artifact.result_abi)


def _indexed_measurements(
    native_counts: Sequence[Mapping[Any, int]], native_width: int, shots: int
) -> list[tuple[int, list[int], int]]:
    """Normalize ordered SDK measurement rows for the native result owner.

    Args:
        native_counts (Sequence[Mapping]): One SDK count map per logical index.
        native_width (int): SDK classical-bit width.
        shots (int): Required number of single-shot coordinates.

    Returns:
        list[tuple[int, list[int], int]]: Indexed native-order bits and counts.

    Raises:
        ExecutionError: If cardinality, counts, or measurement width is invalid.
    """
    if len(native_counts) != shots:
        raise ExecutionError("Q8233: indexed result count disagrees with invocation")
    indexed = []
    for index, counts in enumerate(native_counts):
        if (
            not isinstance(counts, Mapping)
            or any(type(count) is not int or count < 0 for count in counts.values())
            or sum(counts.values()) != 1
        ):
            raise ExecutionError("Q8233: indexed circuit results require one shot")
        for raw_key, count in counts.items():
            if count == 0:
                continue
            text = (
                format(raw_key, f"0{native_width}b")
                if type(raw_key) is int
                else str(raw_key).replace(" ", "")
            )
            if len(text) != native_width or any(bit not in "01" for bit in text):
                raise ExecutionError(
                    "Q8233: indexed measurement width disagrees with template"
                )
            indexed.append((index, [int(bit) for bit in reversed(text)], count))
    return indexed
