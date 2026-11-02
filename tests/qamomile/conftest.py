"""Shared source programs for Qamomile compiler tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from qamomile.compilation import (
    CompilationReport,
    CompiledProgram,
    Invocation,
    Job,
    Reference,
)
from qamomile.ir import Program
from qamomile.persistence import deserialize, serialize
from qamomile.serialization import _TargetProgramView, decode_target_adjoint

_BackendLaunch = Callable[[object, CompiledProgram, Invocation], Job]


@pytest.fixture
def static_export():
    """Require the optional image exporter and system Cairo for export witnesses."""
    import importlib

    try:
        importlib.import_module("cairosvg")
    except (ImportError, OSError) as error:
        pytest.skip(f"optional CairoSVG/system Cairo unavailable: {error}")


def lower_adjoint_calls(program: Program) -> _TargetProgramView:
    """Return the native target-adjoint view used by structural tests."""

    from qamomile._native import compiler as native

    return decode_target_adjoint(
        bytes(native.lower_adjoint_calls_for_target(program.to_bytes()))
    )


_SDK_BACKEND_PREFIXES = {
    "test_qiskit_backend.py": "test_qiskit_",
    "test_quri_parts_backend.py": "test_quri_parts_",
    "test_braket_backend.py": "test_braket_",
    "test_cudaq_backend.py": "test_cudaq_",
    "test_cirq_backend.py": "test_cirq_",
}
_FAST_SDK_BACKEND_SEMANTICS = frozenset(
    {
        "keyed_parameters",
        "target_profile_is_derived_from_the_backend_and_fingerprinted",
        "quantum_records",
        "recursion",
        "sqrt",
        "executes_derived_measurement_results",
        "measures_unsigned_integer_registers",
        "materializes_dynamic_classical_index_at_launch",
        "executes_standard_python_record_results",
        "executes_symbolic_observable_recipe",
        "aggregates_private_measurement_rows_after_result_projection",
        "preserves_private_trailing_measurement_width",
        "executes_boolean_lifting",
        "executes_grover_adaptive_search",
        "executes_signal_composition",
        "zero_layer_qaoa",
        "approximate_qft",
        "classical_array_compatibility",
        "qubit_selection",
        "ibm_runtime_rejects_nonportable_seed",
        "ibm_runtime_measures_unsigned_integer_registers",
        "ibm_runtime_submit_snapshot_cancel_and_restore",
    }
)
_FAST_HUGR_TESTS = frozenset(
    {
        "test_hugr_keyed_parameters",
        "test_hugr_quantum_records",
        "test_hugr_recursion",
        "test_hugr_sqrt",
        "test_hugr_zero_layer_qaoa",
        "test_hugr_approximate_qft",
        "test_hugr_classical_array_compatibility",
        "test_hugr_qubit_selection",
        "test_direct_hugr_package_validates_and_executes",
        "test_hugr_aggregate_projection_fails_closed",
        "test_hugr_executes_a_derived_sampled_result",
        "test_hugr_executes_a_measurement_controlled_tail_loop",
        "test_hugr_executes_complete_boolean_measurement_dataflow",
        "test_hugr_executes_negative_indexed_boolean_result",
        "test_hugr_guards_a_runtime_divisor_without_a_totality_proof",
        "test_hugr_materializes_dynamic_classical_index_at_launch",
        "test_hugr_measures_unsigned_integer_registers",
        "test_helios_submit_snapshot_cancel_and_restore",
        "test_helios_measures_unsigned_integer_registers",
        "test_hugr_preserves_and_executes_measurement_control",
        "test_hugr_preserves_binary64_boolean_semantics",
        "test_hugr_rejects_a_library_type_over_its_port_budget",
        "test_hugr_rejects_a_static_library_loop_over_its_work_budget",
        "test_hugr_executes_boolean_lifting",
        "test_hugr_executes_grover_adaptive_search",
        "test_hugr_executes_signal_composition",
        "test_hugr_executes_symbolic_observable_recipe",
        "test_hugr_requires_native_boolean_lift_planner",
        "test_selene_decodes_only_well_formed_result_streams",
        "test_selene_endpoint_is_public",
        "test_selene_executes_shots_across_processes",
        "test_selene_fails_closed_without_the_optional_runtime",
        "test_selene_rejects_invalid_options",
        "test_selene_requires_result_instrumentation",
    }
)
_MAX_BACKEND_TEST_QUBITS = 10
_MAX_BACKEND_TEST_GATE_DEPTH = 256
_MAX_BACKEND_TEST_EMITTED_INSTRUCTIONS = 1_024
_MAX_BACKEND_TEST_SHOTS = 512
_MAX_FAST_BACKEND_TEST_GATE_DEPTH = 64
_MAX_FAST_BACKEND_TEST_EMITTED_INSTRUCTIONS = 256
_MAX_FAST_BACKEND_TEST_SHOTS = 128


def _resource_upper_int(
    report: CompilationReport,
    *identifiers: str,
) -> int | None:
    """Return the first proved integer upper bound in a compilation report.

    Args:
        report: Compilation report exposing ``metric``.
        *identifiers: Metric identifiers in preference order.

    Returns:
        Proved integer upper bound, or None when no listed metric is available.
    """

    for identifier in identifiers:
        try:
            value = report.metric(identifier)
        except KeyError:
            continue
        _, upper = value.numeric_enclosure()
        if upper is not None and not isinstance(upper, bool) and isinstance(upper, int):
            return upper
    return None


def _backend_endpoint_types(
    test_file: str,
    *,
    backend: object = None,
    module_name: object = None,
) -> tuple[type, ...]:
    """Return endpoint classes selected by a file or a mixed-backend test case.

    Args:
        test_file: Basename of the active Qamomile test file.
        backend: Optional backend label from the active parametrized case.
        module_name: Optional integration module from the active case.

    Returns:
        Reference plus endpoint classes belonging to the active backend.
    """

    dependencies = {
        "qiskit": "qiskit_aer",
        "quration": "pyqret",
        "quri_parts": "qulacs",
        "braket": "braket.circuits",
        "cudaq": "cudaq",
        "cirq": "cirq",
        "hugr": "hugr",
    }
    selected = backend
    if selected is None and isinstance(module_name, str):
        selected = module_name.removeprefix("qamomile.")
    if isinstance(selected, str) and selected in dependencies:
        # This resolver runs in a selected test's fixture, never collection.
        pytest.importorskip(dependencies[selected])
        test_file = f"test_{selected}_backend.py"

    if test_file == "test_qiskit_backend.py":
        from qamomile.qiskit import Aer, IBMRuntime, QiskitBackend

        return (Reference, Aer, IBMRuntime, QiskitBackend)
    if test_file == "test_quration_backend.py":
        from qamomile.quration import QurationLocal

        return (Reference, QurationLocal)
    if test_file == "test_quri_parts_backend.py":
        from qamomile.quri_parts import Qulacs

        return (Reference, Qulacs)
    if test_file == "test_braket_backend.py":
        from qamomile.braket import BraketDevice, BraketLocal

        return (Reference, BraketLocal, BraketDevice)
    if test_file == "test_cudaq_backend.py":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqLocal

        return (Reference, CudaqLocal)
    if test_file == "test_cirq_backend.py":
        from qamomile.cirq import CirqSampler, CirqSimulator

        return (Reference, CirqSimulator, CirqSampler)
    if test_file == "test_hugr_backend.py":
        from qamomile.hugr import Helios, Selene

        return (Reference, Helios, Selene)
    return (Reference,)


def _bounded_backend_launch(
    launch: _BackendLaunch,
    request: pytest.FixtureRequest,
) -> _BackendLaunch:
    """Wrap one endpoint launch with the shared simulator-work bounds.

    Args:
        launch: Unbound endpoint launch method.
        request: Active test request used to select the workload profile.

    Returns:
        Wrapped launch method enforcing the test-only bounds.
    """

    def launch_bounded(
        endpoint: object,
        artifact: CompiledProgram,
        invocation: Invocation,
    ) -> Job:
        """Reject oversized simulator examples before starting a worker.

        Args:
            endpoint: Backend endpoint receiving the artifact.
            artifact: Compiled artifact with a resource report.
            invocation: Validated launch values.

        Returns:
            Job returned by the original endpoint launch.
        """

        acceptance = any(
            request.node.get_closest_marker(marker) is not None
            for marker in ("backend_slow", "hugr_slow", "resource_slow")
        )
        max_depth = (
            _MAX_BACKEND_TEST_GATE_DEPTH
            if acceptance
            else _MAX_FAST_BACKEND_TEST_GATE_DEPTH
        )
        max_instructions = (
            _MAX_BACKEND_TEST_EMITTED_INSTRUCTIONS
            if acceptance
            else _MAX_FAST_BACKEND_TEST_EMITTED_INSTRUCTIONS
        )
        max_shots = (
            _MAX_BACKEND_TEST_SHOTS if acceptance else _MAX_FAST_BACKEND_TEST_SHOTS
        )
        report = artifact.report
        qubits = _resource_upper_int(
            report,
            "qamomile.circuit.virtual_qubits",
            "qamomile.logical.static_circuit_qubits",
            "qamomile.logical.peak_live_qubits",
            "qamomile.logical.allocated_qubits",
        )
        instructions = _resource_upper_int(
            report,
            "qamomile.emitted.instructions",
            "qamomile.logical.semantic_work",
        )
        depth = _resource_upper_int(
            report,
            "qamomile.logical.gate_depth",
            "qamomile.logical.dependency_span",
        )
        assert qubits is None or qubits <= _MAX_BACKEND_TEST_QUBITS, (
            "backend execution tests are limited to "
            f"{_MAX_BACKEND_TEST_QUBITS} qubits; got {qubits}"
        )
        assert depth is None or depth <= max_depth, (
            "backend execution tests are limited to "
            f"gate depth {max_depth}; got {depth}"
        )
        assert instructions is None or instructions <= max_instructions, (
            "backend execution tests must use a bounded shallow example; "
            f"got {instructions} emitted instructions"
        )
        assert invocation.shots <= max_shots, (
            "backend execution tests are limited to "
            f"{max_shots} shots; got {invocation.shots}"
        )
        return launch(endpoint, artifact, invocation)

    return launch_bounded


@pytest.fixture(autouse=True)
def _bound_backend_execution_scale(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every backend execution small enough for a development loop.

    Args:
        request: Current pytest test request.
        monkeypatch: Pytest mutation fixture scoped to this test.

    Returns:
        None.
    """

    callspec = getattr(request.node, "callspec", None)
    parameters = {} if callspec is None else callspec.params
    for endpoint_type in _backend_endpoint_types(
        request.path.name,
        backend=parameters.get("backend"),
        module_name=parameters.get("module_name"),
    ):
        monkeypatch.setattr(
            endpoint_type,
            "launch",
            _bounded_backend_launch(endpoint_type.launch, request),
        )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Separate bounded backend smoke loops from exhaustive acceptance.

    Args:
        items: Fully collected pytest items.

    Returns:
        None.
    """

    for item in items:
        test_name = getattr(item, "originalname", None) or item.name
        sdk_prefix = _SDK_BACKEND_PREFIXES.get(item.path.name)
        if sdk_prefix is not None:
            semantic_name = test_name.removeprefix(sdk_prefix)
            if semantic_name not in _FAST_SDK_BACKEND_SEMANTICS:
                item.add_marker(pytest.mark.backend_slow)
            continue
        if item.path.name != "test_hugr_backend.py":
            continue
        if test_name not in _FAST_HUGR_TESTS:
            item.add_marker(pytest.mark.hugr_slow)


BELL_SOURCE = """
import qamomile as qm

@qm.qkernel
def bell() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)
"""

DERIVED_RESULT_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    bits = qm.measure(qm.qubits(1))
    return not bits[0]
"""

BROADCAST_GATE_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[3]]:
    state = qm.qubits(3)
    state = qm.h(state)
    state = qm.z(state)
    state = qm.h(state)
    return qm.measure(state)
"""

UINT_MEASUREMENT_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(value: qm.Param[qm.UInt[3]]) -> qm.Shot[qm.UInt[3]]:
    return qm.measure(qm.prepare(value))

@qm.qkernel
def sample_reinterpreted() -> qm.Shot[qm.UInt[3]]:
    state = qm.prepare((1, 1, 0))
    return qm.measure(qm.as_uint(state))
"""

BOOLEAN_LIFT_SOURCE = """
import qamomile as qm

@qm.qkernel
def first(bits: qm.Bits[2]) -> qm.Bool:
    return bool(bits[0])

@qm.qkernel
def marked(bits: qm.Bits[2]) -> qm.Bool:
    return first(bits) and bool(bits[1])

@qm.qkernel
def phase_oracle(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.phase_by(marked, state)

@qm.qkernel
def phase_search() -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.h(state, target=1)
    state = qm.adjoint(phase_oracle, state)
    state = qm.h(state, target=0)
    state = qm.h(state, target=1)
    state = qm.x(state, target=0)
    state = qm.x(state, target=1)
    state = qm.cz(state, control=0, target=1)
    state = qm.x(state, target=0)
    state = qm.x(state, target=1)
    state = qm.h(state, target=0)
    return qm.h(state, target=1)

@qm.qkernel
def marked_pair(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    return (marked(bits), bool(bits[0]) ^ bool(bits[1]))

@qm.qkernel
def cyclic(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    return (not bool(bits[1]), bool(bits[0]))

@qm.qkernel
def coherent_embedding(source: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[4]]:
    target = qm.qubits(2)
    source, target = qm.xor_by(marked_pair, source, target)
    source = qm.permute_by(cyclic, source)
    return qm.concat(source, target)

@qm.qkernel
def coherent_search() -> qm.Q[qm.Bits[4]]:
    return coherent_embedding(phase_search())

@qm.qkernel
def bell_state() -> qm.Q[qm.Bits[2]]:
    state = qm.h(qm.qubits(2), target=0)
    return qm.cx(state, control=0, target=1)

@qm.qkernel
def phase_sample() -> qm.Shot[qm.Bits[4]]:
    return qm.measure(coherent_search())
"""


def exercise_boolean_lift_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute Boolean lifts and grouped expectation through one shared artifact."""

    import qamomile as qm

    module = qm.build_source(
        BOOLEAN_LIFT_SOURCE,
        module_name="cross_backend_boolean_lift",
    )
    sample = module.specialize("phase_sample")
    state = module.specialize("bell_state")
    assert qm.run(sample, on=endpoint, shots=4, seed=seed).result.counts == {
        (0, 1, 1, 0): 4
    }
    estimate = qm.expectation(
        state,
        qm.X(0) * qm.X(1) - qm.Y(0) * qm.Y(1) + qm.Z(0) * qm.Z(1),
        on=endpoint,
        shots=4,
        seed=seed,
        grouping="commuting",
    )
    assert estimate.value == 3.0
    assert estimate.shots == 4
    assert len(estimate.groups) == 1


def exercise_derived_result_backend(*, endpoint: object, seed: int = 0) -> None:
    """Require one backend to execute a derived measurement result."""

    import qamomile as qm

    program = qm.build_source(
        DERIVED_RESULT_SOURCE,
        module_name="cross_backend_derived_result",
    ).specialize("sample")
    result = qm.run(program, on=endpoint, shots=4, seed=seed).result
    assert result.counts == {(1,): 4}
    assert result.most_likely_value == 1
    assert result.values == {1: 4}


def exercise_broadcast_gate_backend(*, endpoint: object, seed: int = 0) -> None:
    """Require omitted unary selectors to apply independently to every lane."""

    import qamomile as qm

    program = qm.build_source(
        BROADCAST_GATE_SOURCE,
        module_name="cross_backend_broadcast_gate",
    ).specialize("sample")
    result = qm.run(program, on=endpoint, shots=4, seed=seed).result
    assert result.counts == {(1, 1, 1): 4}


def exercise_uint_measurement_backend(*, endpoint: object, seed: int = 0) -> None:
    """Require integer input and typed integer output for one UInt register."""

    import qamomile as qm

    program = qm.build_source(
        UINT_MEASUREMENT_SOURCE,
        module_name="cross_backend_uint_measurement",
    ).specialize("sample")
    result = qm.run(
        program,
        on=endpoint,
        parameters={"value": 6},
        shots=4,
        seed=seed,
    ).result
    assert result.counts == {(1, 1, 0): 4}
    assert result.values == {6: 4}
    assert result.most_likely_value == 6
    cast_result = qm.run(
        qm.build_source(
            UINT_MEASUREMENT_SOURCE,
            module_name="cross_backend_uint_reshape",
        ).specialize("sample_reinterpreted"),
        on=endpoint,
        shots=4,
        seed=seed,
    ).result
    assert cast_result.counts == {(1, 1, 0): 4}
    assert cast_result.values == {6: 4}
    assert cast_result.most_likely_value == 6


DYNAMIC_TARGET_SOURCE = """
import qamomile as qm

@qm.qkernel
def dynamic_target(target: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.x(state, target=target)
    return qm.measure(state)
"""


def exercise_dynamic_qubit_selection_backend(
    *, endpoint: object, seed: int = 0
) -> None:
    """Require one backend to materialize a launch-time wire selector."""

    import pytest

    import qamomile as qm
    from qamomile.effects import Requirement

    program = qm.build_source(
        DYNAMIC_TARGET_SOURCE,
        module_name="cross_backend_dynamic_target",
    ).specialize("dynamic_target")
    assert Requirement.DYNAMIC_QUBIT_SELECTION in program.entry.requirements
    for target, expected in ((0, (1, 0)), (1, (0, 1))):
        record = qm.run(  # type: ignore[arg-type]
            program,
            on=endpoint,
            parameters={"target": target},
            shots=4,
            seed=seed,
        )
        assert record.artifact.format == "qamomile.launch-materializer.v1"
        assert record.result.counts == {expected: 4}
    with pytest.raises(qm.ExecutionError, match="Q4303"):
        qm.run(  # type: ignore[arg-type]
            program,
            on=endpoint,
            parameters={"target": 2},
            shots=1,
            seed=seed,
        )


DYNAMIC_CLASSICAL_INDEX_SOURCE = """
import qamomile as qm

@qm.qkernel
def dynamic_angle(
    values: qm.Param[qm.Matrix[qm.Real, 2, 2]],
    row: qm.Param[qm.Int],
) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    state = qm.rx(state, values[row, 0])
    return qm.measure(state)
"""


def exercise_dynamic_classical_index_backend(
    *, endpoint: object, seed: int = 0
) -> None:
    """Require launch-time aggregate indexing without expanding its values."""

    import math

    import pytest

    import qamomile as qm
    from qamomile.effects import Requirement

    program = qm.build_source(
        DYNAMIC_CLASSICAL_INDEX_SOURCE,
        module_name="cross_backend_dynamic_classical_index",
    ).specialize("dynamic_angle")
    assert Requirement.DYNAMIC_CLASSICAL_INDEX in program.entry.requirements
    values = ((0.0, 0.0), (math.pi, 0.0))
    for row, expected in ((0, (0,)), (1, (1,))):
        record = qm.run(  # type: ignore[arg-type]
            program,
            on=endpoint,
            parameters={"values": values, "row": row},
            shots=4,
            seed=seed,
        )
        assert record.artifact.format == "qamomile.launch-materializer.v1"
        assert record.result.counts == {expected: 4}
    with pytest.raises(qm.ExecutionError, match="Q4303"):
        qm.run(  # type: ignore[arg-type]
            program,
            on=endpoint,
            parameters={"values": values, "row": 2},
            shots=1,
            seed=seed,
        )


HOST_DISPATCH_SOURCE = """
import qamomile as qm

@qm.qkernel
def host_dispatch(limit: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if limit > 0:
        state = qm.x(state)
    return qm.measure(state)
"""

LAUNCH_LOOP_SOURCE = """
import qamomile as qm

@qm.qkernel
def launch_loop(rounds: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(rounds):
        state = qm.x(state)
    return qm.measure(state)
"""

COLLECTION_LOOPS_SOURCE = """
import qamomile as qm

@qm.qkernel
def collection_loops() -> qm.Bits[3]:
    state = qm.qubits(3)
    for index, enabled in enumerate([True, False, True]):
        if enabled:
            state = qm.x(state, target=index)
    for enabled, target in zip([False, True, False], [0, 1, 2]):
        if enabled:
            state = qm.x(state, target=target)
    return qm.measure(state)
"""


STRUCTURAL_MATH_SOURCE = """
import qamomile as qm

@qm.qkernel
def structural_math(size: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(qm.ceil(qm.log2(size))):
        state = qm.x(state)
    return qm.measure(state)
"""


TYPED_MAP_SOURCE = """
import qamomile as qm

@qm.qkernel
def typed_map(
    size: qm.Const[qm.Int],
    switches: qm.Const[qm.Map[qm.Int, qm.Bool, size]],
    enabled: qm.Const[qm.Map[qm.Int, qm.Bool, size]],
) -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    for target, active in switches.items():
        if active and enabled[target]:
            state = qm.x(state, target=target)
    return qm.measure(state)
"""


MULTI_OWNER_DISPATCH_SOURCE = """
import qamomile as qm

@qm.qkernel
def multi_owner_dispatch(
    first: qm.Param[qm.Bool],
) -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[1]]]:
    left = qm.qubits(1)
    right = qm.qubits(1)
    if first:
        left = qm.x(left)
    else:
        right = qm.x(right)
    return qm.measure(left), qm.measure(right)
"""


def exercise_collection_loops_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute collection loops and immutable mapping through one endpoint."""

    import qamomile as qm

    program = qm.build_source(COLLECTION_LOOPS_SOURCE).specialize("collection_loops")
    assert qm.run(program, on=endpoint, shots=4, seed=seed).result.counts == {
        (1, 1, 1): 4
    }
    mapped = qm.build_source(TYPED_MAP_SOURCE).specialize(
        "typed_map",
        **{
            "switches": {0: True, 1: False},
            "enabled": {0: True, 1: True},
        },
    )
    assert qm.run(mapped, on=endpoint, shots=4, seed=seed).result.counts == {(1, 0): 4}


def exercise_structural_math_backend(*, endpoint: object, seed: int = 0) -> None:
    """Execute one launch-bound structural-math program on a backend endpoint."""

    import qamomile as qm

    program = qm.build_source(STRUCTURAL_MATH_SOURCE).specialize("structural_math")
    artifact = qm.compile(program, to=endpoint.target)  # type: ignore[attr-defined]
    result = qm.execute(
        artifact,
        on=endpoint,
        parameters={"size": 8},
        shots=4,
        seed=seed,
    )
    assert result.counts == {(1,): 4}


def exercise_multi_owner_dispatch_backend(
    *,
    endpoint: object,
    seed: int = 0,
) -> None:
    """Execute one multi-owner host-dispatch program on a backend endpoint."""

    import qamomile as qm

    program = qm.build_source(MULTI_OWNER_DISPATCH_SOURCE).specialize(
        "multi_owner_dispatch"
    )
    artifact = qm.compile(program, to=endpoint.target)  # type: ignore[attr-defined]
    for first, expected in ((False, (0, 1)), (True, (1, 0))):
        result = qm.execute(
            artifact,
            on=endpoint,
            parameters={"first": first},
            shots=4,
            seed=seed,
        )
        assert result.counts == {expected: 4}
        assert result.values == {tuple((bit,) for bit in expected): 4}


BASIS_SOURCE = """
import qamomile as qm

@qm.qkernel
def basis() -> qm.Bits[3]:
    state = qm.prepare((1, 0, 0))
    return qm.measure(state)
"""

RECORD_RESULT_SOURCE = """
from typing import NamedTuple
import qamomile as qm

class Observation(NamedTuple):
    first: qm.Shot[qm.Bits[1]]
    pair: qm.Shot[qm.Bits[2]]

@qm.qkernel
def record_result() -> Observation:
    first = qm.measure(qm.prepare((1,)))
    pair = qm.measure(qm.prepare((0, 1)))
    return Observation(first=first, pair=pair)
"""

COHERENT_RESHAPE_SOURCE = """
import qamomile as qm

@qm.qkernel
def reshaped_basis() -> qm.Bits[3]:
    state = qm.prepare((1, 0, 0))
    left, right = qm.split_at(state, 1)
    left = qm.x(left, target=0)
    right = qm.x(right, target=1)
    state = qm.concat(left, right)
    return qm.measure(state)

@qm.qkernel
def reshaped_bell() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    left, right = qm.split_at(state, 1)
    state = qm.concat(left, right)
    return qm.measure(state)

@qm.qkernel
def branched_step(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    left, right = qm.split_at(state, 1)
    left = qm.h(left)
    right = qm.x(right, target=1)
    return qm.concat(left, right)

@qm.qkernel
def branched_adjoint() -> qm.Bits[3]:
    state = qm.prepare((1, 0, 1))
    state = branched_step(state)
    state = qm.adjoint(branched_step, state)
    return qm.measure(state)

@qm.qkernel
def branched_temporary() -> qm.Unit:
    state = qm.qubits(3)
    left, right = qm.split_at(state, 1)
    left = qm.h(left)
    right = qm.x(right, target=1)
    state = qm.concat(left, right)
    return None
"""

ROTATION_SOURCE = """
import qamomile as qm

@qm.qkernel
def rotate(theta: qm.Param[qm.Real] = 0.0) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.ry(state, theta, target=0)
    return qm.measure(state)
"""

EXTENDED_GATE_SOURCE = """
import qamomile as qm

@qm.qkernel
def extended_gates() -> qm.Bits[3]:
    state = qm.prepare((1, 1, 0))
    state = qm.sdg(state, target=0)
    state = qm.s(state, target=0)
    state = qm.tdg(state, target=1)
    state = qm.t(state, target=1)
    state = qm.p(state, 0.0, target=2)
    state = qm.cp(state, 0.0, control=0, target=2)
    state = qm.rzz(state, 0.0, left=1, right=2)
    state = qm.ccx(state, control1=0, control2=1, target=2)
    return qm.measure(state)
"""

EXTENDED_DISCRETE_GATE_SOURCE = """
import qamomile as qm

@qm.qkernel
def extended_discrete_gates() -> qm.Bits[3]:
    state = qm.prepare((1, 1, 0))
    state = qm.sdg(state, target=0)
    state = qm.s(state, target=0)
    state = qm.tdg(state, target=1)
    state = qm.t(state, target=1)
    state = qm.ccx(state, control1=0, control2=1, target=2)
    return qm.measure(state)
"""

EXPLICIT_ADJOINT_SOURCE = """
import qamomile as qm

@qm.qkernel
def prepare_one(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    state = qm.h(state)
    state = qm.t(state)
    return state

@qm.qkernel
def explicit_adjoint() -> qm.Bits[1]:
    state = qm.qubits(1)
    state = prepare_one(state)
    state = qm.adjoint(prepare_one, state)
    return qm.measure(state)
"""

MEASURE_RESET_SOURCE = """
import qamomile as qm

@qm.qkernel
def measure_and_reset() -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[1]]]:
    state = qm.prepare((1,))
    observed, state = qm.measure_reset(state)
    return (observed, qm.measure(state))
"""

MEASURE_AXES_SOURCE = """
import qamomile as qm

@qm.qkernel
def measure_axes() -> tuple[
    qm.Shot[qm.Bool],
    qm.Shot[qm.Bool],
    qm.Shot[qm.Bool],
    qm.Shot[qm.Bool],
    qm.Shot[qm.Bool],
    qm.Shot[qm.Bool],
    qm.Shot[qm.Bits[1]],
    qm.Shot[qm.Bits[1]],
    qm.Shot[qm.Bits[1]],
]:
    z_state = qm.prepare((1,))
    z_first, z_state = qm.measure_z(z_state)
    z_second, z_state = qm.measure_z(z_state)
    z_final = qm.measure(z_state)

    x_state = qm.qubits(1)
    x_state = qm.h(x_state)
    x_first, x_state = qm.measure_x(x_state)
    x_second, x_state = qm.measure_x(x_state)
    x_state = qm.h(x_state)
    x_final = qm.measure(x_state)

    y_state = qm.qubits(1)
    y_state = qm.h(y_state)
    y_state = qm.s(y_state)
    y_first, y_state = qm.measure_y(y_state)
    y_second, y_state = qm.measure_y(y_state)
    y_state = qm.sdg(y_state)
    y_state = qm.h(y_state)
    y_final = qm.measure(y_state)
    return (
        z_first,
        z_second,
        x_first,
        x_second,
        y_first,
        y_second,
        z_final,
        x_final,
        y_final,
    )
"""

FEEDBACK_SOURCE = """
import qamomile as qm

@qm.qkernel
def feedback() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    flag, state = qm.measure_z(state, 0)
    if flag:
        state = qm.x(state, target=1)
    else:
        state = qm.z(state, target=1)
    return qm.measure(state)
"""

NEGATED_FEEDBACK_SOURCE = """
import qamomile as qm

@qm.qkernel
def negated_feedback() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    flag, state = qm.measure_z(state, 0)
    if not flag:
        state = qm.x(state, target=1)
    return qm.measure(state)
"""

COMPOUND_FEEDBACK_SOURCE = """
import qamomile as qm

@qm.qkernel
def compound_feedback() -> qm.Bits[3]:
    state = qm.prepare((1, 0, 0))
    left, state = qm.measure_z(state, 0)
    right, state = qm.measure_z(state, 1)
    if (left and not right) or (right and not left):
        state = qm.x(state, target=2)
    return qm.measure(state)
"""

RETAINED_HUGR_SOURCE = """
import qamomile as qm

@qm.qkernel
def mark(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    state = qm.x(state, target=0)
    state = qm.x(state, target=2)
    return state

@qm.qkernel
def retained_basis() -> qm.Bits[3]:
    state = qm.qubits(3)
    state = mark(state)
    return qm.measure(state)
"""

MODULAR_CLEANUP_SOURCE = """
import qamomile as qm

@qm.qkernel
def scratch_step(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    state = qm.h(state, target=1)
    state = qm.cx(state, target=0, control=1)
    state = qm.s(state, target=0)
    state = qm.t(state, target=1)
    return state

@qm.qkernel
def modular_cleanup() -> qm.Bits[3]:
    scratch = qm.qubits(2)
    scratch = scratch_step(scratch)
    output = qm.prepare((1, 0, 0))
    return qm.measure(output)
"""

MULTI_INPUT_CLEANUP_SOURCE = """
import qamomile as qm

@qm.qkernel
def combine(
    left: qm.Q[qm.Bits[1]],
    right: qm.Q[qm.Bits[1]],
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    state = qm.concat(left, right)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.split_at(state, 1)

@qm.qkernel
def outer(
    left: qm.Q[qm.Bits[1]],
    right: qm.Q[qm.Bits[1]],
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    left, right = combine(left, right)
    left = qm.z(left)
    return (left, right)

@qm.qkernel
def multi_input_cleanup() -> qm.Bits[1]:
    left = qm.qubits(1)
    right = qm.qubits(1)
    pair = outer(left, right)
    output = qm.prepare((1,))
    return qm.measure(output)
"""

TRAILING_MEASUREMENT_SOURCE = """
import qamomile as qm

@qm.qkernel
def trailing_measurement() -> qm.Bits[1]:
    first_state = qm.qubits(1)
    first = qm.measure(first_state)
    trailing_state = qm.qubits(1)
    ignored = qm.measure(trailing_state)
    return first
"""

JOINT_PRIVATE_MEASUREMENT_SOURCE = """
import qamomile as qm

@qm.qkernel
def joint_private_measurement() -> qm.Bits[1]:
    first_state = qm.qubits(1)
    private_state = qm.h(qm.qubits(1))
    first = qm.measure(first_state)
    ignored = qm.measure(private_state)
    return first
"""

STRUCTURED_RESULT_SOURCE = """
import qamomile as qm

@qm.qkernel
def structured_result() -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[2]]]:
    left = qm.prepare((1,))
    right = qm.prepare((0, 1))
    return (qm.measure(left), qm.measure(right))
"""

CALLABLE_TRANSFORMS_SOURCE = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def callable_transforms() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bits[1]]]:
    index = qm.prepare(True)
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (unchanged, flipped), target)
    index, target = qm.control(flipped, index, target, control_state=0)
    return qm.measure(index), qm.measure(target)
"""

COHERENT_ADJOINT_SOURCE = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def selected(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    index, target = qm.split_at(state, 1)
    index, target = qm.multiplex(index, (unchanged, flipped), target)
    return qm.concat(index, target)

@qm.qkernel
def selected_adjoint_roundtrip() -> qm.Shot[qm.Bits[2]]:
    state = qm.prepare((1, 0))
    state = selected(state)
    state = qm.adjoint(selected, state)
    return qm.measure(state)
"""

INDEXED_CONTROL_SOURCE = """
import qamomile as qm

@qm.qkernel
def flip_target(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def indexed_control_state() -> qm.Q[qm.Bits[3]]:
    controls = qm.prepare((1, 0))
    target = qm.qubits(1)
    system = qm.concat(controls, target)
    controls, target = qm.split_at(system, 2)
    for index in range(2):
        controls, target = qm.control(
            flip_target,
            controls,
            target,
            at=index,
        )
    return qm.concat(controls, target)

@qm.qkernel
def indexed_control_sample() -> qm.Shot[qm.Bits[3]]:
    return qm.measure(indexed_control_state())
"""

CONTROLLED_PHASE_SOURCE = """
import qamomile as qm

@qm.qkernel
def scalar_phase(state: qm.Q[qm.Bits[1]], angle: qm.Real) -> qm.Q[qm.Bits[1]]:
    return qm.global_phase(state, angle)

@qm.qkernel
def controlled_phase() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bits[1]]]:
    control = qm.prepare(False)
    control = qm.h(control)
    target = qm.qubits(1)
    control, target = qm.control(
        scalar_phase,
        control,
        target,
        3.141592653589793,
    )
    control = qm.h(control)
    return qm.measure(control), qm.measure(target)
"""

PAULI_EVOLUTION_SOURCE = """
import qamomile as qm

@qm.qkernel
def pauli_evolution(
    hamiltonian: qm.Const[qm.Observable],
) -> qm.Shot[qm.Bits[2]]:
    state = qm.h(qm.qubits(2), target=0)
    state = qm.pauli_evolve(state, hamiltonian, 3.141592653589793)
    state = qm.h(state, target=0)
    return qm.measure(state)
"""

NORMALIZED_MODEL_SOURCE = """
import qamomile as qm

@qm.qkernel
def normalized_model_sample(
    hamiltonian: qm.Const[qm.Observable],
) -> qm.Shot[qm.Bits[1]]:
    state = qm.h(qm.qubits(1), target=0)
    state = qm.pauli_evolve(state, hamiltonian, 1.5707963267948966)
    state = qm.h(state, target=0)
    return qm.measure(state)
"""


def exercise_normalized_model_backend(*, endpoint: object, seed: int = 0) -> None:
    """Compile and execute one natively normalized model on a backend.

    Args:
        endpoint: Backend endpoint exposing the standard Qamomile target.
        seed: Portable simulator seed.

    Returns:
        None.
    """

    import qamomile as qm
    from qamomile.optimization import BinaryModel

    model = BinaryModel.from_ising({(9,): 4.0}, constant=2.0).normalize_by_abs_max()
    program = qm.build_source(
        NORMALIZED_MODEL_SOURCE,
        module_name="cross_backend_normalized_model",
    ).specialize(
        "normalized_model_sample",
        **{"hamiltonian": model.hamiltonian},
    )
    artifact = qm.compile(program, to=endpoint.target)  # type: ignore[attr-defined]
    assert serialize(deserialize(serialize(artifact))) == serialize(artifact)
    result = qm.run(artifact, on=endpoint, shots=4, seed=seed).result
    assert result.counts == {(1,): 4}
    decoded = model.decode(result)
    assert decoded.lowest.values == {9: -1}
    assert decoded.lowest.energy == -0.5


SYMBOLIC_OBSERVABLE_SOURCE = """
import qamomile as qm

@qm.qkernel
def transverse_field(
    n: qm.Const[qm.Int],
    coefficients: qm.Const[qm.Vector[qm.Real, n]],
) -> qm.Observable:
    hamiltonian = qm.zero_observable(n)
    for index in range(n):
        hamiltonian = hamiltonian + coefficients[index] * qm.X(index)
    return hamiltonian

@qm.qkernel
def symbolic_evolution(
    coefficients: qm.Const[qm.Vector[qm.Real, 2]],
) -> qm.Shot[qm.Bits[2]]:
    hamiltonian = transverse_field(2, coefficients)
    state = qm.pauli_evolve(qm.qubits(2), hamiltonian, 1.5707963267948966)
    return qm.measure(state)

@qm.qkernel
def plus_state() -> qm.Q[qm.Bits[2]]:
    return qm.h(qm.qubits(2), target=0)
"""


def exercise_symbolic_observable_backend(*, endpoint: object, seed: int = 0) -> None:
    """Build and execute one symbolic Observable recipe on a backend.

    Args:
        endpoint: Backend endpoint exposing the standard Qamomile target.
        seed: Portable simulator seed.

    Returns:
        None.
    """

    import qamomile as qm

    module = qm.build_source(
        SYMBOLIC_OBSERVABLE_SOURCE,
        module_name="cross_backend_symbolic_observable",
    )
    constants = {"n": 2, "coefficients": (1.0, 0.0)}
    observable = qm.evaluate(module.specialize("transverse_field", **constants))
    assert observable == qm.zero_observable(2) + qm.X(0)

    program = module.specialize(
        "symbolic_evolution",
        **{"coefficients": constants["coefficients"]},
    )
    artifact = qm.compile(program, to=endpoint.target)  # type: ignore[attr-defined]
    assert serialize(deserialize(serialize(artifact))) == serialize(artifact)
    result = qm.execute(artifact, on=endpoint, shots=4, seed=seed)
    assert result.counts == {(1, 0): 4}

    estimate = qm.expectation(
        module.specialize("plus_state"),
        observable,
        on=endpoint,
        shots=4,
        seed=seed,
    )
    assert estimate.value == 1.0


RETURNED_FUNCTION_SOURCE = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def apply(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return operation(state)

@qm.qkernel
def forward(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
) -> qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    return operation

@qm.qkernel
def forward_twice(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
) -> qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    forwarded = forward(operation)
    return forwarded

@qm.qkernel
def apply_forwarded(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return apply(forward(operation), state)

@qm.qkernel
def retain_or_default(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    flag: qm.Param[qm.Bool],
) -> qm.Param[qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]]:
    return operation if flag else unchanged

@qm.qkernel
def choose(
    flag: qm.Param[qm.Bool],
) -> qm.Param[qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]]:
    operation = flipped if flag else unchanged
    return operation

@qm.qkernel
def sample_inline(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(choose(flag)(qm.qubits(1)))

@qm.qkernel
def sample_forwarded(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(apply(forward_twice(choose(flag=flag)), qm.qubits(1)))

@qm.qkernel
def sample_keyword_order(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(
        apply(state=qm.qubits(1), operation=forward_twice(choose(flag=flag)))
    )

@qm.qkernel
def sample_generated(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(apply_forwarded(choose(flag=flag), qm.qubits(1)))

@qm.qkernel
def sample_nested_choice(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    operation = forward(flipped) if flag else forward(unchanged)
    return qm.measure(apply(operation, qm.qubits(1)))

@qm.qkernel
def sample_retained(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    operation = retain_or_default(flipped, flag)
    return qm.measure(operation(qm.qubits(1)))

@qm.qkernel
def sample_immediate() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(forward(flipped)(qm.qubits(1)))
"""

STATE_PREPARATION_SOURCE = """
import qamomile as qm

@qm.qkernel
def plus_state() -> qm.Q[qm.Bits[1]]:
    state = qm.qubits(1)
    return qm.h(state)
"""
