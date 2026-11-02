"""Validate standalone QIR modules with LLVM and an actual QIR consumer."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

import qamomile as qm
from qamomile.circuit import lower_to_circuit
from qamomile.persistence import deserialize, export_payload, serialize
from qamomile.qir import FORMAT, QIR, load_bitcode, load_ir

from .conftest import BASIS_SOURCE, BELL_SOURCE, DERIVED_RESULT_SOURCE
from .test_stdlib import (
    amplitude_two_sample,
    grover_sample,
    qft_frequency_three,
    qft_roundtrip,
    qpe_sample,
    qsvt_phase_sample,
    static_product_formula_sample,
    static_variational_sample,
)

pytestmark = pytest.mark.qir
pyqir = pytest.importorskip("pyqir")


def _counts(
    artifact: qm.CompiledProgram, shots: int = 8, seed: int = 7
) -> dict[tuple[int, ...], int]:
    """Execute actual exported bitcode and read its ordered flat output array."""
    runner = pytest.importorskip("qirrunner")
    assert (
        artifact.report.metric("qamomile.circuit.virtual_qubits").numeric_enclosure()[1]
        <= 10
    )
    assert shots <= 512
    handler = runner.OutputHandler()
    runner.run_bytes(
        load_bitcode(artifact), shots=shots, rng_seed=seed, output_fn=handler.handle
    )
    counts: Counter[tuple[int, ...]] = Counter()
    row: list[int] = []
    for line in handler.get_output().splitlines():
        fields = line.split("\t")
        if fields[:2] == ["OUTPUT", "RESULT"]:
            row.append(int(fields[2]))
        elif fields[0] == "END":
            assert len(row) == artifact.result_abi.width
            counts[tuple(row)] += 1
            row.clear()
    assert sum(counts.values()) == shots
    return dict(counts)


def _compile(source: str, entry: str = "sample") -> qm.CompiledProgram:
    """Compile a source entry through the public lifecycle."""
    return qm.compile(qm.build_source(source).specialize(entry), to=QIR())


def test_qir_bell_module_is_base_profile_and_llvm_verified() -> None:
    """The standalone module contains the required QIR profile and ABI facts."""
    artifact = _compile(BELL_SOURCE, "bell")
    assert artifact.format == FORMAT
    assert artifact.report.stages.count("rust-qcore-standard-optimization") == 1
    module = pyqir.Module.from_bitcode(pyqir.Context(), load_bitcode(artifact))
    assert module.verify() is None
    ir = load_ir(artifact)
    assert pyqir.Module.from_ir(pyqir.Context(), ir).verify() is None
    assert '"qir_profiles"="base_profile"' in ir
    assert '"required_num_qubits"="2"' in ir
    assert '"required_num_results"="2"' in ir
    assert '"output_labeling_schema"="qamomile.source-order.v1"' in ir
    assert '!"qir_major_version", i32 1' in ir
    assert '!"dynamic_qubit_management", i1 false' in ir
    assert '!"dynamic_result_management", i1 false' in ir
    assert '"irreversible"' in ir and "%Result* writeonly" in ir
    entry = next(
        function for function in module.functions if pyqir.is_entry_point(function)
    )
    assert entry.name == "main" and len(entry.basic_blocks) == 4
    assert "define i64 @main()" in ir and "ret i64 0" in ir
    instructions = [
        instruction
        for block in entry.basic_blocks
        for instruction in block.instructions
    ]
    calls = [
        instruction.callee.name
        for instruction in instructions
        if isinstance(instruction, pyqir.Call)
    ]
    assert calls[0] == "__quantum__rt__initialize"
    first_measurement = calls.index("__quantum__qis__mz__body")
    assert all(
        name.startswith("__quantum__rt__") or name == "__quantum__qis__mz__body"
        for name in calls[first_measurement:]
    )
    assert artifact.report.metric(
        "qamomile.emitted.instructions"
    ).numeric_enclosure() == (len(instructions), len(instructions))


def test_qir_export_roundtrip_is_deterministic_and_executable() -> None:
    """Persistence preserves native ABI facts and the original standalone module."""
    first = _compile(BASIS_SOURCE, "basis")
    second = _compile(BASIS_SOURCE, "basis")
    assert load_bitcode(first) == load_bitcode(second) == export_payload(first)
    assert serialize(first) == serialize(second)
    restored = deserialize(serialize(first))
    assert load_bitcode(restored) == load_bitcode(first)
    assert restored.result_abi == first.result_abi
    assert serialize(restored) == serialize(first)
    assert _counts(restored) == {(1, 0, 0): 8}


def test_qir_runner_samples_bell_with_repeatable_seed() -> None:
    """The external QIR runtime preserves entanglement and measurement order."""
    artifact = _compile(BELL_SOURCE, "bell")
    first = _counts(artifact, shots=64, seed=23)
    assert first == _counts(artifact, shots=64, seed=23)
    assert set(first) == {(0, 0), (1, 1)}


@pytest.mark.parametrize(
    "function",
    [
        qft_roundtrip,
        qft_frequency_three,
        amplitude_two_sample,
        grover_sample,
        qpe_sample,
        qsvt_phase_sample,
    ],
)
def test_qir_shared_static_library_witnesses(function: qm.QKernel) -> None:
    """Fourier, preparation, Grover, QPE and QSVT export and execute unchanged."""
    expected = qm.run(function, on=qm.Reference(), shots=8, seed=7).result.counts
    artifact = qm.compile(function, to=QIR())
    assert _counts(artifact) == expected


def test_qir_records_reordered_duplicate_and_private_measurements() -> None:
    """Only source-selected results are recorded, with duplicates and ordering."""
    source = """
import qamomile as qm
@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bit], qm.Shot[qm.Bit]]:
    bits = qm.measure(qm.prepare((1, 0, 0)))
    return bits[1], bits[0], bits[1]
"""
    artifact = _compile(source)
    assert artifact.result_abi.source_order == (1, 0, 1)
    assert artifact.result_abi.native_width == 3
    assert _counts(artifact) == {(0, 1, 0): 8}
    assert '"required_num_results"="3"' in load_ir(artifact)


def test_qir_specializes_const_angles() -> None:
    """Specialized constants do not become LLVM entry arguments or arithmetic."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(angle: qm.Const[qm.Real]) -> qm.Bits[1]:
    state = qm.rx(qm.qubits(1), angle)
    return qm.measure(state)
"""
    kernel = qm.build_source(source)["sample"]
    artifact = qm.compile(kernel, to=QIR(), angle=3.141592653589793)
    assert artifact.parameter_abi.slots == ()
    assert _counts(artifact) == {(1,): 8}


@pytest.mark.parametrize(
    "function,bindings",
    [
        (static_variational_sample, {"hamiltonian": qm.Z(0)}),
        (
            static_product_formula_sample,
            {"hamiltonians": (qm.X(0), qm.Hamiltonian.zero(num_qubits=1))},
        ),
    ],
)
def test_qir_shared_static_variational_and_evolution(
    function: qm.QKernel, bindings: dict[str, object]
) -> None:
    """Static QAOA, FQAOA and product evolution use the same native library source."""
    program = qm.specialize(function, **bindings)
    expected = qm.run(program, on=qm.Reference(), shots=8, seed=7).result.counts
    assert _counts(qm.compile(program, to=QIR())) == expected


def test_qir_empty_unit_entry() -> None:
    """A zero-resource entry records an empty output array without allocation."""
    source = """
import qamomile as qm
@qm.qkernel
def sample() -> qm.Unit:
    return None
"""
    artifact = _compile(source)
    assert '"required_num_qubits"="0"' in load_ir(artifact)
    assert _counts(artifact) == {(): 8}


@pytest.mark.parametrize(
    "body, annotation",
    [
        ("state = qm.reset(qm.qubits(1))\n    return qm.measure(state)", "qm.Bits[1]"),
        ("state = qm.qubits(1)\n    qm.discard(state)\n    return None", "qm.Unit"),
        (
            "state = qm.qubits(1)\n    bit, state = qm.measure_z(state, target=0)\n    if bit:\n        state = qm.x(state)\n    return qm.measure(state)",
            "qm.Bits[1]",
        ),
    ],
)
def test_qir_rejects_unsupported_runtime_effects(body: str, annotation: str) -> None:
    """Feedback, reset and discard fail closed through native capability planning."""
    source = f"import qamomile as qm\n@qm.qkernel\ndef sample() -> {annotation}:\n    {body}\n"
    with pytest.raises(qm.ArtifactError, match="Q7015"):
        _compile(source)


@pytest.mark.parametrize(
    "parameter, body",
    [
        ("angle: qm.Param[qm.Real]", "state = qm.rx(state, angle)"),
        ("select: qm.Param[qm.Bool]", "if select:\n        state = qm.x(state)"),
    ],
)
def test_qir_rejects_launch_parameters_instead_of_exporting_a_recipe(
    parameter: str, body: str
) -> None:
    """Standalone export cannot silently produce a family or launch materializer."""
    source = f"import qamomile as qm\n@qm.qkernel\ndef sample({parameter}) -> qm.Bits[1]:\n    state = qm.qubits(1)\n    {body}\n    return qm.measure(state)\n"
    with pytest.raises(qm.ArtifactError, match="Q7015"):
        _compile(source)


def test_qir_rejects_computed_classical_output() -> None:
    """The Base Profile cannot silently export raw bits for a computed result."""
    with pytest.raises(qm.CapabilityError, match="Q9002"):
        _compile(DERIVED_RESULT_SOURCE)
    with pytest.raises(qm.CapabilityError, match="Q9002"):
        _compile(
            "import qamomile as qm\n@qm.qkernel\ndef sample() -> qm.Int:\n    return 3\n"
        )


def test_qir_rejects_gates_after_an_independent_measurement() -> None:
    """Profile validation rejects interleaved unitary and irreversible calls."""
    source = """
import qamomile as qm
@qm.qkernel
def sample() -> tuple[qm.Bits[1], qm.Bits[1]]:
    first = qm.measure(qm.prepare((1,)))
    second = qm.measure(qm.prepare((1,)))
    return first, second
"""
    with pytest.raises(qm.CapabilityError, match="all gates before measurements"):
        _compile(source)


def test_qir_rejects_unsupported_coherent_controls() -> None:
    """Three controls fail before allocating a partially emitted LLVM module."""
    source = """
import qamomile as qm
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)
@qm.qkernel
def sample() -> qm.Bits[4]:
    control = qm.qubits(3)
    target = qm.qubits(1)
    control, target = qm.control(flip, control, target)
    return qm.measure(qm.concat(control, target))
"""
    with pytest.raises(qm.CapabilityError, match="up to two controls"):
        _compile(source)


def test_qir_budget_precedes_llvm_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instruction expansion and output costs are bounded before LLVM construction."""
    from qamomile.qir import _emission

    def unexpected_context() -> Any:
        pytest.fail("over-budget emission must not allocate an LLVM context")

    logical = lower_to_circuit(qm.build_source(BELL_SOURCE).specialize("bell"))
    monkeypatch.setattr(pyqir, "Context", unexpected_context)
    # Two gates, two measurements, two result calls and six framing instructions.
    monkeypatch.setattr(_emission, "MAX_QIR_INSTRUCTIONS", 11)
    with pytest.raises(qm.CapabilityError, match="Q9003"):
        _emission.emit_module(logical)
    with pytest.raises(qm.CapabilityError, match="Q9003"):
        _emission.emit_module(replace(logical, measurement_count=1_000_000))


def test_qir_instruction_budget_accepts_the_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The emission budget includes exactly the generated runtime and QIS calls."""
    from qamomile.qir import _emission

    monkeypatch.setattr(_emission, "MAX_QIR_INSTRUCTIONS", 12)
    assert _counts(_compile(BELL_SOURCE, "bell"), shots=2)


def test_qir_sdk_inspection_rejects_foreign_artifacts_and_malformed_bitcode() -> None:
    """Export helpers and the persistence loader reject invalid payloads."""
    from qamomile.qir._payload import QIRExecutable

    foreign = qm.compile(
        qm.build_source(BELL_SOURCE).specialize("bell"), to=qm.Reference()
    )
    for loader in (load_ir, load_bitcode):
        with pytest.raises(qm.ExecutionError, match="Q9006"):
            loader(foreign)
    with pytest.raises(qm.ExecutionError, match="Q9005"):
        QIRExecutable.from_bytes(b"not llvm bitcode")


@pytest.mark.parametrize(
    "gate, controls",
    [
        (gate, 0)
        for gate in (
            "h",
            "x",
            "y",
            "z",
            "s",
            "sdg",
            "t",
            "tdg",
            "rx",
            "ry",
            "rz",
            "p",
            "global_phase",
        )
    ]
    + [
        (gate, 1)
        for gate in (
            "h",
            "x",
            "y",
            "z",
            "s",
            "sdg",
            "t",
            "tdg",
            "rx",
            "ry",
            "rz",
            "p",
            "global_phase",
        )
    ]
    + [(gate, 2) for gate in ("x", "y", "z")],
)
def test_qir_emitted_controlled_operator_preserves_relative_phase(
    gate: str, controls: int
) -> None:
    """Parsed LLVM calls match every column of an independent SDK gate operator."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import Qiskit, load_circuit

    angle = ", 0.713" if gate in {"rx", "ry", "rz", "p", "global_phase"} else ""
    apply = (
        "control, target = qm.control(operation, control, target)"
        if controls
        else "target = operation(target)"
    )
    preparation = f"control = qm.qubits({controls})" if controls else ""
    output = "qm.concat(control, target)" if controls else "target"
    source = f"""
import qamomile as qm
@qm.qkernel
def operation(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.{gate}(state{angle})
@qm.qkernel
def sample() -> qm.Bits[{controls + 1}]:
    {preparation}
    target = qm.qubits(1)
    {apply}
    return qm.measure({output})
"""
    program = qm.build_source(source).specialize("sample")
    artifact = qm.compile(program, to=QIR())
    module = pyqir.Module.from_bitcode(pyqir.Context(), load_bitcode(artifact))
    entry = next(function for function in module.functions if function.name == "main")
    circuit = QuantumCircuit(controls + 1)
    for instruction in (
        instruction
        for block in entry.basic_blocks
        for instruction in block.instructions
    ):
        if not isinstance(
            instruction, pyqir.Call
        ) or not instruction.callee.name.startswith("__quantum__qis__"):
            continue
        name = instruction.callee.name.removeprefix("__quantum__qis__")
        if name == "mz__body":
            continue
        opcode = (
            name.replace("__body", "").replace("s__adj", "sdg").replace("t__adj", "tdg")
        )
        opcode = {"cnot": "cx"}.get(opcode, opcode)
        args = [
            arg.value if isinstance(arg, pyqir.FloatConstant) else pyqir.qubit_id(arg)
            for arg in instruction.args
        ]
        getattr(circuit, opcode)(*args)
    expected_circuit = load_circuit(
        qm.compile(program, to=Qiskit())
    ).remove_final_measurements(inplace=False)
    expected = Operator(expected_circuit).data
    actual = Operator(circuit).data
    # Removing a single overall phase must preserve all controlled relative phases.
    pivot = np.unravel_index(np.argmax(np.abs(expected)), expected.shape)
    phase = actual[pivot] / expected[pivot]
    np.testing.assert_allclose(actual, phase * expected, atol=1e-12, rtol=1e-12)
