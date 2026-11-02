"""Bounded witnesses for native authority and reusable execution preparation."""

from __future__ import annotations

import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.compilation import _materialize_launch_artifact
from qamomile.persistence import deserialize, serialize

_LOOP = """
import qamomile as qm
@qm.qkernel
def sample(rounds: qm.Param[qm.Int], theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for i in range(rounds):
        state = qm.rx(state, theta)
    return qm.measure(state)
"""


def test_launch_cache_keys_only_materialized_parameters(monkeypatch):
    """Changing runtime angles retains the topology and its parameter ABI."""
    from qamomile.qiskit import Qiskit

    target = Qiskit()
    artifact = qm.compile(qm.build_source(_LOOP).specialize("sample"), to=target)
    for current in (artifact, deserialize(serialize(artifact))):
        calls = []
        original = Qiskit.lower_circuit

        def lower(self, logical):
            calls.append(logical)
            return original(self, logical)

        with monkeypatch.context() as patch:
            patch.setattr(Qiskit, "lower_circuit", lower)
            children = [
                _materialize_launch_artifact(
                    current, {"rounds": rounds, "theta": theta}, target=target
                )
                for rounds, theta in ((2, 0.1), (2, 0.2), (3, 0.2), (2, 0.7))
            ]
        assert len(calls) == 3
        assert children[0] is children[1]
        assert children[0].execution_identity == children[3].execution_identity
        assert children[0].execution_identity != children[2].execution_identity
        assert [slot.name for slot in children[0].parameter_abi.slots] == ["theta"]


def test_compilation_consumes_one_verified_owner(monkeypatch):
    """Planning, analysis, ABI and emission receive handles, not source bytes."""
    from qamomile.qiskit import Qiskit

    program = qm.build_source(_LOOP.replace("range(rounds)", "range(2)")).specialize(
        "sample"
    )
    owner = program._native_program()
    seen = []
    for name in (
        "inspect_program",
        "derive_artifact_abi",
        "program_semantic_digest",
        "plan_target_lowering",
        "inspect_symbolic_logical_counts",
        "inspect_static_circuit",
    ):
        original = getattr(native, name)

        def inspect(value, *args, _name=name, _original=original):
            assert value is owner, _name
            seen.append(_name)
            return _original(value, *args)

        monkeypatch.setattr(native, name, inspect)
    qm.compile(program, to=Qiskit())
    assert "inspect_symbolic_logical_counts" in seen
    assert "inspect_static_circuit" in seen


def test_verified_owner_persistence_and_replacement():
    """Handles stay process local and an independent facade verifies itself."""
    program = qm.build_source(_LOOP).specialize("sample")
    with ThreadPoolExecutor(max_workers=3) as pool:
        owners = list(pool.map(lambda _: program._native_program(), range(6)))
    assert all(owner is owners[0] for owner in owners)
    restored = pickle.loads(pickle.dumps(program))
    assert restored == program
    assert restored.to_bytes() == program.to_bytes()
    assert restored._native_program() is not program._native_program()
    replaced = replace(program, module_name="independent_source")
    assert replaced._native_program() is not program._native_program()
    assert replaced.to_bytes() != program.to_bytes()
    with pytest.raises(ValueError):
        native.VerifiedProgramHandle(b"invalid")


@pytest.mark.quri_parts
@pytest.mark.parametrize("angle", ["0.2", "theta * 2.0 + 0.1", "theta * theta"])
def test_qulacs_preparation_reuses_native_templates(monkeypatch, angle):
    """Native conversion is once for constant/affine circuits; fallback is exact."""
    import numpy as np
    import qulacs
    import quri_parts.qulacs.circuit as conversion

    from qamomile.quri_parts import Qulacs
    from qamomile.quri_parts._payload import QuriExecutable

    source = f"""
import qamomile as qm
@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    q = qm.qubits(1)
    return qm.measure(qm.rx(q, {angle}))
"""
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=Qulacs().target
    )
    for current in (artifact, deserialize(serialize(artifact))):
        executable = current.execution_payload(QuriExecutable)
        constant = Mock(wraps=conversion.convert_circuit)
        parametric = Mock(wraps=conversion.convert_parametric_circuit)
        with monkeypatch.context() as patch:
            patch.setattr(conversion, "convert_circuit", constant)
            patch.setattr(conversion, "convert_parametric_circuit", parametric)
            for theta in (0.0, math.pi / 2, 0.0):
                circuit = executable.qulacs_circuit({"theta": theta})
                state = qulacs.QuantumState(1)
                circuit.update_quantum_state(state)
                value = (
                    0.2
                    if angle == "0.2"
                    else (
                        theta * theta if angle == "theta * theta" else theta * 2 + 0.1
                    )
                )
                np.testing.assert_allclose(
                    state.get_vector(),
                    [math.cos(value / 2), -1j * math.sin(value / 2)],
                    atol=1e-12,
                )
                circuit.add_X_gate(0)
        assert constant.call_count == (
            3 if angle == "theta * theta" else int(angle == "0.2")
        )
        assert parametric.call_count == int(angle == "theta * 2.0 + 0.1")


@pytest.mark.quri_parts
def test_qulacs_affine_aggregate_binding_is_isolated_across_threads():
    """SDK parameter order follows first use, with independent bound copies."""
    import numpy as np
    import qulacs
    from quri_parts.qulacs.circuit import convert_circuit

    from qamomile.quri_parts import Qulacs
    from qamomile.quri_parts._payload import QuriExecutable

    source = """
import qamomile as qm
@qm.qkernel
def sample(angles: qm.Param[qm.Vector[qm.Real, 2]]) -> qm.Bits[1]:
    q = qm.qubits(1)
    q = qm.ry(q, angles[1] * 2.0 + 0.1)
    q = qm.rz(q, angles[0] - angles[1])
    return qm.measure(q)
"""
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=Qulacs().target
    )
    executable = artifact.execution_payload(QuriExecutable)
    bindings = [{"angles": values} for values in ((0.2, 0.7), (0.9, -0.3))] * 3

    def vector(circuit):
        state = qulacs.QuantumState(1)
        circuit.update_quantum_state(state)
        return state.get_vector()

    expected = [vector(convert_circuit(executable.bind(row))) for row in bindings]
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(executable.qulacs_circuit, bindings))
    for circuit, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(vector(circuit), reference, atol=1e-12)
        circuit.add_X_gate(0)


@pytest.mark.parametrize("backend", ["cirq", "quri_parts", "braket"])
def test_retained_controls_reduce_modular_projection_without_simulation(backend):
    """The six-logical-wire witness never allocates its large statevector."""
    from importlib import import_module

    from qamomile.circuit import lower_to_circuit

    cls = {"cirq": "Cirq", "quri_parts": "QuriParts", "braket": "Braket"}[backend]
    target = getattr(import_module(f"qamomile.{backend}"), cls)()
    source = """
import qamomile as qm
@qm.qkernel
def state() -> qm.Q[qm.Bits[6]]:
    q, e = qm.split_at(qm.qubits(6), 4)
    q, e = qm.mod_pow(qm.as_uint(q), qm.as_uint(e), base=2, modulus=15)
    return qm.concat(qm.as_bits(q), qm.as_bits(e))
"""
    program = qm.build_source(source).specialize("state")
    projected = lower_to_circuit(
        program, realization_capabilities=target.capabilities.realization_capabilities
    )
    portable = lower_to_circuit(program)
    assert projected.width < portable.width
    assert projected.operation_count < portable.operation_count
    assert any(len(row.controls) > 2 for row in projected.body.instructions)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("cirq", marks=pytest.mark.cirq),
        pytest.param("braket", marks=pytest.mark.braket),
    ],
)
@pytest.mark.parametrize("control_state", [7, 5])
def test_controlled_sdk_operators_preserve_full_unitary(backend, control_state):
    """MCX, controlled SWAP and conditional phase preserve every input sector."""
    import numpy as np

    source = f"""
import qamomile as qm
@qm.qkernel
def pair(q: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    q = qm.x(q, target=0)
    q = qm.swap(q)
    return qm.global_phase(q, 0.37)
@qm.qkernel
def state() -> qm.Q[qm.Bits[5]]:
    c, q = qm.split_at(qm.qubits(5), 3)
    c, q = qm.control(pair, c, q, control_state={control_state})
    return qm.concat(c, q)
"""
    program = qm.build_source(source).specialize("state")
    expected = np.zeros((32, 32), complex)
    for column in range(32):
        row = column
        phase = 1
        if column & 7 == control_state:
            row ^= 1 << 3
            left, right = (row >> 3) & 1, (row >> 4) & 1
            if left != right:
                row ^= (1 << 3) | (1 << 4)
            phase = np.exp(0.37j)
        expected[row, column] = phase
    if backend == "cirq":
        import cirq

        from qamomile.cirq import Cirq
        from qamomile.cirq._payload import CirqExecutable

        payload = Cirq().lower(program).payload
        assert isinstance(payload, CirqExecutable)
        for current in (payload, CirqExecutable.from_bytes(payload.to_bytes())):
            matrix = current.circuit.unitary(
                qubit_order=list(reversed(cirq.LineQubit.range(5)))
            )
            np.testing.assert_allclose(matrix, expected, atol=1e-12)
    else:
        from braket.circuits import Circuit

        from qamomile.braket import Braket
        from qamomile.braket._payload import BraketExecutable

        payload = Braket().lower(program).payload
        assert isinstance(payload, BraketExecutable)
        order = [int(f"{i:05b}"[::-1], 2) for i in range(32)]
        for current in (payload, BraketExecutable.from_bytes(payload.to_bytes())):
            matrix = Circuit.from_ir(current.program()).to_unitary()
            np.testing.assert_allclose(
                matrix[np.ix_(order, order)], expected, atol=1e-12
            )
