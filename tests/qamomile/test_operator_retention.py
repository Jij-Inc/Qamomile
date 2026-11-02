"""Late composite emission preserves complete controlled unitaries."""

from __future__ import annotations

import runpy

import numpy as np
import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.circuit import CircuitInstruction, lower_to_circuit
from qamomile.compilation import RealizationCapability
from qamomile.persistence import deserialize, export_payload, serialize


def test_modular_swap_capability_selects_retention_without_changing_meaning() -> None:
    """The Rust plan retains exactly the swaps accepted by the next consumer."""
    options = ["mod_mul.schoolbook_ripple"]
    fallback = native.plan_mod_pow(2, 1, 2, 3, False, 1, ["mcx"], options)
    retained = native.plan_mod_pow(
        2, 1, 2, 3, False, 1, ["mcx", "controlled_swap"], options
    )
    assert fallback[:3] == retained[:3]
    swaps = [step for step in retained[3] if step[0] == "swap"]
    assert len(swaps) == 2
    assert len(fallback[3]) - len(retained[3]) == 2 * len(swaps)
    assert all(len(wires) == 2 and len(controls) == 3 for _, wires, controls in swaps)
    assert all(gate != "swap" for gate, _, _ in fallback[3])


def test_pauli_native_projection_preserves_control_inverse_and_source() -> None:
    """Native and portable projections share the same unchanged QCore."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def evolution(state: qm.Q[qm.Bits[3]], hamiltonian: qm.Const[qm.Observable], time: qm.Param[qm.Real]) -> qm.Q[qm.Bits[3]]:
    return qm.pauli_evolve(state, hamiltonian, time)
@qm.qkernel
def inverse(state: qm.Q[qm.Bits[3]], hamiltonian: qm.Const[qm.Observable], time: qm.Param[qm.Real]) -> qm.Q[qm.Bits[3]]:
    return qm.adjoint(evolution, state, hamiltonian, time)
@qm.qkernel
def example(hamiltonian: qm.Const[qm.Observable], time: qm.Param[qm.Real]) -> qm.Bits[5]:
    controls, state = qm.split_at(qm.h(qm.qubits(5)), 2)
    controls, state = qm.control(inverse, controls, state, hamiltonian, time)
    return qm.measure(qm.concat(controls, state))
""").specialize("example", hamiltonian=0.2 + 0.7 * qm.X(0) * qm.Y(2))
    before = program.to_bytes()
    retained = lower_to_circuit(
        program,
        realization_capabilities=frozenset({RealizationCapability.PAULI_EVOLUTION}),
    )
    fallback = lower_to_circuit(program)
    evolution = [i for i in retained.body.instructions if i.opcode == "pauli_evolution"]
    assert len(evolution) == 1 and len(evolution[0].controls) == 2
    assert retained.operation_count < fallback.operation_count
    assert "pauli.native" in retained.realizations
    assert all(i.opcode != "pauli_evolution" for i in fallback.body.instructions)
    assert program.to_bytes() == before


@pytest.mark.cudaq
@pytest.mark.parametrize("controls_count", [0, 1, 2])
@pytest.mark.parametrize("time", [-0.37, 0.0, 0.41])
def test_cudaq_pauli_exponential_matches_complex_matrix(
    tmp_path, controls_count: int, time: float
) -> None:
    """Sparse remapping, Y signs, constant phase and all control sectors agree."""
    cudaq = pytest.importorskip("cudaq")
    from qamomile.cudaq._emission import _emit_instruction
    from qamomile.cudaq._integration import _load_source

    width = 3 + controls_count
    wires = (2, 0, 1)
    controls = tuple(range(3, width))
    hamiltonian = 0.23 + 0.7 * qm.X(0) * qm.Y(2) - 0.4 * qm.Y(0) * qm.X(2)
    instruction = CircuitInstruction(
        "pauli_evolution", wires, (hamiltonian, time), controls=controls
    )
    preparation = [f"    q = cudaq.qvector({width})"]
    for index in range(width):
        preparation += [
            f"    ry({0.51 + 0.17 * index}, q[{index}])",
            f"    rz({0.21 + 0.13 * index}, q[{index}])",
        ]
    body = preparation.copy()
    definitions: dict[str, str] = {}
    _emit_instruction(
        instruction, body, indent="    ", parameter_symbols={}, definitions=definitions
    )
    source = "import cudaq\n" + "\n".join(definitions.values())
    source += "\n@cudaq.kernel\ndef probe():\n" + "\n".join(body)
    source += "\n@cudaq.kernel\ndef prepared():\n" + "\n".join(preparation) + "\n"
    path = tmp_path / "pauli_witness.py"
    path.write_text(source)
    scope = runpy.run_path(str(path))
    restored = _load_source(source.replace("def probe():", "def _qamomile_entry():"))
    initial = np.asarray(cudaq.get_state(scope["prepared"])).copy()
    # Independent dense Pauli action in physical little-endian basis order.
    generator = np.zeros((1 << width, 1 << width), dtype=complex)
    for basis in range(1 << width):
        if not all(basis & (1 << control) for control in controls):
            continue
        generator[basis, basis] = 0.23
        for coefficient, x_wire, y_wire in [(0.7, 2, 1), (-0.4, 1, 2)]:
            target = basis ^ (1 << x_wire) ^ (1 << y_wire)
            generator[target, basis] += coefficient * (
                1j if not basis & (1 << y_wire) else -1j
            )
    values, vectors = np.linalg.eigh(generator)
    expected = (vectors * np.exp(-1j * time * values)) @ vectors.conj().T @ initial
    actual = np.asarray(cudaq.get_state(restored))
    # An uncontrolled closed-program phase is intentionally unobservable.
    if not controls:
        expected *= np.exp(1j * time * 0.23)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0)


@pytest.mark.cudaq
@pytest.mark.parametrize("opcode", ["swap", "sdg", "tdg"])
@pytest.mark.parametrize("controls_count", [1, 3])
def test_cudaq_controlled_operators_preserve_complex_amplitudes(
    tmp_path, opcode: str, controls_count: int
) -> None:
    """Retained SDK operators are exact on active and inactive control sectors."""
    cudaq = pytest.importorskip("cudaq")
    from qamomile.cudaq._emission import _emit_controlled_instruction
    from qamomile.cudaq._integration import _load_source

    width = controls_count + 2
    controls = tuple(range(controls_count))
    targets = (
        (controls_count, controls_count + 1) if opcode == "swap" else (controls_count,)
    )
    preparation = [f"    q = cudaq.qvector({width})"]
    for index in range(width):
        preparation += [
            f"    ry({0.3 + 0.19 * index}, q[{index}])",
            f"    rz({0.2 + 0.11 * index}, q[{index}])",
        ]
    body = preparation.copy()
    _emit_controlled_instruction(
        CircuitInstruction(opcode, targets, controls=controls),
        body,
        indent="    ",
        parameter_symbols={},
    )
    assert len(body) == len(preparation) + 1
    source = "import cudaq\n@cudaq.kernel\ndef probe():\n" + "\n".join(body)
    source += "\n@cudaq.kernel\ndef prepared():\n" + "\n".join(preparation) + "\n"
    path = tmp_path / "controlled_witness.py"
    path.write_text(source)
    scope = runpy.run_path(str(path))
    initial = np.asarray(cudaq.get_state(scope["prepared"])).copy()
    expected = initial.copy()
    for basis in range(1 << width):
        if not all(basis & (1 << control) for control in controls):
            continue
        if opcode == "swap":
            destination = basis
            if ((basis >> targets[0]) ^ (basis >> targets[1])) & 1:
                destination ^= (1 << targets[0]) | (1 << targets[1])
            expected[destination] = initial[basis]
        elif basis & (1 << targets[0]):
            expected[basis] *= np.exp(-1j * np.pi / (2 if opcode == "sdg" else 4))
    restored = _load_source(source.replace("def probe():", "def _qamomile_entry():"))
    np.testing.assert_allclose(
        np.asarray(cudaq.get_state(restored)), expected, atol=3e-12, rtol=0
    )


PAULI_INTERFERENCE_SOURCE = """
import qamomile as qm
@qm.qkernel
def evolution(state: qm.Q[qm.Bits[3]], hamiltonian: qm.Const[qm.Observable], time: qm.Param[qm.Real]) -> qm.Q[qm.Bits[3]]:
    return qm.pauli_evolve(state, hamiltonian, time)
@qm.qkernel
def inverse_reference(state: qm.Q[qm.Bits[3]], time: qm.Param[qm.Real]) -> qm.Q[qm.Bits[3]]:
    state = qm.h(state, target=0)
    state = qm.sdg(state, target=2)
    state = qm.h(state, target=2)
    state = qm.cx(state, control=0, target=2)
    state = qm.rz(state, -1.4 * time, target=2)
    state = qm.cx(state, control=0, target=2)
    state = qm.h(state, target=0)
    state = qm.h(state, target=2)
    state = qm.s(state, target=2)
    return qm.global_phase(state, 0.23 * time)
@qm.qkernel
def state(hamiltonian: qm.Const[qm.Observable], time: qm.Param[qm.Real]) -> qm.Q[qm.Bits[5]]:
    whole = qm.h(qm.qubits(5))
    whole = qm.s(whole, target=0)
    whole = qm.t(whole, target=2)
    whole = qm.s(whole, target=4)
    controls, data = qm.split_at(whole, 2)
    controls, data = qm.control(evolution, controls, data, hamiltonian, time, control_state=1)
    controls, data = qm.control(inverse_reference, controls, data, time, control_state=1)
    whole = qm.concat(controls, data)
    whole = qm.sdg(whole, target=4)
    whole = qm.tdg(whole, target=2)
    whole = qm.sdg(whole, target=0)
    return qm.h(whole)
@qm.qkernel
def sample(hamiltonian: qm.Const[qm.Observable], time: qm.Param[qm.Real]) -> qm.Bits[5]:
    return qm.measure(state(hamiltonian, time))
"""


@pytest.mark.cudaq
def test_cudaq_source_artifact_executes_in_fresh_process_without_json(tmp_path) -> None:
    """An independently restored source artifact owns all its helper definitions."""
    import subprocess
    import sys
    from unittest.mock import patch

    cudaq = pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal

    module = qm.build_source(PAULI_INTERFERENCE_SOURCE)
    hamiltonian = 0.23 + 0.7 * qm.X(0) * qm.Y(2)
    program = module.specialize("sample", hamiltonian=hamiltonian)
    with (
        patch.object(
            cudaq.PyKernelDecorator, "to_json", side_effect=AssertionError("JSON used")
        ),
        patch.object(
            cudaq.PyKernelDecorator,
            "from_json",
            side_effect=AssertionError("JSON used"),
        ),
    ):
        artifact = qm.compile(program, to=Cudaq())
        assert serialize(artifact) == serialize(qm.compile(program, to=Cudaq()))
        assert artifact.format == "cudaq.python-source.v1"
        source = export_payload(artifact).decode("utf-8")
        assert source.startswith("import cudaq\n") and "exp_pauli(" in source
        assert "cudaq.control(" in source
        assert "generated_source" not in dict(artifact.metadata.entries)
        assert "realization:pauli.native" in artifact.report.stages
        assert "cudaq-kernel-serialization" not in artifact.report.stages
        restored = deserialize(serialize(artifact))
        endpoint = CudaqLocal()
        assert qm.execute(
            restored, on=endpoint, parameters={"time": 0.41}, shots=4, seed=2
        ).counts == {(0, 0, 0, 0, 0): 4}
        prepared = module.specialize("state", hamiltonian=hamiltonian)
        assert (
            qm.expectation(
                prepared,
                qm.Z(0) + qm.Z(4),
                on=endpoint,
                parameters={"time": -0.37},
                shots=4,
                seed=2,
            ).value
            == 2.0
        )
    artifact_path = tmp_path / "evolution.qart"
    artifact_path.write_bytes(serialize(artifact))
    script = """
import sys
from pathlib import Path
from unittest.mock import patch
import cudaq
import qamomile as qm
from qamomile.cudaq import CudaqLocal
with patch.object(cudaq.PyKernelDecorator, "from_json", side_effect=AssertionError("JSON used")):
    from qamomile.persistence import deserialize
    artifact = deserialize(Path(sys.argv[1]).read_bytes())
    for time in [-0.37, 0.0, 0.41]:
        result = qm.execute(artifact, on=CudaqLocal(), parameters={"time": time}, shots=4, seed=2)
        assert result.counts == {(0, 0, 0, 0, 0): 4}, result.counts
"""
    subprocess.run(
        [sys.executable, "-c", script, str(artifact_path)],
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )


@pytest.mark.cudaq
def test_cudaq_source_loader_rejects_invalid_payloads_and_cleans_inspection_cache() -> (
    None
):
    """Source inspection is temporary and old payloads fail with stable errors."""
    import linecache
    from types import SimpleNamespace

    pytest.importorskip("cudaq")
    from dataclasses import replace

    from qamomile.cudaq import Cudaq, load_kernel
    from qamomile.cudaq._integration import _load_source
    from qamomile.errors import ExecutionError

    before = set(linecache.cache)
    for source in ["def invalid(:", "import cudaq\n", "_qamomile_entry = 1\n"]:
        with pytest.raises(ExecutionError, match="Q8504"):
            _load_source(source)
    assert set(linecache.cache) == before
    program = qm.build_source(
        "import qamomile as qm\n@qm.qkernel\ndef sample() -> qm.Bits[1]:\n    return qm.measure(qm.qubits(1))"
    ).specialize("sample")
    artifact = qm.compile(program, to=Cudaq())
    with pytest.raises(qm.ArtifactError, match="Q7001"):
        load_kernel(replace(artifact, payload=b"\xff"))
    with pytest.raises(ExecutionError, match="Q8503"):
        load_kernel(SimpleNamespace(format="cudaq.kernel-json.v1", payload=b"{}"))
