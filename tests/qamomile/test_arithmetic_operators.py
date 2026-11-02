"""First-class coherent arithmetic source and execution witnesses."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile as qm
from qamomile.persistence import serialize

from ._programs import (
    first_class_comparison_sample,
    first_class_constant_add_sample,
    first_class_mod_add_constant_sample,
    first_class_mod_add_invalid_sample,
    first_class_mod_add_sample,
    first_class_mod_mul_controlled,
    first_class_mod_mul_invalid_sample,
    first_class_mod_mul_roundtrip,
    first_class_mod_mul_sample,
    first_class_mod_mul_state,
    first_class_mod_pow_sample,
    first_class_mul_add_sample,
    stdlib_arithmetic_ux_sample,
)

_ARITHMETIC_CASES = (
    (first_class_constant_add_sample, (1, 0, 1)),
    (first_class_mod_add_sample, (0, 0, 0, 1, 1, 1)),
    (first_class_mod_add_invalid_sample, (1, 0, 1, 1, 1, 1)),
    (first_class_mod_add_constant_sample, (1, 0)),
    (first_class_mul_add_sample, (1, 1, 1, 0, 0, 1, 1, 1)),
    (first_class_mod_mul_sample, (0, 0, 1)),
    (first_class_mod_mul_invalid_sample, (0, 1, 1)),
    (first_class_comparison_sample, (1, 1, 0, 0, 1)),
    (stdlib_arithmetic_ux_sample, (1, 1, 1, 0, 1, 1, 0, 0, 1, 1)),
)


@pytest.mark.parametrize(
    ("program", "opcode"),
    [
        (first_class_constant_add_sample, "quantum.add_constant"),
        (first_class_mod_add_sample, "quantum.mod_add"),
        (first_class_mod_add_constant_sample, "quantum.mod_add_constant"),
        (first_class_mul_add_sample, "quantum.mul_add"),
        (first_class_mod_mul_sample, "quantum.mod_mul"),
        (first_class_mod_pow_sample, "quantum.mod_pow"),
        (first_class_comparison_sample, "quantum.less_than_xor"),
    ],
)
def test_qcore_retains_arithmetic_meaning(program: qm.QKernel, opcode: str) -> None:
    """Each source operation survives as one verified semantic opcode."""

    assert opcode in program.specialize().inspect()


@pytest.mark.parametrize(
    ("program", "expected"),
    _ARITHMETIC_CASES,
)
def test_first_class_arithmetic_reference(
    program: qm.QKernel, expected: tuple[int, ...]
) -> None:
    """Each arithmetic family executes its complete-basis permutation on Reference."""

    artifact = qm.compile(program, to=qm.Reference().target)
    assert b"quantum." in serialize(artifact)
    assert qm.execute(artifact, on=qm.Reference(), shots=4, seed=0).counts == {
        expected: 4
    }


def test_arithmetic_control_adjoint_and_persistence() -> None:
    """A modular multiplier remains exact under transforms and storage."""

    roundtrip = first_class_mod_mul_roundtrip.specialize()
    assert (
        qm.Program.from_bytes(roundtrip.to_bytes()).to_bytes() == roundtrip.to_bytes()
    )
    assert qm.run(roundtrip, on=qm.Reference(), shots=4, seed=0).result.counts == {
        (0, 1, 0): 4
    }
    assert qm.run(
        first_class_mod_mul_controlled, on=qm.Reference(), shots=4, seed=0
    ).result.counts == {(1, 0, 0, 1): 4}


@pytest.mark.parametrize(
    ("expression", "diagnostic"),
    [
        ("qm.mod_mul(target, factor=2, modulus=6)", "Q8050"),
        ("qm.mod_mul(target, factor=2, modulus=9)", "Q8050"),
    ],
)
def test_invalid_modular_parameters_fail_closed(
    expression: str, diagnostic: str
) -> None:
    """Reject noninvertible factors and moduli outside the target width."""

    source = f"""
import qamomile as qm
@qm.qkernel
def invalid() -> qm.Bits[3]:
    target = qm.as_uint(qm.prepare((0, 1, 0)))
    return qm.measure(qm.as_bits({expression}))
"""
    program = qm.build_source(source).specialize("invalid")
    with pytest.raises(qm.ExecutionError, match=diagnostic):
        qm.run(program, on=qm.Reference(), shots=4, seed=0)
    pytest.importorskip("qiskit")
    from qamomile.qiskit import Qiskit

    with pytest.raises(qm.CapabilityError, match=diagnostic):
        qm.compile(program, to=Qiskit())


def test_oversized_modular_layout_fails_before_expansion() -> None:
    """Reject unsupported modulus width without allocating a statevector."""

    pytest.importorskip("qiskit")
    from qamomile.qiskit import Qiskit

    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def oversized() -> qm.Bits[12]:
    target = qm.as_uint(qm.qubits(4))
    addend = qm.as_uint(qm.qubits(4))
    modulus = qm.as_uint(qm.qubits(4))
    target, addend, modulus = qm.mod_add(target, addend, modulus)
    return qm.measure(qm.concat(qm.as_bits(target), qm.as_bits(addend), qm.as_bits(modulus)))
""").specialize("oversized")
    with pytest.raises(qm.CapabilityError, match="Q8051"):
        qm.compile(program, to=Qiskit())


def test_modular_multiplier_has_the_exact_full_unitary() -> None:
    """Check every basis column and complex phase on the SDK circuit."""

    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import Qiskit, to_circuit

    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def multiply() -> qm.Q[qm.UInt[3]]:
    state = qm.as_uint(qm.qubits(3))
    return qm.mod_mul(state, factor=2, modulus=5)
""").specialize("multiply")
    circuit = to_circuit(program, target=Qiskit())
    actual = Operator(circuit).data
    expected = np.zeros((8, 8), dtype=complex)
    for value in range(8):
        image = (2 * value) % 5 if value < 5 else value
        expected[image, value] = 1
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("backend", ["qiskit", "cirq", "quri_parts", "braket", "cudaq"])
def test_first_class_arithmetic_on_every_local_backend(backend: str) -> None:
    """Execute the same arithmetic source through every available local SDK."""

    modules = {
        "qiskit": ("qiskit_aer", "Aer"),
        "cirq": ("cirq", "CirqSimulator"),
        "quri_parts": ("quri_parts", "Qulacs"),
        "braket": ("braket", "BraketLocal"),
        "cudaq": ("cudaq", "CudaqLocal"),
    }
    sdk, endpoint_name = modules[backend]
    pytest.importorskip(sdk)
    target_module = __import__(f"qamomile.{backend}", fromlist=[endpoint_name])
    endpoint = getattr(target_module, endpoint_name)()
    for program, expected in _ARITHMETIC_CASES:
        artifact = qm.compile(program, to=endpoint.target)
        assert isinstance(artifact, qm.CompiledProgram)
        assert qm.execute(artifact, on=endpoint, shots=4, seed=0).counts == {
            expected: 4
        }
    assert (
        qm.expectation(
            first_class_mod_mul_state, qm.Z(2), on=endpoint, shots=4, seed=0
        ).value
        == -1.0
    )


def test_first_class_arithmetic_compiles_to_hugr() -> None:
    """HUGR projects the Rust schedule for each arithmetic family."""

    pytest.importorskip("hugr")
    pytest.importorskip("tket_exts")
    from qamomile.hugr import Hugr

    for program, _ in _ARITHMETIC_CASES:
        artifact = qm.compile(program, to=Hugr())
        assert isinstance(artifact, qm.CompiledProgram)
        assert serialize(artifact)
