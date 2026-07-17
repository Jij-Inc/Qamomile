"""Serialization tests for concrete Pauli LCU block encodings."""

from __future__ import annotations

import math
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.qkernel_callable import (
    qkernel_callable_attrs,
    qkernel_callable_ref,
)
from qamomile.circuit.ir.operation.callable import CallPolicy
from qamomile.circuit.serialization import (
    QAMOMILE_VERSION,
    deserialize,
    deserialize_pauli_lcu_block_encoding,
    serialize,
    serialize_pauli_lcu_block_encoding,
)
from qamomile.circuit.serialization.graph_protobuf import _float_bits
from qamomile.circuit.serialization.proto import qamomile_ir_pb2 as pb
from qamomile.linalg import PauliLCU, PauliLCUTerm
from qamomile.observable import Pauli, PauliOperator

I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_ARTIFACT_KIND = "qamomile.pauli_lcu_block_encoding"


@qmc.qkernel
def _identity_block_case(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Return both block-encoding target registers unchanged.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register to preserve.
        system (qmc.Vector[qmc.Qubit]): System register to preserve.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Unchanged
            registers in their original order.
    """
    return signal, system


def _executor(case: Any) -> Any:
    """Return a local simulator executor for one SDK fixture case.

    Args:
        case (Any): Cross-backend fixture carrying a transpiler and name.

    Returns:
        Any: Backend-specific local executor.
    """
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _artifact(payload: bytes) -> pb.PauliLCUBlockEncodingArtifact:
    """Parse one block-encoding recipe for controlled test mutation.

    Args:
        payload (bytes): Canonical recipe bytes.

    Returns:
        pb.PauliLCUBlockEncodingArtifact: Parsed mutable message.
    """
    message = pb.PauliLCUBlockEncodingArtifact()
    message.ParseFromString(payload)
    return message


def _artifact_bytes(message: pb.PauliLCUBlockEncodingArtifact) -> bytes:
    """Encode one test-mutated recipe deterministically.

    Args:
        message (pb.PauliLCUBlockEncodingArtifact): Message to encode.

    Returns:
        bytes: Deterministic protobuf bytes.
    """
    return message.SerializeToString(deterministic=True)


@pytest.mark.parametrize(
    "lcu",
    [
        PauliLCU(1, ()),
        PauliLCU(1, (PauliLCUTerm(-1j, ()),)),
        PauliLCU.from_matrix(1j * I2 + 0.5 * X2),
    ],
    ids=["zero", "single", "multi"],
)
def test_descriptor_round_trip_restores_factory_qkernel(lcu: PauliLCU) -> None:
    """Zero-, single-, and multi-term recipes restore every descriptor field."""
    encoding = qmc.pauli_lcu_block_encoding(lcu)

    payload = serialize_pauli_lcu_block_encoding(encoding)
    restored = deserialize_pauli_lcu_block_encoding(payload)

    assert isinstance(restored, qmc.PauliLCUBlockEncoding)
    assert isinstance(restored.unitary, qmc.QKernel)
    assert restored.normalization == pytest.approx(encoding.normalization)
    assert restored.num_signal_qubits == encoding.num_signal_qubits
    assert restored.num_system_qubits == encoding.num_system_qubits
    assert serialize_pauli_lcu_block_encoding(restored) == payload
    assert deserialize_pauli_lcu_block_encoding(bytearray(payload)).normalization == (
        pytest.approx(encoding.normalization)
    )


def test_descriptor_recipe_is_distinct_from_qkernel_wire_format() -> None:
    """Dedicated recipe bytes cannot be confused with generic qkernel bytes."""
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(1j * I2 + 0.5 * X2))

    recipe_payload = serialize_pauli_lcu_block_encoding(encoding)
    qkernel_payload = serialize(encoding.unitary)
    message = _artifact(recipe_payload)

    assert recipe_payload != qkernel_payload
    assert message.artifact_kind == _ARTIFACT_KIND
    assert message.qamomile_version == QAMOMILE_VERSION
    with pytest.raises(ValueError):
        deserialize(recipe_payload)
    with pytest.raises(ValueError, match="artifact_kind mismatch"):
        deserialize_pauli_lcu_block_encoding(qkernel_payload)


def test_descriptor_recipe_round_trips_in_a_fresh_interpreter() -> None:
    """A separate Python process restores and canonically reserializes a recipe."""
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(1j * I2 + 0.5 * X2))
    payload = serialize_pauli_lcu_block_encoding(encoding)
    script = """
import sys
from qamomile.circuit.serialization import (
    deserialize_pauli_lcu_block_encoding,
    serialize_pauli_lcu_block_encoding,
)
payload = sys.stdin.buffer.read()
encoding = deserialize_pauli_lcu_block_encoding(payload)
sys.stdout.buffer.write(serialize_pauli_lcu_block_encoding(encoding))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        capture_output=True,
        check=True,
    )

    assert completed.stdout == payload
    assert completed.stderr == b""


def test_descriptor_serializer_rejects_non_descriptor_value() -> None:
    """The adapter has a Pauli-specific public input contract."""
    with pytest.raises(TypeError, match="expected a PauliLCUBlockEncoding"):
        serialize_pauli_lcu_block_encoding(object())  # type: ignore[arg-type]


def test_descriptor_deserializer_rejects_non_bytes_value() -> None:
    """The adapter rejects objects outside the protobuf byte contract."""
    with pytest.raises(TypeError, match="expected bytes"):
        deserialize_pauli_lcu_block_encoding("payload")  # type: ignore[arg-type]


def test_descriptor_serializer_rejects_mismatched_public_fields() -> None:
    """Descriptor normalization and widths must agree with the encoded LCU."""
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(I2))
    mismatched = qmc.PauliLCUBlockEncoding(
        unitary=encoding.unitary,
        normalization=2.0,
        num_signal_qubits=encoding.num_signal_qubits,
        num_system_qubits=encoding.num_system_qubits,
    )

    with pytest.raises(ValueError, match="descriptor fields"):
        serialize_pauli_lcu_block_encoding(mismatched)


def test_descriptor_serializer_rejects_forged_factory_metadata() -> None:
    """Factory metadata on a different unitary body is insufficient."""
    expected = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(X2))
    semantic_arguments = qkernel_callable_attrs(expected.unitary)["semantic_arguments"]
    namespace = qkernel_callable_ref(expected.unitary).namespace

    @qmc.qkernel
    def forged_unitary(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Return registers unchanged despite forged Pauli-X metadata.

        Args:
            signal (qmc.Vector[qmc.Qubit]): Signal register to preserve.
            system (qmc.Vector[qmc.Qubit]): System register to preserve.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Unchanged
                registers.
        """
        return signal, system

    configure_composite(
        forged_unitary,
        name="pauli_lcu_block_encoding",
        namespace=namespace,
        policy=CallPolicy.PRESERVE_BOX,
        semantic_arguments=semantic_arguments,
    )
    forged = qmc.PauliLCUBlockEncoding(
        forged_unitary,
        expected.normalization,
        expected.num_signal_qubits,
        expected.num_system_qubits,
    )

    with pytest.raises(ValueError, match="unitary.*canonical factory output"):
        serialize_pauli_lcu_block_encoding(forged)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("artifact_kind", "not-a-block-encoding", "artifact_kind mismatch"),
        ("qamomile_version", "0.0.0", "qamomile_version mismatch"),
    ],
)
def test_descriptor_deserializer_rejects_wrong_envelope_identity(
    field: str,
    replacement: str,
    match: str,
) -> None:
    """Artifact kind and exact installed version are both load-bearing."""
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(I2))
    message = _artifact(serialize_pauli_lcu_block_encoding(encoding))
    setattr(message, field, replacement)

    with pytest.raises(ValueError, match=match):
        deserialize_pauli_lcu_block_encoding(_artifact_bytes(message))


def test_descriptor_deserializer_rejects_missing_required_fields() -> None:
    """Recipe, term, and operator message presence is validated explicitly."""
    lcu = PauliLCU(
        1,
        (PauliLCUTerm(1.0, (PauliOperator(Pauli.X, 0),)),),
    )
    payload = serialize_pauli_lcu_block_encoding(qmc.pauli_lcu_block_encoding(lcu))

    missing_num_qubits = _artifact(payload)
    missing_num_qubits.ClearField("num_qubits")
    with pytest.raises(ValueError, match="missing num_qubits"):
        deserialize_pauli_lcu_block_encoding(_artifact_bytes(missing_num_qubits))

    missing_coefficient = _artifact(payload)
    missing_coefficient.terms[0].ClearField("coefficient")
    with pytest.raises(ValueError, match="has no coefficient"):
        deserialize_pauli_lcu_block_encoding(_artifact_bytes(missing_coefficient))

    missing_index = _artifact(payload)
    missing_index.terms[0].operators[0].ClearField("index")
    with pytest.raises(ValueError, match="has no index"):
        deserialize_pauli_lcu_block_encoding(_artifact_bytes(missing_index))


@pytest.mark.parametrize(
    ("real", "imag", "match"),
    [
        (math.nan, 0.0, "term 0 is invalid"),
        (math.inf, 0.0, "term 0 is invalid"),
        (0.0, 0.0, "term 0 is invalid"),
        (-0.0, 1.0, "not in canonical form"),
    ],
)
def test_descriptor_deserializer_rejects_invalid_coefficient_bits(
    real: float,
    imag: float,
    match: str,
) -> None:
    """Coefficient components remain finite, nonzero, and canonically signed."""
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(I2))
    message = _artifact(serialize_pauli_lcu_block_encoding(encoding))
    message.terms[0].coefficient.real_bits = _float_bits(real)
    message.terms[0].coefficient.imag_bits = _float_bits(imag)

    with pytest.raises(ValueError, match=match):
        deserialize_pauli_lcu_block_encoding(_artifact_bytes(message))


def test_descriptor_deserializer_rejects_duplicate_pauli_words() -> None:
    """A retained decomposition cannot contain duplicate Pauli words."""
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(I2))
    message = _artifact(serialize_pauli_lcu_block_encoding(encoding))
    duplicate = message.terms.add()
    duplicate.CopyFrom(message.terms[0])
    duplicate.coefficient.real_bits = _float_bits(0.5)

    with pytest.raises(ValueError, match="recipe is invalid"):
        deserialize_pauli_lcu_block_encoding(_artifact_bytes(message))


def test_descriptor_deserializer_rejects_noncanonical_operator_order() -> None:
    """Sparse Pauli operators must remain ordered by increasing qubit index."""
    lcu = PauliLCU(
        2,
        (
            PauliLCUTerm(
                1.0,
                (
                    PauliOperator(Pauli.Z, 0),
                    PauliOperator(Pauli.X, 1),
                ),
            ),
        ),
    )
    message = _artifact(
        serialize_pauli_lcu_block_encoding(qmc.pauli_lcu_block_encoding(lcu))
    )
    first = pb.PauliLCUBlockEncodingOperator()
    first.CopyFrom(message.terms[0].operators[0])
    second = pb.PauliLCUBlockEncodingOperator()
    second.CopyFrom(message.terms[0].operators[1])
    message.terms[0].operators[0].CopyFrom(second)
    message.terms[0].operators[1].CopyFrom(first)

    with pytest.raises(ValueError, match="not in canonical form"):
        deserialize_pauli_lcu_block_encoding(_artifact_bytes(message))


def test_descriptor_deserializer_rejects_duplicate_wire_field() -> None:
    """Equivalent duplicate protobuf fields are not a canonical recipe."""
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(I2))
    payload = serialize_pauli_lcu_block_encoding(encoding)
    kind = _ARTIFACT_KIND.encode()
    assert len(kind) < 128
    duplicate_kind = bytes((0x0A, len(kind))) + kind

    with pytest.raises(ValueError, match="not in canonical form"):
        deserialize_pauli_lcu_block_encoding(payload + duplicate_kind)


def test_round_tripped_unitary_retains_call_time_width_validation() -> None:
    """Restoration returns the factory QKernel rather than a static IR shell."""
    encoding = qmc.pauli_lcu_block_encoding(
        PauliLCU.from_matrix(1j * I2 + 0.5 * X2 + 0.25 * np.diag([1.0, -1.0]))
    )
    restored = deserialize_pauli_lcu_block_encoding(
        serialize_pauli_lcu_block_encoding(encoding)
    )

    @qmc.qkernel
    def wrong_width() -> qmc.Bit:
        """Invoke the restored two-signal unitary with one signal qubit."""
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = restored.unitary(signal, system)
        return qmc.measure(signal[0])

    with pytest.raises(
        ValueError,
        match="requires 2 signal qubits, got 1",
    ):
        wrong_width.build()


def test_round_tripped_phase_executes_through_nested_select_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """Every backend observes a restored identity-term phase in outer SELECT."""
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU(1, (PauliLCUTerm(-1.0, ()),)))
    restored = deserialize_pauli_lcu_block_encoding(
        serialize_pauli_lcu_block_encoding(encoding)
    )
    selected = qmc.select(
        (_identity_block_case, restored.unitary),
        num_index_qubits=1,
    )

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        """Interfere an identity case with the restored negative identity."""
        selector = qmc.qubit_array(1, "selector")
        selector[0] = qmc.h(selector[0])
        signal = qmc.qubit_array(restored.num_signal_qubits, "signal")
        system = qmc.qubit_array(restored.num_system_qubits, "system")
        selector, _, _ = selected(selector, signal, system)
        selector[0] = qmc.h(selector[0])
        return qmc.measure(selector[0])

    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=64).result()
    assert result.results == [(1, 64)]


def test_round_tripped_multi_term_inverse_control_executes_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """Controlled multi-term factory output cancels its inverse everywhere."""
    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(1j * I2 + 0.5 * X2))
    restored = deserialize_pauli_lcu_block_encoding(
        serialize_pauli_lcu_block_encoding(encoding)
    )

    @qmc.qkernel
    def inverse_unitary(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the restored inverse through a controllable qkernel ABI.

        Args:
            signal (qmc.Vector[qmc.Qubit]): Signal register to transform.
            system (qmc.Vector[qmc.Qubit]): System register to transform.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Registers
                after the inverse block-encoding unitary.
        """
        return qmc.inverse(restored.unitary)(signal, system)

    controlled = qmc.control(restored.unitary)
    controlled_inverse = qmc.control(inverse_unitary)

    @qmc.qkernel
    def circuit() -> tuple[qmc.Bit, qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Apply a controlled restored unitary followed by its controlled inverse."""
        outer = qmc.x(qmc.qubit("outer"))
        signal = qmc.qubit_array(restored.num_signal_qubits, "signal")
        system = qmc.qubit_array(restored.num_system_qubits, "system")
        system[0] = qmc.x(system[0])
        outer, signal, system = controlled(outer, signal, system)
        outer, signal, system = controlled_inverse(outer, signal, system)
        return qmc.measure(outer), qmc.measure(signal), qmc.measure(system)

    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=32).result()
    assert result.results == [((1, (0,), (1,)), 32)]


def test_restored_unitary_supports_double_inverse_identity() -> None:
    """Fresh factory output retains the inverse involution fast path."""
    encoding = qmc.pauli_lcu_block_encoding(
        PauliLCU(1, (PauliLCUTerm(-1.0, (PauliOperator(Pauli.X, 0),)),))
    )
    restored = deserialize_pauli_lcu_block_encoding(
        serialize_pauli_lcu_block_encoding(encoding)
    )

    assert qmc.inverse(qmc.inverse(restored.unitary)) is restored.unitary
