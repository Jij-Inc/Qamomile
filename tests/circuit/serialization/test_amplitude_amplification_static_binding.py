"""Tests for serialized amplification-preparation binding slots."""

from __future__ import annotations

import math
from typing import Any

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.block import BlockKind
from qamomile.circuit.serialization import deserialize, serialize

_SCHEDULE = qmc.standard_amplification_schedule(1)


@qmc.qkernel
def _uniform_system_preparation(
    system: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a uniform superposition on the system register.

    Args:
        system (qmc.Vector[qmc.Qubit]): All-zero system register.

    Returns:
        qmc.Vector[qmc.Qubit]: System register in a uniform superposition.
    """
    return qmc.h(system)


@qmc.qkernel
def _preparation_template(
    target: qmc.AmplificationPreparation,
    observable: qmc.Observable,
) -> qmc.Float:
    """Amplify a statically bound preparation and measure its success.

    Args:
        target (qmc.AmplificationPreparation): Static preparation slot.
        observable (qmc.Observable): All-zero signal projector.

    Returns:
        qmc.Float: Expected value of the all-zero signal projector.
    """
    signal = qmc.qubit_array(target.num_signal_qubits, "signal")
    system = qmc.qubit_array(target.num_system_qubits, "system")
    signal, _ = qmc.amplitude_amplification(signal, system, target, _SCHEDULE)
    return qmc.expval(signal, observable)


@qmc.qkernel
def _preparation_sample_template(
    target: qmc.AmplificationPreparation,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Amplify a statically bound preparation and measure both registers.

    Args:
        target (qmc.AmplificationPreparation): Static preparation slot.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured signal and
            system registers.
    """
    signal = qmc.qubit_array(target.num_signal_qubits, "signal")
    system = qmc.qubit_array(target.num_system_qubits, "system")
    signal, system = qmc.amplitude_amplification(signal, system, target, _SCHEDULE)
    return qmc.measure(signal), qmc.measure(system)


def _encoding(single: float, num_system_qubits: int = 1) -> Any:
    """Build a small diagonal block encoding.

    Args:
        single (float): Coefficient of the ``Z_0`` word.
        num_system_qubits (int): System register width. Defaults to ``1``.

    Returns:
        Any: Two-term diagonal block encoding.
    """
    return qmc.ising_z_block_encoding(
        {(): 1.0 + 0.0j, (0,): complex(single)},
        num_system_qubits=num_system_qubits,
    )


def _zero_projector(num_qubits: int) -> qm_o.Hamiltonian:
    """Return the projector onto an all-zero register.

    Args:
        num_qubits (int): Positive register width.

    Returns:
        qm_o.Hamiltonian: Product of ``(I + Z_i) / 2`` over every qubit.
    """
    projector = qm_o.Hamiltonian.identity(num_qubits=num_qubits)
    identity = qm_o.Hamiltonian.identity(num_qubits=num_qubits)
    for index in range(num_qubits):
        projector = projector * (0.5 * (identity + qm_o.Z(index)))
    return projector


def test_unbound_template_uses_only_a_typed_static_manifest_slot() -> None:
    """The preparation parameter is absent from the ordinary value ABI."""
    block = _preparation_sample_template.block

    assert block.kind is BlockKind.HIERARCHICAL
    assert block.label_args == []
    assert block.input_values == []
    assert block.param_slots == ()
    assert len(block.static_bindings) == 1
    slot = block.static_bindings[0]
    assert slot.name == "target"
    assert slot.type_key == "qamomile.stdlib.amplification_preparation"
    assert [field.name for field in slot.fields] == [
        "num_signal_qubits",
        "num_system_qubits",
    ]


def test_static_manifest_serialization_is_deterministic() -> None:
    """The unbound template round-trips without a concrete descriptor."""
    payload = serialize(_preparation_sample_template)
    restored = deserialize(payload)

    assert serialize(restored) == payload
    assert list(restored.signature.parameters) == ["target"]
    assert restored.input_types == {"target": qmc.AmplificationPreparation}
    assert restored.block.label_args == []
    assert len(restored.block.static_bindings) == 1


def test_round_tripped_template_binds_a_composed_preparation(
    qiskit_transpiler: Any,
) -> None:
    """A serialized template amplifies a preparation it never saw."""
    encoding = _encoding(-0.3)
    prepared = qmc.amplification_preparation(encoding, _uniform_system_preparation)
    restored = deserialize(serialize(_preparation_template))

    executable = qiskit_transpiler.transpile(
        restored,
        bindings={
            "target": prepared,
            "observable": _zero_projector(prepared.num_signal_qubits),
        },
    )
    observed = float(executable.run(qiskit_transpiler.executor()).result())

    bare = qiskit_transpiler.transpile(
        _bare_preparation_kernel(prepared),
        bindings={"observable": _zero_projector(prepared.num_signal_qubits)},
    )
    initial = float(bare.run(qiskit_transpiler.executor()).result())
    angle = math.asin(math.sqrt(initial))

    assert executable.quantum_circuit.num_qubits == (
        prepared.num_signal_qubits + prepared.num_system_qubits + 1
    )
    assert observed == pytest.approx(math.sin(3 * angle) ** 2, abs=1e-8)


def _bare_preparation_kernel(prepared: Any) -> qmc.QKernel:
    """Build a kernel measuring the unamplified success probability.

    Args:
        prepared (Any): Preparation descriptor to apply once.

    Returns:
        qmc.QKernel: Kernel returning the all-zero-signal probability.
    """

    @qmc.qkernel
    def kernel(observable: qmc.Observable) -> qmc.Float:
        """Apply the bare preparation and estimate its success probability.

        Args:
            observable (qmc.Observable): All-zero signal projector.

        Returns:
            qmc.Float: Expected value of the all-zero signal projector.
        """
        signal = qmc.qubit_array(prepared.num_signal_qubits, "signal")
        system = qmc.qubit_array(prepared.num_system_qubits, "system")
        signal, _ = prepared.preparation(signal, system)
        return qmc.expval(signal, observable)

    return kernel


def test_one_template_binds_differently_sized_preparations(
    qiskit_transpiler: Any,
) -> None:
    """One serialized template serves preparations of different widths."""
    payload = serialize(_preparation_sample_template)
    restored = deserialize(payload)

    widths = []
    for num_system_qubits in (1, 2, 3):
        prepared = qmc.amplification_preparation(_encoding(-0.4, num_system_qubits))
        executable = qiskit_transpiler.transpile(
            restored,
            bindings={"target": prepared},
        )
        widths.append(executable.quantum_circuit.num_qubits)

    assert widths == [3, 4, 5]
    assert serialize(restored) == payload


def test_binding_rejects_a_foreign_descriptor(qiskit_transpiler: Any) -> None:
    """A block encoding cannot be bound into a preparation slot."""
    restored = deserialize(serialize(_preparation_sample_template))

    with pytest.raises(TypeError, match="AmplificationPreparation"):
        qiskit_transpiler.transpile(
            restored,
            bindings={"target": _encoding(-0.4)},
        )
