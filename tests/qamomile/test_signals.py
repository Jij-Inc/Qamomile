"""Selected-process contracts, phase semantics, and shared SDK witnesses."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.stdlib import BlockEncoding, Region, Signal, select, turn
from qamomile.stdlib._structure import identity

from ._programs import (
    signal_fixed_point_sample,
    signal_fixed_point_state,
    signal_mask_sample,
    signal_mask_state,
    signal_prepare,
    signal_scalar_phase_sample,
    signal_search_process,
    signal_search_sample,
    signal_search_state,
)


def exercise_signals_backend(*, endpoint: object) -> None:
    """Compile and execute sampling, complex-phase, and expectation witnesses."""
    from .test_signal_algebra import exercise_signal_algebra_backend

    exercise_signal_algebra_backend(endpoint=endpoint)
    for source, launches in (
        (signal_search_sample, (({"phases": ((math.pi, math.pi),)}, (1, 1)),)),
        (signal_fixed_point_sample, (({}, (0, 0)),)),
        (
            signal_mask_sample,
            (({"angle": math.pi}, (1, 1, 1, 1)), ({"angle": 0.0}, (1, 1, 0, 1))),
        ),
        (
            signal_scalar_phase_sample,
            (({"angle": math.pi}, (1, 0)), ({"angle": 0.0}, (0, 0))),
        ),
    ):
        artifact = qm.compile(source, to=endpoint.target)
        assert isinstance(artifact, qm.CompiledProgram)
        for parameters, expected in launches:
            result = qm.execute(
                artifact, on=endpoint, parameters=parameters, shots=4, seed=0
            )
            assert result.counts == {expected: 4}
    assert (
        qm.expectation(
            signal_mask_state,
            qm.Z(2),
            on=endpoint,
            parameters={"angle": math.pi},
            shots=4,
            seed=0,
        ).value
        == -1.0
    )
    assert (
        qm.expectation(
            signal_fixed_point_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
            seed=0,
        ).value
        == 1.0
    )


def test_signal_reference_execution() -> None:
    """The same shared source executes on the native reference endpoint."""
    exercise_signals_backend(endpoint=qm.Reference())


def test_regions_are_native_checked_immutable_embeddings() -> None:
    """Patterns retain source order and distinguish full and rank-one spaces."""
    region = Region("1*00*")
    assert region.width == 5
    assert region.payload_indices == (1, 4)
    assert region == Region("1*00*")
    assert Region("***").payload_indices == (0, 1, 2)
    assert Region("010").payload_indices == ()
    assert Region("*" * native.MAX_SIGNAL_REGION_WIDTH).width == 4096
    with pytest.raises(FrozenInstanceError):
        region.pattern = "00000"
    with pytest.raises(TypeError, match="string"):
        Region(3)
    for invalid in ("", "*" * (native.MAX_SIGNAL_REGION_WIDTH + 1)):
        with pytest.raises(ValueError, match="Q7730"):
            Region(invalid)
    for invalid in ("x", "0 1", "é"):
        with pytest.raises(ValueError, match="Q7731"):
            Region(invalid)


def test_region_native_boundary_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent or malformed native producer never selects a Python planner."""
    monkeypatch.setattr(native, "signal_region", None)
    with pytest.raises(TypeError, match="Q7733"):
        Region("0")
    monkeypatch.setattr(native, "signal_region", lambda _: SimpleNamespace(pattern="0"))
    with pytest.raises(TypeError, match="Q7734"):
        Region("0")


def test_signal_selection_preserves_param_abi_and_defines_a_new_target() -> None:
    """A parameterized process remains a regular callable, not a bound circuit."""
    signal = select(signal_search_process, input=Region("00"), output=Region("11"))
    assert signal.process is signal_search_process
    assert signal.ambient_width == 2
    assert signal.normalization == 1.0
    assert signal.error_bound == 0.0
    assert "not established" in signal.inspect()
    assert "qcore.for" in signal.process.module().inspect()
    block = BlockEncoding(identity, 2.0, 1, 1, error_bound=0.01)
    imported = Signal.from_block_encoding(block)
    assert imported.input_region == imported.output_region == Region("0*")
    assert (imported.normalization, imported.error_bound) == (2.0, 0.01)
    fresh = select(imported.process, input=Region("0*"), output=Region("0*"))
    assert (fresh.normalization, fresh.error_bound) == (1.0, 0.0)
    with pytest.raises(TypeError, match="use select"):
        Signal()


@qm.qkernel
def _nonunitary(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """A channel cannot become a coherent selected process."""
    return qm.reset(state)


@qm.qkernel
def _shot_parameter(
    state: qm.Q[qm.Bits[2]], angle: qm.Shot[qm.Real]
) -> qm.Q[qm.Bits[2]]:
    """Shot arguments cannot be treated as a fixed coherent launch recipe."""
    return qm.rz(state, angle, target=0)


@qm.qkernel
def _nested_shot_parameter(
    state: qm.Q[qm.Bits[2]], angles: tuple[qm.Real, qm.Shot[qm.Real]]
) -> qm.Q[qm.Bits[2]]:
    """The launch-stage restriction applies recursively inside products."""
    return qm.rz(state, angles[1], target=0)


@qm.qkernel
def _two_owners(
    left: qm.Q[qm.Bits[2]], right: qm.Q[qm.Bits[2]]
) -> tuple[qm.Q[qm.Bits[2]], qm.Q[qm.Bits[2]]]:
    """Two coherent inputs do not match the selected single-register ABI."""
    return left, right


@pytest.mark.parametrize(
    "process", (_nonunitary, _shot_parameter, _nested_shot_parameter, _two_owners)
)
def test_signal_rejects_unsupported_process_contracts(process: qm.QKernel) -> None:
    """Native effects and recursive type equations guard the selection ABI."""
    with pytest.raises(TypeError, match="Q7735"):
        select(process, input=Region("00"), output=Region("11"))


def test_signal_rejects_mismatched_shapes() -> None:
    """Matching widths and an owned bit-register ABI are required."""
    with pytest.raises(ValueError, match="same ambient"):
        select(signal_prepare, input=Region("0"), output=Region("11"))
    with pytest.raises(TypeError, match="Q7735"):
        select(signal_prepare, input=Region("0"), output=Region("1"))
    with pytest.raises(TypeError, match="Region"):
        turn("00")


def test_signal_persistence_and_resource_identity() -> None:
    """Only ordinary QCore crosses persistence; phase recipes add no opcode."""
    program = signal_fixed_point_sample.specialize()
    payload = program.to_bytes()
    restored = qm.Program.from_bytes(payload)
    assert restored.to_bytes() == payload
    assert qm.evaluate(restored) == (0, 0)
    assert program.to_bytes() == signal_fixed_point_sample.specialize().to_bytes()
    text = program.inspect()
    assert "quantum.signal" not in text
    assert "quantum.region" not in text
    study = qm.resources.logical(signal_search_process)
    assert study.metric(qm.resources.LOGICAL_INPUT_QUBITS).numeric_enclosure() == (2, 2)
    assert study.unresolved()
    with pytest.raises(ValueError, match="unknown contributors"):
        qm.resources.logical(signal_search_process, strict=True)


@pytest.mark.parametrize(
    "pattern", ("*", "0", "1", "**", "00", "01", "10", "11", "1*0*", "0*1*0")
)
def test_turn_matches_complete_complex_diagonal(pattern: str) -> None:
    """All basis columns expose wrong bit order, routing, and scalar phase."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import Qiskit, to_circuit

    region = Region(pattern)
    rotate = turn(region)
    width = region.width
    angle = 0.73

    @qm.qkernel
    def state() -> qm.Q[qm.Bits[width]]:
        return rotate(qm.qubits(width), angle)

    actual = Operator(to_circuit(state, target=Qiskit())).data
    expected = np.diag(
        [
            np.exp(1j * angle)
            if all(
                char == "*" or int(char) == ((index >> lane) & 1)
                for lane, char in enumerate(pattern)
            )
            else 1.0
            for index in range(1 << width)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def test_same_phase_program_has_grover_and_fixed_point_responses() -> None:
    """Circuit inspection preserves different parameterized success probabilities."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import to_circuit

    for angle, expected in ((math.pi, 1.0), (math.pi / 3, 37 / 64)):
        circuit = to_circuit(
            signal_search_state, parameters={"phases": ((angle, angle),)}
        )
        state = Statevector.from_instruction(circuit).data
        assert abs(state[3]) ** 2 == pytest.approx(expected, abs=1e-13)
