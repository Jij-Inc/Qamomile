"""Fixed keyed inputs reuse the same native Param Vector ABI."""

from __future__ import annotations

import math

import pytest

import qamomile as qm
from qamomile.parameters import ParameterMap
from qamomile.persistence import serialize

SOURCE = """
import qamomile as qm
@qm.qkernel
def sample(
    keys: qm.Const[qm.Map[tuple[qm.Int, qm.Int], qm.Int, 2]],
    angles: qm.Param[qm.Vector[qm.Real, 2]],
) -> qm.Shot[qm.Bits[2]]:
    state = qm.prepare((1, 0))
    return qm.measure(qm.rx(state, angles[keys[(3, 7)]], target=1))
"""


def exercise_parameter_map_backend(endpoint, *, seed=7):
    """Rebind a reversed input dictionary three times without recompiling."""
    layout = ParameterMap(((8, 2), (3, 7)))
    program = qm.build_source(SOURCE).specialize("sample", keys=layout.indices)
    restored = qm.Program.from_bytes(program.to_bytes())
    artifact = qm.compile(restored, to=endpoint.target)
    before = serialize(artifact)
    for angle, expected in ((0.0, (1, 0)), (math.pi, (1, 1)), (0.0, (1, 0))):
        parameters = {"angles": layout.pack({(3, 7): angle, (8, 2): 0.75})}
        assert qm.execute(
            artifact, on=endpoint, parameters=parameters, shots=8, seed=seed
        ).counts == {expected: 8}
        assert serialize(artifact) == before

    state_source = SOURCE.replace("qm.Shot[qm.Bits[2]]", "qm.Q[qm.Bits[2]]").replace(
        "return qm.measure(qm.rx(state, angles[keys[(3, 7)]], target=1))",
        "return qm.rx(state, angles[keys[(3, 7)]], target=1)",
    )
    state = qm.build_source(state_source).specialize("sample", keys=layout.indices)
    assert (
        qm.expectation(
            state,
            qm.Z(1),
            on=endpoint,
            parameters={"angles": layout.pack({(3, 7): math.pi, (8, 2): 0.75})},
            shots=8,
            seed=seed,
        ).value
        == -1.0
    )


def test_parameter_map_reuses_reference_artifact():
    """Reference shares the same fixed-key input witness as capable SDKs."""
    exercise_parameter_map_backend(qm.Reference())


def test_parameter_map_order_and_immutability():
    """Declared key order is stable while every packed value tuple is fresh."""
    layout = ParameterMap(((3, 7), (8, 2)))
    assert dict(layout.indices) == {(3, 7): 0, (8, 2): 1}
    assert layout.pack({(8, 2): 4.0, (3, 7): 2.0}) == (2.0, 4.0)
    with pytest.raises(TypeError):
        layout.indices[(3, 7)] = 1
    with pytest.raises(AttributeError):
        layout.keys = ((8, 2),)


@pytest.mark.parametrize(
    "keys,error",
    [((), ValueError), ((1, 1), ValueError), ([1, 2], TypeError), (([],), TypeError)],
)
def test_parameter_map_rejects_invalid_layout(keys, error):
    """Invalid fixed-key layouts fail before compiler calls."""
    with pytest.raises(error):
        ParameterMap(keys)


@pytest.mark.parametrize(
    "values,error",
    [
        ({1: 1.0}, ValueError),
        ({1: 1.0, 3: 2.0}, ValueError),
        ({1: 1.0, 2: 2.0, 3: 3.0}, ValueError),
        ([1, 2], TypeError),
    ],
)
def test_parameter_map_rejects_missing_and_extra_keys(values, error):
    """Packing cannot silently ignore or invent dictionary keys."""
    with pytest.raises(error):
        ParameterMap((1, 2)).pack(values)


def test_parameter_map_native_abi_still_checks_value_types():
    """Packing leaves value type validation with the native parameter ABI."""
    layout = ParameterMap(((8, 2), (3, 7)))
    program = qm.build_source(SOURCE).specialize("sample", keys=layout.indices)
    with pytest.raises(qm.ExecutionError):
        qm.run(
            program,
            on=qm.Reference(),
            parameters={"angles": layout.pack({(8, 2): 0.0, (3, 7): "bad"})},
            shots=1,
        )


def test_parameter_map_checks_node_budget_before_index_allocation(monkeypatch):
    """The host packing layout is bounded before allocating its lookup table."""
    from qamomile._native import compiler as native

    monkeypatch.setattr(native, "MAX_QCORE_MESSAGE_NODES", 2)
    with pytest.raises(ValueError, match="node budget"):
        ParameterMap((0, 1, 2))
