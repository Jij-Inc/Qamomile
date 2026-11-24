"""Integer inputs obey the declared Real semantics at both input stages."""

from __future__ import annotations

import pytest

import qamomile as qm

_ROTATION_SOURCE = """
import qamomile as qm

@qm.qkernel
def encode(
    state: qm.Q[qm.Bits[1]],
    values: qm.Param[qm.Vector[qm.Real, 2]],
) -> qm.Q[qm.Bits[1]]:
    state = qm.h(state)
    state = qm.ry(state, values[0])
    state = qm.rz(state, values[1])
    return state

@qm.qkernel
def overlap(
    left: qm.INPUT_STAGE[qm.Vector[qm.Real, 2]],
    right: qm.INPUT_STAGE[qm.Vector[qm.Real, 2]],
) -> qm.Shot[qm.Bits[1]]:
    state = encode(qm.qubits(1), left)
    state = qm.adjoint(encode, state, right)
    return qm.measure(state)
"""


@pytest.mark.parametrize("stage", ["Param", "Const"])
@pytest.mark.parametrize("values", [(1, -2), (1.0, -2.0)])
def test_real_integer_input_executes_forward_and_adjoint(
    stage: str, values: tuple[int | float, int | float]
) -> None:
    """The same accepted values cancel through the shared circuit lowering."""

    module = qm.build_source(_ROTATION_SOURCE.replace("INPUT_STAGE", stage))
    bindings = {"left": values, "right": values}
    program = module.specialize("overlap", **(bindings if stage == "Const" else {}))
    artifact = qm.compile(program, to=qm.Reference())
    result = qm.execute(
        artifact,
        on=qm.Reference(),
        parameters=bindings if stage == "Param" else {},
        shots=8,
        seed=17,
    )

    assert result.counts == {(0,): 8}


@pytest.mark.parametrize("stage", ["Param", "Const"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(2**53 + 1, id="inexact-positive"),
        pytest.param(-(2**53 + 1), id="inexact-negative"),
        pytest.param(1 << 100, id="bigint"),
        pytest.param(int(float.fromhex("0x1.fffffffffffffp+1023")), id="largest-real"),
    ],
)
def test_real_integer_input_rounds_before_arithmetic(stage: str, value: int) -> None:
    """Real inputs round once before arithmetic, including BigInt inputs."""

    module = qm.build_source(
        f"""
import qamomile as qm

@qm.qkernel
def difference(value: qm.{stage}[qm.Real]) -> qm.Real:
    return (value + 1) - value
"""
    )
    results = []
    for supplied in (value, float(value)):
        bindings = {"value": supplied}
        program = module.specialize(
            "difference", **(bindings if stage == "Const" else {})
        )
        results.append(qm.evaluate(program, **(bindings if stage == "Param" else {})))
        if stage == "Const":
            restored = qm.Program.from_bytes(program.to_bytes())
            assert restored.to_bytes() == program.to_bytes()
            assert qm.evaluate(restored) == results[-1]

    expected = (float(value) + 1) - float(value)
    assert results == [expected, expected]
    assert all(type(result) is float for result in results)


@pytest.mark.parametrize("stage", ["Param", "Const"])
def test_real_integer_input_preserves_other_nested_scalar_types(stage: str) -> None:
    """Only Real leaves are converted inside products and multidimensional arrays."""

    module = qm.build_source(
        f"""
import qamomile as qm

@qm.qkernel
def identity(
    value: qm.{stage}[tuple[qm.Bool, qm.Int, qm.Matrix[qm.Real, 2, 2]]],
) -> tuple[qm.Bool, qm.Int, qm.Matrix[qm.Real, 2, 2]]:
    return value
"""
    )
    value = (True, (1 << 100) + 1, ((0, -2), (3, 2**53 + 1)))
    bindings = {"value": value}
    program = module.specialize("identity", **(bindings if stage == "Const" else {}))
    result = qm.evaluate(program, **(bindings if stage == "Param" else {}))

    assert result == (True, (1 << 100) + 1, ((0.0, -2.0), (3.0, float(2**53 + 1))))
    assert isinstance(result, tuple)
    assert type(result[0]) is bool
    assert type(result[1]) is int
    assert all(type(item) is float for row in result[2] for item in row)


@pytest.mark.parametrize("stage", ["Param", "Const"])
@pytest.mark.parametrize("invalid", [(True, 1), ("0", 1), (1j, 1), (1,), ((0, 1), 2)])
def test_real_integer_input_does_not_relax_type_or_shape_checks(
    stage: str, invalid: object
) -> None:
    """Numeric coercion does not accept Boolean, non-real, or malformed vectors."""

    module = qm.build_source(_ROTATION_SOURCE.replace("INPUT_STAGE", stage))
    bindings = {"left": invalid, "right": (0, 1)}
    if stage == "Const":
        with pytest.raises(qm.SpecializationError, match="Q3006"):
            module.specialize("overlap", **bindings)
    else:
        artifact = qm.compile(module.specialize("overlap"), to=qm.Reference())
        with pytest.raises(qm.ExecutionError, match="Q8109"):
            artifact.parameter_abi.bind(bindings)


@pytest.mark.parametrize("stage", ["Param", "Const"])
@pytest.mark.parametrize(
    "value", [1 << 1024, -(1 << 1024)], ids=["positive", "negative"]
)
def test_real_integer_input_rejects_binary64_overflow(stage: str, value: int) -> None:
    """An integer outside finite Real range fails at the checked input boundary."""

    module = qm.build_source(
        f"""
import qamomile as qm

@qm.qkernel
def identity(value: qm.{stage}[qm.Real]) -> qm.Real:
    return value
"""
    )
    reason = "integer is too large to convert to binary64"
    if stage == "Const":
        with pytest.raises(qm.SpecializationError, match=reason):
            module.specialize("identity", value=value)
    else:
        artifact = qm.compile(module.specialize("identity"), to=qm.Reference())
        with pytest.raises(qm.ExecutionError, match=reason):
            artifact.parameter_abi.bind({"value": value})


def test_integer_real_map_keys_reject_collisions_after_rounding() -> None:
    """Different integer keys cannot silently alias in a declared Real map."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def lookup(values: qm.Const[qm.Map[qm.Real, qm.Int, 2]]) -> qm.Int:
    return values[9007199254740992.0]
"""
    )

    with pytest.raises(qm.SpecializationError, match="Q3007.*duplicate key"):
        module.specialize("lookup", values={2**53: 1, 2**53 + 1: 2})
