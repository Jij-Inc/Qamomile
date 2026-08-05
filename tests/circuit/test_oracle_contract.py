"""Tests for consistent scalar and vector oracle control contracts."""

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.types import QubitType
from qamomile.circuit.ir.value import ArrayValue, array_static_length

_CONTROLLED_ORACLE = qmc.Oracle(
    "controlled_vector_contract",
    num_qubits=2,
    num_control_qubits=1,
)
_VECTOR_ORACLE = qmc.Oracle(
    "vector_contract",
    signature=qmc.CallableSignature(
        inputs=[qmc.Vector[qmc.Qubit]],
        outputs=[qmc.Vector[qmc.Qubit]],
    ),
)


@qmc.qkernel
def _valid_vector_call() -> qmc.Vector[qmc.Qubit]:
    """Return a vector passed through a vector-signature oracle."""
    qubits = qmc.qubit_array(2, "qubits")
    return _VECTOR_ORACLE(qubits)


@qmc.qkernel
def _invalid_vector_call() -> qmc.Vector[qmc.Qubit]:
    """Attempt to bypass an oracle's explicit control requirement."""
    qubits = qmc.qubit_array(2, "qubits")
    return _CONTROLLED_ORACLE(qubits)


def test_vector_oracle_cannot_bypass_declared_controls() -> None:
    """Vector syntax rejects an oracle that requires explicit controls."""
    with pytest.raises(ValueError, match="requires 1 explicit control"):
        _invalid_vector_call.build()


def test_oracle_normalizes_numpy_control_count() -> None:
    """NumPy integer control counts use the same public integral contract."""
    oracle = qmc.opaque(
        "numpy_control_count",
        num_qubits=1,
        num_control_qubits=np.int64(2),
    )

    assert oracle.num_control_qubits == 2
    assert isinstance(oracle.num_control_qubits, int)


def test_controlled_oracle_normalizes_numpy_control_count() -> None:
    """Oracle control transforms normalize NumPy integers before storage."""
    oracle = qmc.opaque("numpy_added_controls", num_qubits=1)

    transformed = qmc.control(oracle, num_controls=np.int64(2))

    assert transformed.added_num_control_qubits == 2
    assert isinstance(transformed.added_num_control_qubits, int)


def test_transformed_oracle_normalizes_numpy_control_count() -> None:
    """Direct transformed-Oracle construction shares the integral contract."""
    oracle = qmc.opaque("numpy_direct_transform", num_qubits=1)

    transformed = qmc.TransformedOracle(
        oracle,
        added_num_control_qubits=np.int64(2),
    )

    assert transformed.added_num_control_qubits == 2
    assert isinstance(transformed.added_num_control_qubits, int)


@pytest.mark.parametrize("num_qubits", [np.int64(0), np.int32(2)])
def test_oracle_normalizes_numpy_target_count(num_qubits: np.integer) -> None:
    """Oracle target widths accept nonnegative NumPy integer scalars."""
    oracle = qmc.opaque("numpy_target_count", num_qubits=num_qubits)

    assert oracle.num_qubits == int(num_qubits)
    assert isinstance(oracle.num_qubits, int)


@pytest.mark.parametrize("num_qubits", [True, False, 1.0])
def test_oracle_rejects_non_integral_target_count(num_qubits: object) -> None:
    """Oracle target widths reject booleans and coercive floats."""
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        qmc.opaque(
            "invalid_target_count",
            num_qubits=num_qubits,  # type: ignore[arg-type]
        )


def test_oracle_rejects_negative_target_count() -> None:
    """Oracle target widths preserve their nonnegative arity contract."""
    with pytest.raises(ValueError, match="num_qubits must be nonnegative"):
        qmc.opaque("negative_target_count", num_qubits=-1)


def test_oracle_uses_identity_equality_and_hashing() -> None:
    """Mutable Oracle definitions remain distinct, hashable identities."""
    first = qmc.opaque("same_fields", num_qubits=1)
    second = qmc.opaque("same_fields", num_qubits=1)

    assert first == first
    assert first != second
    indexed = {first: "first", second: "second"}
    assert len(indexed) == 2
    assert indexed[first] == "first"
    assert indexed[second] == "second"


def test_transformed_oracle_retains_structural_equality_and_hashing() -> None:
    """Frozen transforms compare structurally over one Oracle identity."""
    oracle = qmc.opaque("transform_identity", num_qubits=1)
    equivalent = qmc.control(oracle, num_controls=2)
    repeated = qmc.control(oracle, num_controls=2)
    distinct = qmc.control(
        qmc.opaque("transform_identity", num_qubits=1),
        num_controls=2,
    )

    assert equivalent == repeated
    assert hash(equivalent) == hash(repeated)
    assert equivalent != distinct
    assert {equivalent: "value"}[repeated] == "value"


@pytest.mark.parametrize("inverse", [False, True])
def test_vector_signature_oracle_rejects_control_at_compose_time(
    inverse: bool,
) -> None:
    """A vector-target Oracle fails when control is composed, not when called."""
    oracle = qmc.opaque(
        "vector_target",
        signature=qmc.CallableSignature(
            inputs=[qmc.Vector[qmc.Qubit]],
            outputs=[qmc.Vector[qmc.Qubit]],
        ),
    )
    candidate = qmc.inverse(oracle) if inverse else oracle

    with pytest.raises(TypeError, match="vector-signature oracles"):
        qmc.control(candidate)


def test_vector_signature_oracle_rejects_direct_controlled_wrapper() -> None:
    """Direct transformed-Oracle construction enforces the same scalar ABI."""
    oracle = qmc.opaque(
        "direct_vector_target",
        signature=qmc.CallableSignature(
            inputs=[qmc.Vector[qmc.Qubit]],
            outputs=[qmc.Vector[qmc.Qubit]],
        ),
    )

    with pytest.raises(TypeError, match="vector-signature oracles"):
        qmc.TransformedOracle(oracle, added_num_control_qubits=1)


def test_vector_oracle_preserves_vector_result_shape() -> None:
    """Vector oracle calls retain one vector output at runtime."""
    block = _valid_vector_call.build()

    assert len(block.output_values) == 1
    output = block.output_values[0]
    assert isinstance(output, ArrayValue)
    assert array_static_length(output) == 2
    assert output.type == QubitType()
