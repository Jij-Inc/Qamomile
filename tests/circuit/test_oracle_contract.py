"""Tests for consistent scalar and vector oracle control contracts."""

import pytest

import qamomile.circuit as qmc

_CONTROLLED_ORACLE = qmc.Oracle(
    "controlled_vector_contract",
    num_qubits=2,
    num_control_qubits=1,
)


@qmc.qkernel
def _invalid_vector_call() -> qmc.Vector[qmc.Qubit]:
    """Attempt to bypass an oracle's explicit control requirement."""
    qubits = qmc.qubit_array(2, "qubits")
    return _CONTROLLED_ORACLE(qubits)


def test_vector_oracle_cannot_bypass_declared_controls() -> None:
    """Vector syntax rejects an oracle that requires explicit controls."""
    with pytest.raises(ValueError, match="requires 1 explicit control"):
        _invalid_vector_call.build()


@pytest.mark.parametrize(
    "signature",
    [
        pytest.param(
            qmc.CallableSignature(
                inputs=[qmc.Vector[qmc.Qubit]],
                outputs=[qmc.Vector[qmc.Qubit]],
            ),
            id="vector",
        ),
        pytest.param(
            qmc.CallableSignature(
                inputs=[qmc.VectorView[qmc.Qubit]],
                outputs=[qmc.VectorView[qmc.Qubit]],
            ),
            id="vector-view",
        ),
    ],
)
def test_vector_signature_rejects_explicit_controls(
    signature: qmc.CallableSignature,
) -> None:
    """An Oracle cannot be constructed with no valid call form."""
    with pytest.raises(
        ValueError,
        match="vector signatures cannot declare explicit scalar controls",
    ):
        qmc.opaque(
            "invalid_controlled_vector_signature",
            num_control_qubits=1,
            signature=signature,
        )


@pytest.mark.parametrize(
    "signature",
    [
        pytest.param(
            qmc.CallableSignature(
                inputs=[qmc.Qubit],
                outputs=[qmc.Bit],
            ),
            id="non-passthrough-result",
        ),
        pytest.param(
            qmc.CallableSignature(
                inputs=[qmc.Qubit, qmc.Qubit],
                outputs=[qmc.Qubit],
            ),
            id="different-output-arity",
        ),
        pytest.param(
            qmc.CallableSignature(
                inputs=[qmc.Matrix[qmc.Qubit]],
                outputs=[qmc.Matrix[qmc.Qubit]],
            ),
            id="unsupported-quantum-matrix",
        ),
    ],
)
def test_oracle_rejects_unsupported_callable_signature(
    signature: qmc.CallableSignature,
) -> None:
    """Oracle signatures must describe a supported quantum pass-through."""
    with pytest.raises(ValueError, match="Qubit pass-through contract"):
        qmc.opaque("unsupported_signature", signature=signature)


def test_oracle_rejects_scalar_signature_width_mismatch() -> None:
    """A scalar signature and explicit width must describe the same arity."""
    signature = qmc.CallableSignature(
        inputs=[qmc.Qubit],
        outputs=[qmc.Qubit],
    )

    with pytest.raises(ValueError, match="declares 1 scalar qubits"):
        qmc.opaque(
            "mismatched_signature_width",
            num_qubits=2,
            signature=signature,
        )


def test_oracle_accepts_vector_view_signature() -> None:
    """A vector-view annotation retains the supported vector contract."""
    oracle = qmc.opaque(
        "vector_view_signature",
        signature=qmc.CallableSignature(
            inputs=[qmc.VectorView[qmc.Qubit]],
            outputs=[qmc.VectorView[qmc.Qubit]],
        ),
    )

    assert oracle.num_qubits is None


def test_scalar_signature_rejects_vector_call_syntax() -> None:
    """An explicit scalar signature cannot be called through a vector."""
    oracle = qmc.opaque(
        "scalar_signature",
        signature=qmc.CallableSignature(
            inputs=[qmc.Qubit],
            outputs=[qmc.Qubit],
        ),
    )

    @qmc.qkernel
    def invalid_call() -> qmc.Vector[qmc.Qubit]:
        """Call a scalar-signature oracle with a vector."""
        return oracle(qmc.qubit_array(1, "qubits"))

    with pytest.raises(TypeError, match="declared with a scalar signature"):
        invalid_call.build()


def test_fixed_width_vector_signature_rejects_scalar_call_syntax() -> None:
    """A fixed-width vector signature remains vector-only."""
    oracle = qmc.opaque(
        "fixed_width_vector_signature",
        num_qubits=1,
        signature=qmc.CallableSignature(
            inputs=[qmc.Vector[qmc.Qubit]],
            outputs=[qmc.Vector[qmc.Qubit]],
        ),
    )

    @qmc.qkernel
    def invalid_call(q: qmc.Qubit) -> qmc.Qubit:
        """Call a vector-signature oracle with a scalar."""
        (q,) = oracle(q)
        return q

    with pytest.raises(TypeError, match="declared with a vector signature"):
        invalid_call.build()
