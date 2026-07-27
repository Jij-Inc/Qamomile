"""Regression tests for emit-time qubit-operand aliasing detection (S2).

The frontend rejects the scalar ``cx(q, q)`` alias at trace time by comparing
``logical_id``. But symbolic array indices — ``cx(qs[i], qs[j])`` where ``i``
and ``j`` are loop variables that coincide only at runtime, or after loop
unrolling — resolve to the same physical qubit only at emit time. Before this
check the duplicate escaped Qamomile entirely and surfaced as a raw,
backend-specific failure: Qiskit ``CircuitError: 'duplicate qubit arguments'``,
a CUDA-Q simulator crash, or (on a backend that does not validate) a silently
ill-defined gate. ``emit_gate`` now raises a Qamomile ``QubitAliasError`` naming
the gate, both operands, and the shared physical index — one actionable,
backend-independent diagnostic.

Note: Do NOT use ``from __future__ import annotations`` here — the @qkernel AST
transformer relies on resolved type annotations.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import QubitAliasError

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def diagonal_cx(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Apply ``cx(qs[i], qs[j])`` over the full i, j grid — aliases on i == j."""
    qs = qmc.qubit_array(n, "qs")
    for i in qmc.range(n):
        for j in qmc.range(n):
            a, b = qmc.cx(qs[i], qs[j])
            qs[i] = a
            qs[j] = b
    return qmc.measure(qs)


@qmc.qkernel
def chain_cx(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Apply ``cx(qs[i], qs[i+1])`` — always distinct qubits, never aliases."""
    qs = qmc.qubit_array(n, "qs")
    for i in qmc.range(n - 1):
        a, b = qmc.cx(qs[i], qs[i + 1])
        qs[i] = a
        qs[i + 1] = b
    return qmc.measure(qs)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_diagonal_cx_raises_qubit_alias_error(n):
    """``cx(qs[i], qs[j])`` on the diagonal is rejected with a named error."""
    with pytest.raises(QubitAliasError) as exc_info:
        QiskitTranspiler().transpile(diagonal_cx, bindings={"n": n})
    message = str(exc_info.value)
    assert "CX" in message
    assert "qs[i]" in message and "qs[j]" in message
    assert "distinct qubits" in message


@pytest.mark.parametrize("n", [2, 3, 5])
def test_chain_cx_transpiles(n):
    """The off-diagonal chain ``cx(qs[i], qs[i+1])`` transpiles cleanly."""
    executable = QiskitTranspiler().transpile(chain_cx, bindings={"n": n})
    assert executable is not None


def test_scalar_cx_alias_still_caught_at_trace_time():
    """The scalar ``cx(q, q)`` alias is still rejected by the frontend."""

    @qmc.qkernel
    def scalar_alias() -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit("q")
        _a, _b = qmc.cx(q, q)
        return qmc.measure(qmc.qubit_array(1, "z"))

    with pytest.raises(QubitAliasError):
        scalar_alias.build()


def test_swap_alias_is_rejected():
    """A two-qubit ``swap`` with coinciding indices is also rejected."""

    @qmc.qkernel
    def diagonal_swap(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n):
            for j in qmc.range(n):
                a, b = qmc.swap(qs[i], qs[j])
                qs[i] = a
                qs[j] = b
        return qmc.measure(qs)

    with pytest.raises(QubitAliasError) as exc_info:
        QiskitTranspiler().transpile(diagonal_swap, bindings={"n": 2})
    assert "SWAP" in str(exc_info.value)


def test_toffoli_two_operand_alias_is_rejected():
    """A ``ccx`` whose two controls coincide (i == j) is rejected."""

    @qmc.qkernel
    def diagonal_ccx(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n):
            for j in qmc.range(n):
                a, b, c = qmc.ccx(qs[i], qs[j], qs[2])
                qs[i] = a
                qs[j] = b
                qs[2] = c
        return qmc.measure(qs)

    with pytest.raises(QubitAliasError) as exc_info:
        QiskitTranspiler().transpile(diagonal_ccx, bindings={"n": 3})
    assert "TOFFOLI" in str(exc_info.value)


def test_cp_rotation_alias_is_rejected():
    """A controlled-phase ``cp`` (angle in operands) still detects aliasing.

    ``cp`` carries its rotation angle in ``operands``; ``qubit_operands``
    excludes it, so the two-qubit aliasing check must still see exactly the two
    qubit operands.
    """

    @qmc.qkernel
    def diagonal_cp(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n):
            for j in qmc.range(n):
                a, b = qmc.cp(qs[i], qs[j], 0.5)
                qs[i] = a
                qs[j] = b
        return qmc.measure(qs)

    with pytest.raises(QubitAliasError) as exc_info:
        QiskitTranspiler().transpile(diagonal_cp, bindings={"n": 2})
    assert "CP" in str(exc_info.value)


def test_loop_var_vs_constant_index_alias_is_rejected():
    """``cx(qs[i], qs[0])`` aliases on the iteration where i == 0."""

    @qmc.qkernel
    def cx_to_zero(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n):
            a, b = qmc.cx(qs[i], qs[0])
            qs[i] = a
            qs[0] = b
        return qmc.measure(qs)

    with pytest.raises(QubitAliasError):
        QiskitTranspiler().transpile(cx_to_zero, bindings={"n": 3})


# ---------------------------------------------------------------------------
# Controlled-block (qmc.control) path: shares the emit-time aliasing check via
# the checked append_gate wrapper in controlled_emission.
# ---------------------------------------------------------------------------


@qmc.qkernel
def _x_gate(q: qmc.Qubit) -> qmc.Qubit:
    """Single-qubit X, wrapped so it can be lifted to a controlled block."""
    return qmc.x(q)


_controlled_x = qmc.control(_x_gate)


def test_controlled_block_diagonal_alias_is_rejected():
    """``qmc.control(x)(qs[i], qs[j])`` on the diagonal is rejected.

    Before the controlled-path fix this reached the backend as a raw Qiskit
    ``CircuitError`` and, on CUDA-Q, transpiled silently and crashed the
    simulator at run time.
    """

    @qmc.qkernel
    def diagonal_controlled(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n):
            for j in qmc.range(n):
                qs[i], qs[j] = _controlled_x(qs[i], qs[j])
        return qmc.measure(qs)

    with pytest.raises(QubitAliasError):
        QiskitTranspiler().transpile(diagonal_controlled, bindings={"n": 2})


def test_controlled_block_off_diagonal_transpiles():
    """``qmc.control(x)(qs[i], qs[i+1])`` (distinct qubits) transpiles."""

    @qmc.qkernel
    def chain_controlled(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n - 1):
            qs[i], qs[i + 1] = _controlled_x(qs[i], qs[i + 1])
        return qmc.measure(qs)

    executable = QiskitTranspiler().transpile(chain_controlled, bindings={"n": 3})
    assert executable is not None


def test_controlled_block_fallback_path_rejects_alias(monkeypatch):
    """The block-decomposition fallback also rejects control/target aliasing.

    When ``_blockvalue_to_gate`` cannot produce a native gate (always the case
    on QURI Parts, and on CUDA-Q via a generated helper kernel), the controlled
    block is decomposed gate-by-gate and never routes through ``append_gate``.
    Forcing that path here confirms the shared aliasing check still fires — the
    diagnostic is labelled ``controlled gate (fallback)``.
    """
    from qamomile.circuit.transpiler.circuit_ir import CircuitLoweringPass

    # Force the fallback branch (``unitary_gate is None``).
    monkeypatch.setattr(
        CircuitLoweringPass, "_blockvalue_to_gate", lambda self, *a, **k: None
    )

    @qmc.qkernel
    def diagonal_controlled(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n):
            for j in qmc.range(n):
                qs[i], qs[j] = _controlled_x(qs[i], qs[j])
        return qmc.measure(qs)

    with pytest.raises(QubitAliasError, match="fallback"):
        QiskitTranspiler().transpile(diagonal_controlled, bindings={"n": 2})


# ---------------------------------------------------------------------------
# Inverse-block (qmc.inverse) path: shares the emit-time aliasing check at the
# entry of emit_inverse_block_at_indices, covering all four sub-paths.
# ---------------------------------------------------------------------------


@qmc.qkernel
def _cx_block(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Two-qubit block (``cx``) used to build an inverse block."""
    a, b = qmc.cx(a, b)
    return a, b


_inverse_cx = qmc.inverse(_cx_block)


def test_inverse_block_diagonal_alias_is_rejected():
    """``qmc.inverse(u)(qs[i], qs[j])`` on the diagonal is rejected.

    Before the inverse-path fix this reached the backend as a raw Qiskit
    ``CircuitError`` and, on CUDA-Q, transpiled silently then crashed the
    simulator at run time.
    """

    @qmc.qkernel
    def diagonal_inverse(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n):
            for j in qmc.range(n):
                qs[i], qs[j] = _inverse_cx(qs[i], qs[j])
        return qmc.measure(qs)

    with pytest.raises(QubitAliasError, match="inverse block"):
        QiskitTranspiler().transpile(diagonal_inverse, bindings={"n": 2})


def test_inverse_block_diagonal_alias_is_rejected_on_cudaq():
    """CUDA-Q's inverse adjoint fast path also rejects target aliasing.

    CUDA-Q overrides ``_emit_inverse_block`` with an adjoint-helper fast path
    that bypasses the base ``emit_inverse_block_at_indices`` entry check. Before
    the fix this transpiled silently and crashed the simulator at run time
    (``qpp::applyCTRL(): Subsystems mismatch dimensions!``).
    """
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    @qmc.qkernel
    def diagonal_inverse(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n):
            for j in qmc.range(n):
                qs[i], qs[j] = _inverse_cx(qs[i], qs[j])
        return qmc.measure(qs)

    with pytest.raises(QubitAliasError, match="inverse block"):
        CudaqTranspiler().transpile(diagonal_inverse, bindings={"n": 2})


def test_inverse_block_off_diagonal_transpiles():
    """A non-aliasing inverse block ``inverse(u)(qs[i], qs[i+1])`` transpiles."""

    @qmc.qkernel
    def chain_inverse(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(n, "qs")
        for i in qmc.range(n - 1):
            qs[i], qs[i + 1] = _inverse_cx(qs[i], qs[i + 1])
        return qmc.measure(qs)

    executable = QiskitTranspiler().transpile(chain_inverse, bindings={"n": 3})
    assert executable is not None


def test_reject_duplicate_physical_indices_unit():
    """The shared checker raises exactly when a physical index repeats."""
    from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
        reject_duplicate_physical_indices,
    )

    # No duplicates / trivially short lists: no error.
    reject_duplicate_physical_indices("g", [0, 1, 2])
    reject_duplicate_physical_indices("g", [5])
    reject_duplicate_physical_indices("g", [])

    # Duplicates: QubitAliasError, and operand names are quoted when given.
    with pytest.raises(QubitAliasError, match="qs\\[i\\].*qs\\[j\\]"):
        reject_duplicate_physical_indices("CX", [0, 0], ["qs[i]", "qs[j]"])
    with pytest.raises(QubitAliasError):
        reject_duplicate_physical_indices("TOFFOLI", [2, 5, 2])


# ---------------------------------------------------------------------------
# CUDA-Q / QuriParts cross-backend: the check lives in the shared emit_gate,
# so every backend routed through StandardEmitPass gets the same diagnostic.
# ---------------------------------------------------------------------------


def test_diagonal_cx_raises_on_cudaq():
    """CUDA-Q also raises ``QubitAliasError`` (shared ``emit_gate`` path)."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    with pytest.raises(QubitAliasError):
        CudaqTranspiler().transpile(diagonal_cx, bindings={"n": 2})


def test_diagonal_cx_raises_on_quri_parts():
    """QuriParts also raises ``QubitAliasError`` (shared ``emit_gate`` path)."""
    pytest.importorskip("quri_parts")
    from qamomile.quri_parts import QuriPartsTranspiler

    with pytest.raises(QubitAliasError):
        QuriPartsTranspiler().transpile(diagonal_cx, bindings={"n": 2})
