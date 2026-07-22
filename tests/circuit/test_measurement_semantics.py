"""Tests for destructive measurement and projection/reset semantics."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.transpiler.errors import QubitConsumedError


def test_measure_consumes_qubit_without_resetting_it():
    """``measure`` is terminal: the measured qubit handle cannot be reused."""

    @qmc.qkernel
    def kernel(q: qmc.Qubit) -> qmc.Bit:
        bit = qmc.measure(q)
        _ = qmc.x(q)
        return bit

    with pytest.raises(QubitConsumedError, match="already consumed"):
        kernel.build()


def test_project_z_keeps_projected_qubit_and_reset_returns_fresh_handle():
    """``project_z`` keeps a projected qubit handle; ``reset`` prepares it."""

    @qmc.qkernel
    def kernel(q: qmc.Qubit) -> qmc.Bit:
        q, _bit = qmc.project_z(q)
        q = qmc.reset(q)
        q = qmc.x(q)
        return qmc.measure(q)

    block = kernel.build()
    project_ops = [op for op in block.operations if isinstance(op, ProjectOperation)]
    reset_ops = [op for op in block.operations if isinstance(op, ResetOperation)]
    measure_ops = [op for op in block.operations if isinstance(op, MeasureOperation)]

    assert len(project_ops) == 1
    assert project_ops[0].axis == "z"
    assert len(reset_ops) == 1
    assert len(measure_ops) == 1
    assert reset_ops[0].operands[0].uuid == project_ops[0].results[0].uuid


@pytest.mark.parametrize(
    ("project_fn", "basis_gates"),
    [
        (qmc.project_x, [GateOperationType.H, GateOperationType.H]),
        (
            qmc.project_y,
            [
                GateOperationType.SDG,
                GateOperationType.H,
                GateOperationType.H,
                GateOperationType.S,
            ],
        ),
    ],
)
def test_project_x_and_project_y_lower_through_basis_changes(
    project_fn,
    basis_gates,
):
    """X/Y projection lower to basis changes around one Z projection."""

    @qmc.qkernel
    def kernel(q: qmc.Qubit) -> qmc.Bit:
        q, _bit = project_fn(q)
        return qmc.measure(q)

    block = kernel.build()
    project_ops = [op for op in block.operations if isinstance(op, ProjectOperation)]
    gate_types = [
        op.gate_type for op in block.operations if isinstance(op, GateOperation)
    ]

    assert len(project_ops) == 1
    assert project_ops[0].axis == "z"
    assert gate_types == basis_gates


def test_measure_reset_resource_estimate_counts_reset_as_primitive():
    """``measure_reset`` is estimable as projective measure plus reset."""

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        q = qmc.qubit("q")
        q = qmc.x(q)
        q, _bit = qmc.measure_reset(q)
        return qmc.measure(q)

    estimate = kernel.estimate_resources(trace=True).simplify()
    assert estimate.gates.total == 2
    assert estimate.gates.single_qubit == 2
    assert estimate.depth.measurement_depth == 2
    assert estimate.trace is not None
    assert "reset" in estimate.trace.render()


def test_qiskit_emit_measure_reset_uses_backend_reset():
    """Qiskit emission preserves ``measure_reset`` as measurement plus reset."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        q = qmc.qubit("q")
        q = qmc.x(q)
        q, _bit = qmc.measure_reset(q)
        return qmc.measure(q)

    executable = QiskitTranspiler().transpile(kernel)
    circuit = executable.compiled_quantum[0].circuit
    counts = circuit.count_ops()

    assert counts["measure"] == 2
    assert counts["reset"] == 1


class TestResetBackendUnsupported:
    """A backend without a reset primitive fails with EmitError, not raw Python.

    ``GateEmitter.emit_reset`` raises ``NotImplementedError`` on backends with
    no reset primitive (e.g. QURI Parts). That raw exception used to escape a
    normal qkernel compile; ``StandardEmitPass._checked_emit_reset`` now
    converts it into an actionable ``EmitError``.
    """

    def test_reset_on_unsupported_backend_raises_emit_error(self, monkeypatch):
        """qmc.reset on a reset-less emitter raises EmitError with guidance."""
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.qiskit import QiskitTranspiler

        def _no_reset(self, qubit):
            del self, qubit
            raise NotImplementedError("This backend does not support reset.")

        monkeypatch.setattr(QuantumCircuit, "reset", _no_reset)

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.reset(q)
            return qmc.measure(q)

        with pytest.raises(EmitError, match="cannot emit a qubit reset"):
            QiskitTranspiler().transpile(kernel)

    def test_quri_parts_reset_raises_emit_error(self):
        """The QURI Parts backend rejects reset at its capability boundary.

        The declaration-driven target verification now diagnoses reset
        before materialization, so the error is the ``EmitError``-compatible
        ``TargetCapabilityError`` naming the ``quri_parts`` target.
        """
        pytest.importorskip("quri_parts")
        from qamomile.circuit.transpiler.errors import TargetCapabilityError
        from qamomile.quri_parts import QuriPartsTranspiler

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.reset(q)
            return qmc.measure(q)

        with pytest.raises(
            TargetCapabilityError,
            match="cannot represent a mid-circuit reset",
        ) as excinfo:
            QuriPartsTranspiler().transpile(kernel)
        assert excinfo.value.target == "quri_parts"


def test_quri_parts_rejects_projection_followed_by_gate():
    """QURI Parts must not defer a non-terminal projection to shot end."""
    pytest.importorskip("quri_parts")
    from qamomile.circuit.transpiler.errors import EmitError
    from qamomile.quri_parts import QuriPartsTranspiler

    @qmc.qkernel
    def kernel() -> tuple[qmc.Bit, qmc.Bit]:
        q = qmc.qubit("q")
        q, projected = qmc.project_z(q)
        q = qmc.x(q)
        return projected, qmc.measure(q)

    with pytest.raises(EmitError, match="mid-circuit measurement"):
        QuriPartsTranspiler().transpile(kernel)


@pytest.mark.cudaq
def test_cudaq_measure_reset_selects_runnable_mode():
    """CUDA-Q preserves a measurement taken before resetting the qubit."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    @qmc.qkernel
    def kernel() -> tuple[qmc.Bit, qmc.Bit]:
        q = qmc.x(qmc.qubit("q"))
        q, before_reset = qmc.measure_reset(q)
        return before_reset, qmc.measure(q)

    transpiler = CudaqTranspiler()
    result = (
        transpiler.transpile(kernel).sample(transpiler.executor(), shots=32).result()
    )
    assert dict(result.results) == {(1, 0): 32}


def test_repeated_loop_terminal_measurement_is_mid_circuit():
    """A terminal body measurement feeds the next loop iteration."""
    from qamomile.circuit.transpiler.circuit_ir import (
        ForInstruction,
        LoopVariableExpr,
        MeasureInstruction,
        WireId,
        has_mid_circuit_measurement,
    )

    before = WireId(0)
    measured = WireId(1)
    after = WireId(2)
    loop = ForInstruction(
        indexset=range(2),
        loop_variable=LoopVariableExpr("index"),
        inputs=(before,),
        body=(MeasureInstruction(before, measured, 0),),
        body_outputs=(measured,),
        outputs=(after,),
    )

    assert has_mid_circuit_measurement((loop,))
