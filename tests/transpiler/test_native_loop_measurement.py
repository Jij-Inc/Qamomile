"""Regression tests for measurements inside native backend loops (M2).

A backend whose ``GateEmitter.supports_for_loop()`` is ``True`` (Qiskit)
lowers a ``for i in qmc.range(n):`` body to a native loop instruction,
keeping ``i`` as an opaque backend loop parameter. When the body's only
loop-var-indexed access was a measurement (``measure(q[i])``), the old
``LoopAnalyzer._has_array_element_access`` — which enumerated only
``GateOperation`` / ``BinOp`` / ``ControlledUOperation`` / ``PauliEvolveOp``
— failed to flag the loop for unrolling. Emit then could not resolve
``q[i]`` and silently *warned and dropped the measurement*, producing a
circuit that ran to completion (exit code 0) with a measured bit missing.

The fix makes the scan generic over ``all_input_values()`` (so measurement,
reset and projection operands are covered) and promotes the scalar
``emit_measure`` silent-drop warning to an ``EmitError``.

Note: Do NOT use ``from __future__ import annotations`` in this file — the
@qkernel AST transformer relies on resolved type annotations.
"""

import pytest

import qamomile.circuit as qmc

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def measure_each_in_loop(n: qmc.UInt) -> qmc.Bit:
    """Prepare each of ``n`` qubits in |1> and measure it inside the loop.

    Every qubit is flipped to |1> then measured, so the returned bit (the
    last iteration's measurement) is deterministically 1. The per-iteration
    ``measure(q[i])`` is the construct that used to be silently dropped.
    """
    q = qmc.qubit_array(n, "q")
    acc = qmc.bit(False)
    for i in qmc.range(n):
        q[i] = qmc.x(q[i])
        acc = qmc.measure(q[i])
    return acc


@qmc.qkernel
def reset_each_in_loop(n: qmc.UInt) -> qmc.Bit:
    """Flip each qubit to |1>, reset it, then measure — deterministically 0."""
    q = qmc.qubit_array(n, "q")
    acc = qmc.bit(False)
    for i in qmc.range(n):
        q[i] = qmc.x(q[i])
        q[i] = qmc.reset(q[i])
        acc = qmc.measure(q[i])
    return acc


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_loop_measurement_is_not_dropped_on_qiskit(n):
    """Measuring ``q[i]`` inside a native loop emits one measure per iteration."""
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(measure_each_in_loop, bindings={"n": n})
    circuit = executable.quantum_circuit

    num_measure = sum(1 for instr in circuit.data if instr.operation.name == "measure")
    assert num_measure == n, (
        f"expected {n} measurement instructions (one per unrolled iteration), "
        f"got {num_measure} — the native-loop path silently dropped measurements"
    )


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_loop_measurement_sampling_is_deterministic(n):
    """Sampling the loop-measured bit returns the analytic value (all |1>)."""
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(measure_each_in_loop, bindings={"n": n})
    job = executable.sample(transpiler.executor(), shots=128)
    for value, _count in job.result().results:
        assert value == 1, f"expected measured bit 1 for all shots, got {value}"


@pytest.mark.parametrize("n", [1, 2, 3])
def test_loop_reset_then_measure_is_deterministic(n):
    """Reset inside a native loop is honoured: measuring after reset gives 0."""
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(reset_each_in_loop, bindings={"n": n})
    circuit = executable.quantum_circuit
    num_reset = sum(1 for instr in circuit.data if instr.operation.name == "reset")
    assert num_reset == n, (
        f"expected {n} reset instructions, got {num_reset} — reset inside the "
        f"native loop was not unrolled"
    )
    job = executable.sample(transpiler.executor(), shots=128)
    for value, _count in job.result().results:
        assert value == 0, f"expected measured bit 0 after reset, got {value}"


@qmc.qkernel
def measure_only_loop(n: qmc.UInt) -> qmc.Bit:
    """Measure ``q[i]`` inside the loop with no other loop-var access.

    The only loop-var-indexed access is the measurement, so if the loop
    analyzer is forced not to unroll, emit reaches the scalar-measure
    resolution path — exactly the shape of the original silent-drop bug.
    """
    q = qmc.qubit_array(n, "q")
    acc = qmc.bit(False)
    for i in qmc.range(n):
        acc = qmc.measure(q[i])
    return acc


def test_emit_measure_raises_when_loop_not_unrolled(monkeypatch):
    """The scalar-measure safety net raises ``EmitError`` (never a warning).

    Regression for the emit_measure warn→EmitError promotion: if the loop
    analyzer is forced to skip unrolling (simulating the original bug where
    ``measure(q[i])`` slipped through), emit must fail loudly with a
    measurement-specific ``EmitError`` rather than silently dropping the
    measurement.
    """
    from qamomile.circuit.transpiler.errors import EmitError
    from qamomile.circuit.transpiler.passes.emit_support.loop_analyzer import (
        LoopAnalyzer,
    )

    # Force the native-loop path even though the body measures ``q[i]``.
    monkeypatch.setattr(LoopAnalyzer, "should_unroll", lambda self, op, bindings: False)

    transpiler = QiskitTranspiler()
    with pytest.raises(EmitError, match="[Mm]easurement could not be emitted"):
        transpiler.transpile(measure_only_loop, bindings={"n": 3})


@pytest.mark.parametrize("n", [2, 3])
def test_fresh_ancilla_in_native_loop_is_reset_each_iteration(n):
    """A qubit allocated inside a native loop is reset every iteration.

    Regression for the Qiskit native ``for_loop`` path omitting
    ``emit_qinit_reset=True``: without a per-iteration reset, the second and
    later iterations reuse the ancilla in its post-measurement state, silently
    computing a wrong result. The body flips a fresh ancilla to |1> and
    measures it, so it must read 1 on every iteration.
    """

    @qmc.qkernel
    def fresh_ancilla_loop(m: qmc.UInt) -> qmc.Bit:
        acc = qmc.bit(False)
        for _i in qmc.range(m):
            a = qmc.qubit("a")
            a = qmc.x(a)
            acc = qmc.measure(a)
        return acc

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(fresh_ancilla_loop, bindings={"m": n})
    job = executable.sample(transpiler.executor(), shots=128)
    for value, _count in job.result().results:
        assert value == 1, (
            f"expected measured bit 1 on every iteration, got {value} — a fresh "
            f"ancilla in the native loop was not reset between iterations"
        )
