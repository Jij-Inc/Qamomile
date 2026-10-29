"""Verify measurement results, returned states, and final H2 instructions."""

from __future__ import annotations

import re
from collections import Counter

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import QubitConsumedError
from qamomile.hugr import HugrExecutor, HugrTranspiler, SeleneExecutionOptions
from qamomile.hugr._qir import to_h2_qir
from qamomile.hugr._runtime import prepare_execution

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from hugr import ops

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _terminal(initial: qmc.UInt) -> qmc.Bit:
    """Consume an eigenstate and return its computational-basis bit."""
    qubit = qmc.qubit("qubit")
    if initial:
        qubit = qmc.x(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _reset(initial: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
    """Return the original bit and measure the independently reset state."""
    qubit = qmc.qubit("qubit")
    if initial:
        qubit = qmc.x(qubit)
    qubit, original = qmc.measure_reset(qubit)
    return original, qmc.measure(qubit)


@qmc.qkernel
def _feedback_reset(initial: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
    """Use one measurement result in feedback and two public output positions."""
    qubit = qmc.qubit("qubit")
    if initial:
        qubit = qmc.x(qubit)
    qubit, original = qmc.measure_reset(qubit)
    if original:
        qubit = qmc.x(qubit)
    return original, original, qmc.measure(qubit)


@qmc.qkernel
def _project_z(initial: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
    """Retain both Z eigenstates across repeated projection."""
    qubit = qmc.qubit("qubit")
    if initial:
        qubit = qmc.x(qubit)
    qubit, first = qmc.project_z(qubit)
    qubit, second = qmc.project_z(qubit)
    return first, second, qmc.measure(qubit)


@qmc.qkernel
def _project_x(initial: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
    """Decode the returned X eigenstate after two repeatable projections."""
    qubit = qmc.qubit("qubit")
    if initial:
        qubit = qmc.x(qubit)
    qubit = qmc.h(qubit)
    qubit, first = qmc.project_x(qubit)
    qubit, second = qmc.project_x(qubit)
    qubit = qmc.h(qubit)
    return first, second, qmc.measure(qubit)


@qmc.qkernel
def _project_y(initial: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
    """Decode the returned Y eigenstate after two repeatable projections."""
    qubit = qmc.qubit("qubit")
    if initial:
        qubit = qmc.x(qubit)
    qubit = qmc.h(qubit)
    qubit = qmc.s(qubit)
    qubit, first = qmc.project_y(qubit)
    qubit, second = qmc.project_y(qubit)
    qubit = qmc.sdg(qubit)
    qubit = qmc.h(qubit)
    return first, second, qmc.measure(qubit)


@qmc.qkernel
def _single_projection(initial: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
    """Keep the projected state available for a terminal measurement."""
    qubit = qmc.qubit("qubit")
    if initial:
        qubit = qmc.x(qubit)
    qubit, original = qmc.project_z(qubit)
    return original, qmc.measure(qubit)


@qmc.qkernel
def _vector() -> qmc.Vector[qmc.Bit]:
    """Preserve an asymmetric three-bit pattern in vector index order."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    return qmc.measure(register)


@qmc.qkernel
def _integer() -> qmc.UInt:
    """Decode the low two bits of a three-qubit little-endian integer."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _fixed() -> qmc.Float:
    """Decode weights one half and one from a three-qubit fixed-point value."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    return qmc.measure(qmc.cast(register, qmc.QFixed, int_bits=2))


@qmc.qkernel
def _nested_reset(count: qmc.UInt, initial: qmc.UInt) -> qmc.Bit:
    """Reset inside both branches of a runtime loop before returning a bit."""
    condition = qmc.qubit("selector")
    if initial:
        condition = qmc.x(condition)
    selector = qmc.measure(condition)
    qubit = qmc.x(qmc.qubit("qubit"))
    for _ in qmc.range(count):
        if selector:
            qubit, _bit = qmc.measure_reset(qubit)
        else:
            qubit, _bit = qmc.measure_reset(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _intervening_gate() -> tuple[qmc.Bit, qmc.Bit]:
    """Retain a projection when an X gate separates it from reset."""
    qubit = qmc.x(qmc.qubit("qubit"))
    qubit, original = qmc.project_z(qubit)
    qubit = qmc.x(qubit)
    qubit = qmc.reset(qubit)
    return original, qmc.measure(qubit)


def _counts(package) -> Counter[str]:
    """Count target operations in all nested regions and helper functions."""
    return Counter(
        str(name())
        for module in package.modules
        for _, data in module.nodes()
        if callable(name := getattr(data.op, "name", None))
    )


def _sample(executable, tmp_path, expected, bindings=None):
    """Observe a deterministic public result and preserve the reusable artifact."""
    before = executable.artifact.to_bytes()
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path, seed=37))
    result = executable.sample(executor, shots=4, bindings=bindings).result()
    assert result.results == [(expected, 4)]
    assert executable.artifact.to_bytes() == before


@pytest.mark.parametrize("initial", [0, 1])
@pytest.mark.parametrize(
    "kernel",
    [_terminal, _reset, _project_x, _project_y, _project_z],
    ids=["terminal", "reset", "project-x", "project-y", "project-z"],
)
def test_public_measurement_results_and_returned_eigenstates(tmp_path, kernel, initial):
    """Both eigenvalues survive projection, while reset always returns zero state."""
    transpiler = HugrTranspiler()
    executable = transpiler.transpile(kernel, bindings={"initial": initial})
    transpiler.target.validate(executable.artifact)
    bit = bool(initial)
    if kernel is _terminal:
        expected = bit
    elif kernel is _reset:
        expected = (bit, False)
    else:
        expected = (bit, bit, bit)
    _sample(executable, tmp_path, expected)


def test_terminal_measurement_rejects_reusing_consumed_qubit():
    """Public measurement consumes ownership before target emission."""

    @qmc.qkernel
    def invalid() -> qmc.Bit:
        qubit = qmc.qubit("qubit")
        bit = qmc.measure(qubit)
        qubit = qmc.x(qubit)
        return bit

    with pytest.raises(QubitConsumedError, match="already consumed"):
        HugrTranspiler().compile(invalid)


@pytest.mark.parametrize("initial", [0, 1])
def test_measure_reset_preserves_result_fanout_and_feedback(tmp_path, initial):
    """Fused reset publishes its original bit to every output and condition."""
    transpiler = HugrTranspiler()
    executable = transpiler.transpile(_feedback_reset, bindings={"initial": initial})
    transpiler.target.validate(executable.artifact)
    counts = _counts(executable.artifact)
    assert counts["tket.qsystem.MeasureReset"] == 1
    assert counts["tket.quantum.Measure"] == 0
    assert counts["tket.quantum.Reset"] == 0
    _sample(executable, tmp_path, (bool(initial),) * 3)


@pytest.mark.parametrize(
    "kernel,expected", [(_vector, (True, True, False)), (_integer, 3), (_fixed, 1.5)]
)
def test_aggregate_measurement_keeps_bit_order_without_extra_frees(
    tmp_path, kernel, expected
):
    """Vector, integer, and fixed-point results consume each carrier exactly once."""
    transpiler = HugrTranspiler()
    executable = transpiler.transpile(kernel)
    transpiler.target.validate(executable.artifact)
    counts = _counts(executable.artifact)
    assert counts["tket.quantum.QAlloc"] == 3
    assert counts["tket.quantum.MeasureFree"] == 3
    assert counts["tket.quantum.Measure"] == 0
    assert counts["tket.quantum.QFree"] == 0
    assert counts["tket.bool.read"] == 3
    _sample(executable, tmp_path, expected)


@pytest.mark.parametrize("initial", [0, 1])
def test_measure_reset_fuses_in_nested_runtime_regions(tmp_path, initial):
    """Native measurement/reset remains valid inside conditional loop regions."""
    transpiler = HugrTranspiler()
    executable = transpiler.transpile(
        _nested_reset, parameters=["count"], bindings={"initial": initial}
    )
    transpiler.target.validate(executable.artifact)
    operations = [data.op for _, data in executable.artifact.modules[0].nodes()]
    assert any(isinstance(operation, ops.TailLoop) for operation in operations)
    assert any(isinstance(operation, ops.Conditional) for operation in operations)
    counts = _counts(executable.artifact)
    assert counts["tket.qsystem.MeasureReset"] == 2
    assert counts["tket.quantum.Measure"] == 0
    assert counts["tket.quantum.Reset"] == 0
    assert counts["tket.quantum.QFree"] == 0
    _sample(executable, tmp_path, False, {"count": 2})


def test_projection_and_later_reset_keep_the_intervening_gate(tmp_path):
    """A gate between projection and reset prevents destructive fusion."""
    transpiler = HugrTranspiler()
    executable = transpiler.transpile(_intervening_gate)
    transpiler.target.validate(executable.artifact)
    counts = _counts(executable.artifact)
    assert counts["tket.quantum.Measure"] == 1
    assert counts["tket.quantum.Reset"] == 1
    assert counts["tket.qsystem.MeasureReset"] == 0
    assert counts["tket.quantum.MeasureFree"] == 1
    assert counts["tket.quantum.QFree"] == 0
    _sample(executable, tmp_path, (True, False))


@pytest.mark.parametrize(
    "kernel,resets,branches,measurements",
    [
        (_terminal, 0, 0, 1),
        (_reset, 1, 0, 2),
        (_single_projection, 1, 1, 2),
        (_project_x, 2, 2, 3),
        (_project_y, 2, 2, 3),
    ],
)
def test_final_h2_qir_only_restores_states_that_remain_live(
    kernel, resets, branches, measurements
):
    """Inspect actual bitcode for resets and measurement reconstruction branches."""
    pytest.importorskip("hugr_qir.hugr_to_qir")
    pyqir = pytest.importorskip("pyqir")
    compiled = HugrTranspiler().compile(kernel, bindings={"initial": 1})
    original = compiled.artifact.to_bytes()
    prepared = prepare_execution(compiled)
    before = prepared.package.to_bytes()
    bitcode, outputs = to_h2_qir(prepared.package)
    llvm = str(pyqir.Module.from_bitcode(pyqir.Context(), bitcode))
    assert len(re.findall(r"call void @__quantum__qis__reset__body\(", llvm)) == resets
    assert len(re.findall(r"\bbr i1\b", llvm)) == branches
    assert (
        len(re.findall(r"call void @__quantum__qis__mz__body\(", llvm)) == measurements
    )
    assert set(outputs) == set(prepared.output_tags)
    assert prepared.package.to_bytes() == before
    assert compiled.artifact.to_bytes() == original
