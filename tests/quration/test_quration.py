"""Quration backend tests, including optional PyQret execution coverage."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.stdlib import amplitude_encoding
from qamomile.circuit.transpiler.circuit_ir import (
    BinaryExpr,
    BinaryOperator,
    CircuitBuilder,
    LiteralExpr,
    LoopVariableExpr,
    ParameterExpr,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.quration import QurationTranspiler
from qamomile.quration.materializer import PyQretMaterializer, evaluate_scalar


@qmc.qkernel
def _quration_bell(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Prepare and measure a Bell state with a bound rotation."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.ry(left, theta)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _quration_expval(observable: qmc.Observable) -> qmc.Float:
    """Prepare |+> and evaluate an observable."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.h(qubit)
    return qmc.expval(qubit, observable)


@qmc.qkernel
def _quration_resource_bell() -> tuple[qmc.Bit, qmc.Bit]:
    """Prepare a Clifford Bell state for FTQC resource compilation."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.h(left)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _quration_gate_helper(
    left: qmc.Qubit,
    right: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Exercise Quration's composite two-qubit gate decompositions."""
    left, right = qmc.swap(left, right)
    left, right = qmc.rzz(left, right, 0.25)
    return qmc.cp(left, right, -0.5)


@qmc.composite_gate(name="named_rotation")
def _quration_named_rotation(qubit: qmc.Qubit) -> qmc.Qubit:
    """Provide a named reusable body for PyQret call-graph tests."""
    qubit = qmc.h(qubit)
    return qmc.t(qubit)


@qmc.qkernel
def _quration_named_call() -> qmc.Bit:
    """Invoke one named composite twice before measurement."""
    qubit = qmc.qubit("qubit")
    qubit = _quration_named_rotation(qubit)
    qubit = _quration_named_rotation(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _quration_lowering_coverage(
    observable: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Exercise reusable calls, loop unrolling, and Pauli evolution."""
    qubits = qmc.qubit_array(2, "qubits")
    for _ in qmc.range(2):
        qubits[0], qubits[1] = _quration_gate_helper(qubits[0], qubits[1])
    qubits = qmc.pauli_evolve(qubits, observable, gamma)
    return qmc.measure(qubits)


@qmc.qkernel
def _quration_state_preparation() -> qmc.Vector[qmc.Bit]:
    """Exercise semantic state preparation through the PyQret fallback.

    Returns:
        qmc.Vector[qmc.Bit]: Prepared-state measurements.
    """
    qubits = qmc.qubit_array(2, "qubits")
    qubits = amplitude_encoding(qubits, [1.0, 2.0, 3.0, 4.0])
    return qmc.measure(qubits)


@qmc.qkernel
def _quration_ripple_carry() -> tuple[qmc.Vector[qmc.Bit], qmc.Bit, qmc.Bit]:
    """Exercise semantic full addition through the PyQret fallback.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Bit, qmc.Bit]: Accumulator, carry,
            and overflow measurements.
    """
    left = qmc.qubit_array(2, "left")
    right = qmc.qubit_array(2, "right")
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    _, right, carry, overflow = qmc.ripple_carry_add(left, right, carry, overflow)
    return qmc.measure(right), qmc.measure(carry), qmc.measure(overflow)


@qmc.qkernel
def _quration_multi_controlled_x() -> qmc.Bit:
    """Exercise semantic MCX through PyQret's native bounded realization.

    Returns:
        qmc.Bit: Target measurement.
    """
    controls = qmc.x(qmc.qubit_array(2, "controls"))
    target = qmc.qubit("target")
    _, target = qmc.mcx(controls, target)
    return qmc.measure(target)


def test_quration_scalar_evaluator_handles_literals_loops_and_arithmetic() -> None:
    """PyQret materialization evaluates only concrete scalar expressions."""
    expression = BinaryExpr(
        BinaryOperator.MUL,
        LoopVariableExpr("i"),
        LiteralExpr(0.5),
    )

    assert evaluate_scalar(expression, {"i": 3}) == 1.5


def test_quration_scalar_evaluator_rejects_runtime_parameters() -> None:
    """Unbound circuit parameters fail before entering the PyQret builder."""
    with pytest.raises(EmitError, match="must be supplied through bindings"):
        evaluate_scalar(ParameterExpr("theta"))


def test_quration_missing_dependency_has_actionable_error() -> None:
    """Quration use fails clearly when source-built PyQret is unavailable."""
    if importlib.util.find_spec("pyqret") is not None:
        pytest.skip("PyQret is installed in this environment")

    with pytest.raises(ImportError, match="requires the optional 'pyqret'"):
        QurationTranspiler().transpile(
            _quration_bell,
            bindings={"theta": math.pi / 2},
        )


@pytest.mark.quration
def test_quration_transpiles_and_samples_bell_state() -> None:
    """Quration transpilation executes a sampled algorithm through PyQret."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(
        _quration_bell,
        bindings={"theta": math.pi / 2},
    )

    result = executable.sample(transpiler.executor(seed=3), shots=128).result()
    counts = dict(result.results)
    assert sum(counts.values()) == 128
    assert set(counts) <= {(0, 0), (1, 1)}


@pytest.mark.quration
def test_quration_transpiles_and_executes_expectation_value() -> None:
    """Quration full-state simulation evaluates the expval execution path."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(
        _quration_expval,
        bindings={"observable": qm_o.X(0)},
    )

    result = executable.run(transpiler.executor(seed=5)).result()
    assert result == pytest.approx(1.0, abs=1e-10)


@pytest.mark.quration
def test_quration_executes_calls_loops_decompositions_and_pauli_evolution() -> None:
    """Representative Quration lowering paths produce an executable circuit."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(
        _quration_lowering_coverage,
        bindings={"observable": qm_o.X(0) * qm_o.Y(1), "gamma": 0.125},
    )

    result = executable.sample(transpiler.executor(seed=7), shots=32).result()
    assert sum(count for _, count in result.results) == 32


@pytest.mark.quration
@pytest.mark.parametrize(
    "kernel",
    [
        _quration_state_preparation,
        _quration_ripple_carry,
        _quration_multi_controlled_x,
    ],
)
def test_quration_transpiles_semantic_composites(kernel: qmc.QKernel) -> None:
    """New semantic composites select executable native or fallback paths."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(kernel)

    result = executable.sample(transpiler.executor(seed=11), shots=4).result()
    assert sum(count for _, count in result.results) == 4


@pytest.mark.quration
def test_quration_preserves_named_composites_as_pyqret_calls() -> None:
    """Named Qamomile composites remain nodes in PyQret's call graph."""
    pytest.importorskip("pyqret")
    executable = QurationTranspiler().transpile(_quration_named_call)
    circuit = executable.compiled_quantum[0].circuit

    call_graph = circuit.get_ir().gen_call_graph(display_num_calls=True)
    assert call_graph.count('label = "named_rotation') == 1
    assert 'label = "2"' in call_graph


@pytest.mark.quration
def test_quration_emits_program_global_phase_natively() -> None:
    """A concrete global phase reaches PyQret's intrinsic phase operation."""
    pytest.importorskip("pyqret")
    builder = CircuitBuilder(1, 0, name="global_phase")
    builder.add_global_phase(0.25)

    artifact = PyQretMaterializer().materialize(builder.freeze()).artifact

    assert "GlobalPhase 0.25" in artifact.get_ir().gen_cfg()


@pytest.mark.quration
def test_quration_compiles_ftqc_resources_and_preserves_native_owners() -> None:
    """Quration compilation returns stable pass and resource information."""
    pytest.importorskip("pyqret")
    from pyqret.backend import CompileOption, OptLevel, ScLsFixedV0Option

    topology = Path(__file__).parent / "data" / "plane.yaml"
    option = CompileOption(
        opt_level=OptLevel.O0,
        sc_ls_fixed_v0_option=ScLsFixedV0Option(topology=str(topology)),
    )

    result = QurationTranspiler().compile_resources(
        _quration_resource_bell,
        option,
    )

    assert result.compiler.option.opt_level is OptLevel.O0
    assert result.compile_result.get_run_order()
    assert result.compile_info.gate_count > 0
    assert result.circuit.has_mf()
