"""Tests for the contract-aware resource-estimation input UX."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
import qamomile.observable as qm_o


@qmc.composite_gate(name="phase_u")
def _phase_u(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply a phase rotation (angle does not affect gate counts)."""
    return qmc.p(q, theta)


@qmc.qkernel
def _toy_qpe(bits: qmc.UInt = 4, theta: qmc.Float = 0.1) -> qmc.Vector[qmc.Bit]:
    """A QPE kernel whose ``bits`` argument carries a Python default."""
    counting = qmc.qubit_array(bits, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)
    for k in qmc.range(bits):
        counting[k] = qmc.h(counting[k])
    for k in qmc.range(bits):
        cu = qmc.control(_phase_u)
        counting[k], target = cu(counting[k], target, theta=theta, power=2**k)
    counting = qmc.iqft(counting)
    return qmc.measure(counting)


@qmc.qkernel
def _sized_kernel(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate an n-bit register for symbolic-size substitution tests."""
    return qmc.qubit_array(n, name="reg")


@qmc.qkernel
def _branch_probe(flag: qmc.UInt = 0) -> qmc.Qubit:
    """One gate on the true branch, two on the false branch."""
    q = qmc.qubit("q")
    if flag:
        q = qmc.x(q)
    else:
        q = qmc.h(q)
        q = qmc.z(q)
    return q


@qmc.qkernel
def _cmp_branch(n: qmc.UInt) -> qmc.Qubit:
    """A comparison-predicate branch: one gate if n > 5, else three."""
    q = qmc.qubit("q")
    if n > 5:
        q = qmc.x(q)
    else:
        q = qmc.h(q)
        q = qmc.z(q)
        q = qmc.h(q)
    return q


@qmc.qkernel
def _dual_role(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """``n`` drives both a register size and a branch predicate."""
    reg = qmc.qubit_array(n, name="reg")
    for k in qmc.range(n):
        reg[k] = qmc.h(reg[k])
    extra = qmc.qubit("extra")
    if n > 3:
        extra = qmc.x(extra)
    else:
        extra = qmc.h(extra)
        extra = qmc.z(extra)
    return qmc.measure(reg)


@qmc.qkernel
def _measurement_branch() -> qmc.Bit:
    """A runtime measurement-backed branch that must not be specialized."""
    q = qmc.qubit("q")
    bit = qmc.measure(q)
    r = qmc.qubit("r")
    if bit:
        r = qmc.x(r)
    else:
        r = qmc.h(r)
        r = qmc.z(r)
    return qmc.measure(r)


def test_branch_specialized_on_concrete_flag() -> None:
    """A decidable compile-time branch counts only the taken branch."""
    true_est = _branch_probe.estimate_resources(inputs={"flag": 1})
    false_est = _branch_probe.estimate_resources(inputs={"flag": 0})
    assert true_est.gates.total == 1  # only qmc.x
    assert false_est.gates.total == 2  # qmc.h + qmc.z


@pytest.mark.parametrize(
    "value", [np.int64(1), np.int32(1), np.float64(1.0)], ids=["i64", "i32", "f64"]
)
def test_branch_specialized_on_numpy_scalar(value: object) -> None:
    """NumPy scalar substitution values specialize branches like Python scalars.

    A ``np.int64`` is a very natural notebook/algorithm output; it must decide a
    compile-time branch rather than being silently dropped to the conservative
    estimate.
    """
    assert _branch_probe.estimate_resources(inputs={"flag": value}).gates.total == 1
    assert _cmp_branch.estimate_resources(inputs={"n": np.int64(8)}).gates.total == 1


def test_symbolic_compile_time_branch_stays_piecewise() -> None:
    """A Python default remains a symbolic exact branch during estimation."""
    est = _branch_probe.estimate_resources()
    assert str(est.gates.total) == "Piecewise((1, flag), (2, True))"


def test_comparison_branch_specialized() -> None:
    """A comparison predicate is decided from a substituted value."""
    assert _cmp_branch.estimate_resources(inputs={"n": 8}).gates.total == 1
    assert _cmp_branch.estimate_resources(inputs={"n": 3}).gates.total == 3


@qmc.qkernel
def _two_param_branch(n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
    """A branch predicate over two parameters (needs both to decide)."""
    q = qmc.qubit("q")
    if n > m:
        q = qmc.x(q)
    else:
        q = qmc.h(q)
        q = qmc.z(q)
    return q


@qmc.qkernel
def _positive_branch(n: qmc.UInt) -> qmc.Qubit:
    """A predicate SymPy pre-decides from the positive-integer assumption."""
    q = qmc.qubit("q")
    if n > 0:
        q = qmc.x(q)
    else:
        q = qmc.h(q)
        q = qmc.z(q)
    return q


def _messages(est) -> list[str]:
    """Return the assumption messages of an estimate."""
    return [a.message for a in est.assumptions]


def test_partially_supplied_branch_remains_exact_piecewise() -> None:
    """A partially specialized compile-time branch keeps its exact predicate."""
    est = _two_param_branch.estimate_resources(inputs={"n": 8})
    m = sp.Symbol("m", integer=True, positive=True)
    assert est.gates.total == sp.Piecewise((1, m < 8), (2, True))
    messages = _messages(est)
    assert not any("'n'" in m and "ignored" in m for m in messages)


def test_symbolic_branch_has_no_undecidable_noise() -> None:
    """A fully-symbolic estimate does not emit an undecidable-branch assumption."""
    est = _two_param_branch.estimate_resources()
    assert not any("undecidable" in m for m in _messages(est))


def test_fully_supplied_branch_specializes_without_assumption() -> None:
    """Supplying every predicate operand decides the branch, with no assumption."""
    est = _two_param_branch.estimate_resources(inputs={"n": 8, "m": 3})
    assert est.gates.total == 1
    assert not any("undecidable" in m for m in _messages(est))


def test_pre_decided_predicate_has_no_undecidable_assumption() -> None:
    """A predicate decided by SymPy assumptions does not report undecidable."""
    est = _positive_branch.estimate_resources(inputs={"n": 8})
    assert est.gates.total == 1
    assert not any("undecidable" in m for m in _messages(est))


def test_dual_role_symbol_is_consistent() -> None:
    """A symbol driving both a branch and a size specializes both consistently."""
    est5 = _dual_role.estimate_resources(inputs={"n": 5})
    assert est5.gates.total == 6  # 5 H + 1 (n>3 true branch)
    assert int(est5.qubits) == 6  # 5 reg + 1 extra

    est2 = _dual_role.estimate_resources(inputs={"n": 2})
    assert est2.gates.total == 4  # 2 H + 2 (n>3 false branch)
    assert int(est2.qubits) == 3  # 2 reg + 1 extra


def test_measurement_branch_not_specialized() -> None:
    """A runtime measurement-backed branch stays a conservative maximum."""
    est = _measurement_branch.estimate_resources()
    # 1 H-free path: measure + choice(max(1, 2)) = 2 gates in the branch.
    assert est.gates.total == 2


def test_untaken_branch_allocations_are_not_counted() -> None:
    """A qubit allocated only in the untaken branch does not inflate width."""

    @qmc.qkernel
    def alloc_branch(flag: qmc.UInt = 0) -> qmc.Qubit:
        """Allocate an ancilla only on the false branch."""
        q = qmc.qubit("q")
        if flag:
            q = qmc.x(q)
        else:
            anc = qmc.qubit("anc")
            anc = qmc.h(anc)
            q = qmc.cx(anc, q)[1]
        return q

    taken = alloc_branch.estimate_resources(inputs={"flag": 1})
    # Only |q> is live on the true branch; the false-branch ancilla is gone.
    assert int(taken.qubits) == 1


@qmc.qkernel
def _carried_loop_bound(n: qmc.UInt) -> qmc.Qubit:
    """Use a loop-carried counter as a later quantum-loop bound."""
    count = qmc.uint(0)
    for _ in qmc.range(n):
        count = count + 1
    q = qmc.qubit("q")
    for _ in qmc.range(count):
        q = qmc.x(q)
    return q


@qmc.qkernel
def _carried_branch(n: qmc.UInt) -> qmc.Qubit:
    """Use a loop-carried counter as a later branch condition."""
    count = qmc.uint(0)
    for _ in qmc.range(n):
        count = count + 1
    q = qmc.qubit("q")
    if count == 1:
        q = qmc.x(q)
    else:
        q = qmc.h(q)
        q = qmc.z(q)
    return q


def test_region_arg_drives_later_symbolic_loop() -> None:
    """A carried counter is published as the symbolic result ``n``."""
    estimate = _carried_loop_bound.estimate_resources()
    n = sp.Symbol("n", integer=True, positive=True)

    assert sp.simplify(estimate.gates.total - n) == 0
    assert set(estimate.parameters) == {"n"}


def test_region_arg_drives_later_concrete_loop() -> None:
    """Inputs specialize a carried loop bound."""
    estimate = _carried_loop_bound.estimate_resources(inputs={"n": 3})

    assert estimate.gates.total == 3
    assert estimate.parameters == {}


def test_large_input_keeps_region_loop_symbolic(monkeypatch) -> None:
    """A large estimation input is substituted after loop summarization."""
    from qamomile.circuit.estimator.resource_estimator import ResourceInterpreter

    def fail_concrete_iteration(*args, **kwargs):
        raise AssertionError("estimation input triggered concrete loop execution")

    monkeypatch.setattr(
        ResourceInterpreter,
        "_eval_concrete_region_for",
        fail_concrete_iteration,
    )

    estimate = _carried_loop_bound.estimate_resources(inputs={"n": 2048})
    assert estimate.gates.total == 2048


def test_region_arg_drives_later_branch() -> None:
    """A concrete carried value specializes a later compile-time branch."""
    one = _carried_branch.estimate_resources(inputs={"n": 1})
    two = _carried_branch.estimate_resources(inputs={"n": 2})

    assert one.gates.total == 1
    assert two.gates.total == 2


def test_inputs_include_noop_angle() -> None:
    """Passing all kernel inputs works; an angle that affects no metric is a no-op.

    ``theta`` never appears in any resource expression, so supplying it is a
    recorded no-op rather than an error — the user simply passed the kernel's
    declared inputs.
    """
    est = _toy_qpe.estimate_resources(inputs={"bits": 5, "theta": 0.25})

    assert int(est.qubits) == 6  # 5 counting + 1 target
    ignored = [a.message for a in est.assumptions if "ignored" in a.message]
    assert any("theta" in message for message in ignored)


def test_inputs_force_symbolic_over_python_default() -> None:
    """Estimation keeps Python defaults symbolic until inputs specialize them.

    ``bits`` has default 4, but the symbolic-first estimator must expose the
    expression rather than silently baking the execution default.
    """
    default = _toy_qpe.estimate_resources()
    bits = sp.Symbol("bits", integer=True, positive=True)
    assert sp.simplify(default.qubits - (bits + 1)) == 0

    est = _toy_qpe.estimate_resources(inputs={"bits": 5})
    assert int(est.qubits) == 6


def test_inputs_accept_shift_expression() -> None:
    """An input expression may reintroduce the same symbol name (n -> n+1)."""
    n = sp.Symbol("n", integer=True, positive=True)
    est = _sized_kernel.estimate_resources(inputs={"n": n + 1})
    # The reintroduced symbol still sizes the register correctly.
    assert sp.simplify(est.qubits - (n + 1)) == 0


def test_input_typo_raises() -> None:
    """A name that is neither a free symbol nor a kernel argument raises."""
    with pytest.raises(ValueError, match="neither free symbols"):
        _toy_qpe.estimate_resources(inputs={"bti": 5})


def test_quantum_port_is_not_an_estimation_input() -> None:
    """Quantum port names are rejected instead of treated as structural data."""

    @qmc.qkernel
    def quantum_input(q: qmc.Qubit) -> qmc.Qubit:
        """Apply one gate to a caller-owned quantum input."""
        return qmc.h(q)

    with pytest.raises(ValueError, match="neither free symbols"):
        quantum_input.estimate_resources(inputs={"q": 0})


def test_interleaved_composite_signature_binds_resource_parameter() -> None:
    """Resource estimation reweaves grouped operands to formal order."""

    @qmc.composite_gate(name="interleaved_resource_box")
    def interleaved_resource_box(
        first: qmc.Qubit,
        rounds: qmc.UInt,
        second: qmc.Qubit,
    ) -> tuple[qmc.Qubit, qmc.Qubit]:
        """Apply ``rounds`` X gates and one H gate."""
        for _ in qmc.range(rounds):
            first = qmc.x(first)
        second = qmc.h(second)
        return first, second

    @qmc.qkernel
    def algorithm(rounds: qmc.UInt) -> tuple[qmc.Qubit, qmc.Qubit]:
        """Invoke the interleaved composite on two fresh qubits."""
        first = qmc.qubit("first")
        second = qmc.qubit("second")
        return interleaved_resource_box(first, rounds, second)

    estimate = algorithm.estimate_resources(inputs={"rounds": 3})

    assert estimate.gates.total == 4


def test_inputs_trace_structural_values_and_specialize_scalars() -> None:
    """One input mapping handles structural and symbolic qkernel arguments."""

    @qmc.qkernel
    def observable_probe(n: qmc.UInt, observable: qmc.Observable) -> qmc.Float:
        """Apply ``n`` gates and evaluate one supplied observable."""
        reg = qmc.qubit_array(n, "reg")
        for index in qmc.range(n):
            reg[index] = qmc.h(reg[index])
        return qmc.expval(reg, observable)

    estimate = observable_probe.estimate_resources(
        inputs={"n": 3, "observable": qm_o.Z(0)}
    )

    assert estimate.qubits == 3
    assert estimate.gates.total == 3
    assert estimate.parameters == {}


@pytest.mark.parametrize(
    "angles",
    [np.array([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3]],
    ids=["numpy", "list"],
)
def test_numeric_vector_input_specializes_shape(angles: object) -> None:
    """A numeric vector input determines symbolic loop and register sizes."""

    @qmc.qkernel
    def vector_probe(angles: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
        """Apply one rotation for every supplied angle."""
        reg = qmc.qubit_array(angles.shape[0], "reg")
        for index in qmc.range(angles.shape[0]):
            reg[index] = qmc.rx(reg[index], angles[index])
        return reg

    estimate = vector_probe.estimate_resources(inputs={"angles": angles})

    assert estimate.qubits == 3
    assert estimate.gates.total == 3
    assert estimate.parameters == {}
