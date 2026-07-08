"""Tests for the contract-aware ``substitutions`` estimation UX."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.arithmetic import modmul_const


@qmc.composite_gate(name="phase_u")
@qmc.qkernel
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
def _modmul_kernel(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply one abstract modular multiplication on an n-bit register."""
    reg = qmc.qubit_array(n, name="reg")
    reg = modmul_const(reg)
    return reg


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


@pytest.mark.parametrize("mode", ["bindings", "substitutions"])
def test_branch_specialized_on_concrete_flag(mode: str) -> None:
    """A decidable compile-time branch counts only the taken branch."""
    true_est = _branch_probe.estimate_resources(**{mode: {"flag": 1}})
    false_est = _branch_probe.estimate_resources(**{mode: {"flag": 0}})
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
    assert (
        _branch_probe.estimate_resources(substitutions={"flag": value}).gates.total == 1
    )
    assert (
        _cmp_branch.estimate_resources(substitutions={"n": np.int64(8)}).gates.total
        == 1
    )


def test_branch_undecidable_stays_conservative() -> None:
    """Without a value, an unbound predicate keeps the conservative maximum.

    (The default ``flag=0`` is baked here, so this also confirms the default
    path; the symbolic-max path is covered by the measurement-branch test.)
    """
    est = _branch_probe.estimate_resources()
    assert est.gates.total == 2  # max(1, 2), default flag=0


def test_comparison_branch_specialized() -> None:
    """A comparison predicate is decided from a substituted value."""
    assert _cmp_branch.estimate_resources(substitutions={"n": 8}).gates.total == 1
    assert _cmp_branch.estimate_resources(substitutions={"n": 3}).gates.total == 3


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


def test_partially_supplied_branch_reports_undecidable() -> None:
    """A branch left undecidable by a missing value records why, not a false no-op.

    ``if n > m:`` with only ``n`` supplied cannot be decided, so the estimate
    stays conservative AND records an assumption naming the unresolved ``m`` —
    and must NOT claim the supplied ``n`` "does not affect any resource metric".
    """
    est = _two_param_branch.estimate_resources(substitutions={"n": 8})
    assert est.gates.total == 2  # conservative max(1, 2)
    messages = _messages(est)
    assert any("undecidable" in m and "unresolved: m" in m for m in messages), messages
    assert not any("'n'" in m and "ignored" in m for m in messages)


def test_symbolic_branch_has_no_undecidable_noise() -> None:
    """A fully-symbolic estimate does not emit an undecidable-branch assumption."""
    est = _two_param_branch.estimate_resources()
    assert not any("undecidable" in m for m in _messages(est))


def test_fully_supplied_branch_specializes_without_assumption() -> None:
    """Supplying every predicate operand decides the branch, with no assumption."""
    est = _two_param_branch.estimate_resources(substitutions={"n": 8, "m": 3})
    assert est.gates.total == 1
    assert not any("undecidable" in m for m in _messages(est))


def test_pre_decided_predicate_has_no_undecidable_assumption() -> None:
    """A predicate decided by SymPy assumptions does not report undecidable."""
    est = _positive_branch.estimate_resources(substitutions={"n": 8})
    assert est.gates.total == 1
    assert not any("undecidable" in m for m in _messages(est))


def test_dual_role_symbol_is_consistent() -> None:
    """A symbol driving both a branch and a size specializes both consistently."""
    est5 = _dual_role.estimate_resources(substitutions={"n": 5})
    assert est5.gates.total == 6  # 5 H + 1 (n>3 true branch)
    assert int(est5.qubits) == 6  # 5 reg + 1 extra

    est2 = _dual_role.estimate_resources(substitutions={"n": 2})
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

    taken = alloc_branch.estimate_resources(substitutions={"flag": 1})
    # Only |q> is live on the true branch; the false-branch ancilla is gone.
    assert int(taken.qubits) == 1


def test_substitutions_pass_kernel_inputs_including_noop_angle() -> None:
    """Passing all kernel inputs works; an angle that affects no metric is a no-op.

    ``theta`` never appears in any resource expression, so substituting it is a
    recorded no-op rather than an error — the user simply passed the kernel's
    declared inputs.
    """
    est = _toy_qpe.estimate_resources(substitutions={"bits": 5, "theta": 0.25})

    assert int(est.qubits) == 6  # 5 counting + 1 target
    ignored = [a.message for a in est.assumptions if "ignored" in a.message]
    assert any("theta" in message for message in ignored)


def test_substitutions_force_symbolic_over_python_default() -> None:
    """A substitution overrides a Python-signature default instead of baking it.

    ``bits`` has default 4; ``substitutions={"bits": 5}`` must estimate 5, not
    silently use the default (which would build a 4-bit circuit and ignore the
    request).
    """
    # Without substitutions, the Python default (bits=4) is baked -> 5 qubits.
    default = _toy_qpe.estimate_resources()
    assert int(default.qubits) == 5

    # substitutions={"bits": 5} overrides the default -> 6 qubits (not 5).
    est = _toy_qpe.estimate_resources(substitutions={"bits": 5})
    assert int(est.qubits) == 6


def test_substitutions_accept_shift_expression() -> None:
    """A substitution value may reintroduce the same symbol name (n -> n+1)."""
    n = sp.Symbol("n", integer=True, positive=True)
    est = _modmul_kernel.estimate_resources(substitutions={"n": n + 1})
    # One abstract modmul on an (n+1)-bit register: peak width n+1.
    assert sp.simplify(est.qubits - (n + 1)) == 0


def test_substitutions_typo_raises() -> None:
    """A name that is neither a free symbol nor a kernel argument raises."""
    with pytest.raises(ValueError, match="neither free symbols"):
        _toy_qpe.estimate_resources(substitutions={"bti": 5})


def test_substitutions_bindings_overlap_raises() -> None:
    """A name in both bindings and substitutions is rejected."""
    with pytest.raises(ValueError, match="both bindings and substitutions"):
        _modmul_kernel.estimate_resources(bindings={"n": 4}, substitutions={"n": 4})
