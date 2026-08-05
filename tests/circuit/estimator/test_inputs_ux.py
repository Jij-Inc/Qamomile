"""Tests for the contract-aware resource-estimation input UX."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.qkernel_callable import qkernel_callable_attrs
from qamomile.circuit.ir.operation.callable import (
    CallableImplementation,
    CallPolicy,
    CallTransform,
)


class _ShapeGetterRaises:
    """Expose a shape descriptor whose getter fails."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Raise while reading the deliberately malformed shape.

        Raises:
            RuntimeError: Always, to exercise public error translation.
        """
        raise RuntimeError("shape unavailable")


class _ShapeIteratorRaises:
    """Expose a shape object whose iterator fails."""

    @property
    def shape(self) -> object:
        """Return a deliberately malformed iterable shape.

        Returns:
            object: Shape-like object whose iterator raises.
        """

        class _BrokenShape:
            """Fail when shape discovery requests dimensions."""

            def __iter__(self):
                """Raise instead of yielding dimensions.

                Raises:
                    RuntimeError: Always, to exercise public error translation.
                """
                raise RuntimeError("shape iteration unavailable")

        return _BrokenShape()


class _SequenceLengthRaises(list[float]):
    """Expose a sequence whose length provider fails."""

    def __len__(self) -> int:
        """Raise while discovering the deliberately malformed sequence.

        Raises:
            RuntimeError: Always, to exercise public error translation.
        """
        raise RuntimeError("length unavailable")


class _SequenceIteratorRaises(list[float]):
    """Expose a sequence whose element iterator fails."""

    def __iter__(self):
        """Raise while traversing the deliberately malformed sequence.

        Raises:
            RuntimeError: Always, to exercise public error translation.
        """
        raise RuntimeError("sequence iteration unavailable")


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
def _recursive_ignore_predicate(
    depth: qmc.UInt,
    predicate: qmc.Bit,
    target: qmc.Qubit,
) -> qmc.Qubit:
    """Recurse to one gate without ever reading ``predicate``."""
    if depth == 0:
        target = qmc.t(target)
    else:
        target = _recursive_ignore_predicate(depth - 1, predicate, target)
    return target


@qmc.qkernel
def _recursive_ignore_predicate_caller(
    iterations: qmc.UInt,
    depth: qmc.UInt,
) -> qmc.Qubit:
    """Refresh an ignored predicate after each recursive invocation."""
    predicate = qmc.bit(False)
    target = qmc.qubit("target")
    source = qmc.qubit("source")
    for _index in qmc.range(iterations):
        target = _recursive_ignore_predicate(depth, predicate, target)
        predicate = qmc.measure(source)
    return target


@qmc.qkernel
def _recursive_use_predicate(
    depth: qmc.UInt,
    predicate: qmc.Bit,
    target: qmc.Qubit,
) -> qmc.Qubit:
    """Recurse to a base case that branches on ``predicate``."""
    if depth == 0:
        if predicate:
            target = qmc.t(target)
    else:
        target = _recursive_use_predicate(depth - 1, predicate, target)
    return target


@qmc.qkernel
def _recursive_use_predicate_caller(
    iterations: qmc.UInt,
    depth: qmc.UInt,
) -> qmc.Qubit:
    """Refresh a predicate after a recursive invocation that reads it."""
    predicate = qmc.bit(False)
    target = qmc.qubit("target")
    source = qmc.qubit("source")
    for _index in qmc.range(iterations):
        target = _recursive_use_predicate(depth, predicate, target)
        predicate = qmc.measure(source)
    return target


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


def test_typed_branch_inputs_retain_uint_and_bit_domains() -> None:
    """Branch pruning does not erase scalar parameter validation."""

    @qmc.qkernel
    def bit_branch(flag: qmc.Bit) -> qmc.Qubit:
        """Select one of two gates from a classical bit parameter."""
        target = qmc.qubit("target")
        if flag:
            target = qmc.x(target)
        else:
            target = qmc.h(target)
        return target

    with pytest.raises(ValueError, match="non-integer value"):
        _branch_probe.estimate_resources(inputs={"flag": 1.5})
    with pytest.raises(ValueError, match="negative value"):
        _branch_probe.estimate_resources(inputs={"flag": -1})
    with pytest.raises(ValueError, match="upper bound"):
        bit_branch.estimate_resources(inputs={"flag": 2})
    assert bit_branch.estimate_resources(inputs={"flag": True}).gates.total == 1
    assert bit_branch.estimate_resources(inputs={"flag": False}).gates.total == 1
    for boolean in (False, True):
        with pytest.raises(TypeError, match="expects UIntType, got bool"):
            _branch_probe.estimate_resources(inputs={"flag": boolean})


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
    assert str(est.gates.total) == "Piecewise((1, flag > 0), (2, True))"


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
    m = est.parameters["m"]
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
    n = estimate.parameters["n"]

    assert sp.simplify(estimate.gates.total - n) == 0
    assert set(estimate.parameters) == {"n"}


def test_region_arg_drives_later_concrete_loop() -> None:
    """Inputs specialize a carried loop bound."""
    estimate = _carried_loop_bound.estimate_resources(inputs={"n": 3})

    assert estimate.gates.total == 3
    assert estimate.parameters == {}


def test_large_input_keeps_region_loop_symbolic(monkeypatch) -> None:
    """QKernel and raw IR inputs are substituted after loop summarization."""
    from qamomile.circuit.estimator.resource_estimator import ResourceInterpreter

    def fail_concrete_iteration(*args, **kwargs):
        raise AssertionError("estimation input triggered concrete loop execution")

    monkeypatch.setattr(
        ResourceInterpreter,
        "_eval_concrete_region_for",
        fail_concrete_iteration,
    )

    targets = (
        _carried_loop_bound,
        _carried_loop_bound.block,
        _carried_loop_bound.block.operations,
    )
    for target in targets:
        estimate = qmc.estimate_resources(target, inputs={"n": 2048})
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
    bits = default.parameters["bits"]
    assert sp.simplify(default.qubits - (bits + 1)) == 0

    est = _toy_qpe.estimate_resources(inputs={"bits": 5})
    assert int(est.qubits) == 6


def test_qubit_allocation_width_requires_an_integer() -> None:
    """UInt-sized allocations reject fractional direct and later inputs."""
    symbolic = _sized_kernel.estimate_resources()

    with pytest.raises(ValueError, match="non-integer value"):
        symbolic.substitute(n=1.5)
    with pytest.raises(ValueError, match="non-integer value"):
        _sized_kernel.estimate_resources(inputs={"n": 1.5})


def test_inputs_accept_shift_expression() -> None:
    """An input expression may reintroduce the same symbol name (n -> n+1)."""
    n = sp.Symbol("n", integer=True, positive=True)
    est = _sized_kernel.estimate_resources(inputs={"n": n + 1})
    # The reintroduced symbol still sizes the register correctly.
    assert sp.simplify(est.qubits - (n + 1)) == 0


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("8", id="numeric-string"),
        pytest.param("m", id="symbol-string"),
        pytest.param("2*k+1", id="expression-string"),
        pytest.param([1], id="list"),
        pytest.param(None, id="none"),
    ],
)
def test_scalar_resource_parameters_reject_non_numeric_values(value: object) -> None:
    """Only numeric scalars or explicit SymPy input expressions are accepted."""
    symbolic = _sized_kernel.estimate_resources()

    with pytest.raises(TypeError, match=r"requires a .*numeric scalar"):
        _sized_kernel.estimate_resources(inputs={"n": value})
    with pytest.raises(TypeError, match=r"requires a .*numeric scalar"):
        symbolic.substitute(n=value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(complex(1, 1), id="python-complex"),
        pytest.param(float("nan"), id="python-nan"),
        pytest.param(float("inf"), id="python-infinity"),
        pytest.param(sp.I, id="sympy-imaginary"),
        pytest.param(sp.nan, id="sympy-nan"),
        pytest.param(sp.oo, id="sympy-infinity"),
    ],
)
def test_scalar_resource_parameters_require_finite_real_values(
    value: object,
) -> None:
    """Concrete resource scalars reject complex and non-finite values."""
    symbolic = _sized_kernel.estimate_resources()

    with pytest.raises(ValueError, match="finite and real"):
        _sized_kernel.estimate_resources(inputs={"n": value})
    with pytest.raises(ValueError, match="finite and real"):
        symbolic.substitute(n=value)


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
    assert estimate.calls.calls_by_name == {"expval": 1}
    assert estimate.calls.queries_by_name == {"expval": 1}
    assert estimate.derivation is qmc.EstimateDerivation.MODELED
    assert len(estimate.assumptions) == 1


@pytest.mark.parametrize(
    "angles",
    [
        np.array([0.1, 0.2, 0.3]),
        [0.1, 0.2, 0.3],
        (0.1, 0.2, 0.3),
        range(3),
        SimpleNamespace(shape=(3,)),
        SimpleNamespace(shape=(np.int64(3),)),
        SimpleNamespace(shape=(sp.Integer(3),)),
    ],
    ids=[
        "numpy",
        "list",
        "tuple",
        "sequence",
        "shape-attribute",
        "numpy-shape-dimension",
        "sympy-shape-dimension",
    ],
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
    assert estimate.assumptions == ()


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((3.5,), id="fractional"),
        pytest.param((True,), id="boolean"),
        pytest.param((np.bool_(True),), id="numpy-boolean"),
        pytest.param((-1,), id="negative"),
    ],
)
def test_invalid_array_shape_dimensions_are_rejected(shape: tuple[object, ...]) -> None:
    """Resource inputs reject shape entries that are not nonnegative integers."""

    @qmc.qkernel
    def vector_probe(values: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
        """Allocate one qubit for every supplied vector entry.

        Args:
            values (qmc.Vector[qmc.Float]): Vector whose length sets width.

        Returns:
            qmc.Vector[qmc.Qubit]: Register with one qubit per vector entry.
        """
        return qmc.qubit_array(values.shape[0], "reg")

    with pytest.raises(ValueError, match="nonnegative integers"):
        vector_probe.estimate_resources(inputs={"values": SimpleNamespace(shape=shape)})


def test_array_shape_dimension_comparison_cannot_leak_provider_errors() -> None:
    """Resource inputs normalize integer subclasses before sign validation."""

    class HostileDimension(int):
        """Raise if validation compares the provider-owned object directly."""

        def __lt__(self, other: object) -> bool:
            """Reject direct ordering comparisons.

            Args:
                other (object): Right-hand comparison operand.

            Returns:
                bool: This implementation never returns.

            Raises:
                RuntimeError: Always, to expose a direct comparison.
            """
            raise RuntimeError("provider comparison must not escape")

    @qmc.qkernel
    def vector_probe(values: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
        """Allocate one qubit for every supplied vector entry.

        Args:
            values (qmc.Vector[qmc.Float]): Vector whose length sets width.

        Returns:
            qmc.Vector[qmc.Qubit]: Register with one qubit per vector entry.
        """
        return qmc.qubit_array(values.shape[0], "reg")

    estimate = vector_probe.estimate_resources(
        inputs={"values": SimpleNamespace(shape=(HostileDimension(3),))}
    )

    assert estimate.qubits == 3
    with pytest.raises(ValueError, match="nonnegative integers"):
        vector_probe.estimate_resources(
            inputs={"values": SimpleNamespace(shape=(HostileDimension(-1),))}
        )


def test_noniterable_array_shape_is_rejected_as_invalid_input() -> None:
    """Resource inputs translate a malformed scalar shape into ValueError."""

    @qmc.qkernel
    def vector_probe(values: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
        """Allocate one qubit for every supplied vector entry.

        Args:
            values (qmc.Vector[qmc.Float]): Vector whose length sets width.

        Returns:
            qmc.Vector[qmc.Qubit]: Register with one qubit per vector entry.
        """
        return qmc.qubit_array(values.shape[0], "reg")

    with pytest.raises(ValueError, match="shape must be an iterable"):
        vector_probe.estimate_resources(inputs={"values": SimpleNamespace(shape=3)})


@pytest.mark.parametrize(
    ("values", "message"),
    [
        pytest.param(_ShapeGetterRaises(), "Could not read", id="getter"),
        pytest.param(_ShapeIteratorRaises(), "must be an iterable", id="iterator"),
        pytest.param(
            _SequenceLengthRaises([0.0]),
            "sequence length",
            id="sequence-length",
        ),
        pytest.param(
            _SequenceIteratorRaises([0.0]),
            "iterate over the array sequence",
            id="sequence-iterator",
        ),
    ],
)
def test_array_shape_provider_failures_are_translated(
    values: object,
    message: str,
) -> None:
    """Resource inputs expose malformed shape providers as ValueError."""

    @qmc.qkernel
    def vector_probe(data: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
        """Allocate one qubit for every supplied vector entry.

        Args:
            data (qmc.Vector[qmc.Float]): Vector whose length sets width.

        Returns:
            qmc.Vector[qmc.Qubit]: Register with one qubit per vector entry.
        """
        return qmc.qubit_array(data.shape[0], "reg")

    with pytest.raises(ValueError, match=message) as caught:
        vector_probe.estimate_resources(inputs={"data": values})

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_array_shape_provider_resource_failures_propagate() -> None:
    """Resource failures from a shape provider are not mislabeled as input errors."""

    class ExhaustedShapeProvider:
        """Expose a shape descriptor that reports resource exhaustion."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Raise the simulated process resource failure.

            Raises:
                MemoryError: Always, to verify narrow protocol translation.
            """
            raise MemoryError("shape allocation exhausted")

    @qmc.qkernel
    def vector_probe(data: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
        """Allocate one qubit for every supplied vector entry.

        Args:
            data (qmc.Vector[qmc.Float]): Vector whose length sets width.

        Returns:
            qmc.Vector[qmc.Qubit]: Register with one qubit per vector entry.
        """
        return qmc.qubit_array(data.shape[0], "reg")

    with pytest.raises(MemoryError, match="allocation exhausted"):
        vector_probe.estimate_resources(inputs={"data": ExhaustedShapeProvider()})


def test_ragged_matrix_input_is_rejected() -> None:
    """Resource inputs reject ragged nested sequences before specialization."""

    @qmc.qkernel
    def matrix_probe(values: qmc.Matrix[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
        """Allocate one qubit for every supplied matrix row.

        Args:
            values (qmc.Matrix[qmc.Float]): Matrix whose row count sets width.

        Returns:
            qmc.Vector[qmc.Qubit]: Register with one qubit per matrix row.
        """
        return qmc.qubit_array(values.shape[0], "reg")

    with pytest.raises(ValueError, match="must be rectangular"):
        matrix_probe.estimate_resources(inputs={"values": [[0.1], [0.2, 0.3]]})


@pytest.mark.parametrize(
    "scalar",
    [
        pytest.param(0.5, id="python-scalar"),
        pytest.param(np.array(0.5), id="rank-zero-array"),
    ],
)
def test_scalar_input_does_not_bypass_declared_array_rank(
    scalar: object,
) -> None:
    """A discovered rank-zero input is rejected for an array parameter."""

    @qmc.qkernel
    def matrix_probe(values: qmc.Matrix[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
        """Allocate one qubit for every supplied matrix row.

        Args:
            values (qmc.Matrix[qmc.Float]): Matrix whose row count sets width.

        Returns:
            qmc.Vector[qmc.Qubit]: Register with one qubit per matrix row.
        """
        return qmc.qubit_array(values.shape[0], "reg")

    with pytest.raises(ValueError, match=r"has rank 0.*declares rank 2"):
        matrix_probe.estimate_resources(inputs={"values": scalar})


def test_loop_carried_bit_condition_rejects_only_a_real_backedge() -> None:
    """A delayed Bit condition is valid once but rejected when it must carry."""

    @qmc.qkernel
    def delayed_condition(iterations: qmc.UInt) -> qmc.Qubit:
        """Read the previous predicate before refreshing it each iteration."""
        predicate = qmc.bit(False)
        target = qmc.qubit("target")
        source = qmc.qubit("source")
        for _index in qmc.range(iterations):
            if predicate:
                target = qmc.x(target)
            predicate = qmc.measure(source)
        return target

    delayed_condition.estimate_resources(inputs={"iterations": 0})
    delayed_condition.estimate_resources(inputs={"iterations": 1})
    with pytest.raises(NotImplementedError, match="Loop-carried"):
        delayed_condition.estimate_resources(inputs={"iterations": 2})
    with pytest.raises(NotImplementedError, match="Loop-carried"):
        delayed_condition.estimate_resources()


def test_nested_symbolic_trip_count_uses_maximum_for_backedge_proof() -> None:
    """An inner loop is rejected when any enclosing iteration repeats it."""

    @qmc.qkernel
    def varying_inner_trip_count() -> qmc.Qubit:
        """Run inner loops of lengths one and two."""
        target = qmc.qubit("target")
        source = qmc.qubit("source")
        for outer in qmc.range(1, 3):
            predicate = qmc.bit(False)
            for _index in qmc.range(outer):
                if predicate:
                    target = qmc.t(target)
                predicate = qmc.measure(source)
        return target

    with pytest.raises(NotImplementedError, match="Loop-carried"):
        varying_inner_trip_count.estimate_resources()


def test_nested_translation_invariant_single_trip_has_no_backedge() -> None:
    """An affine inner range that is always length one remains valid."""

    @qmc.qkernel
    def invariant_inner_trip_count() -> qmc.Qubit:
        """Run exactly one inner iteration for every outer value."""
        target = qmc.qubit("target")
        source = qmc.qubit("source")
        for outer in qmc.range(1, 3):
            predicate = qmc.bit(False)
            for _index in qmc.range(outer, outer + 1):
                if predicate:
                    target = qmc.t(target)
                predicate = qmc.measure(source)
        return target

    estimate = invariant_inner_trip_count.estimate_resources()

    assert estimate.gates.total == 0
    assert estimate.measurements.total == 2


def test_loop_carried_bit_validation_honors_selected_refresh_merge() -> None:
    """A definite same-iteration refresh is not mistaken for stale state."""

    @qmc.qkernel
    def refreshed_condition(
        iterations: qmc.UInt,
        refresh: qmc.UInt,
    ) -> qmc.Qubit:
        """Optionally refresh a predicate before its only body read."""
        predicate = qmc.bit(False)
        target = qmc.qubit("target")
        source = qmc.qubit("source")
        for _index in qmc.range(iterations):
            if refresh:
                predicate = qmc.measure(source)
            if predicate:
                target = qmc.x(target)
        return target

    disabled = refreshed_condition.estimate_resources(
        inputs={"iterations": 2, "refresh": 0}
    )
    enabled = refreshed_condition.estimate_resources(
        inputs={"iterations": 2, "refresh": 1}
    )

    assert disabled.gates.total == 0
    assert enabled.gates.total == 2


def test_nested_callable_loop_carried_bit_uses_shared_validation() -> None:
    """A selected callee body receives the same loop-state validation as root."""

    @qmc.qkernel
    def delayed_body(iterations: qmc.UInt, target: qmc.Qubit) -> qmc.Qubit:
        """Read a stale predicate inside a nested callable body."""
        predicate = qmc.bit(False)
        source = qmc.qubit("source")
        for _index in qmc.range(iterations):
            if predicate:
                target = qmc.x(target)
            predicate = qmc.measure(source)
        return target

    @qmc.qkernel
    def caller(iterations: qmc.UInt) -> qmc.Qubit:
        """Invoke the delayed body with one fresh target."""
        return delayed_body(iterations, qmc.qubit("target"))

    caller.estimate_resources(inputs={"iterations": 1})
    with pytest.raises(NotImplementedError, match="Loop-carried"):
        caller.estimate_resources(inputs={"iterations": 2})


def test_unused_callee_bit_does_not_create_a_loop_backedge_read() -> None:
    """An unused formal is absent from the inlined validation view."""

    @qmc.qkernel
    def ignore_predicate(predicate: qmc.Bit, target: qmc.Qubit) -> qmc.Qubit:
        """Apply one gate without reading the predicate.

        Args:
            predicate (qmc.Bit): Deliberately unused caller predicate.
            target (qmc.Qubit): Qubit receiving the gate.

        Returns:
            qmc.Qubit: Updated target qubit.
        """
        return qmc.t(target)

    @qmc.qkernel
    def caller(iterations: qmc.UInt) -> qmc.Qubit:
        """Pass a refreshed predicate to a callee that ignores it.

        Args:
            iterations (qmc.UInt): Number of loop iterations.

        Returns:
            qmc.Qubit: Updated target qubit.
        """
        predicate = qmc.bit(False)
        target = qmc.qubit("target")
        source = qmc.qubit("source")
        for _index in qmc.range(iterations):
            target = ignore_predicate(predicate, target)
            predicate = qmc.measure(source)
        return target

    estimate = caller.estimate_resources(inputs={"iterations": 2})

    assert estimate.gates.total == 2


def test_recursive_unused_callee_bit_does_not_create_a_backedge_read() -> None:
    """Concrete recursive inlining removes an unused carried predicate."""
    estimate = _recursive_ignore_predicate_caller.estimate_resources(
        inputs={"iterations": 2, "depth": 2}
    )

    assert estimate.gates.total == 2


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_recursive_used_callee_bit_preserves_backedge_read(depth: int) -> None:
    """Recursive specialization retains a reached predicate read.

    Args:
        depth (int): Concrete recursion depth before the predicate branch.
    """
    with pytest.raises(NotImplementedError, match="Loop-carried"):
        _recursive_use_predicate_caller.estimate_resources(
            inputs={"iterations": 2, "depth": depth}
        )


def test_callee_compile_time_branch_decides_loop_backedge_read() -> None:
    """A bound dead callee branch does not make its predicate look read."""

    @qmc.qkernel
    def optional_predicate(
        predicate: qmc.Bit,
        target: qmc.Qubit,
        enabled: qmc.UInt,
    ) -> qmc.Qubit:
        """Read the predicate only when the compile-time flag is enabled.

        Args:
            predicate (qmc.Bit): Caller predicate used by the optional branch.
            target (qmc.Qubit): Qubit updated by the optional branch.
            enabled (qmc.UInt): Compile-time branch selector.

        Returns:
            qmc.Qubit: Possibly updated target qubit.
        """
        if enabled:
            if predicate:
                target = qmc.t(target)
        return target

    @qmc.qkernel
    def caller(iterations: qmc.UInt, enabled: qmc.UInt) -> qmc.Qubit:
        """Refresh a predicate after forwarding it to the optional helper.

        Args:
            iterations (qmc.UInt): Number of loop iterations.
            enabled (qmc.UInt): Compile-time helper branch selector.

        Returns:
            qmc.Qubit: Possibly updated target qubit.
        """
        predicate = qmc.bit(False)
        target = qmc.qubit("target")
        source = qmc.qubit("source")
        for _index in qmc.range(iterations):
            target = optional_predicate(predicate, target, enabled)
            predicate = qmc.measure(source)
        return target

    disabled = caller.estimate_resources(inputs={"iterations": 2, "enabled": 0})

    assert disabled.gates.total == 0
    with pytest.raises(NotImplementedError, match="Loop-carried"):
        caller.estimate_resources(inputs={"iterations": 2, "enabled": 1})


def test_selected_strategy_can_remove_a_callee_backedge_read() -> None:
    """Validation inlines the same strategy body as resource evaluation."""

    @qmc.qkernel
    def selected_ignores(predicate: qmc.Bit, target: qmc.Qubit) -> qmc.Qubit:
        """Apply one gate without reading the predicate."""
        return qmc.h(target)

    @qmc.qkernel
    def default_reads(predicate: qmc.Bit, target: qmc.Qubit) -> qmc.Qubit:
        """Conditionally apply one gate from the predicate."""
        if predicate:
            target = qmc.t(target)
        return target

    configured = configure_composite(
        default_reads,
        name="strategy_default_reads",
        policy=CallPolicy.INLINE,
        implementations=[
            CallableImplementation(
                transform=CallTransform.DIRECT,
                strategy="selected",
                body=selected_ignores.block,
            )
        ],
    )
    configured = configured._clone_with_callable_attrs(
        {
            **qkernel_callable_attrs(configured),
            "resource_contract": {
                "quantum_operand_widths": [{"index": 0, "name": "target", "width": 1}]
            },
        }
    )

    @qmc.qkernel
    def caller(iterations: qmc.UInt) -> qmc.Qubit:
        """Refresh a predicate after invoking the configured helper."""
        predicate = qmc.bit(False)
        target = qmc.qubit("target")
        source = qmc.qubit("source")
        for _index in qmc.range(iterations):
            target = configured(predicate, target)
            predicate = qmc.measure(source)
        return target

    selected = caller.estimate_resources(
        inputs={"iterations": 2},
        strategies={"strategy_default_reads": "selected"},
    )

    assert selected.gates.total == 2
    with pytest.raises(NotImplementedError, match="Loop-carried"):
        caller.estimate_resources(inputs={"iterations": 2})


def test_selected_strategy_can_introduce_a_callee_backedge_read() -> None:
    """Validation does not keep an input-unused default body by mistake."""

    @qmc.qkernel
    def selected_reads(predicate: qmc.Bit, target: qmc.Qubit) -> qmc.Qubit:
        """Conditionally apply one gate from the predicate."""
        if predicate:
            target = qmc.t(target)
        return target

    @qmc.qkernel
    def default_ignores(predicate: qmc.Bit, target: qmc.Qubit) -> qmc.Qubit:
        """Apply one gate without reading the predicate."""
        return qmc.h(target)

    configured = configure_composite(
        default_ignores,
        name="strategy_default_ignores",
        policy=CallPolicy.INLINE,
        implementations=[
            CallableImplementation(
                transform=CallTransform.DIRECT,
                strategy="selected",
                body=selected_reads.block,
            )
        ],
    )
    configured = configured._clone_with_callable_attrs(
        {
            **qkernel_callable_attrs(configured),
            "resource_contract": {
                "quantum_operand_widths": [{"index": 0, "name": "target", "width": 1}]
            },
        }
    )

    @qmc.qkernel
    def caller(iterations: qmc.UInt) -> qmc.Qubit:
        """Refresh a predicate after invoking the configured helper."""
        predicate = qmc.bit(False)
        target = qmc.qubit("target")
        source = qmc.qubit("source")
        for _index in qmc.range(iterations):
            target = configured(predicate, target)
            predicate = qmc.measure(source)
        return target

    default = caller.estimate_resources(inputs={"iterations": 2})

    assert default.gates.total == 2
    with pytest.raises(NotImplementedError, match="Loop-carried"):
        caller.estimate_resources(
            inputs={"iterations": 2},
            strategies={"strategy_default_ignores": "selected"},
        )


def test_output_only_array_index_is_validated_across_call_boundaries() -> None:
    """A returned element enforces its bounds with or without a helper call."""

    @qmc.qkernel
    def select_element(
        register: qmc.Vector[qmc.Qubit],
        index: qmc.UInt,
    ) -> qmc.Qubit:
        """Return one dynamically selected register element."""
        return register[index]

    contracted = select_element._clone_with_callable_attrs(
        {
            **qkernel_callable_attrs(select_element),
            "resource_contract": {
                "quantum_operand_widths": [{"index": 0, "name": "register", "width": 2}]
            },
        }
    )

    @qmc.qkernel
    def direct(index: qmc.UInt) -> qmc.Qubit:
        """Return a dynamic element directly from the root qkernel."""
        return qmc.qubit_array(2, "register")[index]

    @qmc.qkernel
    def through_helper(index: qmc.UInt) -> qmc.Qubit:
        """Return a dynamic element through an ordinary helper."""
        return select_element(qmc.qubit_array(2, "register"), index)

    @qmc.qkernel
    def through_contract(index: qmc.UInt) -> qmc.Qubit:
        """Return a dynamic element through a width-contracted helper."""
        return contracted(qmc.qubit_array(2, "register"), index)

    for kernel in (direct, through_helper, through_contract):
        valid = kernel.estimate_resources(inputs={"index": 1})
        assert valid.width.peak_qubits == 2
        with pytest.raises(ValueError, match="in-bounds margin"):
            kernel.estimate_resources(inputs={"index": 2})
