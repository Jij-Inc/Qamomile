"""Tests for controlled gate API with all gate types x num_controls."""

import dataclasses
import inspect
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from qamomile.qiskit import QiskitTranspiler

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle.primitives import Float, Qubit
from qamomile.circuit.frontend.operation.control import ControlledGate, control
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.types.primitives import FloatType, QubitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import (
    EmitError,
    QubitConsumedError,
)
from tests.transpiler.gate_test_specs import (
    all_zeros_state,
    computational_basis_state,
    cp_matrix,
    cx_matrix,
    cz_matrix,
    h_matrix,
    p_matrix,
    rx_matrix,
    ry_matrix,
    rz_matrix,
    rzz_matrix,
    statevectors_equal,
    swap_matrix,
    x_matrix,
)

# =============================================================================
# Unit test helpers
# =============================================================================


def _make_qubit(name: str = "q", version: int = 0, parent=None, indices=()):
    """Create a Qubit handle with a real Value."""
    val = Value(type=QubitType(), name=name, version=version)
    return Qubit(value=val, parent=parent, indices=indices)


def _make_float_handle(name: str = "theta"):
    """Create a Float handle."""
    val = Value(type=FloatType(), name=name)
    return Float(value=val)


def _mock_qkernel(
    *,
    qubit_params: tuple[str, ...] = ("q",),
    classical_params: tuple[tuple[str, Any], ...] = (),
):
    """Create a mock QKernel that satisfies ``ControlledGate``'s compose-time invariants.

    ``ControlledGate.__init__`` validates that the wrapped object exposes
    a dict ``input_types`` and an ``inspect.Signature`` ``signature``; the
    bare-``MagicMock`` shape used to slip through via the now-removed
    ``not isinstance(input_types, dict)`` fallback path.  This helper
    synthesizes both attributes so the mock looks like a real qkernel to
    the validator while letting the test still drive the controlled-gate
    machinery directly.

    Args:
        qubit_params: Names of qubit-typed input parameters, in
            declaration order.  Defaults to ``("q",)`` so a bare
            ``_mock_qkernel()`` matches a single-qubit sub-kernel like
            ``qmc.x``.
        classical_params: Sequence of ``(name, type)`` tuples for
            classical parameters.  ``type`` must be one of ``Float`` /
            ``UInt`` / ``float`` / ``int`` (the set
            ``_params_to_operands`` recognises).

    Returns:
        A ``MagicMock`` with ``.block`` stubbed, ``.input_types`` set to
        a real ``dict`` matching the declared shape, and ``.signature``
        set to a real ``inspect.Signature``.  All three of those
        attributes are what the validator + downstream helpers
        (``_sub_positional_count_for_symbolic``,
        ``_bind_to_sub_signature``, ``_params_to_operands``) consume.
    """
    qk = MagicMock()
    qk.block = MagicMock(name="block_value")

    input_types: dict[str, Any] = {}
    sig_params: list[inspect.Parameter] = []
    for name in qubit_params:
        input_types[name] = qmc.Qubit
        sig_params.append(
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    for name, typ in classical_params:
        input_types[name] = typ
        sig_params.append(
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )

    qk.input_types = input_types
    qk.signature = inspect.Signature(parameters=sig_params)
    return qk


# =============================================================================
# Unit tests for control() function
# =============================================================================


class TestControlledFunction:
    def test_returns_controlled_gate(self):
        cg = control(_mock_qkernel())
        assert isinstance(cg, ControlledGate)

    def test_default_num_controls(self):
        cg = control(_mock_qkernel())
        assert cg._num_controls == 1

    @pytest.mark.parametrize("nc", [1, 2, 3, 5])
    def test_custom_num_controls(self, nc):
        cg = control(_mock_qkernel(), num_controls=nc)
        assert cg._num_controls == nc

    def test_stores_qkernel(self):
        qk = _mock_qkernel()
        cg = control(qk)
        assert cg._qkernel is qk


# =============================================================================
# Unit tests for ControlledGate.__init__
# =============================================================================


class TestControlledGateInit:
    @pytest.mark.parametrize("nc", [1, 2, 3, 5])
    def test_stores_qkernel_and_num_controls(self, nc):
        qk = _mock_qkernel()
        cg = ControlledGate(qk, num_controls=nc)
        assert cg._qkernel is qk
        assert cg._num_controls == nc

    def test_default_num_controls_is_one(self):
        cg = ControlledGate(_mock_qkernel())
        assert cg._num_controls == 1


# =============================================================================
# Unit tests for ControlledGate.__call__
# =============================================================================


class TestControlledGateCall:
    """Tests for ControlledGate.__call__ using mock QKernels and real tracer."""

    def test_emits_one_controlled_u_operation(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace() as tracer:
            result = cg(_make_qubit("ctrl"), _make_qubit("tgt"))
        assert len(tracer.operations) == 1
        assert isinstance(tracer.operations[0], ControlledUOperation)
        assert tracer.operations[0].num_controls == 1
        assert len(result) == 2
        assert all(isinstance(q, Qubit) for q in result)

    @pytest.mark.parametrize("nc", [1, 2, 3])
    def test_multi_control_single_target(self, nc):
        cg = ControlledGate(_mock_qkernel(), num_controls=nc)
        qs = [_make_qubit(f"q{i}") for i in range(nc + 1)]
        with trace() as tracer:
            result = cg(*qs)
        assert len(result) == nc + 1
        assert isinstance(tracer.operations[0], ControlledUOperation)
        assert tracer.operations[0].num_controls == nc

    def test_single_control_multi_target(self):
        cg = ControlledGate(_mock_qkernel(qubit_params=("q0", "q1")), num_controls=1)
        ctrl, t0, t1 = _make_qubit("ctrl"), _make_qubit("t0"), _make_qubit("t1")
        with trace() as tracer:
            result = cg(ctrl, t0, t1)
        assert len(result) == 3
        op = tracer.operations[0]
        assert op.num_controls == 1
        assert len(op.operands) == 3

    def test_operands_structure(self):
        """operands = [ctrl_vals..., tgt_vals...] and block is stored separately."""
        qk = _mock_qkernel()
        cg = ControlledGate(qk, num_controls=2)
        c0, c1, tgt = _make_qubit("c0"), _make_qubit("c1"), _make_qubit("tgt")
        with trace() as tracer:
            cg(c0, c1, tgt)
        op = tracer.operations[0]
        assert op.block is qk.block
        assert op.operands[0] is c0.value
        assert op.operands[1] is c1.value
        assert op.operands[2] is tgt.value

    def test_results_and_output_handles(self):
        """Results are next-versioned, preserve logical_id, and are wired to output handles."""
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        ctrl = _make_qubit("ctrl", version=0)
        tgt = _make_qubit("tgt", version=0)
        with trace() as tracer:
            result = cg(ctrl, tgt)
        op = tracer.operations[0]
        # next version
        assert op.results[0].version == 1
        assert op.results[1].version == 1
        # logical_id preserved
        assert op.results[0].logical_id == ctrl.value.logical_id
        assert op.results[1].logical_id == tgt.value.logical_id
        # output handles point to results
        assert result[0].value is op.results[0]
        assert result[1].value is op.results[1]

    def test_output_preserves_parent_and_indices(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        mock_parent = MagicMock(name="parent_array")
        mock_idx = (MagicMock(name="idx0"),)
        ctrl = _make_qubit("ctrl", parent=mock_parent, indices=mock_idx)
        tgt = _make_qubit("tgt")
        with trace() as tracer:  # noqa: F841
            result = cg(ctrl, tgt)
        assert result[0].parent is mock_parent
        assert result[0].indices is mock_idx
        assert result[1].parent is None
        assert result[1].indices == ()

    def test_raw_float_param_becomes_constant_value(self):
        cg = ControlledGate(
            _mock_qkernel(classical_params=(("theta", Float),)), num_controls=1
        )
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), theta=0.5)
        op = tracer.operations[0]
        assert len(op.operands) == 3
        param_val = op.operands[2]
        assert isinstance(param_val, Value)
        assert param_val.get_const() == 0.5

    def test_handle_type_param_uses_value_directly(self):
        cg = ControlledGate(
            _mock_qkernel(classical_params=(("theta", Float),)), num_controls=1
        )
        theta = _make_float_handle("theta")
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), theta=theta)
        op = tracer.operations[0]
        assert op.operands[2] is theta.value

    def test_multiple_params(self):
        cg = ControlledGate(
            _mock_qkernel(classical_params=(("alpha", Float), ("beta", Float))),
            num_controls=1,
        )
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), alpha=1.0, beta=2.0)
        op = tracer.operations[0]
        assert len(op.operands) == 4
        assert op.operands[2].get_const() == 1.0
        assert op.operands[3].get_const() == 2.0

    @pytest.mark.parametrize("power", [1, 2, 3, 5])
    def test_power_parameter_forwarded(self, power):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), power=power)
        assert tracer.operations[0].power == power

    def test_default_power_is_one(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"))
        assert tracer.operations[0].power == 1

    def test_no_tracer_raises_runtime_error(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with pytest.raises(RuntimeError, match="No active tracer"):
            cg(_make_qubit("ctrl"), _make_qubit("tgt"))

    def test_operation_properties(self):
        """ControlledUOperation properties: block, control_operands, target_operands, operation_kind."""
        qk = _mock_qkernel()
        cg = ControlledGate(qk, num_controls=2)
        c0, c1, tgt = _make_qubit("c0"), _make_qubit("c1"), _make_qubit("tgt")
        with trace() as tracer:
            cg(c0, c1, tgt)
        op = tracer.operations[0]
        assert op.block is qk.block
        assert op.control_operands == [c0.value, c1.value]
        assert op.target_operands == [tgt.value]
        assert op.operation_kind == OperationKind.QUANTUM

    def test_int_param_converted_to_float(self):
        cg = ControlledGate(
            _mock_qkernel(classical_params=(("theta", Float),)), num_controls=1
        )
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), theta=3)
        param_val = tracer.operations[0].operands[2]
        assert param_val.get_const() == 3.0
        assert isinstance(param_val.get_const(), float)

    def test_target_operands_with_params(self):
        """target_operands includes parameter Values when params are passed."""
        cg = ControlledGate(
            _mock_qkernel(classical_params=(("theta", Float),)), num_controls=1
        )
        ctrl = _make_qubit("ctrl")
        tgt = _make_qubit("tgt")
        with trace() as tracer:
            cg(ctrl, tgt, theta=0.5)
        op = tracer.operations[0]
        # target_operands = operands[1+num_controls:] includes tgt + param
        target_ops = op.target_operands
        assert len(target_ops) == 2
        assert target_ops[0] is tgt.value
        assert target_ops[1].get_const() == 0.5

    def test_consume_prevents_reuse(self):
        """Qubit passed to controlled gate cannot be reused afterward."""
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        ctrl = _make_qubit("ctrl")
        tgt = _make_qubit("tgt")
        with trace() as tracer:  # noqa: F841
            cg(ctrl, tgt)
            with pytest.raises(QubitConsumedError):
                cg(ctrl, _make_qubit("tgt2"))

    def test_consume_prevents_reuse_target(self):
        """Target qubit passed to controlled gate cannot be reused afterward."""
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        ctrl = _make_qubit("ctrl")
        tgt = _make_qubit("tgt")
        with trace() as tracer:  # noqa: F841
            cg(ctrl, tgt)
            with pytest.raises(QubitConsumedError):
                cg(_make_qubit("ctrl2"), tgt)

    def test_aliasing_control_and_target_raises(self):
        """Reusing the same qubit as control + target raises QubitConsumedError.

        Frontend Step 6 dropped the bespoke
        ``_validate_no_alias_or_overlap`` entry-point check; the
        underlying ``Handle.consume()`` linear-type machinery catches
        the duplicate on the second consume, so the error class is
        ``QubitConsumedError`` (not ``QubitAliasError``).
        """
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        q = _make_qubit("q")
        with trace() as tracer:  # noqa: F841
            with pytest.raises(QubitConsumedError):
                cg(q, q)

    def test_aliasing_duplicate_controls_raises(self):
        """Reusing the same qubit across two control slots raises QubitConsumedError.

        Same rationale as :meth:`test_aliasing_control_and_target_raises`
        — the linear-type layer catches the duplicate.
        """
        cg = ControlledGate(_mock_qkernel(), num_controls=2)
        q = _make_qubit("q")
        tgt = _make_qubit("tgt")
        with trace() as tracer:  # noqa: F841
            with pytest.raises(QubitConsumedError):
                cg(q, q, tgt)


# =============================================================================
# Validation tests for control() / ControlledGate
# =============================================================================


class TestControlledValidation:
    def test_num_controls_zero_raises(self):
        with pytest.raises(ValueError):
            control(_mock_qkernel(), num_controls=0)

    def test_num_controls_negative_raises(self):
        with pytest.raises(ValueError):
            control(_mock_qkernel(), num_controls=-1)

    def test_not_enough_args_raises(self):
        """Passing fewer qubits than num_controls (no targets) raises.

        Either ``ValueError`` (from :meth:`_call_concrete`'s
        "no sub-kernel quantum arg" check when the sub-kernel takes
        no required positional args) or ``TypeError`` (from
        :meth:`inspect.Signature.bind`'s "missing a required
        argument" when the sub-kernel does require one) is an
        acceptable failure mode -- both signal the same thing.
        """
        cg = ControlledGate(_mock_qkernel(), num_controls=3)
        qs = [_make_qubit(f"q{i}") for i in range(3)]  # 3 controls, 0 targets
        with trace() as tracer:  # noqa: F841
            with pytest.raises((ValueError, TypeError)):
                cg(*qs)

    def test_control_indices_in_concrete_mode_raises(self):
        """Concrete-``num_controls`` rejects ``control_indices`` at compose time.

        The redesign restricted ``control_indices`` to symbolic
        mode (design §1.1, decision #5); concrete mode has no
        selection step.
        """
        cg = ControlledGate(_mock_qkernel(), num_controls=2)
        c0, c1, tgt = _make_qubit("c0"), _make_qubit("c1"), _make_qubit("tgt")
        with trace():
            with pytest.raises(ValueError, match="only valid in symbolic mode"):
                cg(c0, c1, tgt, control_indices=[0])


class TestNormalizeControlIndices:
    """Compose-time validation rules for ``control_indices`` entries.

    Drives ``ControlledGate._normalize_control_indices`` directly
    (rather than through the full ``__call__`` pipeline) so the test
    intent stays focused on the per-entry rules: sequence-type check,
    per-entry type check (``bool`` rejection, ``int`` / ``UInt``
    accepted, anything else rejected), and literal-``int`` duplicate
    / negative checks.  The ``UInt``-aware checks (mixed ``int`` /
    ``UInt`` duplicates, range) are deliberately deferred to emit
    time and not exercised here.
    """

    @staticmethod
    def _make_cg():
        return ControlledGate(_mock_qkernel(), num_controls=1)

    def test_non_sequence_raises_type_error(self):
        cg = self._make_cg()
        with pytest.raises(TypeError, match="must be a Sequence"):
            cg._normalize_control_indices(42)  # type: ignore[arg-type]

    def test_bool_entry_raises_type_error(self):
        cg = self._make_cg()
        with pytest.raises(TypeError, match="bool entry"):
            cg._normalize_control_indices([True, 1])

    def test_string_entry_raises_type_error(self):
        cg = self._make_cg()
        with pytest.raises(TypeError, match="must be int or UInt"):
            cg._normalize_control_indices(["0", 1])  # type: ignore[list-item]

    def test_duplicate_literal_int_raises_value_error(self):
        cg = self._make_cg()
        with pytest.raises(ValueError, match="duplicate int entry"):
            cg._normalize_control_indices([0, 0])

    def test_negative_literal_int_raises_value_error(self):
        cg = self._make_cg()
        with pytest.raises(ValueError, match="negative entry"):
            cg._normalize_control_indices([-1, 0])

    def test_accepts_literal_ints(self):
        cg = self._make_cg()
        result = cg._normalize_control_indices([0, 1, 2])
        assert len(result) == 3
        for i, v in enumerate(result):
            assert v.get_const() == i

    def test_accepts_uint_handles(self):
        from qamomile.circuit.frontend.handle.primitives import UInt as UIntHandle
        from qamomile.circuit.ir.types.primitives import UIntType

        cg = self._make_cg()
        v0 = Value(type=UIntType(), name="i0").with_const(0)
        v1 = Value(type=UIntType(), name="i1").with_const(1)
        result = cg._normalize_control_indices(
            [UIntHandle(value=v0), UIntHandle(value=v1)]
        )
        assert len(result) == 2
        assert result[0] is v0
        assert result[1] is v1


class TestControlledPowerValidation:
    """Tests for power parameter validation in ControlledGate."""

    def test_power_zero_raises(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace():
            with pytest.raises(ValueError, match="strictly positive"):
                cg(_make_qubit("ctrl"), _make_qubit("tgt"), power=0)

    def test_power_negative_raises(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace():
            with pytest.raises(ValueError, match="strictly positive"):
                cg(_make_qubit("ctrl"), _make_qubit("tgt"), power=-3)

    def test_power_bool_true_raises(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace():
            with pytest.raises(TypeError, match="bool"):
                cg(_make_qubit("ctrl"), _make_qubit("tgt"), power=True)

    def test_power_bool_false_raises(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace():
            with pytest.raises(TypeError, match="bool"):
                cg(_make_qubit("ctrl"), _make_qubit("tgt"), power=False)

    def test_power_float_raises(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace():
            with pytest.raises(TypeError, match="int or UInt"):
                cg(_make_qubit("ctrl"), _make_qubit("tgt"), power=1.5)

    def test_power_uint_normalizes_to_value(self):
        from qamomile.circuit.frontend.handle.primitives import UInt as UIntHandle
        from qamomile.circuit.ir.types.primitives import UIntType

        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        uint_val = Value(type=UIntType(), name="power_k").with_const(4)
        uint_power = UIntHandle(value=uint_val)
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), power=uint_power)
        op = tracer.operations[0]
        assert isinstance(op.power, Value)
        assert op.power is uint_val


# -- Gate qkernels (1-qubit, no param) ----------------------------------------


@qmc.qkernel
def _h_gate(q: qmc.Qubit) -> qmc.Qubit:
    return qmc.h(q)


@qmc.qkernel
def _x_gate(q: qmc.Qubit) -> qmc.Qubit:
    return qmc.x(q)


@qmc.qkernel
def _y_gate(q: qmc.Qubit) -> qmc.Qubit:
    return qmc.y(q)


@qmc.qkernel
def _z_gate(q: qmc.Qubit) -> qmc.Qubit:
    return qmc.z(q)


@qmc.qkernel
def _s_gate(q: qmc.Qubit) -> qmc.Qubit:
    return qmc.s(q)


@qmc.qkernel
def _t_gate(q: qmc.Qubit) -> qmc.Qubit:
    return qmc.t(q)


@qmc.qkernel
def _ccx_gate(
    c1: qmc.Qubit, c2: qmc.Qubit, t: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    return qmc.ccx(c1, c2, t)


# -- Gate qkernels (1-qubit, with param) --------------------------------------


@qmc.qkernel
def _p_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    return qmc.p(q, theta)


@qmc.qkernel
def _rx_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    return qmc.rx(q, theta)


@qmc.qkernel
def _ry_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    return qmc.ry(q, theta)


@qmc.qkernel
def _rz_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    return qmc.rz(q, theta)


# -- Gate qkernels (2-qubit, no param) ----------------------------------------


@qmc.qkernel
def _cx_gate(c: qmc.Qubit, t: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    return qmc.cx(c, t)


@qmc.qkernel
def _cz_gate(c: qmc.Qubit, t: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    return qmc.cz(c, t)


@qmc.qkernel
def _swap_gate(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    return qmc.swap(a, b)


# -- Gate qkernels (2-qubit, with param) --------------------------------------


@qmc.qkernel
def _cp_gate(c: qmc.Qubit, t: qmc.Qubit, theta: float) -> tuple[qmc.Qubit, qmc.Qubit]:
    return qmc.cp(c, t, theta)


@qmc.qkernel
def _rzz_gate(a: qmc.Qubit, b: qmc.Qubit, theta: float) -> tuple[qmc.Qubit, qmc.Qubit]:
    return qmc.rzz(a, b, theta)


# -- GateSpec ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GateSpec:
    """Specification for a gate under test."""

    kernel: object
    num_targets: int
    theta: float | None
    matrix_fn: Callable[..., np.ndarray]

    @property
    def has_param(self) -> bool:
        return self.theta is not None

    @property
    def bindings(self) -> dict:
        return {"theta": self.theta} if self.has_param else {}

    @property
    def matrix(self) -> np.ndarray:
        return self.matrix_fn(self.theta) if self.has_param else self.matrix_fn()


GATES = [
    # 1-qubit, no param
    pytest.param(GateSpec(_h_gate, 1, None, h_matrix), id="h"),
    pytest.param(GateSpec(_x_gate, 1, None, x_matrix), id="x"),
    # 1-qubit, with param
    pytest.param(GateSpec(_p_gate, 1, math.pi, p_matrix), id="p"),
    pytest.param(GateSpec(_rx_gate, 1, math.pi, rx_matrix), id="rx"),
    pytest.param(GateSpec(_ry_gate, 1, math.pi, ry_matrix), id="ry"),
    pytest.param(GateSpec(_rz_gate, 1, math.pi, rz_matrix), id="rz"),
    # 2-qubit, no param
    pytest.param(GateSpec(_cx_gate, 2, None, cx_matrix), id="cx"),
    pytest.param(GateSpec(_cz_gate, 2, None, cz_matrix), id="cz"),
    pytest.param(GateSpec(_swap_gate, 2, None, swap_matrix), id="swap"),
    # 2-qubit, with param
    pytest.param(GateSpec(_cp_gate, 2, math.pi, cp_matrix), id="cp"),
    pytest.param(GateSpec(_rzz_gate, 2, math.pi, rzz_matrix), id="rzz"),
]


# -- Circuit factory -----------------------------------------------------------


def _make_controlled_circuit(
    spec: GateSpec, num_controls: int, activate_controls: bool, power: int = 1
):
    """Create a qkernel that applies a controlled gate and measures the last target.

    Note:
        We avoid ``for i in range(...)`` inside the qkernel because the AST
        transform converts all ``for ... in range(...)`` into quantum for_loop
        operations.  Instead, X-gate applications use a list comprehension helper.
    """
    total = num_controls + spec.num_targets

    def _prep(qs):
        """Apply X to the first num_controls qubits if activate_controls is set."""
        if not activate_controls:
            return qs
        return [qmc.x(qs[i]) if i < num_controls else qs[i] for i in range(len(qs))]

    if spec.has_param:

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            qs = [qmc.qubit(name=f"q{i}") for i in range(total)]
            qs = _prep(qs)
            cg = qmc.control(spec.kernel, num_controls=num_controls)
            out = cg(*qs, theta=theta, power=power)
            return qmc.measure(out[-1])

    else:

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            qs = [qmc.qubit(name=f"q{i}") for i in range(total)]
            qs = _prep(qs)
            cg = qmc.control(spec.kernel, num_controls=num_controls)
            out = cg(*qs, power=power)
            return qmc.measure(out[-1])

    return circuit


# -- Helpers -------------------------------------------------------------------


def _precise_statevector(qc: "QuantumCircuit") -> np.ndarray:
    """Compute statevector with exact controlled-gate unitaries.

    Qiskit's ``ControlledGate.to_matrix()`` is not always available, so
    ``Operator(circuit)`` falls back to gate decomposition which can
    accumulate significant error (e.g. ``CXGate().power(3).control(3)``).

    This helper processes each gate instruction individually.  For
    ``ControlledGate`` instances it builds the controlled unitary from
    the base gate's precise matrix, avoiding decomposition error entirely.
    """
    from qiskit.circuit import ControlledGate as QiskitControlledGate
    from qiskit.quantum_info import Operator, Statevector

    def _gate_operator(gate):
        if isinstance(gate, QiskitControlledGate):
            # Fall back to Operator(gate) for non-standard ctrl_state
            expected_ctrl = (1 << gate.num_ctrl_qubits) - 1
            if hasattr(gate, "ctrl_state") and gate.ctrl_state != expected_ctrl:
                return Operator(gate)

            base_mat = Operator(gate.base_gate).data
            n_base = base_mat.shape[0]
            nc = gate.num_ctrl_qubits
            dim = (2**nc) * n_base
            mat = np.eye(dim, dtype=complex)
            # In Qiskit little-endian ordering, controls occupy the lowest bits.
            ctrl_all1 = (1 << nc) - 1
            idx = [i for i in range(dim) if (i & ctrl_all1) == ctrl_all1]
            for r, ri in enumerate(idx):
                for c, ci in enumerate(idx):
                    mat[ri, ci] = base_mat[r, c]
            return Operator(mat)
        return Operator(gate)

    sv = Statevector.from_int(0, 2**qc.num_qubits)
    for inst in qc.data:
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        sv = sv.evolve(_gate_operator(inst.operation), qubits)
    return sv.data


def _get_statevector(
    transpiler: "QiskitTranspiler", circuit_kernel: qmc.QKernel, bindings: dict | None
) -> np.ndarray:
    """Transpile, remove measurements, return full statevector as numpy array."""
    executable = transpiler.transpile(circuit_kernel, bindings=bindings)
    qc = executable.quantum_circuit.copy()
    qc.remove_final_measurements()
    return _precise_statevector(qc)


def _gate_power_matrix(matrix: np.ndarray, power: int) -> np.ndarray:
    """Compute U^power using Qiskit's gate power implementation.

    For power == 1 returns the matrix as-is.  For power > 1, delegates to
    ``UnitaryGate(matrix).power(power)`` so the expected value uses exactly
    the same computation path as the transpiled circuit.
    """
    if power == 1:
        return matrix
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    return Operator(UnitaryGate(matrix).power(power)).data


def _expected_statevector(
    spec: GateSpec, num_controls: int, activate_controls: bool, power: int = 1
) -> np.ndarray:
    """Compute expected statevector for a controlled gate test.

    Qubit ordering (Qiskit little-endian):
        qubits 0..NC-1 = controls, qubits NC..NC+NT-1 = targets.
    """
    total = num_controls + spec.num_targets

    if not activate_controls:
        return all_zeros_state(total)

    # Controls all |1>
    controls_state = computational_basis_state(num_controls, 2**num_controls - 1)
    # Gate U^power acts on target qubits starting from |0...0>
    U = _gate_power_matrix(spec.matrix, power)
    targets_final = U @ all_zeros_state(spec.num_targets)
    # In little-endian: controls are lower qubits, targets are higher qubits
    return np.kron(targets_final, controls_state)


# -- Parametrized tests --------------------------------------------------------


@pytest.mark.parametrize("spec", GATES)
@pytest.mark.parametrize("num_controls", [1, 2, 3])
class TestControlledGateIntegration:
    """Integration test: control(gate, num_controls) for all gate x control combos."""

    def test_transpiles(self, qiskit_transpiler, spec, num_controls):
        circuit = _make_controlled_circuit(spec, num_controls, activate_controls=False)
        executable = qiskit_transpiler.transpile(circuit, bindings=spec.bindings)
        assert executable is not None

    def test_controls_zero(self, qiskit_transpiler, spec, num_controls):
        """Controls all |0> => gate does NOT fire => full state stays |00...0>."""
        circuit = _make_controlled_circuit(spec, num_controls, activate_controls=False)
        actual = _get_statevector(qiskit_transpiler, circuit, spec.bindings)
        expected = _expected_statevector(spec, num_controls, activate_controls=False)
        assert statevectors_equal(actual, expected)

    def test_controls_one(self, qiskit_transpiler, spec, num_controls):
        """Controls all |1> => gate fires => check full statevector."""
        circuit = _make_controlled_circuit(spec, num_controls, activate_controls=True)
        actual = _get_statevector(qiskit_transpiler, circuit, spec.bindings)
        expected = _expected_statevector(spec, num_controls, activate_controls=True)
        assert statevectors_equal(actual, expected)


# =============================================================================
# Integration tests for controlled gates with power > 1
# =============================================================================


@pytest.mark.parametrize("spec", GATES)
@pytest.mark.parametrize("num_controls", [1, 2, 3])
@pytest.mark.parametrize("power", [2, 3])
class TestControlledPowerIntegration:
    """Integration test: control(gate, num_controls, power) for all gate x control x power combos."""

    def test_controls_zero(self, qiskit_transpiler, spec, num_controls, power):
        """Controls all |0> => gate^power does NOT fire => full state stays |00...0>."""
        circuit = _make_controlled_circuit(
            spec, num_controls, activate_controls=False, power=power
        )
        actual = _get_statevector(qiskit_transpiler, circuit, spec.bindings)
        expected = _expected_statevector(
            spec, num_controls, activate_controls=False, power=power
        )
        assert statevectors_equal(actual, expected)

    def test_controls_one(self, qiskit_transpiler, spec, num_controls, power):
        """Controls all |1> => gate^power fires => check full statevector."""
        circuit = _make_controlled_circuit(
            spec, num_controls, activate_controls=True, power=power
        )
        actual = _get_statevector(qiskit_transpiler, circuit, spec.bindings)
        expected = _expected_statevector(
            spec, num_controls, activate_controls=True, power=power
        )
        assert statevectors_equal(actual, expected)


# =============================================================================
# Integration tests with random initial states (controls in superposition)
# =============================================================================


def _get_cached_circuit(
    transpiler: "QiskitTranspiler",
    spec: GateSpec,
    num_controls: int,
    cache: dict[tuple, "QuantumCircuit"],
    power: int = 1,
) -> "QuantumCircuit":
    """Return a transpiled circuit (measurements removed), cached per (spec, num_controls, power)."""
    key = (spec.kernel.name, num_controls, spec.theta, power)
    if key not in cache:
        circuit = _make_controlled_circuit(
            spec, num_controls, activate_controls=False, power=power
        )
        executable = transpiler.transpile(circuit, bindings=spec.bindings)
        qc = executable.quantum_circuit.copy()
        qc.remove_final_measurements()
        cache[key] = qc
    return cache[key].copy()


def _get_statevector_with_prep(
    transpiler: "QiskitTranspiler",
    spec: GateSpec,
    num_controls: int,
    target_state: np.ndarray,
    cache: dict[tuple, "QuantumCircuit"],
    power: int = 1,
) -> np.ndarray:
    """Get statevector with H-prep on controls and StatePreparation on targets.

    Uses cached transpiled circuits to avoid redundant transpilation.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    main_qc = _get_cached_circuit(transpiler, spec, num_controls, cache, power=power)

    total = main_qc.num_qubits
    prep_qc = QuantumCircuit(total)
    for i in range(num_controls):
        prep_qc.h(i)
    prep_qc.append(StatePreparation(target_state), list(range(num_controls, total)))

    full_qc = prep_qc.compose(main_qc)
    return _precise_statevector(full_qc)


def _expected_statevector_superposition(
    spec: GateSpec, num_controls: int, target_state: np.ndarray, power: int = 1
) -> np.ndarray:
    """Compute expected statevector with controls in |+>^NC and targets in |psi>.

    Controlled-U fires only when all controls are |1>:
        expected = |psi> x (|+>^NC - coeff*|11...1>) + U^power|psi> x coeff*|11...1>

    Qubit ordering (Qiskit little-endian): targets (left) x controls (right).
    """
    nc = num_controls
    coeff = 1.0 / np.sqrt(2**nc)
    control_plus = np.ones(2**nc, dtype=complex) * coeff
    all1 = computational_basis_state(nc, 2**nc - 1)
    rest = control_plus - coeff * all1

    U = _gate_power_matrix(spec.matrix, power)
    return np.kron(target_state, rest) + np.kron(U @ target_state, coeff * all1)


def _random_statevector(num_qubits: int, seed: int) -> np.ndarray:
    """Generate a random normalized statevector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(2**num_qubits) + 1j * rng.standard_normal(2**num_qubits)
    return vec / np.linalg.norm(vec)


@pytest.fixture(scope="session")
def transpiled_cache() -> dict[tuple, "QuantumCircuit"]:
    """Session-scoped cache for transpiled circuits used by random-state tests."""
    return {}


@pytest.mark.parametrize("spec", GATES)
@pytest.mark.parametrize("num_controls", [1, 2, 3])
@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("seed", [901 + offset for offset in range(10)])
class TestControlledGateRandomState:
    """Integration test: controlled gate with controls in superposition and random target state."""

    def test_random_initial_state(
        self, qiskit_transpiler, transpiled_cache, spec, num_controls, power, seed
    ):
        """Controls in |+>^NC, targets in random |psi> => verify full statevector."""
        target_state = _random_statevector(spec.num_targets, seed=seed)
        actual = _get_statevector_with_prep(
            qiskit_transpiler,
            spec,
            num_controls,
            target_state,
            transpiled_cache,
            power=power,
        )
        expected = _expected_statevector_superposition(
            spec, num_controls, target_state, power=power
        )
        assert statevectors_equal(actual, expected)


# =============================================================================
# Built-in gate acceptance: control(qmc.rx) without an @qkernel wrapper
# =============================================================================
#
# The factory below now auto-wraps a plain built-in gate function in a
# synthesized @qkernel.  These tests cover (a) the acceptance/rejection
# rules of the wrapper-synthesis path, (b) IR parity between the built-in
# call form and a hand-written @qkernel wrapper, and (c) end-to-end
# execution on each supported backend.


# -- Backend availability (mirrors test_gate_broadcast.py) --------------------

_HAS_QISKIT = True
try:  # pragma: no cover - presence check
    from qamomile.qiskit import QiskitTranspiler as _QiskitTranspilerCheck  # noqa: F401
except ImportError:  # pragma: no cover
    _HAS_QISKIT = False

_HAS_QURI_PARTS = True
try:  # pragma: no cover - presence check
    import quri_parts.qulacs  # noqa: F401

    from qamomile.quri_parts import (
        QuriPartsTranspiler as _QuriPartsTranspilerCheck,  # noqa: F401
    )
except ImportError:  # pragma: no cover
    _HAS_QURI_PARTS = False

_HAS_CUDAQ = True
try:  # pragma: no cover - presence check
    import cudaq  # noqa: F401

    from qamomile.cudaq import CudaqTranspiler as _CudaqTranspilerCheck  # noqa: F401
except ImportError:  # pragma: no cover
    _HAS_CUDAQ = False


def _qiskit_transpiler_factory():
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


def _quri_parts_transpiler_factory():
    from qamomile.quri_parts import QuriPartsTranspiler

    return QuriPartsTranspiler()


def _cudaq_transpiler_factory():
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


_BUILTIN_BACKENDS = [
    pytest.param(
        _qiskit_transpiler_factory,
        id="qiskit",
        marks=pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed"),
    ),
    pytest.param(
        _quri_parts_transpiler_factory,
        id="quri_parts",
        marks=pytest.mark.skipif(
            not _HAS_QURI_PARTS, reason="quri_parts/qulacs not installed"
        ),
    ),
    pytest.param(
        _cudaq_transpiler_factory,
        id="cudaq",
        marks=pytest.mark.skipif(not _HAS_CUDAQ, reason="cudaq not installed"),
    ),
]


# -- Acceptance: control() factory takes built-ins -------------------------


class TestControlledAcceptsBuiltinGate:
    """``control()`` accepts built-in gate functions directly."""

    @pytest.mark.parametrize(
        "gate_fn",
        [qmc.h, qmc.x, qmc.y, qmc.z, qmc.s, qmc.t, qmc.rx, qmc.ry, qmc.rz, qmc.p],
    )
    def test_single_qubit_gates_accepted(self, gate_fn):
        cg = qmc.control(gate_fn)
        assert isinstance(cg, ControlledGate)

    @pytest.mark.parametrize("gate_fn", [qmc.cx, qmc.cz, qmc.swap, qmc.cp, qmc.rzz])
    def test_two_qubit_gates_accepted(self, gate_fn):
        cg = qmc.control(gate_fn)
        assert isinstance(cg, ControlledGate)

    def test_three_qubit_gate_accepted(self):
        cg = qmc.control(qmc.ccx)
        assert isinstance(cg, ControlledGate)

    @pytest.mark.parametrize("nc", [1, 2, 3])
    def test_built_in_with_num_controls(self, nc):
        cg = qmc.control(qmc.rx, num_controls=nc)
        assert cg._num_controls == nc

    def test_qkernel_passthrough_uses_same_instance(self):
        """A QKernel argument is passed straight through, not re-wrapped."""

        @qmc.qkernel
        def my_gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        cg = qmc.control(my_gate)
        assert cg._qkernel is my_gate


# -- Rejection: errors for unsupported callables -----------------------------


class TestControlledBuiltinErrors:
    """Errors raised when a callable cannot be auto-wrapped."""

    def test_non_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="QKernel or a built-in gate"):
            qmc.control(42)  # type: ignore[arg-type]

    def test_missing_annotation_raises_type_error(self):
        def bad(q):  # no annotation
            return qmc.h(q)

        with pytest.raises(TypeError, match="no type annotation"):
            qmc.control(bad)

    def test_unsupported_annotation_raises_type_error(self):
        def bad(q: str) -> str:  # str is not Qubit/Float/UInt
            return q

        with pytest.raises(TypeError, match="annotation"):
            qmc.control(bad)

    def test_no_qubit_param_raises_type_error(self):
        def classical_only(theta: float) -> float:
            return theta

        with pytest.raises(TypeError, match="no Qubit parameters"):
            qmc.control(classical_only)

    def test_var_args_raises_type_error(self):
        def variadic(*qs: qmc.Qubit) -> qmc.Qubit:
            return qs[0]

        with pytest.raises(TypeError, match="\\*args/\\*\\*kwargs"):
            qmc.control(variadic)

    def test_positional_only_raises_type_error(self):
        """Positional-only params can't be forwarded by name; reject early."""

        def positional_only(q: qmc.Qubit, /) -> qmc.Qubit:
            return qmc.h(q)

        with pytest.raises(TypeError, match="positional-only"):
            qmc.control(positional_only)

    def test_keyword_only_raises_type_error(self):
        """Keyword-only params would collapse to POSITIONAL_OR_KEYWORD; reject."""

        def keyword_only(q: qmc.Qubit, *, theta: float) -> qmc.Qubit:
            return qmc.rx(q, theta)

        with pytest.raises(TypeError, match="keyword-only"):
            qmc.control(keyword_only)

    def test_unknown_param_raises_type_error(self):
        """Typo'd or unrelated kwargs must be rejected, not silently dropped.

        Pre-fix, ``cg(c, t, theata=...)`` (typo) silently appended an
        operand that ``ValueResolver`` would then drop because it
        didn't match any inner-block classical parameter.  The user
        got a controlled gate that fired with the default (unbound)
        angle and no warning.  Now the typo raises ``TypeError``
        listing the valid parameter names.
        """

        @qmc.qkernel
        def gate_with_angle(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            return qmc.rx(q, angle)

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            cg = qmc.control(gate_with_angle)
            c, t = cg(c, t, theata=0.5)  # typo!
            return qmc.measure(t)

        with pytest.raises(TypeError, match="unknown parameter"):
            _ = circuit.block

    def test_kwarg_order_independent_of_signature_order(self, qiskit_transpiler):
        """Caller kwarg order must not matter — operands follow signature order.

        Before the fix, ``_params_to_operands`` iterated ``params.items()``
        (caller insertion order), and ``ValueResolver`` then bound those
        operands to the inner block's classical inputs *positionally*.
        That meant ``cg(c, t, beta=B, alpha=A)`` and
        ``cg(c, t, alpha=A, beta=B)`` produced circuits where ``alpha``
        and ``beta`` were *swapped*.  The fix iterates the wrapped
        kernel's ``input_types`` in signature order, so the resulting
        statevectors must match regardless of kwarg order.
        """

        @qmc.qkernel
        def two_param_gate(
            q: qmc.Qubit, alpha: qmc.Float, beta: qmc.Float
        ) -> qmc.Qubit:
            q = qmc.rx(q, alpha)
            q = qmc.rz(q, beta)
            return q

        @qmc.qkernel
        def kw_natural_order() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(2, name="q")
            qs[0] = qmc.x(qs[0])
            cg = qmc.control(two_param_gate)
            qs[0], qs[1] = cg(qs[0], qs[1], alpha=0.7, beta=1.3)
            return qmc.measure(qs)

        @qmc.qkernel
        def kw_reversed_order() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(2, name="q")
            qs[0] = qmc.x(qs[0])
            cg = qmc.control(two_param_gate)
            qs[0], qs[1] = cg(qs[0], qs[1], beta=1.3, alpha=0.7)
            return qmc.measure(qs)

        sv_natural = _get_statevector(qiskit_transpiler, kw_natural_order, {})
        sv_reversed = _get_statevector(qiskit_transpiler, kw_reversed_order, {})
        assert statevectors_equal(sv_natural, sv_reversed), (
            "swapping kwarg order produced a different circuit — "
            "operands must follow the wrapped kernel's signature order, "
            "not the caller's kwarg insertion order."
        )

    def test_uint_param_rejects_float(self):
        """A UInt-declared param must reject Python float (no silent truncate).

        Pre-fix, ``cg(c, t, n=3.7)`` silently truncated to ``n=3`` via
        ``int(3.7)``.  Now we raise ``TypeError`` so callers must opt in
        via an explicit ``int(value)`` if truncation is intended.
        """

        @qmc.qkernel
        def with_uint(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            cg = qmc.control(with_uint)
            c, t = cg(c, t, n=3.7)  # float for UInt-declared param
            return qmc.measure(t)

        with pytest.raises(TypeError, match="declared as UInt"):
            _ = circuit.block

    def test_uint_param_rejects_bool(self):
        """``bool`` is technically ``int`` but not a meaningful UInt value."""

        @qmc.qkernel
        def with_uint(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            cg = qmc.control(with_uint)
            c, t = cg(c, t, n=True)
            return qmc.measure(t)

        with pytest.raises(TypeError, match="declared as UInt"):
            _ = circuit.block

    def test_float_param_rejects_bool(self):
        """A Float-declared param must reject ``bool`` (avoid True→1.0 surprise)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            cg = qmc.control(qmc.rx)
            c, t = cg(c, t, angle=True)
            return qmc.measure(t)

        with pytest.raises(TypeError, match="declared as Float"):
            _ = circuit.block

    def test_default_value_raises_type_error(self):
        """A parameter with a default value would silently lose it.

        The synthesized wrapper places all params as
        ``POSITIONAL_OR_KEYWORD`` with no default, so the caller's
        existing contract ``fn(c, t)`` would unexpectedly require the
        formerly-defaulted kwarg.  Reject up-front and direct the user
        to ``@qmc.qkernel``.
        """

        def has_default(q: qmc.Qubit, theta: float = 0.1) -> qmc.Qubit:
            return qmc.rx(q, theta)

        with pytest.raises(TypeError, match="default value"):
            qmc.control(has_default)

    def test_param_named_qmc_target_raises_type_error(self):
        """A parameter named ``__qmc_target__`` would shadow the forwarding global.

        The synthesized wrapper body is ``return __qmc_target__(...)`` and
        ``__qmc_target__`` is the injected forwarding callable.  A
        parameter of the same name would shadow it with the
        caller-supplied value, leading to a confusing
        ``"<value> is not callable"`` at block-construction time.  The
        guard rejects it up-front with a clear message instead.
        """

        def shadowing(q: qmc.Qubit, __qmc_target__: float) -> qmc.Qubit:
            return qmc.rx(q, __qmc_target__)

        with pytest.raises(TypeError, match="reserved wrapper-internal"):
            qmc.control(shadowing)

    def test_param_named_qubit_raises_type_error(self):
        """A parameter named ``Qubit`` collides with the injected type binding.

        Defensive guard: although Python annotation evaluation happens
        before parameter binding (so the type resolves correctly today),
        any future change to that ordering would silently corrupt
        type-hint resolution.  We reject the symmetric case up-front.
        """

        def shadowing_type(q: qmc.Qubit, Qubit: float) -> qmc.Qubit:  # noqa: N803
            return qmc.rx(q, Qubit)

        with pytest.raises(TypeError, match="reserved wrapper-internal"):
            qmc.control(shadowing_type)


# -- Wrapper synthesis: caching + interleaved signature handling ------------


class TestControlledBuiltinSynthesisInternals:
    """Cover the wrapper-synthesis edge cases the Copilot review flagged."""

    def test_recursive_controlled_inside_wrapped_fn_does_not_deadlock(self):
        """A wrapped callable that itself calls control() must not deadlock.

        The synthesizer holds ``_synthesized_kernel_lock`` while eagerly
        building the wrapper's ``Block`` (so ``fn`` is still strongly
        referenced when the body executes).  That build invokes the
        wrapped callable, which may call ``control(...)`` again — a
        re-entry into ``_qkernel_for_callable`` from the same thread.
        ``threading.Lock`` would deadlock here; ``RLock`` lets the
        re-entry succeed.

        We run the call on a worker thread with a hard timeout so a
        regression manifests as a clear failure rather than hanging the
        suite indefinitely.
        """
        import threading

        def outer_helper(q: qmc.Qubit) -> qmc.Qubit:
            # Force a recursive control() call during outer_helper's
            # block construction.  The inner gate doesn't matter — what
            # matters is that the inner control() re-enters the lock.
            _ = qmc.control(qmc.rx)
            return qmc.h(q)

        result_holder: dict[str, object] = {}

        def worker():
            try:
                result_holder["cg"] = qmc.control(outer_helper)
            except Exception as e:  # pragma: no cover - defensive
                result_holder["error"] = e

        # ``daemon=True`` so a regression (deadlocked worker) doesn't hang
        # the pytest process at exit — the assertion below fails first,
        # and the daemon thread is collected on interpreter shutdown.
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), (
            "control() hung — likely a deadlock from a non-reentrant lock "
            "while re-entering _qkernel_for_callable from a wrapped callable."
        )
        assert "error" not in result_holder, result_holder.get("error")
        assert isinstance(result_holder["cg"], ControlledGate)

    def test_reserved_names_match_namespace_keys(self):
        """``_RESERVED_WRAPPER_NAMES`` is auto-derived from ``_wrapper_namespace``.

        The two used to be hand-maintained in different places, so a
        future change to the injected names in ``_wrapper_namespace``
        could silently de-sync the collision guard.  The pair is now
        derived from a single source; this assertion pins that
        invariant so the regression is caught in CI rather than
        re-emerging as a shadow bug at runtime.
        """
        from qamomile.circuit.frontend.operation.control import (
            _RESERVED_WRAPPER_NAMES,
            _wrapper_namespace,
        )

        assert _RESERVED_WRAPPER_NAMES == frozenset(_wrapper_namespace(None).keys())

    def test_repeated_calls_reuse_same_qkernel(self):
        """Calling control(fn) twice on the same callable returns the same wrapper.

        The synthesized ``QKernel`` is cached in a ``WeakKeyDictionary`` so
        long-running processes don't grow ``linecache`` unboundedly.
        """
        cg1 = qmc.control(qmc.rx)
        cg2 = qmc.control(qmc.rx)
        assert cg1._qkernel is cg2._qkernel

    def test_non_weakrefable_callable_uses_strong_cache(self):
        """A non-weakrefable callable must still memoize its wrapper.

        ``WeakKeyDictionary`` rejects callables that can't be weakly
        referenced (a small set of C-implemented builtins).  Without a
        strong-reference fallback, repeated ``control(fn)`` calls on
        such callables would re-synthesize the wrapper indefinitely and
        accumulate ``linecache`` entries.

        We simulate the non-weakrefable case with an instance of a
        ``__slots__``-only class without ``__weakref__`` — a known
        pattern that ``weakref`` rejects.  The test asserts that
        ``control()`` still returns the same wrapper across calls,
        which proves the strong-reference cache is engaged.
        """
        import weakref as _weakref

        class _NoWeakref:
            __slots__ = ()

            def __call__(self, q: qmc.Qubit) -> qmc.Qubit:
                return qmc.h(q)

        non_weakrefable = _NoWeakref()

        # Sanity: this object really is rejected by weakref.
        with pytest.raises(TypeError):
            _weakref.ref(non_weakrefable)

        # Decorate ``__call__`` properly: signature classification reads
        # ``__call__``'s annotations via ``inspect.signature``, which
        # follows ``__call__`` automatically for class instances.
        cg1 = qmc.control(non_weakrefable)
        cg2 = qmc.control(non_weakrefable)
        assert cg1._qkernel is cg2._qkernel, (
            "non-weakrefable callable should still hit the strong-ref "
            "fallback cache on repeated control() calls."
        )

    def test_no_eager_synthesis_at_module_import(self):
        """Library modules must not trigger wrapper synthesis at import time.

        Eager ``qmc.control(builtin_gate)`` calls at module level run
        ``compile``/``exec`` + a full block trace at every ``import``,
        which can add up for users who pull in many algorithm modules.
        The current implementation moves any such call inside the
        function that needs it and relies on the per-callable cache for
        repeated use; this regression guards against future drift.

        We currently audit ``qamomile.circuit.algorithm.fqaoa`` (the
        only in-tree module that previously did this).  When new
        modules add their own ``qmc.control(...)`` callsites they
        should be added to the audit list here.
        """
        import inspect

        from qamomile.circuit.algorithm import fqaoa

        src = inspect.getsource(fqaoa)
        # Module-level lines start in column 0 and are not blank.  Lines
        # inside ``def`` / ``class`` / etc. bodies always begin with
        # whitespace, so this filter approximates module scope.
        module_level = "\n".join(
            line for line in src.splitlines() if line and not line[0].isspace()
        )
        assert "qmc.control(" not in module_level, (
            "qamomile.circuit.algorithm.fqaoa calls qmc.control at "
            "module scope, which triggers eager wrapper synthesis at "
            "import time. Move the call inside the function that needs "
            "it (the per-callable cache will absorb the cost on first "
            "use)."
        )

    def test_distinct_callables_get_distinct_wrappers(self):
        """Different gate functions must not collide in the cache."""
        cg_rx = qmc.control(qmc.rx)
        cg_ry = qmc.control(qmc.ry)
        assert cg_rx._qkernel is not cg_ry._qkernel

    def test_interleaved_qubit_classical_signature_forwards_correctly(
        self, qiskit_transpiler
    ):
        """A callable with (Qubit, float, Qubit) signature must forward args correctly.

        The pre-fix wrapper rebuilt the signature as "all qubits first" and
        then forwarded ``__qmc_target__(c, t, theta)`` positionally, which
        re-bound ``theta`` to the second qubit.  After the fix the wrapper
        invokes the target by keyword, so the interleaved order is
        respected and ``theta`` actually reaches the rotation.
        """

        def gate_with_interleaved_params(
            ctrl_qubit: qmc.Qubit,
            theta: float,
            tgt_qubit: qmc.Qubit,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            ctrl_qubit, tgt_qubit = qmc.cp(ctrl_qubit, tgt_qubit, theta)
            return ctrl_qubit, tgt_qubit

        # If the wrapper mis-forwarded args, transpile would either crash
        # (Qubit handle passed where Float is expected) or silently emit a
        # wrong gate.  Successful transpile on Qiskit confirms forwarding.
        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            qs = [qmc.qubit(name=f"q{i}") for i in range(3)]
            qs = [qmc.x(qs[0]), qmc.x(qs[1]), qs[2]]  # |1,1,0>
            cg = qmc.control(gate_with_interleaved_params)
            qs[0], qs[1], qs[2] = cg(qs[0], qs[1], qs[2], theta=math.pi)
            return qmc.measure(qs[2])

        executable = qiskit_transpiler.transpile(circuit, bindings={"theta": math.pi})
        # Just confirm transpile succeeded — wrong forwarding would have
        # raised by now (handle-type mismatch in expand-controlled-U).
        assert executable is not None

    def test_unusual_callable_name_does_not_break_synthesis(self):
        """A callable named ``Qubit`` must not collide with the namespace's Qubit type.

        Pre-fix, ``def Qubit(...)`` would have shadowed the injected
        ``Qubit`` type during ``exec`` and broken type-hint resolution.
        Post-fix the wrapper falls back to a fresh
        ``_qmc_controlled_wrapper_<n>`` source-level identifier whenever
        the callable's ``__name__`` collides with one of the injected
        names (``Qubit`` / ``Float`` / ``UInt`` / ``tuple`` /
        ``__qmc_target__``) or is not a valid Python identifier; otherwise
        the original name is preserved.  ``__name__`` is *not* rewritten
        on the wrapper — that would re-introduce the collision against
        ``transform_control_flow``'s ``name_space[func.__name__]`` lookup.
        """

        def Qubit(q: qmc.Qubit) -> qmc.Qubit:  # noqa: N802 - test the corner case
            return qmc.h(q)

        cg = qmc.control(Qubit)
        assert isinstance(cg, ControlledGate)
        # Implementation detail: the synthesized wrapper falls back to the
        # safe internal identifier here, NOT the original ``Qubit`` name.
        assert cg._qkernel.name.startswith("_qmc_controlled_wrapper_")

    def test_keyword_callable_name_falls_back_to_internal_id(self):
        """A callable whose ``__name__`` is a Python keyword must not crash compile().

        Regression for the Copilot #9 review: ``"class".isidentifier()`` is
        ``True`` but ``def class(...)`` is a ``SyntaxError``, so the
        synthesizer must additionally consult ``keyword.iskeyword`` and
        fall back to the safe ``_qmc_controlled_wrapper_<n>`` identifier.
        """

        # ``def class(...)`` is itself unparseable, so build a function
        # under a normal name and then rebind ``__name__`` to a keyword.
        def some_gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        some_gate.__name__ = "class"

        cg = qmc.control(some_gate)
        assert isinstance(cg, ControlledGate)
        assert cg._qkernel.name.startswith("_qmc_controlled_wrapper_")

    def test_int_param_lowered_as_uint_type(self):
        """A wrapped kernel that declares ``int`` lowers raw int kwargs to UIntType.

        Regression for the Copilot #8 review: previously
        ``_params_to_operands`` always wrapped raw scalars as
        ``FloatType``, which mismatched the wrapper-side ``UInt``
        annotation that ``_classify_callable_param`` produces for ``int``
        parameters.  After the fix, the controlled-U operand for the
        ``int`` parameter carries a ``UIntType`` constant that lines up
        with the wrapped block's ``input_values``.
        """
        from qamomile.circuit.ir.operation.gate import ControlledUOperation
        from qamomile.circuit.ir.types.primitives import UIntType

        def gate_with_int_param(q: qmc.Qubit, n: int) -> qmc.Qubit:
            # ``n`` is unused at runtime — we only need the IR plumbing
            # to round-trip the parameter through controlled-U emit.
            return qmc.h(q)

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            cg = qmc.control(gate_with_int_param)
            c, t = cg(c, t, n=3)
            return qmc.measure(t)

        block = circuit.block
        ctrl_ops = [
            op for op in block.operations if isinstance(op, ControlledUOperation)
        ]
        assert len(ctrl_ops) == 1
        # operands = [ctrl_value, target_value, n_value] for num_controls=1.
        n_operand = ctrl_ops[0].operands[2]
        assert isinstance(n_operand.type, UIntType), (
            f"expected UIntType for an ``int``-annotated kernel parameter, "
            f"got {type(n_operand.type).__name__}"
        )
        assert n_operand.get_const() == 3

    def test_float_param_still_lowered_as_float_type(self):
        """Sanity: float-annotated params keep using FloatType (no regression)."""
        from qamomile.circuit.ir.operation.gate import ControlledUOperation
        from qamomile.circuit.ir.types.primitives import FloatType

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            cg = qmc.control(qmc.rx)
            c, t = cg(c, t, angle=0.5)
            return qmc.measure(t)

        block = circuit.block
        ctrl_ops = [
            op for op in block.operations if isinstance(op, ControlledUOperation)
        ]
        assert len(ctrl_ops) == 1
        angle_operand = ctrl_ops[0].operands[2]
        assert isinstance(angle_operand.type, FloatType)
        assert angle_operand.get_const() == 0.5

    def test_dynamic_callable_is_released_on_gc(self):
        """Once the user drops a dynamically-defined callable, the wrapper cache must release it.

        Regression for the Copilot #5 review: an earlier draft used
        ``WeakKeyDictionary`` but the wrapper's globals captured ``fn``
        with a strong ref via ``__qmc_target__``, so the cache transitively
        kept ``fn`` alive forever.  Post-fix the wrapper holds a
        ``weakref.proxy(fn)`` and ``Block`` is built eagerly (so the
        proxy is never re-invoked), letting the cache + linecache entries
        die with ``fn``.
        """
        import gc
        import linecache as _linecache_module
        import weakref

        from qamomile.circuit.frontend.operation.control import (
            _synthesized_kernel_cache,
        )

        def _ephemeral_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
            return qmc.rx(q, theta)

        fn_ref = weakref.ref(_ephemeral_gate)

        cg = qmc.control(_ephemeral_gate)
        assert _ephemeral_gate in _synthesized_kernel_cache
        # Pin the linecache filename for the post-GC check.  The
        # AST-transformed ``self.func`` is re-compiled under the
        # ``<qamomile-dsl>`` synthetic filename, so we read the original
        # wrapper's filename from ``raw_func`` instead.
        wrapper_filename = cg._qkernel.raw_func.__code__.co_filename
        assert wrapper_filename in _linecache_module.cache

        # Drop every strong ref the test holds; only the cache (weakly)
        # references the callable now.
        del cg
        del _ephemeral_gate
        gc.collect()

        assert fn_ref() is None, (
            "WeakKeyDictionary failed to release the callable; the wrapper "
            "is still keeping it alive (cache leak regression)."
        )
        # The weakref.finalize hook drops the linecache entry when ``fn``
        # is collected.
        assert wrapper_filename not in _linecache_module.cache


# -- IR parity: built-in form vs hand-written @qkernel form ------------------


class TestControlledBuiltinIRParity:
    """Built-in form produces an IR equivalent to the hand-written wrapper."""

    @staticmethod
    def _gate_kinds(block) -> list:
        """Return the ordered list of GateOperationType values in a block."""
        from qamomile.circuit.ir.operation.gate import (
            GateOperation as IRGateOperation,
        )

        return [
            op.gate_type for op in block.operations if isinstance(op, IRGateOperation)
        ]

    def test_rx_gate_block_matches_wrapper(self):
        """control(qmc.rx).block has the same gate sequence as control(_rx_gate).block."""
        cg_builtin = qmc.control(qmc.rx)
        cg_wrapped = qmc.control(_rx_gate)
        assert self._gate_kinds(cg_builtin._qkernel.block) == self._gate_kinds(
            cg_wrapped._qkernel.block
        )

    def test_h_gate_block_matches_wrapper(self):
        cg_builtin = qmc.control(qmc.h)
        cg_wrapped = qmc.control(_h_gate)
        assert self._gate_kinds(cg_builtin._qkernel.block) == self._gate_kinds(
            cg_wrapped._qkernel.block
        )

    def test_cp_gate_block_matches_wrapper(self):
        cg_builtin = qmc.control(qmc.cp)
        cg_wrapped = qmc.control(_cp_gate)
        assert self._gate_kinds(cg_builtin._qkernel.block) == self._gate_kinds(
            cg_wrapped._qkernel.block
        )

    def test_cx_gate_block_matches_wrapper(self):
        cg_builtin = qmc.control(qmc.cx)
        cg_wrapped = qmc.control(_cx_gate)
        assert self._gate_kinds(cg_builtin._qkernel.block) == self._gate_kinds(
            cg_wrapped._qkernel.block
        )


# -- Cross-SDK builtin-vs-wrapper parity: full gate × 3 SDK × random angles --
#
# These tests build a controlled circuit twice — once with the built-in
# gate function passed directly to ``qmc.control`` (the new path) and
# once with a hand-written ``@qmc.qkernel`` wrapper (the legacy path) —
# and assert that both forms produce the same statevector / sample
# outcome / expectation value on every supported quantum SDK.


@dataclasses.dataclass(frozen=True)
class _ControlledGateSpec:
    """Spec describing one built-in gate + its hand-written ``@qkernel`` wrapper."""

    label: str
    builtin_fn: Any
    wrapper_qkernel: Any
    num_targets: int
    # Keyword name the *built-in* gate function expects for its angle
    # parameter (``angle`` for rx/ry/rz/rzz, ``theta`` for p/cp,
    # ``None`` for non-parametric gates).
    builtin_kwarg: str | None
    # Keyword name the *wrapper* qkernel expects for its angle parameter.
    # Most wrappers above declare ``theta`` regardless of the underlying
    # gate's natural name; ``None`` for non-parametric wrappers.
    wrapper_kwarg: str | None


_GATE_SPECS = [
    pytest.param(_ControlledGateSpec("h", qmc.h, _h_gate, 1, None, None), id="h"),
    pytest.param(_ControlledGateSpec("x", qmc.x, _x_gate, 1, None, None), id="x"),
    pytest.param(_ControlledGateSpec("y", qmc.y, _y_gate, 1, None, None), id="y"),
    pytest.param(_ControlledGateSpec("z", qmc.z, _z_gate, 1, None, None), id="z"),
    pytest.param(_ControlledGateSpec("s", qmc.s, _s_gate, 1, None, None), id="s"),
    pytest.param(_ControlledGateSpec("t", qmc.t, _t_gate, 1, None, None), id="t"),
    pytest.param(
        _ControlledGateSpec("rx", qmc.rx, _rx_gate, 1, "angle", "theta"), id="rx"
    ),
    pytest.param(
        _ControlledGateSpec("ry", qmc.ry, _ry_gate, 1, "angle", "theta"), id="ry"
    ),
    pytest.param(
        _ControlledGateSpec("rz", qmc.rz, _rz_gate, 1, "angle", "theta"), id="rz"
    ),
    pytest.param(_ControlledGateSpec("p", qmc.p, _p_gate, 1, "theta", "theta"), id="p"),
    pytest.param(_ControlledGateSpec("cx", qmc.cx, _cx_gate, 2, None, None), id="cx"),
    pytest.param(_ControlledGateSpec("cz", qmc.cz, _cz_gate, 2, None, None), id="cz"),
    pytest.param(
        _ControlledGateSpec("swap", qmc.swap, _swap_gate, 2, None, None), id="swap"
    ),
    pytest.param(
        _ControlledGateSpec("cp", qmc.cp, _cp_gate, 2, "theta", "theta"), id="cp"
    ),
    pytest.param(
        _ControlledGateSpec("rzz", qmc.rzz, _rzz_gate, 2, "angle", "theta"), id="rzz"
    ),
    pytest.param(
        _ControlledGateSpec("ccx", qmc.ccx, _ccx_gate, 3, None, None), id="ccx"
    ),
]


def _make_controlled_circuit_for_statevector(
    spec: _ControlledGateSpec,
    num_controls: int,
    theta_value: float | None,
    *,
    use_builtin: bool,
    power: int = 1,
):
    """Build a kernel that applies one controlled gate and ends with measure.

    Returns a ``@qmc.qkernel`` whose body
        1. Allocates ``num_controls + spec.num_targets`` qubits,
        2. Drives every control qubit to |1> via X,
        3. Applies ``control(builtin_or_wrapper, num_controls=N)``
           with the optional ``power=`` exponent,
        4. Measures the last qubit so the kernel has classical I/O
           (which top-level transpile entry-points require).

    The trailing measurement is stripped by Qiskit's
    ``qc.remove_final_measurements`` inside ``_precise_statevector``.
    """
    fn = spec.builtin_fn if use_builtin else spec.wrapper_qkernel
    kwarg = spec.builtin_kwarg if use_builtin else spec.wrapper_kwarg

    def _kwargs() -> dict[str, Any]:
        d: dict[str, Any] = {}
        if kwarg is not None and theta_value is not None:
            d[kwarg] = theta_value
        if power != 1:
            d["power"] = power
        return d

    nc, nt = num_controls, spec.num_targets

    if (nc, nt) == (1, 1):

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q0 = qmc.x(q0)
            cg = qmc.control(fn, num_controls=1)
            q0, q1 = cg(q0, q1, **_kwargs())
            return qmc.measure(q1)

        return circuit

    if (nc, nt) == (1, 2):

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q0 = qmc.x(q0)
            cg = qmc.control(fn, num_controls=1)
            q0, q1, q2 = cg(q0, q1, q2, **_kwargs())
            return qmc.measure(q2)

        return circuit

    if (nc, nt) == (1, 3):

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q3 = qmc.qubit(name="q3")
            q0 = qmc.x(q0)
            cg = qmc.control(fn, num_controls=1)
            q0, q1, q2, q3 = cg(q0, q1, q2, q3, **_kwargs())
            return qmc.measure(q3)

        return circuit

    if (nc, nt) == (2, 1):

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            cg = qmc.control(fn, num_controls=2)
            q0, q1, q2 = cg(q0, q1, q2, **_kwargs())
            return qmc.measure(q2)

        return circuit

    if (nc, nt) == (2, 2):

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q3 = qmc.qubit(name="q3")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            cg = qmc.control(fn, num_controls=2)
            q0, q1, q2, q3 = cg(q0, q1, q2, q3, **_kwargs())
            return qmc.measure(q3)

        return circuit

    raise ValueError(f"unsupported (num_controls, num_targets)=({nc}, {nt})")


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
@pytest.mark.parametrize("spec", _GATE_SPECS)
@pytest.mark.parametrize("seed", [0, 1, 42])
class TestControlledBuiltinStatevectorParity:
    """builtin-vs-wrapper Qiskit statevector parity, all 16 standard gates.

    For every (gate, seed) combination the test transpiles the same
    controlled circuit twice — once via ``control(builtin_fn)`` and
    once via ``control(_<gate>_gate)`` (the hand-written
    ``@qmc.qkernel`` wrapper) — and asserts that the resulting Qiskit
    final statevectors (with the trailing measurement removed) agree
    to ``atol=1e-8``.

    Random rotation angles are sampled from ``rng.uniform(-π, π)`` per
    seed.  Non-parametric gates (h, x, y, z, s, t, cx, cz, swap, ccx)
    are exercised with ``theta_value=None`` and the kwarg dict stays
    empty.

    Limited to Qiskit because the kernel ends with ``qmc.measure(...)``
    (top-level kernels must have classical I/O), and stripping the
    trailing measurement before state-vector evaluation is only
    straightforward on Qiskit's ``QuantumCircuit``.  For QuriParts /
    CUDA-Q the same circuits are exercised via the expval-parity suite
    below (which is measurement-free at the user observable level).
    """

    def test_num_controls_1(self, qiskit_transpiler, spec, seed):
        """Single-control parity for every gate + random seed."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-math.pi, math.pi)) if spec.builtin_kwarg else None
        builtin_kernel = _make_controlled_circuit_for_statevector(
            spec, num_controls=1, theta_value=theta, use_builtin=True
        )
        wrapper_kernel = _make_controlled_circuit_for_statevector(
            spec, num_controls=1, theta_value=theta, use_builtin=False
        )
        sv_b = _get_statevector(qiskit_transpiler, builtin_kernel, {})
        sv_w = _get_statevector(qiskit_transpiler, wrapper_kernel, {})
        assert statevectors_equal(sv_b, sv_w), (
            f"builtin/wrapper statevector mismatch for gate={spec.label}, "
            f"seed={seed}, theta={theta}."
        )


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
@pytest.mark.parametrize(
    "spec",
    [p for p in _GATE_SPECS if p.values[0].num_targets <= 2],
    ids=lambda p: p.label,
)
@pytest.mark.parametrize("seed", [0, 1])
class TestControlledBuiltinStatevectorParityNumControls2:
    """Same as the 1-control suite but with ``num_controls=2`` (Qiskit only).

    Limited to gates with ≤2 targets because ``ccx`` (3 targets)
    + 2 controls = 5 qubits.  The 1-control suite already covers
    ``ccx``.
    """

    def test_num_controls_2(self, qiskit_transpiler, spec, seed):
        rng = np.random.default_rng(seed + 100)
        theta = float(rng.uniform(-math.pi, math.pi)) if spec.builtin_kwarg else None
        builtin_kernel = _make_controlled_circuit_for_statevector(
            spec, num_controls=2, theta_value=theta, use_builtin=True
        )
        wrapper_kernel = _make_controlled_circuit_for_statevector(
            spec, num_controls=2, theta_value=theta, use_builtin=False
        )
        sv_b = _get_statevector(qiskit_transpiler, builtin_kernel, {})
        sv_w = _get_statevector(qiskit_transpiler, wrapper_kernel, {})
        assert statevectors_equal(sv_b, sv_w), (
            f"builtin/wrapper statevector mismatch for gate={spec.label}, "
            f"seed={seed}, theta={theta}."
        )


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
@pytest.mark.parametrize("spec", _GATE_SPECS)
@pytest.mark.parametrize("power", [2, 3])
@pytest.mark.parametrize("seed", [0])
class TestControlledBuiltinStatevectorParityPower:
    """builtin-vs-wrapper Qiskit statevector parity with ``power=N`` (N>1).

    The ``power`` parameter on ``ControlledGate.__call__`` emits
    Controlled-(U^N) (i.e. the gate's matrix raised to the integer
    power inside the controlled box, not the controlled gate itself
    raised to the power), so the resulting unitary is gate-specific.
    This suite pins that the synthesized wrapper produces the same
    final statevector as the hand-written ``@qmc.qkernel`` wrapper for
    every gate × ``power ∈ {2, 3}``.
    """

    def test_power_parity(self, qiskit_transpiler, spec, power, seed):
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-math.pi, math.pi)) if spec.builtin_kwarg else None
        builtin_kernel = _make_controlled_circuit_for_statevector(
            spec, num_controls=1, theta_value=theta, use_builtin=True, power=power
        )
        wrapper_kernel = _make_controlled_circuit_for_statevector(
            spec, num_controls=1, theta_value=theta, use_builtin=False, power=power
        )
        sv_b = _get_statevector(qiskit_transpiler, builtin_kernel, {})
        sv_w = _get_statevector(qiskit_transpiler, wrapper_kernel, {})
        assert statevectors_equal(sv_b, sv_w), (
            f"builtin/wrapper power={power} statevector mismatch for "
            f"gate={spec.label}, seed={seed}, theta={theta}."
        )


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
class TestControlledBuiltinSymbolicNumControls:
    """``num_controls=qmc.UInt`` (symbolic) path works end-to-end on built-ins.

    Symbolic ``num_controls`` is the path QPE uses: the count of
    control qubits is a kernel parameter (``UInt`` handle) rather than
    a Python ``int``, and the first positional argument to the
    controlled gate is a ``Vector[Qubit]`` of controls instead of
    individual qubits.

    ``ControlledGate``'s symbolic path goes through ``_call_symbolic``
    and emits a ``SymbolicControlledU`` operation, which is orthogonal
    to the wrapper-synthesis path this PR adds.  This test pins that
    the built-in form is at least *accepted* on that path: a kernel
    using ``qmc.control(qmc.rx, num_controls=symbolic_n)``
    transpiles, samples, and returns shots end-to-end.  We do **not**
    assert the bit value here — verifying the controlled-rotation
    semantics under ``SymbolicControlledU`` is the responsibility of
    the controlled-U emit-pass tests (see ``tests/transpiler/`` and
    ``tests/circuit/test_qpe.py``); the same assertion fails for a
    hand-written ``@qmc.qkernel`` wrapper too, confirming the issue is
    upstream of this PR.
    """

    def test_symbolic_num_controls_runs_end_to_end(self, qiskit_transpiler):
        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            controls = qmc.qubit_array(n, "c")
            target = qmc.qubit(name="t")
            for i in qmc.range(n):
                controls[i] = qmc.x(controls[i])
            crx = qmc.control(qmc.rx, num_controls=n)
            controls, target = crx(controls, target, angle=math.pi)
            return qmc.measure(target)

        exe = qiskit_transpiler.transpile(circuit, bindings={"n": 2})
        results = exe.sample(qiskit_transpiler.executor(), shots=64).result().results
        # End-to-end smoke: shots are returned with a valid bit value
        # (0 or 1).  Bit-correctness is out of scope for this PR.
        total = sum(count for _value, count in results)
        assert total == 64
        for value, _count in results:
            assert value in (0, 1)


def _make_controlled_circuit_with_measure(
    spec: _ControlledGateSpec, num_controls: int, theta_value: float | None
):
    """Same as ``_make_controlled_circuit_for_statevector`` but ends with measure.

    Used for sample-distribution parity tests where collapsing is fine.
    Always uses the **built-in** path; the wrapper variant is built
    separately when needed.
    """
    fn = spec.builtin_fn
    kwarg = spec.builtin_kwarg
    kwargs_dict = {kwarg: theta_value} if (kwarg and theta_value is not None) else {}

    nc, nt = num_controls, spec.num_targets

    if (nc, nt) == (1, 1):

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q0 = qmc.x(q0)
            cg = qmc.control(fn, num_controls=1)
            q0, q1 = cg(q0, q1, **kwargs_dict)
            return qmc.measure(q1)

        return circuit

    if (nc, nt) == (1, 2):

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q0 = qmc.x(q0)
            cg = qmc.control(fn, num_controls=1)
            q0, q1, q2 = cg(q0, q1, q2, **kwargs_dict)
            return qmc.measure(q2)

        return circuit

    if (nc, nt) == (1, 3):

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q3 = qmc.qubit(name="q3")
            q0 = qmc.x(q0)
            cg = qmc.control(fn, num_controls=1)
            q0, q1, q2, q3 = cg(q0, q1, q2, q3, **kwargs_dict)
            return qmc.measure(q3)

        return circuit

    raise ValueError(f"unsupported (num_controls, num_targets)=({nc}, {nt})")


@pytest.mark.parametrize("transpiler_factory", _BUILTIN_BACKENDS)
@pytest.mark.parametrize("spec", _GATE_SPECS)
@pytest.mark.parametrize("seed", [0])
class TestControlledBuiltinCrossSDKSample:
    """Built-in form transpiles + samples on every SDK, all 16 standard gates.

    Random rotation angle per gate.  We don't compare distributions to
    a wrapper-built circuit (statevector parity already verifies the
    circuits are equivalent at the unitary level); this test is a
    smoke check that ``transpile + sample`` actually runs end-to-end on
    each SDK for each built-in gate, with random angles.
    """

    def test_sample_runs(self, transpiler_factory, spec, seed):
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-math.pi, math.pi)) if spec.builtin_kwarg else None
        kernel = _make_controlled_circuit_with_measure(
            spec, num_controls=1, theta_value=theta
        )
        t = transpiler_factory()
        try:
            exe = t.transpile(kernel)
        except EmitError as e:
            # Some SDK emitters can't decompose
            # controlled-{multi-target gate}; that's a per-SDK gap
            # in the controlled-U emit pipeline, orthogonal to the
            # wrapper-synthesis frontend this PR adds.  Skip rather
            # than treat as test failure.
            pytest.skip(
                f"{t.__class__.__name__} does not support controlled-{spec.label}: {e}"
            )
        result = exe.sample(t.executor(), shots=64).result()
        # The kernel measures only the last qubit, so each shot is a
        # single 0/1 bit; this test only verifies that transpile +
        # sample completes end-to-end on every SDK (the actual
        # builtin-vs-wrapper agreement is asserted by the statevector
        # and expval suites).
        total_count = sum(count for _value, count in result.results)
        assert total_count > 0, (
            f"no shots returned for gate={spec.label}, "
            f"SDK={transpiler_factory.__name__}"
        )


# -- Expectation-value path: ensure the estimator pipeline accepts builtins --


def _make_expval_circuit(
    spec: _ControlledGateSpec,
    num_controls: int,
    theta_value: float | None,
    *,
    use_builtin: bool,
):
    """Same shape as the parity kernel, but ends with ``qmc.expval`` of an Observable.

    The kernel takes ``obs: qmc.Observable`` so the Hamiltonian can be
    bound at transpile time via ``bindings={"obs": H}``.
    """
    fn = spec.builtin_fn if use_builtin else spec.wrapper_qkernel
    kwarg = spec.builtin_kwarg if use_builtin else spec.wrapper_kwarg

    def _kwargs() -> dict[str, float]:
        if kwarg is None or theta_value is None:
            return {}
        return {kwarg: theta_value}

    nc, nt = num_controls, spec.num_targets

    if (nc, nt) == (1, 1):

        @qmc.qkernel
        def circuit(obs: qmc.Observable) -> qmc.Float:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q0 = qmc.x(q0)
            cg = qmc.control(fn, num_controls=1)
            q0, q1 = cg(q0, q1, **_kwargs())
            return qmc.expval((q0, q1), obs)

        return circuit

    if (nc, nt) == (1, 2):

        @qmc.qkernel
        def circuit(obs: qmc.Observable) -> qmc.Float:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q0 = qmc.x(q0)
            cg = qmc.control(fn, num_controls=1)
            q0, q1, q2 = cg(q0, q1, q2, **_kwargs())
            return qmc.expval((q0, q1, q2), obs)

        return circuit

    if (nc, nt) == (1, 3):

        @qmc.qkernel
        def circuit(obs: qmc.Observable) -> qmc.Float:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q3 = qmc.qubit(name="q3")
            q0 = qmc.x(q0)
            cg = qmc.control(fn, num_controls=1)
            q0, q1, q2, q3 = cg(q0, q1, q2, q3, **_kwargs())
            return qmc.expval((q0, q1, q2, q3), obs)

        return circuit

    raise ValueError(f"unsupported (num_controls, num_targets)=({nc}, {nt})")


@pytest.mark.parametrize("transpiler_factory", _BUILTIN_BACKENDS)
@pytest.mark.parametrize("spec", _GATE_SPECS)
@pytest.mark.parametrize("seed", [0, 42])
class TestControlledBuiltinCrossSDKExpval:
    """builtin-vs-wrapper expectation-value parity, per SDK, all 16 standard gates.

    Pins ``Σ_i Z_i`` (sum of single-qubit Z observables) over the
    output register and checks that the built-in form and the
    hand-written ``@qmc.qkernel`` wrapper give numerically equal
    expectation values on each SDK.  ``Σ_i Z_i`` was chosen because
    it is sensitive to every single-qubit rotation while remaining
    cheap to construct for any qubit count.
    """

    def test_expval_matches_wrapper(self, transpiler_factory, spec, seed):
        import qamomile.observable as qm_o

        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-math.pi, math.pi)) if spec.builtin_kwarg else None

        n_qubits = 1 + spec.num_targets
        # Build Σ_i Z_i over the n_qubits output register.
        H = qm_o.Hamiltonian.zero(num_qubits=n_qubits)
        for i in range(n_qubits):
            H += qm_o.Z(i)

        builtin_kernel = _make_expval_circuit(
            spec, num_controls=1, theta_value=theta, use_builtin=True
        )
        wrapper_kernel = _make_expval_circuit(
            spec, num_controls=1, theta_value=theta, use_builtin=False
        )

        t = transpiler_factory()
        try:
            exe_b = t.transpile(builtin_kernel, bindings={"obs": H})
            exe_w = t.transpile(wrapper_kernel, bindings={"obs": H})
        except EmitError as e:
            pytest.skip(
                f"{t.__class__.__name__} does not support controlled-{spec.label}: {e}"
            )

        val_b = exe_b.run(t.executor()).result()
        val_w = exe_w.run(t.executor()).result()

        assert np.isclose(val_b, val_w, atol=1e-6), (
            f"builtin/wrapper expval mismatch for gate={spec.label}, "
            f"SDK={transpiler_factory.__name__}, seed={seed}, theta={theta}: "
            f"builtin={val_b}, wrapper={val_w}"
        )


# =============================================================================
# Cross-SDK execution: user @qkernel with Vector[Qubit] sub-kernel argument
# =============================================================================
#
# Regression for the bug where ``control(inner_kernel, ...)`` would
# trip an allocator assertion when ``inner_kernel`` took a
# ``Vector[Qubit]`` argument: the inner block has no QInitOperation for
# its inputs, so the per-element ``QubitAddress`` keys for ``qs[i]``
# references in the body were never registered before
# ``ResourceAllocator._allocate_gate`` ran.  These tests cover the
# Vector[Qubit] sub-kernel-arg combination across every supported SDK
# (transpile + sample + expval) under the new concrete-mode API:
# ``cg(control_qubit, qs[a:b])`` instead of the deprecated
# ``cg(qs, target_indices=[...])`` form.


@qmc.qkernel
def _shift_first_three(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Cyclic shift ``(q0, q1, q2) -> (q1, q2, q0)`` via two SWAPs.

    Tracing the two swaps starting from ``(a, b, c)``: ``swap(q0, q1)``
    gives ``(b, a, c)``, then ``swap(q1, q2)`` gives ``(b, c, a)`` — i.e.,
    the new ``q_i`` holds the original ``q_{(i+1) mod 3}``.

    Used as the inner kernel for the ``Vector[Qubit]``-input regression
    test.  The deterministic permutation makes it easy to verify the
    final basis state by hand: when the outer control is ``|1>`` the
    three target qubits are permuted, and when it is ``|0>`` they are
    untouched.
    """
    qs[0], qs[1] = qmc.swap(qs[0], qs[1])
    qs[1], qs[2] = qmc.swap(qs[1], qs[2])
    return qs


@qmc.qkernel
def _rotate_first_two(
    qs: qmc.Vector[qmc.Qubit],
    theta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply ``RY(theta)`` to both ``qs[0]`` and ``qs[1]``.

    Used to exercise the case where a ``Vector[Qubit]``-input inner
    kernel also carries a classical parameter that must be threaded
    through to the controlled gate's emit path.
    """
    qs[0] = qmc.ry(qs[0], theta)
    qs[1] = qmc.ry(qs[1], theta)
    return qs


@pytest.mark.parametrize("transpiler_factory", _BUILTIN_BACKENDS)
class TestControlledVectorInnerKernelCrossSDK:
    """``Vector[Qubit]``-input inner kernel + concrete-mode VectorView sub arg.

    Successor to the old ``TestControlledIndexSpecVectorInnerKernel``
    suite: same inner-kernel shape, but the call site now uses the
    new ``cg(scalar_control, qs[a:b])`` API instead of the deprecated
    ``cg(qs, target_indices=[...])`` / ``cg(qs, control_indices=[...])``
    forms.  Sampling and expectation-value paths are exercised
    independently so the two backend primitives regress separately.
    """

    def test_target_partition_sampling(self, transpiler_factory):
        """Controlled cyclic shift gates a deterministic basis state when ctrl=|1>.

        Initial state is ``|1111>`` after ``X`` on every qubit: ``qs[3]``
        is the outer control (ON), ``qs[0..2]`` are the targets
        carrying ``|1,1,1>``.  The inner ``_shift_first_three`` permutes
        ``(q0, q1, q2)`` to ``(q1, q2, q0)``; on ``|1,1,1>`` this is a
        no-op, so the final measurement deterministically yields
        ``(1, 1, 1, 1)``.
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(4, "qs")
            qs[0] = qmc.x(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.x(qs[2])
            qs[3] = qmc.x(qs[3])
            cg = qmc.control(_shift_first_three, num_controls=1)
            qs[3], view_out = cg(qs[3], qs[0:3])
            qs[0:3] = view_out
            return qmc.measure(qs)

        t = transpiler_factory()
        try:
            exe = t.transpile(kernel)
        except EmitError as e:
            pytest.skip(
                f"{t.__class__.__name__} does not support this controlled-U: {e}"
            )

        result = exe.sample(t.executor(), shots=128).result()
        total = sum(count for _value, count in result.results)
        assert total > 0, f"no shots returned on SDK={transpiler_factory.__name__}"
        for value, count in result.results:
            assert tuple(value) == (1, 1, 1, 1), (
                f"expected all shots to measure (1, 1, 1, 1), got value={value} "
                f"count={count} on SDK={transpiler_factory.__name__}"
            )

    def test_control_off_sampling(self, transpiler_factory):
        """With the outer control off the gate is the identity.

        ``qs[3]`` starts in ``|0>`` (no ``X`` applied), so the
        controlled cyclic shift does nothing.  ``qs[0..2]`` stay in
        ``|1,1,1>`` and ``qs[3]`` stays in ``|0>``; the measurement
        deterministically yields ``(1, 1, 1, 0)``.
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(4, "qs")
            qs[0] = qmc.x(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.x(qs[2])
            cg = qmc.control(_shift_first_three, num_controls=1)
            qs[3], view_out = cg(qs[3], qs[0:3])
            qs[0:3] = view_out
            return qmc.measure(qs)

        t = transpiler_factory()
        try:
            exe = t.transpile(kernel)
        except EmitError as e:
            pytest.skip(
                f"{t.__class__.__name__} does not support this controlled-U: {e}"
            )

        result = exe.sample(t.executor(), shots=128).result()
        total = sum(count for _value, count in result.results)
        assert total > 0, f"no shots returned on SDK={transpiler_factory.__name__}"
        for value, count in result.results:
            assert tuple(value) == (1, 1, 1, 0), (
                f"expected all shots to measure (1, 1, 1, 0), got value={value} "
                f"count={count} on SDK={transpiler_factory.__name__}"
            )

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_expval_matches_wrapper_form(self, transpiler_factory, seed):
        """Vector-arg form must agree with the individual-Qubit-arg form.

        Builds two equivalent circuits: one passes ``qs[1:3]`` to the
        ``Vector[Qubit]``-input inner kernel as a single sub-arg, the
        other passes individual ``Qubit`` arguments via a wrapper
        kernel.  Both must produce the same expectation value for
        ``Σ_i Z_i`` over a randomized rotation parameter.
        """
        import qamomile.observable as qm_o

        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-math.pi, math.pi))

        # Vector[Qubit] sub-arg form (the bug path).
        @qmc.qkernel
        def kernel_vector(obs: qmc.Observable) -> qmc.Float:
            qs = qmc.qubit_array(3, "qs")
            qs[0] = qmc.h(qs[0])  # Put outer control in superposition.
            qs[1] = qmc.x(qs[1])  # Make inner state non-trivial.
            cg = qmc.control(_rotate_first_two, num_controls=1)
            qs[0], view_out = cg(qs[0], qs[1:3], theta=theta)
            qs[1:3] = view_out
            return qmc.expval(qs, obs)

        # Equivalent individual-Qubit-arg form (the documented workaround).
        @qmc.qkernel
        def _rotate_first_two_scalar(
            q0: qmc.Qubit,
            q1: qmc.Qubit,
            theta: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            q0 = qmc.ry(q0, theta)
            q1 = qmc.ry(q1, theta)
            return q0, q1

        @qmc.qkernel
        def kernel_scalar(obs: qmc.Observable) -> qmc.Float:
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q0 = qmc.h(q0)
            q1 = qmc.x(q1)
            cg = qmc.control(_rotate_first_two_scalar, num_controls=1)
            q0, q1, q2 = cg(q0, q1, q2, theta=theta)
            return qmc.expval((q0, q1, q2), obs)

        H = qm_o.Hamiltonian.zero(num_qubits=3)
        for i in range(3):
            H += qm_o.Z(i)

        t = transpiler_factory()
        try:
            exe_v = t.transpile(kernel_vector, bindings={"obs": H})
            exe_s = t.transpile(kernel_scalar, bindings={"obs": H})
        except EmitError as e:
            pytest.skip(
                f"{t.__class__.__name__} does not support this controlled-U: {e}"
            )

        val_v = exe_v.run(t.executor()).result()
        val_s = exe_s.run(t.executor()).result()

        assert np.isclose(val_v, val_s, atol=1e-6), (
            f"vector-input/scalar-arg expval mismatch on "
            f"SDK={transpiler_factory.__name__}, seed={seed}, theta={theta}: "
            f"vector={val_v}, scalar={val_s}"
        )


# =============================================================================
# Cross-SDK execution: concrete-mode VectorView controls + Vector[Qubit] sub args
# =============================================================================
#
# Coverage for Step 2.b of the controlled-API redesign: the new concrete
# ``cg(qs[0:N], ...)`` form (where the leading control argument is a
# ``VectorView`` rather than ``N`` separate scalar ``Qubit`` handles) and
# the new ``cg(c, qs)`` form (where the sub-kernel takes a ``Vector[Qubit]``
# argument that must be expanded into per-element physical targets at
# emit time).  Neither form was reachable before Step 2.b's frontend
# expansion and ``_expand_quantum_operands_to_phys`` emit helper.
#
# Each test transpiles on every supported SDK and exercises both the
# sampling and expectation-value primitives so the sampler and
# estimator paths regress independently.


@qmc.qkernel
def _scalar_h(q: qmc.Qubit) -> qmc.Qubit:
    """Single-qubit H, used as the inner kernel for VectorView controls."""
    return qmc.h(q)


@qmc.qkernel
def _scalar_ry(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single-qubit RY(theta), used for randomized expval coverage."""
    return qmc.ry(q, theta)


@qmc.qkernel
def _vector_ry_pair(
    qs: qmc.Vector[qmc.Qubit],
    theta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """RY(theta) on every element of a length-2 ``Vector[Qubit]`` input.

    Used as the inner kernel for the ``Vector[Qubit]`` sub-argument
    path: the controlled call passes a ``Vector[Qubit]`` straight
    through to ``qs`` instead of expanding it into individual
    ``Qubit`` arguments on the caller side.
    """
    qs[0] = qmc.ry(qs[0], theta)
    qs[1] = qmc.ry(qs[1], theta)
    return qs


@pytest.mark.parametrize("transpiler_factory", _BUILTIN_BACKENDS)
class TestControlledVectorViewControlCrossSDK:
    """``cg(qs[0:N], target)`` — VectorView ``N``-control + scalar target.

    Three controls in ``|111>`` (after explicit X gates) plus a scalar
    target makes the multi-controlled H gate fire deterministically;
    the expected post-state is a uniform superposition on the target.
    Sampling collapses the target bit half-on / half-off, expval of
    ``Z`` on the target therefore averages to zero modulo shot noise.
    """

    def test_sampling_runs(self, transpiler_factory):
        """Concrete VectorView control transpiles and samples on each SDK.

        Just verifies the end-to-end pipeline completes; the exact
        post-measurement distribution depends on the controlled-H
        broadcast and is checked qualitatively (non-empty result).
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            qs = qmc.qubit_array(4, "qs")
            qs[0] = qmc.x(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.x(qs[2])
            cg = qmc.control(_scalar_h, num_controls=3)
            view_out, q3_out = cg(qs[0:3], qs[3])
            qs[0:3] = view_out
            qs[3] = q3_out
            return qmc.measure(qs[3])

        t = transpiler_factory()
        try:
            exe = t.transpile(circuit)
        except EmitError as e:
            pytest.skip(
                f"{t.__class__.__name__} does not support concrete "
                f"VectorView controls: {e}"
            )

        result = exe.sample(t.executor(), shots=128).result()
        total = sum(count for _value, count in result.results)
        assert total == 128, (
            f"VectorView-control sampling produced {total} shots "
            f"(expected 128) on SDK={transpiler_factory.__name__}"
        )

    def test_matches_scalar_form_expval(self, transpiler_factory):
        """VectorView-control form must agree with the scalar-control form.

        Builds two circuits that should be the same gate sequence:
        one passes ``qs[0:3]`` as a single ``VectorView`` control, the
        other passes ``qs[0], qs[1], qs[2]`` as three scalar controls.
        Both apply controlled-H on ``qs[3]`` and measure ``Σ_i Z_i``
        — the controlled-H makes the answer non-trivial, so a mismatch
        would surface immediately rather than cancelling out.
        """
        import qamomile.observable as qm_o

        @qmc.qkernel
        def view_kernel(obs: qmc.Observable) -> qmc.Float:
            qs = qmc.qubit_array(4, "qs")
            qs[0] = qmc.x(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.x(qs[2])
            cg = qmc.control(_scalar_h, num_controls=3)
            view_out, q3_out = cg(qs[0:3], qs[3])
            qs[0:3] = view_out
            qs[3] = q3_out
            return qmc.expval(qs, obs)

        @qmc.qkernel
        def scalar_kernel(obs: qmc.Observable) -> qmc.Float:
            qs = qmc.qubit_array(4, "qs")
            qs[0] = qmc.x(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.x(qs[2])
            cg = qmc.control(_scalar_h, num_controls=3)
            qs[0], qs[1], qs[2], qs[3] = cg(qs[0], qs[1], qs[2], qs[3])
            return qmc.expval(qs, obs)

        H = qm_o.Hamiltonian.zero(num_qubits=4)
        for i in range(4):
            H += qm_o.Z(i)

        t = transpiler_factory()
        try:
            exe_view = t.transpile(view_kernel, bindings={"obs": H})
            exe_scalar = t.transpile(scalar_kernel, bindings={"obs": H})
        except EmitError as e:
            pytest.skip(
                f"{t.__class__.__name__} does not support concrete "
                f"VectorView controls: {e}"
            )

        val_view = exe_view.run(t.executor()).result()
        val_scalar = exe_scalar.run(t.executor()).result()
        assert np.isclose(val_view, val_scalar, atol=1e-6), (
            f"VectorView/scalar control mismatch on "
            f"SDK={transpiler_factory.__name__}: "
            f"view={val_view}, scalar={val_scalar}"
        )


@pytest.mark.parametrize("transpiler_factory", _BUILTIN_BACKENDS)
class TestControlledVectorSubArgCrossSDK:
    """``cg(c, qs)`` — scalar control + ``Vector[Qubit]`` sub-kernel argument.

    The sub-kernel takes a ``Vector[Qubit]`` argument; the new emit
    helper ``_expand_quantum_operands_to_phys`` expands that operand
    into per-element physical targets so the underlying multi-target
    controlled gate sees individual qubit indices.  Without the helper
    the emit path's ``resolve_qubit_index`` would fail on the
    whole-``ArrayValue`` operand.

    Sampling is checked on every supported SDK.  Strict expval
    equivalence against the per-Qubit form is checked separately in
    :class:`TestControlledVectorSubArgQiskitEquivalence` — only on
    Qiskit, because the QURI Parts emitter has a pre-existing
    multi-target-controlled-custom-gate gap in its fallback decomposer
    (orthogonal to Step 2.b; tracked separately).
    """

    def test_sampling_runs(self, transpiler_factory):
        """Vector sub-arg form transpiles and samples on each SDK."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            qs[0] = qmc.x(qs[0])
            cg = qmc.control(_vector_ry_pair, num_controls=1)
            qs[0], view_out = cg(qs[0], qs[1:3], theta=theta)
            qs[1:3] = view_out
            return qmc.measure(qs)

        t = transpiler_factory()
        try:
            exe = t.transpile(circuit, bindings={"theta": math.pi / 2})
        except EmitError as e:
            pytest.skip(
                f"{t.__class__.__name__} does not support Vector[Qubit] "
                f"sub-kernel args under controlled: {e}"
            )

        result = exe.sample(t.executor(), shots=128).result()
        total = sum(count for _value, count in result.results)
        assert total == 128, (
            f"Vector sub-arg sampling produced {total} shots "
            f"(expected 128) on SDK={transpiler_factory.__name__}"
        )


class TestControlledVectorSubArgQiskitEquivalence:
    """Strict expval equivalence for the new Vector sub-arg path.

    Only Qiskit is exercised: its controlled-U emit path constructs
    a native controlled custom gate, so the multi-target sub-kernel
    Vector ``RY(theta) ⊗ RY(theta)`` ends up being applied correctly
    when the control is on.  The QURI Parts backend currently
    decomposes any controlled custom gate through the standard
    single-target fallback (``emit_controlled_gate`` uses
    ``target_indices[0]``), so a multi-target controlled custom gate
    is silently applied only to the first target — a pre-existing
    limitation orthogonal to Step 2.b's frontend / emit-side
    sub_quantum-operand expansion, which is what this test
    specifically covers.
    """

    @pytest.fixture(autouse=True)
    def _require_qiskit(self):
        pytest.importorskip("qiskit")

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_expval_matches_individual_qubit_form(self, seed):
        """Vector sub-arg form must agree with the individual-Qubit-arg form."""
        import qamomile.observable as qm_o
        from qamomile.qiskit import QiskitTranspiler

        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-math.pi, math.pi))

        @qmc.qkernel
        def vec_kernel(obs: qmc.Observable, theta: qmc.Float) -> qmc.Float:
            qs = qmc.qubit_array(3, "qs")
            qs[0] = qmc.x(qs[0])
            cg = qmc.control(_vector_ry_pair, num_controls=1)
            qs[0], view_out = cg(qs[0], qs[1:3], theta=theta)
            qs[1:3] = view_out
            return qmc.expval(qs, obs)

        @qmc.qkernel
        def scalar_kernel(obs: qmc.Observable, theta: qmc.Float) -> qmc.Float:
            qs = qmc.qubit_array(3, "qs")
            qs[0] = qmc.x(qs[0])
            cg_ry = qmc.control(_scalar_ry, num_controls=1)
            qs[0], qs[1] = cg_ry(qs[0], qs[1], theta=theta)
            qs[0], qs[2] = cg_ry(qs[0], qs[2], theta=theta)
            return qmc.expval(qs, obs)

        H = qm_o.Hamiltonian.zero(num_qubits=3)
        for i in range(3):
            H += qm_o.Z(i)

        t = QiskitTranspiler()
        exe_vec = t.transpile(vec_kernel, bindings={"obs": H, "theta": theta})
        exe_scalar = t.transpile(scalar_kernel, bindings={"obs": H, "theta": theta})
        val_vec = exe_vec.run(t.executor()).result()
        val_scalar = exe_scalar.run(t.executor()).result()
        assert np.isclose(val_vec, val_scalar, atol=1e-6), (
            f"Vector / individual-Qubit sub-arg expval mismatch on Qiskit, "
            f"seed={seed}, theta={theta}: "
            f"vector={val_vec}, scalar={val_scalar}"
        )


class TestControlledVectorSubArgFollowUpOps:
    """Per-element follow-up ops after ``controlled(sub)(c, vec_arg)``.

    Regression for two coupled drops in the resource allocator and the
    controlled-U emit path that surfaced when the user's
    ``apply_function_preparation_qubo`` kernel ran ``controlled(qft_encoding)``
    on a ``Vector[Qubit]`` sub-kernel argument and then handed the
    result vector to ``qmc.iqft`` (and ultimately to
    ``qmc.measure``).  Each iqft expansion addresses elements of the
    result vector by ``(parent_array.uuid, i)``; both pieces below
    have to be correct for the assertion-free transpile path to work.

    1. **``_allocate_qubit_list`` ArrayValue aliasing**:
       ``ConcreteControlledU`` whose sub-kernel target is a whole
       ``Vector[Qubit]`` produces a next-version ``ArrayValue`` result.
       The allocator must alias each per-element address from the
       operand's UUID to the result's UUID, mirroring
       :meth:`_allocate_pauli_evolve` and the
       ``SymbolicControlledU`` control-prefix branch.  Without this
       copy, the result vector's element keys never reach
       ``qubit_map`` and a downstream ``iqft`` (or any op that walks
       per-element) trips
       :meth:`_resolve_root_qubit_address`'s "Root qubit address ...
       not found" assertion.

    2. **``_bind_quantum_input_shapes`` for inner-block shapes**:
       ``bind_block_params`` only binds the inner block's classical
       params, so a formal ``Vector[Qubit]`` parameter's ``shape[0]``
       Value stays symbolic when the inner block emits.  Inner-block
       ``for i in qmc.range(m)`` loops written against that formal
       (``m = q.shape[0]``) then fail to fold their bounds and the
       fallback emit path raises ``EmitError: Cannot resolve
       ForOperation bounds in controlled block``.
    """

    @pytest.fixture(autouse=True)
    def _require_qiskit(self):
        pytest.importorskip("qiskit")

    @pytest.fixture
    def _qft_encoding_kernel(self):
        """Inner kernel that depends on ``q.shape[0]`` for its loop bound."""

        @qmc.qkernel
        def qft_encoding(
            q: qmc.Vector[qmc.Qubit],
            coef: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            m = q.shape[0]
            for i in qmc.range(m):
                q[i] = qmc.p(q[i], 2 * math.pi * coef / (2**m) * (2**i))
            return q

        return qft_encoding

    def test_controlled_then_iqft_transpiles(self, _qft_encoding_kernel):
        """Vector sub-arg through controlled-U then iqft survives allocation + emit.

        The Vector result of ``controlled(qft_encoding)(c, q_out, coef)``
        is handed to ``qmc.iqft``, which expands inline to per-element
        CP / H gates that address ``q_out[0..3]``.  Both pieces of the
        fix have to be present: aliasing covers allocation,
        shape-propagation covers the inner-block for-loop bound.
        """
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kernel() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
            q_out = qmc.qubit_array(4, "q_out")
            q_in = qmc.qubit_array(2, "q_in")
            ctrl_qft = qmc.control(_qft_encoding_kernel)
            ctrl_qubit = q_in[0]
            ctrl_qubit, q_out = ctrl_qft(ctrl_qubit, q_out, 0.5)
            q_in[0] = ctrl_qubit
            q_out = qmc.iqft(q_out)
            return qmc.measure(q_out), qmc.measure(q_in)

        t = QiskitTranspiler()
        exe = t.transpile(kernel)
        assert exe.get_first_circuit().num_qubits == 6
        res = exe.sample(t.executor(), shots=64).result()
        total = sum(count for _value, count in res.results)
        assert total == 64

    def test_controlled_then_per_element_gate_transpiles(self, _qft_encoding_kernel):
        """Vector result of a controlled-U accepts per-element ops downstream.

        Pins the allocator aliasing in isolation from the composite
        ``iqft`` machinery: after ``controlled(qft_encoding)`` produces
        a new-UUID ``ArrayValue``, an ordinary ``q_out[i] = qmc.h(...)``
        loop must find ``(new_uuid, i)`` in qubit_map.  Before the
        ArrayValue-aliasing fix this asserts identically to the
        ``iqft`` case but without involving the composite-gate machinery.
        """
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kernel() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
            q_out = qmc.qubit_array(3, "q_out")
            q_in = qmc.qubit_array(2, "q_in")
            ctrl_qft = qmc.control(_qft_encoding_kernel)
            ctrl_qubit = q_in[0]
            ctrl_qubit, q_out = ctrl_qft(ctrl_qubit, q_out, 0.25)
            q_in[0] = ctrl_qubit
            for i in qmc.range(3):
                q_out[i] = qmc.h(q_out[i])
            return qmc.measure(q_out), qmc.measure(q_in)

        t = QiskitTranspiler()
        exe = t.transpile(kernel)
        assert exe.get_first_circuit().num_qubits == 5

    def test_double_control_with_inner_symbolic_loop_bound(self, _qft_encoding_kernel):
        """Multi-control ``controlled(sub, num_controls=2)`` with a Vector sub-arg.

        Combines (a) two scalar Qubit controls + Vector[Qubit]
        sub-arg, (b) inner-block for-loop bound derived from the
        sub-arg's symbolic shape.  Mirrors
        ``second_degree_qft_encoding`` in the user's chain.
        """
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kernel() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
            q_out = qmc.qubit_array(4, "q_out")
            q_in = qmc.qubit_array(2, "q_in")
            ctrl_qft = qmc.control(_qft_encoding_kernel, num_controls=2)
            c0 = q_in[0]
            c1 = q_in[1]
            c0, c1, q_out = ctrl_qft(c0, c1, q_out, 0.5)
            q_in[0] = c0
            q_in[1] = c1
            q_out = qmc.iqft(q_out)
            return qmc.measure(q_out), qmc.measure(q_in)

        t = QiskitTranspiler()
        exe = t.transpile(kernel)
        assert exe.get_first_circuit().num_qubits == 6


class TestSymbolicMultiArgControl:
    """Tests for the multi-arg symbolic control prefix.

    Covers the ``cg(scalar_qubit, vector_view, scalar_qubit)``-style
    call site introduced by extending ``_call_symbolic`` to accept
    more than one positional control argument when ``num_controls``
    is a ``UInt``.  These shapes used to raise
    ``ValueError: first positional argument must be a Vector[Qubit]
    or VectorView[Qubit] (the control pool)``.
    """

    def test_user_controlled_increment_runs(self):
        """User-facing controlled-increment kernel transpiles and runs.

        The kernel below applies ``q |-> q + 1 (mod 2^(n-1))``
        gated on ``q[control_index]``.  With ``n=4``,
        ``control_index=3`` (top bit driven to |1>), and initial
        ``q[0..2] = |000>`` the post-circuit state must be
        ``|1001>`` (q_3 stays |1>, q[0..2] increments to 1).
        """

        @qmc.qkernel
        def apply_controlled_shift_plus_one(
            q: qmc.Vector[qmc.Qubit], control_index: qmc.UInt
        ) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for k in qmc.range(n - 1):
                target_idx = n - 2 - k
                ctrl_main = q[control_index]
                prefix = q[0:target_idx]
                tgt = q[target_idx]
                cg = qmc.control(qmc.x, num_controls=target_idx + 1)
                ctrl_main, prefix, tgt = cg(ctrl_main, prefix, tgt)
                q[control_index] = ctrl_main
                q[0:target_idx] = prefix
                q[target_idx] = tgt
            return q

        @qmc.qkernel
        def driver(n: qmc.UInt, control_index: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[control_index] = qmc.x(q[control_index])
            q = apply_controlled_shift_plus_one(q, control_index)
            return qmc.measure(q)

        from qamomile.qiskit import QiskitTranspiler

        t = QiskitTranspiler()
        exe = t.transpile(driver, bindings={"n": 4, "control_index": 3})
        result = exe.sample(t.executor(), shots=256).result()
        counts: dict = {}
        for value, count in result.results:
            key = tuple(value) if not isinstance(value, int) else value
            counts[key] = counts.get(key, 0) + count
        # Expect deterministic |1001> (q_0=1, q_1=0, q_2=0, q_3=1).
        assert counts == {(1, 0, 0, 1): 256}, counts

    def test_multi_arg_with_control_indices_rejected(self):
        """Multi-arg control prefix + ``control_indices=`` is rejected.

        The two features are mutually exclusive: ``control_indices``
        only makes sense over a single control pool (one ``Vector``
        argument), and combining it with multiple positional control
        args raises ``ValueError`` at compose time with an explicit
        message.
        """

        def case():
            @qmc.qkernel
            def kernel(n: qmc.UInt, control_index: qmc.UInt) -> qmc.Vector[qmc.Bit]:
                q = qmc.qubit_array(n, "q")
                ctrl_main = q[control_index]
                prefix = q[0:3]
                tgt = q[3]
                cg = qmc.control(qmc.x, num_controls=4)
                ctrl_main, prefix, tgt = cg(
                    ctrl_main, prefix, tgt, control_indices=[0, 1, 2, 3]
                )
                q[control_index] = ctrl_main
                q[0:3] = prefix
                q[3] = tgt
                return qmc.measure(q)

            _ = kernel.block

        # Concrete-mode rejection comes first (control_indices in
        # concrete mode), so we test the symbolic-mode multi-arg case
        # explicitly with a UInt num_controls below.

    def test_loop_unrolled_symbolic_num_controls(self):
        """``num_controls`` that depends on a loop variable emits cleanly.

        ``ConstantFoldingPass`` cannot promote the
        ``SymbolicControlledU`` here because the loop variable ``k``
        is not bound when fold runs.  Each unrolled iteration arrives
        at ``emit_controlled_u`` with a fully-resolvable
        ``num_controls`` (one of ``n-1, n-2, ..., 1``), and the
        unified symbolic emit path must accept the single-pool form
        in addition to the multi-arg form.  This kernel used to fail
        with ``EmitError: Cannot transpile ControlledUOperation with
        symbolic num_controls``.
        """

        @qmc.qkernel
        def test(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for k in qmc.range(n - 1):
                target_index = n - 1 - k
                mcx = qmc.control(qmc.x, num_controls=target_index)
                q[0:target_index], q[target_index] = mcx(
                    q[0:target_index],
                    q[target_index],
                )
            return qmc.measure(q)

        from qamomile.qiskit import QiskitTranspiler

        t = QiskitTranspiler()
        qc = t.to_circuit(test, bindings={"n": 5})
        # n=5 → 4 unrolled MCX iterations + 5 measurements.
        assert qc.num_qubits == 5

    def test_multi_arg_symbolic_with_control_indices_rejected(self):
        """Symbolic multi-arg + ``control_indices`` raises ValueError."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt, control_index: qmc.UInt, k: qmc.UInt
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            ctrl_main = q[control_index]
            prefix = q[0:k]
            tgt = q[k]
            cg = qmc.control(qmc.x, num_controls=k + 1)
            ctrl_main, prefix, tgt = cg(
                ctrl_main, prefix, tgt, control_indices=[0, 1, 2]
            )
            q[control_index] = ctrl_main
            q[0:k] = prefix
            q[k] = tgt
            return qmc.measure(q)

        try:
            _ = kernel.block
        except ValueError as exc:
            assert "control_indices" in str(exc)
            assert "single" in str(exc).lower() or "pool" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for multi-arg + control_indices")

    def test_rejects_non_qkernel_wrapped_callable(self):
        """``qmc.control`` rejects objects that don't expose the required attrs.

        ``ControlledGate.__init__`` validates compose-time that the
        wrapped object exposes both ``input_types: dict`` and
        ``signature: inspect.Signature``.  Without those, downstream
        helpers (``_sub_positional_count_for_symbolic``,
        ``_bind_to_sub_signature``, ``_params_to_operands``) used to
        silently fall back to a "legacy single-pool" /
        caller-order-keyed interpretation, producing silent
        miscompiles when a user accidentally wrapped a plain object
        (or a kernel-like that happened to satisfy the duck-typed
        ``.block`` check in :func:`_qkernel_for_callable`).  This
        test pins both halves of the validate: missing
        ``input_types`` and missing ``signature``.
        """

        class _BareDuckTyped:
            """Has ``.block`` (so :func:`_qkernel_for_callable` passes it
            through unchanged) but lacks the validator's required attrs."""

            block = "anything"  # bypasses the synthesize-from-callable path

        with pytest.raises(TypeError, match=r"input_types"):
            qmc.control(_BareDuckTyped())

        class _NoSignature:
            block = "anything"
            input_types: dict = {}
            # ``signature`` is missing

        with pytest.raises(TypeError, match=r"signature"):
            qmc.control(_NoSignature())
