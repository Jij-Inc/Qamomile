"""Tests for controlled gate API with all gate types x num_controls."""

import dataclasses
import math
from collections.abc import Callable
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from qamomile.qiskit import QiskitTranspiler

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle.primitives import Float, Qubit
from qamomile.circuit.frontend.operation.controlled import ControlledGate, controlled
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.types.primitives import FloatType, QubitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import QubitAliasError, QubitConsumedError
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


def _mock_qkernel():
    """Create a mock QKernel with a block property."""
    qk = MagicMock()
    qk.block = MagicMock(name="block_value")
    return qk


# =============================================================================
# Unit tests for controlled() function
# =============================================================================


class TestControlledFunction:
    def test_returns_controlled_gate(self):
        cg = controlled(_mock_qkernel())
        assert isinstance(cg, ControlledGate)

    def test_default_num_controls(self):
        cg = controlled(_mock_qkernel())
        assert cg._num_controls == 1

    @pytest.mark.parametrize("nc", [1, 2, 3, 5])
    def test_custom_num_controls(self, nc):
        cg = controlled(_mock_qkernel(), num_controls=nc)
        assert cg._num_controls == nc

    def test_stores_qkernel(self):
        qk = _mock_qkernel()
        cg = controlled(qk)
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
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
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
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), theta=0.5)
        op = tracer.operations[0]
        assert len(op.operands) == 3
        param_val = op.operands[2]
        assert isinstance(param_val, Value)
        assert param_val.get_const() == 0.5

    def test_handle_type_param_uses_value_directly(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        theta = _make_float_handle("theta")
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), theta=theta)
        op = tracer.operations[0]
        assert op.operands[2] is theta.value

    def test_multiple_params(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
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
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), theta=3)
        param_val = tracer.operations[0].operands[2]
        assert param_val.get_const() == 3.0
        assert isinstance(param_val.get_const(), float)

    def test_target_operands_with_params(self):
        """target_operands includes parameter Values when params are passed."""
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
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
        """Same qubit as both control and target raises QubitAliasError."""
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        q = _make_qubit("q")
        with trace() as tracer:  # noqa: F841
            with pytest.raises(QubitAliasError):
                cg(q, q)

    def test_aliasing_duplicate_controls_raises(self):
        """Same qubit used as multiple controls raises QubitAliasError."""
        cg = ControlledGate(_mock_qkernel(), num_controls=2)
        q = _make_qubit("q")
        tgt = _make_qubit("tgt")
        with trace() as tracer:  # noqa: F841
            with pytest.raises(QubitAliasError):
                cg(q, q, tgt)


# =============================================================================
# Validation tests for controlled() / ControlledGate
# =============================================================================


class TestControlledValidation:
    def test_num_controls_zero_raises(self):
        with pytest.raises(ValueError):
            controlled(_mock_qkernel(), num_controls=0)

    def test_num_controls_negative_raises(self):
        with pytest.raises(ValueError):
            controlled(_mock_qkernel(), num_controls=-1)

    def test_not_enough_args_raises(self):
        """Passing fewer qubits than num_controls (no targets) raises ValueError."""
        cg = ControlledGate(_mock_qkernel(), num_controls=3)
        qs = [_make_qubit(f"q{i}") for i in range(3)]  # 3 controls, 0 targets
        with trace() as tracer:  # noqa: F841
            with pytest.raises(ValueError):
                cg(*qs)


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
            cg = qmc.controlled(spec.kernel, num_controls=num_controls)
            out = cg(*qs, theta=theta, power=power)
            return qmc.measure(out[-1])

    else:

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            qs = [qmc.qubit(name=f"q{i}") for i in range(total)]
            qs = _prep(qs)
            cg = qmc.controlled(spec.kernel, num_controls=num_controls)
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
    """Integration test: controlled(gate, num_controls) for all gate x control combos."""

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
    """Integration test: controlled(gate, num_controls, power) for all gate x control x power combos."""

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
# Built-in gate acceptance: controlled(qmc.rx) without an @qkernel wrapper
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

    from qamomile.quri_parts import QuriPartsTranspiler as _QuriPartsTranspilerCheck  # noqa: F401
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


# -- Acceptance: controlled() factory takes built-ins -------------------------


class TestControlledAcceptsBuiltinGate:
    """``controlled()`` accepts built-in gate functions directly."""

    @pytest.mark.parametrize(
        "gate_fn",
        [qmc.h, qmc.x, qmc.y, qmc.z, qmc.s, qmc.t, qmc.rx, qmc.ry, qmc.rz, qmc.p],
    )
    def test_single_qubit_gates_accepted(self, gate_fn):
        cg = qmc.controlled(gate_fn)
        assert isinstance(cg, ControlledGate)

    @pytest.mark.parametrize("gate_fn", [qmc.cx, qmc.cz, qmc.swap, qmc.cp, qmc.rzz])
    def test_two_qubit_gates_accepted(self, gate_fn):
        cg = qmc.controlled(gate_fn)
        assert isinstance(cg, ControlledGate)

    def test_three_qubit_gate_accepted(self):
        cg = qmc.controlled(qmc.ccx)
        assert isinstance(cg, ControlledGate)

    @pytest.mark.parametrize("nc", [1, 2, 3])
    def test_built_in_with_num_controls(self, nc):
        cg = qmc.controlled(qmc.rx, num_controls=nc)
        assert cg._num_controls == nc

    def test_qkernel_passthrough_uses_same_instance(self):
        """A QKernel argument is passed straight through, not re-wrapped."""

        @qmc.qkernel
        def my_gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        cg = qmc.controlled(my_gate)
        assert cg._qkernel is my_gate


# -- Rejection: errors for unsupported callables -----------------------------


class TestControlledBuiltinErrors:
    """Errors raised when a callable cannot be auto-wrapped."""

    def test_non_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="QKernel or a built-in gate"):
            qmc.controlled(42)  # type: ignore[arg-type]

    def test_missing_annotation_raises_type_error(self):
        def bad(q):  # no annotation
            return qmc.h(q)

        with pytest.raises(TypeError, match="no type annotation"):
            qmc.controlled(bad)

    def test_unsupported_annotation_raises_type_error(self):
        def bad(q: str) -> str:  # str is not Qubit/Float/UInt
            return q

        with pytest.raises(TypeError, match="annotation"):
            qmc.controlled(bad)

    def test_no_qubit_param_raises_type_error(self):
        def classical_only(theta: float) -> float:
            return theta

        with pytest.raises(TypeError, match="no Qubit parameters"):
            qmc.controlled(classical_only)

    def test_var_args_raises_type_error(self):
        def variadic(*qs: qmc.Qubit) -> qmc.Qubit:
            return qs[0]

        with pytest.raises(TypeError, match="\\*args/\\*\\*kwargs"):
            qmc.controlled(variadic)


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
        """controlled(qmc.rx).block has the same gate sequence as controlled(_rx_gate).block."""
        cg_builtin = qmc.controlled(qmc.rx)
        cg_wrapped = qmc.controlled(_rx_gate)
        assert self._gate_kinds(cg_builtin._qkernel.block) == self._gate_kinds(
            cg_wrapped._qkernel.block
        )

    def test_h_gate_block_matches_wrapper(self):
        cg_builtin = qmc.controlled(qmc.h)
        cg_wrapped = qmc.controlled(_h_gate)
        assert self._gate_kinds(cg_builtin._qkernel.block) == self._gate_kinds(
            cg_wrapped._qkernel.block
        )

    def test_cp_gate_block_matches_wrapper(self):
        cg_builtin = qmc.controlled(qmc.cp)
        cg_wrapped = qmc.controlled(_cp_gate)
        assert self._gate_kinds(cg_builtin._qkernel.block) == self._gate_kinds(
            cg_wrapped._qkernel.block
        )

    def test_cx_gate_block_matches_wrapper(self):
        cg_builtin = qmc.controlled(qmc.cx)
        cg_wrapped = qmc.controlled(_cx_gate)
        assert self._gate_kinds(cg_builtin._qkernel.block) == self._gate_kinds(
            cg_wrapped._qkernel.block
        )


# -- Cross-backend execution: built-in form transpiles + runs ----------------


def _make_controlled_builtin_circuit(gate_fn, theta: float | None = None):
    """Build a kernel that flips control(s) to |1> then applies ``controlled(gate_fn)``."""
    if theta is None:

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            c = qmc.x(c)
            cg = qmc.controlled(gate_fn)
            c, t = cg(c, t)
            return qmc.measure(t)

    else:

        @qmc.qkernel
        def circuit(angle: qmc.Float) -> qmc.Bit:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            c = qmc.x(c)
            cg = qmc.controlled(gate_fn)
            c, t = cg(c, t, angle=angle)
            return qmc.measure(t)

    return circuit


@pytest.mark.parametrize("transpiler_factory", _BUILTIN_BACKENDS)
class TestControlledBuiltinCrossBackend:
    """controlled(builtin) transpiles + samples on every supported backend."""

    def test_sample_controlled_x_flips_target(self, transpiler_factory):
        """controlled(qmc.x) with control=|1> flips the target every shot."""
        circuit = _make_controlled_builtin_circuit(qmc.x, theta=None)
        t = transpiler_factory()
        exe = t.transpile(circuit)
        results = exe.sample(t.executor(), shots=64).result().results
        for value, _count in results:
            assert value == 1, (
                f"expected 1, got {value} on {transpiler_factory().__class__.__name__}"
            )

    def test_sample_controlled_rx_pi_flips_target(self, transpiler_factory):
        """controlled(qmc.rx) with control=|1> and angle=pi flips the target every shot.

        ``angle`` is bound at transpile time (rather than promoted to a
        runtime parameter) because QuriParts' controlled-rotation emit
        path does not yet support symbolic angles inside controlled
        gates — see ``LinearMappedParametricQuantumCircuit`` errors when
        binding parameters used inside CRX.  This test is about the
        wrapper-synthesis path, not about parametric controlled-rotation
        support, so a concrete angle keeps the assertion meaningful on
        every backend.
        """
        circuit = _make_controlled_builtin_circuit(qmc.rx, theta=math.pi)
        t = transpiler_factory()
        exe = t.transpile(circuit, bindings={"angle": math.pi})
        results = exe.sample(t.executor(), shots=64).result().results
        for value, _count in results:
            assert value == 1


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
class TestControlledBuiltinStatevectorParity:
    """Built-in form produces the same Qiskit statevector as the wrapped form."""

    @pytest.mark.parametrize("num_controls", [1, 2])
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_rx_builtin_matches_wrapper(self, qiskit_transpiler, num_controls, seed):
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-math.pi, math.pi))

        spec_builtin = GateSpec(qmc.rx, 1, theta, rx_matrix)
        spec_wrapper = GateSpec(_rx_gate, 1, theta, rx_matrix)

        circ_builtin = _make_controlled_circuit(
            spec_builtin, num_controls, activate_controls=True
        )
        circ_wrapper = _make_controlled_circuit(
            spec_wrapper, num_controls, activate_controls=True
        )

        sv_builtin = _get_statevector(
            qiskit_transpiler, circ_builtin, spec_builtin.bindings
        )
        sv_wrapper = _get_statevector(
            qiskit_transpiler, circ_wrapper, spec_wrapper.bindings
        )
        assert statevectors_equal(sv_builtin, sv_wrapper)

    @pytest.mark.parametrize("num_controls", [1, 2])
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_cp_builtin_matches_wrapper(self, qiskit_transpiler, num_controls, seed):
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-math.pi, math.pi))

        spec_builtin = GateSpec(qmc.cp, 2, theta, cp_matrix)
        spec_wrapper = GateSpec(_cp_gate, 2, theta, cp_matrix)

        circ_builtin = _make_controlled_circuit(
            spec_builtin, num_controls, activate_controls=True
        )
        circ_wrapper = _make_controlled_circuit(
            spec_wrapper, num_controls, activate_controls=True
        )

        sv_builtin = _get_statevector(
            qiskit_transpiler, circ_builtin, spec_builtin.bindings
        )
        sv_wrapper = _get_statevector(
            qiskit_transpiler, circ_wrapper, spec_wrapper.bindings
        )
        assert statevectors_equal(sv_builtin, sv_wrapper)

    def test_h_builtin_matches_wrapper(self, qiskit_transpiler):
        spec_builtin = GateSpec(qmc.h, 1, None, h_matrix)
        spec_wrapper = GateSpec(_h_gate, 1, None, h_matrix)

        circ_builtin = _make_controlled_circuit(spec_builtin, 1, activate_controls=True)
        circ_wrapper = _make_controlled_circuit(spec_wrapper, 1, activate_controls=True)

        sv_builtin = _get_statevector(qiskit_transpiler, circ_builtin, {})
        sv_wrapper = _get_statevector(qiskit_transpiler, circ_wrapper, {})
        assert statevectors_equal(sv_builtin, sv_wrapper)


# -- Expectation-value path: ensure the estimator pipeline accepts builtins --


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
class TestControlledBuiltinExpval:
    """Built-in form is exercised through the expectation-value path too."""

    def test_expval_matches_wrapper(self, qiskit_transpiler):
        """<Z_t> on (X|1>)|0> after controlled-X should equal -1 for both forms."""
        import qamomile.observable as qm_o

        @qmc.qkernel
        def circuit_builtin(obs: qmc.Observable) -> qmc.Float:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            c = qmc.x(c)
            cg = qmc.controlled(qmc.x)
            c, t = cg(c, t)
            return qmc.expval((c, t), obs)

        @qmc.qkernel
        def circuit_wrapper(obs: qmc.Observable) -> qmc.Float:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            c = qmc.x(c)
            cg = qmc.controlled(_x_gate)
            c, t = cg(c, t)
            return qmc.expval((c, t), obs)

        # Z on the target qubit (index 1).
        H = qm_o.Hamiltonian.zero(num_qubits=2)
        H += qm_o.Z(1)

        exe_b = qiskit_transpiler.transpile(circuit_builtin, bindings={"obs": H})
        exe_w = qiskit_transpiler.transpile(circuit_wrapper, bindings={"obs": H})

        val_b = exe_b.run(qiskit_transpiler.executor()).result()
        val_w = exe_w.run(qiskit_transpiler.executor()).result()

        # Both controls=|1> and target=|0> → CX flips target → final state |11>
        # → <Z_target> on |1> = -1.
        assert np.isclose(val_b, -1.0, atol=1e-6)
        assert np.isclose(val_w, -1.0, atol=1e-6)
        assert np.isclose(val_b, val_w, atol=1e-9)
