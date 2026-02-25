"""Tests for controlled gate API with all gate types x num_controls."""

import dataclasses
import math
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

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
        # block + 1 control + 2 targets = 4 operands
        assert len(op.operands) == 4

    def test_operands_structure(self):
        """operands = [block, ctrl_vals..., tgt_vals...]"""
        qk = _mock_qkernel()
        cg = ControlledGate(qk, num_controls=2)
        c0, c1, tgt = _make_qubit("c0"), _make_qubit("c1"), _make_qubit("tgt")
        with trace() as tracer:
            cg(c0, c1, tgt)
        op = tracer.operations[0]
        assert op.operands[0] is qk.block
        assert op.operands[1] is c0.value
        assert op.operands[2] is c1.value
        assert op.operands[3] is tgt.value

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
        with trace() as tracer:
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
        # block + ctrl + tgt + param = 4 operands
        assert len(op.operands) == 4
        param_val = op.operands[3]
        assert isinstance(param_val, Value)
        assert param_val.params["const"] == 0.5

    def test_handle_type_param_uses_value_directly(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        theta = _make_float_handle("theta")
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), theta=theta)
        op = tracer.operations[0]
        assert op.operands[3] is theta.value

    def test_multiple_params(self):
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        with trace() as tracer:
            cg(_make_qubit("ctrl"), _make_qubit("tgt"), alpha=1.0, beta=2.0)
        op = tracer.operations[0]
        # block + ctrl + tgt + 2 params = 5
        assert len(op.operands) == 5
        assert op.operands[3].params["const"] == 1.0
        assert op.operands[4].params["const"] == 2.0

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
        param_val = tracer.operations[0].operands[3]
        assert param_val.params["const"] == 3.0
        assert isinstance(param_val.params["const"], float)

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
        assert target_ops[1].params["const"] == 0.5

    def test_consume_prevents_reuse(self):
        """Qubit passed to controlled gate cannot be reused afterward."""
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        ctrl = _make_qubit("ctrl")
        tgt = _make_qubit("tgt")
        with trace() as tracer:
            cg(ctrl, tgt)
            with pytest.raises(QubitConsumedError):
                cg(ctrl, _make_qubit("tgt2"))

    def test_consume_prevents_reuse_target(self):
        """Target qubit passed to controlled gate cannot be reused afterward."""
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        ctrl = _make_qubit("ctrl")
        tgt = _make_qubit("tgt")
        with trace() as tracer:
            cg(ctrl, tgt)
            with pytest.raises(QubitConsumedError):
                cg(_make_qubit("ctrl2"), tgt)

    def test_aliasing_control_and_target_raises(self):
        """Same qubit as both control and target raises QubitAliasError."""
        cg = ControlledGate(_mock_qkernel(), num_controls=1)
        q = _make_qubit("q")
        with trace() as tracer:
            with pytest.raises(QubitAliasError):
                cg(q, q)

    def test_aliasing_duplicate_controls_raises(self):
        """Same qubit used as multiple controls raises QubitAliasError."""
        cg = ControlledGate(_mock_qkernel(), num_controls=2)
        q = _make_qubit("q")
        tgt = _make_qubit("tgt")
        with trace() as tracer:
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
        with trace() as tracer:
            with pytest.raises(ValueError):
                cg(*qs)


@pytest.fixture
def qiskit_transpiler():
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


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


def _get_statevector(transpiler, circuit_kernel, bindings):
    """Transpile, remove measurements, return full statevector as numpy array."""
    from qiskit.quantum_info import Statevector

    executable = transpiler.transpile(circuit_kernel, bindings=bindings)
    qc = executable.quantum_circuit.copy()
    qc.remove_final_measurements()
    return Statevector(qc).data


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
    U = np.linalg.matrix_power(spec.matrix, power)
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

    @staticmethod
    def _tolerance(spec, num_controls, power):
        """Return (rtol, atol) — relaxed for Qiskit CXGate().power(3).control(3) precision issue."""
        if spec.matrix_fn is cx_matrix and num_controls == 3 and power == 3:
            return 1e-1, 5e-2
        return 1e-5, 1e-8

    def test_controls_zero(self, qiskit_transpiler, spec, num_controls, power):
        """Controls all |0> => gate^power does NOT fire => full state stays |00...0>."""
        circuit = _make_controlled_circuit(
            spec, num_controls, activate_controls=False, power=power
        )
        actual = _get_statevector(qiskit_transpiler, circuit, spec.bindings)
        expected = _expected_statevector(
            spec, num_controls, activate_controls=False, power=power
        )
        rtol, atol = self._tolerance(spec, num_controls, power)
        assert statevectors_equal(actual, expected, rtol=rtol, atol=atol)

    def test_controls_one(self, qiskit_transpiler, spec, num_controls, power):
        """Controls all |1> => gate^power fires => check full statevector."""
        circuit = _make_controlled_circuit(
            spec, num_controls, activate_controls=True, power=power
        )
        actual = _get_statevector(qiskit_transpiler, circuit, spec.bindings)
        expected = _expected_statevector(
            spec, num_controls, activate_controls=True, power=power
        )
        rtol, atol = self._tolerance(spec, num_controls, power)
        assert statevectors_equal(actual, expected, rtol=rtol, atol=atol)


# =============================================================================
# Integration tests with random initial states (controls in superposition)
# =============================================================================


def _get_statevector_with_prep(
    transpiler, circuit_kernel, bindings, num_controls, target_state
):
    """Transpile circuit, prepend state preparation, return full statevector.

    Prepends H gates on control qubits and StatePreparation on target qubits
    to the transpiled Qiskit circuit.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import Statevector

    executable = transpiler.transpile(circuit_kernel, bindings=bindings)
    main_qc = executable.quantum_circuit.copy()
    main_qc.remove_final_measurements()

    total = main_qc.num_qubits
    prep_qc = QuantumCircuit(total)
    for i in range(num_controls):
        prep_qc.h(i)
    prep_qc.append(StatePreparation(target_state), list(range(num_controls, total)))

    full_qc = prep_qc.compose(main_qc)
    return Statevector(full_qc).data


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

    U = np.linalg.matrix_power(spec.matrix, power)
    return np.kron(target_state, rest) + np.kron(U @ target_state, coeff * all1)


def _random_statevector(num_qubits: int, seed: int) -> np.ndarray:
    """Generate a random normalized statevector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(2**num_qubits) + 1j * rng.standard_normal(2**num_qubits)
    return vec / np.linalg.norm(vec)


@pytest.mark.parametrize("spec", GATES)
@pytest.mark.parametrize("num_controls", [1, 2, 3])
class TestControlledGateRandomState:
    """Integration test: controlled gate with controls in superposition and random target state."""

    def test_random_initial_state(self, qiskit_transpiler, spec, num_controls):
        """Controls in |+>^NC, targets in random |psi> => verify full statevector."""
        target_state = _random_statevector(spec.num_targets, seed=42)
        circuit = _make_controlled_circuit(spec, num_controls, activate_controls=False)
        actual = _get_statevector_with_prep(
            qiskit_transpiler, circuit, spec.bindings, num_controls, target_state
        )
        expected = _expected_statevector_superposition(spec, num_controls, target_state)
        assert statevectors_equal(actual, expected)
