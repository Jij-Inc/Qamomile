"""Regression tests for symbolic controlled-U resource allocation.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The qkernel AST transformer relies on resolved type annotations.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit import modular_decrement
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.resource_allocator import (
    ResourceAllocator,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)


@qmc.qkernel
def symbolic_indexed_mcz_kernel(
    total: qmc.UInt,
    control_qubit: qmc.UInt,
    signal_start: qmc.UInt,
    num_signal: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Apply an MCZ with a scalar symbolic-index control.

    Args:
        total (qmc.UInt): Number of qubits in the allocated register.
        control_qubit (qmc.UInt): Scalar control index into ``q``.
        signal_start (qmc.UInt): Start index of the signal slice.
        num_signal (qmc.UInt): Number of signal qubits, also used as
            the symbolic control count.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement bits for the whole register.
    """
    q = qmc.qubit_array(total, name="q")
    target = signal_start + num_signal - 1
    controls_start = signal_start
    controls_stop = signal_start + num_signal - 1
    mcz = qmc.control(qmc.z, num_controls=num_signal)
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit],
        q[controls_start:controls_stop],
        q[target],
    )
    return qmc.measure(q)


@qmc.qkernel
def repeated_symbolic_mcz_kernel(
    total: qmc.UInt,
    control_qubit: qmc.UInt,
    signal_start: qmc.UInt,
    num_signal: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Apply the symbolic-index MCZ shape eight times.

    Args:
        total (qmc.UInt): Number of qubits in the allocated register.
        control_qubit (qmc.UInt): Scalar control index into ``q``.
        signal_start (qmc.UInt): Start index of the signal slice.
        num_signal (qmc.UInt): Number of signal qubits, also used as
            the symbolic control count.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement bits for the whole register.
    """
    q = qmc.qubit_array(total, name="q")
    target = signal_start + num_signal - 1
    controls_start = signal_start
    controls_stop = signal_start + num_signal - 1
    mcz = qmc.control(qmc.z, num_controls=num_signal)
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit], q[controls_start:controls_stop], q[target]
    )
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit], q[controls_start:controls_stop], q[target]
    )
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit], q[controls_start:controls_stop], q[target]
    )
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit], q[controls_start:controls_stop], q[target]
    )
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit], q[controls_start:controls_stop], q[target]
    )
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit], q[controls_start:controls_stop], q[target]
    )
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit], q[controls_start:controls_stop], q[target]
    )
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit], q[controls_start:controls_stop], q[target]
    )
    return qmc.measure(q)


@qmc.qkernel
def symbolic_mcz_then_use_control_result(
    total: qmc.UInt,
    control_qubit: qmc.UInt,
    signal_start: qmc.UInt,
    num_signal: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Use the scalar control result after symbolic-index MCZ.

    Args:
        total (qmc.UInt): Number of qubits in the allocated register.
        control_qubit (qmc.UInt): Scalar control index into ``q``.
        signal_start (qmc.UInt): Start index of the signal slice.
        num_signal (qmc.UInt): Number of signal qubits, also used as
            the symbolic control count.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement bits for the whole register.
    """
    q = qmc.qubit_array(total, name="q")
    target = signal_start + num_signal - 1
    controls_start = signal_start
    controls_stop = signal_start + num_signal - 1
    mcz = qmc.control(qmc.z, num_controls=num_signal)
    q[control_qubit], q[controls_start:controls_stop], q[target] = mcz(
        q[control_qubit],
        q[controls_start:controls_stop],
        q[target],
    )
    q[control_qubit] = qmc.x(q[control_qubit])
    return qmc.measure(q)


@qmc.qkernel
def loop_symbolic_scalar_control_mcz_kernel(
    total: qmc.UInt,
    num_offsets: qmc.UInt,
    num_controls: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Apply symbolic-count MCZ with a loop-indexed scalar control.

    Args:
        total (qmc.UInt): Number of qubits in the allocated register.
        num_offsets (qmc.UInt): Number of loop iterations to emit.
        num_controls (qmc.UInt): Symbolic control count for the MCZ.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement bits for the whole register.
    """
    q = qmc.qubit_array(total, name="q")
    for offset in qmc.range(num_offsets):
        mcz = qmc.control(qmc.z, num_controls=num_controls)
        q[offset], q[3:4], q[4] = mcz(q[offset], q[3:4], q[4])
    return qmc.measure(q)


@qmc.qkernel
def constant_index_symbolic_num_controls_kernel(
    num_controls: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Apply literal-index MCZ with a bound symbolic control count.

    Args:
        num_controls (qmc.UInt): Symbolic control count bound at
            transpile time.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement bits for the whole register.
    """
    q = qmc.qubit_array(6, name="q")
    mcz = qmc.control(qmc.z, num_controls=num_controls)
    q[5], q[3:4], q[4] = mcz(q[5], q[3:4], q[4])
    return qmc.measure(q)


@qmc.qkernel
def constant_indexed_mcz_kernel() -> qmc.Vector[qmc.Bit]:
    """Apply the same MCZ shape with literal indices and count.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement bits for the whole register.
    """
    q = qmc.qubit_array(6, name="q")
    mcz = qmc.control(qmc.z, num_controls=2)
    q[5], q[3:4], q[4] = mcz(q[5], q[3:4], q[4])
    return qmc.measure(q)


@qmc.qkernel
def one_x(q: qmc.Qubit) -> qmc.Qubit:
    """Apply X to one scalar qubit.

    Args:
        q (qmc.Qubit): Input qubit.

    Returns:
        qmc.Qubit: Updated qubit after applying X.
    """
    q = qmc.x(q)
    return q


@qmc.qkernel
def symbolic_controlled_x_kernel(
    total: qmc.UInt,
    control_qubit: qmc.UInt,
    target_qubit: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Apply controlled-X to symbolic-index scalar operands.

    Args:
        total (qmc.UInt): Number of qubits in the allocated register.
        control_qubit (qmc.UInt): Scalar control index into ``q``.
        target_qubit (qmc.UInt): Scalar target index into ``q``.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement bits for the whole register.
    """
    q = qmc.qubit_array(total, name="q")
    cx = qmc.control(qmc.x, num_controls=1)
    q[control_qubit], q[target_qubit] = cx(q[control_qubit], q[target_qubit])
    return qmc.measure(q)


@qmc.qkernel
def symbolic_inverse_kernel(
    total: qmc.UInt,
    target_qubit: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Apply inverse subkernel to a symbolic-index scalar operand.

    Args:
        total (qmc.UInt): Number of qubits in the allocated register.
        target_qubit (qmc.UInt): Scalar target index into ``q``.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement bits for the whole register.
    """
    q = qmc.qubit_array(total, name="q")
    q[target_qubit] = qmc.inverse(one_x)(q[target_qubit])
    return qmc.measure(q)


@qmc.qkernel
def controlled_modular_decrement_kernel(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Apply controlled modular decrement to an n-qubit register.

    Args:
        n (qmc.UInt): Number of qubits in the modular register.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement bits for the modular register.
    """
    q = qmc.qubit_array(n, name="q")
    c = qmc.qubit(name="c")
    c = qmc.x(c)
    shift = qmc.control(modular_decrement, num_controls=1)
    c, q = shift(c, q)
    return qmc.measure(q)


@qmc.qkernel
def scalar_input_symbolic_control_then_use(
    ctrl: qmc.Qubit,
    other_ctrl: qmc.Qubit,
    target: qmc.Qubit,
    num_controls: qmc.UInt,
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply symbolic controlled-X to scalar inputs and reuse the control.

    Args:
        ctrl (qmc.Qubit): First scalar control qubit.
        other_ctrl (qmc.Qubit): Second scalar control qubit.
        target (qmc.Qubit): Target qubit.
        num_controls (qmc.UInt): Symbolic control count.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]: Updated first control,
        second control, and target qubits.
    """
    cx = qmc.control(qmc.x, num_controls=num_controls)
    ctrl, other_ctrl, target = cx(ctrl, other_ctrl, target)
    ctrl = qmc.h(ctrl)
    return ctrl, other_ctrl, target


def _symbolic_mcz_bindings() -> dict[str, int]:
    """Return bindings for a six-qubit symbolic-index MCZ.

    Returns:
        dict[str, int]: Bindings that make the MCZ use qubits 3, 4,
        and 5 of a six-qubit register.
    """
    return {
        "total": 6,
        "control_qubit": 5,
        "signal_start": 3,
        "num_signal": 2,
    }


def _qiskit_circuit(kernel, bindings: dict | None = None):
    """Transpile a kernel and return the first Qiskit circuit.

    Args:
        kernel: QKernel to transpile with the Qiskit backend.
        bindings (dict | None): Compile-time bindings passed to
            ``transpile``. Defaults to None.

    Returns:
        QuantumCircuit: First emitted Qiskit circuit.
    """
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    exe = QiskitTranspiler().transpile(kernel, bindings=bindings)
    return exe.compiled_quantum[0].circuit


def _quri_parts_width(kernel, bindings: dict | None = None) -> int:
    """Transpile a kernel and return the QURI Parts circuit width.

    Args:
        kernel: QKernel to transpile with the QURI Parts backend.
        bindings (dict | None): Compile-time bindings passed to
            ``transpile``. Defaults to None.

    Returns:
        int: Number of qubits in the emitted QURI Parts circuit.
    """
    pytest.importorskip("quri_parts")
    from qamomile.quri_parts import QuriPartsTranspiler

    exe = QuriPartsTranspiler().transpile(kernel, bindings=bindings)
    return exe.compiled_quantum[0].circuit.qubit_count


def _cudaq_width(kernel, bindings: dict | None = None) -> int:
    """Transpile a kernel and return the CUDA-Q artifact width.

    Args:
        kernel: QKernel to transpile with the CUDA-Q backend.
        bindings (dict | None): Compile-time bindings passed to
            ``transpile``. Defaults to None.

    Returns:
        int: Number of qubits in the emitted CUDA-Q artifact.
    """
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    exe = CudaqTranspiler().transpile(kernel, bindings=bindings)
    return exe.compiled_quantum[0].circuit.num_qubits


def test_qiskit_repeated_symbolic_mcz_does_not_allocate_phantoms():
    """Repeated symbolic scalar controls keep the emitted Qiskit width."""
    circuit = _qiskit_circuit(repeated_symbolic_mcz_kernel, _symbolic_mcz_bindings())
    assert circuit.num_qubits == 6


def test_qiskit_literal_index_symbolic_count_does_not_allocate_phantom():
    """A literal array element with symbolic count aliases its real wire."""
    circuit = _qiskit_circuit(
        constant_index_symbolic_num_controls_kernel,
        {"num_controls": 2},
    )
    assert circuit.num_qubits == 6


def test_qiskit_symbolic_mcz_then_use_control_result_stays_on_wire():
    """Using the scalar control result after MCZ keeps the original wire."""
    circuit = _qiskit_circuit(
        symbolic_mcz_then_use_control_result,
        _symbolic_mcz_bindings(),
    )
    assert circuit.num_qubits == 6

    x_ops = [inst for inst in circuit.data if inst.operation.name == "x"]
    assert len(x_ops) == 1
    assert [circuit.find_bit(qubit).index for qubit in x_ops[0].qubits] == [5]


def test_qiskit_loop_symbolic_scalar_control_does_not_allocate_phantom():
    """Loop-local scalar controls receive bindings at emit without phantoms."""
    circuit = _qiskit_circuit(
        loop_symbolic_scalar_control_mcz_kernel,
        {"total": 6, "num_offsets": 2, "num_controls": 2},
    )
    assert circuit.num_qubits == 6


def test_qiskit_unresolved_symbolic_control_index_raises_emit_error():
    """Leaving the symbolic scalar control index unbound fails explicitly."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    with pytest.raises(EmitError, match="Cannot resolve scalar quantum operand"):
        QiskitTranspiler().transpile(
            symbolic_indexed_mcz_kernel,
            bindings={"total": 6, "signal_start": 3, "num_signal": 2},
        )


@pytest.mark.parametrize(
    ("kernel", "bindings", "expected_width"),
    [
        (constant_indexed_mcz_kernel, {}, 6),
        (
            symbolic_controlled_x_kernel,
            {"total": 6, "control_qubit": 5, "target_qubit": 4},
            6,
        ),
        (symbolic_inverse_kernel, {"total": 6, "target_qubit": 4}, 6),
    ],
)
def test_qiskit_non_offender_symbolic_paths_keep_width(
    kernel,
    bindings: dict,
    expected_width: int,
):
    """Neighboring symbolic paths keep their existing emitted width."""
    circuit = _qiskit_circuit(kernel, bindings)
    assert circuit.num_qubits == expected_width


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 5])
def test_qiskit_controlled_modular_decrement_keeps_expected_width(n_qubits: int):
    """Controlled modular decrement itself does not allocate phantom qubits."""
    circuit = _qiskit_circuit(
        controlled_modular_decrement_kernel,
        {"n": n_qubits},
    )
    assert circuit.num_qubits == n_qubits + 1


@pytest.mark.parametrize(
    ("kernel", "bindings", "expected_width"),
    [
        (repeated_symbolic_mcz_kernel, _symbolic_mcz_bindings(), 6),
        (constant_index_symbolic_num_controls_kernel, {"num_controls": 2}, 6),
        (symbolic_mcz_then_use_control_result, _symbolic_mcz_bindings(), 6),
    ],
)
@pytest.mark.quri_parts
def test_quri_parts_symbolic_controlled_u_does_not_allocate_phantoms(
    kernel,
    bindings: dict,
    expected_width: int,
):
    """Shared allocator fixes symbolic controlled-U width for QURI Parts."""
    assert _quri_parts_width(kernel, bindings) == expected_width


@pytest.mark.parametrize(
    ("kernel", "bindings", "expected_width"),
    [
        (repeated_symbolic_mcz_kernel, _symbolic_mcz_bindings(), 6),
        (constant_index_symbolic_num_controls_kernel, {"num_controls": 2}, 6),
        (symbolic_mcz_then_use_control_result, _symbolic_mcz_bindings(), 6),
    ],
)
@pytest.mark.cudaq
def test_cudaq_symbolic_controlled_u_does_not_allocate_phantoms(
    kernel,
    bindings: dict,
    expected_width: int,
):
    """Shared allocator fixes symbolic controlled-U width for CUDA-Q."""
    assert _cudaq_width(kernel, bindings) == expected_width


def test_scalar_input_symbolic_control_allocator_preserves_fresh_result_alias():
    """Fresh scalar controls still map their results for later use."""
    block = scalar_input_symbolic_control_then_use.build(parameters=["num_controls"])
    qubit_map, _ = ResourceAllocator(ValueResolver()).allocate(block.operations)
    assert max(qubit_map.values()) + 1 == 3
