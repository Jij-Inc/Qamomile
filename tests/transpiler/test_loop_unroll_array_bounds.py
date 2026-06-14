"""Regression tests for out-of-range array indexing during loop unrolling.

A loop-unrolled access like ``angles[i]`` whose concrete iteration index
exceeds the compile-time bound array used to be silently swallowed by the
emit-time ``ValueResolver``: every out-of-range iteration fell through to
symbolic-parameter creation and shared a single phantom runtime parameter
named ``angles[i]``. Binding that phantom then ran all out-of-range
iterations with one shared value — a silent-wrong-result hazard.

Transpilation must instead fail fast with an ``EmitError`` naming the
array, index, and length, while in-range unrolling and runtime parameter
arrays keep working unchanged.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)
from qamomile.qiskit.transpiler import QiskitTranspiler

# ==============================================================================
# Kernel definitions at module level (required for inspect.getsource to work)
# ==============================================================================


@qmc.qkernel
def rx_loop_indexed(
    rx_angles: qmc.Vector[qmc.Float], num_rotations: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    """Apply ``rx(rx_angles[i])`` once per loop iteration on a single qubit."""
    q = qmc.qubit_array(1, "q")
    for i in qmc.range(num_rotations):
        q[0] = qmc.rx(q[0], rx_angles[i])
    return qmc.measure(q)


# ==============================================================================
# End-to-end transpile behaviour
# ==============================================================================


def test_out_of_range_loop_index_raises_at_transpile():
    """Unrolling past the bound array length fails at transpile time with an
    error naming the array, the offending index, and the array length."""
    transpiler = QiskitTranspiler()

    with pytest.raises(
        EmitError,
        match=r"Index 3 is out of range .* 'rx_angles' of length 3",
    ):
        transpiler.transpile(
            rx_loop_indexed,
            bindings={"rx_angles": [0.1, 0.2, 0.3], "num_rotations": 5},
        )


def test_in_range_loop_unrolling_binds_each_element():
    """In-range unrolling still bakes each bound angle into its own gate."""
    transpiler = QiskitTranspiler()

    exe = transpiler.transpile(
        rx_loop_indexed,
        bindings={"rx_angles": [0.1, 0.2, 0.3], "num_rotations": 3},
    )

    circuit = exe.compiled_quantum[0].circuit
    emitted_angles = [
        float(inst.operation.params[0])
        for inst in circuit.data
        if inst.operation.name == "rx"
    ]
    assert emitted_angles == [0.1, 0.2, 0.3]


def test_runtime_parameter_array_keeps_per_element_parameters():
    """A runtime parameter array still yields one runtime parameter per
    unrolled element and stays bindable at sample time."""
    transpiler = QiskitTranspiler()

    exe = transpiler.transpile(
        rx_loop_indexed,
        bindings={"num_rotations": 3},
        parameters=["rx_angles"],
    )

    circuit = exe.compiled_quantum[0].circuit
    assert sorted(p.name for p in circuit.parameters) == [
        "rx_angles[0]",
        "rx_angles[1]",
        "rx_angles[2]",
    ]

    job = exe.sample(
        transpiler.executor(),
        bindings={"rx_angles": [0.1, 0.2, 0.3]},
        shots=50,
    )
    assert len(job.result().results) > 0


# ==============================================================================
# Emit-time ValueResolver unit behaviour
# ==============================================================================


def _array_element(index: Value, name: str = "angles") -> Value:
    """Create an element Value of a length-3 float array.

    Args:
        index (Value): Index Value for the element access. May be
            constant or symbolic.
        name (str): Parent array name used for binding lookup. Defaults
            to ``"angles"``.

    Returns:
        Value: A Value representing ``name[index]``.
    """
    parent = ArrayValue(
        type=FloatType(),
        name=name,
        shape=(Value(type=UIntType(), name="dim").with_const(3),),
    )
    return Value(
        type=FloatType(),
        name=f"{name}[i]",
        parent_array=parent,
        element_indices=(index,),
    )


class TestIndexIntoBoundArray:
    def test_out_of_range_concrete_index_raises(self):
        """A concrete out-of-range index into a bound array raises EmitError."""
        resolver = ValueResolver()
        element = _array_element(Value(type=UIntType(), name="i").with_const(5))

        with pytest.raises(
            EmitError, match=r"Index 5 is out of range .* 'angles' of length 3"
        ):
            resolver.resolve_classical_value(
                element, bindings={"angles": [0.1, 0.2, 0.3]}
            )

    def test_in_range_concrete_index_resolves(self):
        """Sanity check: an in-range concrete index resolves to the element."""
        resolver = ValueResolver()
        element = _array_element(Value(type=UIntType(), name="i").with_const(1))

        resolved = resolver.resolve_classical_value(
            element, bindings={"angles": [0.1, 0.2, 0.3]}
        )
        assert resolved == 0.2

    def test_symbolic_index_stays_unresolved(self):
        """An unresolved symbolic index keeps falling back to ``None``."""
        resolver = ValueResolver()
        element = _array_element(Value(type=UIntType(), name="i"))

        resolved = resolver.resolve_classical_value(
            element, bindings={"angles": [0.1, 0.2, 0.3]}
        )
        assert resolved is None

    def test_runtime_parameter_array_is_never_indexed(self):
        """A runtime parameter array returns ``None`` even for an
        out-of-range concrete index: its bound data is a placeholder."""
        resolver = ValueResolver(parameters={"angles"})
        element = _array_element(Value(type=UIntType(), name="i").with_const(5))

        resolved = resolver.resolve_classical_value(
            element, bindings={"angles": [0.1, 0.2, 0.3]}
        )
        assert resolved is None
