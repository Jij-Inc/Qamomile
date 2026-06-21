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

import numpy as np
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


@pytest.mark.parametrize(
    ("array_len", "num_rotations"),
    [(3, 5), (1, 2), (2, 4), (4, 7)],
)
def test_out_of_range_loop_index_raises_at_transpile(array_len, num_rotations):
    """Unrolling past the bound array length fails at transpile time.

    The first out-of-range iteration is ``i == array_len`` (indices
    ``0 .. array_len - 1`` are valid), so the error must name that index
    and the array length, regardless of how many extra iterations follow.
    """
    transpiler = QiskitTranspiler()
    angles = [0.1 * (k + 1) for k in range(array_len)]

    with pytest.raises(
        EmitError,
        match=(
            rf"Index {array_len} is out of range .* "
            rf"'rx_angles' of length {array_len}"
        ),
    ):
        transpiler.transpile(
            rx_loop_indexed,
            bindings={"rx_angles": angles, "num_rotations": num_rotations},
        )


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("num_rotations", [1, 2, 3, 5])
def test_in_range_loop_unrolling_binds_each_element(seed, num_rotations):
    """In-range unrolling bakes each bound angle into its own gate.

    Exercised across register sizes and randomized angles (including the
    boundary angles 0 and pi) so the per-iteration binding is verified for
    more than one fixed vector.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=num_rotations).tolist()
    # Pin the boundary angles for the leading iterations where they fit.
    for boundary_index, boundary_angle in ((0, 0.0), (1, np.pi)):
        if boundary_index < num_rotations:
            angles[boundary_index] = boundary_angle
    transpiler = QiskitTranspiler()

    exe = transpiler.transpile(
        rx_loop_indexed,
        bindings={"rx_angles": angles, "num_rotations": num_rotations},
    )

    circuit = exe.compiled_quantum[0].circuit
    emitted_angles = [
        float(inst.operation.params[0])
        for inst in circuit.data
        if inst.operation.name == "rx"
    ]
    np.testing.assert_allclose(emitted_angles, angles, rtol=0, atol=1e-12)


@pytest.mark.parametrize("num_rotations", [1, 2, 3])
def test_runtime_parameter_array_keeps_per_element_parameters(num_rotations):
    """A runtime parameter array yields one runtime parameter per unrolled
    element and stays bindable at sample time, across register sizes."""
    rng = np.random.default_rng(num_rotations)
    angles = rng.uniform(-np.pi, np.pi, size=num_rotations).tolist()
    transpiler = QiskitTranspiler()

    exe = transpiler.transpile(
        rx_loop_indexed,
        bindings={"num_rotations": num_rotations},
        parameters=["rx_angles"],
    )

    circuit = exe.compiled_quantum[0].circuit
    assert {p.name for p in circuit.parameters} == {
        f"rx_angles[{k}]" for k in range(num_rotations)
    }

    job = exe.sample(
        transpiler.executor(),
        bindings={"rx_angles": angles},
        shots=50,
    )
    assert len(job.result().results) > 0


# ==============================================================================
# Emit-time ValueResolver unit behaviour
# ==============================================================================


def _array_element(index: Value, name: str = "angles", length: int = 3) -> Value:
    """Create an element Value of a one-dimensional float array.

    Args:
        index (Value): Index Value for the element access. May be
            constant or symbolic.
        name (str): Parent array name used for binding lookup. Defaults
            to ``"angles"``.
        length (int): Declared length of the parent array. Defaults to 3.

    Returns:
        Value: A Value representing ``name[index]``.
    """
    parent = ArrayValue(
        type=FloatType(),
        name=name,
        shape=(Value(type=UIntType(), name="dim").with_const(length),),
    )
    return Value(
        type=FloatType(),
        name=f"{name}[i]",
        parent_array=parent,
        element_indices=(index,),
    )


class TestIndexIntoBoundArray:
    @pytest.mark.parametrize("oob_index", [3, 4, 10])
    def test_out_of_range_concrete_index_raises(self, oob_index):
        """A concrete out-of-range index into a bound array raises EmitError
        naming the offending index and the array length."""
        resolver = ValueResolver()
        element = _array_element(Value(type=UIntType(), name="i").with_const(oob_index))

        with pytest.raises(
            EmitError,
            match=rf"Index {oob_index} is out of range .* 'angles' of length 3",
        ):
            resolver.resolve_classical_value(
                element, bindings={"angles": [0.1, 0.2, 0.3]}
            )

    @pytest.mark.parametrize(("index", "expected"), [(0, 0.1), (1, 0.2), (2, 0.3)])
    def test_in_range_concrete_index_resolves(self, index, expected):
        """An in-range concrete index resolves to the corresponding element."""
        resolver = ValueResolver()
        element = _array_element(Value(type=UIntType(), name="i").with_const(index))

        resolved = resolver.resolve_classical_value(
            element, bindings={"angles": [0.1, 0.2, 0.3]}
        )
        assert resolved == pytest.approx(expected)

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

    def test_zero_dim_container_surfaces_emit_error_not_typeerror(self):
        """A 0-d container raises on both indexing and ``len()``; the
        out-of-range ``EmitError`` must still surface rather than being
        masked by a ``TypeError`` while formatting the length note."""
        resolver = ValueResolver()
        element = _array_element(Value(type=UIntType(), name="i").with_const(0))

        with pytest.raises(EmitError, match=r"Index 0 is out of range .* 'angles'"):
            resolver.resolve_classical_value(
                element, bindings={"angles": np.array(5.0)}
            )
