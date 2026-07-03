"""Tests pinning the explicit rejection of rank>1 quantum registers.

The quantum addressing path is rank-1 throughout (flat ``QubitAddress``,
``shape[0]``-based wire allocation, ``element_indices[0]``-only
resolution), so a higher-rank quantum register used to silently alias
distinct elements onto the same physical qubit. Six guards now reject
rank>1 quantum registers loudly (see the "Rank>1 quantum registers are
explicitly rejected" entry of ``LIMITATIONS.md``); this module pins each
guard firing plus non-over-firing controls for the rank-1 quantum and
classical higher-rank paths.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.func_to_block import create_dummy_input
from qamomile.circuit.frontend.handle import (
    Float,
    Matrix,
    Qubit,
    Tensor,
    UInt,
    Vector,
)
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types import QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _populate_input_qubit_map,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    QubitAddress,
)
from qamomile.circuit.transpiler.passes.emit_support.resource_allocator import (
    ResourceAllocator,
)


def _quantum_array_value(name: str, dims: tuple[int, ...]) -> ArrayValue:
    """Build a quantum ``ArrayValue`` with constant shape dimensions.

    Args:
        name (str): Name for the ArrayValue.
        dims (tuple[int, ...]): Concrete shape dimensions.

    Returns:
        ArrayValue: A ``QubitType`` ArrayValue carrying one constant
            shape Value per dimension.
    """
    shape = tuple(
        Value(type=UIntType(), name=f"{name}_dim{i}").with_const(dim)
        for i, dim in enumerate(dims)
    )
    return ArrayValue(type=QubitType(), name=name, shape=shape)


class TestRankGuardsFire:
    """Each of the six rank>1 rejection guards raises loudly."""

    @pytest.mark.parametrize(
        "shape, rank",
        [((2, 2), 2), ((2, 3), 2), ((2, 2, 2), 3)],
    )
    def test_qubit_array_tuple_shape_rejected(self, shape, rank):
        """``qubit_array`` rejects rank>1 tuple shapes before any tracer
        access (no tracer is active here, so reaching the tracer would
        raise ``RuntimeError`` instead)."""
        with pytest.raises(NotImplementedError, match=f"rank-{rank}"):
            qmc.qubit_array(shape, "m")

    def test_vector_wrapper_cannot_smuggle_rank2_value(self):
        """``ArrayBase.__post_init__`` keys off ``value.shape``, so a
        rank-2 quantum ArrayValue is rejected even inside a ``Vector``
        wrapper."""
        array_value = _quantum_array_value("m", (2, 2))
        with pytest.raises(NotImplementedError, match="(?i)rank-2"):
            Vector[Qubit](value=array_value, _shape=(2, 2))

    def test_arraybase_create_matrix_qubit_rejected(self):
        """``ArrayBase.create`` routes through ``__post_init__``, so a
        quantum ``Matrix`` cannot be created via the factory either."""
        with pytest.raises(NotImplementedError, match="(?i)rank-2"):
            Matrix.create((2, 2), "m", Qubit)

    @pytest.mark.parametrize(
        "annotation, rank",
        [(Matrix[Qubit], 2), (Tensor[Qubit], 3)],
    )
    def test_create_dummy_input_rejects_quantum_higher_rank(self, annotation, rank):
        """``create_dummy_input`` guards the kernel-parameter path, which
        bypasses ``__post_init__`` via ``object.__new__``."""
        with pytest.raises(NotImplementedError, match=f"rank-{rank}"):
            create_dummy_input(annotation, "m")

    def test_kernel_with_matrix_qubit_param_fails_to_build(self):
        """A kernel declaring a ``Matrix[Qubit]`` parameter raises when
        its block is traced (the ``create_dummy_input`` guard)."""

        @qmc.qkernel
        def takes_matrix(m: Matrix[Qubit]) -> Matrix[Qubit]:
            return m

        with pytest.raises(NotImplementedError, match="rank-2"):
            takes_matrix.build()

    def test_draw_path_rejects_matrix_qubit_param(self):
        """The ``draw(name=size)`` visualization path rejects rank>1
        quantum parameters instead of realizing them as 1-D registers."""

        @qmc.qkernel
        def takes_matrix(m: Matrix[Qubit]) -> Matrix[Qubit]:
            return m

        with pytest.raises(NotImplementedError, match="rank-2"):
            takes_matrix._build_graph_for_visualization(m=2)

    def test_resource_allocator_rejects_rank2_qinit(self):
        """The allocator's QInit branch rejects rank>1 registers in
        hand-built IR (defense in depth behind the frontend guards)."""
        array_value = _quantum_array_value("m", (2, 2))
        qinit = QInitOperation(operands=[], results=[array_value])
        with pytest.raises(EmitError, match="rank-2"):
            ResourceAllocator().allocate([qinit])

    def test_input_qubit_map_rejects_rank2_input(self):
        """The controlled-U input mapper rejects rank>1 quantum inputs in
        hand-built IR (defense in depth behind the frontend guards).
        The rank guard fires before any length resolution, so the
        ``emit_pass`` argument is never touched."""
        array_value = _quantum_array_value("m", (2, 2))
        with pytest.raises(EmitError, match="rank-2"):
            _populate_input_qubit_map(None, [array_value], 4, {}, {})


class TestRankGuardsDoNotOverfire:
    """Rank-1 quantum and higher-rank classical paths keep working."""

    def test_qubit_array_scalar_and_one_tuple_shapes_ok(self):
        """``qubit_array`` still accepts a scalar size and a 1-tuple."""
        with trace():
            from_scalar = qmc.qubit_array(3, "q_scalar")
            from_tuple = qmc.qubit_array((3,), "q_tuple")
        assert isinstance(from_scalar, Vector)
        assert isinstance(from_tuple, Vector)
        assert len(from_scalar.value.shape) == 1
        assert len(from_tuple.value.shape) == 1

    def test_vector_qubit_param_builds_and_draws(self):
        """``Vector[Qubit]`` parameters keep working on both the symbolic
        build path and the ``draw(name=size)`` visualization path."""

        @qmc.qkernel
        def takes_vector(q: Vector[Qubit]) -> Vector[Qubit]:
            return q

        assert takes_vector.build() is not None
        assert takes_vector._build_graph_for_visualization(q=3) is not None

    def test_classical_matrix_and_tensor_params_ok(self):
        """Classical ``Matrix[Float]`` / ``Tensor[UInt]`` parameters are
        untouched by the quantum rank guards."""

        @qmc.qkernel
        def takes_classical(m: Matrix[Float], t: Tensor[UInt]) -> Float:
            return m[0, 0]

        assert takes_classical.build() is not None

    def test_classical_matrix_create_ok(self):
        """``ArrayBase.create`` still builds classical rank-2 arrays."""
        with trace():
            matrix = Matrix.create((2, 2), "m", Float)
        assert len(matrix.value.shape) == 2

    def test_resource_allocator_rank1_allocates_all_elements(self):
        """The allocator still assigns one wire per element of a 1-D
        quantum register."""
        array_value = _quantum_array_value("q", (3,))
        qinit = QInitOperation(operands=[], results=[array_value])
        qubit_map, _ = ResourceAllocator().allocate([qinit])
        for i in range(3):
            assert qubit_map[QubitAddress(array_value.uuid, i)] == i
        assert len(qubit_map) == 3

    def test_input_qubit_map_rank1_maps_all_elements(self):
        """The controlled-U input mapper still maps each element of a 1-D
        quantum input plus the base address."""
        array_value = _quantum_array_value("q", (3,))
        qubit_map: dict = {}
        _populate_input_qubit_map(None, [array_value], 3, {}, qubit_map)
        for i in range(3):
            assert qubit_map[QubitAddress(array_value.uuid, i)] == i
        assert qubit_map[QubitAddress(array_value.uuid)] == 0
