"""Tests for dynamic array size resolution in transpiler.

These tests verify that the transpiler correctly handles:
1. Dynamic qubit array sizes derived from input array shapes (hi.shape[0] → hi_dim0)
2. Nested array access patterns (edges[e, 0])
3. Measurement of dynamically-sized qubit arrays
4. Numpy integer types in array element resolution

These tests were added to prevent regression of bugs fixed in the emit pass.
"""

from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types.primitives import QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import (
    EmitError,
    QamomileCompileError,
    ResolutionFailureReason,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import QubitAddress
from qamomile.circuit.transpiler.passes.emit_support.resource_allocator import (
    ResourceAllocator,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import ValueResolver
from qamomile.qiskit.transpiler import QiskitTranspiler

Backend = tuple[str, Any, Any]

# ==============================================================================
# Kernel definitions at module level (required for inspect.getsource to work)
# ==============================================================================


@qmc.qkernel
def kernel_dynamic_size(
    hi: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Kernel with qubit array sized by input vector shape."""
    n = hi.shape[0]
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return qmc.measure(q)


@qmc.qkernel
def kernel_matrix_size(
    edges: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Kernel with qubit array sized by matrix dimension."""
    n = edges.shape[0]
    q = qmc.qubit_array(n, name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_size_from_uint_scalar(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Measure a qubit array sized by a bound UInt scalar.

    Args:
        n (qmc.UInt): Compile-time bound scalar that determines the
            qubit-array size.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement result for the allocated qubits.
    """
    q = qmc.qubit_array(n, name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_size_from_uint_element(
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Measure a qubit array sized by a bound UInt vector element.

    Args:
        sizes (qmc.Vector[qmc.UInt]): Compile-time bound vector whose first
            element determines the qubit-array size.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement result for the allocated qubits.
    """
    q = qmc.qubit_array(sizes[0], name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_size_from_uint_slice_element(
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Measure a qubit array sized by a bound UInt vector-view element.

    Args:
        sizes (qmc.Vector[qmc.UInt]): Compile-time bound vector whose sliced
            view element determines the qubit-array size.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement result for the allocated qubits.
    """
    q = qmc.qubit_array(sizes[1:3][0], name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_size_from_uint_strided_slice_element(
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Measure a qubit array sized by a strided vector-view element.

    Args:
        sizes (qmc.Vector[qmc.UInt]): Compile-time bound vector whose strided
            view element determines the qubit-array size.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement result for the allocated qubits.
    """
    q = qmc.qubit_array(sizes[0:3:2][1], name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_size_from_uint_default_start_slice_element(
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Measure a qubit array sized by a default-start vector-view element.

    Args:
        sizes (qmc.Vector[qmc.UInt]): Compile-time bound vector whose
            default-start sliced view element determines the qubit-array size.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement result for the allocated qubits.
    """
    q = qmc.qubit_array(sizes[:3][0], name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_size_from_uint_default_stop_slice_element(
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Measure a qubit array sized by a default-stop vector-view element.

    Args:
        sizes (qmc.Vector[qmc.UInt]): Compile-time bound vector whose
            default-stop sliced view element determines the qubit-array size.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement result for the allocated qubits.
    """
    q = qmc.qubit_array(sizes[1:][0], name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_size_from_symbolic_uint_element(
    sizes: qmc.Vector[qmc.UInt],
    i: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Measure a qubit array sized by a symbolic UInt vector index.

    Args:
        sizes (qmc.Vector[qmc.UInt]): Compile-time bound vector indexed by a
            runtime symbolic parameter.
        i (qmc.UInt): Runtime symbolic index into ``sizes``.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement result for the allocated qubits.
    """
    q = qmc.qubit_array(sizes[i], name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_size_from_negative_index_element(
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Measure a qubit array sized by a negative-index vector element.

    Args:
        sizes (qmc.Vector[qmc.UInt]): Vector accessed with an unsupported
            Python-style negative index.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement result for the allocated qubits.
    """
    q = qmc.qubit_array(sizes[-1], name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_size_from_negative_view_element(
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Measure a qubit array sized by a negative-index vector-view element.

    Args:
        sizes (qmc.Vector[qmc.UInt]): Vector whose sliced view is accessed
            with an unsupported Python-style negative index.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement result for the allocated qubits.
    """
    q = qmc.qubit_array(sizes[1:3][-1], name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_nested_index(
    edges: qmc.Matrix[qmc.UInt],
    n_edges: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Kernel with nested array index access."""
    q = qmc.qubit_array(3, name="q")
    for e in qmc.range(n_edges):
        i = edges[e, 0]
        j = edges[e, 1]
        q[i], q[j] = qmc.cx(q[i], q[j])
    return qmc.measure(q)


@qmc.qkernel
def kernel_ising_cost(
    q: qmc.Vector[qmc.Qubit],
    edges: qmc.Matrix[qmc.UInt],
    Jij: qmc.Vector[qmc.Float],
    hi: qmc.Vector[qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """QAOA cost layer with nested array access."""
    num_e = edges.shape[0]
    for e in qmc.range(num_e):
        i = edges[e, 0]
        j = edges[e, 1]
        angle = 2 * Jij[e] * gamma
        q[i], q[j] = qmc.rzz(q[i], q[j], angle)
    n = hi.shape[0]
    for i in qmc.range(n):
        angle = 2 * hi[i] * gamma
        q[i] = qmc.rz(q[i], angle)
    return q


@qmc.qkernel
def kernel_x_mixer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """QAOA mixer layer."""
    n = q.shape[0]
    for i in qmc.range(n):
        angle = 2 * beta
        q[i] = qmc.rx(q[i], angle)
    return q


@qmc.qkernel
def kernel_qaoa_state(
    edges: qmc.Matrix[qmc.UInt],
    Jij: qmc.Vector[qmc.Float],
    hi: qmc.Vector[qmc.Float],
    p: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """QAOA state preparation with dynamic qubit array."""
    n = hi.shape[0]
    q = qmc.qubit_array(n, name="qaoa_state")
    for iter in qmc.range(p):
        gamma = gammas[iter]
        beta = betas[iter]
        q = kernel_ising_cost(q, edges, Jij, hi, gamma)
        q = kernel_x_mixer(q, beta)
    return q


@qmc.qkernel
def kernel_qaoa(
    edges: qmc.Matrix[qmc.UInt],
    Jij: qmc.Vector[qmc.Float],
    hi: qmc.Vector[qmc.Float],
    p: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Full QAOA circuit with measurement."""
    q = kernel_qaoa_state(edges, Jij, hi, p, gammas, betas)
    return qmc.measure(q)


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def sdk_backend(request) -> Backend:
    """Yield an installed SDK backend for cross-backend execution.

    Args:
        request (pytest.FixtureRequest): Parametrized pytest request whose
            value selects the backend name.

    Returns:
        Backend: Tuple of backend name, transpiler, and executor. Optional
            SDK dependencies are skipped with ``pytest.importorskip``.

    Raises:
        AssertionError: If the fixture parameter is not a known backend name.
    """
    name = request.param
    if name == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "quri_parts":
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        return name, transpiler, transpiler.executor()
    if name == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        transpiler = CudaqTranspiler()
        return name, transpiler, transpiler.executor()
    raise AssertionError(f"Unknown backend: {name}")


def _counts(result: Any) -> dict[tuple[int, ...], int]:
    """Convert a sample result into counts keyed by bit tuples.

    Args:
        result (Any): Backend sample result exposing ``results`` as
            bitstring/count pairs.

    Returns:
        dict[tuple[int, ...], int]: Counts keyed by integer bit tuples.
    """
    counts: dict[tuple[int, ...], int] = {}
    for bits, count in result.results:
        bit_tuple = tuple(int(bit) for bit in bits)
        counts[bit_tuple] = counts.get(bit_tuple, 0) + int(count)
    return counts


def _assert_all_zero_counts(
    name: str,
    counts: dict[tuple[int, ...], int],
    *,
    width: int,
    shots: int,
) -> None:
    """Assert that deterministic all-zero sampling returned every shot.

    Args:
        name (str): Backend name used in the assertion message.
        counts (dict[tuple[int, ...], int]): Observed counts keyed by bit
            tuples.
        width (int): Expected bitstring width.
        shots (int): Expected total shot count for the all-zero outcome.

    Raises:
        AssertionError: If the counts contain any non-zero outcome or an
            unexpected shot count.
    """
    expected = tuple(0 for _ in range(width))
    assert counts == {expected: shots}, f"{name}: unexpected counts {counts}"


# ==============================================================================
# Test classes
# ==============================================================================


class TestDynamicArraySizeResolution:
    """Tests for dynamic qubit array size resolution.

    Verifies that qubit arrays sized by input array shapes (e.g., hi.shape[0])
    are correctly allocated and can be used in circuits.
    """

    def test_qubit_array_size_from_vector_shape(self):
        """Test that qubit array size can be determined from input vector shape.

        This tests the hi_dim0 naming pattern resolution in ResourceAllocator._resolve_size().
        """
        transpiler = QiskitTranspiler()
        hi = np.array([0.1, 0.2, 0.3])  # 3 elements → 3 qubits

        executor = transpiler.transpile(kernel_dynamic_size, bindings={"hi": hi})

        # Run the circuit to verify it executes without error
        job = executor.sample(transpiler.executor(), bindings={}, shots=100)
        result = job.result()
        assert result is not None
        # Should have measurement results for 3 qubits
        for bitstring, _count in result.results:
            assert len(bitstring) == 3

    def test_qubit_array_size_from_matrix_shape(self):
        """Test that qubit array size works with 2D array shapes."""
        transpiler = QiskitTranspiler()
        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.uint64)  # 3 edges

        executor = transpiler.transpile(kernel_matrix_size, bindings={"edges": edges})

        job = executor.sample(transpiler.executor(), bindings={}, shots=100)
        result = job.result()
        assert result is not None
        for bitstring, _count in result.results:
            assert len(bitstring) == 3

    @pytest.mark.parametrize("n", [5, np.uint64(5)])
    def test_qubit_array_size_from_bound_uint_scalar(self, n: int):
        """Test that Python and NumPy integer scalar sizes resolve."""
        transpiler = QiskitTranspiler()

        executor = transpiler.transpile(
            kernel_size_from_uint_scalar,
            bindings={"n": n},
        )

        circuit = executor.compiled_quantum[0].circuit
        assert circuit.num_qubits == 5
        assert circuit.num_clbits == 5
        counts = _counts(executor.sample(transpiler.executor(), shots=16).result())
        _assert_all_zero_counts("qiskit", counts, width=5, shots=16)

    def test_qubit_array_size_from_bound_uint_vector_element(self):
        """Test that a bound UInt vector-element size executes deterministically."""
        transpiler = QiskitTranspiler()

        executor = transpiler.transpile(
            kernel_size_from_uint_element,
            bindings={"sizes": np.array([5], dtype=np.uint64)},
        )

        circuit = executor.compiled_quantum[0].circuit
        assert circuit.num_qubits == 5
        assert circuit.num_clbits == 5
        counts = _counts(executor.sample(transpiler.executor(), shots=16).result())
        _assert_all_zero_counts("qiskit", counts, width=5, shots=16)

    def test_qubit_array_size_from_bound_uint_vector_view_element(self):
        """Test that a bound UInt vector-view size executes deterministically."""
        transpiler = QiskitTranspiler()

        executor = transpiler.transpile(
            kernel_size_from_uint_slice_element,
            bindings={"sizes": np.array([2, 5, 7], dtype=np.uint64)},
        )

        circuit = executor.compiled_quantum[0].circuit
        assert circuit.num_qubits == 5
        assert circuit.num_clbits == 5
        counts = _counts(executor.sample(transpiler.executor(), shots=16).result())
        _assert_all_zero_counts("qiskit", counts, width=5, shots=16)

    def test_qubit_array_size_from_bound_uint_strided_vector_view_element(self):
        """Test that a strided UInt vector-view size remaps to its root index."""
        transpiler = QiskitTranspiler()

        executor = transpiler.transpile(
            kernel_size_from_uint_strided_slice_element,
            bindings={"sizes": np.array([2, 7, 5], dtype=np.uint64)},
        )

        circuit = executor.compiled_quantum[0].circuit
        assert circuit.num_qubits == 5
        assert circuit.num_clbits == 5
        counts = _counts(executor.sample(transpiler.executor(), shots=16).result())
        _assert_all_zero_counts("qiskit", counts, width=5, shots=16)

    @pytest.mark.parametrize(
        ("kernel", "sizes", "width"),
        [
            (
                kernel_size_from_uint_default_start_slice_element,
                np.array([5, 7, 11], dtype=np.uint64),
                5,
            ),
            (
                kernel_size_from_uint_default_stop_slice_element,
                np.array([2, 5, 7], dtype=np.uint64),
                5,
            ),
        ],
    )
    def test_qubit_array_size_from_implicit_bound_vector_view_element(
        self,
        kernel: Any,
        sizes: np.ndarray,
        width: int,
    ):
        """Test that implicit slice bounds resolve for vector-view elements."""
        transpiler = QiskitTranspiler()

        executor = transpiler.transpile(kernel, bindings={"sizes": sizes})

        circuit = executor.compiled_quantum[0].circuit
        assert circuit.num_qubits == width
        assert circuit.num_clbits == width
        counts = _counts(executor.sample(transpiler.executor(), shots=16).result())
        _assert_all_zero_counts("qiskit", counts, width=width, shots=16)

    @pytest.mark.parametrize(
        ("kernel", "sizes", "width"),
        [
            (
                kernel_size_from_uint_element,
                np.array([5], dtype=np.uint64),
                5,
            ),
            (
                kernel_size_from_uint_slice_element,
                np.array([2, 5, 7], dtype=np.uint64),
                5,
            ),
            (
                kernel_size_from_uint_strided_slice_element,
                np.array([2, 7, 5], dtype=np.uint64),
                5,
            ),
        ],
    )
    def test_bound_uint_vector_element_executes_on_sdk_backends(
        self,
        sdk_backend: Backend,
        kernel: Any,
        sizes: np.ndarray,
        width: int,
    ):
        """Test UInt vector-element size allocation on every SDK backend."""
        name, transpiler, executor = sdk_backend

        executable = transpiler.transpile(
            kernel,
            bindings={"sizes": sizes},
        )
        counts = _counts(executable.sample(executor, shots=16).result())

        _assert_all_zero_counts(name, counts, width=width, shots=16)

    @pytest.mark.parametrize("n", [-1, np.int64(-3)])
    def test_negative_scalar_size_raises(self, n: int):
        """Test that negative scalar sizes are rejected."""
        transpiler = QiskitTranspiler()

        with pytest.raises(QamomileCompileError, match="Cannot resolve array size"):
            transpiler.transpile(kernel_size_from_uint_scalar, bindings={"n": n})

    @pytest.mark.parametrize(
        "sizes",
        [
            np.array([-1], dtype=np.int64),
            np.array([-3], dtype=np.int64),
        ],
    )
    def test_negative_vector_element_size_raises(self, sizes: np.ndarray):
        """Test that negative vector-element sizes are rejected."""
        transpiler = QiskitTranspiler()

        with pytest.raises(QamomileCompileError, match="Cannot resolve array size"):
            transpiler.transpile(
                kernel_size_from_uint_element, bindings={"sizes": sizes}
            )

    def test_symbolic_uint_vector_element_size_raises(self):
        """Test that symbolic element indices do not fall back to array length."""
        transpiler = QiskitTranspiler()

        with pytest.raises(QamomileCompileError, match="Cannot resolve array size"):
            transpiler.transpile(
                kernel_size_from_symbolic_uint_element,
                bindings={"sizes": np.array([2, 5], dtype=np.uint64)},
                parameters=["i"],
            )

    def test_negative_bound_index_size_rejected(self):
        """Test that an index bound to ``-1`` does not Python-wrap the container.

        Binding ``i=-1`` previously made ``sizes[i]`` silently resolve to the
        last container element (7 qubits); call-time specialization now folds
        the negative constant into the element access, where the frontend
        rejection fires.
        """
        transpiler = QiskitTranspiler()

        with pytest.raises(NotImplementedError, match="Negative index"):
            transpiler.transpile(
                kernel_size_from_symbolic_uint_element,
                bindings={"sizes": np.array([2, 5, 7], dtype=np.uint64), "i": -1},
            )

    def test_non_integral_vector_element_size_raises(self):
        """Test that non-integral element sizes are not silently truncated."""
        transpiler = QiskitTranspiler()

        with pytest.raises(QamomileCompileError, match="Cannot resolve array size"):
            transpiler.transpile(
                kernel_size_from_uint_element,
                bindings={"sizes": np.array([5.5], dtype=np.float64)},
            )

    def test_bool_vector_element_size_raises(self):
        """Test that boolean element sizes are not coerced to one or zero."""
        transpiler = QiskitTranspiler()

        with pytest.raises(QamomileCompileError, match="Cannot resolve array size"):
            transpiler.transpile(
                kernel_size_from_uint_element,
                bindings={"sizes": np.array([True], dtype=np.bool_)},
            )

    @pytest.mark.parametrize(
        "kernel",
        [
            kernel_size_from_negative_index_element,
            kernel_size_from_negative_view_element,
        ],
    )
    def test_negative_const_element_index_size_rejected_at_trace(self, kernel: Any):
        """Test that constant negative element indices are rejected at trace.

        ``sizes[-1]`` previously resolved by Python container accident while
        ``sizes[1:3][-1]`` silently affine-composed to ``sizes[0]`` and
        produced a wrong-sized circuit; both now fail loudly before IR is
        built.
        """
        transpiler = QiskitTranspiler()

        with pytest.raises(NotImplementedError, match="Negative index"):
            transpiler.transpile(
                kernel,
                bindings={"sizes": np.array([2, 5, 7], dtype=np.uint64)},
            )

    def test_allocator_size_from_const_uint_vector_element(self):
        """Test that const-array metadata resolves an array element size."""
        dim = Value(type=UIntType(), name="dim_0").with_const(1)
        parent = ArrayValue(
            type=UIntType(),
            name="sizes",
            shape=(dim,),
        ).with_array_runtime_metadata(const_array=(5,))
        idx = Value(type=UIntType(), name="idx_0").with_const(0)
        size = Value(
            type=UIntType(),
            name="sizes[0]",
            parent_array=parent,
            element_indices=(idx,),
        )
        q = ArrayValue(type=QubitType(), name="q", shape=(size,))
        allocator = ResourceAllocator()

        qubit_map, clbit_map = allocator.allocate([QInitOperation([], [q])])

        assert len(qubit_map) == 5
        assert clbit_map == {}

    def test_allocator_size_from_const_uint_vector_view_element(self):
        """Test that a const-array VectorView element resolves through its root."""
        root_dim = Value(type=UIntType(), name="dim_0").with_const(3)
        root = ArrayValue(
            type=UIntType(),
            name="sizes",
            shape=(root_dim,),
        ).with_array_runtime_metadata(const_array=(2, 5, 7))
        view_dim = Value(type=UIntType(), name="view_dim_0").with_const(2)
        start = Value(type=UIntType(), name="slice_start").with_const(1)
        step = Value(type=UIntType(), name="slice_step").with_const(1)
        view = ArrayValue(
            type=UIntType(),
            name="sizes_slice",
            shape=(view_dim,),
            slice_of=root,
            slice_start=start,
            slice_step=step,
        )
        idx = Value(type=UIntType(), name="idx_0").with_const(0)
        size = Value(
            type=UIntType(),
            name="sizes_slice[0]",
            parent_array=view,
            element_indices=(idx,),
        )
        q = ArrayValue(type=QubitType(), name="q", shape=(size,))
        allocator = ResourceAllocator()

        qubit_map, clbit_map = allocator.allocate([QInitOperation([], [q])])

        assert len(qubit_map) == 5
        assert clbit_map == {}

    def test_allocator_runtime_parameter_vector_element_size_stays_unresolved(self):
        """Test that runtime parameter array elements are not indexed."""
        dim = Value(type=UIntType(), name="dim_0").with_const(1)
        parent = ArrayValue(
            type=UIntType(),
            name="sizes",
            shape=(dim,),
        )
        idx = Value(type=UIntType(), name="idx_0").with_const(0)
        size = Value(
            type=UIntType(),
            name="sizes[0]",
            parent_array=parent,
            element_indices=(idx,),
        )
        q = ArrayValue(type=QubitType(), name="q", shape=(size,))
        allocator = ResourceAllocator(ValueResolver(parameters={"sizes"}))

        with pytest.raises(EmitError, match="Cannot resolve array size"):
            allocator.allocate(
                [QInitOperation([], [q])],
                bindings={"sizes": np.array([5], dtype=np.uint64)},
            )

    def test_allocator_negative_symbolic_element_index_stays_unresolved(self):
        """Test that a negative-resolved element index does not Python-wrap.

        A symbolic index bound to ``-1`` must not silently read the last
        container element (``container[-1]``); the size stays unresolved
        and surfaces as the standard unresolved-size error.
        """
        dim = Value(type=UIntType(), name="dim_0").with_const(3)
        parent = ArrayValue(
            type=UIntType(),
            name="sizes",
            shape=(dim,),
        ).with_array_runtime_metadata(const_array=(2, 5, 7))
        idx = Value(type=UIntType(), name="i")
        size = Value(
            type=UIntType(),
            name="sizes[i]",
            parent_array=parent,
            element_indices=(idx,),
        )
        q = ArrayValue(type=QubitType(), name="q", shape=(size,))
        allocator = ResourceAllocator()

        with pytest.raises(EmitError, match="Cannot resolve array size"):
            allocator.allocate([QInitOperation([], [q])], bindings={"i": -1})

    def test_allocator_negative_symbolic_view_element_index_stays_unresolved(self):
        """Test that a negative-resolved view-local index is not affine-composed.

        For a view over ``sizes`` with ``start=1, step=1``, a local index
        bound to ``-1`` must not compose to root index ``0`` (which would
        silently allocate ``sizes[0]`` qubits); the size stays unresolved.
        """
        root_dim = Value(type=UIntType(), name="dim_0").with_const(3)
        root = ArrayValue(
            type=UIntType(),
            name="sizes",
            shape=(root_dim,),
        ).with_array_runtime_metadata(const_array=(2, 5, 7))
        view_dim = Value(type=UIntType(), name="view_dim_0").with_const(2)
        start = Value(type=UIntType(), name="slice_start").with_const(1)
        step = Value(type=UIntType(), name="slice_step").with_const(1)
        view = ArrayValue(
            type=UIntType(),
            name="sizes_slice",
            shape=(view_dim,),
            slice_of=root,
            slice_start=start,
            slice_step=step,
        )
        idx = Value(type=UIntType(), name="i")
        size = Value(
            type=UIntType(),
            name="sizes_slice[i]",
            parent_array=view,
            element_indices=(idx,),
        )
        q = ArrayValue(type=QubitType(), name="q", shape=(size,))
        allocator = ResourceAllocator()

        with pytest.raises(EmitError, match="Cannot resolve array size"):
            allocator.allocate([QInitOperation([], [q])], bindings={"i": -1})

    def test_resolver_negative_view_qubit_index_fails(self):
        """Test that emit-time qubit routing refuses negative view indices.

        Without the guard, a view-local index bound to ``-1`` composes to
        root index ``start + step * (-1)`` — a valid-but-wrong physical
        wire — instead of failing with ``NEGATIVE_INDEX``.
        """
        root_dim = Value(type=UIntType(), name="dim_0").with_const(3)
        root = ArrayValue(type=QubitType(), name="q", shape=(root_dim,))
        view_dim = Value(type=UIntType(), name="view_dim_0").with_const(2)
        start = Value(type=UIntType(), name="slice_start").with_const(1)
        step = Value(type=UIntType(), name="slice_step").with_const(1)
        view = ArrayValue(
            type=QubitType(),
            name="q_slice",
            shape=(view_dim,),
            slice_of=root,
            slice_start=start,
            slice_step=step,
        )
        idx = Value(type=UIntType(), name="i")
        element = Value(
            type=QubitType(),
            name="q_slice[i]",
            parent_array=view,
            element_indices=(idx,),
        )
        qubit_map = {QubitAddress(root.uuid, k): k for k in range(3)}
        resolver = ValueResolver()

        result = resolver.resolve_qubit_index_detailed(
            element, qubit_map, bindings={"i": -1}
        )

        assert not result.success
        assert result.failure_reason is ResolutionFailureReason.NEGATIVE_INDEX


class TestNestedArrayAccess:
    """Tests for nested array access resolution.

    Verifies that qubit indices computed from array lookups (e.g., edges[e, 0])
    are correctly resolved during loop unrolling.
    """

    def test_nested_array_index_resolution(self):
        """Test that nested array access like q[edges[e, 0]] works correctly.

        This tests the numpy integer type handling in _index_into_array().
        """
        transpiler = QiskitTranspiler()
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint64)

        executor = transpiler.transpile(
            kernel_nested_index,
            bindings={"edges": edges, "n_edges": 2},
        )

        job = executor.sample(transpiler.executor(), bindings={}, shots=100)
        result = job.result()
        assert result is not None


class TestDynamicMeasurement:
    """Tests for measurement of dynamically-sized qubit arrays.

    Verifies that _emit_measure_vector correctly handles dynamic sizes.
    """

    def test_measure_dynamic_qubit_array(self):
        """Test that measurement works with dynamically-sized qubit arrays.

        This tests the _emit_measure_vector fix for dynamic size resolution.
        """
        transpiler = QiskitTranspiler()
        hi = np.array([0.1, 0.2, 0.3, 0.4])  # 4 elements

        executor = transpiler.transpile(kernel_dynamic_size, bindings={"hi": hi})

        # This should not raise "No counts for experiment" error
        job = executor.sample(transpiler.executor(), bindings={}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0
        # Verify all bitstrings have correct length
        for bitstring, _count in result.results:
            assert len(bitstring) == 4


class TestQAOAPattern:
    """Integration tests for QAOA-like circuit patterns.

    These tests verify that the combination of all dynamic resolution features
    works correctly in realistic QAOA circuits.
    """

    def test_qaoa_full_pattern(self):
        """Test full QAOA pattern with dynamic sizes and nested array access."""
        # Test data - triangle graph
        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.uint64)
        Jij = np.array([0.5, 0.3, 0.2])
        hi = np.array([0.1, 0.2, 0.15])  # 3 qubits

        transpiler = QiskitTranspiler()
        executor = transpiler.transpile(
            kernel_qaoa,
            bindings={"edges": edges, "Jij": Jij, "hi": hi, "p": 2},
            parameters=["betas", "gammas"],
        )

        # Run with concrete parameter values
        job = executor.sample(
            transpiler.executor(),
            bindings={"betas": [0.1, 0.2], "gammas": [0.3, 0.4]},
            shots=100,
        )
        result = job.result()

        assert result is not None
        assert len(result.results) > 0
        # Verify all bitstrings have correct length (3 qubits)
        for bitstring, _count in result.results:
            assert len(bitstring) == 3
