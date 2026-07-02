"""Regression tests for classical array element assignment in qkernels.

Classical element assignment (``bits[1] = bits[0]``) used to emit no IR at
all: ``ArrayBase._return_element`` only released frontend borrow
bookkeeping for classical element types, so the write was silently dropped
and the compiled program returned the pre-store contents (see the
quantum-side counterpart fixed on the reject-foreign-qubit-assign branch).
The write is now a real ``StoreArrayElementOperation``: compile-time
resolvable stores fold into ``const_array`` metadata, measurement-derived
stores execute host-side in a classical segment, and every unsupported
route (slice views, multi-dimensional arrays, if/else branches, quantum
segment leakage, self-referential loop updates) fails loudly instead of
miscompiling.
"""

from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.transpiler.errors import (
    EmitError,
    ValidationError,
)

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler  # noqa: E402

Backend = tuple[str, Any, Any]


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def backend(request) -> Backend:
    """Yield ``(name, transpiler, executor)`` for each installed SDK backend."""
    name = request.param
    if name == "qiskit":
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
    raise AssertionError(f"unknown backend {name}")


def _counts(result: Any) -> dict[Any, int]:
    """Convert a sample result to a count map keyed by output tuples."""
    counts: dict[Any, int] = {}
    for bits, count in result.results:
        key = tuple(
            tuple(int(b) for b in part) if isinstance(part, tuple) else int(part)
            for part in bits
        )
        counts[key] = counts.get(key, 0) + int(count)
    return counts


# ---------------------------------------------------------------------------
# Vector[Bit] (measurement-derived)
# ---------------------------------------------------------------------------


@qmc.qkernel
def copy_bit_kernel() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(2, "qs")
    qs[0] = qmc.x(qs[0])
    bits = qmc.measure(qs)
    bits[1] = bits[0]
    return bits


def test_measured_bit_store_repro(backend):
    """The original silent-drop repro: bits[1] = bits[0] must yield (1, 1).

    Before the fix the write was dropped and sampling returned (1, 0).
    """
    name, transpiler, executor = backend
    exe = transpiler.transpile(copy_bit_kernel)
    counts = _counts(exe.sample(executor, shots=100).result())
    assert counts == {(1, 1): 100}, f"{name}: got {counts}"


@qmc.qkernel
def chained_bit_kernel() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(3, "qs")
    qs[0] = qmc.x(qs[0])
    bits = qmc.measure(qs)
    bits[1] = bits[0]
    bits[2] = bits[1]
    return bits


def test_measured_bit_chained_stores(backend):
    """Chained stores read the post-store contents of the previous store."""
    name, transpiler, executor = backend
    exe = transpiler.transpile(chained_bit_kernel)
    counts = _counts(exe.sample(executor, shots=100).result())
    assert counts == {(1, 1, 1): 100}, f"{name}: got {counts}"


@qmc.qkernel
def bit_literal_kernel() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(2, "qs")
    bits = qmc.measure(qs)
    bits[1] = 1
    return bits


def test_measured_bit_literal_store(backend):
    """A Python literal 0/1 can be stored into a measured Vector[Bit]."""
    name, transpiler, executor = backend
    exe = transpiler.transpile(bit_literal_kernel)
    counts = _counts(exe.sample(executor, shots=100).result())
    assert counts == {(0, 1): 100}, f"{name}: got {counts}"


@qmc.qkernel
def two_register_kernel() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    qs = qmc.qubit_array(2, "qs")
    zeros = qmc.qubit_array(2, "zeros")
    for i in qmc.range(2):
        qs[i] = qmc.x(qs[i])
    bits = qmc.measure(qs)
    dst = qmc.measure(zeros)
    for i in qmc.range(2):
        dst[i] = bits[i]
    return dst, bits


def test_measured_bit_loop_store_between_registers(backend):
    """A loop-indexed store copies one register's readout into another's."""
    name, transpiler, executor = backend
    exe = transpiler.transpile(two_register_kernel)
    counts = _counts(exe.sample(executor, shots=100).result())
    assert counts == {((1, 1), (1, 1)): 100}, f"{name}: got {counts}"


def test_run_returns_stored_bits():
    """run() resolves the stored contents, matching the sample() path."""
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(copy_bit_kernel)
    result = exe.run(transpiler.executor()).result()
    assert tuple(int(b) for b in result) == (1, 1)


# ---------------------------------------------------------------------------
# Vector[Float] / Vector[UInt] (binding-derived)
# ---------------------------------------------------------------------------


@qmc.qkernel
def float_store_kernel(
    vals: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Float], qmc.Vector[qmc.Bit]]:
    vals[1] = vals[0]
    qs = qmc.qubit_array(1, "qs")
    bits = qmc.measure(qs)
    return vals, bits


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_float_element_store_and_return(seed):
    """A binding-derived Vector[Float] store is visible in the returned array."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(-np.pi, np.pi, size=2).tolist()
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(float_store_kernel, bindings={"vals": values})
    out_vals, out_bits = exe.run(transpiler.executor()).result()
    np.testing.assert_allclose(out_vals, (values[0], values[0]), atol=1e-12)
    assert tuple(out_bits) == (0,)


@qmc.qkernel
def uint_store_kernel(
    vals: qmc.Vector[qmc.UInt],
) -> tuple[qmc.Vector[qmc.UInt], qmc.Vector[qmc.Bit]]:
    vals[0] = vals[2]
    vals[1] = 7
    qs = qmc.qubit_array(1, "qs")
    bits = qmc.measure(qs)
    return vals, bits


def test_uint_element_and_literal_store():
    """Element-copy and literal stores both land in a Vector[UInt] output."""
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(uint_store_kernel, bindings={"vals": [1, 2, 3]})
    out_vals, _ = exe.run(transpiler.executor()).result()
    assert tuple(int(v) for v in out_vals) == (3, 7, 3)


@qmc.qkernel
def angle_store_kernel(vals: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(1, "qs")
    vals[1] = vals[0]
    qs[0] = qmc.rx(qs[0], vals[1])
    return qmc.measure(qs)


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_stored_element_folds_into_gate_angle(seed):
    """Reading a stored element as a gate angle bakes the post-store value.

    The store folds at compile time (all inputs bound), so the emitted rx
    angle must equal the stored value — not the stale pre-store binding.
    """
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-np.pi, np.pi))
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(angle_store_kernel, bindings={"vals": [angle, 0.0]})
    circuit = exe.compiled_quantum[0].circuit
    emitted = [
        float(inst.operation.params[0])
        for inst in circuit.data
        if inst.operation.name == "rx"
    ]
    np.testing.assert_allclose(emitted, [angle], atol=1e-12)


def test_stored_pi_angle_flips_qubit(backend):
    """End-to-end: rx(pi) through a stored element flips the qubit."""
    name, transpiler, executor = backend
    exe = transpiler.transpile(
        angle_store_kernel, bindings={"vals": [float(np.pi), 0.0]}
    )
    counts = _counts(exe.sample(executor, shots=100).result())
    assert counts == {(1,): 100}, f"{name}: got {counts}"


# ---------------------------------------------------------------------------
# Loop-index writes
# ---------------------------------------------------------------------------


@qmc.qkernel
def loop_fill_kernel(
    dst: qmc.Vector[qmc.UInt],
) -> tuple[qmc.Vector[qmc.UInt], qmc.Vector[qmc.Bit]]:
    n = dst.shape[0]
    for i in qmc.range(n):
        dst[i] = 9
    qs = qmc.qubit_array(1, "qs")
    bits = qmc.measure(qs)
    return dst, bits


@pytest.mark.parametrize("length", [1, 2, 3, 5])
def test_loop_index_literal_fill(length):
    """A loop-indexed literal store fills every slot across register sizes."""
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(loop_fill_kernel, bindings={"dst": [0] * length})
    out_vals, _ = exe.run(transpiler.executor()).result()
    assert tuple(int(v) for v in out_vals) == (9,) * length


@qmc.qkernel
def loop_copy_kernel(
    src: qmc.Vector[qmc.Float], dst: qmc.Vector[qmc.Float]
) -> tuple[qmc.Vector[qmc.Float], qmc.Vector[qmc.Bit]]:
    n = dst.shape[0]
    for i in qmc.range(n):
        dst[i] = src[i]
    qs = qmc.qubit_array(1, "qs")
    bits = qmc.measure(qs)
    return dst, bits


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("length", [1, 2, 5])
def test_loop_index_cross_array_copy(seed, length):
    """A loop-indexed store copies another array element-by-element."""
    rng = np.random.default_rng(seed)
    src = rng.uniform(-1.0, 1.0, size=length).tolist()
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        loop_copy_kernel, bindings={"src": src, "dst": [0.0] * length}
    )
    out_vals, _ = exe.run(transpiler.executor()).result()
    np.testing.assert_allclose(out_vals, src, atol=1e-12)


@qmc.qkernel
def loop_self_referential_kernel(
    vals: qmc.Vector[qmc.UInt], n: qmc.UInt
) -> tuple[qmc.Vector[qmc.UInt], qmc.Vector[qmc.Bit]]:
    for i in qmc.range(1, n):
        vals[i] = vals[0]
    qs = qmc.qubit_array(1, "qs")
    bits = qmc.measure(qs)
    return vals, bits


def test_loop_self_referential_store_raises():
    """Reading the written array inside a runtime loop fails at compile time.

    The single traced store references a fixed pre-loop array version, so
    iteration >= 2 would silently read stale contents; the transpiler
    rejects the kernel at compile time (pre-fold in partial_eval, again in
    analyze) instead of diverging from Python semantics.
    """
    transpiler = QiskitTranspiler()
    with pytest.raises(ValidationError, match="same array"):
        transpiler.transpile(
            loop_self_referential_kernel,
            bindings={"vals": [5, 0, 0]},
            parameters=["n"],
        )


@qmc.qkernel
def loop_arithmetic_self_referential_kernel(
    vals: qmc.Vector[qmc.UInt], n: qmc.UInt
) -> tuple[qmc.Vector[qmc.UInt], qmc.Vector[qmc.Bit]]:
    for i in qmc.range(n):
        vals[i] = vals[0] + 1
    qs = qmc.qubit_array(1, "qs")
    bits = qmc.measure(qs)
    return vals, bits


def test_loop_self_referential_store_via_arithmetic_rejected():
    """A same-array read behind a BinOp (``vals[i] = vals[0] + 1``) is rejected.

    The old immediate-operand runtime guard missed this arithmetic-intermediary
    form and silently wrote the folded pre-loop value every iteration.
    """
    transpiler = QiskitTranspiler()
    with pytest.raises(ValidationError, match="same array"):
        transpiler.transpile(
            loop_arithmetic_self_referential_kernel,
            bindings={"vals": [10, 0, 0]},
            parameters=["n"],
        )


# ---------------------------------------------------------------------------
# Rejected forms (loud errors instead of silent drops)
# ---------------------------------------------------------------------------


def test_type_mismatch_rejected():
    """Storing a float literal into a Vector[UInt] raises TypeError at build."""

    @qmc.qkernel
    def bad_type(vals: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
        vals[0] = 1.5
        qs = qmc.qubit_array(1, "qs")
        return qmc.measure(qs)

    with pytest.raises(TypeError, match="element type UInt"):
        QiskitTranspiler().transpile(bad_type, bindings={"vals": [1]})


def test_cross_type_handle_rejected():
    """Storing a measured Bit handle into a Vector[UInt] raises TypeError."""

    @qmc.qkernel
    def bad_handle(vals: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(1, "qs")
        bits = qmc.measure(qs)
        vals[0] = bits[0]
        return bits

    with pytest.raises(TypeError, match="Cannot assign Bit"):
        QiskitTranspiler().transpile(bad_handle, bindings={"vals": [1]})


def test_bool_literal_rejected_for_uint():
    """bool is not a valid UInt element value (mirrors index handling)."""

    @qmc.qkernel
    def bad_bool(vals: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
        vals[0] = True
        qs = qmc.qubit_array(1, "qs")
        return qmc.measure(qs)

    with pytest.raises(TypeError, match="bool"):
        QiskitTranspiler().transpile(bad_bool, bindings={"vals": [1]})


def test_out_of_range_store_rejected():
    """A constant index past the array length raises IndexError at build."""

    @qmc.qkernel
    def out_of_range(vals: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
        vals[5] = 1
        qs = qmc.qubit_array(1, "qs")
        return qmc.measure(qs)

    with pytest.raises(IndexError, match="out of range"):
        QiskitTranspiler().transpile(out_of_range, bindings={"vals": [1, 2]})


def test_negative_index_store_rejected():
    """Negative constant indices are rejected before any IR is built."""

    @qmc.qkernel
    def negative(vals: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
        vals[-1] = 1
        qs = qmc.qubit_array(1, "qs")
        return qmc.measure(qs)

    with pytest.raises(NotImplementedError, match="Negative index"):
        QiskitTranspiler().transpile(negative, bindings={"vals": [1, 2]})


def test_view_store_rejected():
    """Classical element stores through a slice view are rejected loudly."""

    @qmc.qkernel
    def view_store() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(4, "qs")
        bits = qmc.measure(qs)
        view = bits[0:2]
        view[0] = bits[3]
        return bits

    with pytest.raises(NotImplementedError, match="slice view"):
        QiskitTranspiler().transpile(view_store)


def test_matrix_store_rejected():
    """Multi-dimensional classical stores are rejected loudly."""

    @qmc.qkernel
    def matrix_store(m: qmc.Matrix[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        m[0, 1] = m[0, 0]
        qs = qmc.qubit_array(1, "qs")
        return qmc.measure(qs)

    with pytest.raises(NotImplementedError, match="1-D"):
        QiskitTranspiler().transpile(matrix_store, bindings={"m": [[1.0, 2.0]]})


def test_if_branch_store_rejected():
    """Stores inside a measurement-backed if/else branch are rejected.

    Array values have no phi merge, so the store would apply regardless of
    the branch taken.
    """

    @qmc.qkernel
    def if_store() -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        bits = qmc.measure(qs)
        if bits[0]:
            bits[1] = bits[0]
        return bits

    with pytest.raises(ValidationError, match="if/else branch"):
        QiskitTranspiler().transpile(if_store)


def test_store_in_quantum_loop_rejected():
    """A store inside a mixed quantum/classical loop body fails at emit.

    The mixed body routes the whole loop into the quantum segment; the
    store cannot be emitted there and must not be silently skipped.
    """

    @qmc.qkernel
    def mixed_loop(vals: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        qs = qmc.qubit_array(2, "qs")
        for i in qmc.range(2):
            qs[i] = qmc.h(qs[i])
            vals[i] = 0.5
        return qmc.measure(qs)

    with pytest.raises(EmitError, match="reached the quantum segment"):
        QiskitTranspiler().transpile(mixed_loop, bindings={"vals": [0.0, 0.0]})


def test_runtime_parameter_store_feeding_gate_rejected():
    """A non-foldable store whose element feeds a gate angle fails at emit.

    With the array left as a runtime parameter the store cannot fold, so
    the stale-metadata guard must surface an error instead of silently
    emitting the pre-store binding as the angle.
    """
    with pytest.raises(EmitError, match="reached the quantum segment"):
        QiskitTranspiler().transpile(angle_store_kernel, parameters=["vals"])


# ---------------------------------------------------------------------------
# IR structure and serialization
# ---------------------------------------------------------------------------


def test_store_op_shape_in_ir():
    """The frontend emits one store op with [array, value, index] operands."""
    transpiler = QiskitTranspiler()
    block = transpiler.inline(transpiler.to_block(copy_bit_kernel))
    stores = [
        op for op in block.operations if isinstance(op, StoreArrayElementOperation)
    ]
    assert len(stores) == 1
    store = stores[0]
    assert len(store.operands) == 3
    assert store.array.uuid != store.results[0].uuid
    assert store.array.logical_id == store.results[0].logical_id
    assert store.results[0].version == store.array.version + 1
    # Stale compile-time payloads must not survive onto the result version.
    assert store.results[0].metadata.scalar is None
    assert store.results[0].metadata.array_runtime is None
    # The block returns the post-store version, not the measure result.
    assert block.output_values[0].uuid == store.results[0].uuid


def test_store_op_serialize_roundtrip():
    """A block containing a store op round-trips through JSON and msgpack."""
    from qamomile.circuit.ir import serialize

    transpiler = QiskitTranspiler()
    block = transpiler.inline(transpiler.to_block(copy_bit_kernel))

    for dump, load in (
        (serialize.dump_json, serialize.load_json),
        (serialize.dump_msgpack, serialize.load_msgpack),
    ):
        restored = load(dump(block))
        restored_ops = [type(op).__name__ for op in restored.operations]
        assert restored_ops == [type(op).__name__ for op in block.operations]
        restored_store = next(
            op
            for op in restored.operations
            if isinstance(op, StoreArrayElementOperation)
        )
        assert len(restored_store.operands) == 3
        assert restored_store.results[0].uuid == block.output_values[0].uuid


def test_store_op_canonicalize_stable():
    """Canonicalization accepts store ops and hashes them stably."""
    from qamomile.circuit.ir import canonical

    transpiler = QiskitTranspiler()
    block = transpiler.inline(transpiler.to_block(copy_bit_kernel))
    canon = canonical.canonicalize(block)
    assert canonical.content_hash(block) == canonical.content_hash(canon)
