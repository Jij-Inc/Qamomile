"""Tests for the IR serialization pipeline (JSON + msgpack).

The intermediate dict schema lives in
:mod:`qamomile.circuit.ir.serialize.schema`. Encoders and decoders
should produce byte-stable round-trips for ``AFFINE`` / ``ANALYZED``
Blocks across a representative spread of IR features (gates,
measurement, control flow, ControlledU, composite gates).
"""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest

import qamomile.circuit as qmc

# Real algorithm kernels used by TestRealAlgorithmRoundTrip below.
from qamomile.circuit.algorithm.basic import (  # noqa: E402
    cz_entangling_layer,
    phase_gadget,
    rx_layer,
    superposition_vector,
)
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.parameter import ParamKind, ParamSlot
from qamomile.circuit.ir.serialize import (
    SCHEMA_VERSION,
    dump_json,
    dump_msgpack,
    from_dict,
    load_json,
    load_msgpack,
    to_dict,
)
from qamomile.circuit.ir.serialize.numpy_io import array_to_dict, dict_to_array
from qamomile.circuit.ir.types.primitives import (
    FloatType,
)
from qamomile.circuit.ir.value import (
    ArrayRuntimeMetadata,
    ArrayValue,
    ScalarMetadata,
    Value,
    ValueMetadata,
)
from qamomile.circuit.stdlib.qft import iqft, qft  # noqa: E402
from qamomile.circuit.transpiler.passes.inline import InlinePass

# ---------------------------------------------------------------------------
# Fixture kernels (representative IR shapes)
# ---------------------------------------------------------------------------


def _to_affine(kernel: qmc.QKernel) -> Block:
    """Return an AFFINE block for ``kernel`` without a backend transpiler.

    Args:
        kernel (qmc.QKernel): A ``@qkernel``-decorated function.

    Returns:
        Block: The kernel's traced block after running ``InlinePass``.
    """
    return InlinePass().run(kernel.block)


@qmc.qkernel
def _scalar_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single-qubit kernel with a Float parameter."""
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return q


@qmc.qkernel
def _measure_kernel(q: qmc.Qubit) -> qmc.Bit:
    """Kernel that exercises a measurement-derived classical bit."""
    q = qmc.h(q)
    return qmc.measure(q)


@qmc.qkernel
def _loop_kernel(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Symbolic ``ForOperation`` over a Vector input."""
    n = qs.shape[0]
    for i in qmc.range(n):
        qs[i] = qmc.h(qs[i])
    return qs


@qmc.qkernel
def _phase(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single-qubit phase rotation used inside a controlled-U body."""
    return qmc.p(q, theta)


@qmc.qkernel
def _controlled_phase(
    ctrl: qmc.Qubit, target: qmc.Qubit, theta: qmc.Float
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Top-level kernel that embeds a ``ControlledUOperation``."""
    op = qmc.controlled(_phase)
    ctrl, target = op(ctrl, target, theta=theta)
    return ctrl, target


# ---------------------------------------------------------------------------
# Smoke + structure preservation
# ---------------------------------------------------------------------------


class TestRoundTripStructure:
    """Encoder + decoder preserve kind, operation order, and param_slots."""

    def test_json_round_trip_preserves_kind_and_op_count(self):
        """``load_json(dump_json(block))`` produces structurally similar IR."""
        block = _to_affine(_scalar_gate)
        restored = load_json(dump_json(block))
        assert restored.kind == block.kind
        assert len(restored.operations) == len(block.operations)
        assert [type(op).__name__ for op in restored.operations] == [
            type(op).__name__ for op in block.operations
        ]

    def test_msgpack_round_trip_preserves_kind_and_op_count(self):
        """msgpack round-trip mirrors the JSON one."""
        block = _to_affine(_scalar_gate)
        restored = load_msgpack(dump_msgpack(block))
        assert restored.kind == block.kind
        assert len(restored.operations) == len(block.operations)

    def test_label_args_preserved(self):
        """``label_args`` round-trips byte-for-byte."""
        block = _to_affine(_scalar_gate)
        restored = load_json(dump_json(block))
        assert restored.label_args == block.label_args

    def test_param_slots_preserved(self):
        """``param_slots`` round-trips with name, kind, ndim intact."""
        block = _to_affine(_scalar_gate)
        restored = load_json(dump_json(block))
        assert len(restored.param_slots) == len(block.param_slots)
        for original, restored_slot in zip(block.param_slots, restored.param_slots):
            assert original.name == restored_slot.name
            assert original.kind == restored_slot.kind
            assert original.ndim == restored_slot.ndim
            assert isinstance(restored_slot.type, type(original.type))

    def test_input_value_types_preserved(self):
        """Input Value types match across the round-trip."""
        block = _to_affine(_scalar_gate)
        restored = load_json(dump_json(block))
        assert [type(v.type).__name__ for v in restored.input_values] == [
            type(v.type).__name__ for v in block.input_values
        ]


class TestRoundTripIRFeatures:
    """Round-trip works for IR features beyond plain gate sequences."""

    def test_measurement_round_trip(self):
        """Kernels with ``MeasureOperation`` survive both wire formats."""
        block = _to_affine(_measure_kernel)
        assert load_json(dump_json(block)).kind == block.kind
        assert load_msgpack(dump_msgpack(block)).kind == block.kind

    def test_for_loop_round_trip(self):
        """Symbolic ``ForOperation`` survives serialization."""
        block = _to_affine(_loop_kernel)
        restored = load_json(dump_json(block))
        # The top-level operation list should retain the for loop wrapper.
        op_names = [type(op).__name__ for op in restored.operations]
        assert "ForOperation" in op_names

    def test_controlled_u_round_trip(self):
        """``ControlledUOperation`` with nested unitary block round-trips."""
        block = _to_affine(_controlled_phase)
        restored = load_json(dump_json(block))
        op_names = [type(op).__name__ for op in restored.operations]
        assert any(name.endswith("ControlledU") for name in op_names), op_names

    def test_msgpack_smaller_than_json_for_loop_kernel(self):
        """msgpack should produce at least as compact bytes as JSON.

        Acts as a regression guard against accidental textification of
        binary fields in the msgpack path.
        """
        block = _to_affine(_loop_kernel)
        assert len(dump_msgpack(block)) <= len(dump_json(block))


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    """Schema-version envelope is enforced."""

    def test_envelope_has_current_version(self):
        """``to_dict`` emits the current ``SCHEMA_VERSION`` in the envelope."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        assert d["schema_version"] == SCHEMA_VERSION

    def test_load_rejects_future_version(self):
        """A higher ``schema_version`` than the loader supports raises."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        d["schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema_version"):
            from_dict(d)

    def test_load_rejects_past_version(self):
        """A lower ``schema_version`` raises until migration is implemented."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        d["schema_version"] = 0
        with pytest.raises(ValueError, match="schema_version"):
            from_dict(d)


# ---------------------------------------------------------------------------
# numpy wrapper
# ---------------------------------------------------------------------------


class TestNumpyWrapper:
    """The numpy ndarray wrapper preserves dtype, shape, and values."""

    @pytest.mark.parametrize(
        "dtype",
        ["float64", "float32", "int64", "int32", "uint8", "complex128"],
    )
    def test_array_round_trip_dtype(self, dtype):
        """Each allow-listed dtype round-trips losslessly."""
        rng = np.random.default_rng(seed=0)
        arr = (rng.random(size=(2, 3)) * 100).astype(dtype)
        restored = dict_to_array(array_to_dict(arr))
        assert restored.dtype.name == arr.dtype.name
        assert restored.shape == arr.shape
        assert np.array_equal(restored, arr)

    def test_disallowed_dtype_rejected(self):
        """A dtype outside the allow-list raises on encode."""
        arr = np.array(["a", "b"])
        with pytest.raises(ValueError, match="allow-list"):
            array_to_dict(arr)

    def test_decoder_rejects_corrupt_length(self):
        """A length mismatch in the wrapper raises on decode."""
        wrapper = array_to_dict(np.array([1.0, 2.0, 3.0]))
        wrapper["data"] = wrapper["data"][:-1]  # truncate one byte
        with pytest.raises(ValueError, match="data length"):
            dict_to_array(wrapper)

    def test_numpy_array_in_param_slot_bound_value(self):
        """A numpy ``bound_value`` on a ParamSlot round-trips via JSON.

        Construct a Block manually carrying a numpy bound_value so we
        can verify both the encoder routes through ``array_to_dict``
        and the JSON path converts bytes ↔ base64 transparently.
        """
        original = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        slot = ParamSlot(
            name="thetas",
            type=FloatType(),
            kind=ParamKind.COMPILE_TIME_BOUND,
            ndim=1,
            bound_value=original,
        )
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            param_slots=(slot,),
        )
        restored = load_json(dump_json(block))
        restored_slot = restored.param_slots[0]
        assert isinstance(restored_slot.bound_value, np.ndarray)
        assert np.array_equal(restored_slot.bound_value, original)
        assert restored_slot.bound_value.dtype == original.dtype


# ---------------------------------------------------------------------------
# Safety: closed dispatch tables refuse unknown tags
# ---------------------------------------------------------------------------


class TestDecoderSafety:
    """The decoder refuses unknown ``$type`` tags rather than resolve them."""

    def test_unknown_operation_tag_rejected(self):
        """An unrecognized operation tag raises ``ValueError``."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        d["block"]["operations"][0]["$type"] = "__rogue__"
        with pytest.raises(ValueError, match="Operation"):
            from_dict(d)

    def test_unknown_value_type_tag_rejected(self):
        """An unrecognized value-type tag raises ``ValueError``."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        d["block"]["value_table"][0]["value_type"]["$type"] = "__rogue__"
        with pytest.raises(ValueError, match="ValueType"):
            from_dict(d)


# ---------------------------------------------------------------------------
# Unsupported scope: HIERARCHICAL / CallBlockOperation
# ---------------------------------------------------------------------------


class TestUnsupportedKind:
    """``to_dict`` rejects unsupported block kinds and CallBlockOperation."""

    def test_rejects_hierarchical(self):
        """HIERARCHICAL blocks must be inlined first."""
        hierarchical = dataclasses.replace(
            _scalar_gate.block, kind=BlockKind.HIERARCHICAL
        )
        with pytest.raises(ValueError, match="AFFINE"):
            to_dict(hierarchical)

    def test_rejects_call_block_operation(self):
        """A residual ``CallBlockOperation`` raises ``NotImplementedError``."""
        block = _to_affine(_scalar_gate)
        bad = dataclasses.replace(
            block,
            operations=[*block.operations, CallBlockOperation(block=block)],
        )
        with pytest.raises(NotImplementedError, match="CallBlockOperation"):
            to_dict(bad)


# ---------------------------------------------------------------------------
# Deterministic byte stability under canonical UUIDs
# ---------------------------------------------------------------------------


class TestEncodedShape:
    """The encoded dict has the documented top-level shape."""

    def test_envelope_keys(self):
        """The envelope carries exactly ``schema_version`` and ``block``."""
        d = to_dict(_to_affine(_scalar_gate))
        assert set(d) == {"schema_version", "block"}

    def test_block_has_value_table_and_operations(self):
        """The block dict includes ``value_table`` and ``operations`` lists."""
        block_dict = to_dict(_to_affine(_scalar_gate))["block"]
        assert isinstance(block_dict["value_table"], list)
        assert isinstance(block_dict["operations"], list)
        assert block_dict["$type"] == "Block"


# ---------------------------------------------------------------------------
# Manual construction with metadata UUID references
# ---------------------------------------------------------------------------


class TestManualConstruction:
    """Hand-built Blocks with metadata round-trip without going through tracing."""

    def test_array_runtime_metadata_round_trip(self):
        """``ArrayRuntimeMetadata`` element UUIDs round-trip exactly."""
        e0 = Value(type=FloatType(), name="e0")
        e1 = Value(type=FloatType(), name="e1")
        arr = ArrayValue(
            type=FloatType(),
            name="arr",
            metadata=ValueMetadata(
                array_runtime=ArrayRuntimeMetadata(
                    element_uuids=(e0.uuid, e1.uuid),
                    element_logical_ids=(e0.logical_id, e1.logical_id),
                ),
            ),
        )
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[e0, e1],
            output_values=cast(list[Value], [arr]),
            label_args=["e0", "e1"],
            output_names=["arr"],
        )
        restored = load_json(dump_json(block))
        restored_arr = restored.output_values[0]
        assert restored_arr.metadata.array_runtime is not None
        assert restored_arr.metadata.array_runtime.element_uuids == (e0.uuid, e1.uuid)

    def test_scalar_metadata_round_trip(self):
        """``ScalarMetadata`` parameter_name / const_value round-trip."""
        v = Value(
            type=FloatType(),
            name="theta",
            metadata=ValueMetadata(
                scalar=ScalarMetadata(const_value=0.5, parameter_name="theta")
            ),
        )
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[v],
            output_values=[v],
            label_args=["theta"],
        )
        restored = load_json(dump_json(block))
        assert restored.input_values[0].metadata.scalar == ScalarMetadata(
            const_value=0.5, parameter_name="theta"
        )

    def test_extra_x_gate_changes_bytes(self):
        """Adding a gate changes the serialized bytes (regression guard)."""
        block = _to_affine(_scalar_gate)
        baseline = dump_json(block)
        last_result = block.operations[-1].results[0]
        new_q = last_result.next_version()
        modified = dataclasses.replace(
            block,
            operations=[
                *block.operations,
                GateOperation.fixed(
                    GateOperationType.X,
                    qubits=[last_result],
                    results=[new_q],
                ),
            ],
            output_values=[new_q],
        )
        assert baseline != dump_json(modified)


# ---------------------------------------------------------------------------
# Real-algorithm round-trip
# ---------------------------------------------------------------------------


@qmc.qkernel
def _bell_state() -> qmc.Vector[qmc.Qubit]:
    """Bell state: H on q0, then CX(q0, q1). Two-qubit, two-op kernel."""
    qs = qmc.qubit_array(2, "qs")
    qs[0] = qmc.h(qs[0])
    qs[0], qs[1] = qmc.cx(qs[0], qs[1])
    return qs


@qmc.qkernel
def _ansatz_h_rx_cz(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
    """A small variational ansatz: superposition + Rx layer + CZ ladder.

    Stresses ``ForOperation`` over a symbolic ``Vector[Float]`` plus
    multi-qubit ``cz`` tuple assignment, both common shapes in real
    QAOA / VQE ansatze.
    """
    n = thetas.shape[0]
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], thetas[i])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
    return q


@qmc.qkernel
def _algo_superposition_5() -> qmc.Vector[qmc.Qubit]:
    """Exercise the ``superposition_vector`` stdlib kernel via composition."""
    return superposition_vector(5)  # type: ignore[arg-type]


@qmc.qkernel
def _algo_rx_then_cz(
    thetas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Compose two ``qamomile.circuit.algorithm.basic`` kernels.

    Forces ``InlinePass`` to flatten two distinct ``CallBlockOperation``s
    and confirms the serialized form still survives.
    """
    n = thetas.shape[0]
    q = qmc.qubit_array(n, "q")
    q = rx_layer(q, thetas, 0)  # type: ignore[arg-type]
    q = cz_entangling_layer(q)
    return q


@qmc.qkernel
def _algo_phase_gadget(
    indices: qmc.Vector[qmc.UInt], angle: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    """Exercise the ``phase_gadget`` stdlib kernel (CX ladder + RZ).

    ``indices`` and ``angle`` are bound at build time so the gadget's
    ``k`` (number of qubits in the term) is concrete.
    """
    q = qmc.qubit_array(3, "q")
    return phase_gadget(q, indices, angle)


@qmc.qkernel
def _qft_3() -> qmc.Vector[qmc.Qubit]:
    """QFT on 3 qubits — exercises ``CompositeGateOperation`` + nested block."""
    qs = qmc.qubit_array(3, "qs")
    return qft(qs)


@qmc.qkernel
def _qft_then_iqft_4() -> qmc.Vector[qmc.Qubit]:
    """QFT followed by IQFT — two composite gates in sequence."""
    qs = qmc.qubit_array(4, "qs")
    qs = qft(qs)
    qs = iqft(qs)
    return qs


_REAL_KERNELS = [
    pytest.param(_bell_state, {}, id="bell"),
    pytest.param(
        _ansatz_h_rx_cz,
        {"thetas": np.array([0.1, 0.2, 0.3, 0.4])},
        id="ansatz_h_rx_cz",
    ),
    pytest.param(_algo_superposition_5, {}, id="superposition_5"),
    pytest.param(
        _algo_rx_then_cz,
        {"thetas": np.array([0.5, 0.6, 0.7])},
        id="rx_then_cz",
    ),
    pytest.param(
        _algo_phase_gadget,
        {"indices": np.array([0, 1, 2], dtype=np.int64), "angle": 0.7},
        id="phase_gadget",
    ),
    pytest.param(_qft_3, {}, id="qft_3"),
    pytest.param(_qft_then_iqft_4, {}, id="qft_then_iqft_4"),
]


def _build_real(kernel, build_kwargs):
    """Build an AFFINE block for a real-algorithm kernel.

    Uses ``kernel.build(**kwargs)`` when bindings are provided
    (typical for kernels with ``Vector[Float]`` parameters) and falls
    back to the cached ``kernel.block`` otherwise; either way the
    block is then run through ``InlinePass`` so the result is AFFINE.

    Args:
        kernel: A ``@qkernel``-decorated function.
        build_kwargs: Bindings for ``kernel.build``. Empty dict means
            use ``kernel.block`` directly.

    Returns:
        Block: The AFFINE block ready for serialization.
    """
    raw = kernel.build(**build_kwargs) if build_kwargs else kernel.block
    return InlinePass().run(raw)


class TestRealAlgorithmRoundTrip:
    """End-to-end round-trip on representative real-world kernels.

    Each kernel is built into an AFFINE block, serialized through one
    of the wire formats, decoded back, and re-encoded into the
    intermediate dict. ``to_dict`` equality on the two dicts is the
    strongest equivalence we can express without depending on the
    canonical-form pass (PR #389, separate branch): UUIDs round-trip
    verbatim, every operation field is preserved, and the value table
    is rebuilt in the same order.
    """

    @pytest.mark.parametrize("kernel,build_kwargs", _REAL_KERNELS)
    def test_json_round_trip_to_dict_equal(self, kernel, build_kwargs):
        """``to_dict(load_json(dump_json(b))) == to_dict(b)`` for real kernels."""
        block = _build_real(kernel, build_kwargs)
        original = to_dict(block)
        restored = to_dict(load_json(dump_json(block)))
        assert restored == original

    @pytest.mark.parametrize("kernel,build_kwargs", _REAL_KERNELS)
    def test_msgpack_round_trip_to_dict_equal(self, kernel, build_kwargs):
        """``to_dict(load_msgpack(dump_msgpack(b))) == to_dict(b)`` for real kernels."""
        block = _build_real(kernel, build_kwargs)
        original = to_dict(block)
        restored = to_dict(load_msgpack(dump_msgpack(block)))
        assert restored == original

    @pytest.mark.parametrize("kernel,build_kwargs", _REAL_KERNELS)
    def test_round_trip_preserves_op_sequence(self, kernel, build_kwargs):
        """Operation type and order are byte-stable across the round-trip."""
        block = _build_real(kernel, build_kwargs)
        restored = load_json(dump_json(block))
        assert [type(op).__name__ for op in restored.operations] == [
            type(op).__name__ for op in block.operations
        ]

    @pytest.mark.parametrize("kernel,build_kwargs", _REAL_KERNELS)
    def test_round_trip_preserves_value_uuids(self, kernel, build_kwargs):
        """Every Value UUID present in the original survives the round-trip."""
        block = _build_real(kernel, build_kwargs)
        restored = load_json(dump_json(block))
        original_uuids = {v.uuid for v in block.input_values} | {
            v.uuid for v in block.output_values
        }
        restored_uuids = {v.uuid for v in restored.input_values} | {
            v.uuid for v in restored.output_values
        }
        assert restored_uuids == original_uuids

    def test_qft_round_trip_preserves_nested_block(self):
        """QFT's nested ``implementation_block`` survives byte-for-byte.

        This is a focused regression test: the encoder must emit the
        nested unitary block separately rather than via ``repr`` (the
        same fix that addressed Copilot's review on PR #389 for
        ControlledU's nested block).
        """
        block = _build_real(_qft_3, {})
        restored = load_msgpack(dump_msgpack(block))
        original_d = to_dict(block)
        restored_d = to_dict(restored)
        # Find the CompositeGateOperation in operations
        composite = next(
            op
            for op in original_d["block"]["operations"]
            if op["$type"] == "CompositeGateOperation"
        )
        composite_r = next(
            op
            for op in restored_d["block"]["operations"]
            if op["$type"] == "CompositeGateOperation"
        )
        assert composite["implementation_block"] == composite_r["implementation_block"]
