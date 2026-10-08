"""Regression tests: affine_validate descends into operation-owned blocks.

`AffineValidationPass` used to recurse through control-flow nesting only
(``HasNestedOps``); blocks OWNED by an operation — ``ControlledUOperation.block``,
``InverseBlockOperation.source_block`` / ``implementation_block``, un-inlined
``InvokeOperation.definition.body`` — were never entered by any transpile-time
backstop. An IR-level double consume placed inside a controlled block passed
``affine_validate`` (the identical tamper at top level raised), passed the full
pipeline, and executed a physically wrong circuit (``{(0,0): 256}`` where the
untampered kernel gives ``{(1,1): 256}``). Reachable from hand-built or
deserialized IR, or a future pass bug — exactly the backstop's mandate.

Each owned block is now validated as an independent affine scope (fresh
consumed map; its ``input_values`` are the ownership boundary), recursively,
so boxed-in-boxed nesting and control flow inside a boxed body are covered.
"""

import dataclasses

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.gate import ControlledUOperation, GateOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.transpiler.errors import AffineTypeError
from qamomile.circuit.transpiler.passes.affine_validate import (
    operation_owned_blocks,
)
from qamomile.qiskit import QiskitTranspiler


def _tamper_double_consume(inner):
    """Rewire the 2nd gate of a block to re-consume the 1st gate's input.

    Args:
        inner (Block): Block containing at least two ``GateOperation``s.

    Returns:
        Block: Copy of ``inner`` whose second gate operand aliases the
        first gate's (already consumed) input value.
    """
    gates = [o for o in inner.operations if isinstance(o, GateOperation)]
    assert len(gates) >= 2, f"need >=2 inner gates, got {len(gates)}"
    g1, g2 = gates[0], gates[1]
    bad_g2 = g2.replace_values({g2.operands[0].uuid: g1.operands[0]})
    return dataclasses.replace(
        inner, operations=[bad_g2 if o is g2 else o for o in inner.operations]
    )


def _replace_op(block, old, new):
    """Return a copy of ``block`` with ``old`` swapped for ``new``.

    Args:
        block (Block): Block whose operation list is rewritten.
        old (Operation): Operation instance to replace (identity match).
        new (Operation): Replacement operation.

    Returns:
        Block: Copy with the swapped operation list.
    """
    return dataclasses.replace(
        block, operations=[new if o is old else o for o in block.operations]
    )


def _find(cls, operations):
    """Return the first operation of type ``cls``.

    Args:
        cls (type): Operation class to search for.
        operations (list[Operation]): Operations to scan.

    Returns:
        Operation: First match.

    Raises:
        AssertionError: If no operation of ``cls`` is present.
    """
    for op in operations:
        if isinstance(op, cls):
            return op
    raise AssertionError(f"no {cls.__name__} in block")


@qmc.qkernel
def _two_gate_sub(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    a = qmc.x(a)
    b = qmc.x(b)
    return a, b


_ctrl = qmc.control(_two_gate_sub)


@qmc.qkernel
def _controlled_main() -> tuple[qmc.Bit, qmc.Bit]:
    c = qmc.qubit("c")
    a = qmc.qubit("a")
    b = qmc.qubit("b")
    c = qmc.x(c)  # control ON so the boxed body fires
    c, a, b = _ctrl(c, a, b)
    return qmc.measure(a), qmc.measure(b)


def _tampered_controlled_block(transpiler):
    """Build ``_controlled_main`` and tamper its controlled body.

    Args:
        transpiler (QiskitTranspiler): Pipeline entry used for building.

    Returns:
        Block: Inlined AFFINE block whose ``ControlledUOperation.block``
        contains an IR-level double consume.
    """
    block = transpiler.inline(transpiler.to_block(_controlled_main))
    cu = _find(ControlledUOperation, block.operations)
    bad_cu = dataclasses.replace(cu, block=_tamper_double_consume(cu.block))
    return _replace_op(block, cu, bad_cu)


def _run_pipeline(transpiler, block):
    """Run affine_validate through emit, mirroring ``transpile()``'s tail.

    Args:
        transpiler (QiskitTranspiler): Pipeline entry.
        block (Block): Inlined AFFINE block.

    Returns:
        Any: Emitted executable.
    """
    b = transpiler.affine_validate(block)
    b = transpiler.partial_eval(b)
    b = transpiler.analyze(b)
    b = transpiler.classical_lowering(b)
    b = transpiler.validate_symbolic_shapes(b)
    return transpiler.emit(transpiler.plan(b))


class TestTamperedOwnedBlocksRejected:
    """IR-level double consumes inside owned blocks are caught by the backstop."""

    def test_tampered_controlled_block_rejected_at_affine_validate(self):
        """The tamper that used to pass affine_validate now raises there."""
        tr = QiskitTranspiler()
        with pytest.raises(AffineTypeError, match="already consumed"):
            tr.affine_validate(_tampered_controlled_block(tr))

    def test_tampered_controlled_block_never_reaches_emit(self):
        """The full pipeline stops at affine_validate instead of silently
        emitting the physically wrong circuit."""
        tr = QiskitTranspiler()
        with pytest.raises(AffineTypeError, match="already consumed"):
            _run_pipeline(tr, _tampered_controlled_block(tr))

    def test_top_level_tamper_still_rejected(self):
        """Regression pin: the pre-existing top-level double-consume
        detection is unchanged."""

        @qmc.qkernel
        def chain() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.x(q)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        block = tr.inline(tr.to_block(chain))
        with pytest.raises(AffineTypeError, match="already consumed"):
            tr.affine_validate(_tamper_double_consume(block))

    def test_tampered_inverse_block_rejected(self):
        """A double consume inside an InverseBlockOperation's owned block
        is caught too — both owned block fields are descended."""

        @qmc.qkernel
        def inv_main() -> tuple[qmc.Bit, qmc.Bit]:
            a = qmc.qubit("a")
            b = qmc.qubit("b")
            a, b = qmc.inverse(_two_gate_sub)(a, b)
            return qmc.measure(a), qmc.measure(b)

        tr = QiskitTranspiler()
        block = tr.inline(tr.to_block(inv_main))
        inv = _find(InverseBlockOperation, block.operations)
        owned = operation_owned_blocks(inv)
        assert owned, "inverse op should own at least one block"
        target = owned[0]
        tampered = _tamper_double_consume(target)
        if inv.source_block is target:
            bad_inv = dataclasses.replace(inv, source_block=tampered)
        else:
            bad_inv = dataclasses.replace(inv, implementation_block=tampered)
        with pytest.raises(AffineTypeError, match="already consumed"):
            tr.affine_validate(_replace_op(block, inv, bad_inv))

    def test_boxed_in_boxed_tamper_rejected(self):
        """Recursion reaches a controlled block nested INSIDE another
        controlled block."""

        @qmc.qkernel
        def one_q_two_gates(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.x(q)
            q = qmc.h(q)
            return q

        inner_ctrl = qmc.control(one_q_two_gates)

        @qmc.qkernel
        def outer_sub(c2: qmc.Qubit, q: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            c2, q = inner_ctrl(c2, q)
            return c2, q

        outer_ctrl = qmc.control(outer_sub)

        @qmc.qkernel
        def nested_main() -> tuple[qmc.Bit, qmc.Bit]:
            c = qmc.qubit("c")
            a = qmc.qubit("a")
            b = qmc.qubit("b")
            c, a, b = outer_ctrl(c, a, b)
            return qmc.measure(a), qmc.measure(b)

        tr = QiskitTranspiler()
        block = tr.inline(tr.to_block(nested_main))
        outer_cu = _find(ControlledUOperation, block.operations)
        inner_cu = _find(ControlledUOperation, outer_cu.block.operations)
        bad_inner = dataclasses.replace(
            inner_cu, block=_tamper_double_consume(inner_cu.block)
        )
        bad_outer = dataclasses.replace(
            outer_cu, block=_replace_op(outer_cu.block, inner_cu, bad_inner)
        )
        with pytest.raises(AffineTypeError, match="already consumed"):
            tr.affine_validate(_replace_op(block, outer_cu, bad_outer))


class TestLegitimateOwnedBlocksStillPass:
    """Fresh-scope descent must not false-positive on valid recipes."""

    def test_untampered_controlled_kernel_executes(self):
        """The untampered controlled kernel passes validation and executes
        the physically correct circuit: control ON, both targets flipped."""
        pytest.importorskip("qiskit_aer")
        from qiskit_aer import AerSimulator

        tr = QiskitTranspiler()
        exe = _run_pipeline(tr, tr.inline(tr.to_block(_controlled_main)))
        job = exe.sample(
            tr.executor(backend=AerSimulator(seed_simulator=42)), shots=256
        )
        assert dict(job.result().results) == {(1, 1): 256}

    def test_untampered_inverse_kernel_executes(self):
        """An inverse kernel passes validation and executes: the body is one
        self-inverse X per qubit, so its inverse is X per qubit again and
        both |0> targets measure 1."""
        pytest.importorskip("qiskit_aer")
        from qiskit_aer import AerSimulator

        @qmc.qkernel
        def inv_main() -> tuple[qmc.Bit, qmc.Bit]:
            a = qmc.qubit("a")
            b = qmc.qubit("b")
            a, b = qmc.inverse(_two_gate_sub)(a, b)
            return qmc.measure(a), qmc.measure(b)

        tr = QiskitTranspiler()
        exe = _run_pipeline(tr, tr.inline(tr.to_block(inv_main)))
        job = exe.sample(
            tr.executor(backend=AerSimulator(seed_simulator=42)), shots=256
        )
        assert dict(job.result().results) == {(1, 1): 256}


class TestOwnedBlockEnumeration:
    """The walker enumerates exactly the block-owning operation families."""

    def test_controlled_and_plain_ops_enumerated(self):
        """ControlledUOperation exposes its body; plain gates own nothing."""
        tr = QiskitTranspiler()
        block = tr.inline(tr.to_block(_controlled_main))
        cu = _find(ControlledUOperation, block.operations)
        assert operation_owned_blocks(cu) == [cu.block]
        gate = _find(GateOperation, block.operations)
        assert operation_owned_blocks(gate) == []

    def test_invoke_definition_bodies_enumerated(self):
        """An un-inlined invoke exposes BOTH the standard definition body
        and every implementation body — the same both-kinds enumeration
        ``region_validation`` uses, so a tampered implementation body
        cannot hide from the descent."""
        from qamomile.circuit.ir.operation.callable import (
            CallableDef,
            CallableImplementation,
            CallableRef,
            InvokeOperation,
        )

        tr = QiskitTranspiler()
        block = tr.inline(tr.to_block(_controlled_main))
        cu = _find(ControlledUOperation, block.operations)
        standard_body = cu.block
        implementation_body = block  # any distinct Block object works

        invoke = InvokeOperation(
            operands=[],
            results=[],
            target=CallableRef(namespace="user", name="probe"),
            definition=CallableDef(
                ref=CallableRef(namespace="user", name="probe"),
                body=standard_body,
                implementations=[
                    CallableImplementation(body=implementation_body),
                    CallableImplementation(body=None),
                ],
            ),
        )
        assert operation_owned_blocks(invoke) == [
            standard_body,
            implementation_body,
        ]
