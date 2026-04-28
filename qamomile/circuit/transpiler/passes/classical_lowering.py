"""Classical-op lowering pass: identify runtime-evaluation classical ops.

Walks the block, identifies ``CompOp`` / ``CondOp`` / ``NotOp`` / ``BinOp``
instances whose operand dataflow traces back to a ``MeasureOperation``
(i.e. cannot be folded at compile-time, by emit-time loop unrolling, or
by ``compile_time_if_lowering``), and replaces them with the equivalent
``RuntimeClassicalExpr``.

Why this pass exists:

The pre-``RuntimeClassicalExpr`` design left runtime classical ops in
their compile-time IR form (``CompOp`` etc.) all the way to emit, where
the emit pass had to fold-or-translate via ``evaluate_classical_predicate``
+ ``_build_runtime_predicate_expr``. This put backend-specific lowering
logic inside the emit pass and used the ``bindings`` dict as a polymorphic
slot holding either Python scalars (fold result) or backend ``Expr``
objects.

By identifying runtime classical ops at IR level and giving them their
own node type, we:

- Make "runtime evaluation required" structurally explicit in the IR.
- Move backend lowering to a dedicated emit hook (``_emit_runtime_classical_expr``).
- Preserve the existing fold path for ops that *can* fold at compile or
  emit time (loop-bound or parameter-bound) — those are not measurement-
  derived and stay as ``CompOp``/``CondOp``/``NotOp``/``BinOp``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    RuntimeClassicalExpr,
    runtime_kind_from_binop,
    runtime_kind_from_compop,
    runtime_kind_from_condop,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.value import ValueBase
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.analyze import (
    build_dependency_graph,
    find_measurement_derived_values,
    find_measurement_results,
)


class ClassicalLoweringPass(Pass[Block, Block]):
    """Lower measurement-derived classical ops to ``RuntimeClassicalExpr``.

    Input: Block with ``BlockKind.ANALYZED``.
    Output: Block with ``BlockKind.ANALYZED`` (same kind; only op rewrites).

    The pass:

    1. Builds a measurement-taint set using the same dataflow utilities
       as ``AnalyzePass`` (forward propagation from ``MeasureOperation``
       results through the dependency graph).
    2. Walks operations recursively (through ``HasNestedOps``).
    3. For each ``CompOp`` / ``CondOp`` / ``NotOp`` / ``BinOp`` whose
       result UUID is in the taint set, replaces it with an equivalent
       ``RuntimeClassicalExpr`` (same operands and result Value, only
       the op type and kind enum change).
    4. Non-tainted classical ops are left unchanged so the existing fold
       paths (compile-time fold in ``compile_time_if_lowering``,
       emit-time fold in ``evaluate_classical_predicate``) continue to
       handle them.

    The dependency graph and taint set are computed once, walked once,
    so the pass is O(N) where N is the number of operations.
    """

    @property
    def name(self) -> str:
        return "classical_lowering"

    def run(self, input: Block) -> Block:
        if input.kind != BlockKind.ANALYZED:
            # Defensive: the pass is designed to run after AnalyzePass.
            # Tolerate other kinds (e.g. tests passing AFFINE blocks
            # directly) by computing taint from scratch.
            pass

        dep_graph = build_dependency_graph(input.operations)
        measurement_uuids = find_measurement_results(input.operations)
        tainted = find_measurement_derived_values(dep_graph, measurement_uuids)

        new_ops = self._rewrite_operations(input.operations, tainted)
        return dataclasses.replace(input, operations=new_ops)

    # ------------------------------------------------------------------
    # Rewrite walker
    # ------------------------------------------------------------------

    def _rewrite_operations(
        self,
        operations: list[Operation],
        tainted: set[str],
    ) -> list[Operation]:
        """Walk operations recursively; rewrite tainted classical ops."""
        new_ops: list[Operation] = []
        for op in operations:
            if isinstance(op, (CompOp, CondOp, NotOp, BinOp)) and self._is_tainted(
                op, tainted
            ):
                new_ops.append(self._lower(op))
                continue
            if isinstance(op, HasNestedOps):
                rewritten_lists = [
                    self._rewrite_operations(body, tainted)
                    for body in op.nested_op_lists()
                ]
                new_ops.append(op.rebuild_nested(rewritten_lists))
                continue
            new_ops.append(op)
        return new_ops

    def _is_tainted(self, op: Operation, tainted: set[str]) -> bool:
        """True when any operand or result of ``op`` is measurement-derived.

        We check both operands and results: a result UUID landing in
        ``tainted`` confirms the op participates in the measurement
        dataflow chain. Checking operands is redundant (the dataflow
        propagation derived the result from those operands) but cheap and
        defends against subtle graph-construction edge cases.
        """
        for v in op.results:
            if isinstance(v, ValueBase) and v.uuid in tainted:
                return True
        for v in op.operands:
            if isinstance(v, ValueBase) and v.uuid in tainted:
                return True
        return False

    def _lower(self, op: Operation) -> RuntimeClassicalExpr:
        """Convert a compile-time classical op to ``RuntimeClassicalExpr``.

        Preserves operands and result Values verbatim — only the op type
        and the unified ``kind`` enum change. This means downstream
        consumers reading the result Value see the same UUID/name; only
        the op-type-based dispatch differs.
        """
        if isinstance(op, BinOp):
            kind = runtime_kind_from_binop(cast(Any, op.kind))
        elif isinstance(op, CompOp):
            kind = runtime_kind_from_compop(cast(Any, op.kind))
        elif isinstance(op, CondOp):
            kind = runtime_kind_from_condop(cast(Any, op.kind))
        elif isinstance(op, NotOp):
            from qamomile.circuit.ir.operation.arithmetic_operations import (
                RuntimeOpKind,
            )

            kind = RuntimeOpKind.NOT
        else:  # pragma: no cover — guarded by caller
            raise TypeError(f"Cannot lower {type(op).__name__} to RuntimeClassicalExpr")

        return RuntimeClassicalExpr(
            operands=list(op.operands),
            results=list(op.results),
            kind=kind,
        )
