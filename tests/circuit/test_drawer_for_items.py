"""Regression tests for ForItems rendering of bound Dict parameters.

When a kernel parameter ``Dict`` is bound (via ``bindings={...}`` at
``transpile``/``draw`` time), the visualization analyzer must materialize
its entries from runtime metadata so that ``fold_loops=False`` can unroll
the corresponding ``for k, v in d.items():`` loop. The bug fixed here:
``_materialize_dict_entries`` previously checked truthiness of the
returned tuple, so a deliberately-empty bound mapping (``{}``) was
indistinguishable from a never-bound dict — both produced ``None`` and
the analyzer fell into the fold path, rendering an empty loop as a
folded box even with ``fold_loops=False``.
"""

import matplotlib

matplotlib.use("Agg")

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.qaoa import ising_cost
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VFoldedKind,
    VGate,
    VInlineBlock,
    VSkip,
    VUnfoldedKind,
    VUnfoldedSequence,
)


@qmc.qkernel
def kernel_with_dict(
    n: qmc.UInt,
    coeffs: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Apply Hadamard then a coeff-driven RZ to each qubit, then measure."""
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    for i, c in coeffs.items():
        q[i] = qmc.rz(q[i], angle=c)
    return qmc.measure(q)


def _walk_visual_nodes(nodes):
    """Yield all VisualNodes reachable from a list of root nodes."""
    for node in nodes:
        yield node
        if isinstance(node, VInlineBlock):
            yield from _walk_visual_nodes(node.children)
        elif isinstance(node, VUnfoldedSequence):
            for iteration in node.iterations:
                yield from _walk_visual_nodes(iteration)


def _build_visual_circuit(kernel, *, fold_loops, **bindings):
    """Trace ``kernel`` and run the visualization analyzer in isolation.

    Args:
        kernel: A ``QKernel`` to trace.
        fold_loops (bool): Forwarded to ``CircuitAnalyzer``.
        **bindings: Concrete arguments for the kernel (passed through
            ``_build_graph_for_visualization``).

    Returns:
        VisualCircuit: The analyzed visual IR ready for layout/render.
    """
    block = kernel._build_graph_for_visualization(**bindings)
    analyzer = CircuitAnalyzer(
        block,
        DEFAULT_STYLE,
        inline=False,
        fold_loops=fold_loops,
        expand_composite=False,
        inline_depth=None,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(block)
    return analyzer.build_visual_ir(block, qubit_map, qubit_names, num_qubits)


class TestEmptyBoundDictForItems:
    """An empty bound Dict must not render as a folded ForItems box."""

    def test_empty_dict_under_fold_loops_false_skips(self):
        """Empty bound Dict yields VSkip, not VFoldedBlock(FOR_ITEMS)."""
        vc = _build_visual_circuit(kernel_with_dict, fold_loops=False, n=3, coeffs={})
        nodes = list(_walk_visual_nodes(vc.children))

        folded_for_items = [
            n
            for n in nodes
            if isinstance(n, VFoldedBlock) and n.kind == VFoldedKind.FOR_ITEMS
        ]
        assert not folded_for_items, (
            "Empty bound Dict ForItems must not render as a folded box "
            f"under fold_loops=False; got: {folded_for_items}"
        )

        # The empty loop should appear as VSkip (zero-iteration unfold).
        assert any(isinstance(n, VSkip) for n in nodes), (
            "Empty bound Dict ForItems should collapse to a VSkip node."
        )

    def test_empty_dict_under_fold_loops_true_still_folds(self):
        """fold_loops=True keeps the canonical folded view, even when empty."""
        vc = _build_visual_circuit(kernel_with_dict, fold_loops=True, n=3, coeffs={})
        nodes = list(_walk_visual_nodes(vc.children))
        folded_for_items = [
            n
            for n in nodes
            if isinstance(n, VFoldedBlock) and n.kind == VFoldedKind.FOR_ITEMS
        ]
        assert len(folded_for_items) == 1, (
            "fold_loops=True should still render the ForItems loop as a "
            f"single folded box; got {len(folded_for_items)}: {folded_for_items}"
        )

    def test_non_empty_dict_unfolds_to_one_iteration_per_entry(self):
        """fold_loops=False with a non-empty bound Dict produces N iterations."""
        coeffs = {0: 0.1, 2: -0.3}
        vc = _build_visual_circuit(
            kernel_with_dict, fold_loops=False, n=3, coeffs=coeffs
        )
        unfolded_for_items = [
            n
            for n in _walk_visual_nodes(vc.children)
            if isinstance(n, VUnfoldedSequence) and n.kind == VUnfoldedKind.FOR_ITEMS
        ]
        assert len(unfolded_for_items) == 1, (
            f"Expected exactly one unfolded ForItems sequence; got "
            f"{len(unfolded_for_items)}"
        )
        assert len(unfolded_for_items[0].iterations) == len(coeffs), (
            "Unfolded ForItems should produce one iteration per Dict entry; "
            f"got {len(unfolded_for_items[0].iterations)} for "
            f"{len(coeffs)} entries"
        )


class TestForItemsValueVarSubstitutesIntoBinOpLabels:
    """ForItems value-var values must substitute into BinOp gate labels.

    When a parameter Dict is bound and ForItems unrolls each entry, the
    value variable (e.g. ``Jij``) holds a concrete numeric value for that
    iteration. Gate angles built as ``Jij * gamma`` must render with the
    concrete coefficient (``0.5 * gamma``) rather than leaking the loop
    variable's display name (``Jij * gamma``). This used to regress
    because ``_format_binop_operand`` only looked up operands by
    ``logical_id``, while ForItems unrolling stores entry values under
    string keys (``_loop_<var>``), and the same fallback that
    ``_evaluate_value`` uses was missing from the BinOp formatter.
    """

    def test_ising_cost_rzz_labels_show_concrete_coefficients(self):
        """One RZZ per quad entry, each label has the entry's coefficient."""
        coeffs = {(0, 1): 0.5, (1, 2): -0.3, (0, 2): 0.7}
        block = ising_cost._build_graph_for_visualization(q=3, quad=coeffs, linear={})
        analyzer = CircuitAnalyzer(
            block,
            DEFAULT_STYLE,
            inline=False,
            fold_loops=False,
            expand_composite=False,
            inline_depth=None,
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(block)
        vc = analyzer.build_visual_ir(block, qubit_map, qubit_names, num_qubits)
        labels = [
            n.label for n in _walk_visual_nodes(vc.children) if isinstance(n, VGate)
        ]

        assert len(labels) == len(coeffs), (
            f"Expected {len(coeffs)} RZZ gates; got {len(labels)}: {labels}"
        )

        for label in labels:
            assert "Jij" not in label, (
                f"Loop value-var name 'Jij' leaked into label {label!r}"
            )

        # Every concrete coefficient should appear in exactly one label.
        expected_substrings = {f"{v}*gamma" for v in coeffs.values()}
        seen = {sub for sub in expected_substrings if any(sub in lbl for lbl in labels)}
        assert seen == expected_substrings, (
            f"Concrete coefficients missing from labels; "
            f"expected substrings {sorted(expected_substrings)}, "
            f"saw {sorted(seen)} in {labels}"
        )
