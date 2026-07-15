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

    def test_empty_dict_under_fold_loops_true_renders_nothing(self):
        """fold_loops=True renders no ForItems box for an empty bound Dict.

        The zero-trip trace guard (``should_trace_items_loop``, mirroring
        ``qmc.range``'s guard) skips tracing the body of a
        compile-time-known EMPTY dict entirely, so no ForItemsOperation
        exists in the IR to draw — the same behavior as ``range(0)``.
        """
        vc = _build_visual_circuit(kernel_with_dict, fold_loops=True, n=3, coeffs={})
        nodes = list(_walk_visual_nodes(vc.children))
        folded_for_items = [
            n
            for n in nodes
            if isinstance(n, VFoldedBlock) and n.kind == VFoldedKind.FOR_ITEMS
        ]
        assert not folded_for_items, (
            "An empty bound Dict is skipped at trace time; no ForItems "
            f"box should exist to fold; got: {folded_for_items}"
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


def test_vector_key_elements_rebind_for_each_unfolded_entry() -> None:
    """Each bound Vector key supplies its own concrete element values."""

    @qmc.qkernel
    def circuit(
        data: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    ) -> qmc.Bit:
        """Scale each entry value by the second element of its Vector key."""
        q = qmc.qubit("q")
        for key, value in qmc.items(data):
            q = qmc.rx(q, value * key[1])
        return qmc.measure(q)

    vc = _build_visual_circuit(
        circuit,
        fold_loops=False,
        data={(2, 3): 0.5, (4, 5): 0.5},
    )
    labels = [
        node.label
        for node in _walk_visual_nodes(vc.children)
        if isinstance(node, VGate)
    ]

    assert labels == ["$R_x$(1.50)", "$R_x$(2.50)", "M"]


class TestForLoopVariableSubstitutesIntoArrayIndices:
    """ForOperation loop indices into parameter arrays must substitute.

    When a ``for layer in qmc.range(p):`` loop is unrolled at
    visualization time, gate-angle expressions like
    ``gammas[layer]`` must render as ``gammas[0]`` and ``gammas[1]``
    in the unrolled iterations — not as the literal symbolic
    ``gammas[layer]`` repeated for every iteration. This used to
    regress because the array-element access on a parameter array
    reports ``is_parameter()`` True with ``parameter_name() ==
    "gammas[layer]"`` (the pre-formatted string with the loop
    variable baked in), so ``_format_binop_operand`` returned that
    verbatim from its named-parameter branch before the
    array-element branch ever got a chance to recurse on the
    indices.
    """

    def test_unrolled_for_substitutes_loop_index_into_parameter_array(self):
        """Each unrolled `gammas[layer]` access shows the iteration index.

        Routes the kernel through the public `Transpiler.to_block +
        inline` pipeline so the analyzer sees the same fully-flattened
        Block the docs notebooks render. The kernel below is a
        miniature stand-in for the QAOA layer loop: it iterates
        `for layer in qmc.range(p):` and pulls `gammas[layer]` /
        `betas[layer]` out of two parameter arrays per iteration.
        """
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def two_param_loops(
            p: qmc.UInt,
            n: qmc.UInt,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            for layer in qmc.range(p):
                for i in qmc.range(n):
                    q[i] = qmc.rz(q[i], angle=gammas[layer])
                for i in qmc.range(n):
                    q[i] = qmc.rx(q[i], angle=betas[layer])
            return qmc.measure(q)

        t = QiskitTranspiler()
        block = t.to_block(
            two_param_loops,
            bindings={"p": 2, "n": 3},
            parameters=["gammas", "betas"],
        )
        block = t.inline(block)

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

        # No label should contain the bare loop variable name.
        for label in labels:
            assert "[layer]" not in label, (
                f"Loop variable 'layer' leaked into label {label!r}; expected "
                f"the unrolled iteration to substitute it as a concrete index"
            )

        # Both iteration indices should appear for both parameter arrays.
        # ``_format_symbolic_param`` TeX-converts ``gammas`` →
        # ``${\gamma}s$`` (and ``betas`` → ``${\beta}s$``) because the
        # base ``gamma`` / ``beta`` is in the Greek symbol map; the
        # ``[0]`` / ``[1]`` index sits inside the same ``$...$`` math
        # span, so we assert against the TeX form rather than the bare
        # ``gammas[0]`` string.
        joined = " ".join(labels)
        for needle in (
            r"{\gamma}s[0]",
            r"{\gamma}s[1]",
            r"{\beta}s[0]",
            r"{\beta}s[1]",
        ):
            assert needle in joined, (
                f"Expected unrolled iterations to include `{needle}` "
                f"somewhere in {labels}"
            )
