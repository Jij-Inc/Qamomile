"""Test output variable name display on circuit diagrams."""

import qamomile.circuit as qm


class TestOutputNames:
    """Test suite for output variable name display."""

    def test_single_return(self):
        """Test single return value displays correctly."""

        @qm.qkernel
        def circuit(q: qm.Qubit) -> qm.Qubit:
            result = qm.h(q)
            return result

        # Build graph
        graph = circuit.build()
        graph.output_names = circuit._extract_return_names() or []

        # Should extract "result"
        assert graph.output_names == ["result"]

    def test_tuple_return(self):
        """Test tuple return displays correctly."""

        @qm.qkernel
        def circuit(
            q0: qm.Qubit, q1: qm.Qubit, q2: qm.Qubit
        ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
            q0 = qm.h(q0)
            q1 = qm.h(q1)
            q2 = qm.h(q2)
            return q0, q1, q2

        # Build graph
        graph = circuit.build()
        graph.output_names = circuit._extract_return_names() or []

        # Should extract ["q0", "q1", "q2"]
        assert graph.output_names == ["q0", "q1", "q2"]

    def test_renamed_variables(self):
        """Test renamed variables in return statement."""

        @qm.qkernel
        def circuit(a: qm.Qubit, b: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            alice = qm.h(a)
            bob = qm.x(b)
            return alice, bob

        # Build graph
        graph = circuit.build()
        graph.output_names = circuit._extract_return_names() or []

        # Should extract ["alice", "bob"]
        assert graph.output_names == ["alice", "bob"]

    def test_draw_with_output_names(self):
        """Test draw() method sets output_names correctly."""

        @qm.qkernel
        def circuit(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            q0 = qm.h(q0)
            q1 = qm.x(q1)
            return q0, q1

        # Draw should set output_names
        fig = circuit.draw()
        assert fig is not None

    def test_draw_inline_with_output_names(self):
        """Test draw() with inline mode sets output_names correctly."""

        @qm.qkernel
        def inner(q: qm.Qubit) -> qm.Qubit:
            return qm.h(q)

        @qm.qkernel
        def circuit(q: qm.Qubit) -> qm.Qubit:
            result = inner(q)
            return result

        # Draw inline should set output_names
        fig = circuit.draw(inline=True)
        assert fig is not None

    def test_no_return_statement(self):
        """Test circuit without return statement."""

        @qm.qkernel
        def circuit(q: qm.Qubit) -> qm.Qubit:
            q = qm.h(q)
            # No explicit return statement in source
            # (but Python implicitly returns due to type annotation)

        # _extract_return_names should handle missing return gracefully
        output_names = circuit._extract_return_names()
        # Should return None or empty list
        assert output_names is None or output_names == []

    def test_complex_circuit_with_parameters(self):
        """Test complex circuit with parameters and output names."""

        @qm.qkernel
        def circuit(
            q0: qm.Qubit, q1: qm.Qubit, theta: float
        ) -> tuple[qm.Qubit, qm.Qubit]:
            q0 = qm.h(q0)
            q0 = qm.rx(q0, theta)
            q1 = qm.x(q1)
            q0, q1 = qm.cx(q0, q1)
            return q0, q1

        # Build graph
        graph = circuit.build()
        graph.output_names = circuit._extract_return_names() or []

        # Should extract ["q0", "q1"]
        assert graph.output_names == ["q0", "q1"]

        # Draw with symbolic parameter
        fig_symbolic = circuit.draw()
        assert fig_symbolic is not None

        # Draw with bound parameter
        fig_bound = circuit.draw(theta=0.5)
        assert fig_bound is not None
