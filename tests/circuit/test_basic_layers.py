"""Tests for basic rotation and entanglement layers."""

import sympy as sp
from qamomile.circuit.estimator import count_gates
from qamomile.circuit.algorithm.basic import (
    rx_layer,
    ry_layer,
    rz_layer,
    cz_entangling_layer,
)


def test_rx_layer_gate_count():
    """Test that rx_layer produces n RX gates."""
    counts = count_gates(rx_layer.block)
    q_dim0 = sp.Symbol("q_dim0")
    for n in [1, 3, 5]:
        single = counts.single_qubit.subs(q_dim0, n)
        two = counts.two_qubit.subs(q_dim0, n)
        assert single == n
        assert two == 0


def test_ry_layer_gate_count():
    """Test that ry_layer produces n RY gates."""
    counts = count_gates(ry_layer.block)
    q_dim0 = sp.Symbol("q_dim0")
    for n in [1, 3, 5]:
        single = counts.single_qubit.subs(q_dim0, n)
        two = counts.two_qubit.subs(q_dim0, n)
        assert single == n
        assert two == 0


def test_rz_layer_gate_count():
    """Test that rz_layer produces n RZ gates."""
    counts = count_gates(rz_layer.block)
    q_dim0 = sp.Symbol("q_dim0")
    for n in [1, 3, 5]:
        single = counts.single_qubit.subs(q_dim0, n)
        two = counts.two_qubit.subs(q_dim0, n)
        assert single == n
        assert two == 0


def test_cz_entangling_layer_gate_count():
    """Test that cz_entangling_layer produces n-1 CZ gates."""
    counts = count_gates(cz_entangling_layer.block)
    q_dim0 = sp.Symbol("q_dim0")
    for n in [2, 4, 6]:
        single = counts.single_qubit.subs(q_dim0, n)
        two = counts.two_qubit.subs(q_dim0, n)
        assert single == 0
        assert two == n - 1
