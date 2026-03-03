"""GateCount dataclass for gate counting results.

Extracted to its own module to break the circular dependency between
``_catalog.py`` (which classifies gates into GateCount) and
``gate_counter.py`` (which imports classification functions from _catalog).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ._utils import _strip_nonneg_max


@dataclass
class GateCount:
    """Gate count breakdown for a quantum circuit.

    All counts are SymPy expressions that may contain symbols
    for parametric problem sizes.

    Attributes:
        total: Total number of gates
        single_qubit: Number of single-qubit gates
        two_qubit: Number of two-qubit gates
        t_gates: Number of T gates (critical for fault tolerance)
        clifford_gates: Number of Clifford gates (H, S, CNOT, CZ, etc.)
    """

    total: sp.Expr
    single_qubit: sp.Expr
    two_qubit: sp.Expr
    multi_qubit: sp.Expr
    t_gates: sp.Expr
    clifford_gates: sp.Expr
    rotation_gates: sp.Expr
    oracle_calls: dict[str, sp.Expr] = field(default_factory=dict)
    oracle_queries: dict[str, sp.Expr] = field(default_factory=dict)

    def __add__(self, other: GateCount) -> GateCount:
        """Add two gate counts together."""
        merged_oracle = dict(self.oracle_calls)
        for name, count in other.oracle_calls.items():
            if name in merged_oracle:
                merged_oracle[name] = merged_oracle[name] + count
            else:
                merged_oracle[name] = count
        merged_queries = dict(self.oracle_queries)
        for name, count in other.oracle_queries.items():
            if name in merged_queries:
                merged_queries[name] = merged_queries[name] + count
            else:
                merged_queries[name] = count
        return GateCount(
            total=self.total + other.total,
            single_qubit=self.single_qubit + other.single_qubit,
            two_qubit=self.two_qubit + other.two_qubit,
            multi_qubit=self.multi_qubit + other.multi_qubit,
            t_gates=self.t_gates + other.t_gates,
            clifford_gates=self.clifford_gates + other.clifford_gates,
            rotation_gates=self.rotation_gates + other.rotation_gates,
            oracle_calls=merged_oracle,
            oracle_queries=merged_queries,
        )

    def __mul__(self, factor: sp.Expr | int) -> GateCount:
        """Multiply gate counts by a factor."""
        factor_expr = sp.Integer(factor) if isinstance(factor, int) else factor
        return GateCount(
            total=self.total * factor_expr,
            single_qubit=self.single_qubit * factor_expr,
            two_qubit=self.two_qubit * factor_expr,
            multi_qubit=self.multi_qubit * factor_expr,
            t_gates=self.t_gates * factor_expr,
            clifford_gates=self.clifford_gates * factor_expr,
            rotation_gates=self.rotation_gates * factor_expr,
            oracle_calls={
                name: count * factor_expr for name, count in self.oracle_calls.items()
            },
            oracle_queries={
                name: count * factor_expr for name, count in self.oracle_queries.items()
            },
        )

    __rmul__ = __mul__

    def max(self, other: GateCount) -> GateCount:
        """Element-wise maximum of two gate counts."""

        def _merge_max(
            a: dict[str, sp.Expr], b: dict[str, sp.Expr]
        ) -> dict[str, sp.Expr]:
            merged: dict[str, sp.Expr] = {}
            for key in set(a.keys()) | set(b.keys()):
                if key in a and key in b:
                    merged[key] = sp.Max(a[key], b[key])
                elif key in a:
                    merged[key] = a[key]
                else:
                    merged[key] = b[key]
            return merged

        return GateCount(
            total=sp.Max(self.total, other.total),
            single_qubit=sp.Max(self.single_qubit, other.single_qubit),
            two_qubit=sp.Max(self.two_qubit, other.two_qubit),
            multi_qubit=sp.Max(self.multi_qubit, other.multi_qubit),
            t_gates=sp.Max(self.t_gates, other.t_gates),
            clifford_gates=sp.Max(self.clifford_gates, other.clifford_gates),
            rotation_gates=sp.Max(self.rotation_gates, other.rotation_gates),
            oracle_calls=_merge_max(self.oracle_calls, other.oracle_calls),
            oracle_queries=_merge_max(self.oracle_queries, other.oracle_queries),
        )

    def simplify(self) -> GateCount:
        """Simplify all SymPy expressions."""
        return GateCount(
            total=_strip_nonneg_max(sp.simplify(self.total)),
            single_qubit=_strip_nonneg_max(sp.simplify(self.single_qubit)),
            two_qubit=_strip_nonneg_max(sp.simplify(self.two_qubit)),
            multi_qubit=_strip_nonneg_max(sp.simplify(self.multi_qubit)),
            t_gates=_strip_nonneg_max(sp.simplify(self.t_gates)),
            clifford_gates=_strip_nonneg_max(sp.simplify(self.clifford_gates)),
            rotation_gates=_strip_nonneg_max(sp.simplify(self.rotation_gates)),
            oracle_calls={
                name: _strip_nonneg_max(sp.simplify(count))
                for name, count in self.oracle_calls.items()
            },
            oracle_queries={
                name: _strip_nonneg_max(sp.simplify(count))
                for name, count in self.oracle_queries.items()
            },
        )

    @staticmethod
    def zero() -> GateCount:
        """Return a zero gate count."""
        return GateCount(
            total=sp.Integer(0),
            single_qubit=sp.Integer(0),
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
            rotation_gates=sp.Integer(0),
        )
