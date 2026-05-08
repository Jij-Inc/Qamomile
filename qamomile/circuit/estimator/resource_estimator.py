"""Unified resource estimation interface.

This module combines all resource metrics (qubits, gates)
into a single interface for comprehensive circuit analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sympy as sp

from qamomile.circuit.estimator.gate_counter import GateCount, count_gates
from qamomile.circuit.estimator.qubits_counter import qubits_counter

if TYPE_CHECKING:
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.operation.operation import Operation


@dataclass
class ResourceEstimate:
    """Comprehensive resource estimate for a quantum circuit.

    All metrics are SymPy expressions that may contain symbols
    for parametric problem sizes.

    Attributes:
        qubits: Logical qubit count
        gates: Gate count breakdown (total, single_qubit, two_qubit, t_gates, clifford)
        parameters: Dictionary mapping symbol names to their SymPy symbols
    """

    qubits: sp.Expr
    gates: GateCount
    parameters: dict[str, sp.Symbol] = field(default_factory=dict)

    def substitute(self, **values: int | float) -> ResourceEstimate:
        """Substitute concrete values for parameters.

        Args:
            **values: Parameter name -> concrete value mappings

        Returns:
            New ResourceEstimate with substituted values

        Example:
            >>> est = estimate_resources(circuit)
            >>> concrete = est.substitute(n=100, p=3)
            >>> print(concrete.qubits)  # 100 (instead of 'n')
        """
        # Find matching symbols from the parameters dict
        subs_dict = {}
        for key, val in values.items():
            # Look for symbol in parameters dict
            if key in self.parameters:
                subs_dict[self.parameters[key]] = val
            else:
                # Try creating a symbol (for cases where parameters weren't tracked)
                subs_dict[sp.Symbol(key, integer=True, positive=True)] = val

        def _subs_eval(expr: sp.Expr) -> sp.Expr:
            if isinstance(expr, (int, float)):  # type: ignore[unreachable]
                return sp.Integer(expr)  # type: ignore[unreachable]
            return expr.subs(subs_dict).doit()

        return ResourceEstimate(
            qubits=_subs_eval(self.qubits),
            gates=GateCount(
                total=_subs_eval(self.gates.total),
                single_qubit=_subs_eval(self.gates.single_qubit),
                two_qubit=_subs_eval(self.gates.two_qubit),
                multi_qubit=_subs_eval(self.gates.multi_qubit),
                t_gates=_subs_eval(self.gates.t_gates),
                clifford_gates=_subs_eval(self.gates.clifford_gates),
                rotation_gates=_subs_eval(self.gates.rotation_gates),
                oracle_calls={
                    name: _subs_eval(val)
                    for name, val in self.gates.oracle_calls.items()
                },
                oracle_queries={
                    name: _subs_eval(val)
                    for name, val in self.gates.oracle_queries.items()
                },
            ),
            parameters=self.parameters,
        )

    def simplify(self) -> ResourceEstimate:
        """Simplify all SymPy expressions.

        Returns:
            New ResourceEstimate with simplified expressions
        """
        return ResourceEstimate(
            qubits=sp.simplify(self.qubits),
            gates=self.gates.simplify(),
            parameters=self.parameters,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization.

        Returns:
            Dictionary with all metrics as strings (for JSON/YAML export)

        Example:
            >>> est = estimate_resources(circuit)
            >>> data = est.to_dict()
            >>> import json
            >>> print(json.dumps(data, indent=2))
        """
        return {
            "qubits": str(self.qubits),
            "gates": {
                "total": str(self.gates.total),
                "single_qubit": str(self.gates.single_qubit),
                "two_qubit": str(self.gates.two_qubit),
                "multi_qubit": str(self.gates.multi_qubit),
                "t_gates": str(self.gates.t_gates),
                "clifford_gates": str(self.gates.clifford_gates),
                "rotation_gates": str(self.gates.rotation_gates),
                "oracle_calls": {
                    name: str(val) for name, val in self.gates.oracle_calls.items()
                },
                "oracle_queries": {
                    name: str(val) for name, val in self.gates.oracle_queries.items()
                },
            },
            "parameters": {k: str(v) for k, v in self.parameters.items()},
        }

    def __str__(self) -> str:
        """Pretty-print the resource estimate."""
        lines = [
            "Resource Estimate:",
            f"  Qubits: {self.qubits}",
            "  Gates:",
            f"    Total: {self.gates.total}",
            f"    Single-qubit: {self.gates.single_qubit}",
            f"    Two-qubit: {self.gates.two_qubit}",
            f"    Multi-qubit: {self.gates.multi_qubit}",
            f"    T gates: {self.gates.t_gates}",
            f"    Clifford gates: {self.gates.clifford_gates}",
            f"    Rotation gates: {self.gates.rotation_gates}",
        ]
        if self.gates.oracle_calls:
            lines.append("  Oracle Calls:")
            for name, count in self.gates.oracle_calls.items():
                lines.append(f"    {name}: {count}")
        if self.gates.oracle_queries:
            lines.append("  Oracle Queries:")
            for name, count in self.gates.oracle_queries.items():
                lines.append(f"    {name}: {count}")
        if self.parameters:
            lines.append("  Parameters:")
            for name, symbol in self.parameters.items():
                lines.append(f"    {name}: {symbol}")
        return "\n".join(lines)


def estimate_resources(
    block: Block | list[Operation],
    *,
    bindings: dict[str, Any] | None = None,
) -> ResourceEstimate:
    """Estimate all resources for a quantum circuit.

    This is the main entry point for comprehensive resource estimation.
    Combines qubit counting and gate counting.

    Args:
        block: Block or list of Operations to analyze
        bindings: Optional concrete parameter bindings (scalars and dicts).

    Returns:
        ResourceEstimate with qubits, gates, and parameters

    Example:
        >>> import qamomile.circuit as qm
        >>> from qamomile.circuit.estimator import estimate_resources
        >>>
        >>> @qm.qkernel
        >>> def bell_state() -> qm.Vector[qm.Qubit]:
        ...     q = qm.qubit_array(2)
        ...     q[0] = qm.h(q[0])
        ...     q[0], q[1] = qm.cx(q[0], q[1])
        ...     return q
        >>>
        >>> est = estimate_resources(bell_state.block)
        >>> print(est.qubits)  # 2
        >>> print(est.gates.total)  # 2
        >>> print(est.gates.two_qubit)  # 1

    Example with parametric size:
        >>> @qm.qkernel
        >>> def ghz_state(n: qm.UInt) -> qm.Vector[qm.Qubit]:
        ...     q = qm.qubit_array(n)
        ...     q[0] = qm.h(q[0])
        ...     for i in qm.range(n - 1):
        ...         q[i], q[i+1] = qm.cx(q[i], q[i+1])
        ...     return q
        >>>
        >>> est = estimate_resources(ghz_state.block)
        >>> print(est.qubits)  # n
        >>> print(est.gates.total)  # n
        >>> print(est.gates.two_qubit)  # n - 1
        >>>
        >>> # Substitute concrete value
        >>> concrete = est.substitute(n=100)
        >>> print(concrete.qubits)  # 100
        >>> print(concrete.gates.total)  # 100
    """
    # Count qubits
    qubit_count = qubits_counter(block)

    # Count gates
    gate_count = count_gates(block)

    # Substitute dict cardinality and scalar symbols if bindings provided
    if bindings is not None:
        all_subs: dict[sp.Symbol, int] = {}
        for key, val in bindings.items():
            if isinstance(val, dict):
                all_subs[sp.Symbol(f"|{key}|", integer=True, positive=True)] = len(val)
            elif isinstance(val, (int, float)):
                all_subs[sp.Symbol(key, integer=True, positive=True)] = int(val)
        if all_subs:
            gate_count = GateCount(
                total=gate_count.total.subs(all_subs),
                single_qubit=gate_count.single_qubit.subs(all_subs),
                two_qubit=gate_count.two_qubit.subs(all_subs),
                multi_qubit=gate_count.multi_qubit.subs(all_subs),
                t_gates=gate_count.t_gates.subs(all_subs),
                clifford_gates=gate_count.clifford_gates.subs(all_subs),
                rotation_gates=gate_count.rotation_gates.subs(all_subs),
                oracle_calls={
                    name: count.subs(all_subs)
                    for name, count in gate_count.oracle_calls.items()
                },
                oracle_queries={
                    name: count.subs(all_subs)
                    for name, count in gate_count.oracle_queries.items()
                },
            )
            qubit_count = qubit_count.subs(all_subs)

    # Collect all symbols (parameters)
    all_symbols: set[sp.Symbol] = set()
    for expr in [
        qubit_count,
        gate_count.total,
        gate_count.single_qubit,
        gate_count.two_qubit,
        gate_count.multi_qubit,
        gate_count.t_gates,
        gate_count.clifford_gates,
        gate_count.rotation_gates,
    ]:
        all_symbols.update(expr.free_symbols)  # type: ignore[arg-type]
    for oracle_expr in gate_count.oracle_calls.values():
        all_symbols.update(oracle_expr.free_symbols)  # type: ignore[arg-type]
    for oracle_expr in gate_count.oracle_queries.values():
        all_symbols.update(oracle_expr.free_symbols)  # type: ignore[arg-type]

    parameters = {str(sym): sym for sym in sorted(all_symbols, key=str)}

    return ResourceEstimate(
        qubits=qubit_count,
        gates=gate_count,
        parameters=parameters,
    )
