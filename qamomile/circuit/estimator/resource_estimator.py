"""Unified resource estimation interface.

This module combines all resource metrics (qubits, gates, depth)
into a single interface for comprehensive circuit analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sympy as sp

from qamomile.circuit.estimator.depth_estimator import CircuitDepth, estimate_depth
from qamomile.circuit.estimator.gate_counter import GateCount, count_gates
from qamomile.circuit.estimator.qubits_counter import qubits_counter

if TYPE_CHECKING:
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue
    from qamomile.circuit.ir.operation.operation import Operation


@dataclass
class ResourceEstimate:
    """Comprehensive resource estimate for a quantum circuit.

    All metrics are SymPy expressions that may contain symbols
    for parametric problem sizes.

    Attributes:
        qubits: Logical qubit count
        gates: Gate count breakdown (total, single_qubit, two_qubit, t_gates, clifford)
        depth: Circuit depth breakdown (total_depth, t_depth, two_qubit_depth)
        parameters: Dictionary mapping symbol names to their SymPy symbols
    """

    qubits: sp.Expr
    gates: GateCount
    depth: CircuitDepth
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
            ),
            depth=CircuitDepth(
                total_depth=_subs_eval(self.depth.total_depth),
                t_depth=_subs_eval(self.depth.t_depth),
                two_qubit_depth=_subs_eval(self.depth.two_qubit_depth),
                multi_qubit_depth=_subs_eval(self.depth.multi_qubit_depth),
                rotation_depth=_subs_eval(self.depth.rotation_depth),
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
            depth=self.depth.simplify(),
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
                    name: str(val)
                    for name, val in self.gates.oracle_calls.items()
                },
            },
            "depth": {
                "total_depth": str(self.depth.total_depth),
                "t_depth": str(self.depth.t_depth),
                "two_qubit_depth": str(self.depth.two_qubit_depth),
                "multi_qubit_depth": str(self.depth.multi_qubit_depth),
                "rotation_depth": str(self.depth.rotation_depth),
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
        lines.extend([
            "  Depth:",
            f"    Total: {self.depth.total_depth}",
            f"    T-depth: {self.depth.t_depth}",
            f"    Two-qubit depth: {self.depth.two_qubit_depth}",
            f"    Multi-qubit depth: {self.depth.multi_qubit_depth}",
            f"    Rotation depth: {self.depth.rotation_depth}",
        ])
        if self.parameters:
            lines.append("  Parameters:")
            for name, symbol in self.parameters.items():
                lines.append(f"    {name}: {symbol}")
        return "\n".join(lines)


def estimate_resources(
    block: BlockValue | Block | list[Operation],
) -> ResourceEstimate:
    """Estimate all resources for a quantum circuit.

    This is the main entry point for comprehensive resource estimation.
    Combines qubit counting, gate counting, and depth estimation.

    Args:
        block: BlockValue, Block, or list of Operations to analyze

    Returns:
        ResourceEstimate with qubits, gates, depth, and parameters

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
        >>> print(est.depth.total_depth)  # 2

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

    # Estimate depth
    circuit_depth = estimate_depth(block)

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
        circuit_depth.total_depth,
        circuit_depth.t_depth,
        circuit_depth.two_qubit_depth,
        circuit_depth.multi_qubit_depth,
        circuit_depth.rotation_depth,
    ]:
        all_symbols.update(expr.free_symbols)
    for oracle_expr in gate_count.oracle_calls.values():
        all_symbols.update(oracle_expr.free_symbols)

    parameters = {str(sym): sym for sym in sorted(all_symbols, key=str)}

    return ResourceEstimate(
        qubits=qubit_count,
        gates=gate_count,
        depth=circuit_depth,
        parameters=parameters,
    )
