"""Materialize compiler-owned definitions only when Qiskit requests synthesis."""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any

from qiskit.circuit import Gate

from qamomile.circuit import DeferredDefinition, LoopIndex, ParameterValue


class DeferredGate(Gate):
    """Retain an immutable native owner and copy-local SDK parameter bindings.

    Args:
        label (str): Display label; never used to infer operation semantics.
        definition (DeferredDefinition): Checked native definition owner.
        scope (dict[ParameterValue | LoopIndex, Any]): Current SDK bindings.
    """

    def __init__(
        self,
        label: str,
        definition: DeferredDefinition,
        scope: dict[ParameterValue | LoopIndex, Any],
    ) -> None:
        """Initialize an unsynthesized gate with explicit free parameters.

        Args:
            label (str): Display label.
            definition (DeferredDefinition): Native local definition.
            scope (dict[ParameterValue | LoopIndex, Any]): Scalar binding scope.
        """
        symbols = sorted(
            {symbol for value in scope.values() for symbol in value.parameters},
            key=lambda value: (value.name, str(value.uuid)),
        )
        self._owner = definition
        self._scope = dict(scope)
        self._symbols = tuple(symbols)
        # SDK synthesis dispatches on operation names. Source labels are display
        # metadata and must never select a standard gate's equivalence rules.
        super().__init__(
            f"qamomile_{uuid.uuid4().hex}",
            definition.owner.width,
            list(symbols),
            label=label,
        )

    def _define(self) -> None:
        """Emit the checked body and apply this copy's current parameter values."""
        from qiskit import QuantumCircuit

        from ._emission import _EmissionParameters, _emit_block

        circuit = QuantumCircuit(self.num_qubits, name=self.name)
        _emit_block(
            circuit, self._owner.materialize(), _EmissionParameters(self._scope)
        )
        bindings = dict(zip(self._symbols, self.params, strict=True))
        if bindings:
            circuit.assign_parameters(bindings, inplace=True, strict=False)
        self.definition = circuit

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> DeferredGate:
        """Share only the immutable native owner when Qiskit copies this gate.

        Args:
            memo (dict[int, Any] | None): Existing deep-copy identity map.

        Returns:
            DeferredGate: Independent mutable SDK gate with a shared native owner.
        """
        memo = {} if memo is None else memo
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        for name, value in self.__dict__.items():
            setattr(copied, name, value if name == "_owner" else deepcopy(value, memo))
        return copied
