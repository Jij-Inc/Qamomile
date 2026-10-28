"""Declare the internal service contract for resource interpretation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import sympy as sp

from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._opaque import (
    OpaqueCostContext,
    _OpaqueInvocationTransform,
)
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.select import SelectOperation


class _InterpreterContract(ABC):
    """Define services supplied across the linear interpreter layers."""

    @abstractmethod
    def _strategy_for(self, operation: InvokeOperation) -> str | None:
        """Return the selected implementation strategy.

        Args:
            operation (InvokeOperation): Invocation to inspect.

        Returns:
            str | None: Selected strategy, or ``None``.
        """
        ...

    @abstractmethod
    def _apply_condition_values(
        self,
        expression: sp.Expr,
        *,
        record_usage: bool = True,
    ) -> sp.Expr:
        """Apply known scalar inputs to a symbolic expression.

        Args:
            expression (sp.Expr): Expression to specialize.
            record_usage (bool): Whether to record consumed inputs. Defaults
                to ``True``.

        Returns:
            sp.Expr: Specialized expression.
        """
        ...

    @staticmethod
    @abstractmethod
    def _concrete_scalar(expression: sp.Expr) -> int | None:
        """Return a concrete integer when one is known.

        Args:
            expression (sp.Expr): Expression to inspect.

        Returns:
            int | None: Concrete integer, or ``None``.
        """
        ...

    @abstractmethod
    def _opaque_cost_context(
        self,
        operation: InvokeOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr,
        strategy: str | None,
    ) -> tuple[OpaqueCostContext, _OpaqueInvocationTransform]:
        """Build opaque definition and invocation contexts.

        Args:
            operation (InvokeOperation): Invocation to model.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr): Inherited control count.
            strategy (str | None): Selected implementation strategy.

        Returns:
            tuple[OpaqueCostContext, _OpaqueInvocationTransform]: Definition
            context and call-site transform.
        """
        ...

    @abstractmethod
    def _resolve_opaque_definition_cost(
        self,
        operation: InvokeOperation,
        cost: Any,
        context: OpaqueCostContext,
    ) -> ResourceEstimate:
        """Resolve one opaque definition-level cost.

        Args:
            operation (InvokeOperation): Invocation being priced.
            cost (Any): Fixed estimate or cost callback.
            context (OpaqueCostContext): Definition-level context.

        Returns:
            ResourceEstimate: Resolved definition cost.
        """
        ...

    @abstractmethod
    def eval_operations(
        self,
        operations: list[Operation],
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
        initial_allocations: Mapping[str, ResourceExpr] | None = None,
        allow_control_batching: bool = True,
    ) -> ResourceEstimate:
        """Evaluate one sequential operation region.

        Args:
            operations (list[Operation]): Operations in program order.
            resolver (ExprResolver): Resolver for the region scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.
            initial_allocations (Mapping[str, ResourceExpr] | None): Live
                allocations at entry. Defaults to ``None``.
            allow_control_batching (bool): Whether to consider a shared control
                ladder. Defaults to ``True``.

        Returns:
            ResourceEstimate: Sequential region estimate.
        """
        ...

    @abstractmethod
    def eval_controlled_u(
        self,
        operation: ControlledUOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a controlled callable.

        Args:
            operation (ControlledUOperation): Controlled operation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Controlled-call estimate.
        """
        ...

    @abstractmethod
    def eval_expval(
        self,
        operation: ExpvalOp,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate an expectation-value operation.

        Args:
            operation (ExpvalOp): Expectation-value operation.
            resolver (ExprResolver): Resolver for the quantum operand.

        Returns:
            ResourceEstimate: Expectation-value estimate.
        """
        ...

    @abstractmethod
    def eval_for(
        self,
        operation: ForOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a range loop.

        Args:
            operation (ForOperation): Loop operation.
            resolver (ExprResolver): Resolver for the enclosing scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Loop estimate.
        """
        ...

    @abstractmethod
    def eval_for_items(
        self,
        operation: ForItemsOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate an items loop.

        Args:
            operation (ForItemsOperation): Items-loop operation.
            resolver (ExprResolver): Resolver for the enclosing scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Items-loop estimate.
        """
        ...

    @abstractmethod
    def eval_gate(
        self,
        operation: GateOperation,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a primitive gate.

        Args:
            operation (GateOperation): Gate operation.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Gate estimate.
        """
        ...

    @abstractmethod
    def eval_global_phase(
        self,
        operation: GlobalPhaseOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a global phase.

        Args:
            operation (GlobalPhaseOperation): Global-phase operation.
            resolver (ExprResolver): Resolver for the phase value.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Global-phase estimate.
        """
        ...

    @abstractmethod
    def eval_if(
        self,
        operation: IfOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a conditional branch.

        Args:
            operation (IfOperation): Conditional operation.
            resolver (ExprResolver): Resolver for the enclosing scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Branch estimate.
        """
        ...

    @abstractmethod
    def eval_inverse_block(
        self,
        operation: InverseBlockOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate an inverse callable body.

        Args:
            operation (InverseBlockOperation): Inverse operation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Inverse-call estimate.
        """
        ...

    @abstractmethod
    def eval_invoke(
        self,
        operation: InvokeOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a callable invocation.

        Args:
            operation (InvokeOperation): Invocation operation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Invocation estimate.
        """
        ...

    @abstractmethod
    def eval_measure(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate a measurement operation.

        Args:
            operation (Operation): Measurement-like operation.
            resolver (ExprResolver): Resolver for vector widths.

        Returns:
            ResourceEstimate: Measurement estimate.
        """
        ...

    @abstractmethod
    def eval_pauli_evolve(
        self,
        operation: PauliEvolveOp,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a Pauli evolution.

        Args:
            operation (PauliEvolveOp): Pauli-evolution operation.
            resolver (ExprResolver): Resolver for its operands.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Pauli-evolution estimate.
        """
        ...

    @abstractmethod
    def eval_project(self, operation: ProjectOperation) -> ResourceEstimate:
        """Evaluate a projective measurement.

        Args:
            operation (ProjectOperation): Projection operation.

        Returns:
            ResourceEstimate: Projection estimate.
        """
        ...

    @abstractmethod
    def eval_qinit(
        self,
        operation: QInitOperation,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate a quantum allocation.

        Args:
            operation (QInitOperation): Allocation operation.
            resolver (ExprResolver): Resolver for symbolic dimensions.

        Returns:
            ResourceEstimate: Allocation estimate.
        """
        ...

    @abstractmethod
    def eval_reset(self, operation: ResetOperation) -> ResourceEstimate:
        """Evaluate a reset operation.

        Args:
            operation (ResetOperation): Reset operation.

        Returns:
            ResourceEstimate: Reset estimate.
        """
        ...

    @abstractmethod
    def eval_select(
        self,
        operation: SelectOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a SELECT operation.

        Args:
            operation (SelectOperation): SELECT operation.
            resolver (ExprResolver): Resolver for the call site.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: SELECT estimate.
        """
        ...

    @abstractmethod
    def eval_while(
        self,
        operation: WhileOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a while loop.

        Args:
            operation (WhileOperation): While-loop operation.
            resolver (ExprResolver): Resolver for the enclosing scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: While-loop estimate.
        """
        ...
