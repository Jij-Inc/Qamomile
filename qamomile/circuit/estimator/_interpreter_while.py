"""Interpret while loops for resource estimation."""

from __future__ import annotations

import dataclasses

import sympy as sp

from qamomile.circuit.estimator._call_liveness import (
    _captured_quantum_allocations,
    _with_operation_output_summary,
)
from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._interpreter_dataflow import (
    _require_uncontrolled_operation,
)
from qamomile.circuit.estimator._interpreter_for import _ForInterpreter
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
)
from qamomile.circuit.estimator._scopes import (
    build_while_scope,
)
from qamomile.circuit.ir.operation.control_flow import (
    WhileOperation,
)


class _WhileInterpreter(_ForInterpreter):
    """Add runtime while-loop resource evaluation."""

    def eval_while(
        self,
        operation: WhileOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a while loop using its declared or symbolic trip count.

        Args:
            operation (WhileOperation): While-loop operation.
            resolver (ExprResolver): Resolver for the outer scope.
            controls (ResourceExpr | int): Surrounding controls. Defaults to
                zero.

        Returns:
            ResourceEstimate: Repeated loop-body estimate.

        Raises:
            ValueError: If the loop is nested under coherent quantum control.
            NotImplementedError: If the loop carries a rebound value whose
                recurrence the symbolic while-loop model cannot represent.
            AssertionError: If deterministic trip-count name reservation is
                unexpectedly missing for the operation.
        """
        _require_uncontrolled_operation(operation, controls)
        if operation.loop_carried_rebinds or operation.region_args:
            names = sorted(
                {rebind.var_name for rebind in operation.loop_carried_rebinds}
                | {arg.var_name for arg in operation.region_args}
            )
            variables = ", ".join(names) or "(unnamed)"
            raise NotImplementedError(
                "Resource estimation does not support loop-carried values in "
                f"WhileOperation ({variables}). A symbolic trip count alone "
                "cannot determine the carried recurrence."
            )
        self._reserve_while_trip_count_names((operation,), resolver)
        operation_key = (resolver.structural_scope, id(operation))
        cached_name = self._run_state.while_trip_count_names.get(operation_key)
        if cached_name is None or cached_name[0] is not operation:
            raise AssertionError("WhileOperation trip-count name was not reserved.")
        trip_count_name = cached_name[1]
        child, trip_count = build_while_scope(
            operation,
            resolver,
            trip_count_name=trip_count_name,
        )
        with self._guarded_constraint_scope(sp.Gt(trip_count, _ZERO)):
            inner = self.eval_operations(
                operation.operations,
                child,
                controls=controls,
                initial_allocations=_captured_quantum_allocations(
                    operation.operations,
                    child,
                    self._run_state.allocation_owners_by_uuid,
                ),
            )
            inner = _with_constraints(
                inner,
                *self._loop_iteration_array_constraints(
                    operation,
                    child,
                ),
            )
        estimate = _with_operation_output_summary(
            inner.repeat(trip_count),
            operation,
            resolver,
            active_when=sp.Gt(trip_count, _ZERO),
            body_output_sizes=inner._output_sizes,
            allocation_owners_by_uuid=self._run_state.allocation_owners_by_uuid,
        )
        assumption = ResourceAssumption(
            "runtime while resources use the declared trip count and a "
            "wire-local dependency envelope",
            source="while",
        )
        estimate = estimate._with_metadata(
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
        )
        array_taint = self._guard_array_updates(
            operation.operations,
            resolver,
            estimate,
            active_when=sp.Gt(trip_count, _ZERO),
        )
        estimate = dataclasses.replace(
            estimate,
            _measurement_taint_conditions={
                **estimate._measurement_taint_conditions,
                **array_taint,
            },
        )
        return _with_constraints(
            estimate,
            *self._loop_initial_array_constraints(
                operation,
                resolver,
            ),
        )
