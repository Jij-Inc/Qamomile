"""Expose the complete IR resource interpreter."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, cast

import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator._config import (
    UnknownResourcePolicy,
)
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._gate_models import (
    _classify_pauli_evolve_depth,
    _estimate_named_gate,
    _pauli_terms_share_local_basis,
)
from qamomile.circuit.estimator._interpreter_transforms import (
    _TransformedCallInterpreter,
)
from qamomile.circuit.estimator._quantum_values import _qubit_value_size
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resolver_indices import (
    _resolve_concrete_array_payload,
)
from qamomile.circuit.estimator._resource_algebra import (
    _wrap_trace,
)
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    EstimateDerivation,
    EstimateQuality,
    ResourceExpr,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_expressions import (
    _expr,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    ResourceAssumption,
    ResourceTraceNode,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)
from qamomile.observable.hamiltonian import (
    HERMITIAN_IMAG_ATOL,
    PAULI_TERM_ZERO_ATOL,
    _pauli_strings_anticommute,
)


class ResourceInterpreter(_TransformedCallInterpreter):
    """Abstractly interpret IR operations into resource algebra values."""

    def eval_pauli_evolve(
        self,
        operation: PauliEvolveOp,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a Pauli-gadget decomposition for a bound Hamiltonian.

        Basis changes and parity ladders stay uncontrolled under an enclosing
        controlled evolution; only the axial rotation is controlled. A
        Hamiltonian constant contributes a controlled relative global phase.

        Args:
            operation (PauliEvolveOp): Pauli evolution operation.
            resolver (ExprResolver): Resolver for observable and time operands.
            controls (ResourceExpr | int): Surrounding coherent controls.
                Defaults to zero.

        Returns:
            ResourceEstimate: Clean-ancilla Pauli-gadget resources, or a modeled
            opaque/zero estimate when the configured unknown policy permits
            an unbound Hamiltonian.

        Raises:
            ValueError: If the Hamiltonian is unbound under ``ERROR`` policy,
                is non-Hermitian, or requires more qubits than its target
                register provides.
        """
        hamiltonian = self._resolve_hamiltonian_binding(operation, resolver)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            assumption = ResourceAssumption(
                "PauliEvolveOp requires a bound Hamiltonian for gate resources.",
                source="PauliEvolveOp",
            )
            if self.config.unknown_policy is UnknownResourcePolicy.ERROR:
                raise ValueError(
                    "Cannot estimate PauliEvolveOp without a bound "
                    "Hamiltonian; supply the observable through inputs or "
                    "bindings, or select a non-error unknown resource policy."
                )
            calls = CallResources.zero()
            if self.config.unknown_policy is UnknownResourcePolicy.OPAQUE_CALL:
                calls = CallResources(
                    calls_by_name={"pauli_evolve": _ONE},
                    queries_by_name={"pauli_evolve": _ONE},
                )
            return ResourceEstimate(
                calls=calls,
                derivation=EstimateDerivation.MODELED,
                trace=ResourceTraceNode(
                    "pauli_evolve",
                    "modeled",
                    assumptions=(assumption,),
                ),
            )._with_metadata(
                assumptions=(assumption,),
                quality=EstimateQuality.UNKNOWN,
            )

        register_constraint = _ResourceConstraint(
            expression=_qubit_value_size(operation.qubits, resolver),
            minimum=hamiltonian.num_qubits,
            label=(
                "Pauli evolution register width for a "
                f"{hamiltonian.num_qubits}-qubit Hamiltonian"
            ),
            unit="qubit",
        )
        register_constraint.validate()

        gamma = self._apply_condition_values(resolver.resolve(operation.gamma))
        constant = complex(hamiltonian.constant)
        if not math.isfinite(constant.real) or not math.isfinite(constant.imag):
            raise ValueError(
                "PauliEvolveOp requires finite Hamiltonian coefficients, but "
                f"the constant is {hamiltonian.constant}."
            )
        if abs(constant.imag) > HERMITIAN_IMAG_ATOL:
            raise ValueError(
                "PauliEvolveOp requires a Hermitian Hamiltonian (real "
                "constant), but the constant has a nonzero imaginary part."
            )
        for _operators, coefficient in hamiltonian:
            numeric_coefficient = complex(coefficient)
            if not math.isfinite(numeric_coefficient.real) or not math.isfinite(
                numeric_coefficient.imag
            ):
                raise ValueError(
                    "PauliEvolveOp requires finite Hamiltonian coefficients, "
                    f"but a term coefficient is {coefficient}."
                )
            if abs(numeric_coefficient.imag) > HERMITIAN_IMAG_ATOL:
                raise ValueError(
                    "PauliEvolveOp requires a Hermitian Hamiltonian (real "
                    "Pauli coefficients), but a term has a nonzero imaginary "
                    "part."
                )
        if gamma.is_zero is True:
            return _with_constraints(
                ResourceEstimate.zero("pauli_evolve"),
                register_constraint,
            )

        term_estimates: list[ResourceEstimate] = []
        active_pauli_terms: list[tuple[qm_o.PauliOperator, ...]] = []
        for operators, coefficient in hamiltonian:
            resolved_coefficient = complex(coefficient)
            if abs(resolved_coefficient) < PAULI_TERM_ZERO_ATOL or not operators:
                continue
            active_pauli_terms.append(operators)
            x_count = sum(operator.pauli == qm_o.Pauli.X for operator in operators)
            y_count = sum(operator.pauli == qm_o.Pauli.Y for operator in operators)
            basis_h_gate = _estimate_named_gate(
                "h",
                _ZERO,
                control_decomposition=self.config.control_decomposition,
            )
            basis_h = basis_h_gate.repeat(2 * (x_count + y_count))
            basis_s_gate = _estimate_named_gate(
                "s",
                _ZERO,
                control_decomposition=self.config.control_decomposition,
            )
            basis_s = basis_s_gate.repeat(2 * y_count)
            parity_gate = _estimate_named_gate(
                "cx",
                _ZERO,
                control_decomposition=self.config.control_decomposition,
            )
            parity = parity_gate.repeat(2 * max(0, len(operators) - 1))
            rotation = _estimate_named_gate(
                "rz",
                _expr(controls),
                control_decomposition=self.config.control_decomposition,
            )
            term = basis_h.seq(basis_s).seq(parity).seq(rotation)
            term = dataclasses.replace(
                term,
                depth=_classify_pauli_evolve_depth(
                    operators,
                    basis_h_layer=basis_h_gate.depth,
                    basis_s_layer=basis_s_gate.depth,
                    parity_layer=parity_gate.depth,
                    rotation=rotation.depth,
                ),
            )
            term_estimates.append(term)

        if abs(constant) >= PAULI_TERM_ZERO_ATOL:
            phase = -gamma * sp.Float(constant.real)
            term_estimates.append(
                self._estimate_global_phase_expression(
                    cast(sp.Expr, phase),
                    controls=controls,
                )
            )
        estimate = ResourceEstimate.seq_all(term_estimates)
        trotter_assumption = None
        if not _pauli_terms_share_local_basis(active_pauli_terms) and any(
            _pauli_strings_anticommute(left, right)
            for index, left in enumerate(active_pauli_terms)
            for right in active_pauli_terms[index + 1 :]
        ):
            trotter_assumption = ResourceAssumption(
                "Noncommuting Pauli terms are counted as one first-order "
                "Lie-Trotter product-formula step in Hamiltonian term order.",
                source="PauliEvolveOp",
            )
        trace = _wrap_trace(
            "pauli_evolve",
            estimate.trace,
            source_kind="body",
        )
        if trotter_assumption is not None:
            trace = dataclasses.replace(
                trace,
                assumptions=(*trace.assumptions, trotter_assumption),
            )
        active_estimate = dataclasses.replace(
            estimate,
            trace=trace,
        )
        if trotter_assumption is not None:
            active_estimate = active_estimate._with_metadata(
                assumptions=(trotter_assumption,),
                approximation=ApproximationStatus.APPROXIMATE,
            )
        if gamma.is_zero is not False:
            active_estimate = ResourceEstimate.zero("pauli_evolve").conditional(
                active_estimate,
                sp.Eq(gamma, _ZERO),
            )
        return _with_constraints(active_estimate, register_constraint)

    def _resolve_hamiltonian_binding(
        self,
        operation: PauliEvolveOp,
        resolver: ExprResolver,
    ) -> Any:
        """Resolve a Pauli evolution's observable through nested call scopes.

        Args:
            operation (PauliEvolveOp): Pauli evolution operation.
            resolver (ExprResolver): Resolver for the current callable scope.

        Returns:
            Any: Bound Hamiltonian value, or ``None`` when no binding matches.

        Raises:
            ValueError: If a concrete Hamiltonian array element has a negative
                or out-of-range index, or malformed bound data.
        """
        observable = operation.observable
        if observable.is_array_element():
            resolved_indices: list[int] = []
            for index in observable.element_indices:
                resolved_index = self._apply_condition_values(
                    resolver.resolve(index),
                    record_usage=False,
                )
                if not (resolved_index.is_number and resolved_index.is_integer is True):
                    break
                concrete_index = int(resolved_index)
                if concrete_index < 0:
                    raise ValueError(
                        "PauliEvolveOp Hamiltonian element index must be "
                        f"non-negative, got {concrete_index}."
                    )
                resolved_indices.append(concrete_index)
            else:
                parent = observable.parent_array
                if parent is not None:
                    payload = _resolve_concrete_array_payload(
                        parent,
                        {
                            **self.bindings,
                            **self._run_state.condition_values,
                        },
                        resolve_expression=resolver.resolve,
                        specialize=lambda expression: self._apply_condition_values(
                            expression,
                            record_usage=False,
                        ),
                        source="PauliEvolveOp Hamiltonian",
                    )
                    if payload is not None:
                        try:
                            resolved_element = payload
                            for index in resolved_indices:
                                resolved_element = resolved_element[index]
                        except (IndexError, KeyError, TypeError) as error:
                            raise ValueError(
                                "PauliEvolveOp Hamiltonian element exceeds its "
                                "bound component array."
                            ) from error
                        if isinstance(resolved_element, qm_o.Hamiltonian):
                            return resolved_element
            element_bindings = dict(self.bindings)
            for index in observable.element_indices:
                resolved_index = self._apply_condition_values(
                    resolver.resolve(index),
                    record_usage=False,
                )
                if resolved_index.is_number and resolved_index.is_integer is True:
                    element_bindings[index.uuid] = int(resolved_index)
            try:
                resolved_element = ValueResolver().resolve_bound_value(
                    observable,
                    element_bindings,
                )
            except EmitError as error:
                raise ValueError(
                    f"Cannot resolve PauliEvolveOp Hamiltonian array element: {error}"
                ) from error
            if isinstance(resolved_element, qm_o.Hamiltonian):
                return resolved_element
        candidates = [observable.name, observable.uuid]
        resolved = resolver.resolve(observable)
        if isinstance(resolved, sp.Symbol):
            candidates.append(_symbol_display_name(resolved))
        for candidate in candidates:
            if candidate in self.bindings:
                return self.bindings[candidate]
        return None
