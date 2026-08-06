"""Interpret primitive quantum resource operations."""

from __future__ import annotations

import dataclasses

import sympy as sp

from qamomile.circuit.estimator._clean_ancilla_projection import (
    _estimate_clean_ancilla_gate,
)
from qamomile.circuit.estimator._clifford_t_decomposition import (
    _PHASE_CLASS_CODES,
    _canonical_phase_gate_name,
    _CanonicalPhaseClass,
    _clifford_t_clean_ancillas,
    _clifford_t_conservative_condition,
)
from qamomile.circuit.estimator._clifford_t_depth import _clifford_t_gate_depth
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._gate_models import (
    _classify_gate,
    _estimate_named_gate_in_basis,
    _gate_has_rotation,
)
from qamomile.circuit.estimator._interpreter_region_analysis import (
    _RegionAnalysisInterpreter,
)
from qamomile.circuit.estimator._quantum_values import (
    _count_qinit,
    _qubit_value_size,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_algebra import (
    _conditional_depth,
    _wrap_trace,
)
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    ControlDecomposition,
    EstimateDerivation,
    EstimateQuality,
    GateBasis,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_expressions import (
    _expr,
    _piecewise,
    _resource_activity_condition,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    DepthResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    ResourceTraceNode,
    WidthResources,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.operation import (
    Operation,
    QInitOperation,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
)


class _PrimitiveInterpreter(_RegionAnalysisInterpreter):
    """Add primitive quantum operation handlers."""

    def eval_gate(
        self,
        operation: GateOperation,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a primitive gate operation.

        Args:
            operation (GateOperation): Gate operation.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.

        Returns:
            ResourceEstimate: Primitive gate resources.

        Raises:
            ValueError: If abstract controls are requested in the Clifford+T
                basis or the selected basis lacks a controlled lowering.
            NotImplementedError: If the clean-ancilla model has no registered
                lowering for the primitive.
        """
        if (
            self.config.basis is GateBasis.LOGICAL
            and self.config.control_decomposition
            is ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
        ):
            return _estimate_clean_ancilla_gate(
                ResourceEstimate.zero(),
                operation,
                _expr(controls),
            )
        if (
            self.config.basis is GateBasis.CLIFFORD_T
            and self.config.control_decomposition is ControlDecomposition.ABSTRACT
            and _expr(controls) != _ZERO
        ):
            raise ValueError(
                "Clifford+T estimation cannot preserve a controlled primitive "
                "as abstract. Select the clean-ancilla Toffoli control "
                "decomposition."
            )

        gates = _classify_gate(
            operation,
            num_controls=controls,
            basis=self.config.basis,
            precision=self.config.precision,
        )
        name = operation.gate_type.name.lower() if operation.gate_type else "gate"
        depth = (
            _clifford_t_gate_depth(operation, _expr(controls), self.config.precision)
            if self.config.basis is GateBasis.CLIFFORD_T
            else None
        )
        estimate = ResourceEstimate.primitive(name, gates, depth=depth)
        estimate = dataclasses.replace(
            estimate,
            basis=self.config.basis,
            control_decomposition=self.config.control_decomposition,
            precision=(
                self.config.precision
                if self.config.basis is GateBasis.CLIFFORD_T
                else None
            ),
        )
        if self.config.basis is GateBasis.CLIFFORD_T:
            clean_ancillas = _clifford_t_clean_ancillas(operation, _expr(controls))
            if clean_ancillas != _ZERO:
                estimate = dataclasses.replace(
                    estimate,
                    width=WidthResources(
                        clean_ancilla_qubits=clean_ancillas,
                        peak_qubits=clean_ancillas,
                    ),
                )
        if self.config.basis is GateBasis.CLIFFORD_T:
            conservative_when = _clifford_t_conservative_condition(
                name,
                _expr(controls),
            )
            if conservative_when is not sp.false:
                estimate = estimate._with_metadata(
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=conservative_when,
                )
            if _gate_has_rotation(operation):
                estimate = estimate._with_metadata(
                    quality=EstimateQuality.UNKNOWN,
                    approximation=ApproximationStatus.APPROXIMATE,
                )
        return estimate

    def eval_global_phase(
        self,
        operation: GlobalPhaseOperation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate a zero-qubit phase under its surrounding controls.

        The target-neutral estimator assigns a standalone global phase no
        logical gate cost. A target materializer may still synthesize it with
        gates or a clean carrier. With one or more coherent controls it
        becomes a phase gate on one control, with the remaining controls
        guarding that gate.

        Args:
            operation (GlobalPhaseOperation): Phase operation to estimate.
            resolver (ExprResolver): Resolver for the phase operand.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.

        Returns:
            ResourceEstimate: Zero when uncontrolled, otherwise one logical
            phase-gate contribution with the correct arity.
        """
        phase = self._apply_condition_values(resolver.resolve(operation.phase))
        return self._estimate_global_phase_expression(phase, controls=controls)

    def _estimate_global_phase_expression(
        self,
        phase: sp.Expr,
        *,
        controls: ResourceExpr | int,
    ) -> ResourceEstimate:
        """Estimate a resolved global-phase expression under controls.

        Args:
            phase (sp.Expr): Resolved phase angle in radians.
            controls (ResourceExpr | int): Number of coherent controls.

        Returns:
            ResourceEstimate: Zero for an unobservable global phase or an
            estimate for the relative phase induced by coherent controls.
        """
        control_count = _expr(controls)
        if control_count == _ZERO:
            return ResourceEstimate.zero("global_phase")
        phase_class = _CanonicalPhaseClass(phase)
        zero = ResourceEstimate.zero("global_phase")

        def gate_estimate(gate_name: str) -> ResourceEstimate:
            """Estimate one canonical relative-phase gate.

            Args:
                gate_name (str): Canonical phase gate name.

            Returns:
                ResourceEstimate: Gate estimate under all but one control.
            """
            estimate = _estimate_named_gate_in_basis(
                gate_name,
                control_count - _ONE,
                basis=self.config.basis,
                control_decomposition=self.config.control_decomposition,
                precision=self.config.precision,
            )
            return dataclasses.replace(
                estimate,
                trace=_wrap_trace(
                    "controlled_global_phase",
                    estimate.trace,
                    source_kind="decomposition",
                ),
            )

        if phase.is_number:
            gate_name = _canonical_phase_gate_name(phase)
            if gate_name is None:
                phase_estimate = zero
            else:
                if self.config.basis is not GateBasis.CLIFFORD_T:
                    # The shared control implementation preserves every nontrivial
                    # angle as P(theta), including special Clifford angles.
                    gate_name = "p"
                phase_estimate = gate_estimate(gate_name)
        elif self.config.basis is not GateBasis.CLIFFORD_T:
            phase_estimate = zero.conditional(
                gate_estimate("p"),
                sp.Eq(phase_class, _PHASE_CLASS_CODES[None]),
            )
        else:
            phase_estimate = gate_estimate("p")
            for gate_name in ("tdg", "sdg", "t", "s", "z"):
                phase_estimate = gate_estimate(gate_name).conditional(
                    phase_estimate,
                    sp.Eq(phase_class, _PHASE_CLASS_CODES[gate_name]),
                )
            phase_estimate = zero.conditional(
                phase_estimate,
                sp.Eq(phase_class, _PHASE_CLASS_CODES[None]),
            )

        if control_count.is_number and control_count.is_integer:
            return phase_estimate
        return zero.conditional(
            phase_estimate,
            sp.Eq(control_count, _ZERO),
        )

    def _with_zero_control_bracket(
        self,
        estimate: ResourceEstimate,
        *,
        zero_controls: ResourceExpr | int,
        surrounding_controls: ResourceExpr | int = 0,
        active_when: ResourceExpr | int | None = None,
    ) -> ResourceEstimate:
        """Bracket an estimate with X gates for zero-valued controls.

        One X is applied before and after the controlled region for every
        zero-valued control. ``surrounding_controls`` applies only to those X
        gates, as happens for SELECT nested under an outer controlled region.
        Brackets that normalize an operation's own controls are unconditional
        and therefore leave this argument at zero.

        Args:
            estimate (ResourceEstimate): Controlled-region estimate.
            zero_controls (ResourceExpr | int): Number of controls whose
                activation bit is zero.
            surrounding_controls (ResourceExpr | int): Controls inherited by
                the bracket X gates. Defaults to zero.
            active_when (ResourceExpr | int | None): Optional expression that
                must be positive for the region to emit. Defaults to ``None``.

        Returns:
            ResourceEstimate: Bracketed resource estimate.
        """
        zeros = _expr(zero_controls)
        if zeros == _ZERO:
            return estimate
        bracket_gate = _estimate_named_gate_in_basis(
            "x",
            _expr(surrounding_controls),
            basis=self.config.basis,
            control_decomposition=self.config.control_decomposition,
            precision=self.config.precision,
        )
        side = bracket_gate.repeat(zeros)
        if _expr(surrounding_controls) == _ZERO:
            side = dataclasses.replace(
                side,
                depth=_conditional_depth(
                    bracket_gate.depth,
                    DepthResources.zero(),
                    sp.Gt(zeros, _ZERO),
                ),
            )
        bracketed = side.seq(estimate).seq(side)
        if active_when is None:
            return bracketed
        active = _expr(active_when)
        if active.is_number:
            return bracketed if active > 0 else estimate
        return bracketed.conditional(
            estimate,
            _resource_activity_condition(active),
        )

    def eval_qinit(
        self,
        operation: QInitOperation,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate a qubit allocation.

        Args:
            operation (QInitOperation): Qubit initialization operation.
            resolver (ExprResolver): Resolver for symbolic array shapes.

        Returns:
            ResourceEstimate: Width-only allocation estimate.
        """
        count = _count_qinit(operation, resolver)
        result = operation.results[0]
        self._run_state.allocation_owners_by_uuid[result.uuid] = result.logical_id
        dimension_constraints = (
            tuple(
                _ResourceConstraint(
                    expression=resolver.resolve(dimension),
                    minimum=0,
                    label=f"Qubit allocation dimension {position}",
                    unit="element",
                )
                for position, dimension in enumerate(result.shape)
            )
            if isinstance(result, ArrayValue)
            else ()
        )
        for constraint in dimension_constraints:
            constraint.validate()
        width = WidthResources(
            allocated_qubits=count,
            peak_qubits=count,
        )
        return _with_constraints(
            ResourceEstimate(
                width=width,
                trace=ResourceTraceNode(
                    "qinit",
                    "primitive",
                    summary=f"qubits={count}",
                ),
                _allocation_sites={operation.results[0].uuid: count},
            ),
            *dimension_constraints,
        )

    def eval_measure(
        self,
        operation: Operation,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate a measurement operation.

        Args:
            operation (Operation): Measurement-like operation.
            resolver (ExprResolver): Resolver for vector operand dimensions.

        Returns:
            ResourceEstimate: Measurement depth estimate.

        Raises:
            TypeError: If ``operation`` is not a supported measurement IR
                operation.
        """
        if isinstance(operation, MeasureOperation):
            measured_qubits = _ONE
        elif isinstance(operation, MeasureVectorOperation):
            measured_qubits = (
                _qubit_value_size(operation.operands[0], resolver)
                if operation.operands
                else _ZERO
            )
        elif isinstance(operation, MeasureQFixedOperation):
            type_width = (
                _qubit_value_size(operation.operands[0], resolver)
                if operation.operands
                else _ZERO
            )
            measured_qubits = (
                type_width if type_width != _ZERO else sp.Integer(operation.num_bits)
            )
        else:  # pragma: no cover - dispatch admits only measurement operations.
            raise TypeError(
                f"Unsupported measurement operation {type(operation).__name__}."
            )
        layer = _piecewise(
            _ONE,
            _ZERO,
            sp.Gt(measured_qubits, _ZERO),
        )
        return ResourceEstimate(
            measurements=MeasurementResources(total=measured_qubits),
            depth=DepthResources(depth=layer, measurement_depth=layer),
            trace=ResourceTraceNode(type(operation).__name__, "primitive"),
        )

    def eval_expval(
        self,
        operation: ExpvalOp,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate an abstract expectation-value measurement.

        The semantic operation consumes the input state, but its concrete gate
        and shot cost depends on observable grouping, basis rotations, and the
        executor's sampling policy. The estimator therefore records one
        abstract query and one measurement layer without inventing a backend
        decomposition.

        Args:
            operation (ExpvalOp): Expectation-value operation.
            resolver (ExprResolver): Resolver for the quantum operand width.

        Returns:
            ResourceEstimate: Modeled abstract query and measurement depth.
        """
        measured_qubits = (
            _qubit_value_size(operation.qubits, resolver)
            if operation.operands
            else _ZERO
        )
        layer = _piecewise(
            _ONE,
            _ZERO,
            sp.Gt(measured_qubits, _ZERO),
        )
        assumption = ResourceAssumption(
            "expectation-value gate and shot costs depend on observable "
            "grouping, basis rotations, and executor sampling policy",
            source="ExpvalOp",
        )
        return ResourceEstimate(
            depth=DepthResources(depth=layer, measurement_depth=layer),
            calls=CallResources(
                calls_by_name={"expval": _ONE},
                queries_by_name={"expval": _ONE},
            ),
            assumptions=(assumption,),
            derivation=EstimateDerivation.MODELED,
            quality=EstimateQuality.UNKNOWN,
            trace=ResourceTraceNode(
                "expval",
                "opaque",
                summary="one abstract expectation query",
                assumptions=(assumption,),
            ),
        )

    def eval_project(self, operation: ProjectOperation) -> ResourceEstimate:
        """Evaluate a projective measurement operation.

        The public X/Y projection helpers normally lower to basis-change gates
        around a Z projection before this interpreter runs. Hand-built or
        deserialized IR can still carry an X/Y axis, so estimate those semantic
        basis changes here as well instead of silently treating every axis as
        a bare measurement.

        Args:
            operation (ProjectOperation): Projection operation.

        Returns:
            ResourceEstimate: Measurement-like resource estimate.
        """
        measurement = ResourceEstimate(
            measurements=MeasurementResources(total=_ONE),
            depth=DepthResources(depth=_ONE, measurement_depth=_ONE),
            trace=ResourceTraceNode("project_z", "primitive"),
        )
        basis_changes = {
            "x": (("h",), ("h",)),
            "y": (("sdg", "h"), ("h", "s")),
            "z": ((), ()),
        }
        before, after = basis_changes[operation.axis]
        estimate = ResourceEstimate.zero()
        for gate_name in before:
            estimate = estimate.seq(
                _estimate_named_gate_in_basis(
                    gate_name,
                    _ZERO,
                    basis=self.config.basis,
                    control_decomposition=self.config.control_decomposition,
                    precision=self.config.precision,
                )
            )
        estimate = estimate.seq(measurement)
        for gate_name in after:
            estimate = estimate.seq(
                _estimate_named_gate_in_basis(
                    gate_name,
                    _ZERO,
                    basis=self.config.basis,
                    control_decomposition=self.config.control_decomposition,
                    precision=self.config.precision,
                )
            )
        return dataclasses.replace(
            estimate,
            trace=_wrap_trace(
                f"project_{operation.axis}",
                estimate.trace,
                source_kind="decomposition",
            ),
        )

    def eval_reset(self, operation: ResetOperation) -> ResourceEstimate:
        """Evaluate a reset operation.

        Args:
            operation (ResetOperation): Reset operation.

        Returns:
            ResourceEstimate: Reset primitive resource estimate.
        """
        return ResourceEstimate(
            resets=ResetResources(total=_ONE),
            depth=DepthResources(depth=_ONE, reset_depth=_ONE),
            trace=ResourceTraceNode(
                name="reset",
                source_kind="primitive",
                summary="resets=1",
            ),
        )
