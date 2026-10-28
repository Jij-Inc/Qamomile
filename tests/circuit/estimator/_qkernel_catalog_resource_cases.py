"""Test-case definitions for qkernel-catalog resource estimates.

The catalog cases intentionally leave every expected result as ``None``.  A
case starts asserting values as soon as its corresponding expectation is
filled in.  Until then, the test still evaluates the estimation path before
skipping, so invalid inputs and unsupported symbolic construction are visible.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import sympy as sp

import qamomile.circuit as qmc


@dataclasses.dataclass(frozen=True)
class ExpectedResourceEstimate:
    """Describe every public resource field expected from one estimate.

    All resource records use their public zero defaults, making omitted
    numeric fields explicit assertions of zero rather than partial matches.
    Assumptions are represented by source only because their explanatory
    messages may change without changing the estimate semantics. ``trace`` is
    excluded because it is diagnostic output rather than a resource result.
    """

    width: qmc.WidthResources = dataclasses.field(default_factory=qmc.WidthResources)
    gates: qmc.GateResources = dataclasses.field(default_factory=qmc.GateResources)
    depth: qmc.DepthResources = dataclasses.field(default_factory=qmc.DepthResources)
    calls: qmc.CallResources = dataclasses.field(default_factory=qmc.CallResources)
    measurements: qmc.MeasurementResources = dataclasses.field(
        default_factory=qmc.MeasurementResources
    )
    resets: qmc.ResetResources = dataclasses.field(default_factory=qmc.ResetResources)
    parameters: frozenset[str] = frozenset()
    assumption_sources: tuple[str, ...] = ()
    derivation: qmc.EstimateDerivation = qmc.EstimateDerivation.STRUCTURAL
    quality: qmc.EstimateQuality = qmc.EstimateQuality.EXACT
    approximation: qmc.ApproximationStatus = qmc.ApproximationStatus.EXACT
    control_decomposition: qmc.ControlDecomposition = (
        qmc.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
    )


SymbolicExpectationFactory: TypeAlias = Callable[
    [Mapping[str, sp.Symbol]], ExpectedResourceEstimate
]


@dataclasses.dataclass(frozen=True)
class CatalogEstimateOptions:
    """Store the estimator configuration used by one catalog case."""

    control_decomposition: qmc.ControlDecomposition = (
        qmc.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
    )
    unknown_policy: qmc.UnknownResourcePolicy = qmc.UnknownResourcePolicy.ERROR
    strategies: tuple[tuple[str, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class CatalogResourceSpecialization:
    """Describe one concrete specialization through both public routes."""

    id: str
    inputs: Mapping[str, Any]
    substitutions: Mapping[str, int | float]
    expected_from_inputs: ExpectedResourceEstimate | None = None
    expected_from_substitution: ExpectedResourceEstimate | None = None


@dataclasses.dataclass(frozen=True)
class CatalogResourceCase:
    """Describe one qkernel and estimator-configuration variant."""

    catalog_id: str
    variant: str = "default"
    options: CatalogEstimateOptions = dataclasses.field(
        default_factory=CatalogEstimateOptions
    )
    symbolic_inputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    expected_symbolic: ExpectedResourceEstimate | SymbolicExpectationFactory | None = (
        None
    )
    specializations: tuple[CatalogResourceSpecialization, ...] = ()

    @property
    def id(self) -> str:
        """Return the stable pytest and cache identifier for this case."""

        return f"{self.catalog_id}[{self.variant}]"


class CatalogEstimationRoute(enum.StrEnum):
    """Identify the public estimation route expected to fail."""

    SYMBOLIC = "symbolic"
    INPUTS = "inputs"
    SUBSTITUTE = "substitute"


@dataclasses.dataclass(frozen=True)
class CatalogResourceErrorCase:
    """Describe a catalog route that should raise a public exception."""

    id: str
    catalog_id: str
    route: CatalogEstimationRoute
    options: CatalogEstimateOptions = dataclasses.field(
        default_factory=CatalogEstimateOptions
    )
    symbolic_inputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    inputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    substitutions: Mapping[str, int | float] = dataclasses.field(default_factory=dict)
    exception_type: type[Exception] = ValueError
    match: str | None = None


def _specialization(
    id: str,
    *,
    inputs: Mapping[str, Any],
    substitutions: Mapping[str, int | float],
    expected_from_inputs: ExpectedResourceEstimate | None = None,
    expected_from_substitution: ExpectedResourceEstimate | None = None,
) -> CatalogResourceSpecialization:
    """Build one direct-input and late-substitution specialization pair."""

    return CatalogResourceSpecialization(
        id=id,
        inputs=inputs,
        substitutions=substitutions,
        expected_from_inputs=expected_from_inputs,
        expected_from_substitution=expected_from_substitution,
    )


def _no_input_specialization(
    *,
    expected_from_inputs: ExpectedResourceEstimate | None = None,
    expected_from_substitution: ExpectedResourceEstimate | None = None,
) -> CatalogResourceSpecialization:
    """Build the direct-input and substitution routes for a no-argument kernel."""

    return _specialization(
        id="no-inputs",
        inputs={},
        substitutions={},
        expected_from_inputs=expected_from_inputs,
        expected_from_substitution=expected_from_substitution,
    )


CATALOG_RESOURCE_CASES: tuple[CatalogResourceCase, ...] = (
    # Single-qubit gates.
    CatalogResourceCase(
        catalog_id="single_h",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(total=1, single_qubit=1, clifford=1),
            depth=qmc.DepthResources(depth=1, clifford_depth=1, gate_depth=1),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_x",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(total=1, single_qubit=1, clifford=1),
            depth=qmc.DepthResources(depth=1, clifford_depth=1, gate_depth=1),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_y",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(total=1, single_qubit=1, clifford=1),
            depth=qmc.DepthResources(depth=1, clifford_depth=1, gate_depth=1),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_z",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(total=1, single_qubit=1, clifford=1),
            depth=qmc.DepthResources(depth=1, clifford_depth=1, gate_depth=1),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_t",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(total=1, single_qubit=1, t=1, non_clifford=1),
            depth=qmc.DepthResources(
                depth=1, t_depth=1, gate_depth=1, non_clifford_depth=1
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_tdg",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(total=1, single_qubit=1, t=1, non_clifford=1),
            depth=qmc.DepthResources(
                depth=1, t_depth=1, gate_depth=1, non_clifford_depth=1
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_s",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(total=1, single_qubit=1, clifford=1),
            depth=qmc.DepthResources(depth=1, gate_depth=1, clifford_depth=1),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_sdg",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(total=1, single_qubit=1, clifford=1),
            depth=qmc.DepthResources(depth=1, gate_depth=1, clifford_depth=1),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_p",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(
                total=1, single_qubit=1, rotation=1, non_clifford=1
            ),
            depth=qmc.DepthResources(
                depth=1, rotation_depth=1, gate_depth=1, non_clifford_depth=1
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_rx",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(
                total=1, single_qubit=1, rotation=1, non_clifford=1
            ),
            depth=qmc.DepthResources(
                depth=1, rotation_depth=1, gate_depth=1, non_clifford_depth=1
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_ry",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(
                total=1, single_qubit=1, rotation=1, non_clifford=1
            ),
            depth=qmc.DepthResources(
                depth=1, rotation_depth=1, gate_depth=1, non_clifford_depth=1
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_rz",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(
                total=1, single_qubit=1, rotation=1, non_clifford=1
            ),
            depth=qmc.DepthResources(
                depth=1, rotation_depth=1, gate_depth=1, non_clifford_depth=1
            ),
        ),
    ),
    # Two-qubit gates.
    CatalogResourceCase(
        catalog_id="single_cx",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=2,
                peak_qubits=2,
            ),
            gates=qmc.GateResources(total=1, two_qubit=1, clifford=1),
            depth=qmc.DepthResources(depth=1, gate_depth=1, clifford_depth=1),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_cz",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=2,
                peak_qubits=2,
            ),
            gates=qmc.GateResources(total=1, two_qubit=1, clifford=1),
            depth=qmc.DepthResources(depth=1, gate_depth=1, clifford_depth=1),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_cp",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=2,
                peak_qubits=2,
            ),
            gates=qmc.GateResources(total=1, two_qubit=1, rotation=1, non_clifford=1),
            depth=qmc.DepthResources(
                depth=1, gate_depth=1, rotation_depth=1, non_clifford_depth=1
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_swap",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=2,
                peak_qubits=2,
            ),
            gates=qmc.GateResources(total=1, two_qubit=1, clifford=1),
            depth=qmc.DepthResources(depth=1, gate_depth=1, clifford_depth=1),
        ),
    ),
    CatalogResourceCase(
        catalog_id="single_rzz",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=2,
                peak_qubits=2,
            ),
            gates=qmc.GateResources(total=1, two_qubit=1, rotation=1, non_clifford=1),
            depth=qmc.DepthResources(
                depth=1, gate_depth=1, rotation_depth=1, non_clifford_depth=1
            ),
        ),
    ),
    # Basic circuits.
    CatalogResourceCase(
        catalog_id="no_operation",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                ),
                substitutions={"n": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=parameters["n"],
                peak_qubits=parameters["n"],
            ),
            parameters=frozenset({"n"}),
        ),
    ),
    CatalogResourceCase(
        catalog_id="only_measurements",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    measurements=qmc.MeasurementResources(
                        total=3,
                    ),
                    depth=qmc.DepthResources(depth=1, measurement_depth=1),
                ),
                substitutions={"n": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    measurements=qmc.MeasurementResources(
                        total=3,
                    ),
                    depth=qmc.DepthResources(depth=1, measurement_depth=1),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=parameters["n"],
                peak_qubits=parameters["n"],
            ),
            measurements=qmc.MeasurementResources(
                total=parameters["n"],
            ),
            depth=qmc.DepthResources(
                depth=sp.Piecewise(
                    (1, parameters["n"] > 0),
                    (0, True),
                ),
                measurement_depth=sp.Piecewise(
                    (1, parameters["n"] > 0),
                    (0, True),
                ),
            ),
            parameters=frozenset({"n"}),
        ),
    ),
    CatalogResourceCase(
        catalog_id="simple_for_loop",
        specializations=(
            _specialization(
                id="m-3",
                inputs={"m": 3},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=1,
                        peak_qubits=1,
                    ),
                    gates=qmc.GateResources(total=3, single_qubit=3, clifford=3),
                    depth=qmc.DepthResources(depth=3, gate_depth=3, clifford_depth=3),
                ),
                substitutions={"m": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=1,
                        peak_qubits=1,
                    ),
                    gates=qmc.GateResources(total=3, single_qubit=3, clifford=3),
                    depth=qmc.DepthResources(depth=3, gate_depth=3, clifford_depth=3),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=1,
                peak_qubits=1,
            ),
            gates=qmc.GateResources(
                total=parameters["m"],
                single_qubit=parameters["m"],
                clifford=parameters["m"],
            ),
            depth=qmc.DepthResources(
                depth=parameters["m"],
                gate_depth=parameters["m"],
                clifford_depth=parameters["m"],
            ),
            parameters=frozenset({"m"}),
        ),
    ),
    CatalogResourceCase(
        catalog_id="all_rx",
        specializations=(
            _specialization(
                id="three-qubits",
                inputs={"n": 3, "thetas": [0.1, 0.2, 0.3]},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=3, single_qubit=3, rotation=3, non_clifford=3
                    ),
                    depth=qmc.DepthResources(
                        depth=1, rotation_depth=1, gate_depth=1, non_clifford_depth=1
                    ),
                ),
                substitutions={"n": 3, "thetas_dim0": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=3, single_qubit=3, rotation=3, non_clifford=3
                    ),
                    depth=qmc.DepthResources(
                        depth=1, rotation_depth=1, gate_depth=1, non_clifford_depth=1
                    ),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=parameters["n"],
                peak_qubits=parameters["n"],
            ),
            gates=qmc.GateResources(
                total=parameters["n"],
                single_qubit=parameters["n"],
                rotation=parameters["n"],
                non_clifford=parameters["n"],
            ),
            depth=qmc.DepthResources(
                depth=sp.Piecewise(
                    (1, parameters["n"] > 0),
                    (0, True),
                ),
                rotation_depth=sp.Piecewise(
                    (1, parameters["n"] > 0),
                    (0, True),
                ),
                gate_depth=sp.Piecewise(
                    (1, parameters["n"] > 0),
                    (0, True),
                ),
                non_clifford_depth=sp.Piecewise(
                    (1, parameters["n"] > 0),
                    (0, True),
                ),
            ),
            parameters=frozenset({"n", "thetas_dim0"}),
        ),
    ),
    CatalogResourceCase(
        catalog_id="naive_toffoli_decomposition",
        specializations=(_no_input_specialization(),),
        # https://arxiv.org/pdf/1210.0974
        # But Qamomile sees the dependency, but counts as 0 depth for non-target gates.
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=3,
                peak_qubits=3,
            ),
            gates=qmc.GateResources(
                total=16,
                single_qubit=10,
                two_qubit=6,
                clifford=9,
                non_clifford=7,
                t=7,
            ),
            depth=qmc.DepthResources(
                depth=12,
                gate_depth=12,
                clifford_depth=8,
                non_clifford_depth=5,
                t_depth=5,
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="commutated_toffoli_decomposition",
        specializations=(_no_input_specialization(),),
        # https://arxiv.org/pdf/1210.0974
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=3,
                peak_qubits=3,
            ),
            gates=qmc.GateResources(
                total=16,
                single_qubit=10,
                two_qubit=6,
                clifford=9,
                non_clifford=7,
                t=7,
            ),
            depth=qmc.DepthResources(
                depth=12,
                gate_depth=12,
                clifford_depth=8,
                non_clifford_depth=4,
                t_depth=4,
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="optimal_toffoli_decomposition",
        specializations=(_no_input_specialization(),),
        # https://arxiv.org/pdf/1210.0974
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=3,
                peak_qubits=3,
            ),
            gates=qmc.GateResources(
                total=17,
                single_qubit=10,
                two_qubit=7,
                clifford=10,
                non_clifford=7,
                t=7,
            ),
            depth=qmc.DepthResources(
                depth=9,
                gate_depth=9,
                clifford_depth=7,
                non_clifford_depth=3,
                t_depth=3,
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="optimal_toffoli_decomposition_loop",
        specializations=(
            _specialization(
                id="m-2",
                inputs={"m": 2},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=17 * 3,
                        single_qubit=10 * 3,
                        two_qubit=7 * 3,
                        clifford=10 * 3,
                        non_clifford=7 * 3,
                        t=7 * 3,
                    ),
                    depth=qmc.DepthResources(
                        depth=9 * 3,
                        gate_depth=9 * 3,
                        clifford_depth=7 * 3,
                        non_clifford_depth=3 * 3,
                        t_depth=3 * 3,
                    ),
                    assumption_sources=("m",),
                    quality=qmc.EstimateQuality.CONSERVATIVE,
                ),
                substitutions={"m": 2},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=17 * 3,
                        single_qubit=10 * 3,
                        two_qubit=7 * 3,
                        clifford=10 * 3,
                        non_clifford=7 * 3,
                        t=7 * 3,
                    ),
                    depth=qmc.DepthResources(
                        depth=9 * 3,
                        gate_depth=9 * 3,
                        clifford_depth=7 * 3,
                        non_clifford_depth=3 * 3,
                        t_depth=3 * 3,
                    ),
                    assumption_sources=("m",),
                    quality=qmc.EstimateQuality.CONSERVATIVE,
                ),
            ),
        ),
        # https://arxiv.org/pdf/1210.0974
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=3,
                peak_qubits=3,
            ),
            gates=qmc.GateResources(
                total=17 * (parameters["m"] + 1),
                single_qubit=10 * (parameters["m"] + 1),
                two_qubit=7 * (parameters["m"] + 1),
                clifford=10 * (parameters["m"] + 1),
                non_clifford=7 * (parameters["m"] + 1),
                t=7 * (parameters["m"] + 1),
            ),
            depth=qmc.DepthResources(
                depth=9 * (parameters["m"] + 1),
                gate_depth=9 * (parameters["m"] + 1),
                clifford_depth=7 * (parameters["m"] + 1),
                non_clifford_depth=3 * (parameters["m"] + 1),
                t_depth=3 * (parameters["m"] + 1),
            ),
            parameters=frozenset({"m"}),
            assumption_sources=("m",),
            quality=qmc.EstimateQuality.CONSERVATIVE,
        ),
    ),
    # Entanglement.
    CatalogResourceCase(
        catalog_id="bell_state",
        specializations=(_no_input_specialization(),),
        expected_symbolic=ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=2,
                peak_qubits=2,
            ),
            gates=qmc.GateResources(total=2, single_qubit=1, two_qubit=1, clifford=2),
            depth=qmc.DepthResources(depth=2, gate_depth=2, clifford_depth=2),
        ),
    ),
    CatalogResourceCase(
        catalog_id="linear_entanglement",
        # https://bloqade.quera.comdev/digital/examples/qasm2/ghz/
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(total=2, two_qubit=2, clifford=2),
                    depth=qmc.DepthResources(depth=2, gate_depth=2, clifford_depth=2),
                ),
                substitutions={"n": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(total=2, two_qubit=2, clifford=2),
                    depth=qmc.DepthResources(depth=2, gate_depth=2, clifford_depth=2),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=parameters["n"],
                peak_qubits=parameters["n"],
            ),
            gates=qmc.GateResources(
                total=sp.Max(parameters["n"] - 1, 0),
                two_qubit=sp.Max(parameters["n"] - 1, 0),
                clifford=sp.Max(parameters["n"] - 1, 0),
            ),
            depth=qmc.DepthResources(
                depth=sp.Max(parameters["n"] - 1, 0),
                gate_depth=sp.Max(parameters["n"] - 1, 0),
                clifford_depth=sp.Max(parameters["n"] - 1, 0),
            ),
            parameters=frozenset({"n"}),
        ),
    ),
    CatalogResourceCase(
        catalog_id="full_entanglement",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(total=3, two_qubit=3, clifford=3),
                    depth=qmc.DepthResources(depth=3, gate_depth=3, clifford_depth=3),
                ),
                substitutions={"n": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(total=3, two_qubit=3, clifford=3),
                    depth=qmc.DepthResources(depth=3, gate_depth=3, clifford_depth=3),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=parameters["n"],
                peak_qubits=parameters["n"],
            ),
            gates=qmc.GateResources(
                total=parameters["n"] * (parameters["n"] - 1) / 2,
                two_qubit=parameters["n"] * (parameters["n"] - 1) / 2,
                clifford=parameters["n"] * (parameters["n"] - 1) / 2,
            ),
            depth=qmc.DepthResources(
                depth=sp.Max(2 * parameters["n"] - 3, 0),
                clifford_depth=sp.Max(2 * parameters["n"] - 3, 0),
                gate_depth=sp.Max(2 * parameters["n"] - 3, 0),
            ),
            parameters=frozenset({"n"}),
        ),
    ),
    CatalogResourceCase(
        catalog_id="ghz_state",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=3, single_qubit=1, two_qubit=2, clifford=3
                    ),
                    depth=qmc.DepthResources(depth=3, gate_depth=3, clifford_depth=3),
                ),
                substitutions={"n": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=3, single_qubit=1, two_qubit=2, clifford=3
                    ),
                    depth=qmc.DepthResources(depth=3, gate_depth=3, clifford_depth=3),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=parameters["n"],
                peak_qubits=parameters["n"],
            ),
            gates=qmc.GateResources(
                total=parameters["n"],
                single_qubit=1,
                two_qubit=parameters["n"] - 1,
                clifford=parameters["n"],
            ),
            depth=qmc.DepthResources(
                depth=parameters["n"],
                gate_depth=parameters["n"],
                clifford_depth=parameters["n"],
            ),
            assumption_sources=("qkernel input domain",),
            parameters=frozenset({"n"}),
        ),
    ),
    CatalogResourceCase(
        catalog_id="parallel_ghz_state",
        # https://arxiv.org/pdf/2212.03668
        specializations=(
            _specialization(
                id="m-3",
                inputs={"m": 3},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=8,
                        peak_qubits=8,
                    ),
                    gates=qmc.GateResources(
                        total=8, single_qubit=1, two_qubit=7, clifford=8
                    ),
                    depth=qmc.DepthResources(depth=4, gate_depth=4, clifford_depth=4),
                ),
                substitutions={"m": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=8,
                        peak_qubits=8,
                    ),
                    gates=qmc.GateResources(
                        total=8, single_qubit=1, two_qubit=7, clifford=8
                    ),
                    depth=qmc.DepthResources(depth=4, gate_depth=4, clifford_depth=4),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=2 ** parameters["m"],
                peak_qubits=2 ** parameters["m"],
            ),
            gates=qmc.GateResources(
                total=2 ** parameters["m"],
                single_qubit=1,
                two_qubit=2 ** parameters["m"] - 1,
                clifford=2 ** parameters["m"],
            ),
            depth=qmc.DepthResources(
                depth=parameters["m"] + 1,
                gate_depth=parameters["m"] + 1,
                clifford_depth=parameters["m"] + 1,
            ),
            parameters=frozenset({"m"}),
        ),
    ),
    # QFT and inverse QFT.
    CatalogResourceCase(
        catalog_id="qft",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=7,
                        single_qubit=3,
                        two_qubit=4,
                        clifford=4,
                        rotation=3,
                        non_clifford=3,
                    ),
                    depth=qmc.DepthResources(
                        depth=6,
                        gate_depth=6,
                        clifford_depth=4,
                        rotation_depth=3,
                        non_clifford_depth=3,
                    ),
                ),
                substitutions={"n": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=7,
                        single_qubit=3,
                        two_qubit=4,
                        clifford=4,
                        rotation=3,
                        non_clifford=3,
                    ),
                    depth=qmc.DepthResources(
                        depth=6,
                        gate_depth=6,
                        clifford_depth=4,
                        rotation_depth=3,
                        non_clifford_depth=3,
                    ),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=parameters["n"],
                peak_qubits=parameters["n"],
            ),
            gates=qmc.GateResources(
                total=parameters["n"] * (parameters["n"] + 1) / 2
                + parameters["n"] // 2,
                single_qubit=parameters["n"],
                two_qubit=parameters["n"] * (parameters["n"] - 1) / 2
                + parameters["n"] // 2,
                clifford=parameters["n"] + parameters["n"] // 2,
                rotation=parameters["n"] * (parameters["n"] - 1) / 2,
                non_clifford=parameters["n"] * (parameters["n"] - 1) / 2,
            ),
            depth=qmc.DepthResources(
                depth=sp.Piecewise(
                    (sp.Max(0, 2 * parameters["n"] - 1) + 1, parameters["n"] > 1),
                    (sp.Max(0, 2 * parameters["n"] - 1), True),
                ),
                gate_depth=sp.Piecewise(
                    (sp.Max(0, 2 * parameters["n"] - 1) + 1, parameters["n"] > 1),
                    (sp.Max(0, 2 * parameters["n"] - 1), True),
                ),
                clifford_depth=sp.Piecewise(
                    (parameters["n"] + 1, parameters["n"] > 1),
                    (parameters["n"], True),
                ),
                non_clifford_depth=sp.Max(0, 2 * parameters["n"] - 3),
                rotation_depth=sp.Max(0, 2 * parameters["n"] - 3),
            ),
            parameters=frozenset({"n"}),
        ),
    ),
    CatalogResourceCase(
        catalog_id="iqft",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                expected_from_inputs=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=7,
                        single_qubit=3,
                        two_qubit=4,
                        clifford=4,
                        rotation=3,
                        non_clifford=3,
                    ),
                    depth=qmc.DepthResources(
                        depth=6,
                        gate_depth=6,
                        clifford_depth=4,
                        rotation_depth=3,
                        non_clifford_depth=3,
                    ),
                ),
                substitutions={"n": 3},
                expected_from_substitution=ExpectedResourceEstimate(
                    width=qmc.WidthResources(
                        allocated_qubits=3,
                        peak_qubits=3,
                    ),
                    gates=qmc.GateResources(
                        total=7,
                        single_qubit=3,
                        two_qubit=4,
                        clifford=4,
                        rotation=3,
                        non_clifford=3,
                    ),
                    depth=qmc.DepthResources(
                        depth=6,
                        gate_depth=6,
                        clifford_depth=4,
                        rotation_depth=3,
                        non_clifford_depth=3,
                    ),
                ),
            ),
        ),
        expected_symbolic=lambda parameters: ExpectedResourceEstimate(
            width=qmc.WidthResources(
                allocated_qubits=parameters["n"],
                peak_qubits=parameters["n"],
            ),
            gates=qmc.GateResources(
                total=parameters["n"] * (parameters["n"] + 1) / 2
                + parameters["n"] // 2,
                single_qubit=parameters["n"],
                two_qubit=parameters["n"] * (parameters["n"] - 1) / 2
                + parameters["n"] // 2,
                clifford=parameters["n"] + parameters["n"] // 2,
                rotation=parameters["n"] * (parameters["n"] - 1) / 2,
                non_clifford=parameters["n"] * (parameters["n"] - 1) / 2,
            ),
            depth=qmc.DepthResources(
                depth=sp.Piecewise(
                    (sp.Max(0, 2 * parameters["n"] - 1) + 1, parameters["n"] > 1),
                    (sp.Max(0, 2 * parameters["n"] - 1), True),
                ),
                gate_depth=sp.Piecewise(
                    (sp.Max(0, 2 * parameters["n"] - 1) + 1, parameters["n"] > 1),
                    (sp.Max(0, 2 * parameters["n"] - 1), True),
                ),
                clifford_depth=sp.Piecewise(
                    (parameters["n"] + 1, parameters["n"] > 1),
                    (parameters["n"], True),
                ),
                non_clifford_depth=sp.Max(0, 2 * parameters["n"] - 3),
                rotation_depth=sp.Max(0, 2 * parameters["n"] - 3),
            ),
            parameters=frozenset({"n"}),
        ),
    ),
    # Oracle algorithms.
    CatalogResourceCase(
        catalog_id="hadamard_test",
        specializations=(_no_input_specialization(),),
    ),
    CatalogResourceCase(
        catalog_id="swap_test",
        specializations=(_no_input_specialization(),),
    ),
    CatalogResourceCase(
        catalog_id="simplest_oracle",
        specializations=(_no_input_specialization(),),
    ),
    CatalogResourceCase(
        catalog_id="deutsch",
        specializations=(_no_input_specialization(),),
    ),
    CatalogResourceCase(
        catalog_id="deutsch_jozsa",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                substitutions={"n": 3},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="simon",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                substitutions={"n": 3},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="teleportation",
        specializations=(_no_input_specialization(),),
    ),
    # Phase estimation.
    CatalogResourceCase(
        catalog_id="phase_gate_qpe",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3, "theta": 0.25},
                substitutions={"n": 3},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="opaque_oracle_qpe",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                substitutions={"n": 3},
            ),
        ),
    ),
    # Variational and optimization circuits.
    CatalogResourceCase(
        catalog_id="hardware_efficient_ansatz",
        specializations=(
            _specialization(
                id="three-qubits-two-layers",
                inputs={
                    "n": 3,
                    "num_layers": 2,
                    "thetas": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    "phis": [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
                },
                substitutions={
                    "n": 3,
                    "num_layers": 2,
                    "thetas_dim0": 2,
                    "thetas_dim1": 3,
                    "phis_dim0": 2,
                    "phis_dim1": 3,
                },
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="qaoa_state_umbiguous",
        symbolic_inputs={
            "quad": {(0, 1): 1.0, (1, 2): -0.5},
            "linear": {0: 0.25, 2: -1.0},
        },
        specializations=(
            _specialization(
                id="three-qubits-two-layers",
                inputs={
                    "n": 3,
                    "num_layers": 2,
                    "gammas": [0.1, 0.2],
                    "betas": [0.3, 0.4],
                },
                substitutions={
                    "n": 3,
                    "num_layers": 2,
                    "gammas_dim0": 2,
                    "betas_dim0": 2,
                },
            ),
        ),
    ),
    # Multi-controlled circuits.
    CatalogResourceCase(
        catalog_id="network_decomposition_controlled_z",
        specializations=(
            _specialization(
                id="n-4",
                inputs={"n": 4},
                substitutions={"n": 4},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="naive_multi_controlled_z",
        specializations=(
            _specialization(
                id="n-4",
                inputs={"n": 4},
                substitutions={"n": 4},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="naive_multi_controlled_z",
        variant="abstract-control",
        options=CatalogEstimateOptions(
            control_decomposition=qmc.ControlDecomposition.ABSTRACT
        ),
        specializations=(
            _specialization(
                id="n-4",
                inputs={"n": 4},
                substitutions={"n": 4},
            ),
        ),
    ),
    # Grover and quantum counting.
    CatalogResourceCase(
        catalog_id="grover_network_decomposition",
        specializations=(
            _specialization(
                id="n-4-iterations-2",
                inputs={"n": 4, "n_iters": 2},
                substitutions={"n": 4, "n_iters": 2},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="grover_naive_multi_controlled_z",
        specializations=(
            _specialization(
                id="n-4-iterations-2",
                inputs={"n": 4, "n_iters": 2},
                substitutions={"n": 4, "n_iters": 2},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="grover_naive_multi_controlled_z",
        variant="abstract-control",
        options=CatalogEstimateOptions(
            control_decomposition=qmc.ControlDecomposition.ABSTRACT
        ),
        specializations=(
            _specialization(
                id="n-4-iterations-2",
                inputs={"n": 4, "n_iters": 2},
                substitutions={"n": 4, "n_iters": 2},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="quantum_counting",
        specializations=(
            _specialization(
                id="counting-2-search-3",
                inputs={"n": 2, "m": 3},
                substitutions={"n": 2, "m": 3},
            ),
        ),
    ),
    # Arithmetic.
    CatalogResourceCase(
        catalog_id="maj",
        specializations=(_no_input_specialization(),),
    ),
    CatalogResourceCase(
        catalog_id="maj_loop",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                substitutions={"n": 3},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="uma_2_cnot",
        specializations=(_no_input_specialization(),),
    ),
    CatalogResourceCase(
        catalog_id="uma_2_cnot_loop",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                substitutions={"n": 3},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="uma_3_cnot",
        specializations=(_no_input_specialization(),),
    ),
    CatalogResourceCase(
        catalog_id="uma_3_cnot_loop",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                substitutions={"n": 3},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="simple_ripple_carry_adder_2_cnot",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                substitutions={"n": 3},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="simple_ripple_carry_adder_3_cnot",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                substitutions={"n": 3},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="draper_inplace_qc_adder",
        specializations=(
            _specialization(
                id="n-3-num-2-factor-1",
                inputs={"n": 3, "num": 2, "factor": 1},
                substitutions={"n": 3, "num": 2, "factor": 1},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="ttk_adder",
        specializations=(
            _specialization(
                id="n-3",
                inputs={"n": 3},
                substitutions={"n": 3},
            ),
        ),
    ),
    CatalogResourceCase(
        catalog_id="cdkm_adder",
        specializations=(
            _specialization(
                id="n-4",
                inputs={"n": 4},
                substitutions={"n": 4},
            ),
        ),
    ),
)


# Populate this tuple when a catalog route intentionally raises.  Keeping
# failures separate prevents successful resource expectations from carrying
# exception-only fields.
CATALOG_RESOURCE_ERROR_CASES: tuple[CatalogResourceErrorCase, ...] = ()
