"""Resource-estimation coverage for explicit Suzuki–Trotter algorithms."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.trotter import (
    _trotter_evolve,
    trotterized_time_evolution,
)
from qamomile.circuit.estimator import estimate_resources
from qamomile.circuit.estimator._product_formula import _suzuki_trotter_contract
from qamomile.circuit.serialization import deserialize, serialize


@qmc.qkernel
def _trotter_resource_circuit(
    hamiltonian: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
    step: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the public Trotter helper to one allocated qubit."""
    qubits = qmc.qubit_array(1, name="qubits")
    return trotterized_time_evolution(
        qubits,
        hamiltonian,
        order,
        gamma,
        step,
    )


@qmc.qkernel
def _controlled_trotter_resource_circuit(
    terms: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply controlled Trotter evolution with compile-time structure."""
    control = qmc.qubit("control")
    qubits = qmc.qubit_array(1, name="qubits")
    controlled = qmc.control(_trotter_evolve, num_controls=1)
    control, qubits = controlled(
        control,
        qubits,
        terms,
        order,
        gamma,
        2,
    )
    return qubits


@qmc.qkernel
def _selected_trotter_resource_circuit(
    terms: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply Trotter evolution as each branch of a SELECT operation."""
    index = qmc.qubit("index")
    qubits = qmc.qubit_array(1, name="qubits")
    index, qubits = qmc.select([_trotter_evolve, _trotter_evolve])(
        index,
        qubits,
        terms,
        order,
        gamma,
        1,
    )
    return qubits


@qmc.qkernel
def _sliced_trotter_resource_circuit(
    components: qmc.Vector[qmc.Observable],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Evolve under a concrete view of a larger Hamiltonian vector."""
    qubits = qmc.qubit_array(1, name="qubits")
    selected = components[0:2]
    return trotterized_time_evolution(qubits, selected, 2, gamma, 1)


@qmc.qkernel
def _dynamic_sliced_trotter_resource_circuit(
    components: qmc.Vector[qmc.Observable],
    start: qmc.UInt,
    stop: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Evolve under a view whose bounds come from estimator inputs."""
    qubits = qmc.qubit_array(1, name="qubits")
    selected = components[start:stop]
    return trotterized_time_evolution(qubits, selected, 2, gamma, 1)


@qmc.qkernel
def _computed_slice_trotter_helper(
    qubits: qmc.Vector[qmc.Qubit],
    terms: qmc.Vector[qmc.Observable],
    start: qmc.UInt,
    stop: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply Trotter evolution to a caller-computed component view."""
    selected = terms[start:stop]
    return trotterized_time_evolution(qubits, selected, 2, gamma, 1)


@qmc.qkernel
def _computed_slice_trotter_resource_circuit(
    components: qmc.Vector[qmc.Observable],
    selector: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Pass computed view bounds through an ordinary nested qkernel."""
    qubits = qmc.qubit_array(1, name="qubits")
    return _computed_slice_trotter_helper(
        qubits,
        components,
        selector + 1,
        selector + 3,
        gamma,
    )


@qmc.qkernel
def _controlled_dynamic_slice_trotter_resource_circuit(
    components: qmc.Vector[qmc.Observable],
    start: qmc.UInt,
    stop: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Pass a dynamic component view through a controlled call boundary."""
    control = qmc.qubit("control")
    qubits = qmc.qubit_array(1, name="qubits")
    selected = components[start:stop]
    control, qubits = qmc.control(_trotter_evolve)(
        control,
        qubits,
        selected,
        2,
        gamma,
        1,
    )
    return qubits


@qmc.qkernel
def _derived_order_trotter_resource_circuit(
    components: qmc.Vector[qmc.Observable],
    base_order: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Derive the concrete Suzuki order from another scalar input."""
    qubits = qmc.qubit_array(1, name="qubits")
    return trotterized_time_evolution(
        qubits,
        components,
        base_order + 2,
        gamma,
        1,
    )


@qmc.qkernel
def _conditionally_inactive_trotter_resource_circuit(
    enabled: qmc.UInt,
    components: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Keep Trotter structure in a compile-time inactive branch."""
    qubits = qmc.qubit_array(1, name="qubits")
    if enabled == 1:
        qubits = trotterized_time_evolution(
            qubits,
            components,
            order,
            gamma,
            1,
        )
    else:
        qubits[0] = qmc.x(qubits[0])
    return qubits


@qmc.qkernel
def _computed_inactive_trotter_helper(
    qubits: qmc.Vector[qmc.Qubit],
    enabled: qmc.Bit,
    components: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply Trotter evolution only when a computed caller flag is active."""
    if enabled:
        qubits = trotterized_time_evolution(
            qubits,
            components,
            order,
            gamma,
            1,
        )
    else:
        qubits[0] = qmc.x(qubits[0])
    return qubits


@qmc.qkernel
def _computed_inactive_trotter_resource_circuit(
    selector: qmc.UInt,
    components: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Pass one computed Boolean condition through a nested callable."""
    qubits = qmc.qubit_array(1, name="qubits")
    return _computed_inactive_trotter_helper(
        qubits,
        selector == 0,
        components,
        order,
        gamma,
    )


@qmc.qkernel
def _zero_power_controlled_trotter_resource_circuit(
    components: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Keep a Trotter callable behind an identity controlled operation."""
    control = qmc.qubit("control")
    qubits = qmc.qubit_array(1, name="qubits")
    control, qubits = qmc.control(_trotter_evolve)(
        control,
        qubits,
        components,
        order,
        gamma,
        1,
        power=0,
    )
    return qubits


@qmc.qkernel
def _zero_trip_trotter_resource_circuit(
    iterations: qmc.UInt,
    components: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Keep a Trotter call in a concretely empty range loop."""
    qubits = qmc.qubit_array(1, name="qubits")
    for _index in qmc.range(iterations):
        qubits = trotterized_time_evolution(
            qubits,
            components,
            order,
            gamma,
            1,
        )
    return qubits


@qmc.qkernel
def _legacy_bit_with_trotter_resource_circuit(
    iterations: qmc.UInt,
    enabled: qmc.UInt,
    zero_iterations: qmc.UInt,
    components: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Combine legacy Bit carry validation with inactive Trotter paths."""
    predicate = qmc.bit(False)
    source = qmc.qubit("source")
    qubits = qmc.qubit_array(1, name="qubits")
    for _index in qmc.range(iterations):
        if predicate:
            qubits[0] = qmc.x(qubits[0])
        predicate = qmc.measure(source)
    if enabled == 1:
        qubits = trotterized_time_evolution(
            qubits,
            components,
            order,
            gamma,
            1,
        )
    control = qmc.qubit("control")
    control, qubits = qmc.control(_trotter_evolve)(
        control,
        qubits,
        components,
        order,
        gamma,
        1,
        power=0,
    )
    for _index in qmc.range(zero_iterations):
        qubits = trotterized_time_evolution(
            qubits,
            components,
            order,
            gamma,
            1,
        )
    return qubits


NONCOMMUTING_COMPONENTS = [qm_o.Z(0), qm_o.X(0)]


@pytest.mark.parametrize(
    ("order", "gates_per_step", "rotations_per_step", "cliffords_per_step"),
    (
        (1, 4, 2, 2),
        (2, 5, 3, 2),
        (4, 25, 15, 10),
    ),
)
def test_trotter_resources_follow_order_and_step_count(
    order: int,
    gates_per_step: int,
    rotations_per_step: int,
    cliffords_per_step: int,
) -> None:
    """Every concrete component is counted in each explicit formula step."""
    steps = 3
    estimate = _trotter_resource_circuit.estimate_resources(
        inputs={
            "hamiltonian": NONCOMMUTING_COMPONENTS,
            "order": order,
            "gamma": 1.0,
            "step": steps,
        }
    )

    assert estimate.gates.total == gates_per_step * steps
    assert estimate.gates.single_qubit == gates_per_step * steps
    assert estimate.gates.rotation == rotations_per_step * steps
    assert estimate.gates.clifford == cliffords_per_step * steps
    assert estimate.depth.depth == gates_per_step * steps
    assert estimate.width.allocated_qubits == 1
    assert estimate.width.peak_qubits == 1
    assert estimate.width.clean_ancilla_qubits == 0
    assert estimate.width.dirty_ancilla_qubits == 0
    assert estimate.derivation is qmc.EstimateDerivation.STRUCTURAL
    assert estimate.approximation is qmc.ApproximationStatus.APPROXIMATE


def test_symbolic_trotter_step_count_stays_compact() -> None:
    """The outer step loop is multiplied symbolically, not replayed."""
    estimate = _trotter_resource_circuit.estimate_resources(
        inputs={
            "hamiltonian": NONCOMMUTING_COMPONENTS,
            "order": 2,
            "gamma": 1.0,
        }
    )

    step = estimate.parameters["step"]
    assert sp.simplify(estimate.gates.total - 5 * step) == 0
    assert sp.simplify(estimate.depth.depth - 5 * step) == 0
    concrete = estimate.substitute(step=3)
    assert concrete.gates.total == 15
    assert concrete.depth.depth == 15


def test_trotter_approximation_distinguishes_exact_special_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inner and outer product-formula approximations remain distinguishable."""
    with monkeypatch.context() as patch:
        patch.setattr(
            qm_o,
            "commutator",
            lambda _left, _right: pytest.fail(
                "single-basis components must take the linear commutation path"
            ),
        )
        commuting = _trotter_resource_circuit.estimate_resources(
            inputs={
                "hamiltonian": [qm_o.Z(0), 2.0 * qm_o.Z(0)],
                "order": 2,
                "gamma": 1.0,
                "step": 3,
            }
        )
    zero_time = _trotter_resource_circuit.estimate_resources(
        inputs={
            "hamiltonian": NONCOMMUTING_COMPONENTS,
            "order": 2,
            "gamma": 0.0,
            "step": 3,
        }
    )
    outer_only = _trotter_resource_circuit.estimate_resources(
        inputs={
            "hamiltonian": NONCOMMUTING_COMPONENTS,
            "order": 2,
            "gamma": 1.0,
            "step": 3,
        }
    )
    inner_only = _trotter_resource_circuit.estimate_resources(
        inputs={
            "hamiltonian": [
                qm_o.Z(0) + qm_o.X(0),
                2.0 * (qm_o.Z(0) + qm_o.X(0)),
            ],
            "order": 2,
            "gamma": 1.0,
            "step": 3,
        }
    )
    inner_and_outer = _trotter_resource_circuit.estimate_resources(
        inputs={
            "hamiltonian": [
                qm_o.Z(0) + qm_o.X(0),
                qm_o.Z(0) + qm_o.Y(0),
            ],
            "order": 2,
            "gamma": 1.0,
            "step": 3,
        }
    )

    def messages(estimate: qmc.ResourceEstimate) -> tuple[str, ...]:
        """Return assumption messages for one estimate.

        Args:
            estimate (qmc.ResourceEstimate): Estimate to inspect.

        Returns:
            tuple[str, ...]: Assumption message strings.
        """
        return tuple(assumption.message for assumption in estimate.assumptions)

    assert commuting.approximation is qmc.ApproximationStatus.EXACT
    assert zero_time.approximation is qmc.ApproximationStatus.EXACT
    assert zero_time.gates.total == 0
    assert any("component Hamiltonians" in message for message in messages(outer_only))
    assert not any("Pauli terms" in message for message in messages(outer_only))
    assert any("Pauli terms" in message for message in messages(inner_only))
    assert not any(
        "component Hamiltonians" in message for message in messages(inner_only)
    )
    assert any("Pauli terms" in message for message in messages(inner_and_outer))
    assert any(
        "component Hamiltonians" in message for message in messages(inner_and_outer)
    )


def test_trotter_structure_resolution_handles_expressions_and_views() -> None:
    """Structural inputs resolve through arithmetic and sliced object arrays."""
    derived = _derived_order_trotter_resource_circuit.estimate_resources(
        inputs={
            "components": NONCOMMUTING_COMPONENTS,
            "base_order": 2,
            "gamma": 1.0,
        }
    )
    sliced = _sliced_trotter_resource_circuit.estimate_resources(
        inputs={
            "components": np.asarray(
                [qm_o.Z(0), 2.0 * qm_o.Z(0), qm_o.X(0)],
                dtype=object,
            ),
            "gamma": 1.0,
        }
    )
    dynamic_sliced = _dynamic_sliced_trotter_resource_circuit.estimate_resources(
        inputs={
            "components": np.asarray(
                [qm_o.Z(0), 2.0 * qm_o.Z(0), qm_o.X(0)],
                dtype=object,
            ),
            "start": 0,
            "stop": 2,
            "gamma": 1.0,
        }
    )
    nested_computed_slice = _computed_slice_trotter_resource_circuit.estimate_resources(
        inputs={
            "components": np.asarray(
                [qm_o.Z(0), 2.0 * qm_o.Z(0), qm_o.X(0)],
                dtype=object,
            ),
            "selector": 0,
            "gamma": 1.0,
        }
    )
    controlled_dynamic_slice = (
        _controlled_dynamic_slice_trotter_resource_circuit.estimate_resources(
            inputs={
                "components": np.asarray(
                    [qm_o.Z(0), 2.0 * qm_o.Z(0), qm_o.X(0)],
                    dtype=object,
                ),
                "start": 0,
                "stop": 2,
                "gamma": 1.0,
            }
        )
    )
    sequence = estimate_resources(
        _derived_order_trotter_resource_circuit.block.operations,
        inputs={
            "components": NONCOMMUTING_COMPONENTS,
            "base_order": 2,
            "gamma": 1.0,
        },
    )

    assert derived.gates.total == 25
    assert derived.approximation is qmc.ApproximationStatus.APPROXIMATE
    assert sliced.gates.total == 3
    assert sliced.approximation is qmc.ApproximationStatus.EXACT
    assert dynamic_sliced.gates.total == 3
    assert dynamic_sliced.approximation is qmc.ApproximationStatus.EXACT
    assert nested_computed_slice.gates.total == 5
    assert nested_computed_slice.approximation is qmc.ApproximationStatus.APPROXIMATE
    assert controlled_dynamic_slice.gates.total == 3
    assert controlled_dynamic_slice.approximation is qmc.ApproximationStatus.EXACT
    assert sequence.gates.total == 25


@pytest.mark.parametrize(
    ("order", "step", "message"),
    (
        (0, 1, "Suzuki-Trotter order"),
        (3, 1, "odd Suzuki-Trotter order"),
        (2, 0, "Suzuki-Trotter step count"),
    ),
)
def test_trotter_resource_contract_validates_algorithm_inputs(
    order: int,
    step: int,
    message: str,
) -> None:
    """Estimator inputs obey the same order and step domain as transpilation."""
    with pytest.raises(ValueError, match=message):
        _trotter_resource_circuit.estimate_resources(
            inputs={
                "hamiltonian": NONCOMMUTING_COMPONENTS,
                "order": order,
                "gamma": 1.0,
                "step": step,
            }
        )


def test_suzuki_contract_validation_belongs_to_estimator() -> None:
    """Generic IR roles become mandatory only for the recognized family."""
    incomplete = {
        "resource_contract": {
            "product_formula": {
                "kind": "suzuki_trotter",
                "hamiltonian_operand": 1,
            }
        }
    }

    with pytest.raises(ValueError, match="order_operand must be"):
        _suzuki_trotter_contract(
            incomplete,
            source="incomplete_trotter",
            operand_count=5,
        )


def test_estimator_rejects_unrecognized_product_formula_family() -> None:
    """Unknown families cannot silently report exact Suzuki semantics."""
    attrs = {
        "resource_contract": {
            "product_formula": {
                "kind": "custom_formula",
                "generator_operand": 0,
            }
        }
    }

    with pytest.raises(ValueError, match="kind must be 'suzuki_trotter'"):
        _suzuki_trotter_contract(
            attrs,
            source="custom_formula",
            operand_count=1,
        )


def test_suzuki_contract_rejects_duplicate_operand_roles() -> None:
    """Suzuki-specific roles must address distinct callable operands."""
    attrs = {
        "resource_contract": {
            "product_formula": {
                "kind": "suzuki_trotter",
                "hamiltonian_operand": 1,
                "order_operand": 2,
                "time_operand": 3,
                "steps_operand": 3,
            }
        }
    }

    with pytest.raises(ValueError, match="operand positions must be distinct"):
        _suzuki_trotter_contract(
            attrs,
            source="invalid_trotter",
            operand_count=5,
        )


def test_trotter_resource_estimation_requires_concrete_order() -> None:
    """Every callable boundary rejects order before recursive inlining."""
    restored = deserialize(serialize(_trotter_resource_circuit))
    cases = (
        (
            _trotter_resource_circuit,
            {
                "hamiltonian": NONCOMMUTING_COMPONENTS,
                "gamma": 1.0,
                "step": 2,
            },
        ),
        (
            _trotter_resource_circuit.block,
            {
                "hamiltonian": NONCOMMUTING_COMPONENTS,
                "gamma": 1.0,
                "step": 2,
            },
        ),
        (
            _trotter_resource_circuit.block.operations,
            {
                "hamiltonian": NONCOMMUTING_COMPONENTS,
                "gamma": 1.0,
                "step": 2,
            },
        ),
        (
            _trotter_evolve,
            {
                "q": 1,
                "hamiltonian": NONCOMMUTING_COMPONENTS,
                "gamma": 1.0,
                "step": 2,
            },
        ),
        (
            _controlled_trotter_resource_circuit,
            {
                "terms": NONCOMMUTING_COMPONENTS,
                "gamma": 1.0,
            },
        ),
        (
            _selected_trotter_resource_circuit,
            {
                "terms": NONCOMMUTING_COMPONENTS,
                "gamma": 1.0,
            },
        ),
        (
            restored,
            {
                "hamiltonian": NONCOMMUTING_COMPONENTS,
                "gamma": 1.0,
                "step": 2,
            },
        ),
        (
            _legacy_bit_with_trotter_resource_circuit,
            {
                "iterations": 1,
                "enabled": 1,
                "zero_iterations": 0,
                "components": NONCOMMUTING_COMPONENTS,
                "gamma": 1.0,
            },
        ),
    )

    for kernel, inputs in cases:
        with pytest.raises(ValueError, match="requires product-formula order"):
            estimate_resources(kernel, inputs=inputs)


def test_trotter_contract_survives_serialization_and_control() -> None:
    """Callable metadata remains active after persistence and transforms."""
    restored = deserialize(serialize(_trotter_resource_circuit))
    inputs = {
        "hamiltonian": NONCOMMUTING_COMPONENTS,
        "order": 2,
        "gamma": 1.0,
        "step": 2,
    }

    restored_estimate = estimate_resources(restored, inputs=inputs, trace=True)
    controlled_estimate = _controlled_trotter_resource_circuit.estimate_resources(
        inputs={
            "terms": NONCOMMUTING_COMPONENTS,
            "order": 2,
            "gamma": 1.0,
        }
    )
    selected_estimate = _selected_trotter_resource_circuit.estimate_resources(
        inputs={
            "terms": NONCOMMUTING_COMPONENTS,
            "order": 2,
            "gamma": 1.0,
        }
    )

    assert restored_estimate.gates.total == 10
    assert restored_estimate.approximation is qmc.ApproximationStatus.APPROXIMATE
    assert controlled_estimate.approximation is qmc.ApproximationStatus.APPROXIMATE
    assert selected_estimate.approximation is qmc.ApproximationStatus.APPROXIMATE
    assert controlled_estimate.gates.total == 10
    assert controlled_estimate.gates.two_qubit == 6
    assert restored_estimate.trace is not None
    assert "finite-step Suzuki-Trotter" in restored_estimate.trace.render()


def test_inactive_trotter_branch_does_not_require_its_structure_inputs() -> None:
    """Inactive evaluation paths do not require product-formula structure."""
    estimate = _conditionally_inactive_trotter_resource_circuit.estimate_resources(
        inputs={
            "enabled": 0,
            "components": NONCOMMUTING_COMPONENTS,
            "gamma": 1.0,
        }
    )
    sequence = estimate_resources(
        _conditionally_inactive_trotter_resource_circuit.block.operations,
        inputs={
            "enabled": 0,
        },
    )
    nested = _computed_inactive_trotter_resource_circuit.estimate_resources(
        inputs={
            "selector": 1,
            "components": NONCOMMUTING_COMPONENTS,
            "gamma": 1.0,
        }
    )
    zero_power = _zero_power_controlled_trotter_resource_circuit.estimate_resources(
        inputs={
            "components": NONCOMMUTING_COMPONENTS,
            "gamma": 1.0,
        }
    )
    zero_trip = _zero_trip_trotter_resource_circuit.estimate_resources(
        inputs={
            "iterations": 0,
            "components": NONCOMMUTING_COMPONENTS,
            "gamma": 1.0,
        }
    )
    mixed = _legacy_bit_with_trotter_resource_circuit.estimate_resources(
        inputs={
            "iterations": 1,
            "enabled": 0,
            "zero_iterations": 0,
            "components": NONCOMMUTING_COMPONENTS,
            "gamma": 1.0,
        }
    )

    assert estimate.gates.total == 1
    assert estimate.approximation is qmc.ApproximationStatus.EXACT
    assert sequence.gates.total == 1
    assert nested.gates.total == 1
    assert nested.approximation is qmc.ApproximationStatus.EXACT
    assert zero_power.gates.total == 0
    assert zero_power.approximation is qmc.ApproximationStatus.EXACT
    assert zero_trip.gates.total == 0
    assert zero_trip.approximation is qmc.ApproximationStatus.EXACT
    assert mixed.gates.total == 0
    assert mixed.measurements.total == 1
    assert mixed.approximation is qmc.ApproximationStatus.EXACT
