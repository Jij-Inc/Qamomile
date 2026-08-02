"""Tests for directly composable opaque Oracle transforms."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.callable import CallTransform, InvokeOperation
from qamomile.circuit.serialization import deserialize, serialize


def _only_invoke(kernel: object) -> InvokeOperation:
    """Return the only invocation in a test kernel.

    Args:
        kernel (object): Qkernel-like object exposing a built block.

    Returns:
        InvokeOperation: The kernel's single invocation.
    """
    operations = [
        operation
        for operation in kernel.block.operations  # type: ignore[attr-defined]
        if isinstance(operation, InvokeOperation)
    ]
    assert len(operations) == 1
    return operations[0]


def _partially_bound_same_named_cost() -> tuple[qmc.ResourceEstimate, sp.Symbol]:
    """Return a gate cost whose surviving parameter has public alias ``n__2``."""
    positive_n = sp.Symbol("n", integer=True, positive=True)
    nonnegative_n = sp.Symbol("n", integer=True, nonnegative=True)
    cost = qmc.ResourceEstimate(
        gates=qmc.GateResources(
            total=positive_n + nonnegative_n,
            single_qubit=positive_n + nonnegative_n,
        ),
    ).substitute(n=1)
    return cost, cost.parameters["n__2"]


def test_inverse_oracle_emits_one_inverse_invoke() -> None:
    """Direct inverse keeps the original opaque definition and base cost."""
    cost = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=2, single_qubit=2),
    )
    oracle = qmc.opaque("direct_inverse", num_qubits=1, cost=cost)

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        """Apply a bodyless Oracle inverse directly."""
        target = qmc.qubit("target")
        (target,) = qmc.inverse(oracle)(target)
        return target

    operation = _only_invoke(circuit)

    assert operation.transform is CallTransform.INVERSE
    assert operation.definition is not None
    assert operation.definition.opaque_cost is cost
    assert operation.num_declared_control_qubits == 0
    assert operation.num_added_control_qubits == 0
    assert circuit.estimate_resources().gates == cost.gates


@pytest.mark.parametrize("uses_callback", [False, True], ids=["fixed", "callback"])
def test_inverse_oracle_preserves_partial_public_symbol_alias(
    uses_callback: bool,
) -> None:
    """Fixed and callback costs retain aliases through repeat and inverse."""
    cost, remaining = _partially_bound_same_named_cost()

    def callback(ctx: qmc.OpaqueCostContext) -> qmc.ResourceEstimate:
        """Return the definition-level partially bound cost.

        Args:
            ctx (qmc.OpaqueCostContext): Definition-level opaque-call context.

        Returns:
            qmc.ResourceEstimate: Partially bound ordinary definition cost.
        """
        assert ctx.definition_control_qubits == 0
        return cost

    oracle = qmc.opaque(
        f"aliased_inverse_{uses_callback}",
        num_qubits=1,
        cost=callback if uses_callback else cost,
    )

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        """Apply an inverse bodyless Oracle with a symbolic cost."""
        (target,) = qmc.inverse(oracle)(qmc.qubit("target"))
        return target

    estimate = circuit.estimate_resources()

    assert estimate.parameters == {"n__2": remaining}
    assert estimate.substitute(n__2=3).gates.total == 4


def test_declared_open_oracle_control_preserves_partial_public_symbol_alias() -> None:
    """Open-control X brackets cannot rename a surviving cost parameter."""
    cost, remaining = _partially_bound_same_named_cost()
    oracle = qmc.opaque(
        "aliased_declared_open",
        num_qubits=1,
        num_control_qubits=1,
        cost=cost,
    )

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
        """Apply one definition-level open-controlled Oracle."""
        return oracle(
            qmc.qubit("target"),
            controls=(qmc.qubit("declared"),),
            control_value=0,
        )

    estimate = circuit.estimate_resources()

    assert estimate.parameters == {"n__2": remaining}
    specialized = estimate.substitute(n__2=3)
    assert specialized.gates.total == 6
    assert specialized.gates.single_qubit == 6


@pytest.mark.parametrize("inverse_first", [False, True])
def test_control_and_inverse_oracle_orders_share_one_transformed_invoke(
    inverse_first: bool,
) -> None:
    """Both transform orders preserve the Oracle D/A control partition."""
    oracle = qmc.opaque(
        "ordered_transform",
        num_qubits=1,
        num_control_qubits=1,
        cost=qmc.ResourceEstimate(
            gates=qmc.GateResources(total=1, single_qubit=1),
        ),
    )
    transformed = (
        qmc.control(
            qmc.inverse(oracle),
            num_controls=2,
            control_value=1,
        )
        if inverse_first
        else qmc.inverse(
            qmc.control(
                oracle,
                num_controls=2,
                control_value=1,
            )
        )
    )

    @qmc.qkernel
    def circuit() -> tuple[
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Apply the transformed Oracle to two added and one declared control."""
        added_0 = qmc.qubit("added_0")
        added_1 = qmc.qubit("added_1")
        declared = qmc.qubit("declared")
        target = qmc.qubit("target")
        return transformed(added_0, added_1, declared, target)

    operation = _only_invoke(circuit)

    assert operation.transform is CallTransform.CONTROLLED_INVERSE
    assert operation.num_control_qubits == 3
    assert operation.num_declared_control_qubits == 1
    assert operation.num_added_control_qubits == 2
    assert operation.control_value == 5
    assert operation.definition is not None
    assert operation.definition.attrs["num_control_qubits"] == 1
    assert operation.definition.attrs["num_added_control_qubits"] == 0


def test_nested_oracle_controls_compose_lsb_first_patterns() -> None:
    """New control groups prepend without losing an existing open pattern."""
    oracle = qmc.opaque(
        "nested_controls",
        num_qubits=1,
        num_control_qubits=1,
    )
    transformed = qmc.control(
        qmc.control(
            qmc.inverse(oracle),
            num_controls=2,
            control_value=1,
        ),
        num_controls=1,
        control_value=0,
    )

    @qmc.qkernel
    def circuit() -> tuple[
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Apply three added controls around one declared control."""
        added_outer = qmc.qubit("added_outer")
        added_inner_0 = qmc.qubit("added_inner_0")
        added_inner_1 = qmc.qubit("added_inner_1")
        declared = qmc.qubit("declared")
        target = qmc.qubit("target")
        return transformed(
            added_outer,
            added_inner_0,
            added_inner_1,
            declared,
            target,
        )

    operation = _only_invoke(circuit)

    assert operation.transform is CallTransform.CONTROLLED_INVERSE
    assert operation.num_declared_control_qubits == 1
    assert operation.num_added_control_qubits == 3
    assert operation.control_value == 10


def test_controlled_oracle_keeps_declared_open_pattern_separate() -> None:
    """A call-site D pattern combines with the A pattern without ambiguity."""
    oracle = qmc.opaque(
        "declared_open_transform",
        num_qubits=1,
        num_control_qubits=2,
    )
    transformed = qmc.control(oracle, num_controls=2, control_value=1)

    @qmc.qkernel
    def circuit() -> tuple[
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Apply A=01 and D=10 as separate LSB-first control groups."""
        return transformed(
            qmc.qubit("added_0"),
            qmc.qubit("added_1"),
            qmc.qubit("declared_0"),
            qmc.qubit("declared_1"),
            qmc.qubit("target"),
            declared_control_value=2,
        )

    operation = _only_invoke(circuit)

    assert operation.num_added_control_qubits == 2
    assert operation.num_declared_control_qubits == 2
    assert operation.control_value == 9


def test_opaque_callback_sees_only_definition_controls_after_composition() -> None:
    """A callback runs once at D while A and inherited K are projected later."""
    observed: list[qmc.OpaqueCostContext] = []

    def callback(ctx: qmc.OpaqueCostContext) -> qmc.ResourceEstimate:
        """Record the definition-level context and return one base gate.

        Args:
            ctx (qmc.OpaqueCostContext): Definition-level Oracle context.

        Returns:
            qmc.ResourceEstimate: One ordinary application of the Oracle.
        """
        observed.append(ctx)
        return qmc.ResourceEstimate(
            gates=qmc.GateResources(total=1, two_qubit=1),
            calls=qmc.CallResources(
                queries_by_name={"callback_transform": 1},
            ),
            basis=ctx.basis,
            control_decomposition=ctx.control_decomposition,
            precision=ctx.precision,
        )

    oracle = qmc.opaque(
        "callback_transform",
        num_qubits=1,
        num_control_qubits=1,
        cost=callback,
    )
    transformed = qmc.control(qmc.inverse(oracle), num_controls=2)
    fixed_oracle = qmc.opaque(
        "fixed_transform",
        num_qubits=1,
        num_control_qubits=1,
        cost=qmc.ResourceEstimate(
            gates=qmc.GateResources(total=1, two_qubit=1),
            calls=qmc.CallResources(
                queries_by_name={"fixed_transform": 1},
            ),
        ),
    )
    fixed_transformed = qmc.control(qmc.inverse(fixed_oracle), num_controls=2)

    @qmc.qkernel
    def layer(
        added_0: qmc.Qubit,
        added_1: qmc.Qubit,
        declared: qmc.Qubit,
        target: qmc.Qubit,
    ) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit, qmc.Qubit]:
        """Apply the Oracle with its definition and call-site controls."""
        return transformed(added_0, added_1, declared, target)

    @qmc.qkernel
    def circuit() -> tuple[
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Add one inherited control around the transformed Oracle."""
        return qmc.control(layer)(
            qmc.qubit("inherited"),
            qmc.qubit("added_0"),
            qmc.qubit("added_1"),
            qmc.qubit("declared"),
            qmc.qubit("target"),
        )

    @qmc.qkernel
    def fixed_layer(
        added_0: qmc.Qubit,
        added_1: qmc.Qubit,
        declared: qmc.Qubit,
        target: qmc.Qubit,
    ) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit, qmc.Qubit]:
        """Apply the fixed-cost Oracle under the same A controls."""
        return fixed_transformed(added_0, added_1, declared, target)

    @qmc.qkernel
    def fixed_circuit() -> tuple[
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Add the same inherited K control around the fixed-cost Oracle."""
        return qmc.control(fixed_layer)(
            qmc.qubit("inherited"),
            qmc.qubit("added_0"),
            qmc.qubit("added_1"),
            qmc.qubit("declared"),
            qmc.qubit("target"),
        )

    estimate = circuit.estimate_resources()
    fixed_estimate = fixed_circuit.estimate_resources()
    expected = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1, two_qubit=1),
    ).controlled(3)

    assert len(observed) == 1
    assert observed[0].definition_control_qubits == 1
    assert estimate.gates == expected.gates
    assert fixed_estimate.gates == expected.gates
    assert estimate.width.clean_ancilla_qubits == expected.width.clean_ancilla_qubits
    assert (
        fixed_estimate.width.clean_ancilla_qubits == expected.width.clean_ancilla_qubits
    )
    assert estimate.calls.queries_by_name == {"callback_transform": 1}
    assert fixed_estimate.calls.queries_by_name == {"fixed_transform": 1}


@pytest.mark.parametrize("inverse_first", [False, True])
def test_nonunitary_oracle_cost_fails_closed_for_both_transform_orders(
    inverse_first: bool,
) -> None:
    """Neither direct transform order can invert or control a measurement cost."""
    oracle = qmc.opaque(
        "nonunitary_transform",
        num_qubits=1,
        cost=qmc.ResourceEstimate(
            measurements=qmc.MeasurementResources(total=1),
        ),
    )
    transformed = (
        qmc.control(qmc.inverse(oracle))
        if inverse_first
        else qmc.inverse(qmc.control(oracle))
    )

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
        """Apply a transformed bodyless non-unitary cost."""
        return transformed(
            qmc.qubit("control"),
            qmc.qubit("target"),
        )

    with pytest.raises(ValueError, match="non-unitary"):
        circuit.estimate_resources()


def test_transformed_oracle_round_trip_preserves_transform_and_cost() -> None:
    """Serialization retains one CONTROLLED_INVERSE call and its fixed cost."""
    oracle = qmc.opaque(
        "serialized_transform",
        num_qubits=1,
        num_control_qubits=1,
        cost=qmc.ResourceEstimate(
            gates=qmc.GateResources(total=3, single_qubit=3),
        ),
    )
    transformed = qmc.control(qmc.inverse(oracle), num_controls=2)

    @qmc.qkernel
    def circuit() -> tuple[
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
        qmc.Qubit,
    ]:
        """Apply the transformed Oracle before serialization."""
        return transformed(
            qmc.qubit("added_0"),
            qmc.qubit("added_1"),
            qmc.qubit("declared"),
            qmc.qubit("target"),
        )

    restored = deserialize(serialize(circuit))
    operation = _only_invoke(restored)

    assert operation.transform is CallTransform.CONTROLLED_INVERSE
    assert operation.num_declared_control_qubits == 1
    assert operation.num_added_control_qubits == 2
    assert qmc.estimate_resources(restored).gates == circuit.estimate_resources().gates


def test_serialized_oracles_keep_independent_same_named_cost_symbols() -> None:
    """Round-tripped Oracle definitions retain separate cost namespaces."""
    left_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    right_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    left_oracle = qmc.opaque(
        "serialized_left_symbol",
        num_qubits=1,
        cost=qmc.ResourceEstimate(
            gates=qmc.GateResources(total=left_symbol),
        ),
    )
    right_oracle = qmc.opaque(
        "serialized_right_symbol",
        num_qubits=1,
        cost=qmc.ResourceEstimate(
            gates=qmc.GateResources(total=right_symbol),
        ),
    )

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
        """Apply independently costed bodyless Oracles."""
        (left,) = left_oracle(qmc.qubit("left"))
        (right,) = right_oracle(qmc.qubit("right"))
        return left, right

    restored = deserialize(serialize(circuit))
    estimate = qmc.estimate_resources(restored)

    assert set(estimate.parameters) == {"n", "n__2"}
    assert estimate.substitute(n=2, n__2=5).gates.total == 7


def test_serialized_repeated_oracle_keeps_one_shared_cost_symbol() -> None:
    """Repeated calls retain one fixed Oracle's shared model parameter."""
    shared_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    oracle = qmc.opaque(
        "serialized_shared_symbol",
        num_qubits=1,
        cost=qmc.ResourceEstimate(
            gates=qmc.GateResources(total=shared_symbol),
        ),
    )

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
        """Invoke one symbolically costed Oracle twice."""
        (left,) = oracle(qmc.qubit("left"))
        (right,) = oracle(qmc.qubit("right"))
        return left, right

    restored = deserialize(serialize(circuit))
    estimate = qmc.estimate_resources(restored)

    assert set(estimate.parameters) == {"n"}
    assert estimate.substitute(n=3).gates.total == 6


def test_serialized_oracles_share_cost_symbol_across_local_aliases() -> None:
    """Payload-wide slots preserve a symbol despite definition-local aliases."""
    shadow_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    shared_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    left_oracle = qmc.opaque(
        "serialized_alias_collision_left",
        num_qubits=1,
        cost=qmc.ResourceEstimate(
            gates=qmc.GateResources(total=shadow_symbol + shared_symbol),
        ),
    )
    right_oracle = qmc.opaque(
        "serialized_alias_collision_right",
        num_qubits=1,
        cost=qmc.ResourceEstimate(
            gates=qmc.GateResources(total=shared_symbol),
        ),
    )

    @qmc.qkernel
    def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
        """Invoke definitions that assign different local aliases to one symbol."""
        (left,) = left_oracle(qmc.qubit("left"))
        (right,) = right_oracle(qmc.qubit("right"))
        return left, right

    restored = deserialize(serialize(circuit))
    estimate = qmc.estimate_resources(restored)

    assert set(estimate.parameters) == {"n", "n__2"}
    assert estimate.substitute(n=2, n__2=5).gates.total == 12


def test_serialized_oracle_cost_symbol_stays_distinct_from_kernel_input() -> None:
    """A fixed-cost parameter does not merge with a same-named qkernel input."""
    cost_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    oracle = qmc.opaque(
        "serialized_input_collision",
        num_qubits=1,
        cost=qmc.ResourceEstimate(
            gates=qmc.GateResources(total=cost_symbol),
        ),
    )

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Qubit:
        """Repeat X before invoking a same-named symbolic-cost Oracle."""
        target = qmc.qubit("target")
        for _ in qmc.range(n):
            target = qmc.x(target)
        (target,) = oracle(target)
        return target

    restored = deserialize(serialize(circuit))
    estimate = qmc.estimate_resources(restored)

    assert set(estimate.parameters) == {"n", "n__2"}
    assert estimate.substitute(n=2, n__2=5).gates.total == 7


def test_inverse_oracle_preserves_vector_call_shape() -> None:
    """Inverse-only Oracle wrappers return a Vector for vector input."""
    oracle = qmc.opaque("inverse_vector", num_qubits=2)

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        """Apply one inverse Oracle to a two-qubit vector."""
        targets = qmc.qubit_array(2, "targets")
        return qmc.inverse(oracle)(targets)

    operation = _only_invoke(circuit)

    assert operation.transform is CallTransform.INVERSE
    assert operation.num_target_qubits == 2


def test_double_inverse_oracle_returns_original_definition() -> None:
    """Two direct inverse transforms cancel to the original Oracle object."""
    oracle = qmc.opaque("double_inverse", num_qubits=1)

    assert qmc.inverse(qmc.inverse(oracle)) is oracle
