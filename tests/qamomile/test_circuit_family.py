"""Public launch-dispatched circuit-family lowering tests."""

from __future__ import annotations

import pytest

import qamomile as qm
from qamomile.circuit import (
    ClassicalExpr,
    MeasurementBit,
    circuit_result_postprocessing,
    lower_to_circuit,
    lower_to_circuit_at,
    lower_to_circuit_family,
    postprocess_circuit_counts,
)
from qamomile.compilation import ResultABI

native = pytest.importorskip("qamomile._qamomile_native")


_BOOLEAN_DISPATCH_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool], theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    state = qm.rz(state, theta)
    return qm.measure(state)
"""


def test_public_circuit_rejects_missing_native_projection_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python never restores removed recursive projection implementations."""

    import qamomile.circuit as circuit_module

    inspect = native.inspect_static_circuit

    class BrokenProjection:
        """Hide one required field from an otherwise valid projection."""

        def __init__(self, projection: object, hidden: str) -> None:
            """Retain the valid native delegate."""

            self._projection = projection
            self._hidden = hidden

        def __getattr__(self, name: str) -> object:
            """Forward every field except the selected native fact."""

            if name == self._hidden:
                raise AttributeError(name)
            return getattr(self._projection, name)

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    return qm.measure(qm.qubits(1))
"""
    ).specialize("sample")
    projection = inspect(program.to_bytes())

    for hidden in (
        "operation_count",
        "parameter_names",
        "result_route",
        "realizations",
    ):
        monkeypatch.setattr(
            native,
            "inspect_static_circuit",
            lambda _payload, _capabilities, _lowering, hidden=hidden: BrokenProjection(
                projection, hidden
            ),
        )
        with pytest.raises(TypeError, match="native circuit projection violates"):
            circuit_module.lower_to_circuit(program)


def test_public_circuit_reuses_branch_local_fresh_wires() -> None:
    """Mutually exclusive fresh owners share one bounded physical wire range."""

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def choose() -> qm.Q[qm.Bits[1]]:
    bit = qm.measure(qm.prepare(True))
    if bit:
        return qm.qubits(1)
    return qm.qubits(1)

@qm.qkernel
def sample() -> qm.Bits[1]:
    return qm.measure(choose())
""",
        module_name="public_divergent_return_owners",
    ).specialize("sample")

    circuit = lower_to_circuit(program)

    assert circuit.width == 2
    assert qm.evaluate(program) == (0,)


def test_public_circuit_retains_measurement_result_expressions() -> None:
    """The inspection API exposes shot-time result postprocessing exactly."""

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    bit = qm.measure(qm.h(qm.prepare(True)))
    return not bit, bit
""",
        module_name="public_measurement_result_expression",
    ).specialize("sample")

    circuit = lower_to_circuit(program)

    measured = MeasurementBit(0)
    assert circuit.result_value == (
        ClassicalExpr("classical.not", (measured,)),
        measured,
    )
    assert circuit.requires_result_postprocessing
    assert circuit.result_abi == ResultABI.from_program(program)
    assert b'"result":{"tuple"' in circuit.to_bytes()


def test_public_circuit_restores_portable_result_literals() -> None:
    """Native circuit inspection preserves non-scalar portable constants."""

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def value() -> qm.Complex:
    state = qm.qubits(1)
    qm.discard(state)
    return 1.25 - 2.5j
""",
        module_name="public_portable_result_literal",
    ).specialize("value")

    circuit = lower_to_circuit(program)

    assert circuit.result_value == complex(1.25, -2.5)
    assert not circuit.requires_result_postprocessing
    assert b'"result":{"complex":[1.25,-2.5]}' in circuit.to_bytes()


def test_result_postprocessing_evaluates_raw_measurement_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One backend-neutral plan maps raw measurements to logical samples."""

    program = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bool]:
    bit, state = qm.measure_z(qm.qubits(1), 0)
    qm.discard(state)
    return bit != enabled
""",
        module_name="public_result_postprocessing",
    ).specialize("sample")
    circuit = lower_to_circuit(program)
    plan = circuit_result_postprocessing(circuit)

    assert plan is not None
    assert not plan.startswith(b"{")
    assert circuit.measurement_count == 1
    monkeypatch.setattr(
        ResultABI,
        "encode_value",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("Python result encoding dispatch was used")
        ),
    )
    assert postprocess_circuit_counts(
        plan,
        {(0,): 2, (1,): 3},
        {"enabled": True},
        circuit.result_abi,
    ) == {(1,): 2, (0,): 3}


def test_installed_result_postprocessor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale native extension cannot restore Python expression semantics."""

    abi = ResultABI(qm.Bool, (0,), 1)
    monkeypatch.delattr(native, "postprocess_circuit_counts")

    with pytest.raises(qm.ExecutionError, match="native result postprocessing"):
        postprocess_circuit_counts(b"sealed-plan", {(0,): 1}, {}, abi)


def test_public_circuit_family_selects_boolean_variants() -> None:
    """One Pythonic family selects exact circuits and removes its key ABI."""

    program = qm.build_source(
        _BOOLEAN_DISPATCH_SOURCE,
        module_name="public_boolean_dispatch_family",
    ).specialize("sample")

    family = lower_to_circuit_family(program)
    disabled = family.select({"enabled": False, "theta": 0.5})
    enabled = family.select({"enabled": True, "theta": 0.5})

    assert [parameter.name for parameter in family.dispatch_parameters] == ["enabled"]
    assert [slot.name for slot in family.parameter_abi.slots] == ["enabled", "theta"]
    assert [slot.name for slot in enabled.parameter_abi.slots] == ["theta"]
    assert [instruction.opcode for instruction in disabled.body.instructions] == [
        "rz",
        "measure",
    ]
    assert [instruction.opcode for instruction in enabled.body.instructions] == [
        "x",
        "rz",
        "measure",
    ]


def test_public_circuit_family_selects_through_rust() -> None:
    """An installed compiler never re-evaluates launch predicates in Python."""

    program = qm.build_source(
        _BOOLEAN_DISPATCH_SOURCE,
        module_name="public_native_boolean_dispatch",
    ).specialize("sample")
    family = lower_to_circuit_family(program)
    selected = family.select({"enabled": True, "theta": 0.5})

    assert [instruction.opcode for instruction in selected.body.instructions] == [
        "x",
        "rz",
        "measure",
    ]


def test_public_circuit_family_requires_native_predicate_parameter_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale bridge cannot restore Python predicate-parameter discovery."""

    program = qm.build_source(
        _BOOLEAN_DISPATCH_SOURCE,
        module_name="public_native_predicate_parameter_names",
    ).specialize("sample")
    projection = native.inspect_static_circuit_family(program.to_bytes())

    class BrokenPredicate:
        """Hide the native parameter-name projection from one predicate."""

        def __init__(self, predicate: object) -> None:
            """Retain the otherwise valid native predicate."""

            self._predicate = predicate

        def __getattr__(self, name: str) -> object:
            """Forward every field except the required parameter names."""

            if name == "parameter_names":
                raise AttributeError(name)
            return getattr(self._predicate, name)

    class BrokenFamily:
        """Replace only the predicate projection in an otherwise valid family."""

        def __getattr__(self, name: str) -> object:
            """Forward the family fields while wrapping its predicates."""

            if name == "predicates":
                return [BrokenPredicate(item) for item in projection.predicates]
            return getattr(projection, name)

    monkeypatch.setattr(
        native,
        "inspect_static_circuit_family",
        lambda _payload, _capabilities, _lowering: BrokenFamily(),
    )

    with pytest.raises(TypeError, match="native circuit-family projection violates"):
        lower_to_circuit_family(program)


def test_installed_family_selector_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale installed extension cannot restore Python dispatch semantics."""

    program = qm.build_source(
        _BOOLEAN_DISPATCH_SOURCE,
        module_name="public_stale_native_dispatch",
    ).specialize("sample")
    family = lower_to_circuit_family(program)
    monkeypatch.delattr(native, "select_circuit_family_variant")

    with pytest.raises(qm.CapabilityError, match="selection is unavailable"):
        family.select({"enabled": True, "theta": 0.5})


def test_public_circuit_family_validates_dispatch_values() -> None:
    """Selection reports missing and ill-typed host values at the family ABI."""

    program = qm.build_source(
        _BOOLEAN_DISPATCH_SOURCE,
        module_name="public_boolean_dispatch_validation",
    ).specialize("sample")
    family = lower_to_circuit_family(program)

    with pytest.raises(
        qm.CapabilityError, match="missing host-dispatch parameter"
    ) as missing:
        family.select({"theta": 0.5})
    assert missing.value.diagnostic.code == "Q7221"
    with pytest.raises(qm.CapabilityError, match="must be Bool") as mismatch:
        family.select({"enabled": 1, "theta": 0.5})
    assert mismatch.value.diagnostic.code == "Q7222"
    with pytest.raises(qm.CapabilityError, match="missing host-dispatch parameter"):
        lower_to_circuit_at(program, {"theta": 0.5})
    with pytest.raises(qm.CapabilityError, match="must be Bool"):
        lower_to_circuit_at(program, {"enabled": 1, "theta": 0.5})


def test_public_dispatch_binding_uses_native_defaults_and_ignores_other_slots() -> None:
    """Both dispatch routes bind only control values through the Rust ABI."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(
    theta: qm.Param[qm.Real],
    enabled: qm.Param[qm.Bool] = True,
) -> qm.Bits[1]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    state = qm.rz(state, theta)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="public_native_dispatch_defaults",
    ).specialize("sample")

    family = lower_to_circuit_family(program)
    selected = family.select({"ignored": object()})
    materialized = lower_to_circuit_at(program, {"ignored": object()})

    assert [slot.name for slot in selected.parameter_abi.slots] == ["theta"]
    assert [instruction.opcode for instruction in selected.body.instructions] == [
        "x",
        "rz",
        "measure",
    ]
    assert materialized == selected


def test_public_dispatch_binding_fails_closed_without_native_binder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Circuit selection cannot restore a Python type/default implementation."""

    program = qm.build_source(
        _BOOLEAN_DISPATCH_SOURCE,
        module_name="public_stale_native_dispatch_binding",
    ).specialize("sample")
    family = lower_to_circuit_family(program)
    monkeypatch.delattr(native, "bind_invocation")

    with pytest.raises(
        qm.CapabilityError, match="native invocation binding is unavailable"
    ):
        family.select({"enabled": True})


def test_public_circuit_family_selects_integer_predicate_variants() -> None:
    """An unbounded integer is evaluated once instead of value-enumerated."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(limit: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if limit > 0:
        state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="public_unbounded_dispatch_family",
    ).specialize("sample")

    family = lower_to_circuit_family(program)

    negative = family.select({"limit": -10})
    zero = family.select({"limit": 0})
    positive = family.select({"limit": 10})
    huge = family.select({"limit": 1 << 200})

    assert [parameter.name for parameter in family.dispatch_parameters] == ["limit"]
    assert len(family.dispatch_predicates) == 1
    assert [instruction.opcode for instruction in negative.body.instructions] == [
        "measure"
    ]
    assert zero == negative
    assert huge == positive
    assert [instruction.opcode for instruction in positive.body.instructions] == [
        "x",
        "measure",
    ]


def test_public_circuit_family_retains_a_parameter_used_after_dispatch() -> None:
    """A control parameter stays in a child ABI when the circuit still uses it."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if theta > 0.0:
        state = qm.x(state)
    state = qm.rz(state, theta)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="public_retained_dispatch_parameter",
    ).specialize("sample")
    family = lower_to_circuit_family(program)

    selected = family.select({"theta": 0.5})

    assert [slot.name for slot in selected.parameter_abi.slots] == ["theta"]
    assert [instruction.opcode for instruction in selected.body.instructions] == [
        "x",
        "rz",
        "measure",
    ]


def test_public_circuit_family_retains_a_result_only_dispatch_parameter() -> None:
    """A control parameter stays in a child ABI when its result still uses it."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bool]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    bit, state = qm.measure_z(state, 0)
    qm.discard(state)
    return bit == enabled
"""
    family = lower_to_circuit_family(
        qm.build_source(
            source,
            module_name="public_result_only_dispatch_parameter",
        ).specialize("sample")
    )

    assert len(family.variants) == 2
    for variant in family.variants:
        assert [slot.name for slot in variant.circuit.parameter_abi.slots] == [
            "enabled"
        ]


def test_public_circuit_family_builds_branch_local_predicate_subtrees() -> None:
    """Different predicates in sibling branches form one complete decision tree."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(value: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if value > 0:
        if value > 10:
            state = qm.x(state)
    else:
        if value < -10:
            state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="public_branch_local_dispatch_predicates",
    ).specialize("sample")
    family = lower_to_circuit_family(program)

    assert len(family.variants) == 4
    assert len(family.dispatch_predicates) == 3
    for value, expected in ((-20, "x"), (-2, "measure"), (2, "measure"), (20, "x")):
        circuit = family.select({"value": value})
        opcodes = [instruction.opcode for instruction in circuit.body.instructions]
        assert ("x" if "x" in opcodes else "measure") == expected


def test_public_circuit_family_reuses_an_identical_predicate() -> None:
    """Repeated control does not create contradictory duplicate decisions."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(value: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if value > 0:
        state = qm.x(state)
    if value > 0:
        state = qm.z(state)
    return qm.measure(state)
"""
    family = lower_to_circuit_family(
        qm.build_source(
            source,
            module_name="public_reused_dispatch_predicate",
        ).specialize("sample")
    )

    assert len(family.dispatch_predicates) == 1
    assert len(family.variants) == 2
    assert [
        instruction.opcode
        for instruction in family.select({"value": 1}).body.instructions
    ] == [
        "x",
        "z",
        "measure",
    ]


def test_public_circuit_family_evaluates_a_fixed_aggregate_path() -> None:
    """A scalar component can dispatch without enumerating its parent vector."""

    source = """
import qamomile as qm

@qm.qkernel
def sample(values: qm.Param[qm.Vector[qm.Int, 2]]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if values[1] > 0:
        state = qm.x(state)
    return qm.measure(state)
"""
    family = lower_to_circuit_family(
        qm.build_source(
            source,
            module_name="public_aggregate_dispatch_predicate",
        ).specialize("sample")
    )

    disabled = family.select({"values": (100, -1)})
    enabled = family.select({"values": (-100, 1)})

    assert [instruction.opcode for instruction in disabled.body.instructions] == [
        "measure"
    ]
    assert [instruction.opcode for instruction in enabled.body.instructions] == [
        "x",
        "measure",
    ]


def test_public_launch_materialization_expands_a_parameterized_for_loop() -> None:
    """One exact launch count expands in Rust without value enumeration."""

    source = """
import qamomile as qm

@qm.qkernel
def repeat(rounds: qm.Param[qm.Int], theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(rounds):
        state = qm.x(state)
    state = qm.rz(state, theta)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="public_launch_for_materialization",
    ).specialize("repeat")

    zero = lower_to_circuit_at(program, {"rounds": -1})
    three = lower_to_circuit_at(program, {"rounds": 3})

    assert [slot.name for slot in three.parameter_abi.slots] == ["theta"]
    assert [instruction.opcode for instruction in zero.body.instructions] == [
        "rz",
        "measure",
    ]
    assert [instruction.opcode for instruction in three.body.instructions] == [
        "x",
        "x",
        "x",
        "rz",
        "measure",
    ]


def test_public_launch_materialization_enforces_the_loop_work_budget() -> None:
    """An invocation cannot force unbounded native circuit expansion."""

    source = """
import qamomile as qm

@qm.qkernel
def repeat(rounds: qm.Param[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(rounds):
        state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="public_launch_loop_budget",
    ).specialize("repeat")

    with pytest.raises(qm.CapabilityError, match="1000001 iterations"):
        lower_to_circuit_at(program, {"rounds": 1_000_001})
