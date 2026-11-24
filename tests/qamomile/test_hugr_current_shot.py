"""Certified local HUGR current-shot execution and finite Int acceptance."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from qamomile.hugr._compiler import _DirectHugrLowerer
from qamomile.hugr._module import _finish_hugr_package
from qamomile.ir import Operation, Value, attributes
from qamomile.types import Bool, Int, Real

pytestmark = pytest.mark.hugr


def _lowerer(origin=0):
    """Create an isolated direct emitter for target-level acceptance witnesses."""
    from hugr.build import Module

    module = Module()
    function = module.define_function("main", [], [], visibility="Public")
    lowerer = _DirectHugrLowerer(
        SimpleNamespace(functions=[]),
        function,
        definitions={},
        adjoint_helpers={},
        boolean_plans={},
        current_shot_origin=origin,
    )
    return module, function, lowerer


def _runner(module, function, lowerer, outputs, path):
    """Build a real Selene runner retaining all observable result records."""
    import selene_sim

    lowerer._record_result_bits(
        tuple(lowerer._materialize_boolean_condition(value) for value in outputs)
    )
    function.set_outputs()
    function.add_state_order(lowerer.previous_measurement, function.output_node)
    package, _ = _finish_hugr_package(module, record_results=True)
    return selene_sim.build(package, name="qamomile_current_shot", build_dir=path)


def _rows(runner, shots, processes=1):
    """Observe source-ordered bits from a fresh invocation of one cached runner."""
    import selene_sim

    return [
        tuple(
            value
            for _, value in sorted(
                (int(tag.removeprefix("qamomile_")), value)
                for tag, value in shot
                if tag.startswith("qamomile_")
            )
        )
        for shot in runner.run_shots(
            selene_sim.Quest(),
            n_qubits=1,
            n_shots=shots,
            n_processes=processes,
            shot_offset=0,
            shot_increment=1,
        )
    ]


def test_native_current_shot_restarts_and_preserves_worker_identity(tmp_path):
    """Repeated reads, table access and arithmetic preserve every logical slot."""
    module, function, lowerer = _lowerer()
    index = lowerer._emit_current_shot()
    repeated = lowerer._emit_current_shot()
    remainder = lowerer._emit_dynamic_int(
        Operation("classical.mod", (), (Value(0, Int),)), (index, 3)
    )
    selected = lowerer._emit_int_index(
        Operation("classical.index", (), (Value(1, Bool),)),
        (False, True, False),
        remainder,
    )
    runner = _runner(
        module,
        function,
        lowerer,
        (
            lowerer._int_compare("eq", index, repeated),
            lowerer._int_compare("eq", index, 0),
            selected,
            lowerer._int_compare("eq", index, 8),
        ),
        tmp_path,
    )
    for shots, processes in ((1, 1), (4, 1), (9, 1), (9, 2), (9, 3), (4, 2)):
        assert _rows(runner, shots, processes) == [
            (1, int(i == 0), int(i % 3 == 1), int(i == 8)) for i in range(shots)
        ]


@pytest.mark.parametrize("origin", [0, 1])
def test_certified_origin_normalization_first_and_last(origin, tmp_path):
    """Both certified native origins have the same zero-based source meaning."""
    from hugr.std.int import IntVal

    from qamomile._native import compiler as native

    policy = native.current_shot_policy(origin)
    module, function, lowerer = _lowerer(origin)
    first = lowerer._normalize_current_shot_wire(function.load(IntVal(origin, 6))[0])
    last = lowerer._normalize_current_shot_wire(
        function.load(IntVal(origin + policy.max_shots - 1, 6))[0]
    )
    runner = _runner(
        module,
        function,
        lowerer,
        (
            lowerer._int_compare("eq", first, 0),
            lowerer._int_compare("eq", last, policy.integer_max),
        ),
        tmp_path,
    )
    assert _rows(runner, 1) == [(1, 1)]


@pytest.mark.parametrize("value", [-(1 << 31), -1, 0, 1, (1 << 31) - 1])
def test_finite_int_to_real_is_exact_at_numeric_boundaries(value, tmp_path):
    """Signed32 extremes convert exactly to finite binary64 values on Selene."""
    from hugr.std.int import IntVal

    module, function, lowerer = _lowerer()
    integer = lowerer._checked_int(function.load(IntVal(value, 6))[0])
    real = lowerer._emit_dynamic_int(
        Operation("classical.cast", (), (Value(0, Real),), attributes(target="Real")),
        (integer,),
    )
    comparison = lowerer._emit_dynamic_real(
        Operation(
            "classical.compare_chain",
            (),
            (Value(1, Bool),),
            attributes(operators=("eq",)),
        ),
        (real, float(value)),
    )
    runner = _runner(module, function, lowerer, (comparison,), tmp_path)
    assert _rows(runner, 1) == [(1,)]


@pytest.mark.parametrize("origin", [0, 1])
def test_current_shot_to_real_uses_normalized_coordinate(origin, tmp_path):
    """Real casts see zero-origin coordinates at both certified domain limits."""
    from hugr.std.int import IntVal

    from qamomile._native import compiler as native

    policy = native.current_shot_policy(origin)
    module, function, lowerer = _lowerer(origin)
    comparisons = []
    for value in (0, policy.integer_max):
        integer = lowerer._normalize_current_shot_wire(
            function.load(IntVal(origin + value, 6))[0]
        )
        real = lowerer._emit_dynamic_int(
            Operation(
                "classical.cast", (), (Value(0, Real),), attributes(target="Real")
            ),
            (integer,),
        )
        comparisons.append(
            lowerer._emit_dynamic_real(
                Operation(
                    "classical.compare_chain",
                    (),
                    (Value(1, Bool),),
                    attributes(operators=("eq",)),
                ),
                (real, float(value)),
            )
        )
    runner = _runner(module, function, lowerer, comparisons, tmp_path)
    assert _rows(runner, 1) == [(1, 1)]


@pytest.mark.parametrize("stage", ["Const", "Param"])
def test_current_shot_real_rotation_matches_reference_across_workers(stage):
    """Source Real casts drive deterministic rotations in logical shot order."""
    import math

    import qamomile as qm
    from qamomile.hugr import Selene
    from qamomile.persistence import deserialize, serialize

    source = f"""
import qamomile as qm

@qm.qkernel
def rotate(theta: qm.{stage}[qm.Real]) -> qm.Shot[qm.Bit]:
    index = qm.get_current_shot()
    state = qm.prepare(False)
    state = qm.ry(state, theta * float(index))
    return qm.measure(state)
"""
    const_bindings = {"theta": math.pi} if stage == "Const" else {}
    parameters = {"theta": math.pi} if stage == "Param" else {}
    program = qm.build_source(source).specialize("rotate", **const_bindings)
    endpoint = Selene(n_processes=2)
    artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))
    for shots in (1, 9, 4):
        expected = qm.run(
            program, on=qm.Reference(), parameters=parameters, shots=shots
        ).result
        observed = qm.execute(artifact, on=endpoint, parameters=parameters, shots=shots)
        assert observed.values == expected.values
        assert observed.shots == shots


@pytest.mark.parametrize("left,right", [(-7, 3), (7, -3), (-7, -3), (7, 3)])
def test_finite_int_division_matches_source_floor_semantics(left, right, tmp_path):
    """Signed division and modulo retain Python's floor and divisor-sign rule."""
    from hugr.std.int import IntVal

    from qamomile.hugr._compiler import _IntWire

    module, function, lowerer = _lowerer()
    value = _IntWire(function.load(IntVal(left, 6))[0])
    quotient = lowerer._emit_dynamic_int(
        Operation("classical.floordiv", (), (Value(0, Int),)), (value, right)
    )
    remainder = lowerer._emit_dynamic_int(
        Operation("classical.mod", (), (Value(1, Int),)), (value, right)
    )
    runner = _runner(
        module,
        function,
        lowerer,
        (
            lowerer._int_compare("eq", quotient, left // right),
            lowerer._int_compare("eq", remainder, left % right),
        ),
        tmp_path,
    )
    assert Counter(_rows(runner, 1)) == {(1, 1): 1}


@pytest.mark.parametrize(
    "origin,raw", [(0, -1), (1, 0), (0, 1 << 31), (1, (1 << 31) + 1)]
)
def test_raw_counter_range_failure_prevents_observation(origin, raw, tmp_path):
    """Invalid native counters terminate before a source result can be recorded."""
    from hugr.std.int import IntVal
    from selene_sim.exceptions import SelenePanicError

    module, function, lowerer = _lowerer(origin)
    index = lowerer._normalize_current_shot_wire(function.load(IntVal(raw, 6))[0])
    runner = _runner(
        module, function, lowerer, (lowerer._int_compare("eq", index, 0),), tmp_path
    )
    with pytest.raises(SelenePanicError, match="Q8694"):
        _rows(runner, 1)


@pytest.mark.parametrize(
    "opcode,left,right",
    [
        ("add", (1 << 31) - 1, 1),
        ("sub", -(1 << 31), 1),
        ("mul", 65536, 65536),
        ("floordiv", -(1 << 31), -1),
    ],
)
def test_finite_int_overflow_never_wraps(opcode, left, right, tmp_path):
    """Exact calculations outside the finite target domain fail explicitly."""
    from hugr.std.int import IntVal
    from selene_sim.exceptions import SelenePanicError

    from qamomile.hugr._compiler import _IntWire

    module, function, lowerer = _lowerer()
    result = lowerer._emit_dynamic_int(
        Operation(f"classical.{opcode}", (), (Value(0, Int),)),
        (_IntWire(function.load(IntVal(left, 6))[0]), right),
    )
    runner = _runner(
        module, function, lowerer, (lowerer._int_compare("eq", result, 0),), tmp_path
    )
    with pytest.raises(SelenePanicError, match="Q8695"):
        _rows(runner, 1)


@pytest.mark.parametrize("entry", ["is_first", "stable", "alternating", "selected"])
def test_current_shot_source_matches_reference_and_survives_persistence(entry):
    """Source kernels retain context through calls, loops, gates and saved packages."""
    import qamomile as qm
    from qamomile.hugr import Selene
    from qamomile.persistence import deserialize, serialize

    from .test_current_shot import SOURCE

    program = qm.build_source(SOURCE).specialize(entry)
    endpoint = Selene(n_processes=2)
    artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))
    for shots in (1, 9, 4):
        expected = qm.run(program, on=qm.Reference(), shots=shots).result
        observed = qm.execute(artifact, on=endpoint, shots=shots)
        assert observed.values == expected.values
        assert observed.shots == shots


def test_current_shot_requires_certified_target_convention():
    """Bare HUGR and uncertified Helios cannot promise logical invocation identity."""
    import qamomile as qm
    from qamomile.hugr import Helios, Hugr, Selene

    from .test_current_shot import SOURCE

    program = qm.build_source(SOURCE).specialize("is_first")
    for target in (
        Hugr(current_shot_origin=None),
        Helios(client=SimpleNamespace()).target,
    ):
        with pytest.raises(qm.ArtifactError, match="logical_shot_context"):
            qm.compile(program, to=target)
    assert Selene().target.fingerprint == Hugr().fingerprint
    assert (
        Hugr(current_shot_origin=0).fingerprint
        != Hugr(current_shot_origin=1).fingerprint
    )


def test_selene_rejects_counter_mismatch_and_count_limit_before_execution():
    """Convention mismatch and oversized identity domains fail before launch."""
    from dataclasses import replace

    import qamomile as qm
    from qamomile.hugr import Selene
    from qamomile.ir import FrozenMapping

    from .test_current_shot import SOURCE

    endpoint = Selene()
    artifact = qm.compile(
        qm.build_source(SOURCE).specialize("is_first"), to=endpoint.target
    )
    metadata = dict(artifact.metadata.entries)
    metadata["current_shot_origin"] = 1
    mismatched = replace(artifact, metadata=FrozenMapping.from_mapping(metadata))
    with pytest.raises(qm.ExecutionError, match="different counter convention"):
        endpoint.launch(mismatched, qm.Invocation.create(shots=4))
    with pytest.raises(ValueError, match="current.shot|shot"):
        endpoint.launch(artifact, qm.Invocation.create(shots=(1 << 31) + 1))


@pytest.mark.parametrize("runtime_version", ["0.2.16", "0.2.18", "0.3.1", "0.3.3"])
def test_selene_rejects_unqualified_current_shot_runtime(runtime_version, monkeypatch):
    """Nearby unqualified releases cannot inherit the tested shot schedule."""
    import qamomile as qm
    from qamomile.hugr import Selene, _selene

    from .test_current_shot import SOURCE

    endpoint = Selene()
    artifact = qm.compile(
        qm.build_source(SOURCE).specialize("is_first"), to=endpoint.target
    )
    monkeypatch.setattr(_selene, "version", lambda name: runtime_version)
    with pytest.raises(qm.ExecutionError, match="certified for Selene"):
        endpoint.launch(artifact, qm.Invocation.create(shots=4))
