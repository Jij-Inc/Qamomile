"""Adversarial tests for QCore's persistent semantic boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest

import qamomile as qm
from qamomile.effects import Effect, Requirement
from qamomile.ir import Constant, Operation, Program, Region, Value, attributes
from qamomile.operations import (
    GATES_BY_NAME,
    KNOWN_OPCODES,
    OPERATION_EFFECTS,
    OPERATION_REQUIREMENTS,
    QUANTUM_OPCODES,
    TERMINATOR_KINDS,
    infer_classical_result_type,
)
from qamomile.types import unify_dimensions

_SINGLE_QUBIT_SOURCE = """
import qamomile as qm

@qm.qkernel
def single() -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.h(state)
    return qm.measure(state)
"""

_CLASSICAL_IF_SOURCE = """
import qamomile as qm

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Int:
    value = 1
    if flag:
        value = 2
    else:
        value = 3
    return value
"""

_STAGE_JOIN_SOURCE = """
import qamomile as qm

@qm.qkernel
def choose(flag: qm.Bool, observed: qm.Shot[qm.Int]) -> qm.Shot[qm.Int]:
    if flag:
        result = 0
    else:
        result = observed
    return result
"""

_MATRIX_ROW_SOURCE = """
import qamomile as qm

@qm.qkernel
def row(values: qm.Matrix[qm.Real, 2, 3]) -> qm.Vector[qm.Real, 3]:
    return values[0]
"""

_CLASSICAL_ARITHMETIC_SOURCE = """
import qamomile as qm

@qm.qkernel
def add(left: qm.Param[qm.Int], right: qm.Int) -> qm.Int:
    return left + right
"""

_DEPENDENT_CALL_SOURCE = """
import qamomile as qm

@qm.qkernel
def allocate(width: qm.Const[qm.Int]) -> qm.Q[qm.Bits[width]]:
    return qm.qubits(width)

@qm.qkernel
def allocate_one() -> qm.Q[qm.Bits[1]]:
    return allocate(1)
"""


def test_semantic_catalog_covers_every_quantum_operation_effect() -> None:
    """The verifier's closed opcode set and effect facts cannot drift apart."""

    assert QUANTUM_OPCODES <= OPERATION_EFFECTS.keys()
    assert OPERATION_EFFECTS.keys() <= KNOWN_OPCODES
    assert OPERATION_REQUIREMENTS.keys() <= KNOWN_OPCODES
    assert TERMINATOR_KINDS == {
        "return",
        "unreachable",
        "yield",
        "break",
        "continue",
        "external",
    }


def test_semantic_catalog_authorities_are_immutable() -> None:
    """Process-local consumers cannot mutate the compiler's central facts."""

    with pytest.raises(TypeError):
        GATES_BY_NAME["h"] = GATES_BY_NAME["x"]  # type: ignore[index]
    with pytest.raises(TypeError):
        OPERATION_EFFECTS["quantum.h"] = frozenset({Effect.QUANTUM})  # type: ignore[index]
    with pytest.raises(TypeError):
        OPERATION_REQUIREMENTS["quantum.h"] = frozenset({Requirement.ROTATION})  # type: ignore[index]


def test_python_operation_catalog_fails_closed_without_native_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source metadata cannot fall back to a second handwritten vocabulary."""

    import qamomile.operations as operations

    monkeypatch.setattr(operations.native, "semantic_operation_catalog", None)
    with pytest.raises(RuntimeError, match="Q4501:.*unavailable"):
        operations._load_native_catalog()

    monkeypatch.setattr(
        operations.native,
        "semantic_operation_catalog",
        lambda: (["not-an-operation-row"], []),
    )
    with pytest.raises(RuntimeError, match="Q4501:.*malformed"):
        operations._load_native_catalog()


def test_module_boundary_rejects_duplicate_parameter_names() -> None:
    """Persistent callable names remain an unambiguous source-level ABI."""

    module = qm.build_source(_CLASSICAL_ARITHMETIC_SOURCE)
    entry = module.function("add")
    duplicate = replace(entry.parameters[1], name=entry.parameters[0].name)

    with pytest.raises(qm.VerificationError, match="Q1017:.*duplicate parameter"):
        replace(
            module,
            functions=(replace(entry, parameters=(entry.parameters[0], duplicate)),),
        )


def test_module_function_resolution_uses_native_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local selection has one native owner and a checked Python projection."""

    module = qm.build_source(_CLASSICAL_ARITHMETIC_SOURCE)
    entry = module.function("add")

    assert module.function(entry.symbol) is entry
    with pytest.raises(qm.SpecializationError, match="Q3001:.*found 0"):
        module.function("missing")
    with pytest.raises(qm.SpecializationError, match="Q3001:.*found 0"):
        module.selected_digest("missing")

    local_name = entry.symbol.rsplit(".", 1)[-1]
    ambiguous = replace(
        module,
        functions=(entry, replace(entry, symbol=f"another.{local_name}")),
    )
    with pytest.raises(qm.SpecializationError, match="Q3001:.*found 2"):
        ambiguous.function(local_name)

    from qamomile._native import compiler as native

    monkeypatch.setattr(native, "resolve_module_entrypoint", None)
    with pytest.raises(qm.SpecializationError, match="Q3000"):
        module.function("add")

    monkeypatch.setattr(native, "resolve_module_entrypoint", lambda *_: object())
    with pytest.raises(qm.VerificationError, match="Q4002:.*malformed symbol"):
        module.function("add")

    monkeypatch.setattr(native, "resolve_module_entrypoint", lambda *_: "unknown")
    with pytest.raises(qm.VerificationError, match="Q4002:.*unknown symbol"):
        module.function("add")


def test_module_boundary_checks_defaults_and_constant_values() -> None:
    """Portable literals must inhabit the types declared beside them."""

    module = qm.build_source(_CLASSICAL_ARITHMETIC_SOURCE)
    entry = module.function("add")
    wrong_default = replace(entry.parameters[0], default="not an Int", has_default=True)
    with pytest.raises(qm.VerificationError, match="Q1057: default for parameter"):
        replace(
            module,
            functions=(
                replace(entry, parameters=(wrong_default, entry.parameters[1])),
            ),
        )

    with pytest.raises(qm.VerificationError, match="Q1058: module constant"):
        replace(module, constants=(Constant("answer", qm.Int, True),))


def test_program_serialization_requires_native_concrete_entrypoint_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One Rust verification rejects an unspecialized persistent program."""

    import qamomile.serialization as serialization
    import qamomile.types as type_module

    module = qm.build_source(_DEPENDENT_CALL_SOURCE)
    entry = module.function("allocate")
    raw = Program(
        module_name=module.module_name,
        functions=(entry,),
        constants=(),
        entrypoint=entry.symbol,
        const_bindings=(("width", 3),),
    )
    inspect_program = serialization.native.inspect_program
    calls = 0

    def inspect_once(payload: bytes) -> object:
        """Count and forward the sole native executable-program check."""

        nonlocal calls
        calls += 1
        return inspect_program(payload)

    def reject_python_type_walk(_type_spec: object) -> object:
        """Prove persistence no longer rediscovers dimensions in Python."""

        raise AssertionError("Python repeated symbolic-dimension discovery")

    monkeypatch.setattr(serialization.native, "inspect_program", inspect_once)
    monkeypatch.setattr(type_module, "symbolic_dimensions", reject_python_type_walk)

    with pytest.raises(qm.VerificationError, match="Q1059:.*symbolic dimensions"):
        raw.to_bytes()
    assert calls == 1


def test_program_encoding_fails_closed_without_a_native_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executable bytes cannot escape through a missing or malformed verifier."""

    import qamomile.serialization as serialization

    program = qm.build_source(_SINGLE_QUBIT_SOURCE).specialize("single")
    monkeypatch.setattr(serialization.native, "inspect_program", None)
    with pytest.raises(qm.VerificationError, match="Q4002:.*unavailable"):
        program.to_bytes()

    monkeypatch.setattr(
        serialization.native, "inspect_program", lambda _payload: object()
    )
    with pytest.raises(qm.VerificationError, match="Q4002:.*malformed header"):
        program.to_bytes()


def test_module_boundary_rejects_unknown_opcode() -> None:
    """Persistent QCore accepts only its explicitly versioned opcode set."""

    module = qm.build_source(_SINGLE_QUBIT_SOURCE)
    entry = module.function("single")
    operations = tuple(
        replace(operation, opcode="quantum.clone")
        if operation.opcode == "quantum.h"
        else operation
        for operation in entry.body.operations
    )

    with pytest.raises(qm.VerificationError, match="Q1032: unknown QCore opcode"):
        replace(
            module,
            functions=(
                replace(entry, body=replace(entry.body, operations=operations)),
            ),
        )


def test_module_boundary_enforces_rank_reducing_array_index_type() -> None:
    """A forged matrix row cannot claim to be a scalar in persistent QCore."""

    module = qm.build_source(_MATRIX_ROW_SOURCE)
    entry = module.function("row")
    access = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "classical.index"
    )
    forged = replace(access, results=(replace(access.results[0], type=qm.Real),))
    operations = tuple(
        forged if operation is access else operation
        for operation in entry.body.operations
    )

    with pytest.raises(qm.VerificationError, match="malformed classical.index"):
        replace(
            module,
            functions=(
                replace(entry, body=replace(entry.body, operations=operations)),
            ),
        )


def test_module_boundary_derives_every_classical_result_type() -> None:
    """A producer cannot forge the result type of an ordinary classical op."""

    module = qm.build_source(_CLASSICAL_ARITHMETIC_SOURCE)
    entry = module.function("add")
    addition = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "classical.add"
    )
    forged = replace(addition, results=(replace(addition.results[0], type=qm.Bool),))

    with pytest.raises(qm.VerificationError, match="malformed classical.add"):
        replace(
            module,
            functions=(
                replace(
                    entry,
                    body=replace(
                        entry.body,
                        operations=tuple(
                            forged if operation is addition else operation
                            for operation in entry.body.operations
                        ),
                    ),
                ),
            ),
        )


def test_local_peer_equations_are_rigid_but_call_dimensions_remain_flexible() -> None:
    """Local peers preserve binders and nested stages while calls instantiate."""

    with pytest.raises(ValueError, match="classical.select alternatives"):
        infer_classical_result_type(
            "classical.select",
            (qm.Bool, qm.Bits["n"], qm.Bits["m"]),
        )
    with pytest.raises(ValueError, match="classical.select alternatives"):
        infer_classical_result_type(
            "classical.select",
            (
                qm.Bool,
                tuple[qm.Param[qm.Int]],
                tuple[qm.Shot[qm.Int]],
            ),
        )

    assert (
        infer_classical_result_type(
            "classical.select",
            (
                qm.Bool,
                qm.Param[qm.Bits["n"]],
                qm.Shot[qm.Bits["n"]],
            ),
        )
        == qm.Shot[qm.Bits["n"]]
    )
    assert unify_dimensions(qm.Bits["n"], qm.Bits[4]) == {"n": 4}

    import qamomile.types as types

    scheme = qm.Param[tuple[qm.Bits["n"], qm.Bits["m"]]]  # noqa: F821
    instantiated = types.substitute_dimensions(scheme, {"n": 4, "m": "remaining"})
    assert (
        instantiated
        == (  # noqa: F821
            qm.Param[tuple[qm.Bits[4], qm.Bits["remaining"]]]  # noqa: F821
        )
    )
    assert types.symbolic_dimensions(instantiated) == frozenset({"remaining"})


def test_python_classical_type_inference_fails_closed_without_native_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source adapter cannot restore the retired Python type equations."""

    import qamomile.operations as operations

    monkeypatch.setattr(operations.native, "infer_classical_result_type", None)
    with pytest.raises(TypeError, match="Q4505:.*unavailable"):
        infer_classical_result_type("classical.add", (qm.Int, qm.Int))

    monkeypatch.setattr(
        operations.native,
        "infer_classical_result_type",
        lambda *_: b"\x1a\x01\xff",
    )
    with pytest.raises(RuntimeError, match="Q4506:.*malformed"):
        infer_classical_result_type("classical.add", (qm.Int, qm.Int))


def test_python_type_relations_fail_closed_without_native_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source type boundaries cannot restore retired Python relations."""

    import qamomile.types as types

    monkeypatch.setattr(types.native, "is_type_assignable", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        types.is_type_assignable(qm.Int, qm.Int)

    monkeypatch.setattr(types.native, "is_type_assignable", lambda *_: "yes")
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        types.is_type_assignable(qm.Int, qm.Int)

    monkeypatch.setattr(types.native, "are_local_peer_types", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        types.local_peer_types(qm.Int, qm.Int)

    monkeypatch.setattr(types.native, "are_local_peer_types", lambda *_: "yes")
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        types.local_peer_types(qm.Int, qm.Int)

    monkeypatch.setattr(types.native, "literal_conforms_type", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        types.value_conforms(qm.Int, 1)

    monkeypatch.setattr(types.native, "literal_conforms_type", lambda *_: "yes")
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        types.value_conforms(qm.Int, 1)

    monkeypatch.setattr(types.native, "infer_literal_type", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        types.infer_literal_type(1)

    monkeypatch.setattr(types.native, "infer_literal_type", lambda *_: b"bad")
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        types.infer_literal_type(1)

    monkeypatch.setattr(types.native, "unify_type_dimensions", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        types.unify_dimensions(qm.Bits["n"], qm.Bits[4])

    monkeypatch.setattr(
        types.native,
        "unify_type_dimensions",
        lambda *_: [("n", 0, None)],
    )
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        types.unify_dimensions(qm.Bits["n"], qm.Bits[4])

    monkeypatch.setattr(types.native, "substitute_type_dimensions", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        types.substitute_dimensions(qm.Bits["n"], {"n": 4})

    monkeypatch.setattr(types.native, "substitute_type_dimensions", lambda *_: b"bad")
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        types.substitute_dimensions(qm.Bits["n"], {"n": 4})

    monkeypatch.setattr(types.native, "type_symbolic_dimensions", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        types.symbolic_dimensions(qm.Bits["n"])

    monkeypatch.setattr(
        types.native,
        "type_symbolic_dimensions",
        lambda *_: ["n", "n"],
    )
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        types.symbolic_dimensions(qm.Bits["n"])

    monkeypatch.setattr(types.native, "inspect_type_shape", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        types.logical_width(qm.Bits[4])

    monkeypatch.setattr(
        types.native,
        "inspect_type_shape",
        lambda *_: (0, None, None, None),
    )
    with pytest.raises(RuntimeError, match="Q4512:.*invalid width"):
        types.logical_width(qm.Bits[4])

    monkeypatch.setattr(types.native, "inspect_type_admissibility", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        qm.Q[qm.Bit]

    monkeypatch.setattr(
        types.native,
        "inspect_type_admissibility",
        lambda *_: (True, True),
    )
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        qm.Q[qm.Bit]


def test_python_value_conformance_is_a_native_literal_relation() -> None:
    """Host normalization feeds the same literal relation used by Rust ABIs."""

    import qamomile.types as types

    assert types.value_conforms(qm.Param[qm.Vector[qm.Int, 2]], [1, 2])
    assert not types.value_conforms(qm.Param[qm.Vector[qm.Int, 3]], [1, 2])
    assert types.value_conforms(qm.UFixed[2, 2], 1.25)
    assert not types.value_conforms(qm.UFixed[2, 2], 1.125)
    assert types.value_conforms(qm.Int, 2**100)
    assert types.value_conforms(qm.UFixed[2048, 0], 2**2000)
    assert not types.value_conforms(qm.Int, True)
    pair = types.RecordType("Pair", (("left", qm.Int), ("right", qm.Bool)))
    assert types.value_conforms(pair, {"left": 1, "right": True})
    assert not types.value_conforms(pair, {"left": True, "right": 1})
    assert types.value_conforms(qm.Observable, qm.Z(0) + 0.5 * qm.X(1))
    assert not types.value_conforms(qm.Q[qm.Bits[1]], (0,))
    assert not types.value_conforms(qm.Int, object())


def test_python_literal_inference_is_a_native_introduction_relation() -> None:
    """Source literals receive their unique unstaged type from Rust."""

    import qamomile.types as types

    assert types.infer_literal_type(None) == qm.Unit
    assert types.infer_literal_type(True) == qm.Bool
    assert types.infer_literal_type(2**100) == qm.Int
    assert types.infer_literal_type(1.5) == qm.Real
    assert types.infer_literal_type(1 + 2j) == qm.Complex
    assert types.infer_literal_type((1, False, (2.0,))) == types.as_type(
        tuple[qm.Int, qm.Bool, tuple[qm.Real]]
    )
    with pytest.raises(qm.TypeCheckError, match="Cannot infer"):
        types.infer_literal_type("explicit declaration required")


def test_python_type_constructors_use_native_admissibility() -> None:
    """Coherent and availability modalities share the Rust decoder rules."""

    import qamomile.types as types

    assert qm.Q[tuple[qm.Bits["n"], qm.UFixed[0, 3]]].inner == types.as_type(  # noqa: F821
        tuple[qm.Bits["n"], qm.UFixed[0, 3]]  # noqa: F821
    )
    assert qm.Param[tuple[qm.Int, qm.Bool]].inner == types.as_type(
        tuple[qm.Int, qm.Bool]
    )
    with pytest.raises(qm.TypeCheckError, match=r"Q\[T\] requires"):
        qm.Q[qm.Real]
    with pytest.raises(qm.TypeCheckError, match="unstaged classical data"):
        qm.Param[qm.Q[qm.Bits[1]]]
    with pytest.raises(qm.TypeCheckError, match="unstaged classical data"):
        qm.Shot[qm.Const[qm.Int]]
    with pytest.raises(qm.TypeCheckError, match="Classical arrays"):
        qm.Vector[qm.Q[qm.Bits[1]], 2]


def test_python_linearity_uses_native_quantum_containment(monkeypatch) -> None:
    """Source ownership queries consume the compiler's cached type fact."""

    import qamomile.types as types

    classical = types.as_type(tuple[qm.Int, qm.Param[qm.Bool]])
    quantum = types.as_type(tuple[qm.Int, qm.Q[qm.Bits[1]]])
    record = types.RecordType("Owned", (("tag", qm.Int), ("state", qm.Q[qm.Bits[2]])))

    assert not classical.is_quantum
    assert not classical.is_linear
    assert quantum.is_quantum
    assert quantum.is_linear
    assert record.is_quantum
    assert record.is_linear

    monkeypatch.setattr(types.native, "inspect_type_admissibility", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        _ = quantum.is_quantum

    monkeypatch.setattr(
        types.native,
        "inspect_type_admissibility",
        lambda *_: (False, False),
    )
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        _ = record.is_linear


def test_python_common_stage_uses_native_availability_order(monkeypatch) -> None:
    """Source stage joins retain their API while Rust owns the ordering."""

    import qamomile.types as types

    assert types.common_stage() is None
    assert types.common_stage(qm.Int, qm.Const[qm.Bool]) == "Const"
    assert types.common_stage(qm.Param[qm.Int], qm.Const[qm.Bool]) == "Param"
    assert types.common_stage(qm.Param[qm.Int], qm.Shot[qm.Bool]) == "Shot"
    assert types.common_stage(tuple[qm.Shot[qm.Int], qm.Bool]) is None

    monkeypatch.setattr(types.native, "common_type_stage", None)
    with pytest.raises(TypeError, match="Q4511:.*unavailable"):
        types.common_stage(qm.Int)

    monkeypatch.setattr(types.native, "common_type_stage", lambda *_: "Later")
    with pytest.raises(RuntimeError, match="Q4512:.*malformed"):
        types.common_stage(qm.Int)


def test_call_result_dimension_must_match_direct_const_argument() -> None:
    """A producer cannot forge a dependent result width past a literal call."""

    module = qm.build_source(_DEPENDENT_CALL_SOURCE)
    entry = module.function("allocate_one")
    call = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.call"
    )
    forged = replace(
        call,
        results=(replace(call.results[0], type=qm.Q[qm.Bits[2]]),),
    )

    with pytest.raises(qm.VerificationError, match="Q1028: call"):
        replace(
            module,
            functions=tuple(
                replace(
                    function,
                    result_type=qm.Q[qm.Bits[2]],
                    body=replace(
                        function.body,
                        operations=tuple(
                            forged if operation is call else operation
                            for operation in function.body.operations
                        ),
                    ),
                )
                if function is entry
                else function
                for function in module.functions
            ),
        )


def test_call_result_dimension_must_be_derived_from_an_operand() -> None:
    """A claimed result width cannot invent a binder from a negative scalar."""

    module = qm.build_source(_DEPENDENT_CALL_SOURCE)
    entry = module.function("allocate_one")
    constant = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.constant"
    )

    with pytest.raises(qm.VerificationError, match="Q1028: call"):
        replace(
            module,
            functions=tuple(
                replace(
                    function,
                    body=replace(
                        function.body,
                        operations=tuple(
                            replace(operation, attributes=attributes(value=-1))
                            if operation is constant
                            else operation
                            for operation in function.body.operations
                        ),
                    ),
                )
                if function is entry
                else function
                for function in module.functions
            ),
        )


def test_changed_if_result_cannot_erase_param_control_dependency() -> None:
    """Persistent QCore reconstructs the stage of a branch-defined result."""

    module = qm.build_source(_CLASSICAL_IF_SOURCE)
    entry = module.function("choose")
    conditional = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.if"
    )
    value_index = next(
        index
        for index, result in enumerate(conditional.results)
        if result.hint == "value"
    )
    forged_regions: list[Region] = []
    for region in conditional.regions:
        block = region.entry
        lift = next(
            operation
            for operation in block.operations
            if operation.opcode == "qcore.stage_lift"
            and operation.results[0].id == block.terminator.operands[value_index]
        )
        forged_regions.append(
            Region(
                (
                    replace(
                        block,
                        operations=tuple(
                            operation
                            for operation in block.operations
                            if operation is not lift
                        ),
                        terminator=replace(
                            block.terminator,
                            operands=tuple(
                                lift.operands[0] if index == value_index else operand
                                for index, operand in enumerate(
                                    block.terminator.operands
                                )
                            ),
                        ),
                    ),
                )
            )
        )
    forged_conditional = replace(
        conditional,
        results=tuple(
            replace(result, type=qm.Int) if index == value_index else result
            for index, result in enumerate(conditional.results)
        ),
        regions=tuple(forged_regions),
    )

    with pytest.raises(qm.VerificationError, match="Q1053: qcore.if"):
        replace(
            module,
            functions=(
                replace(
                    entry,
                    body=replace(
                        entry.body,
                        operations=tuple(
                            forged_conditional
                            if operation is conditional
                            else operation
                            for operation in entry.body.operations
                        ),
                    ),
                ),
            ),
        )


def test_module_boundary_rejects_gate_that_clones_an_owned_value() -> None:
    """A forged gate cannot turn one owner into two independently used owners."""

    module = qm.build_source(_SINGLE_QUBIT_SOURCE)
    entry = module.function("single")
    gate = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "quantum.h"
    )
    measurement = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "quantum.measure"
    )
    clone = Value(
        10_000,
        gate.results[0].type,
        "forged_clone",
        gate.results[0].region,
    )
    ignored = Value(10_001, measurement.results[0].type, "ignored")
    operations = tuple(
        replace(operation, results=(*operation.results, clone))
        if operation is gate
        else operation
        for operation in entry.body.operations
    ) + (replace(measurement, operands=(clone.id,), results=(ignored,)),)

    with pytest.raises(
        qm.VerificationError, match="Q2002:.*aliases another live value"
    ):
        replace(
            module,
            functions=(
                replace(entry, body=replace(entry.body, operations=operations)),
            ),
        )


def test_module_boundary_requires_boolean_classical_condition() -> None:
    """A qcore.if condition cannot be forged from an integer SSA value."""

    module = qm.build_source(_CLASSICAL_IF_SOURCE)
    entry = module.function("choose")
    condition = Value(10_000, qm.Int, "forged_condition")
    constant = Operation(
        "qcore.constant",
        (),
        (condition,),
        attributes=(("value", 0),),
    )
    operations: list[Operation] = []
    for operation in entry.body.operations:
        if operation.opcode == "qcore.if":
            operations.append(constant)
            operations.append(
                replace(
                    operation,
                    operands=(condition.id, *operation.operands[1:]),
                )
            )
        else:
            operations.append(operation)

    with pytest.raises(qm.VerificationError, match="Q1039: qcore.if"):
        replace(
            module,
            functions=(
                replace(
                    entry,
                    body=replace(entry.body, operations=tuple(operations)),
                ),
            ),
        )


def test_module_boundary_rejects_result_free_fallthrough_loops() -> None:
    """An empty loop result row is reserved for proven callable exits."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def from_for() -> qm.Q[qm.Bool]:
    state = qm.prepare(False)
    for index in range(1):
        state = qm.x(state)
    return state

""",
        module_name="forged_result_free_fallthrough_loop",
    )
    entry = module.function("from_for")
    loop = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.for"
    )
    forged = replace(loop, results=())
    body = replace(
        entry.body,
        operations=tuple(
            forged if operation is loop else operation
            for operation in entry.body.operations
        ),
    )

    with pytest.raises(
        qm.VerificationError,
        match=r"Q1002: SSA value .* is not defined",
    ):
        replace(
            module,
            functions=tuple(
                replace(function, body=body) if function is entry else function
                for function in module.functions
            ),
        )


def test_source_rejects_quantum_value_as_a_python_condition() -> None:
    """Python control flow cannot pretend that a coherent value is classical."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Bool:
    state = qm.prepare(True)
    if state:
        state = qm.x(state)
    return qm.measure(state)
"""

    with pytest.raises(qm.TypeCheckError, match=r"condition.*Q\[Bool\]"):
        qm.build_source(source)


def test_module_boundary_rejects_extra_structured_region_blocks() -> None:
    """Hidden blocks cannot be smuggled behind the sole structured entry block."""

    module = qm.build_source(_CLASSICAL_IF_SOURCE)
    entry = module.function("choose")
    operation = next(
        item for item in entry.body.operations if item.opcode == "qcore.if"
    )
    first = operation.regions[0].entry
    forged = replace(
        operation,
        regions=(Region((first, first)), operation.regions[1]),
    )
    operations = tuple(
        forged if item is operation else item for item in entry.body.operations
    )

    with pytest.raises(
        qm.VerificationError, match="Q1011: SSA value .* more than once"
    ):
        replace(
            module,
            functions=(
                replace(entry, body=replace(entry.body, operations=operations)),
            ),
        )


def test_source_rejects_nonunitary_controlled_callable() -> None:
    """Only a structurally proved unitary can cross coherent control."""

    source = """
import qamomile as qm

@qm.qkernel
def reset_state(state: qm.Q[qm.Bool]) -> qm.Q[qm.Bool]:
    return qm.reset(state)

@qm.qkernel
def invalid() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    controls = qm.prepare(True)
    target = qm.prepare(False)
    controls, target = qm.control(reset_state, controls, target)
    return qm.measure(controls), qm.measure(target)
"""

    with pytest.raises(qm.VerificationError, match="Q1051: invalid controlled"):
        qm.build_source(source)


def test_module_boundary_rejects_stage_downgrade() -> None:
    """A forged stage lift cannot erase Shot-time availability."""

    module = qm.build_source(_STAGE_JOIN_SOURCE)
    entry = module.function("choose")
    conditional = next(
        item for item in entry.body.operations if item.opcode == "qcore.if"
    )
    branch = conditional.regions[0].entry
    lift = next(item for item in branch.operations if item.opcode == "qcore.stage_lift")
    forged = replace(
        lift,
        operands=(branch.arguments[1].id,),
        results=(replace(lift.results[0], type=qm.Int),),
    )
    operations = tuple(forged if item is lift else item for item in branch.operations)
    forged_conditional = replace(
        conditional,
        regions=(
            Region((replace(branch, operations=operations),)),
            conditional.regions[1],
        ),
    )
    outer = tuple(
        forged_conditional if item is conditional else item
        for item in entry.body.operations
    )

    with pytest.raises(qm.VerificationError, match="Q1010: yield/result mismatch"):
        replace(
            module,
            functions=(replace(entry, body=replace(entry.body, operations=outer)),),
        )


def test_unpacking_multiple_owned_values_uses_distinct_projections() -> None:
    """A quantum product splits into disjoint regions rather than aliases."""

    source = """
import qamomile as qm

@qm.qkernel
def pair() -> qm.Unit:
    first = qm.qubits(1)
    second = qm.qubits(1)
    states = (first, second)
    left, right = states
    qm.discard(left)
    qm.discard(right)
    return None
"""
    module = qm.build_source(source)
    entry = module.function("pair")
    unpack = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.unpack"
    )

    assert unpack.results[0].region != unpack.results[1].region
    assert unpack.results[0].region is not None
    assert unpack.results[1].region is not None
    assert unpack.results[0].region.projections == ("0",)
    assert unpack.results[1].region.projections == ("1",)


def test_operation_effects_are_derived_and_not_stored() -> None:
    """A measurement's nonunitary effects have no forgeable cache."""

    module = qm.build_source(_SINGLE_QUBIT_SOURCE)
    entry = module.function("single")
    measurement = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "quantum.measure"
    )

    assert measurement.effects == frozenset(
        {
            Effect.QUANTUM,
            Effect.MEASUREMENT,
            Effect.PROBABILISTIC,
            Effect.RELEASE,
        }
    )
    assert "effects" not in measurement.__dataclass_fields__


def test_module_boundary_ignores_function_contract_caches() -> None:
    """Target checks use the body-derived contract, never a producer cache."""

    module = qm.build_source(_SINGLE_QUBIT_SOURCE)
    entry = module.function("single")
    assert entry.contract is not None
    requirements = entry.requirements - {Requirement.MEASUREMENT}
    forged = replace(
        entry,
        contract=replace(entry.contract, requirements=requirements),
    )

    reconstructed = replace(module, functions=(forged,)).function("single")

    assert Requirement.MEASUREMENT in reconstructed.requirements
    assert reconstructed.contract != forged.contract


def test_module_boundary_reconstructs_control_availability() -> None:
    """Param control cannot be relabeled as an ordinary compile-time branch."""

    module = qm.build_source(_CLASSICAL_IF_SOURCE)
    entry = module.function("choose")
    operations = tuple(
        replace(
            operation,
            attributes=tuple(
                (name, "Value" if name == "stage" else value)
                for name, value in operation.attributes
            ),
        )
        if operation.opcode == "qcore.if"
        else operation
        for operation in entry.body.operations
    )

    with pytest.raises(qm.VerificationError, match="Q1049: qcore.if"):
        replace(
            module,
            functions=(
                replace(entry, body=replace(entry.body, operations=operations)),
            ),
        )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (tuple((f"row_{index}", "x") for index in range(17)), "more than 16 rows"),
        ((("Origin Name", "x"),), "invalid name"),
        ((("origin", "x"), ("origin", "y")), "duplicate name"),
        ((("doc", "x" * 4096),), "exceeds 4096 bytes"),
    ],
)
def test_intent_rows_are_bounded_by_the_native_verifier(
    rows: tuple[tuple[str, object], ...], message: str
) -> None:
    """Intent bounds live in Rust and reject before any Python projection."""

    from qamomile.serialization import (
        _parse_sealed_package,
        _seal_package,
        encode_literal,
    )

    module = qm.build_source(
        "import qamomile as qm\n\n@qm.qkernel\ndef answer() -> qm.Int:\n    return 1\n",
        module_name="bounded",
    )
    package = _parse_sealed_package(module.to_bytes())
    for name, value in rows:
        encode_literal(value, package.functions[0].intent.add(name=name).value)
    with pytest.raises(qm.SerializationError, match=f"Q1060.*{message}"):
        qm.Module.from_bytes(_seal_package(package))

    package = _parse_sealed_package(module.to_bytes())
    operation = package.functions[0].body.operations[0]
    for name, value in rows:
        encode_literal(value, operation.intent.add(name=name).value)
    with pytest.raises(qm.SerializationError, match=f"Q1060.*{message}"):
        qm.Module.from_bytes(_seal_package(package))
