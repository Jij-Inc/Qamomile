"""Tests for the non-executing Python source frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest

import qamomile as qm
from qamomile.effects import Effect, Requirement
from qamomile.types import ArrayType, WordType, is_type_assignable

from .conftest import (
    MEASURE_AXES_SOURCE,
    MEASURE_RESET_SOURCE,
    RETURNED_FUNCTION_SOURCE,
)

if TYPE_CHECKING:
    MissingRecordField = qm.Int

_TOP_LEVEL_ROUNDS = 2
_SUMMARY_WIDTH = 3


class SourceSummary(NamedTuple):
    """A normal Python record referenced by a file-backed Qamomile function."""

    total: qm.Int
    accepted: qm.Bool


class SizedSummary(NamedTuple):
    """A record whose annotation refers to a portable module constant."""

    bits: qm.Bits[_SUMMARY_WIDTH]


@qm.qkernel
def top_level_summary(total: qm.Int, accepted: qm.Bool) -> SourceSummary:
    """Construct a named classical result without a Qamomile-specific type API."""

    return SourceSummary(total=total, accepted=accepted)


@qm.qkernel
def top_level_sized_summary(bits: qm.Bits[3]) -> SizedSummary:
    """Exercise dependency closure through a record annotation."""

    return SizedSummary(bits=bits)


@qm.qkernel
def top_level_repeat(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    """Provide a file-backed declaration with one referenced module literal."""

    for _ in range(_TOP_LEVEL_ROUNDS):
        state = qm.x(state)
    return state


def test_qkernel_is_the_only_source_decorator() -> None:
    """Normal source UX excludes alternate decorators and compiler internals."""

    assert callable(qm.qkernel)
    assert callable(qm.multiplex)
    assert not hasattr(top_level_repeat, "program")
    for obsolete in (
        "function",
        "FunctionRef",
        "Fn",
        "load_module",
        "load_program",
        "ModuleImage",
        "ProgramImage",
        "ArtifactEnvelope",
        "ExecutableProgram",
        "Product",
        "measure_bit",
        "project_x",
        "project_y",
        "project_z",
        "select",
        "program",
        "unitary",
        "isometry",
        "channel",
        "instrument",
    ):
        assert not hasattr(qm, obsolete)
    assert not hasattr(qm, "plan")
    for internal in (
        "Operation",
        "Requirement",
        "CallableContract",
        "CleanupPlan",
        "ErrorBudget",
        "CompilationTarget",
    ):
        assert not hasattr(qm, internal)
    assert qm.Program.__name__ == "Program"


def test_qkernel_isolates_one_file_declaration_and_its_captures() -> None:
    """Passing a function does not make unrelated file contents semantic."""

    module = top_level_repeat.module()

    assert [function.symbol for function in module.functions] == [
        "tests.qamomile.test_source_compiler.top_level_repeat"
    ]
    assert "range(2)" in top_level_repeat.source_text
    assert "_TOP_LEVEL_ROUNDS" not in top_level_repeat.source_text
    assert "pytest" not in top_level_repeat.source_text
    assert (
        "test_function_is_the_only_source_decorator" not in top_level_repeat.source_text
    )


def test_function_ref_retains_standard_python_record_declarations() -> None:
    """File isolation closes the referenced NamedTuple into portable source."""

    program = top_level_summary.specialize()
    result = qm.evaluate(program, total=5, accepted=True)

    assert dict(result) == {"total": 5, "accepted": True}
    assert "class SourceSummary(NamedTuple)" in top_level_summary.source_text
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()

    sized = top_level_sized_summary.specialize()
    assert dict(qm.evaluate(sized, bits=(1, 0, 1))) == {"bits": (1, 0, 1)}
    assert "_SUMMARY_WIDTH = 3" in top_level_sized_summary.source_text


def test_function_ref_closes_notebook_local_record_declarations() -> None:
    """Non-file-backed handles retain typed records and their dimensions."""

    width = 3

    class NotebookPayload(NamedTuple):
        """One nested notebook-local payload record."""

        bits: qm.Bits[width]

    class NotebookSummary(NamedTuple):
        """One notebook-local result record."""

        payload: NotebookPayload
        accepted: qm.Bool = False

    @qm.qkernel
    def notebook_summary(bits: qm.Bits[3]) -> NotebookSummary:
        """Construct a record whose declaration is outside this source slice."""

        return NotebookSummary(payload=NotebookPayload(bits=bits))

    program = notebook_summary.specialize()
    result = qm.evaluate(program, bits=(1, 0, 1))

    assert dict(result) == {
        "payload": {"bits": (1, 0, 1)},
        "accepted": False,
    }
    assert "class NotebookPayload(NamedTuple)" in notebook_summary.source_text
    assert "class NotebookSummary(NamedTuple)" in notebook_summary.source_text
    assert "width = 3" in notebook_summary.source_text
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()


def test_function_ref_rejects_record_without_source_form_annotations() -> None:
    """Runtime-only record metadata cannot become an alternative type parser."""

    DynamicSummary = NamedTuple("DynamicSummary", [("total", qm.Int)])

    with pytest.raises(qm.SourceUnavailableError, match="Q5104.*source-form"):

        @qm.qkernel
        def dynamic_summary(total: qm.Int) -> DynamicSummary:
            return DynamicSummary(total)


def test_function_ref_rejects_cyclic_notebook_record_annotations() -> None:
    """Notebook record capture stays iterative and rejects recursive schemas."""

    class RecursiveSummary(NamedTuple):
        """One unsupported recursive record."""

        child: RecursiveSummary

    with pytest.raises(qm.SourceUnavailableError, match="Q5104.*cycle"):

        @qm.qkernel
        def recursive_summary(value: RecursiveSummary) -> RecursiveSummary:
            return value


def test_function_ref_does_not_invent_unresolved_record_annotations() -> None:
    """Missing annotation names remain errors for the Rust-owned type checker."""

    class UnresolvedSummary(NamedTuple):
        """One record whose field type is deliberately unavailable."""

        value: MissingRecordField

    @qm.qkernel
    def unresolved_summary(value: qm.Int) -> UnresolvedSummary:
        return UnresolvedSummary(value)

    assert "MissingRecordField = None" not in unresolved_summary.source_text
    with pytest.raises(qm.FrontendError, match="MissingRecordField"):
        unresolved_summary.specialize()


def test_named_tuple_uses_existing_product_ssa_through_python_syntax() -> None:
    """Construction, fields, indexing, and destructuring add no record opcode."""

    source = """
from typing import NamedTuple
import qamomile as qm

class Summary(NamedTuple):
    total: qm.Int
    accepted: qm.Bool

class Pair(NamedTuple):
    left: qm.Int
    right: qm.Int

@qm.qkernel
def summarize(total: qm.Int, accepted: qm.Bool) -> Summary:
    return Summary(accepted=accepted, total=total)

@qm.qkernel
def score(summary: Summary) -> qm.Int:
    total, accepted = summary
    if accepted:
        return summary.total
    return summary[0] - 1

@qm.qkernel
def score_param(summary: qm.Param[Summary]) -> qm.Int:
    if summary.accepted:
        return summary.total
    return 0

@qm.qkernel
def pair_total(pair: Pair) -> qm.Int:
    result = 0
    for value in pair:
        result += value
    return result
"""
    module = qm.build_source(source, module_name="records")
    summary = module.specialize("summarize")
    score = module.specialize("score")

    assert dict(qm.evaluate(summary, total=8, accepted=False)) == {
        "total": 8,
        "accepted": False,
    }
    assert qm.evaluate(score, summary={"total": 8, "accepted": True}) == 8
    assert qm.evaluate(score, summary={"total": 8, "accepted": False}) == 7
    parametric = module.specialize("score_param")
    assert Effect.HOST_CONTROL in parametric.entry.effects
    assert Requirement.HOST_DISPATCH in parametric.entry.requirements
    assert (
        qm.evaluate(
            parametric,
            summary={"total": 9, "accepted": True},
        )
        == 9
    )
    assert (
        qm.evaluate(module.specialize("pair_total"), pair={"left": 3, "right": 4}) == 7
    )
    assert "qcore.tuple" in summary.inspect()
    assert "qcore.record" not in summary.inspect()


def test_named_tuple_requires_its_normal_python_import() -> None:
    """The frontend does not make an otherwise invalid Python name implicit."""

    source = """
import qamomile as qm

class Summary(NamedTuple):
    total: qm.Int

@qm.qkernel
def summarize() -> Summary:
    return Summary(1)
"""
    with pytest.raises(qm.FrontendError, match="unknown logical type 'Summary'"):
        qm.build_source(source)


def test_external_function_kind_is_literal_and_checked() -> None:
    """Only opaque boundaries need an explicit denotation contract."""

    external = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...
"""
    assert qm.build_source(external).function("oracle").external

    with pytest.raises(qm.FrontendError, match="callable-kind string"):
        qm.build_source(external.replace('kind="unitary"', "kind=qm.Int"))
    with pytest.raises(ValueError, match="unknown Qamomile callable kind"):
        qm.qkernel(kind="oracle")


def test_function_reference_remains_source_only_at_runtime() -> None:
    """Static call signatures must not turn source handles into host functions."""

    @qm.qkernel
    def add(left: int, right: int) -> int:
        return left + right

    with pytest.raises(TypeError, match="source declarations"):
        add(1, 2)


def test_closed_function_values_select_and_invoke_through_rust_lowering() -> None:
    """A runtime choice of code lowers to ordinary calls and structured control."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    state = (flipped if flag else unchanged)(qm.qubits(1))
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="function_values")
    program = module.specialize("choose")

    assert qm.evaluate(program, flag=False) == (0,)
    assert qm.evaluate(program, flag=True) == (1,)
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()
    assert qm.resources.logical(program).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (0, 1)
    inspection = program.inspect()
    assert "qcore.if" in inspection
    assert "qcore.call" in inspection
    assert " = function_ref(" not in inspection
    assert " = invoke(" not in inspection
    assert "QKernel[" not in inspection


def test_single_function_argument_survives_for_carries() -> None:
    """Dead-tag elimination retains structured uses of a fixed callable."""

    source = """
import qamomile as qm

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def repeat_for(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    for _ in range(rounds):
        state = operation(state)
    return state

@qm.qkernel
def execute_for() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(repeat_for(flipped, qm.qubits(1), 2))
"""
    module = qm.build_source(source, module_name="function_loop_carries")
    for_program = module.specialize("execute_for")
    restored_for = qm.Program.from_bytes(for_program.to_bytes())

    assert qm.evaluate(restored_for) == (0,)
    assert "qcore.for" in restored_for.inspect()
    assert qm.resources.logical(restored_for).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (2, 2)
    assert restored_for.to_bytes() == for_program.to_bytes()


def test_nullary_function_values_produce_classical_and_quantum_results() -> None:
    """A selected nullary callable lowers without a fabricated Unit argument."""

    source = """
import qamomile as qm

@qm.qkernel
def zero() -> qm.Int:
    return 0

@qm.qkernel
def one() -> qm.Int:
    return 1

@qm.qkernel
def empty() -> qm.Q[qm.Bits[1]]:
    return qm.qubits(1)

@qm.qkernel
def flipped() -> qm.Q[qm.Bits[1]]:
    return qm.x(qm.qubits(1))

@qm.qkernel
def choose_value(flag: qm.Param[qm.Bool]) -> qm.Param[qm.Int]:
    factory: qm.QKernel[qm.Int] = one if flag else zero
    return factory()

@qm.qkernel
def choose_state(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    factory: qm.QKernel[qm.Q[qm.Bits[1]]] = flipped if flag else empty
    return qm.measure(factory())

@qm.qkernel
def choose_shot() -> qm.Shot[qm.Bits[1]]:
    flag, observed = qm.measure_z(qm.prepare((1,)), 0)
    qm.discard(observed)
    factory: qm.QKernel[qm.Q[qm.Bits[1]]] = flipped if flag else empty
    return qm.measure(factory())
"""
    module = qm.build_source(source, module_name="nullary_function_values")
    value = module.specialize("choose_value")
    state = module.specialize("choose_state")
    shot = module.specialize("choose_shot")
    restored = qm.Program.from_bytes(state.to_bytes())

    assert qm.evaluate(value, flag=False) == 0
    assert qm.evaluate(value, flag=True) == 1
    assert qm.evaluate(restored, flag=False) == (0,)
    assert qm.evaluate(restored, flag=True) == (1,)
    assert qm.evaluate(shot) == (1,)
    assert qm.resources.logical(restored).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (0, 1)
    inspection = restored.inspect()
    assert "qcore.if" in inspection
    assert "qcore.call" in inspection
    assert "QKernel[" not in inspection
    assert "function_ref" not in inspection
    assert " = invoke(" not in inspection


def test_shot_time_function_choice_preserves_linear_state() -> None:
    """A measured selector chooses one function without duplicating its state."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def choose() -> qm.Shot[qm.Bits[1]]:
    flag, state = qm.measure_z(qm.prepare((1,)), 0)
    state = (flipped if flag else unchanged)(state)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="shot_function_value").specialize(
        "choose"
    )

    assert qm.evaluate(program) == (0,)
    assert "stage='Shot'" in program.inspect()


def test_local_function_aliases_are_eliminated_before_qcore() -> None:
    """Named and chained function choices retain ordinary Python call syntax."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[2]]:
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]] = (
        flipped if flag else unchanged
    )
    selected = operation
    left = selected(qm.qubits(1))
    right = selected(qm.qubits(1))
    return qm.measure(qm.concat(left, right))
"""
    program = qm.build_source(source, module_name="function_aliases").specialize(
        "choose"
    )

    assert qm.evaluate(program, flag=False) == (0, 0)
    assert qm.evaluate(program, flag=True) == (1, 1)
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()
    assert qm.resources.logical(program).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (0, 2)
    inspection = program.inspect()
    assert "qcore.if" in inspection
    assert "qcore.call" in inspection
    assert "QKernel[" not in inspection
    assert "function_ref" not in inspection
    assert " = invoke(" not in inspection


def test_local_function_aliases_merge_across_nested_conditionals() -> None:
    """Branch-selected callables remain transient across nested Python control."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def choose(
    outer: qm.Param[qm.Bool],
    inner: qm.Param[qm.Bool],
) -> qm.Shot[qm.Bits[2]]:
    operation = unchanged
    if outer:
        if inner:
            operation = flipped
    left = operation(qm.qubits(1))
    right = operation(qm.qubits(1))
    return qm.measure(qm.concat(left, right))
"""
    program = qm.build_source(
        source, module_name="conditional_function_aliases"
    ).specialize("choose")

    assert qm.evaluate(program, outer=False, inner=False) == (0, 0)
    assert qm.evaluate(program, outer=False, inner=True) == (0, 0)
    assert qm.evaluate(program, outer=True, inner=False) == (0, 0)
    assert qm.evaluate(program, outer=True, inner=True) == (1, 1)
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()
    assert qm.resources.logical(program).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (0, 2)
    inspection = program.inspect()
    assert inspection.count("qcore.if") >= 2
    assert "QKernel[" not in inspection
    assert "function_ref" not in inspection
    assert " = invoke(" not in inspection


def test_invariant_function_aliases_cross_loops_without_becoming_carries() -> None:
    """Invariant callables compose through bounded loops and loop else regions."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    operation = flipped if flag else unchanged
    state = qm.qubits(1)
    for _ in range(1):
        state = operation(state)
    for _ in range(1):
        local_operation = operation
        state = local_operation(state)
    else:
        state = operation(state)
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="loop_invariant_function_aliases")
    program = module.specialize("choose")

    assert qm.evaluate(program, flag=False) == (0,)
    assert qm.evaluate(program, flag=True) == (1,)
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()
    assert qm.resources.logical(program).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (0, 3)
    inspection = program.inspect()
    assert "qcore.for" in inspection
    assert module.inspect().count("qcore.for") >= 2
    assert "QKernel[" not in inspection
    assert "function_ref" not in inspection
    assert " = invoke(" not in inspection


def test_qkernel_results_are_rejected_at_the_source_boundary() -> None:
    """QKernel values are static inputs, not source-level return values."""

    with pytest.raises(qm.FrontendError, match="cannot return QKernel values"):
        qm.build_source(
            RETURNED_FUNCTION_SOURCE,
            module_name="returned_function",
        )


def test_function_value_type_and_invocation_errors_are_source_addressed() -> None:
    """Function values reject incompatible branches and invalid calls early."""

    assert str(qm.QKernel[qm.Int]) == "QKernel[Int]"
    assert str(qm.QKernel[qm.Int, qm.Real]) == "QKernel[Int, Real]"
    assert str(qm.QKernel[qm.Int, qm.Bool, qm.Real]) == "QKernel[Int, Bool, Real]"
    assert not qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]].is_linear
    with pytest.raises(qm.TypeCheckError, match="QKernel expects"):
        qm.QKernel[()]

    open_result = """
import qamomile as qm

@qm.qkernel
def forward(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
) -> qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    return operation
"""
    with pytest.raises(qm.FrontendError, match="cannot return QKernel values"):
        qm.build_source(open_result)

    recursive_result = """
import qamomile as qm

@qm.qkernel
def choose() -> qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    return choose()
"""
    with pytest.raises(qm.FrontendError, match="cannot return QKernel values"):
        qm.build_source(recursive_result)

    incompatible = """
import qamomile as qm

@qm.qkernel
def one(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def two(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return state

@qm.qkernel
def choose(flag: qm.Param[qm.Bool], state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return (one if flag else two)(state)
"""
    with pytest.raises(qm.TypeCheckError, match="incompatible types"):
        qm.build_source(incompatible)

    wrong_arity = """
import qamomile as qm

@qm.qkernel
def add(left: qm.Int, right: qm.Int) -> qm.Int:
    return left + right

@qm.qkernel
def choose(flag: qm.Param[qm.Bool], value: qm.Int) -> qm.Int:
    return (add if flag else add)(value)
"""
    with pytest.raises(qm.FrontendError, match="expects 2 positional arguments"):
        qm.build_source(wrong_arity)

    wrong_type = wrong_arity.replace(
        "(add if flag else add)(value)",
        "(add if flag else add)(value, False)",
    )
    with pytest.raises(qm.TypeCheckError, match="expected Int, got Bool"):
        qm.build_source(wrong_type)

    nullary_with_argument = """
import qamomile as qm

@qm.qkernel
def zero() -> qm.Int:
    return 0

@qm.qkernel
def choose(flag: qm.Param[qm.Bool], value: qm.Int) -> qm.Int:
    return (zero if flag else zero)(value)
"""
    with pytest.raises(qm.FrontendError, match="expects 0 positional arguments"):
        qm.build_source(nullary_with_argument)

    inconsistent_dimensions = """
import qamomile as qm

@qm.qkernel
def keep(
    state: qm.Q[qm.Bits[width]],
    weights: qm.Vector[qm.Real, width],
) -> qm.Q[qm.Bits[width]]:
    return state

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Q[qm.Bits[2]]:
    operation = keep if flag else keep
    return operation(qm.qubits(2), [0.0])
"""
    with pytest.raises(qm.TypeCheckError, match="dimensions do not match"):
        qm.build_source(inconsistent_dimensions)


def test_closed_function_parameters_are_specialized_before_qcore() -> None:
    """A higher-order helper becomes ordinary closed calls in native lowering."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def apply(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return operation(state)

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    if flag:
        state = apply(flipped, state)
    else:
        state = apply(unchanged, state)
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="closed_function_parameter")
    program = module.specialize("choose")
    restored = qm.Program.from_bytes(program.to_bytes())

    assert qm.evaluate(restored, flag=False) == (0,)
    assert qm.evaluate(restored, flag=True) == (1,)
    assert {function.symbol.rsplit(".", 1)[-1] for function in module.functions} >= {
        "choose",
        "flipped",
        "unchanged",
    }
    assert all(
        function.symbol.rsplit(".", 1)[-1] != "apply" for function in module.functions
    )
    inspection = module.inspect()
    assert "QKernel[" not in inspection
    assert "function_ref" not in inspection
    assert "invoke(" not in inspection


def test_adjoint_function_parameter_rejects_nonunitary_candidate() -> None:
    """Selected function values gain adjoints only for verified unitaries."""

    source = """
import qamomile as qm

@qm.qkernel
def clear(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.reset(state)

@qm.qkernel
def inverse(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(operation, state)

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(inverse(clear, qm.qubits(1)))
"""
    with pytest.raises(qm.VerificationError, match="Q1029: invalid adjoint callable"):
        qm.build_source(source)


def test_selected_function_value_can_be_adjointed_after_selection() -> None:
    """Runtime choice resolves to verified static adjoints without function IR."""

    source = """
import qamomile as qm

@qm.qkernel
def identity(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def sample(choose_flip: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    operation = flip if choose_flip else identity
    return qm.measure(qm.adjoint(operation, qm.qubits(1)))
"""
    program = qm.build_source(source).specialize("sample")
    assert qm.evaluate(program, choose_flip=False) == (0,)
    assert qm.evaluate(program, choose_flip=True) == (1,)
    assert "qcore.adjoint_call" in program.inspect()
    assert "function_ref" not in program.inspect()


def test_function_parameters_forward_through_dependent_higher_order_helpers() -> None:
    """Native monomorphization forwards one callable and instantiates its width."""

    source = """
import qamomile as qm

@qm.qkernel
def flip_last(
    state: qm.Q[qm.Bits[2]],
    target: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[2]]:
    return qm.x(state, target=target)

@qm.qkernel
def apply(
    operation: qm.QKernel[qm.Q[qm.Bits[width]], qm.Const[qm.Int], qm.Q[qm.Bits[width]]],
    state: qm.Q[qm.Bits[width]],
    target: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[width]]:
    return operation(state, target)

@qm.qkernel
def twice(
    operation: qm.QKernel[qm.Q[qm.Bits[width]], qm.Const[qm.Int], qm.Q[qm.Bits[width]]],
    state: qm.Q[qm.Bits[width]],
    target: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[width]]:
    return apply(operation, apply(operation, state, target), target)

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[2]]:
    return qm.measure(twice(flip_last, qm.qubits(2), 1))
"""
    module = qm.build_source(source, module_name="forwarded_function_parameter")
    program = module.specialize("sample")

    assert qm.evaluate(program) == (0, 0)
    assert "QKernel[" not in program.inspect()
    assert sum(".__fn_" in function.symbol for function in module.functions) == 2


def test_selected_function_parameter_is_specialized_before_qcore() -> None:
    """A selected closed function crosses a higher-order helper as static calls."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def apply(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return operation(state)

@qm.qkernel
def sample(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    operation = flipped if flag else unchanged
    return qm.measure(apply(operation, qm.qubits(1)))
"""
    module = qm.build_source(source, module_name="dynamic_function_parameter")
    program = module.specialize("sample")

    assert qm.evaluate(program, flag=False) == (0,)
    assert qm.evaluate(program, flag=True) == (1,)
    restored = qm.Program.from_bytes(program.to_bytes())
    assert qm.evaluate(restored, flag=True) == (1,)
    assert qm.resources.logical(restored).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (0, 1)
    assert qm.Module.from_bytes(module.to_bytes()).to_bytes() == module.to_bytes()
    assert "QKernel[" not in module.inspect()
    generated = [
        function for function in module.functions if ".__fn_" in function.symbol
    ]
    assert generated
    assert all("qcore.constant" not in function.inspect() for function in generated)


def test_inline_function_parameter_selection_precedes_argument_effects() -> None:
    """Callable selection is lowered before an inline quantum allocation argument."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def apply(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return operation(state)

@qm.qkernel
def sample(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(
        apply(flipped if flag else unchanged, qm.qubits(1))
    )
"""
    module = qm.build_source(source, module_name="inline_function_parameter")
    sample = module.function("sample").inspect()
    program = module.specialize("sample")

    assert qm.evaluate(program, flag=False) == (0,)
    assert qm.evaluate(program, flag=True) == (1,)
    assert sample.index("qcore.if(%0)") < sample.index("quantum.alloc")


def test_multiple_selected_function_parameters_preserve_candidate_pairs() -> None:
    """Two selected callables retain their correlated specialization choices."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def apply_two(
    first: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    second: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return second(first(state))

@qm.qkernel
def sample(
    flip_first: qm.Param[qm.Bool],
    flip_second: qm.Param[qm.Bool],
) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(
        apply_two(
            flipped if flip_first else unchanged,
            flipped if flip_second else unchanged,
            qm.qubits(1),
        )
    )
"""
    module = qm.build_source(source, module_name="paired_function_parameters")
    program = module.specialize("sample")

    for first in (False, True):
        for second in (False, True):
            assert qm.evaluate(
                program,
                flip_first=first,
                flip_second=second,
            ) == (int(first) ^ int(second),)
    assert sum(".__fn_" in function.symbol for function in module.functions) == 4
    assert "QKernel[" not in module.inspect()


def test_selected_function_parameter_lifts_a_classical_result_stage() -> None:
    """A callable's Param selection remains visible on a pure helper result."""

    source = """
import qamomile as qm

@qm.qkernel
def add_one(value: qm.Int) -> qm.Int:
    return value + 1

@qm.qkernel
def add_two(value: qm.Int) -> qm.Int:
    return value + 2

@qm.qkernel
def apply(
    operation: qm.QKernel[qm.Int, qm.Int],
    value: qm.Int,
) -> qm.Int:
    return operation(value)

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Param[qm.Int]:
    return apply(add_one if flag else add_two, 10)

@qm.qkernel
def choose_direct(flag: qm.Param[qm.Bool]) -> qm.Param[qm.Int]:
    return (add_one if flag else add_two)(10)
"""
    module = qm.build_source(source, module_name="staged_function_parameter")
    program = module.specialize("choose")
    direct = module.specialize("choose_direct")

    assert qm.evaluate(program, flag=False) == 12
    assert qm.evaluate(program, flag=True) == 11
    assert qm.evaluate(direct, flag=False) == 12
    assert qm.evaluate(direct, flag=True) == 11
    assert "Param[Int]" in program.inspect()


def test_multiple_function_parameter_stages_join_at_shot() -> None:
    """Nested Param and Shot callable choices retain the latest availability."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(value: qm.Bool) -> qm.Bool:
    return value

@qm.qkernel
def inverted(value: qm.Bool) -> qm.Bool:
    return not value

@qm.qkernel
def apply_two(
    first: qm.QKernel[qm.Bool, qm.Bool],
    second: qm.QKernel[qm.Bool, qm.Bool],
    value: qm.Bool,
) -> qm.Bool:
    return second(first(value))

@qm.qkernel
def choose(param_flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bool]:
    shot_flag, observed = qm.measure_z(qm.prepare((1,)), 0)
    qm.discard(observed)
    return apply_two(
        inverted if param_flag else unchanged,
        inverted if shot_flag else unchanged,
        False,
    )
"""
    program = qm.build_source(
        source, module_name="joined_function_parameter_stages"
    ).specialize("choose")

    assert qm.evaluate(program, param_flag=False) is True
    assert qm.evaluate(program, param_flag=True) is False
    assert "Shot[Bool]" in program.inspect()


def test_multi_argument_function_values_preserve_stage_and_ownership() -> None:
    """A selected binary callable evaluates each typed argument exactly once."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]], angle: qm.Real) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def rotated(state: qm.Q[qm.Bits[1]], angle: qm.Real) -> qm.Q[qm.Bits[1]]:
    return qm.rx(state, angle)

@qm.qkernel
def choose(
    flag: qm.Param[qm.Bool],
    angle: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[1]]:
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Real, qm.Q[qm.Bits[1]]] = unchanged
    for _ in range(1):
        operation = rotated if flag else operation
    return qm.measure(operation(qm.qubits(1), angle))
"""
    program = qm.build_source(
        source, module_name="multi_argument_function_value"
    ).specialize("choose")
    restored = qm.Program.from_bytes(program.to_bytes())

    assert qm.evaluate(restored, flag=False, angle=3.141592653589793) == (0,)
    assert qm.evaluate(restored, flag=True, angle=3.141592653589793) == (1,)
    assert qm.resources.logical(restored).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (0, 1)
    inspection = restored.inspect()
    assert "qcore.call" in inspection
    assert "QKernel[" not in inspection
    assert "function_ref" not in inspection

    dependent_source = """
import qamomile as qm

@qm.qkernel
def unchanged(
    state: qm.Q[qm.Bits[width]],
    width: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[width]]:
    return state

@qm.qkernel
def flipped(
    state: qm.Q[qm.Bits[width]],
    width: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[width]]:
    return qm.x(state, target=width - 1)

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[2]]:
    operation = flipped if flag else unchanged
    return qm.measure(operation(qm.qubits(2), 2))
"""
    dependent = qm.build_source(
        dependent_source, module_name="dependent_multi_argument_function_value"
    ).specialize("choose")

    assert qm.evaluate(dependent, flag=False) == (0, 0)
    assert qm.evaluate(dependent, flag=True) == (0, 1)


def test_function_values_accept_source_ordered_keyword_arguments() -> None:
    """Rust binds mixed and reversed keywords before eliminating the callable."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(
    state: qm.Q[qm.Bits[width]],
    width: qm.Const[qm.Int],
    angle: qm.Real,
) -> qm.Q[qm.Bits[width]]:
    return state

@qm.qkernel
def rotated(
    state: qm.Q[qm.Bits[width]],
    width: qm.Const[qm.Int],
    angle: qm.Real,
) -> qm.Q[qm.Bits[width]]:
    return qm.rx(state, angle, target=width - 1)

@qm.qkernel
def choose(
    flag: qm.Param[qm.Bool],
    angle: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    operation = rotated if flag else unchanged
    selected = operation
    for _ in range(1):
        selected = operation
    state = selected(qm.qubits(2), angle=angle, width=2)
    return qm.measure(state)

@qm.qkernel
def choose_reversed(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[2]]:
    operation = rotated if flag else unchanged
    state = operation(angle=3.141592653589793, width=2, state=qm.qubits(2))
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="keyword_function_values")
    mixed = module.specialize("choose")
    reversed_call = module.specialize("choose_reversed")
    restored = qm.Program.from_bytes(mixed.to_bytes())

    assert qm.evaluate(restored, flag=False, angle=3.141592653589793) == (0, 0)
    assert qm.evaluate(restored, flag=True, angle=3.141592653589793) == (0, 1)
    assert qm.evaluate(reversed_call, flag=False) == (0, 0)
    assert qm.evaluate(reversed_call, flag=True) == (0, 1)
    assert qm.resources.logical(restored).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (0, 1)
    inspection = restored.inspect()
    assert "qcore.if" in inspection
    assert "qcore.call" in inspection
    assert "QKernel[" not in inspection
    assert " = invoke(" not in inspection


def test_static_calls_preserve_python_argument_evaluation_order() -> None:
    """Native lowering schedules keyword operands without changing the call ABI."""

    source = """
import qamomile as qm
from typing import NamedTuple

class Pair(NamedTuple):
    first: qm.Int
    second: qm.Int

@qm.qkernel
def first_value() -> qm.Int:
    return 1

@qm.qkernel
def second_value() -> qm.Int:
    return 2

@qm.qkernel
def component() -> qm.Int:
    return 0

@qm.qkernel
def pair(first: qm.Int, second: qm.Int) -> Pair:
    return Pair(first, second)

@qm.qkernel
def rotate(
    state: qm.Q[qm.Bits[1]],
    first: qm.Real,
    second: qm.Real,
) -> qm.Q[qm.Bits[1]]:
    return qm.rx(state, first + second)

@qm.qkernel
def rotate_other(
    state: qm.Q[qm.Bits[1]],
    first: qm.Real,
    second: qm.Real,
) -> qm.Q[qm.Bits[1]]:
    return qm.ry(state, first + second)

@qm.qkernel
def apply(
    operation: qm.QKernel[
        qm.Q[qm.Bits[1]],
        qm.Real,
        qm.Real,
        qm.Q[qm.Bits[1]],
    ],
    state: qm.Q[qm.Bits[1]],
    first: qm.Real,
    second: qm.Real,
) -> qm.Q[qm.Bits[1]]:
    return operation(state, first, second)

@qm.qkernel
def first_angle() -> qm.Real:
    return 0.0

@qm.qkernel
def second_angle() -> qm.Real:
    return 0.0

@qm.qkernel
def direct() -> Pair:
    return pair(second=second_value(), first=first_value())

@qm.qkernel
def record() -> Pair:
    return Pair(second=second_value(), first=first_value())

@qm.qkernel
def higher_order_entry() -> qm.Bits[1]:
    state = apply(
        second=second_angle(),
        operation=rotate,
        first=first_angle(),
        state=qm.qubits(1),
    )
    return qm.measure(state)

@qm.qkernel
def adjoint_entry() -> qm.Bits[1]:
    state = qm.adjoint(
        rotate,
        qm.qubits(1),
        second=second_angle(),
        first=first_angle(),
    )
    return qm.measure(state)

@qm.qkernel
def control_entry() -> qm.Bits[3]:
    controls, state = qm.control(
        rotate,
        qm.qubits(2),
        qm.qubits(1),
        second=second_angle(),
        at=component(),
        first=first_angle(),
    )
    return qm.measure(qm.concat(controls, state))

@qm.qkernel
def multiplex_entry() -> qm.Bits[2]:
    index, state = qm.multiplex(
        qm.qubits(1),
        (rotate, rotate_other),
        qm.qubits(1),
        second=second_angle(),
        first=first_angle(),
    )
    return qm.measure(qm.concat(index, state))
"""
    module = qm.build_source(source, module_name="argument_order")

    def assert_call_order(entrypoint: str, *callees: str) -> None:
        program = module.specialize(entrypoint)
        inspection = program.inspect()
        positions = [
            inspection.index(f"callee='argument_order.{callee}'") for callee in callees
        ]
        assert positions == sorted(positions)
        assert "evaluation_order" not in inspection

    assert_call_order("direct", "second_value", "first_value", "pair")
    assert_call_order("record", "second_value", "first_value")
    assert_call_order("higher_order_entry", "second_angle", "first_angle")
    assert_call_order("adjoint_entry", "second_angle", "first_angle", "rotate")
    assert_call_order(
        "control_entry", "second_angle", "component", "first_angle", "rotate"
    )
    assert_call_order("multiplex_entry", "second_angle", "first_angle")
    assert qm.evaluate(module.specialize("direct")) == {"first": 1, "second": 2}
    assert qm.evaluate(module.specialize("record")) == {"first": 1, "second": 2}
    assert qm.evaluate(module.specialize("higher_order_entry")) == (0,)
    assert qm.evaluate(module.specialize("adjoint_entry")) == (0,)
    assert qm.evaluate(module.specialize("control_entry")) == (0, 0, 0)
    assert qm.evaluate(module.specialize("multiplex_entry")) == (0, 0)


def test_named_tuple_defaults_are_portable_and_preserve_call_order() -> None:
    """Record defaults become ordinary literals after explicit operands."""

    source = """
from typing import NamedTuple

import qamomile as qm

class Settings(NamedTuple):
    count: qm.Int
    enabled: qm.Bool = True
    scale: qm.Real = 0.5

@qm.qkernel
def defaulted() -> Settings:
    return Settings(scale=2.0, count=4)

@qm.qkernel
def overridden() -> Settings:
    return Settings(5, False, 1.25)
"""
    module = qm.build_source(source, module_name="record_defaults")
    defaulted = qm.Program.from_bytes(module.specialize("defaulted").to_bytes())
    overridden = module.specialize("overridden")

    assert qm.evaluate(defaulted) == {
        "count": 4,
        "enabled": True,
        "scale": 2.0,
    }
    assert qm.evaluate(overridden) == {
        "count": 5,
        "enabled": False,
        "scale": 1.25,
    }
    assert "evaluation_order" not in defaulted.inspect()


@pytest.mark.parametrize(
    ("fields", "message"),
    (
        (
            "first: qm.Int = 1\n    second: qm.Int",
            "without a default follows a field with a default",
        ),
        (
            "value: qm.Int = DEFAULT",
            "default for record field 'value' is not a portable literal",
        ),
        ("enabled: qm.Bool = 1", "expected Bool, got Int"),
    ),
)
def test_named_tuple_defaults_reject_ambiguous_or_nonportable_values(
    fields: str,
    message: str,
) -> None:
    """Record declarations reject defaults that cannot enter portable HIR."""

    source = f"""
from typing import NamedTuple

import qamomile as qm

DEFAULT = 1

class Settings(NamedTuple):
    {fields}
"""

    with pytest.raises((qm.FrontendError, qm.TypeCheckError), match=message):
        qm.build_source(source)


def test_function_value_keywords_fail_closed_at_the_source_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown, duplicate, missing, mismatched, and unavailable binding fails early."""

    template = """
import qamomile as qm

@qm.qkernel
def add(left: qm.Int, right: qm.Int) -> qm.Int:
    return left + right

@qm.qkernel
def choose(flag: qm.Param[qm.Bool], value: qm.Int) -> qm.Int:
    operation = add if flag else add
    return CALL
"""
    failures = (
        ("operation(left=value, unknown=value)", "unknown function parameter"),
        ("operation(value, left=value)", "supplied twice"),
        ("operation(left=value)", "missing argument"),
        ("operation(left=False, right=value)", "does not match its declared type"),
    )
    for call, message in failures:
        with pytest.raises(qm.FrontendError, match=message):
            qm.build_source(template.replace("CALL", call))

    incompatible_labels = """
import qamomile as qm

@qm.qkernel
def left(value: qm.Int) -> qm.Int:
    return value

@qm.qkernel
def right(item: qm.Int) -> qm.Int:
    return item

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Int:
    return (left if flag else right)(0)
"""
    with pytest.raises(qm.TypeCheckError, match="incompatible parameter names"):
        qm.build_source(incompatible_labels)

    from qamomile._native import compiler as native

    monkeypatch.setattr(native, "infer_function_application", None)
    with pytest.raises(TypeError, match="Q4511"):
        qm.build_source(template.replace("CALL", "operation(value, right=value)"))


def test_function_alias_selection_is_materialized_at_assignment() -> None:
    """Rebinding a selector cannot retroactively change an assigned callable."""

    rebound_dependency = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def choose(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    operation = flipped if flag else unchanged
    flag = not flag
    return qm.measure(operation(qm.qubits(1)))
"""
    rebound = qm.build_source(rebound_dependency).specialize("choose")
    assert qm.evaluate(rebound, flag=False) == (0,)
    assert qm.evaluate(rebound, flag=True) == (1,)

    loop_carried = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def choose_for(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    operation = unchanged
    state = qm.qubits(1)
    for index in range(2):
        operation = flipped if flag and index == 0 else operation
        state = operation(state)
    state = operation(state)
    return qm.measure(state)

@qm.qkernel
def choose_break(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    operation = unchanged
    for _ in range(2):
        operation = flipped if flag else operation
        break
    return qm.measure(operation(qm.qubits(1)))
"""
    module = qm.build_source(loop_carried, module_name="loop_carried_function_aliases")
    for entrypoint, gate_range in {
        "choose_for": (0, 3),
        "choose_break": (0, 1),
    }.items():
        program = module.specialize(entrypoint)
        restored = qm.Program.from_bytes(program.to_bytes())
        assert qm.evaluate(program, flag=False) == (0,)
        assert qm.evaluate(program, flag=True) == (1,)
        assert qm.evaluate(restored, flag=False) == (0,)
        assert qm.evaluate(restored, flag=True) == (1,)
        assert (
            qm.resources.logical(program)
            .metric(qm.resources.LOGICAL_GATES)
            .numeric_enclosure()
            == gate_range
        )
        inspection = program.inspect()
        assert "QKernel[" not in inspection
        assert "function_ref" not in inspection
        assert "qcore.call" in inspection


def test_build_does_not_execute_module_or_decorated_body() -> None:
    """Source compilation must remain independent of Python execution."""

    source = """
import qamomile as qm

raise RuntimeError("the source module must never execute")

@qm.qkernel
def answer() -> qm.Int:
    raise RuntimeError("the decorated body must never execute")
    return 42
"""
    with pytest.raises(qm.FrontendError, match="Raise"):
        # The body is parsed and rejected as source syntax, proving that the
        # RuntimeError itself was not raised by Python execution.
        qm.build_source(source)


def test_ordinary_python_control_flow_is_structured_and_executable() -> None:
    """For, if, break, continue, and loop-else retain Python semantics."""

    source = """
import qamomile as qm

@qm.qkernel
def control(limit: qm.Const[qm.Int]) -> qm.Int:
    total = 0
    for index in range(limit):
        if index == 2:
            continue
        total += index
    else:
        total += 10
    for _ in range(13):
        if total >= 13:
            break
        total += 1
    return total
"""
    module = qm.build_source(source, module_name="control_example")
    program = module.specialize("control", limit=4)

    assert "qcore.for" in module.inspect()
    assert qm.evaluate(program) == 14


def test_identity_comparison_is_rejected_as_nonportable() -> None:
    """Python object identity must not leak into persistent semantics."""

    source = """
import qamomile as qm

@qm.qkernel
def identity(value: qm.Int) -> qm.Bool:
    return value is 1
"""
    with pytest.raises(qm.FrontendError, match="object identity"):
        qm.build_source(source)


def test_unresolved_python_global_is_not_captured() -> None:
    """Opaque module globals must be explicit portable declarations."""

    source = """
import qamomile as qm

opaque = object()

@qm.qkernel
def leak() -> qm.Int:
    return opaque
"""
    with pytest.raises(qm.FrontendError, match="not a local"):
        qm.build_source(source)


def test_function_factory_closes_callable_aliases_and_literal_values() -> None:
    """Host composition becomes ordinary linked calls, not a second QCore feature."""

    @qm.qkernel
    def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
        return qm.x(state)

    def repeat(operation: qm.QKernel, rounds: int) -> qm.QKernel:
        @qm.qkernel
        def repeated(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
            for _ in range(rounds):
                state = operation(state)
            return state

        return repeated

    once = repeat(flip, 1)
    twice = repeat(flip, 2)

    @qm.qkernel
    def execute() -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[1]]]:
        left = once(qm.prepare((0,)))
        right = twice(qm.prepare((0,)))
        return qm.measure(left), qm.measure(right)

    assert qm.evaluate(execute.specialize()) == ((1,), (0,))
    assert qm.resources.logical(execute).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (3, 3)
    assert once.function_name != twice.function_name
    assert "operation" not in twice.source_text
    assert "range(2)" in twice.source_text


def test_function_ref_closes_a_separately_imported_module_dependency() -> None:
    """Primary function UX links imported declarations into portable source."""

    from ._imported_function_fixture import imported_flip

    @qm.qkernel
    def imported_composition() -> qm.Shot[qm.Bits[1]]:
        state = imported_flip(qm.qubits(1))
        return qm.measure(state)

    program = imported_composition.specialize()
    restored = qm.Program.from_bytes(program.to_bytes())

    assert qm.evaluate(restored) == (1,)
    assert "imported_flip" in imported_composition.source_text
    assert "_imported_function_fixture" not in imported_composition.source_text
    assert restored.to_bytes() == program.to_bytes()


def test_function_factory_closes_immutable_callable_products() -> None:
    """A host tuple of functions becomes one linked total multiplex expression."""

    @qm.qkernel
    def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
        return state

    @qm.qkernel
    def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
        return qm.x(state)

    cases = (unchanged, flipped)

    @qm.qkernel
    def selected() -> qm.Shot[qm.Bits[2]]:
        index, target = qm.split_at(qm.prepare((1, 0)), 1)
        index, target = qm.multiplex(index, cases, target)
        return qm.measure(qm.concat(index, target))

    program = selected.specialize()
    assert qm.evaluate(program) == (1, 1)
    assert "qcore.select_call" in program.inspect()
    assert "cases" not in selected.source_text


def test_multiplex_rejects_an_incomplete_case_row() -> None:
    """A two-bit coherent index requires exactly four compatible cases."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def invalid() -> qm.Q[qm.Bits[3]]:
    index = qm.qubits(2)
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (unchanged, flipped), target)
    return qm.concat(index, target)
"""
    with pytest.raises(qm.FrontendError, match="Q5093:.*requires exactly 4 cases"):
        qm.build_source(source)


def test_aggregate_loops_thread_internal_values_through_nested_control() -> None:
    """Lowering-owned collection carries survive nested Python if/for regions."""

    source = """
import qamomile as qm

@qm.qkernel
def aggregate_control() -> qm.Int:
    total = 0
    for pair in ((1, True), (2, False), (3, True)):
        value, include = pair
        if include:
            total += value
    for row in ((4, 5), (6, 7)):
        for value in row:
            total += value
    return total
"""
    program = qm.build_source(source).specialize("aggregate_control")

    assert qm.evaluate(program) == 26
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()


def test_staged_record_iteration_retains_param_across_structured_exit() -> None:
    """Record elements retain Param availability across a structured exit."""

    source = """
from typing import NamedTuple

import qamomile as qm


class Pair(NamedTuple):
    left: qm.Int
    right: qm.Int


@qm.qkernel
def positive_prefix_total(pair: qm.Param[Pair]) -> qm.Int:
    result = 0
    for value in pair:
        if value < 0:
            return result
        result += value
    return result
"""
    program = qm.build_source(source).specialize("positive_prefix_total")

    assert qm.evaluate(program, pair={"left": 4, "right": 5}) == 9
    assert qm.evaluate(program, pair={"left": 4, "right": -1}) == 4
    assert Requirement.HOST_DISPATCH in program.entry.requirements


def test_function_factory_closes_dimensions_used_only_by_annotations() -> None:
    """Dependent source shapes need no dummy reference in the function body."""

    def identity(width: int) -> qm.QKernel:
        @qm.qkernel
        def fixed(
            state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            return state

        return fixed

    identity_three = identity(3)

    assert "qm.Q[qm.Bits[3]]" in identity_three.source_text
    assert "width" not in identity_three.source_text
    assert (
        identity_three.module().function(identity_three.function_name).result_type
        == (qm.Q[qm.Bits[3]])
    )


def test_function_factory_rejects_an_opaque_capture_eagerly() -> None:
    """Closure conversion admits data, not arbitrary live Python objects."""

    opaque = object()

    with pytest.raises(qm.SourceUnavailableError, match="portable immutable literal"):

        @qm.qkernel
        def invalid() -> qm.Int:
            return opaque  # type: ignore[return-value]


def test_const_specialization_is_explicit_typed_and_canonical() -> None:
    """Only Const inputs specialize and defaults become canonical bindings."""

    source = """
import qamomile as qm

@qm.qkernel
def choose(flag: qm.Const[qm.Bool] = False) -> qm.Int:
    if flag:
        return 1
    return 2

@qm.qkernel
def choose_at_launch(flag: qm.Param[qm.Bool]) -> qm.Int:
    if flag:
        return 1
    return 2
"""
    module = qm.build_source(source)
    defaulted = module.specialize("choose")
    explicit = module.specialize("choose", flag=False)

    assert defaulted.to_bytes() == explicit.to_bytes()
    assert "qcore.if" not in defaulted.inspect()
    assert qm.evaluate(defaulted) == 2
    with pytest.raises(qm.SpecializationError, match="unknown Const"):
        module.specialize("choose_at_launch", flag=True)
    with pytest.raises(qm.SpecializationError, match=r"expects Const\[Bool\]"):
        module.specialize("choose", flag="false")


@pytest.mark.parametrize(
    "body",
    [
        "state = qm.h(state, typo=0)",
        "state = qm.h(state, 0, 1)",
        "state = qm.rx(state)",
        "state = qm.cx(state, control=0, target=0)",
    ],
)
def test_gate_schema_errors_are_reported_at_the_source(body: str) -> None:
    """Primitive gate arity and selectors never drift to backend failures."""

    source = f"""
import qamomile as qm

@qm.qkernel
def invalid(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    {body}
    return state
"""
    with pytest.raises(qm.FrontendError):
        qm.build_source(source)


def test_gate_default_selectors_are_validated_from_the_native_catalog() -> None:
    """An omitted selector retains the Rust-declared lane during source checks."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.cx(state)
"""
    with pytest.raises(qm.FrontendError, match="Q5075:.*target=1.*width 1"):
        qm.build_source(source)


def test_unused_quantum_transform_is_rejected() -> None:
    """Forgetting to rebind an owned transform is not silently uncomputed."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Unit:
    state = qm.qubits(1)
    qm.h(state)
    return None
"""
    with pytest.raises(qm.FrontendError, match="unused expression"):
        qm.build_source(source)


def test_explicit_adjoint_is_a_typed_function_application() -> None:
    """Users can apply a unitary inverse without exposing an IR operation."""

    source = """
import qamomile as qm

@qm.qkernel
def prepare_one(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    state = qm.h(state)
    state = qm.t(state)
    return state

@qm.qkernel
def undo() -> qm.Bits[1]:
    state = qm.qubits(1)
    state = prepare_one(state)
    state = qm.adjoint(prepare_one, state)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="explicit_inverse").specialize("undo")

    assert "qcore.adjoint_call" in program.inspect()
    assert qm.evaluate(program, seed=4) == (0,)
    assert qm.run(program, on=qm.Reference(), shots=32, seed=4).result.counts == {
        (0,): 32
    }


def test_pauli_measurement_is_typed_nondestructive_and_repeatable() -> None:
    """A basis measurement returns a Shot bit while retaining the collapsed owned state."""

    program = qm.build_source(MEASURE_AXES_SOURCE).specialize("measure_axes")

    assert qm.evaluate(program, seed=2) == (
        True,
        True,
        False,
        False,
        False,
        False,
        (1,),
        (0,),
        (0,),
    )
    assert Requirement.MID_CIRCUIT_MEASUREMENT in program.entry.requirements
    assert Effect.MEASUREMENT in program.entry.effects


def test_basis_measurement_requires_a_target_for_multi_qubit_state() -> None:
    """A scalar basis measurement cannot silently select lane zero."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Shot[qm.Bool]:
    bit, state = qm.measure_x(qm.qubits(2))
    qm.discard(state)
    return bit
"""
    with pytest.raises(qm.FrontendError, match=r"measure_x requires target="):
        qm.build_source(source)


def test_explicit_adjoint_accepts_inference_and_rejects_channel_bodies() -> None:
    """Adjointability is inferred from a body rather than a decorator spelling."""

    source = """
import qamomile as qm

@qm.qkernel
def helper(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def valid() -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.adjoint(helper, state)
    return qm.measure(state)
"""
    assert qm.evaluate(qm.build_source(source).specialize("valid")) == (1,)

    invalid = source.replace("return qm.x(state)", "return qm.reset(state)")
    with pytest.raises(qm.VerificationError, match="invalid adjoint callable"):
        qm.build_source(invalid)


def test_measure_reset_returns_observation_and_clean_owned_state() -> None:
    """Destructive observation is explicit while ownership remains functional."""

    program = qm.build_source(MEASURE_RESET_SOURCE).specialize("measure_and_reset")
    study = qm.resources.logical(program)

    assert qm.evaluate(program) == ((1,), (0,))
    assert Effect.MEASUREMENT in program.entry.effects
    assert Effect.RESET in program.entry.effects
    assert Requirement.MEASUREMENT in program.entry.requirements
    assert Requirement.MID_CIRCUIT_MEASUREMENT in program.entry.requirements
    assert Requirement.RESET in program.entry.requirements
    assert study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
        2,
        2,
    )
    assert study.metric(qm.resources.LOGICAL_RESETS).numeric_enclosure() == (1, 1)


def test_break_skips_loop_else_and_reaches_trailing_return() -> None:
    """A loop-else return does not make a possible break path unreachable."""

    source = """
import qamomile as qm

@qm.qkernel
def loop(limit: qm.Const[qm.Int]) -> qm.Int:
    total = 0
    for index in range(limit):
        if index == 1:
            break
        total += index
    else:
        return 9
    return total + 1
"""
    module = qm.build_source(source)
    assert qm.evaluate(module.specialize("loop", limit=0)) == 9
    assert qm.evaluate(module.specialize("loop", limit=3)) == 1


def test_for_target_is_a_distinct_ssa_value_from_its_previous_binding() -> None:
    """The induction value shadows, then replaces, an existing Python local."""

    source = """
import qamomile as qm

@qm.qkernel
def nonempty() -> qm.Int:
    index = 5
    total = 0
    for index in range(2):
        total += index
    return 10 * total + index

@qm.qkernel
def empty() -> qm.Int:
    index = 5
    for index in range(0):
        pass
    return index
"""
    module = qm.build_source(source)
    assert qm.evaluate(module.specialize("nonempty")) == 11
    assert qm.evaluate(module.specialize("empty")) == 5


def test_for_iterates_fixed_python_aggregates_through_the_same_region_op() -> None:
    """List iteration reuses qcore.for instead of a collection-specific loop."""

    source = """
import qamomile as qm

@qm.qkernel
def total() -> qm.Int:
    result = 0
    for value in [1, 2, 3]:
        result += value
    return result
"""
    program = qm.build_source(source).specialize("total")
    assert qm.evaluate(program) == 6
    assert program.inspect().count("qcore.for") == 1


def test_enumerate_and_zip_lower_param_aggregates_through_structured_for() -> None:
    """Python collection loops retain Param values in one shared QCore loop form."""

    source = """
import qamomile as qm

@qm.qkernel
def enumerate_total(
    values: qm.Param[qm.Vector[qm.Int, 3]],
    start: qm.Param[qm.Int],
) -> qm.Int:
    result = 0
    for index, value in enumerate(values, start=start):
        result += index * value
    return result

@qm.qkernel
def zip_total(
    left: qm.Param[qm.Vector[qm.Int, 3]],
    right: qm.Param[qm.Vector[qm.Int, 3]],
) -> qm.Int:
    result = 0
    for first, second in zip(left, right):
        result += first * second
    return result

@qm.qkernel
def zip_reused(values: qm.Param[qm.Vector[qm.Int, 3]]) -> qm.Int:
    result = 0
    for first, second in zip(values, values):
        result += first * second
    return result
"""
    module = qm.build_source(source)
    enumerated = module.specialize("enumerate_total")
    zipped = module.specialize("zip_total")
    reused = module.specialize("zip_reused")

    assert qm.evaluate(enumerated, values=(2, 3, 4), start=5) == 56
    assert qm.evaluate(zipped, left=(1, 2, 3), right=(4, 5, 6)) == 32
    assert qm.evaluate(reused, values=(1, 2, 3)) == 14
    assert enumerated.inspect().count("qcore.for") == 1
    assert enumerated.inspect().count("classical.index") == 1
    assert "classical.add" in enumerated.inspect()
    assert zipped.inspect().count("qcore.for") == 1
    assert zipped.inspect().count("classical.index") == 2


def test_zip_rejects_unproven_python_shortest_iteration() -> None:
    """Different static extents fail instead of silently inventing zip truncation."""

    source = """
import qamomile as qm

@qm.qkernel
def mismatch() -> qm.Int:
    total = 0
    for left, right in zip([1, 2], [3, 4, 5]):
        total += left + right
    return total
"""

    with pytest.raises(qm.FrontendError, match="Q5085"):
        qm.build_source(source)


def test_collection_loops_do_not_fallback_from_native_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native coverage rejection cannot select a Python ownership lowerer."""

    from qamomile import _qamomile_native as native

    def unsupported(_payload: bytes) -> bytes:
        """Report one valid HIR construct outside native coverage."""

        raise ValueError("Q7101: simulated HIR coverage gap")

    monkeypatch.setattr(native, "lower_hir", unsupported)
    source = """
import qamomile as qm

@qm.qkernel
def fallback() -> qm.Int:
    total = 0
    for index, value in enumerate([2, 3], start=4):
        total += index * value
    for left, right in zip([1, 2], [3, 4]):
        total += left * right
    return total
"""

    with pytest.raises(qm.CapabilityError, match="Q7101"):
        qm.build_source(source)


def test_structural_math_is_typed_and_folded_by_rust() -> None:
    """Ceil and log2 retain Param staging and close before circuit lowering."""

    source = """
import qamomile as qm

@qm.qkernel
def width(size: qm.Param[qm.Int]) -> qm.Int:
    return qm.ceil(qm.log2(size))

@qm.qkernel
def signed_rounding() -> qm.Int:
    return qm.ceil(-1.25)
"""
    module = qm.build_source(source)
    width = module.specialize("width")

    assert width.inspect().count("classical.log2") == 1
    assert width.inspect().count("classical.ceil") == 1
    assert qm.Program.from_bytes(width.to_bytes()).to_bytes() == width.to_bytes()
    assert qm.evaluate(width, size=9) == 4
    assert qm.evaluate(width, size=1 << 4096) == 4096
    assert qm.evaluate(module.specialize("signed_rounding")) == -1


def test_structural_math_reports_domain_and_type_errors() -> None:
    """Invalid static domains fail explicitly instead of leaking backend math."""

    bad_domain = """
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Real:
    return qm.log2(0)
"""
    bad_type = """
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Real:
    return qm.log2(True)
"""

    with pytest.raises(qm.SpecializationError, match="Q3008.*strictly positive"):
        qm.build_source(bad_domain).specialize("invalid")
    with pytest.raises(qm.TypeCheckError, match="qm.log2 expects Int or Real"):
        qm.build_source(bad_type)


def test_missing_native_hir_entrypoint_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete required compiler cannot restore Python HIR semantics."""

    from qamomile import _qamomile_native as native

    monkeypatch.delattr(native, "lower_hir")
    source = """
import qamomile as qm

@qm.qkernel
def width(size: qm.Param[qm.Int]) -> qm.Int:
    return qm.ceil(qm.log2(size))
"""

    with pytest.raises(qm.CapabilityError, match="Q7100"):
        qm.build_source(source)


def test_hir_persistence_requires_the_native_sealer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed source HIR never selects a Python checksum implementation."""

    from qamomile import _qamomile_native as native

    original = native.seal_hir_package
    calls = 0

    def recording_seal(payload: bytes) -> object:
        """Record and forward one native HIR sealing request."""

        nonlocal calls
        calls += 1
        return original(payload)

    monkeypatch.setattr(native, "seal_hir_package", recording_seal)
    qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
""",
        module_name="native_hir_sealing",
    )
    assert calls == 1

    monkeypatch.setattr(native, "seal_hir_package", None)
    with pytest.raises(qm.CapabilityError, match="Q7100"):
        qm.build_source(
            """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
"""
        )

    monkeypatch.setattr(native, "seal_hir_package", lambda _payload: object())
    with pytest.raises(qm.VerificationError, match="Q7003"):
        qm.build_source(
            """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
"""
        )


def test_call_site_instantiates_classical_control_stage() -> None:
    """A Param argument makes helper control host-dispatched at its call site."""

    source = """
import qamomile as qm

@qm.qkernel
def choose(flag: qm.Bool) -> qm.Int:
    if flag:
        return 1
    return 2

@qm.qkernel
def main(flag: qm.Param[qm.Bool]) -> qm.Int:
    return choose(flag)
"""
    program = qm.build_source(source).specialize("main")
    assert Effect.HOST_CONTROL in program.entry.effects
    assert Requirement.HOST_DISPATCH in program.entry.requirements


def test_boolean_and_conditional_expressions_preserve_python_laziness() -> None:
    """Unselected partial expressions are represented by regions, not evaluated."""

    source = """
import qamomile as qm

@qm.qkernel
def short_and() -> qm.Bool:
    return False and (1 // 0 == 0)

@qm.qkernel
def short_or() -> qm.Bool:
    return True or (1 // 0 == 0)

@qm.qkernel
def conditional() -> qm.Int:
    return 1 if True else 1 // 0

@qm.qkernel
def runtime_conditional(flag: qm.Param[qm.Bool]) -> qm.Int:
    return 7 if flag else 1 // 0

@qm.qkernel
def captured_conditional(
    flag: qm.Param[qm.Bool],
    left: qm.Param[qm.Int],
    right: qm.Param[qm.Int],
) -> qm.Int:
    return left if flag else right

@qm.qkernel
def quantum_conditional(flag: qm.Const[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.x(state) if flag else qm.z(state)
    return qm.measure(state)
"""
    module = qm.build_source(source)
    assert qm.evaluate(module.specialize("short_and")) is False
    assert qm.evaluate(module.specialize("short_or")) is True
    assert qm.evaluate(module.specialize("conditional")) == 1
    assert (
        qm.run(
            module.specialize("runtime_conditional"),
            on=qm.Reference(),
            parameters={"flag": True},
        ).result.value
        == 7
    )
    assert (
        qm.run(
            module.specialize("captured_conditional"),
            on=qm.Reference(),
            parameters={"flag": False, "left": 2, "right": 3},
        ).result.value
        == 3
    )
    assert qm.evaluate(module.specialize("quantum_conditional", flag=True)) == (1,)
    assert module.inspect().count("qcore.if") == 6


def test_annotated_module_constant_must_match_its_logical_type() -> None:
    """Portable constant annotations cannot certify a mismatched literal."""

    source = """
import qamomile as qm

BAD: qm.Bool = 3

@qm.qkernel
def main() -> qm.Bool:
    return BAD
"""
    with pytest.raises(qm.TypeCheckError):
        qm.build_source(source)


def test_union_annotation_requires_real_tagged_sum_semantics() -> None:
    """A Python union must not masquerade as an inhabitable product record."""

    source = """
import qamomile as qm

@qm.qkernel
def ambiguous(value: int | None) -> qm.Int:
    return 0
"""
    with pytest.raises(qm.FrontendError, match="Q5063"):
        qm.build_source(source)


def test_const_dimensions_specialize_launch_parameter_shapes() -> None:
    """Dependent ABI dimensions become concrete before artifact execution."""

    source = """
import qamomile as qm

@qm.qkernel
def width(
    size: qm.Const[qm.Int],
    bits: qm.Param[qm.Bits[size]],
) -> qm.Int:
    return len(bits)

@qm.qkernel
def allocate(size: qm.Const[qm.Int]) -> qm.Unit:
    state = qm.qubits(size)
    qm.discard(state)
    return None
"""
    module = qm.build_source(source)
    program = module.specialize("width", size=2)

    assert "Param[Bits[2]]" in program.inspect()
    assert qm.evaluate(program, bits=(0, 1)) == 2
    with pytest.raises(qm.ExecutionError, match=r"expects Param\[Bits\[2\]\]"):
        qm.evaluate(program, bits=(0, 1, 0))
    with pytest.raises(qm.SpecializationError, match="must be positive"):
        module.specialize("allocate", size=0)


def test_unstaged_helper_parameters_are_stage_polymorphic() -> None:
    """Pure helpers inherit call-site availability without duplicating source."""

    source = """
import qamomile as qm

@qm.qkernel
def choose(flag: qm.Bool) -> qm.Int:
    if flag:
        return 1
    return 0

@qm.qkernel
def launch(flag: qm.Param[qm.Bool]) -> qm.Int:
    return choose(flag)
"""
    program = qm.build_source(source).specialize("launch")

    assert qm.evaluate(program, flag=False) == 0
    assert qm.evaluate(program, flag=True) == 1
    assert Effect.HOST_CONTROL in program.entry.effects
    assert Requirement.HOST_DISPATCH in program.entry.requirements


def test_branch_join_lifts_earlier_values_to_common_availability() -> None:
    """A branch may safely lift an ordinary value to a Shot result stage."""

    source = """
import qamomile as qm

@qm.qkernel
def choose(flag: qm.Bool, observed: qm.Shot[qm.Int]) -> qm.Shot[qm.Int]:
    if flag:
        result = 0
    else:
        result = observed
    return result
"""
    program = qm.build_source(source).specialize("choose")

    assert "qcore.stage_lift" in program.inspect()
    assert qm.evaluate(program, flag=True, observed=7) == 0
    assert qm.evaluate(program, flag=False, observed=7) == 7


def test_stage_polymorphism_does_not_invent_unrelated_host_control() -> None:
    """A Param argument does not taint a callee's independent static loop."""

    source = """
import qamomile as qm

@qm.qkernel
def scale(value: qm.Real) -> qm.Real:
    result = value
    for _ in range(2):
        result = result + value
    return result

@qm.qkernel
def launch(value: qm.Param[qm.Real]) -> qm.Real:
    return scale(value)
"""
    program = qm.build_source(source).specialize("launch")

    assert Effect.HOST_CONTROL not in program.entry.effects
    assert Requirement.HOST_DISPATCH not in program.entry.requirements


def test_unchanged_loop_carries_do_not_taint_later_control() -> None:
    """Sequential loops preserve each forwarded carry's own dependency."""

    source = """
import qamomile as qm

@qm.qkernel
def layers(
    state: qm.Q[qm.Bits[width]],
    width: qm.Const[qm.Int],
    angle: qm.Real,
) -> qm.Q[qm.Bits[width]]:
    for index in range(width):
        state = qm.rz(state, angle, target=index)
    for index in range(width):
        state = qm.rx(state, angle, target=index)
    return state

@qm.qkernel
def launch(angle: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(layers(qm.qubits(2), 2, angle))
"""
    program = qm.build_source(source).specialize("launch")

    assert Effect.HOST_CONTROL not in program.entry.effects
    assert Requirement.HOST_DISPATCH not in program.entry.requirements


def test_static_calls_discharge_internal_parameterized_execution() -> None:
    """Only the selected entrypoint ABI imposes a runtime parameter contract."""

    source = """
import qamomile as qm

@qm.qkernel
def rotate(
    state: qm.Q[qm.Bits[1]],
    angle: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[1]]:
    return qm.rx(state, angle)

@qm.qkernel
def launch() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(rotate(qm.qubits(1), 0.0))
"""
    program = qm.build_source(source).specialize("launch")

    assert Requirement.PARAMETERIZED_EXECUTION not in program.entry.requirements


def test_total_boolean_expressions_preserve_python_spelling() -> None:
    """Boolean chains, comparison chains, and conditionals compile as pure SSA."""

    source = """
import qamomile as qm

@qm.qkernel
def logic(a: qm.Param[qm.Bool], b: qm.Param[qm.Bool], x: qm.Param[qm.Int]) -> qm.Int:
    flag = a and not b
    value = 1 if flag else 2
    if 0 < x < 3 and flag:
        return value
    return 0
"""
    program = qm.build_source(source).specialize("logic")

    assert qm.evaluate(program, a=True, b=False, x=2) == 1
    assert qm.evaluate(program, a=True, b=True, x=2) == 0
    assert qm.evaluate(program, a=True, b=False, x=4) == 0


def test_aggregate_comparison_observes_nested_parameter_availability() -> None:
    """Comparing a parameterized aggregate produces a parameterized Bool."""

    source = """
import qamomile as qm

@qm.qkernel
def compare_words(
    left: qm.Param[qm.UInt[3]],
    right: qm.Param[qm.UInt[3]],
) -> qm.Param[qm.Bool]:
    return (left, right) == (left, right)
"""
    program = qm.build_source(source).specialize("compare_words")

    assert str(program.entry.result_type) == "Param[Bool]"
    assert qm.evaluate(program, left=5, right=0) is True


def test_fixed_aggregate_iteration_lowers_to_structured_for() -> None:
    """Ordinary iteration over a fixed typed aggregate executes portably."""

    source = """
import qamomile as qm

@qm.qkernel
def total() -> qm.Int:
    values = [1, 2, 3, 4]
    result = 0
    for value in values:
        result += value
    return result
"""
    program = qm.build_source(source).specialize("total")

    assert "qcore.for" in program.inspect()
    assert qm.evaluate(program) == 10


def test_immutable_typed_map_iterates_and_looks_up_during_specialization() -> None:
    """Map syntax lowers to fixed records with Rust-owned lookup semantics."""

    source = """
import qamomile as qm

@qm.qkernel
def total(
    size: qm.Const[qm.Int],
    left: qm.Const[qm.Map[tuple[qm.Int, qm.Int], qm.Real, size]],
    right: qm.Const[qm.Map[tuple[qm.Int, qm.Int], qm.Real, size]],
) -> qm.Real:
    result = 0.0
    for key, value in left.items():
        result += value * right[key]
    return result
"""
    module = qm.build_source(source, module_name="typed_map")
    program = module.specialize(
        "total",
        **{
            "left": {(0, 1): 2.0, (1, 2): 3.0},
            "right": {(0, 1): 4.0, (1, 2): 5.0},
        },
    )
    restored = qm.Program.from_bytes(program.to_bytes())

    assert "classical.map_get" in module.inspect()
    assert qm.evaluate(program) == 23.0
    assert qm.evaluate(restored) == 23.0
    assert program.to_bytes() == restored.to_bytes()


def test_typed_map_rejects_runtime_mutable_mapping_semantics() -> None:
    """Map parameters are Const so lookup cannot leak into backend execution."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid(values: qm.Param[qm.Map[qm.Int, qm.Real, 2]]) -> qm.Real:
    return values[0]
"""

    with pytest.raises(qm.FrontendError, match=r"Map parameters must use Const"):
        qm.build_source(source)

    mutation = """
import qamomile as qm

@qm.qkernel
def invalid(values: qm.Const[qm.Map[qm.Int, qm.Real, 2]]) -> qm.Real:
    values[0] = 2.0
    return values[0]
"""
    with pytest.raises(qm.FrontendError, match=r"Q5057.*only local names"):
        qm.build_source(mutation)


def test_typed_map_reports_a_missing_key_from_native_evaluation() -> None:
    """Lookup failure remains a compiler diagnostic, never Python dict fallback."""

    source = """
import qamomile as qm

@qm.qkernel
def lookup(
    size: qm.Const[qm.Int],
    keys: qm.Const[qm.Map[qm.Int, qm.Int, size]],
    values: qm.Const[qm.Map[qm.Int, qm.Int, size]],
) -> qm.Int:
    result = 0
    for key, _ in keys.items():
        result += values[key]
    return result
"""
    program = qm.build_source(source).specialize(
        "lookup",
        **{"keys": {0: 1}, "values": {1: 2}},
    )

    with pytest.raises(qm.ExecutionError, match=r"Q6204.*mapping key is missing"):
        qm.evaluate(program)


@pytest.mark.parametrize("size", [1024, 1025, 4096])
def test_large_typed_map_lookup_specializes_and_round_trips(size: int) -> None:
    """Map lookup follows package budgets rather than a separate entry cap."""

    source = """
import qamomile as qm

@qm.qkernel
def lookup(
    size: qm.Const[qm.Int],
    values: qm.Const[qm.Map[qm.Int, qm.Int, size]],
) -> qm.Int:
    return values[0] + values[size - 1]
"""
    module = qm.build_source(source)

    program = module.specialize("lookup", values=dict(enumerate(range(size))))
    restored = qm.Program.from_bytes(program.to_bytes())

    assert dict(program.const_bindings)["size"] == size
    assert qm.evaluate(restored) == size - 1
    assert restored.to_bytes() == program.to_bytes()
    assert "classical.map_get" not in program.inspect()


@pytest.mark.parametrize("backend", ["reference", "qiskit"])
def test_large_typed_map_drives_one_quantum_gate(backend: str) -> None:
    """A 1025-entry coefficient table selects the last angle without expansion."""
    from qamomile.qiskit import Aer

    source = """
import qamomile as qm

@qm.qkernel
def sample(angles: qm.Const[qm.Map[qm.Int, qm.Real, 1025]]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.rx(qm.qubits(1), angles[1024]))
"""
    angles = dict.fromkeys(range(1025), 0.0)
    angles[1024] = 3.141592653589793
    program = qm.build_source(source).specialize("sample", angles=angles)
    restored = qm.Program.from_bytes(program.to_bytes())
    endpoint = qm.Reference() if backend == "reference" else Aer()
    assert qm.run(restored, on=endpoint, shots=8, seed=7).result.counts == {(1,): 8}
    assert qm.resources.logical(restored).metric(
        qm.resources.LOGICAL_GATES
    ).numeric_enclosure() == (1, 1)


def test_large_typed_map_still_rejects_duplicate_keys_at_native_selection() -> None:
    """An untrusted table above the former cap must still have unique keys."""
    from qamomile import _qamomile_native as native
    from qamomile.ir import FrozenMapping
    from qamomile.serialization import encode_selection_request

    source = """
import qamomile as qm

@qm.qkernel
def lookup(values: qm.Const[qm.Map[qm.Int, qm.Int, 1025]]) -> qm.Int:
    return values[0]
"""
    module = qm.build_source(source)
    entries = tuple(
        FrozenMapping.from_mapping({"key": index, "value": index})
        for index in (*range(1024), 0)
    )
    request = encode_selection_request(module, "lookup", (("values", entries),))

    with pytest.raises(ValueError, match=r"Q3007.*duplicate key"):
        native.prepare_program(request)


def test_matrix_is_a_rank_two_structural_array() -> None:
    """Matrix and nested-vector spelling share one rank-aware semantic type."""

    assert qm.Matrix[qm.Real, 2, 3] == qm.Vector[qm.Vector[qm.Real, 3], 2]
    assert str(qm.Vector[qm.Matrix[qm.Int, 2, 3], 4]) == "Tensor[Int, 4, 2, 3]"
    assert isinstance(qm.Matrix[qm.Real, 2, 3], ArrayType)
    assert isinstance(qm.Bits[3], WordType)
    assert qm.Bits[3] != qm.Vector[qm.Bit, 3]
    assert str(qm.Map[qm.Int, qm.Real, 2]) == "Map[Int, Real, 2]"


def test_unsigned_fixed_point_is_data_under_the_quantum_modality() -> None:
    """Fixed point needs no special quantum handle or measurement primitive."""

    assert str(qm.UFixed[2, 3]) == "UFixed[2, 3]"
    assert str(qm.Q[qm.UFixed[0, 3]]) == "Q[UFixed[0, 3]]"
    with pytest.raises(qm.TypeCheckError, match="at least one bit"):
        qm.UFixed[0, 0]


def test_structured_control_tracks_only_real_control_dependencies() -> None:
    """Changed values join control stage while identical captures pass through."""

    source = """
import qamomile as qm

@qm.qkernel
def choose(flag: qm.Param[qm.Bool], unchanged: qm.Int) -> tuple[qm.Int, qm.Int]:
    changed = 0
    if flag:
        changed = 1
    else:
        changed = 2
    return changed, unchanged
"""
    entry = qm.build_source(source).function("choose")
    conditional = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.if"
    )
    result_types = {result.hint: result.type for result in conditional.results}

    assert result_types["changed"] == qm.Param[qm.Int]
    assert result_types["unchanged"] == qm.Int


def test_for_stage_inference_uses_the_bound_availability() -> None:
    """A parameterized bound raises bounded-loop carries to Param."""

    source = """
import qamomile as qm

@qm.qkernel
def count(limit: qm.Param[qm.Int]) -> qm.Int:
    value = 0
    for _ in range(limit):
        value += 1
    return value
"""
    module = qm.build_source(source)
    for_op = next(
        operation
        for operation in module.function("count").body.operations
        if operation.opcode == "qcore.for"
    )

    assert for_op.attribute("stage") == "Param"
    assert for_op.results[0].type == qm.Param[qm.Int]


def test_call_stage_preserves_callee_availability_and_aggregate_bounds() -> None:
    """Call typing joins argument stages without erasing an explicit Shot result."""

    source = """
import qamomile as qm

@qm.qkernel
def observe() -> qm.Shot[qm.Bool]:
    return qm.measure(qm.prepare(True))

@qm.qkernel
def forward() -> qm.Shot[qm.Bool]:
    return observe()
"""
    entry = qm.build_source(source).function("forward")
    call = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.call"
    )

    assert call.results[0].type == qm.Shot[qm.Bool]
    assert is_type_assignable(
        tuple[qm.Int, qm.Int],
        tuple[qm.Shot[qm.Int], qm.Param[qm.Int]],
    )
    assert not is_type_assignable(
        qm.Param[tuple[qm.Int, qm.Int]],
        tuple[qm.Shot[qm.Int], qm.Param[qm.Int]],
    )


def test_matrix_access_supports_rows_and_python_multi_indices() -> None:
    """One index removes one dimension and ``matrix[i, j]`` composes accesses."""

    source = """
import qamomile as qm

@qm.qkernel
def element(
    values: qm.Param[qm.Matrix[qm.Real, 2, 3]],
    row: qm.Param[qm.Int],
    column: qm.Param[qm.Int],
) -> qm.Real:
    selected_row = values[row]
    return selected_row[column] + values[row, column]
"""
    program = qm.build_source(source).specialize("element")

    assert (
        qm.evaluate(
            program,
            values=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
            row=1,
            column=2,
        )
        == 12.0
    )
    assert program.entry.parameters[0].value.type == qm.Param[qm.Matrix[qm.Real, 2, 3]]


def test_tensor_access_closes_the_rank_reduction_rule() -> None:
    """Rank-three annotations use the same comma-index syntax to a scalar."""

    source = """
import qamomile as qm

@qm.qkernel
def element(
    values: qm.Param[qm.Tensor[qm.Int, 2, 2, 2]],
) -> qm.Int:
    return values[1, 0, 1]
"""
    program = qm.build_source(source).specialize("element")

    assert (
        qm.evaluate(
            program,
            values=(((0, 1), (2, 3)), ((4, 5), (6, 7))),
        )
        == 5
    )


def test_nested_list_infers_matrix_and_round_trips_records() -> None:
    """Rectangular nested lists infer Matrix without losing record elements."""

    source = """
from typing import NamedTuple
import qamomile as qm

class Cell(NamedTuple):
    value: qm.Int

@qm.qkernel
def table() -> qm.Matrix[Cell, 2, 2]:
    return [[Cell(1), Cell(2)], [Cell(3), Cell(4)]]
"""
    program = qm.Program.from_bytes(
        qm.build_source(source).specialize("table").to_bytes()
    )

    assert tuple(tuple(dict(cell) for cell in row) for row in qm.evaluate(program)) == (
        ({"value": 1}, {"value": 2}),
        ({"value": 3}, {"value": 4}),
    )


def test_fixed_array_slices_lower_to_existing_index_and_array_hir() -> None:
    """Exact slices retain rank and staging without a persistent slice opcode."""

    source = """
import qamomile as qm

@qm.qkernel
def reverse_stride(
    values: qm.Param[qm.Vector[qm.Int, 5]],
) -> qm.Param[qm.Vector[qm.Int, 3]]:
    return values[::-2]

@qm.qkernel
def select_rows(
    values: qm.Param[qm.Matrix[qm.Int, 4, 2]],
) -> qm.Param[qm.Matrix[qm.Int, 2, 2]]:
    return values[1:4:2]

@qm.qkernel
def make_values(offset: qm.Param[qm.Int]) -> qm.Param[qm.Vector[qm.Int, 5]]:
    return [offset, offset + 1, offset + 2, offset + 3, offset + 4]

@qm.qkernel
def slice_call(offset: qm.Param[qm.Int]) -> qm.Param[qm.Vector[qm.Int, 3]]:
    return make_values(offset)[::-2]
"""
    module = qm.build_source(source, module_name="fixed_slices")
    vector = qm.Program.from_bytes(module.specialize("reverse_stride").to_bytes())
    matrix = module.specialize("select_rows")
    called = module.specialize("slice_call")

    assert qm.evaluate(vector, values=(0, 1, 2, 3, 4)) == (4, 2, 0)
    assert qm.evaluate(
        matrix,
        values=((0, 1), (2, 3), (4, 5), (6, 7)),
    ) == ((2, 3), (6, 7))
    assert qm.evaluate(called, offset=10) == (14, 12, 10)
    assert "classical.slice" not in vector.inspect()
    assert "static_slice" not in vector.inspect()
    assert vector.inspect().count("classical.index") == 3
    assert called.inspect().count("qcore.call") == 1


@pytest.mark.parametrize(
    ("slice_expression", "message"),
    (
        ("values[start:]", "slice start must be an exact integer literal"),
        ("values[::0]", "slice step cannot be zero"),
        ("values[1:1]", "empty slice has no nonempty fixed-array type"),
        ("values[1:3, 0]", "fixed array slice must be the only subscript"),
    ),
)
def test_fixed_array_slices_reject_dynamic_or_unrepresentable_shapes(
    slice_expression: str,
    message: str,
) -> None:
    """Slices fail before HIR when their result shape is not fixed and nonempty."""

    source = f"""
import qamomile as qm

@qm.qkernel
def sliced(
    values: qm.Param[qm.Vector[qm.Int, 4]],
    start: qm.Param[qm.Int],
) -> qm.Param[qm.Vector[qm.Int, 2]]:
    return {slice_expression}
"""

    with pytest.raises(qm.FrontendError, match=message):
        qm.build_source(source)


def test_matrix_iteration_yields_typed_rows() -> None:
    """Nested ordinary for loops follow the same rank-reducing index rule."""

    source = """
import qamomile as qm

@qm.qkernel
def total(values: qm.Param[qm.Matrix[qm.Int, 2, 3]]) -> qm.Int:
    result = 0
    for row in values:
        for value in row:
            result += value
    return result
"""
    program = qm.build_source(source).specialize("total")

    assert qm.evaluate(program, values=((1, 2, 3), (4, 5, 6))) == 21


def test_matrix_row_cannot_masquerade_as_a_scalar() -> None:
    """The frontend rejects the historical Matrix-to-scalar type confusion."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid(values: qm.Matrix[qm.Real, 2, 2]) -> qm.Real:
    return values[0]
"""

    with pytest.raises(qm.TypeCheckError, match=r"expected Real, got Vector"):
        qm.build_source(source)


def test_symbolic_size_record_table_replaces_specialized_dict_iteration() -> None:
    """Normal Python iterates an immutable coefficient table with a Const extent."""

    source = """
import qamomile as qm
from typing import NamedTuple

class Term(NamedTuple):
    left: qm.Int
    right: qm.Int
    weight: qm.Real

@qm.qkernel
def ising(
    n_terms: qm.Const[qm.Int],
    terms: qm.Const[qm.Vector[Term, n_terms]],
    gamma: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[3]]:
    state = qm.qubits(3)
    for term in terms:
        state = qm.rzz(
            state,
            gamma * term.weight,
            left=term.left,
            right=term.right,
        )
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="coefficient_table")
    symbolic = qm.resources.logical(module["ising"])
    program = module.specialize(
        "ising",
        **{
            "terms": (
                {"left": 0, "right": 1, "weight": 1.0},
                {"left": 1, "right": 2, "weight": -0.5},
            ),
        },
    )
    study = qm.resources.logical(program)

    assert symbolic.evaluate(n_terms=5).metric(
        qm.resources.LOGICAL_ROTATION_GATES
    ).numeric_enclosure() == (5, 5)
    assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
        2,
        2,
    )
    assert (
        qm.run(
            program,
            on=qm.Reference(),
            parameters={"gamma": 0.5},
            shots=4,
            seed=2,
        ).result.shots
        == 4
    )

    with pytest.raises(qm.SpecializationError, match="conflicting|expects 2"):
        module.specialize(
            "ising",
            **{
                "n_terms": 3,
                "terms": (
                    {"left": 0, "right": 1, "weight": 1.0},
                    {"left": 1, "right": 2, "weight": -0.5},
                ),
            },
        )


def test_for_target_rebinding_matches_python_on_zero_and_nonzero_trips() -> None:
    """An existing local becomes the last induction value only after a trip."""

    source = """
import qamomile as qm

@qm.qkernel
def rebound(limit: qm.Const[qm.Int]) -> tuple[qm.Int, qm.Int]:
    index = 5
    total = 0
    for index in range(limit):
        total += index
    return index, total
"""
    module = qm.build_source(source)

    assert qm.evaluate(module.specialize("rebound", limit=0)) == (5, 0)
    assert qm.evaluate(module.specialize("rebound", limit=2)) == (1, 1)


def test_dependent_helper_shapes_instantiate_at_each_call_site() -> None:
    """One shape-polymorphic helper accepts distinct fixed-width states."""

    source = """
import qamomile as qm

@qm.qkernel
def first(state: qm.Q[qm.Bits[n]]) -> qm.Q[qm.Bits[n]]:
    return qm.x(state, target=0)

@qm.qkernel
def main() -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[2]]]:
    one = first(qm.qubits(1))
    two = first(qm.qubits(2))
    return (qm.measure(one), qm.measure(two))
"""
    program = qm.build_source(source).specialize("main")
    result = qm.run(program, on=qm.Reference(), shots=3, seed=4).result

    assert result.counts == {(1, 1, 0): 3}
    assert result.values == {((1,), (1, 0)): 3}


def test_dependent_shape_names_are_alpha_equivalent_across_calls() -> None:
    """A caller and callee need not coordinate dependent-binder spelling."""

    source = """
import qamomile as qm

@qm.qkernel
def helper(state: qm.Q[qm.Bits[n]]) -> qm.Q[qm.Bits[n]]:
    return qm.x(state, target=0)

@qm.qkernel
def caller(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:
    return helper(state)
"""
    function = qm.build_source(source).function("caller")

    assert str(function.result_type) == "Q[Bits[width]]"


def test_affine_dimensions_infer_and_persist_canonically() -> None:
    """Call inference solves bounded affine dimensions before QCore lowering."""

    source = """
import qamomile as qm

@qm.qkernel
def offset(state: qm.Q[qm.Bits[3 * n + 3]]) -> qm.Q[qm.Bits[3 * n + 3]]:
    return qm.x(state, target=0)

@qm.qkernel
def pair(
    prefix: qm.Q[qm.Bits[n]],
    combined: qm.Q[qm.Bits[n + m]],
) -> tuple[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[n + m]]]:
    return prefix, combined

@qm.qkernel
def main() -> qm.Shot[qm.Bits[9]]:
    changed = offset(qm.qubits(6))
    prefix, combined = pair(qm.qubits(1), qm.qubits(2))
    return qm.measure(qm.concat(changed, prefix, combined))
"""
    module = qm.build_source(source, module_name="affine_dimensions")
    function = module.function("offset")
    program = module.specialize("main")
    restored = qm.Program.from_bytes(program.to_bytes())

    assert str(function.parameters[0].value.type) == "Q[Bits[3*n+3]]"
    assert restored.entry.result_type == qm.Shot[qm.Bits[9]]
    assert qm.run(restored, on=qm.Reference(), shots=3).result.counts == {
        (1, 0, 0, 0, 0, 0, 0, 0, 0): 3
    }


@pytest.mark.parametrize("expression", ["n / 2", "n % 2", "n**2"])
def test_polynomial_dimensions_reject_nonpolynomial_forms(expression: str) -> None:
    """Dependent dimensions exclude division, modulus, and exponent syntax."""

    source = f"""
import qamomile as qm

@qm.qkernel
def invalid(state: qm.Q[qm.Bits[{expression}]]) -> qm.Q[qm.Bits[{expression}]]:
    return state
"""
    with pytest.raises(qm.FrontendError, match="Q5065"):
        qm.build_source(source)


def test_specialized_entry_const_instantiates_a_dependent_helper_call() -> None:
    """An entry Const binding remains the call result's operand-derived width."""

    source = """
import qamomile as qm

@qm.qkernel
def allocate(width: qm.Const[qm.Int]) -> qm.Q[qm.Bits[width]]:
    return qm.qubits(width)

@qm.qkernel
def main(width: qm.Const[qm.Int]) -> qm.Q[qm.Bits[width]]:
    return allocate(width)
"""
    program = qm.build_source(source, module_name="bound_call").specialize(
        "main", **{"width": 3}
    )

    assert program.entry.result_type == qm.Q[qm.Bits[3]]
    assert (
        qm.Program.from_bytes(program.to_bytes()).entry.result_type == qm.Q[qm.Bits[3]]
    )


def test_declared_calls_delegate_dimension_inference_to_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defaults and keywords share one fail-closed native result-type query."""

    source = """
import qamomile as qm

@qm.qkernel
def allocate(
    width: qm.Const[qm.Int],
    irrelevant: qm.Int = -1,
) -> qm.Q[qm.Bits[width]]:
    return qm.qubits(width)

@qm.qkernel
def keyword_call() -> qm.Shot[qm.Bits[2]]:
    return qm.measure(allocate(irrelevant=-7, width=2))

@qm.qkernel
def default_call() -> qm.Shot[qm.Bits[2]]:
    return qm.measure(allocate(2))
"""
    module = qm.build_source(source, module_name="native_declared_calls")

    assert qm.evaluate(module.specialize("keyword_call")) == (0, 0)
    assert qm.evaluate(module.specialize("default_call")) == (0, 0)

    from qamomile._native import compiler as native

    native_bind = native.bind_declared_function_arguments
    monkeypatch.setattr(native, "bind_declared_function_arguments", None)
    with pytest.raises(TypeError, match="Q4511"):
        qm.build_source(source)

    monkeypatch.setattr(
        native,
        "bind_declared_function_arguments",
        lambda *_: [0, 0],
    )
    with pytest.raises(RuntimeError, match="Q4512"):
        qm.build_source(source)
    monkeypatch.setattr(native, "bind_declared_function_arguments", native_bind)

    monkeypatch.setattr(native, "infer_declared_function_application", None)
    with pytest.raises(TypeError, match="Q4511"):
        qm.build_source(source)

    monkeypatch.setattr(
        native,
        "infer_declared_function_application",
        lambda *_: object(),
    )
    with pytest.raises(RuntimeError, match="Q4512"):
        qm.build_source(source)


@pytest.mark.parametrize(
    ("call", "code", "message"),
    (
        ("add(value, value, value)", "Q5048", "too many arguments"),
        ("add(left=value, unknown=value)", "Q5049", "unknown parameter"),
        ("add(value, left=value)", "Q5050", "supplied twice"),
        ("add(left=value)", "Q5051", "missing argument"),
    ),
)
def test_declared_call_binding_errors_come_from_rust(
    call: str,
    code: str,
    message: str,
) -> None:
    """Ordinary calls retain stable source diagnostics from the native binder."""

    source = f"""
import qamomile as qm

@qm.qkernel
def add(left: qm.Int, right: qm.Int) -> qm.Int:
    return left + right

@qm.qkernel
def choose(value: qm.Int) -> qm.Int:
    return {call}
"""
    with pytest.raises(qm.FrontendError, match=rf"{code}.*{message}"):
        qm.build_source(source)


def test_loop_exit_lattice_accepts_proven_all_path_returns() -> None:
    """Static nonempty loops and loop-else returns need no trailing return."""

    source = """
import qamomile as qm

@qm.qkernel
def from_nonempty() -> qm.Int:
    for value in range(1):
        return value + 7

@qm.qkernel
def from_else(limit: qm.Const[qm.Int]) -> qm.Int:
    for value in range(limit):
        if value > 3:
            return value
    else:
        return 11
"""
    module = qm.build_source(source)

    assert qm.evaluate(module.specialize("from_nonempty")) == 7
    assert qm.evaluate(module.specialize("from_else", limit=2)) == 11


def test_all_return_loops_have_no_phantom_owned_results() -> None:
    """Non-fallthrough loops do not synthesize dead quantum carry cleanup."""

    source = """
import qamomile as qm

@qm.qkernel
def from_for() -> qm.Q[qm.Bool]:
    state = qm.prepare(False)
    for index in range(1):
        return qm.x(state)
    else:
        return state
"""
    module = qm.build_source(source, module_name="all_return_owned_loops")

    function = module.function("from_for")
    loop = next(
        operation
        for operation in function.body.operations
        if operation.opcode == "qcore.for"
    )

    assert loop.results == ()
    assert function.body.terminator.kind == "unreachable"
    assert [region.entry.terminator.kind for region in loop.regions][-2:] == [
        "return",
        "return",
    ]
    assert all(
        operation.opcode != "qcore.forget" for operation in function.body.operations
    )


def test_while_is_retained_as_structured_control() -> None:
    """Source loops remain abstract through build and specialization."""

    source = """
import qamomile as qm

@qm.qkernel
def rejected(limit: qm.Const[qm.Int]) -> qm.Int:
    value = 0
    while value < limit:
        value += 1
    return value
"""
    program = qm.build_source(source, module_name="structured_while").specialize(
        "rejected", limit=3
    )
    assert qm.evaluate(program) == 3


def test_loop_exit_lattice_rejects_possibly_empty_return_loop() -> None:
    """A possibly empty loop still requires a result on its fallthrough path."""

    source = """
import qamomile as qm

@qm.qkernel
def incomplete(limit: qm.Param[qm.Int]) -> qm.Int:
    for value in range(limit):
        return value
"""
    with pytest.raises(qm.FrontendError, match="may exit without returning"):
        qm.build_source(source)


def test_list_construction_preserves_parameter_availability() -> None:
    """Aggregating and indexing a Param value cannot erase its stage."""

    source = """
import qamomile as qm

@qm.qkernel
def keep_stage(value: qm.Param[qm.Int]) -> qm.Param[qm.Int]:
    values = [value]
    return values[0]
"""
    module = qm.build_source(source)
    program = module.specialize("keep_stage")

    assert "Param[Vector[Int, 1]]" in program.inspect()
    assert qm.evaluate(program, value=7) == 7


@pytest.mark.parametrize(
    "expression",
    [
        "1.5 << 2",
        "(1 + 2j) // 2",
        "1 < (1, 2)",
        "int((1, 2))",
    ],
)
def test_invalid_classical_operator_domains_fail_at_build(expression: str) -> None:
    """Invalid Python operator domains produce compiler diagnostics, not TypeError."""

    source = f"""
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Int:
    return {expression}
"""
    with pytest.raises(qm.QamomileError):
        qm.build_source(source)


def test_constant_fold_failure_is_a_specialization_diagnostic() -> None:
    """Partial evaluation translates arithmetic failures to a stable diagnostic."""

    source = """
import qamomile as qm

@qm.qkernel
def divide(value: qm.Const[qm.Int]) -> qm.Int:
    return 1 // value
"""
    module = qm.build_source(source)
    with pytest.raises(qm.SpecializationError, match="Q3008"):
        module.specialize("divide", value=0)


def test_primitive_gate_family_has_typed_reference_semantics() -> None:
    """Adjoint phases, Toffoli, controlled phase, and ZZ rotation stay abstract."""

    source = """
import qamomile as qm

@qm.qkernel
def gate_family() -> qm.Bits[3]:
    state = qm.prepare((1, 1, 0))
    state = qm.sdg(state, target=0)
    state = qm.s(state, target=0)
    state = qm.tdg(state, target=1)
    state = qm.t(state, target=1)
    state = qm.p(state, 0.0, target=2)
    state = qm.cp(state, 0.0, control=0, target=2)
    state = qm.rzz(state, 0.0, left=1, right=2)
    state = qm.ccx(state, control1=0, control2=1, target=2)
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("gate_family")
    study = qm.resources.logical(program)

    assert qm.run(program, on=qm.Reference(), shots=4).result.counts == {(1, 1, 1): 4}
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (8, 8)
    assert study.metric(qm.resources.LOGICAL_TOFFOLI_GATES).numeric_enclosure() == (
        1,
        1,
    )


def test_as_uint_reinterprets_bits_and_persists_result_abi() -> None:
    """A coherent UInt reshape survives QCore persistence and typed decoding."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.UInt[3]]:
    state = qm.prepare((1, 0, 1))
    return qm.measure(qm.as_uint(state))
"""
    program = qm.build_source(source).specialize("sample")
    restored = qm.Program.from_bytes(program.to_bytes())

    assert "qcore.reshape" in restored.inspect()
    result = qm.run(restored, on=qm.Reference(), shots=4, seed=2).result
    assert result.counts == {(1, 0, 1): 4}
    assert result.values == {5: 4}


def test_q_bits_is_the_canonical_coherent_bit_spelling() -> None:
    """The explicit bits modality has one canonical public spelling."""

    assert qm.Q[qm.Bits[3]] == qm.Q[qm.Bits[3]]
    assert str(qm.Q[qm.Bits[3]]) == "Q[Bits[3]]"
    removed_name = "Q" + "State"
    assert not hasattr(qm, removed_name)

    source = f"""
import qamomile as qm

@qm.qkernel
def invalid(state: qm.{removed_name}[1]) -> qm.Q[qm.Bits[1]]:
    return state
"""
    with pytest.raises(qm.FrontendError, match="unsupported type constructor"):
        qm.build_source(source)


def test_as_bits_reinterprets_uint_without_applying_gates() -> None:
    """A UInt can return to raw coherent bits through one structural reshape."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[3]]:
    value = qm.as_uint(qm.prepare((1, 0, 1)))
    return qm.measure(qm.as_bits(value))
"""
    program = qm.build_source(source).specialize("sample")

    assert program.inspect().count("qcore.reshape") == 2
    result = qm.run(program, on=qm.Reference(), shots=4, seed=2).result
    assert result.counts == {(1, 0, 1): 4}


@pytest.mark.parametrize(
    "operator",
    ["+", "-"],
)
def test_uint_binary_arithmetic_returns_both_linear_successors(
    operator: str,
) -> None:
    """Unsigned arithmetic updates the left register and retains the right."""

    left_bits = "(1, 0)" if operator == "+" else "(1, 0)"
    right_bits = "(0, 1)" if operator == "+" else "(0, 1)"
    source = f"""
import qamomile as qm

@qm.qkernel
def arithmetic() -> qm.Shot[qm.Bits[4]]:
    left = qm.as_uint(qm.prepare({left_bits}))
    right = qm.as_uint(qm.prepare({right_bits}))
    left, right = left {operator} right
    state = qm.concat(qm.as_bits(left), qm.as_bits(right))
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("arithmetic")
    result = qm.run(program, on=qm.Reference(), shots=4, seed=2).result

    assert (
        "quantum.add_into" in program.inspect()
        or "quantum.subtract_into" in program.inspect()
    )
    assert result.counts == {(1, 1, 0, 1): 4}


def test_uint_augmented_assignment_rejects_a_hidden_retained_successor() -> None:
    """Two-register += cannot silently discard the retained right register."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Q[qm.UInt[2]]:
    left = qm.as_uint(qm.qubits(2))
    right = qm.as_uint(qm.qubits(2))
    left += right
    return left
"""
    with pytest.raises(qm.QamomileError):
        qm.build_source(source)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("qm.as_uint(qm.prepare(True))", r"expects Q\[Bits\[N\]\]"),
        ("qm.as_uint(qm.qubits(2), width=2)", "expects exactly one state"),
    ],
)
def test_as_uint_rejects_invalid_source_shapes(
    expression: str,
    message: str,
) -> None:
    """The UInt view fails before specialization for invalid source calls."""

    source = f"""
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Q[qm.UInt[1]]:
    return {expression}
"""
    with pytest.raises(qm.QamomileError, match=message):
        qm.build_source(source)


def test_linked_functions_carry_origin_and_docstring_intent() -> None:
    """Every compiled function records its declaring kernel and docstring."""

    from ._imported_function_fixture import imported_flip

    @qm.qkernel
    def documented_composition() -> qm.Shot[qm.Bits[1]]:
        """Flip through an imported helper."""

        return qm.measure(imported_flip(qm.qubits(1)))

    module = documented_composition.module()
    primary = module.function(documented_composition.function_name)
    helper = module.function("imported_flip")

    assert imported_flip.symbol.endswith("_imported_function_fixture.imported_flip")
    assert dict(primary.intent) == {
        "origin": documented_composition.symbol,
        "doc": "Flip through an imported helper.",
    }
    assert dict(helper.intent) == {
        "origin": imported_flip.symbol,
        "doc": "Flip one state from a separately imported Python module.",
    }
    assert f"intent{{origin={imported_flip.symbol!r}" in helper.inspect()

    @qm.qkernel(retain_docstrings=False)
    def undocumented_composition() -> qm.Shot[qm.Bits[1]]:
        """Flip through an imported helper."""

        return qm.measure(imported_flip(qm.qubits(1)))

    bare = undocumented_composition.module()
    assert all("doc" not in dict(function.intent) for function in bare.functions)
    assert dict(bare.function("imported_flip").intent) == {
        "origin": imported_flip.symbol
    }
    assert qm.evaluate(undocumented_composition.specialize()) == (1,)


def test_build_source_docstrings_and_origins_are_intent_not_meaning() -> None:
    """Docstrings and origins change persisted bytes but never the identity."""

    documented = '''
import qamomile as qm

@qm.qkernel
def flip() -> qm.Bits[1]:
    """Flip one qubit."""
    return qm.measure(qm.x(qm.qubits(1)))
'''
    retained = qm.build_source(documented, module_name="m")
    stripped = qm.build_source(documented, module_name="m", retain_docstrings=False)
    attributed = qm.build_source(
        documented, module_name="m", origins={"flip": "pkg.flip"}
    )

    assert retained.function("flip").intent == (("doc", "Flip one qubit."),)
    assert stripped.function("flip").intent == ()
    assert dict(attributed.function("flip").intent) == {
        "origin": "pkg.flip",
        "doc": "Flip one qubit.",
    }
    assert retained.to_bytes() != stripped.to_bytes() != attributed.to_bytes()
    assert (
        retained.selected_digest("flip")
        == stripped.selected_digest("flip")
        == attributed.selected_digest("flip")
    )
    long_doc = documented.replace("Flip one qubit.", "x" * 5000)
    (doc,) = [
        value
        for name, value in qm.build_source(long_doc, module_name="m")
        .function("flip")
        .intent
        if name == "doc"
    ]
    assert doc == "x" * 4000
