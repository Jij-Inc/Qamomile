"""Keep HUGR whole and indexed runtime bindings compatible with the public ABI."""

from __future__ import annotations

import copy
import dataclasses
import math
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.types import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value
from qamomile.circuit.transpiler import (
    CompilationMetadata,
    CompiledProgram,
    ProgramABI,
    dict_param_key,
    flatten_user_bindings,
)

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from hugr import tys
from hugr.build import Module
from hugr.package import Package
from hugr.qsystem.result import QsysResult
from hugr.std.float import FLOAT_T
from hugr.std.int import int_t

from qamomile.hugr import (
    HugrExecutable,
    HugrExecutor,
    HugrTranspiler,
    SeleneExecutionOptions,
)
from qamomile.hugr._nexus import NexusTransport
from qamomile.hugr._runtime import (
    prepare_execution,
    resolve_runtime_bindings,
    validate_runtime_bindings,
)

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _rotations(
    angles: qmc.Matrix[qmc.Float], gammas: qmc.Vector[qmc.Float], label: qmc.UInt
) -> tuple[qmc.Vector[qmc.Bit], qmc.Matrix[qmc.Float], qmc.UInt]:
    """Measure indexed rotations and retain their structured classical inputs.

    Args:
        angles (qmc.Matrix[qmc.Float]): Two-by-two runtime angle matrix.
        gammas (qmc.Vector[qmc.Float]): Two runtime angle contributions.
        label (qmc.UInt): Independent scalar parameter.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Matrix[qmc.Float], qmc.UInt]: Measured
        bits, original matrix, and scalar label.
    """
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0] = qmc.ry(qubits[0], angles[0, 1] + gammas[0])
    qubits[1] = qmc.ry(qubits[1], angles[1, 0] + gammas[1])
    return qmc.measure(qubits), angles, label


def _identity_program(value, native):
    """Build an independent native identity with the given public carrier.

    Args:
        value (ValueLike): Public structured or scalar input and output.
        native (Any): Corresponding independently specified native carrier.

    Returns:
        CompiledProgram: Identity artifact with a single public argument.
    """
    module = Module()
    main = module.define_function("main", [native], [native], visibility="Public")
    main.set_outputs(*main.inputs())
    return CompiledProgram(
        Package([module.hugr], []),
        ProgramABI({value.name: value}, [value]),
        CompilationMetadata("hugr", "program_graph"),
    )


@pytest.fixture(scope="module")
def rotations():
    """Compile a parameterized matrix, vector, and scalar exactly once.

    Returns:
        HugrExecutable: Executable retaining three runtime arguments.

    Raises:
        EmitError: If the independent kernel cannot be lowered to HUGR.
    """
    return HugrTranspiler().transpile(
        _rotations,
        parameters=["angles", "gammas", "label"],
        parameter_shapes={"angles": (2, 2), "gammas": (2,)},
    )


@pytest.fixture
def whole():
    """Provide typed values with a deterministic asymmetric measurement.

    Returns:
        dict[str, Any]: Matrix, vector, and scalar values for one public run.
    """
    return {
        "angles": [[0.25, math.pi], [0.0, 0.5]],
        "gammas": [0.0, 0.0],
        "label": 2**63 + 7,
    }


@pytest.mark.parametrize("form", ["whole", "indexed", "mixed", "rows"])
def test_public_run_and_sample_accept_shared_array_binding_forms(
    tmp_path, rotations, whole, form
):
    """Real execution preserves matrix order for full, scalar, and row bindings."""
    if form == "whole":
        bindings = whole
    elif form == "indexed":
        bindings = flatten_user_bindings(whole)
    elif form == "mixed":
        bindings = flatten_user_bindings({"angles": whole["angles"]}) | {
            "gammas": np.asarray(whole["gammas"]),
            "label": whole["label"],
        }
    else:
        bindings = {
            "angles[0]": whole["angles"][0],
            "angles[1]": whole["angles"][1],
            "gammas": whole["gammas"],
            "label": whole["label"],
        }
    original = copy.deepcopy(bindings)
    artifact = rotations.artifact.to_bytes()
    executor = HugrExecutor(options=SeleneExecutionOptions(seed=31, build_dir=tmp_path))
    expected = ((True, False), ((0.25, math.pi), (0.0, 0.5)), whole["label"])
    assert rotations.run(executor, bindings=bindings).result() == expected
    assert rotations.sample(executor, shots=3, bindings=bindings).result().results == [
        (expected, 3)
    ]
    assert rotations.artifact.to_bytes() == artifact
    assert bindings.keys() == original.keys()
    for key in bindings:
        np.testing.assert_equal(bindings[key], original[key])


def test_nested_tuple_indexed_bindings_retain_scalar_domains(tmp_path):
    """Indexed tuple reconstruction preserves Bit, UInt, and Float carriers."""
    value = TupleValue(
        name="pair",
        elements=(
            Value(type=BitType(), name="flag"),
            TupleValue(
                name="nested",
                elements=(
                    Value(type=UIntType(), name="count"),
                    Value(type=FloatType(), name="angle"),
                ),
            ),
        ),
    )
    compiled = _identity_program(
        value, tys.Tuple(tys.Bool, tys.Tuple(int_t(6), FLOAT_T))
    )
    executable = HugrExecutable(compiled)
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))
    indexed = {"pair[0]": True, "pair[1][0]": 2**63 + 7, "pair[1][1]": 0.5}
    expected = (True, (2**63 + 7, 0.5))
    assert (
        prepare_execution(compiled, indexed).package.to_bytes()
        == prepare_execution(compiled, {"pair": expected}).package.to_bytes()
    )
    assert executable.run(executor, bindings=indexed).result() == expected
    assert executable.sample(executor, shots=2, bindings=indexed).result().results == [
        (expected, 2)
    ]
    with pytest.raises(TypeError, match="UInt"):
        executable.run(executor, bindings=indexed | {"pair[1][0]": True})
    with pytest.raises(ValueError, match="Bit"):
        executable.run(executor, bindings=indexed | {"pair[0]": 2})


@pytest.fixture(scope="module")
def fixed_dictionary():
    """Build an independent native dictionary identity with explicit tuple keys.

    Returns:
        CompiledProgram: Native identity preserving two fixed tuple-key entries.
    """
    keys = ((0, 1), (2, 3))
    entries = tuple(
        (
            TupleValue(
                name=f"key{index}",
                elements=tuple(
                    Value(type=UIntType(), name="key").with_const(component)
                    for component in key
                ),
            ),
            Value(type=FloatType(), name=f"coefficient{index}"),
        )
        for index, key in enumerate(keys)
    )
    value = DictValue(name="coeffs", entries=entries)
    pair_type = tys.Tuple(tys.Tuple(int_t(6), int_t(6)), FLOAT_T)
    native = tys.Tuple(pair_type, pair_type)
    return _identity_program(value, native)


def test_fixed_dictionary_forms_execute_in_abi_order(tmp_path, fixed_dictionary):
    """Dictionary keys use shared repr names and preserve the ABI's entry order."""
    executable = HugrExecutable(fixed_dictionary)
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))
    expected = {(0, 1): 0.25, (2, 3): 0.5}
    reversed_order = {"coeffs": {(2, 3): 0.5, (0, 1): 0.25}}
    indexed = {
        dict_param_key("coeffs", (2, 3)): 0.5,
        dict_param_key("coeffs", (0, 1)): 0.25,
    }
    for bindings in (reversed_order, indexed):
        actual = executable.run(executor, bindings=bindings).result()
        assert actual == expected
        assert list(actual) == list(expected)
        assert executable.sample(
            executor, shots=2, bindings=bindings
        ).result().results == [(expected, 2)]
    first = prepare_execution(fixed_dictionary, reversed_order)
    second = prepare_execution(fixed_dictionary, indexed)
    assert first.package.to_bytes() == second.package.to_bytes()
    assert first.abi_sha256 == second.abi_sha256
    assert list(reversed_order["coeffs"]) == [(2, 3), (0, 1)]


@pytest.mark.parametrize(
    "bindings,exception,match",
    [
        ({"coeffs[(0, 1)]": 0.25}, ValueError, "Missing"),
        ({"coeffs": {(0, 1): 0.25, (9, 9): 0.5}}, ValueError, "incompatible keys"),
        ({"coeffs": {(0.0, 1): 0.25, (2, 3): 0.5}}, TypeError, "UInt"),
        ({"coeffs": {(False, 1): 0.25, (2, 3): 0.5}}, TypeError, "UInt"),
        (
            {"coeffs[(0, 1)]": 0.25, "coeffs[(2, 3)]": 0.5, "coeffs[(9, 9)]": 1.0},
            ValueError,
            "Unexpected",
        ),
        (
            {"coeffs": {(0, 1): 0.25, (2, 3): 0.5}, "coeffs[(0, 1)]": 0.25},
            ValueError,
            "Conflicting",
        ),
    ],
)
def test_fixed_dictionary_binding_errors(fixed_dictionary, bindings, exception, match):
    """Indexed convenience cannot bypass dictionary key and scalar validation."""
    with pytest.raises(exception, match=match):
        prepare_execution(fixed_dictionary, bindings)


def test_integer_dictionary_keys_use_shared_indexed_names(tmp_path):
    """Integer dictionary indices name keys rather than entry positions."""
    value = DictValue(
        name="coeffs",
        entries=(
            (
                Value(type=UIntType(), name="key").with_const(9),
                Value(type=FloatType(), name="coefficient"),
            ),
        ),
    )
    compiled = _identity_program(value, tys.Tuple(tys.Tuple(int_t(6), FLOAT_T)))
    executable = HugrExecutable(compiled)
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))
    assert executable.run(executor, bindings={"coeffs[9]": 0.25}).result() == {9: 0.25}
    with pytest.raises(ValueError, match=r"Missing.*coeffs\[9\]"):
        executable.run(executor, bindings={"coeffs[0]": 0.25})


def test_dictionary_runtime_keys_require_a_whole_mapping(tmp_path):
    """An ABI carrying runtime keys never guesses dictionary contents from strings."""
    value = DictValue(
        name="coeffs",
        entries=(
            (
                Value(type=UIntType(), name="key"),
                Value(type=FloatType(), name="coefficient"),
            ),
        ),
    )
    compiled = _identity_program(value, tys.Tuple(tys.Tuple(int_t(6), FLOAT_T)))
    executable = HugrExecutable(compiled)
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))
    assert executable.run(executor, bindings={"coeffs": {9: 0.25}}).result() == {
        9: 0.25
    }
    with pytest.raises(ValueError, match="requires a whole mapping"):
        executable.run(executor, bindings={"coeffs[9]": 0.25})


@pytest.mark.parametrize("operation", ["run", "sample"])
@pytest.mark.parametrize(
    "change,exception,match",
    [
        ({"angles[0][1]": None}, TypeError, "Float"),
        ({"angles[0][1]": True}, TypeError, "Float"),
        ({"angles[0][1]": math.inf}, ValueError, "finite"),
        ({"angles[0][1]": math.nan}, ValueError, "finite"),
        ({"angles[0][1]": Decimal("1e10000")}, ValueError, "finite"),
        ({"angles[2][0]": 0.0}, ValueError, "Unexpected"),
        ({"angles[-1][0]": 0.0}, ValueError, "Unexpected"),
        ({"angles[0, 1]": 0.0}, ValueError, "Unexpected"),
        ({"unknown": []}, ValueError, "Unexpected"),
        ({"unknown": {"invalid": 1.0}}, ValueError, "Unexpected"),
        ({17: 0.0}, TypeError, "names must be strings"),
        ({"label": -1}, ValueError, "UInt"),
    ],
)
def test_indexed_validation_finishes_before_submission(
    monkeypatch, rotations, whole, operation, change, exception, match
):
    """Public jobs reject indexed errors before the provider sees any package."""
    submit = Mock(
        side_effect=AssertionError("Invalid bindings must never be submitted")
    )
    executor = HugrExecutor()
    monkeypatch.setattr(executor, "submit", submit)
    bindings = flatten_user_bindings(whole) | change
    with pytest.raises(exception, match=match):
        getattr(rotations, operation)(executor, bindings=bindings)
    submit.assert_not_called()


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_container_and_descendant_is_rejected_in_either_order(
    rotations, whole, reverse
):
    """Whole/indexed collisions cannot depend on insertion order or equal values."""
    bindings = whole | {"angles[0][0]": 0.25}
    if reverse:
        bindings = dict(reversed(list(bindings.items())))
    with pytest.raises(ValueError, match="Conflicting"):
        rotations.run(HugrExecutor(), bindings=bindings)


@pytest.mark.parametrize("angles", [[0.0] * 4, [[0.0], [0.0]], [[0.0], [0.0, 1.0]]])
def test_whole_array_shape_validation_is_preserved(rotations, whole, angles):
    """Flattening cannot erase wrong rank, wrong dimensions, or ragged arrays."""
    with pytest.raises(ValueError, match="shape"):
        rotations.run(HugrExecutor(), bindings=whole | {"angles": angles})


@pytest.mark.parametrize("row", [{0: 0.25, 1: math.pi}, [0.25], [[0.25, math.pi]]])
def test_partial_rows_cannot_bypass_shape_validation(rotations, whole, row):
    """A indexed row keeps the same shape contract as its whole-array counterpart."""
    bindings = {
        "angles[0]": row,
        "angles[1]": whole["angles"][1],
        "gammas": whole["gammas"],
        "label": whole["label"],
    }
    with pytest.raises(ValueError, match="shape"):
        rotations.run(HugrExecutor(), bindings=bindings)


def test_duplicate_fixed_dictionary_abi_keys_are_rejected(fixed_dictionary):
    """Malformed fixed dictionary metadata cannot duplicate one public entry."""
    original = fixed_dictionary.abi.public_inputs["coeffs"]
    duplicate = dataclasses.replace(original, entries=(original.entries[0],) * 2)
    compiled = dataclasses.replace(
        fixed_dictionary, abi=ProgramABI({"coeffs": duplicate}, [duplicate])
    )
    with pytest.raises(ValueError, match="duplicate fixed keys"):
        prepare_execution(compiled, {"coeffs": {(0, 1): 0.25}})


def test_missing_indexed_leaf_is_rejected(rotations, whole):
    """The native whole-array ABI requires every element, including unused ones."""
    indexed = flatten_user_bindings(whole)
    del indexed["angles[1][1]"]
    with pytest.raises(ValueError, match=r"Missing.*angles\[1\]\[1\]"):
        rotations.run(HugrExecutor(), bindings=indexed)


def test_identity_validation_and_resolution_accept_indexed_inputs(rotations, whole):
    """Artifact-free validation shares binding reconstruction with ordinary jobs."""
    compiled = CompiledProgram(rotations.artifact, rotations.abi, rotations.metadata)
    descriptor = dataclasses.replace(compiled, artifact=())
    indexed = flatten_user_bindings(whole)
    validate_runtime_bindings(descriptor, indexed)
    resolved = resolve_runtime_bindings(descriptor, indexed)
    assert tuple(resolved) == ("angles", "gammas", "label")
    assert resolved["angles"] == ((0.25, math.pi), (0.0, 0.5))
    with pytest.raises(ValueError, match="finite"):
        validate_runtime_bindings(descriptor, indexed | {"angles[1][1]": math.inf})


def test_resolution_canonicalizes_array_and_scalar_values_without_aliasing(
    rotations, whole
):
    """Host consumers receive the same canonical scalar types as native execution."""
    compiled = CompiledProgram(rotations.artifact, rotations.abi, rotations.metadata)
    values = np.asarray(
        [[Decimal("0.25"), np.float64(math.pi)], [Decimal("0.0"), Decimal("0.5")]],
        dtype=object,
    )
    bindings = {
        "angles": values,
        "gammas": [Decimal("0.0"), np.float64(0.0)],
        "label": np.uint64(whole["label"]),
    }
    resolved = resolve_runtime_bindings(compiled, bindings)
    assert resolved["angles"] == ((0.25, math.pi), (0.0, 0.5))
    assert all(type(item) is float for row in resolved["angles"] for item in row)
    assert all(type(item) is float for item in resolved["gammas"])
    assert type(resolved["label"]) is int
    values[0, 0] = Decimal("9.0")
    bindings["gammas"][0] = Decimal("8.0")
    assert resolved["angles"][0][0] == 0.25
    assert resolved["gammas"][0] == 0.0


def test_resolution_canonicalizes_fixed_dictionary_values_and_order(fixed_dictionary):
    """Dictionary normalization retains canonical keys, ABI order, and ownership."""
    values = {
        (np.int64(2), np.int64(3)): Decimal("0.5"),
        (np.int64(0), np.int64(1)): np.float64(0.25),
    }
    resolved = resolve_runtime_bindings(fixed_dictionary, {"coeffs": values})["coeffs"]
    assert list(resolved) == [(0, 1), (2, 3)]
    assert all(type(component) is int for key in resolved for component in key)
    assert all(type(item) is float for item in resolved.values())
    assert resolved[(0, 1)] + 1.0 == 1.25
    assert resolved[(2, 3)] + 1.0 == 1.5
    assert resolved is not values
    assert list(values) == [(2, 3), (0, 1)]


def test_empty_array_resolution_preserves_shape_and_ownership():
    """Canonical empty arrays retain invisible dimensions in independent storage."""
    value = ArrayValue(
        type=FloatType(),
        name="values",
        shape=tuple(
            Value(type=UIntType(), name="size").with_const(size) for size in (0, 2)
        ),
    )
    compiled = _identity_program(value, tys.Tuple())
    values = np.empty((0, 2))
    resolved = resolve_runtime_bindings(compiled, {"values": values})["values"]
    assert isinstance(resolved, np.ndarray)
    assert resolved.shape == values.shape
    assert resolved.dtype == values.dtype
    assert resolved is not values
    with pytest.raises(ValueError, match="Missing"):
        resolve_runtime_bindings(compiled)


@pytest.mark.parametrize("validate", [prepare_execution, validate_runtime_bindings])
def test_float_dictionary_key_normalization_cannot_drop_entries(validate):
    """Distinct Python keys that round to one native Float cannot silently collapse."""
    value = DictValue(
        name="coeffs",
        entries=tuple(
            (Value(type=FloatType(), name="key"), Value(type=FloatType(), name="value"))
            for _ in range(2)
        ),
    )
    pair = tys.Tuple(FLOAT_T, FLOAT_T)
    compiled = _identity_program(value, tys.Tuple(pair, pair))
    with pytest.raises(ValueError, match="duplicate canonical keys"):
        validate(compiled, {"coeffs": {2**53: 0.25, 2**53 + 1: 0.5}})


@pytest.mark.parametrize("operation", ["run", "sample"])
def test_restore_accepts_equivalent_whole_and_indexed_bindings(
    monkeypatch, rotations, whole, operation
):
    """Remote snapshot fingerprints depend on values and ABI, not spelling or order."""
    records = [
        ("qamomile.output.0.0", True),
        ("qamomile.output.0.1", False),
        *[
            (f"qamomile.output.1.{index}", value)
            for index, value in enumerate([0.25, math.pi, 0.0, 0.5])
        ],
        ("qamomile.output.2.0", whole["label"]),
    ]
    shots = 1 if operation == "run" else 3
    native = SimpleNamespace(id="binding-job", job_type="execute")
    client = SimpleNamespace(
        HeliosConfig=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
        hugr=SimpleNamespace(upload=Mock(return_value="binding-program")),
        start_execute_job=Mock(return_value=native),
        jobs=SimpleNamespace(
            get=Mock(return_value=native),
            status=Mock(return_value=SimpleNamespace(status="COMPLETED")),
            cancel=Mock(),
            results=Mock(
                return_value=[
                    SimpleNamespace(
                        download_result=lambda: QsysResult([records] * shots)
                    )
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executor = HugrExecutor("helios")
    job = getattr(rotations, operation)(executor, bindings=whole, shots=shots)
    indexed = dict(reversed(list(flatten_user_bindings(whole).items())))
    restored = rotations.restore(executor, job.snapshot(), bindings=indexed)
    expected = ((True, False), ((0.25, math.pi), (0.0, 0.5)), whole["label"])
    if operation == "run":
        assert restored.result() == expected
    else:
        assert restored.result().results == [(expected, shots)]
    client.start_execute_job.assert_called_once()
    with pytest.raises(ValueError, match="program and runtime bindings"):
        rotations.restore(
            executor, job.snapshot(), bindings=indexed | {"angles[0][0]": 0.75}
        )
