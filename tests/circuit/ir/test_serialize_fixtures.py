"""Golden-snapshot tests for the IR serialization wire format.

This module pins a handful of representative serialized kernels at the
*current* ``SCHEMA_VERSION`` to checked-in fixture files (one JSON and
one msgpack file per kernel). The tests then attempt to load every
fixture and assert basic structural invariants.

Why this exists:

1. **Backward-compat regression guard**: if a future change to the
   encoder / decoder accidentally drops a field, renames a tag,
   reorders a structural slot, or otherwise breaks the current
   schema, the load step will raise — or the asserted shape will
   fail — and the test surfaces the regression before the change
   lands.
2. **Intentional schema-bump latch**: when ``SCHEMA_VERSION`` is
   bumped on purpose, ``from_dict`` rejects the old-version fixtures
   and this test fails.  The fix is to introduce a migrator from the
   prior version to the new one (or to regenerate the fixtures and
   acknowledge the break).  The test will refuse to silently pass in
   either direction.

The fixture files are produced by :func:`_regenerate_fixtures` at the
bottom of this module. Run

    uv run python tests/circuit/ir/test_serialize_fixtures.py

after a schema bump or when adding new fixture kernels. Re-running the
generator produces structurally-equivalent payloads but with fresh
``uuid4`` UUIDs, so the bytes are *not* expected to be stable across
regenerations — the tests therefore check structural invariants only,
not byte equality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.basic import superposition_vector
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.parameter import ParamKind
from qamomile.circuit.ir.serialize import (
    SCHEMA_VERSION,
    dump_json,
    dump_msgpack,
    load_json,
    load_msgpack,
)
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.stdlib.qft import qft
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.inline import InlinePass

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / f"serialize_v{SCHEMA_VERSION}"


# ---------------------------------------------------------------------------
# Fixture kernels (must stay in lock-step with the checked-in files)
# ---------------------------------------------------------------------------


@qmc.qkernel
def _fix_bell() -> qmc.Vector[qmc.Qubit]:
    """Bell state — simplest non-trivial kernel: H + CX on 2 qubits."""
    qs = qmc.qubit_array(2, "qs")
    qs[0] = qmc.h(qs[0])
    qs[0], qs[1] = qmc.cx(qs[0], qs[1])
    return qs


@qmc.qkernel
def _fix_measure() -> qmc.Bit:
    """Measure-after-H — covers ``MeasureOperation``."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    return qmc.measure(q)


@qmc.qkernel
def _fix_ansatz_rx(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
    """Variational ansatz with Rx layer — covers ForOperation + Vector[Float] parameter."""
    n = thetas.shape[0]
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], thetas[i])
    return q


@qmc.qkernel
def _fix_superposition() -> qmc.Vector[qmc.Qubit]:
    """``algorithm.basic.superposition_vector`` via a wrapper — covers InvokeOperation inline."""
    return superposition_vector(4)  # type: ignore[arg-type]


@qmc.qkernel
def _fix_qft() -> qmc.Vector[qmc.Qubit]:
    """QFT on 3 qubits — covers InvokeOperation + nested body."""
    qs = qmc.qubit_array(3, "qs")
    return qft(qs)


@qmc.qkernel
def _fix_if_merge() -> qmc.Bit:
    """Measurement-conditioned if-else — covers ``IfOperation`` yield refs.

    Both branches rebind ``r``, so the ``IfOperation`` carries one branch
    merge; the fixture pins the ``true_yield_refs`` / ``false_yield_refs``
    wire shape.
    """
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    bit = qmc.measure(q)
    r = qmc.qubit(name="r")
    if bit:
        r = qmc.x(r)
    else:
        r = qmc.h(r)
    return qmc.measure(r)


@qmc.qkernel
def _fix_loop_carry(n: qmc.UInt) -> qmc.UInt:
    """Carried accumulation — cover ``ForOperation`` region arguments.

    ``total = total + i`` is represented by a ``RegionArg`` record, so
    the fixture pins its ``init`` / ``block_arg`` / ``yielded`` / ``result``
    references together with the loop's generic ``result_refs``.
    """
    q = qmc.qubit(name="q")
    qmc.measure(q)
    total = qmc.uint(0)
    for i in qmc.range(n):
        total = total + i
    return total


@qmc.qkernel
def _fix_expval(theta: qmc.Float, H: qmc.Observable) -> qmc.Float:
    """A toy VQE-style kernel — covers ``ExpvalOp`` + ``ObservableType`` parameter.

    Returns the expectation value of an Observable parameter against a
    single-qubit state prepared with H + Rx. Pins the wire form of
    ``ExpvalOp``, the Observable runtime-parameter encoding, and the
    classical Float return shape.
    """
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return qmc.expval(q, H)  # type: ignore[arg-type]


# Mapping from fixture name → (builder, expected structural shape).
#
# Each entry's ``builder`` returns an AFFINE block; ``expected`` is a
# dict of structural invariants checked after load.

FIXTURE_KERNELS: list[tuple[str, Callable[[], Block], dict[str, Any]]] = [
    (
        "bell",
        lambda: InlinePass().run(_fix_bell.block),
        {
            "kind": BlockKind.AFFINE,
            "op_types": ["QInitOperation", "GateOperation", "GateOperation"],
            "param_slot_count": 0,
        },
    ),
    (
        "measure",
        lambda: InlinePass().run(_fix_measure.block),
        {
            "kind": BlockKind.AFFINE,
            "op_types": ["QInitOperation", "GateOperation", "MeasureOperation"],
            "param_slot_count": 0,
        },
    ),
    (
        "ansatz_rx",
        lambda: InlinePass().run(
            _fix_ansatz_rx.build(thetas=np.array([0.1, 0.2, 0.3]))
        ),
        {
            "kind": BlockKind.AFFINE,
            # qubit_array init + H layer for-loop + Rx layer for-loop.
            "op_types": ["QInitOperation", "ForOperation", "ForOperation"],
            "param_slot_count": 1,
            "param_slot_names": ["thetas"],
        },
    ),
    (
        "superposition_4",
        lambda: InlinePass().run(_fix_superposition.block),
        {
            "kind": BlockKind.AFFINE,
            "op_types": ["QInitOperation", "ForOperation"],
            "param_slot_count": 0,
        },
    ),
    (
        "qft_3",
        lambda: InlinePass().run(_fix_qft.block),
        {
            "kind": BlockKind.AFFINE,
            "op_types": ["QInitOperation", "InvokeOperation"],
            "param_slot_count": 0,
        },
    ),
    (
        "if_merge",
        lambda: InlinePass().run(_fix_if_merge.block),
        {
            "kind": BlockKind.AFFINE,
            "op_types": [
                "QInitOperation",
                "GateOperation",
                "MeasureOperation",
                "QInitOperation",
                "IfOperation",
                "MeasureOperation",
            ],
            "param_slot_count": 0,
        },
    ),
    (
        "loop_carry",
        lambda: InlinePass().run(_fix_loop_carry.build(parameters=["n"])),
        {
            "kind": BlockKind.AFFINE,
            "op_types": [
                "QInitOperation",
                "MeasureOperation",
                "ForOperation",
            ],
            "param_slot_count": 1,
            "param_slot_names": ["n"],
        },
    ),
    (
        "measure_analyzed",
        # Same source kernel as ``measure``, but advanced through
        # ``analyze`` so the fixture covers BlockKind.ANALYZED — the
        # other supported top-level kind. Detects regressions that
        # break only the ANALYZED path (e.g., a future pass changes
        # the dependency-analysis shape that some encoder field
        # depends on). ``analyze`` requires classical I/O on
        # entrypoint kernels, so we use ``measure`` (returns ``Bit``)
        # rather than ``bell`` (returns ``Vector[Qubit]``).
        lambda: AnalyzePass().run(InlinePass().run(_fix_measure.block)),
        {
            "kind": BlockKind.ANALYZED,
            "op_types": ["QInitOperation", "GateOperation", "MeasureOperation"],
            "param_slot_count": 0,
        },
    ),
    (
        "vqe_expval",
        lambda: InlinePass().run(_fix_expval.block),
        {
            "kind": BlockKind.AFFINE,
            # H, Rx, then ExpvalOp returning a Float.
            "op_types": [
                "QInitOperation",
                "GateOperation",
                "GateOperation",
                "ExpvalOp",
            ],
            "param_slot_count": 2,
            "param_slot_names": ["theta", "H"],
            # Both slots are RUNTIME_PARAMETER (theta is auto-detected,
            # H is always RUNTIME for Observable per the qkernel
            # tracer).
            "param_slot_kinds": [
                ParamKind.RUNTIME_PARAMETER,
                ParamKind.RUNTIME_PARAMETER,
            ],
            "param_slot_types": [FloatType, ObservableType],
        },
    ),
    (
        "ansatz_rx_bound",
        # Identical IR shape to ``ansatz_rx`` above, but reasserted
        # here with explicit bound_value checks so the test fails if
        # the numpy round-trip path in ParamSlot.bound_value breaks.
        lambda: InlinePass().run(
            _fix_ansatz_rx.build(thetas=np.array([0.1, 0.2, 0.3]))
        ),
        {
            "kind": BlockKind.AFFINE,
            "op_types": ["QInitOperation", "ForOperation", "ForOperation"],
            "param_slot_count": 1,
            "param_slot_names": ["thetas"],
            "param_slot_kinds": [ParamKind.COMPILE_TIME_BOUND],
            "bound_value_numpy_array": {
                "thetas": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            },
        },
    ),
]


# ---------------------------------------------------------------------------
# Tests: load the checked-in fixtures and verify structural invariants
# ---------------------------------------------------------------------------


def _fixture_path(name: str, ext: str) -> Path:
    """Return the path to the ``name``.``ext`` fixture file.

    Args:
        name (str): The kernel fixture name (e.g., ``"bell"``).
        ext (str): The file extension (``"json"`` or ``"msgpack"``).

    Returns:
        Path: The expected on-disk path for the fixture.
    """
    return _FIXTURES_DIR / f"{name}.{ext}"


def _assert_shape(block: Block, expected: dict) -> None:
    """Assert ``block`` matches the structural shape recorded in ``expected``.

    Args:
        block (Block): The loaded Block.
        expected (dict): Structural invariants; supported keys:
            ``kind`` (BlockKind), ``op_types`` (exact ordered list of
            top-level operation type names), ``must_contain_op_types``
            (set of operation type names that must appear at least
            once at the top level), ``param_slot_count`` (int), and
            ``param_slot_names`` (list[str]).

    Raises:
        AssertionError: If any invariant is violated.
    """
    if "kind" in expected:
        assert block.kind == expected["kind"], (block.kind, expected["kind"])
    actual_op_types = [type(op).__name__ for op in block.operations]
    if "op_types" in expected:
        assert actual_op_types == expected["op_types"], (
            f"op type sequence mismatch: got {actual_op_types!r}, "
            f"expected {expected['op_types']!r}"
        )
    if "must_contain_op_types" in expected:
        missing = expected["must_contain_op_types"] - set(actual_op_types)
        assert not missing, (
            f"expected op types {missing!r} missing from {actual_op_types!r}"
        )
    if "param_slot_count" in expected:
        assert len(block.param_slots) == expected["param_slot_count"], (
            f"param_slot count {len(block.param_slots)} != "
            f"expected {expected['param_slot_count']}"
        )
    if "param_slot_names" in expected:
        names = [s.name for s in block.param_slots]
        assert names == expected["param_slot_names"], (
            f"param_slot names {names!r} != expected {expected['param_slot_names']!r}"
        )
    if "param_slot_kinds" in expected:
        kinds = [s.kind for s in block.param_slots]
        assert kinds == expected["param_slot_kinds"], (
            f"param_slot kinds {kinds!r} != expected {expected['param_slot_kinds']!r}"
        )
    if "param_slot_types" in expected:
        type_classes = [type(s.type) for s in block.param_slots]
        assert type_classes == expected["param_slot_types"], (
            f"param_slot types {[t.__name__ for t in type_classes]!r} "
            f"!= expected {[t.__name__ for t in expected['param_slot_types']]!r}"
        )
    if "bound_value_numpy_array" in expected:
        slots_by_name = {s.name: s for s in block.param_slots}
        for slot_name, expected_array in expected["bound_value_numpy_array"].items():
            assert slot_name in slots_by_name, (
                f"expected param_slot {slot_name!r} for numpy bound_value check"
            )
            actual = slots_by_name[slot_name].bound_value
            assert isinstance(actual, np.ndarray), (
                f"bound_value for {slot_name!r} is not a numpy array; "
                f"got {type(actual).__name__}"
            )
            assert actual.dtype == expected_array.dtype, (
                f"bound_value dtype {actual.dtype!r} != "
                f"expected {expected_array.dtype!r}"
            )
            assert np.array_equal(actual, expected_array), (
                f"bound_value values for {slot_name!r} differ"
            )


@pytest.mark.parametrize(
    "name,_builder,expected",
    [(n, b, e) for n, b, e in FIXTURE_KERNELS],
    ids=[n for n, _, _ in FIXTURE_KERNELS],
)
def test_json_fixture_still_loads(name, _builder, expected):
    """The current JSON fixture for ``name`` deserializes under the current loader.

    Catches accidental backward-compat regressions and intentional
    schema bumps that lack a migrator.
    """
    path = _fixture_path(name, "json")
    assert path.exists(), (
        f"Fixture {path} is missing. Did SCHEMA_VERSION bump without "
        f"regenerating fixtures? Run "
        f"`uv run python tests/circuit/ir/test_serialize_fixtures.py`."
    )
    block = load_json(path.read_bytes())
    _assert_shape(block, expected)


@pytest.mark.parametrize(
    "name,_builder,expected",
    [(n, b, e) for n, b, e in FIXTURE_KERNELS],
    ids=[n for n, _, _ in FIXTURE_KERNELS],
)
def test_msgpack_fixture_still_loads(name, _builder, expected):
    """The current msgpack fixture for ``name`` deserializes under the current loader.

    Same intent as the JSON counterpart, but exercises the binary
    pipeline so a regression in either path is caught.
    """
    path = _fixture_path(name, "msgpack")
    assert path.exists(), (
        f"Fixture {path} is missing. Did SCHEMA_VERSION bump without "
        f"regenerating fixtures? Run "
        f"`uv run python tests/circuit/ir/test_serialize_fixtures.py`."
    )
    block = load_msgpack(path.read_bytes())
    _assert_shape(block, expected)


def test_fixtures_directory_matches_schema_version():
    """The fixture directory name reflects the current ``SCHEMA_VERSION``.

    If this fails after a schema bump, the version directory must be
    rolled (and a migrator added, or the prior fixtures dropped
    along with explicit acknowledgement of the break).
    """
    assert _FIXTURES_DIR.name == f"serialize_v{SCHEMA_VERSION}"
    assert _FIXTURES_DIR.is_dir(), f"Fixture directory {_FIXTURES_DIR} is missing"


# ---------------------------------------------------------------------------
# Regeneration entrypoint (run as a script)
# ---------------------------------------------------------------------------


def _regenerate_fixtures() -> None:
    """Overwrite the fixture files with freshly-serialized kernels.

    Run this when:

    - ``SCHEMA_VERSION`` has been bumped and a migrator is in place,
    - a new fixture kernel was added to :data:`FIXTURE_KERNELS`,
    - an intentional encoder change should be reflected in the
      golden snapshots.

    Re-running the script produces structurally-equivalent but not
    byte-identical output (Values carry fresh ``uuid4`` UUIDs each
    build). The tests assert structural invariants only, so the
    fixture bytes do not need to be reproducible.
    """
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder, _expected in FIXTURE_KERNELS:
        block = builder()
        json_path = _fixture_path(name, "json")
        msgpack_path = _fixture_path(name, "msgpack")
        json_path.write_bytes(dump_json(block))
        msgpack_path.write_bytes(dump_msgpack(block))
        print(
            f"regenerated {json_path.name} ({json_path.stat().st_size} B) "
            f"and {msgpack_path.name} ({msgpack_path.stat().st_size} B)"
        )


if __name__ == "__main__":
    _regenerate_fixtures()
