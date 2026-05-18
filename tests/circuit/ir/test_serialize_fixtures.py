"""Golden-snapshot tests for the IR serialization wire format.

This module pins a handful of representative serialized kernels at the
*current* ``SCHEMA_VERSION`` to checked-in fixture files (one JSON and
one msgpack file per kernel). The tests then attempt to load every
fixture and assert basic structural invariants.

Why this exists:

1. **Backward-compat regression guard**: if a future change to the
   encoder / decoder accidentally drops a field, renames a tag,
   reorders a structural slot, or otherwise breaks the v1 schema, the
   load step will raise — or the asserted shape will fail — and the
   test surfaces the regression before the change lands.
2. **Intentional schema-bump latch**: when ``SCHEMA_VERSION`` is
   bumped on purpose, ``from_dict`` rejects the v1 fixtures and this
   test fails. The fix is to introduce a migrator from v1 to the new
   version (or to drop the fixtures and acknowledge the break). The
   test will refuse to silently pass in either direction.

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
from qamomile.circuit.ir.serialize import (
    SCHEMA_VERSION,
    dump_json,
    dump_msgpack,
    load_json,
    load_msgpack,
)
from qamomile.circuit.stdlib.qft import qft
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
    """``algorithm.basic.superposition_vector`` via a wrapper — covers CallBlockOperation inline."""
    return superposition_vector(4)  # type: ignore[arg-type]


@qmc.qkernel
def _fix_qft() -> qmc.Vector[qmc.Qubit]:
    """QFT on 3 qubits — covers CompositeGateOperation + nested implementation_block."""
    qs = qmc.qubit_array(3, "qs")
    return qft(qs)


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
            "op_types": ["QInitOperation", "CompositeGateOperation"],
            "param_slot_count": 0,
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
            f"param_slot names {names!r} != expected "
            f"{expected['param_slot_names']!r}"
        )


@pytest.mark.parametrize(
    "name,_builder,expected",
    [(n, b, e) for n, b, e in FIXTURE_KERNELS],
    ids=[n for n, _, _ in FIXTURE_KERNELS],
)
def test_v1_json_fixture_still_loads(name, _builder, expected):
    """The v1 JSON fixture for ``name`` deserializes under the current loader.

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
def test_v1_msgpack_fixture_still_loads(name, _builder, expected):
    """The v1 msgpack fixture for ``name`` deserializes under the current loader.

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
    rolled (and a migrator added, or the v1 fixtures dropped along
    with explicit acknowledgement of the break).
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
