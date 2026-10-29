"""Execute unchanged Helios HUGR uploads under the zero-based device policy."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from unittest.mock import Mock

import pytest

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")
pytest.importorskip("selene_sim")

from hugr.qsystem.result import QsysResult

import qamomile.circuit as qmc
from qamomile.circuit.transpiler import JobSnapshot
from qamomile.hugr import (
    HugrExecutor,
    HugrTranspiler,
    NexusExecutionOptions,
    SeleneExecutionOptions,
)
from qamomile.hugr._nexus import NexusTransport
from qamomile.hugr._runtime import prepare_execution
from qamomile.hugr._shot_range import SHOT_SUCCESS_TAG
from tests.hugr.test_nexus import _client

pytestmark = pytest.mark.hugr

_SHOTS = 4


@qmc.qkernel
def _indices() -> tuple[qmc.UInt, qmc.UInt, qmc.UInt]:
    a = qmc.get_current_shot()
    b = qmc.get_current_shot()
    return a, b, a * 2 + 3


@qmc.qkernel
def _first_shot() -> qmc.Bit:
    q = qmc.qubit("q")
    if qmc.get_current_shot() == 0:
        q = qmc.x(q)
    return qmc.measure(q)


@qmc.qkernel
def _table_and_condition(values: qmc.Vector[qmc.Bit]) -> tuple[qmc.Bit, qmc.Bit]:
    shot = qmc.get_current_shot()
    q = qmc.qubit("q")
    if shot == 0:
        q = qmc.x(q)
    return values[shot], qmc.measure(q)


@qmc.qkernel
def _nested_read() -> qmc.UInt:
    return qmc.get_current_shot()


@qmc.qkernel
def _nested_entry() -> tuple[qmc.UInt, qmc.Bit]:
    shot = _nested_read()
    q = qmc.qubit("q")
    if shot == 0:
        q = qmc.x(q)
    return shot, qmc.measure(q)


@qmc.qkernel
def _typed_outputs(
    values: qmc.Vector[qmc.Float],
) -> tuple[qmc.Bit, qmc.UInt, qmc.Float]:
    shot = qmc.get_current_shot()
    return shot == 0, shot, values[shot]


@qmc.qkernel
def _no_shot() -> tuple[qmc.Bit, qmc.UInt, qmc.Float]:
    return qmc.measure(qmc.qubit("q")), qmc.uint(7), qmc.float_(0.25)


def _record_kinds(package):
    kinds = {
        "tket.result.result_bool": "BOOL",
        "tket.result.result_uint": "UINT",
        "tket.result.result_f64": "FLOAT",
    }
    records = {}
    for graph in package.modules:
        for _, data in graph.nodes():
            name = getattr(data.op, "name", None)
            kind = kinds.get(name().partition("<")[0]) if callable(name) else None
            if kind is not None:
                tag = data.op.args[0].value
                assert tag not in records
                records[tag] = kind
    return records


@pytest.fixture(scope="module")
def selene():
    return HugrExecutor("selene", options=SeleneExecutionOptions(seed=11))


@pytest.mark.parametrize(
    "kernel,bindings,expected",
    [
        (_indices, {}, [(i, i, 2 * i + 3) for i in range(_SHOTS)]),
        (_first_shot, {}, [True, False, False, False]),
        (
            _table_and_condition,
            {"values": [False, True, False, True]},
            [(False, True), (True, False), (False, False), (True, False)],
        ),
        (_nested_entry, {}, [(i, i == 0) for i in range(_SHOTS)]),
        (
            _typed_outputs,
            {"values": [0.125, 0.25, 0.5, 0.875]},
            [(i == 0, i, value) for i, value in enumerate([0.125, 0.25, 0.5, 0.875])],
        ),
    ],
)
def test_uploaded_helios_hugr_preserves_zero_based_reads(
    monkeypatch, selene, kernel, bindings, expected
):
    program = HugrTranspiler().transpile(kernel, bindings=bindings)
    artifact_before = program._compiled.artifact.to_bytes()
    prepared = prepare_execution(program._compiled)
    package_before = prepared.package.to_bytes()
    original_kinds = _record_kinds(prepared.package)
    original_rows = selene.submit(prepared.package, _SHOTS).result()
    expected_rows = [
        value if isinstance(value, tuple) else (value,) for value in expected
    ]
    assert [
        tuple(row[tag] for tag in prepared.output_tags) for row in original_rows
    ] == expected_rows
    client = _client()
    uploads = []
    rows_by_upload = []

    def upload(*, hugr_package, **kwargs):
        uploads.append(hugr_package)
        assert hugr_package.to_bytes() == package_before
        assert SHOT_SUCCESS_TAG.encode() not in hugr_package.to_bytes()
        assert b"qamomile.h2_shot_range.v1" not in hugr_package.to_bytes()
        kinds = _record_kinds(hugr_package)
        assert kinds == original_kinds
        rows = selene.submit(hugr_package, _SHOTS).result()
        rows_by_upload.append(rows)
        downloaded = QsysResult(
            [
                [(f"USER:{kinds[tag]}:{tag}", value) for tag, value in row.items()]
                for row in rows
            ]
        )
        client.jobs.results.return_value[0].download_result.return_value = downloaded
        return "uploaded-hugr"

    client.hugr.upload.side_effect = upload
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executor = HugrExecutor(
        "nexus", options=NexusExecutionOptions(system_name="Helios-1E", max_cost=1.0)
    )
    for _ in range(2):
        job = program.sample(executor, shots=_SHOTS)
        result = job.result().results
        assert dict(result) == dict(Counter(expected))
        for value, _ in result:
            row = value if isinstance(value, tuple) else (value,)
            assert tuple(type(item) for item in row) == tuple(
                type(item) for item in expected_rows[0]
            )
        assert [
            tuple(row[tag] for tag in prepared.output_tags)
            for row in rows_by_upload[-1]
        ] == expected_rows
        assert all(set(row) == set(prepared.output_tags) for row in rows_by_upload[-1])
    assert uploads[0].to_bytes() == uploads[1].to_bytes()
    assert prepared.package.to_bytes() == package_before
    assert program._compiled.artifact.to_bytes() == artifact_before

    saved = JobSnapshot.from_dict(json.loads(json.dumps(job.snapshot().to_dict())))
    assert len(saved.executions) == 1
    assert saved.executions[0].context == {
        "shots": str(_SHOTS),
        "package_sha256": hashlib.sha256(package_before).hexdigest(),
        "abi_sha256": prepared.abi_sha256,
    }
    monkeypatch.setattr(
        "qamomile.hugr.transpiler.HugrTranspiler.compile",
        Mock(side_effect=AssertionError("Restoration must not compile a new package")),
    )
    restored_executor = HugrExecutor(
        "nexus", options=NexusExecutionOptions(system_name="H2-1E")
    )
    assert dict(program.restore(restored_executor, saved).result().results) == dict(
        Counter(expected)
    )
    assert client.hugr.upload.call_count == 2
    assert client.start_execute_job.call_count == 2
    from qnexus.models.references import ResultVersions

    downloads = client.jobs.results.return_value[0].download_result.call_args_list
    assert len(downloads) == 3
    assert all(call.kwargs == {"version": ResultVersions.RAW} for call in downloads)
    assert prepared.package.to_bytes() == package_before
    assert program._compiled.artifact.to_bytes() == artifact_before


def test_repeated_helios_submissions_reuse_the_unchanged_package(selene):
    prepared = prepare_execution(HugrTranspiler().compile(_indices))
    before = prepared.package.to_bytes()
    client = _client()
    transport = NexusTransport(
        NexusExecutionOptions(system_name="Helios-1E", max_cost=1.0), client=client
    )
    for _ in range(2):
        transport.submit(prepared.package, _SHOTS)
        uploaded = client.hugr.upload.call_args.kwargs["hugr_package"]
        assert uploaded is prepared.package
        assert uploaded.to_bytes() == before
        rows = selene.submit(uploaded, _SHOTS).result()
        assert [tuple(row[tag] for tag in prepared.output_tags) for row in rows] == [
            (i, i, 2 * i + 3) for i in range(_SHOTS)
        ]
    assert client.hugr.upload.call_count == 2
    assert client.start_execute_job.call_count == 2
    assert prepared.package.to_bytes() == before


def test_helios_without_shot_reads_uploads_the_unchanged_package(monkeypatch, selene):
    prepared = prepare_execution(HugrTranspiler().compile(_no_shot))
    before = prepared.package.to_bytes()
    client = _client()
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    executor = HugrExecutor(
        "nexus", options=NexusExecutionOptions(system_name="Helios-1E", max_cost=1.0)
    )
    executor.submit(prepared.package, _SHOTS)
    uploaded = client.hugr.upload.call_args.kwargs["hugr_package"]
    assert uploaded is prepared.package
    assert uploaded.to_bytes() == before
    records = selene.submit(uploaded, _SHOTS).result()
    assert [tuple(row[tag] for tag in prepared.output_tags) for row in records] == [
        (False, 7, 0.25)
    ] * _SHOTS
