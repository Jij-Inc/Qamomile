"""Inspect provider information without changing the base endpoint job contract."""

from __future__ import annotations

import pytest

import qamomile as qm
from qamomile.compilation import CompletedJob

from .conftest import BELL_SOURCE


def test_submit_exposes_metadata_for_an_existing_basic_job() -> None:
    """Existing endpoint jobs need no new methods to remain compatible."""
    endpoint = qm.Reference()
    artifact = qm.compile(qm.build_source(BELL_SOURCE).specialize("bell"), to=endpoint)
    job = qm.submit(artifact, on=endpoint)

    assert isinstance(job, qm.SubmittedJob)
    assert job.metadata() == {"job_id": job.job_id}
    assert isinstance(CompletedJob("local", job.result()), qm.Job)


@pytest.mark.parametrize("invalid", [None, [("cost", 1)], {1: "invalid key"}])
def test_submit_rejects_malformed_provider_metadata(monkeypatch, invalid) -> None:
    """A provider cannot expose malformed metadata through the public facade."""
    monkeypatch.setattr(CompletedJob, "metadata", lambda self: invalid, raising=False)
    endpoint = qm.Reference()
    artifact = qm.compile(qm.build_source(BELL_SOURCE).specialize("bell"), to=endpoint)
    job = qm.submit(artifact, on=endpoint)

    with pytest.raises(qm.ExecutionError, match="Q8150"):
        job.metadata()


def test_submit_returns_detached_provider_metadata(monkeypatch) -> None:
    """Editing a returned report cannot mutate the provider's retained mapping."""
    reported = {"reported_cost": 2.0, "job_id": "incorrect-provider-id"}
    monkeypatch.setattr(CompletedJob, "metadata", lambda self: reported, raising=False)
    endpoint = qm.Reference()
    artifact = qm.compile(qm.build_source(BELL_SOURCE).specialize("bell"), to=endpoint)
    job = qm.submit(artifact, on=endpoint)

    metadata = job.metadata()
    assert metadata == {"reported_cost": 2.0, "job_id": job.job_id}
    assert isinstance(metadata, dict)
    metadata["reported_cost"] = 100.0
    assert reported["reported_cost"] == 2.0
    assert job.metadata()["reported_cost"] == 2.0
