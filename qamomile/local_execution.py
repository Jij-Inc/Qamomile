"""Complete local SDK sampling without crossing a process boundary."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from .compilation import CompiledProgram, CompletedJob, Invocation, Job, SampleResult
from .errors import ExecutionError


def launch_local_sample(
    artifact: CompiledProgram,
    invocation: Invocation,
    *,
    endpoint: str,
    artifact_format: str,
    job_prefix: str,
    execute: Callable[[], SampleResult],
) -> Job:
    """Validate an artifact and complete one sample job in this process.

    Args:
        artifact (CompiledProgram): Sealed target artifact to execute.
        invocation (Invocation): Validated launch parameters, shots, and seed.
        endpoint (str): Endpoint identity for format errors.
        artifact_format (str): CompiledProgram format accepted by the endpoint.
        job_prefix (str): Stable prefix for the completed job identifier.
        execute (Callable[[], SampleResult]): SDK execution in this process.

    Returns:
        Job: Completed job containing source-ordered sample counts.

    Raises:
        ExecutionError: If the artifact format or SDK result is incompatible.
    """

    if artifact.format != artifact_format:
        raise ExecutionError(
            f"Q8308: {endpoint} accepts {artifact_format!r}, got {artifact.format!r}"
        )
    result = execute()
    if not isinstance(result, SampleResult):
        raise ExecutionError(f"Q8309: {endpoint} returned a non-sample result")
    return CompletedJob(sample_job_id(job_prefix, artifact, invocation), result)


def sample_job_id(
    job_prefix: str,
    artifact: CompiledProgram,
    invocation: Invocation,
) -> str:
    """Derive one stable local sampling job identifier.

    Args:
        job_prefix (str): Endpoint-specific identifier prefix.
        artifact (CompiledProgram): Sealed artifact supplying the source digest.
        invocation (Invocation): Validated execution values.

    Returns:
        str: Deterministic identifier for this endpoint and invocation.
    """

    identity = hashlib.sha256(
        artifact.source_digest + repr(invocation).encode()
    ).hexdigest()[:20]
    return f"{job_prefix}:{identity}"
