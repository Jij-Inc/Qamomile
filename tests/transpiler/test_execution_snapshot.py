"""Verify portable execution trees and strict local-result serialization."""

import dataclasses
import json
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from qamomile.circuit.transpiler.execution_handle import (
    CompletedExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
)
from qamomile.circuit.transpiler.execution_snapshot import (
    ExecutionSnapshot,
    ExecutionSnapshotKind,
)


class _PendingHandle(ExecutionHandle):
    """Expose a provider-like lazy value and record result retrieval."""

    def __init__(self, value: Any) -> None:
        """Initialize a deferred test value.

        Args:
            value (Any): Result exposed only when requested.
        """
        self.value = value
        self.result_calls = 0
        self.status_calls = 0

    def result(self, timeout: float | None = None) -> Any:
        """Record retrieval of the controlled result.

        Args:
            timeout (float | None): Ignored compatibility wait limit.

        Returns:
            Any: The configured test value.
        """
        self.result_calls += 1
        return self.value

    def status(self) -> JobStatus:
        """Record polling of the deliberately unfinished execution.

        Returns:
            JobStatus: The running state.
        """
        self.status_calls += 1
        return JobStatus.RUNNING


def _assert_same_value(actual: Any, expected: Any) -> None:
    """Compare type-preserving persistence without numerical approximation.

    Float hex strings compare the exact saved representation, including signed
    zero. No numerical computation occurs in these serialization tests.

    Args:
        actual (Any): Restored native value.
        expected (Any): Original native value.

    Raises:
        AssertionError: If a native type, value, or container order differs.
    """
    assert type(actual) is type(expected)
    if isinstance(expected, float):
        assert actual.hex() == expected.hex()
    elif isinstance(expected, (tuple, list)):
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_same_value(actual_item, expected_item)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            _assert_same_value(actual[key], expected[key])
    else:
        assert actual == expected


@pytest.mark.parametrize("kind", list(ExecutionSnapshotKind))
def test_node_kind_accepts_strings_and_enum_members(kind):
    """Public kind values normalize once while the wire format stays a string."""
    kwargs = (
        {"reference": ExecutionReference("test", ("id",))}
        if kind is ExecutionSnapshotKind.REMOTE
        else {}
    )
    for supplied in (kind, kind.value):
        snapshot = ExecutionSnapshot(supplied, **kwargs)
        assert snapshot.kind is kind
        assert type(snapshot.to_dict()["kind"]) is str
        assert snapshot.to_dict()["kind"] == kind.value
        assert ExecutionSnapshot.from_dict(snapshot.to_dict()).kind is kind


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        True,
        0,
        2**100,
        -7,
        0.0,
        -0.0,
        0.25,
        "value",
        "",
        [],
        (),
        {},
        {"0": 100, "11": 2},
        (True, 1, 1.0, [None, "x"], {"type": "tuple", "items": (0.25,)}),
    ],
)
def test_local_values_preserve_exact_types_through_json_and_restore(value):
    """Portable local values retain scalar types and container boundaries."""
    snapshot = ExecutionSnapshot("local", value=value)
    payload = json.loads(json.dumps(snapshot.to_dict(), allow_nan=False))
    decoded = ExecutionSnapshot.from_dict(payload)
    restore_remote = Mock(side_effect=AssertionError("must not contact a provider"))

    handle = decoded.restore(restore_remote)

    assert decoded == snapshot
    assert decoded.references() == ()
    _assert_same_value(handle.result(), value)
    assert handle.status() is JobStatus.COMPLETED
    restore_remote.assert_not_called()


@pytest.mark.parametrize("seed", [19, 43, 107])
def test_randomized_native_numbers_round_trip_without_rounding(seed):
    """JSON persistence retains varied finite float magnitudes and native ints."""
    rng = np.random.default_rng(seed)
    values = (
        (rng.standard_normal(20) * 10.0 ** rng.integers(-200, 200, size=20)).tolist(),
        rng.integers(-(2**60), 2**60, size=20).tolist(),
    )
    snapshot = ExecutionSnapshot("local", value=values)
    decoded = ExecutionSnapshot.from_dict(
        json.loads(json.dumps(snapshot.to_dict(), allow_nan=False))
    )

    _assert_same_value(decoded.restore(Mock()).result(), values)


def test_local_values_are_detached_from_inputs_payloads_and_restored_handles():
    """Mutating one representation cannot silently change another representation."""
    original = {"values": ([0.25],)}
    snapshot = ExecutionSnapshot("local", value=original)
    original["values"][0][0] = 2.0
    payload = snapshot.to_dict()
    restored = ExecutionSnapshot.from_dict(payload)
    payload["value"]["items"]["values"]["items"][0]["items"][0]["value"] = 3.0
    handle = snapshot.restore(Mock())
    handle.result()["values"][0][0] = 4.0

    _assert_same_value(snapshot.value, {"values": ([0.25],)})
    _assert_same_value(restored.value, {"values": ([0.25],)})
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.value = None


@pytest.mark.parametrize(
    "value",
    [
        complex(1, 2),
        {1: "x"},
        {"bad": object()},
        {1, 2},
        b"bytes",
        np.float64(0.25),
        np.int64(1),
        np.array([1]),
    ],
)
def test_local_values_reject_unsupported_types(value):
    """Arbitrary Python and NumPy objects cannot enter a portable snapshot."""
    with pytest.raises(
        TypeError, match="unsupported local result type|string dictionary keys"
    ):
        ExecutionSnapshot("local", value=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_local_values_reject_nonfinite_floats(value):
    """Nonstandard JSON numbers are rejected before persistence."""
    with pytest.raises(ValueError, match="finite"):
        ExecutionSnapshot("local", value={"nested": [value]})


def test_local_values_reject_cyclic_containers():
    """Recursive Python containers fail with a clear snapshot diagnostic."""
    value = []
    value.append(value)

    with pytest.raises(ValueError, match="nesting|cycle"):
        ExecutionSnapshot("local", value=value)


def test_remote_leaf_preserves_multiple_provider_ids_and_detaches_context():
    """Physical provider jobs remain inside one restorable execution leaf."""
    reference = ExecutionReference(
        "provider",
        ("task-a", "task-b"),
        target="device",
        group_id="group",
        context={"kind": "batch"},
    )
    snapshot = ExecutionSnapshot("remote", reference=reference)
    reference.context["kind"] = "changed"
    copied_reference = snapshot.references()[0]
    copied_reference.context["kind"] = "also changed"

    decoded = ExecutionSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))

    assert decoded == snapshot
    assert decoded.reference.job_ids == ("task-a", "task-b")
    assert decoded.reference.context == {"kind": "batch"}


def test_nested_groups_restore_native_batches_and_local_positions_without_waiting():
    """Tree restoration preserves values and groups without waiting on remote jobs."""
    first = ExecutionReference("test", ("task-a", "task-b"))
    second = ExecutionReference("test", ("task-c",))
    snapshot = ExecutionSnapshot(
        "composite",
        children=(
            ExecutionSnapshot("local", value=0.25),
            ExecutionSnapshot(
                "composite",
                children=(
                    ExecutionSnapshot("remote", reference=first),
                    ExecutionSnapshot("local", value=[1, 2]),
                    ExecutionSnapshot("remote", reference=second),
                ),
            ),
            ExecutionSnapshot("local", value=None),
        ),
    )
    remote_handles = (_PendingHandle((0.5, -0.5)), _PendingHandle(0.75))
    reattach = Mock(side_effect=remote_handles)
    decoded = ExecutionSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))

    handle = decoded.restore(reattach)

    assert decoded.references() == (first, second)
    assert [call.args for call in reattach.call_args_list] == [(first,), (second,)]
    assert all(
        remote.result_calls == remote.status_calls == 0 for remote in remote_handles
    )
    _assert_same_value(handle.result(), (0.25, ((0.5, -0.5), [1, 2], 0.75), None))
    assert all(remote.result_calls == 1 for remote in remote_handles)


def test_empty_groups_restore_without_provider_calls():
    """Empty composites retain the empty tuple result."""
    snapshot = ExecutionSnapshot("composite")
    callback = Mock(side_effect=AssertionError("unexpected provider access"))

    assert ExecutionSnapshot.from_dict(snapshot.to_dict()) == snapshot
    assert snapshot.references() == ()
    assert snapshot.restore(callback).result() == ()
    callback.assert_not_called()


def test_remote_restore_rejects_an_incompatible_callback_result():
    """Provider callbacks must return the common execution lifecycle contract."""
    snapshot = ExecutionSnapshot(
        "remote", reference=ExecutionReference("test", ("id",))
    )

    with pytest.raises(TypeError, match="return an ExecutionHandle"):
        snapshot.restore(Mock(return_value=object()))


def test_restore_does_not_swallow_provider_failures():
    """Restoration errors retain their provider diagnostic."""
    snapshot = ExecutionSnapshot(
        "remote", reference=ExecutionReference("test", ("id",))
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        snapshot.restore(Mock(side_effect=RuntimeError("provider unavailable")))


@pytest.mark.parametrize(
    "kwargs, exception, message",
    [
        ({"kind": 1}, TypeError, "kind"),
        ({"kind": "unknown"}, ValueError, "kind"),
        ({"kind": "remote"}, TypeError, "requires an ExecutionReference"),
        (
            {"kind": "local", "reference": ExecutionReference("test", ("id",))},
            ValueError,
            "must not contain",
        ),
        (
            {"kind": "local", "children": (ExecutionSnapshot("local"),)},
            ValueError,
            "must not contain",
        ),
        ({"kind": "composite", "value": 0.25}, ValueError, "must not contain"),
        ({"kind": "composite", "children": (0.25,)}, TypeError, "children"),
        ({"kind": "composite", "children": None}, TypeError, "children"),
        (
            {
                "kind": "remote",
                "reference": ExecutionReference("test", ("id",)),
                "value": 0.25,
            },
            ValueError,
            "must not contain",
        ),
        (
            {
                "kind": "remote",
                "reference": ExecutionReference("test", ("id",)),
                "children": (ExecutionSnapshot("local"),),
            },
            ValueError,
            "must not contain",
        ),
    ],
)
def test_constructor_rejects_inconsistent_node_fields(kwargs, exception, message):
    """An execution node has exactly one interpretation before serialization."""
    with pytest.raises(exception, match=message):
        ExecutionSnapshot(**kwargs)


@pytest.mark.parametrize(
    "payload, exception, message",
    [
        (None, TypeError, "mapping"),
        ({}, ValueError, "missing fields: kind"),
        ({"kind": 1}, TypeError, "kind"),
        ({"kind": "unknown"}, ValueError, "kind"),
        ({"kind": "local"}, ValueError, "missing fields: value"),
        ({"kind": "local", "value": None}, TypeError, "mapping"),
        (
            {"kind": "local", "value": {"type": "none", "value": None}, "extra": 1},
            ValueError,
            "unknown fields: extra",
        ),
        (
            {"kind": "local", "value": {"type": "none", "value": None}, "children": []},
            ValueError,
            "unknown fields: children",
        ),
        ({"kind": "remote"}, ValueError, "missing fields: reference"),
        ({"kind": "remote", "reference": {}}, ValueError, "missing fields"),
        (
            {
                "kind": "remote",
                "reference": {"provider": "test", "job_ids": ["id"], "extra": 1},
            },
            ValueError,
            "unknown fields: extra",
        ),
        (
            {"kind": "remote", "reference": {"provider": "test", "job_ids": []}},
            ValueError,
            "job_ids",
        ),
        ({"kind": "composite"}, ValueError, "missing fields: children"),
        ({"kind": "composite", "children": {}}, TypeError, "must be a list"),
        ({"kind": "composite", "children": [None]}, TypeError, "mapping"),
        (
            {"kind": "composite", "children": [], "reference": None},
            ValueError,
            "unknown fields: reference",
        ),
    ],
)
def test_serialized_nodes_reject_missing_unknown_or_incompatible_fields(
    payload, exception, message
):
    """Corrupted nodes are diagnosed instead of silently dropping data."""
    with pytest.raises(exception, match=message):
        ExecutionSnapshot.from_dict(payload)


@pytest.mark.parametrize(
    "value, exception, message",
    [
        ({}, ValueError, "missing fields: type"),
        ({"type": 1}, TypeError, "type must be a string"),
        ({"type": "object", "value": {}}, ValueError, "unsupported"),
        ({"type": "none"}, ValueError, "missing fields: value"),
        ({"type": "none", "value": False}, TypeError, "does not match"),
        ({"type": "bool", "value": 1}, TypeError, "does not match"),
        ({"type": "int", "value": True}, TypeError, "does not match"),
        ({"type": "float", "value": 1}, TypeError, "does not match"),
        ({"type": "float", "value": float("inf")}, ValueError, "finite"),
        ({"type": "str", "value": 1}, TypeError, "does not match"),
        (
            {
                "type": "tuple",
                "items": (),
            },
            TypeError,
            "must be a list",
        ),
        (
            {"type": "list", "items": [], "value": []},
            ValueError,
            "unknown fields: value",
        ),
        (
            {"type": "dict", "items": {1: {"type": "none", "value": None}}},
            TypeError,
            "string keys",
        ),
        ({"type": "dict", "items": []}, TypeError, "mapping"),
        ({"type": "list", "items": [1]}, TypeError, "mapping"),
    ],
)
def test_local_codec_rejects_corrupted_tags_and_payloads(value, exception, message):
    """Explicit type tags cannot coerce corrupted payloads into valid results."""
    with pytest.raises(exception, match=message):
        ExecutionSnapshot.from_dict({"kind": "local", "value": value})


def test_serialized_tree_rejects_cycles_with_a_clear_diagnostic():
    """In-memory corrupted payloads cannot recurse without a diagnostic limit."""
    payload = {"kind": "composite", "children": []}
    payload["children"].append(payload)

    with pytest.raises(ValueError, match="nesting|cycle"):
        ExecutionSnapshot.from_dict(payload)


def test_constructor_rejects_trees_that_exceed_the_decoder_nesting_limit():
    """A successfully constructed tree always fits the decoder nesting limit."""
    snapshot = ExecutionSnapshot("local", value=0.25)
    for _ in range(100):
        snapshot = ExecutionSnapshot("composite", children=(snapshot,))

    decoded = ExecutionSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert decoded.to_dict() == snapshot.to_dict()
    with pytest.raises(ValueError, match="nesting"):
        ExecutionSnapshot("composite", children=(snapshot,))


def test_serialized_local_value_rejects_cycles_with_a_clear_diagnostic():
    """Tagged local container cycles are rejected during decoding."""
    value = {"type": "list", "items": []}
    value["items"].append(value)

    with pytest.raises(ValueError, match="nesting|cycle"):
        ExecutionSnapshot.from_dict({"kind": "local", "value": value})


def test_mutated_local_values_are_validated_again_before_serialization_or_restore():
    """Exposed native containers cannot smuggle unsupported values into persistence."""
    snapshot = ExecutionSnapshot("local", value=[])
    snapshot.value.append(object())

    with pytest.raises(TypeError, match="unsupported local result type"):
        snapshot.to_dict()
    with pytest.raises(TypeError, match="unsupported local result type"):
        snapshot.restore(Mock())


def test_local_dictionary_cannot_be_confused_with_codec_tags():
    """User dictionaries containing reserved-looking names remain ordinary data."""
    value = {"kind": "remote", "type": "tuple", "value": 0.25, "items": ["x"]}

    restored = ExecutionSnapshot.from_dict(
        ExecutionSnapshot("local", value=value).to_dict()
    )

    _assert_same_value(restored.restore(Mock()).result(), value)
    assert isinstance(restored.restore(Mock()), CompletedExecutionHandle)
