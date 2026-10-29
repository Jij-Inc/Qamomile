"""Exercise destination-specific settings before any Nexus provider operation."""

from types import SimpleNamespace

import pytest

from qamomile.hugr._nexus import NexusExecutionOptions, NexusTransport
from tests.hugr.test_nexus import _client, _package

pytestmark = pytest.mark.hugr


@pytest.mark.parametrize("system", ["Helios-1", "Helios-1E", "Helios-1SC"])
def test_helios_requires_an_explicit_cost_before_authentication(system):
    client = _client()
    transport = NexusTransport(NexusExecutionOptions(system_name=system), client=client)
    with pytest.raises(ValueError, match="max_cost"):
        transport.submit(_package(), 1)
    client.login.assert_not_called()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


def test_native_cost_is_forwarded_to_the_per_program_sdk_field():
    client = _client()
    config = SimpleNamespace(type="HeliosConfig", system_name="Helios-1E", max_cost=2.5)
    transport = NexusTransport(
        NexusExecutionOptions(backend_config=config), client=client
    )
    transport.submit(_package(), 1)
    assert client.start_execute_job.call_args.kwargs["max_cost"] == 2.5
    assert client.start_execute_job.call_args.kwargs["backend_config"] is config


def test_conflicting_native_and_public_costs_fail_before_authentication():
    client = _client()
    config = SimpleNamespace(type="HeliosConfig", system_name="Helios-1E", max_cost=2.5)
    transport = NexusTransport(
        NexusExecutionOptions(backend_config=config, max_cost=1.0), client=client
    )
    with pytest.raises(ValueError, match="max_cost.*conflict"):
        transport.submit(_package(), 1)
    client.login.assert_not_called()
    client.hugr.upload.assert_not_called()
    client.start_execute_job.assert_not_called()


@pytest.mark.parametrize("cost", [0.0, 2.5])
def test_matching_explicit_limits_are_forwarded_including_zero(cost):
    client = _client()
    config = SimpleNamespace(
        type="HeliosConfig", system_name="Helios-1E", max_cost=cost
    )
    NexusTransport(
        NexusExecutionOptions(backend_config=config, max_cost=cost), client=client
    ).submit(_package(), 1)
    assert client.start_execute_job.call_args.kwargs["max_cost"] == cost


@pytest.mark.parametrize("cost", [-1, True, float("inf"), float("nan")])
def test_invalid_native_limits_fail_before_authentication(cost):
    client = _client()
    config = SimpleNamespace(
        type="HeliosConfig", system_name="Helios-1E", max_cost=cost
    )
    transport = NexusTransport(
        NexusExecutionOptions(backend_config=config), client=client
    )
    with pytest.raises(ValueError, match="backend_config.max_cost"):
        transport.submit(_package(), 1)
    client.login.assert_not_called()
    client.hugr.upload.assert_not_called()


def test_shot_origin_is_not_a_public_option():
    import inspect

    parameters = inspect.signature(NexusExecutionOptions).parameters
    assert "shot_origin" not in parameters
    assert "h2_shot_origin" not in parameters


@pytest.mark.parametrize(
    "system,config,expected_target,expected_origin",
    [
        ("H2-1E", None, "H2-1E", 1),
        ("H2-1", None, "H2-1", 1),
        ("Helios-1E", None, "Helios-1E", 1),
        ("Helios-1", None, "Helios-1", 1),
        (
            "Helios-1",
            SimpleNamespace(type="QuantinuumConfig", device_name="H2-1E"),
            "H2-1E",
            1,
        ),
        (
            "H2-1E",
            SimpleNamespace(type="HeliosConfig", system_name="Helios-1"),
            "Helios-1",
            1,
        ),
    ],
)
def test_internal_origin_uses_the_resolved_destination(
    system, config, expected_target, expected_origin
):
    from qamomile.hugr._nexus import _destination

    destination = _destination(
        NexusExecutionOptions(system_name=system, backend_config=config)
    )
    assert destination.target == expected_target
    assert destination.shot_origin == expected_origin


def test_h2_submission_does_not_require_a_helios_cost_cap(monkeypatch):
    from tests.hugr.test_h2_nexus import _client as h2_client

    monkeypatch.setattr(
        "qamomile.hugr._nexus.normalize_h2_shots",
        lambda package, shots, origin: (package, False),
    )
    monkeypatch.setattr("qamomile.hugr._nexus.to_h2_qir", lambda _: (b"BC\xc0\xde", {}))
    client = h2_client()
    NexusTransport(NexusExecutionOptions(system_name="H2-1E"), client=client).submit(
        _package(), 1
    )
    assert "max_cost" not in client.start_execute_job.call_args.kwargs
