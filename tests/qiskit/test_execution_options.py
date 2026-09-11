"""Verify SDK-independent Runtime option validation and ownership."""

from __future__ import annotations

import sys
from collections import UserDict
from typing import Any

import pytest

from qamomile.qiskit import QiskitExecutionOptions


@pytest.mark.parametrize(
    ("max_execution_time", "resilience_level", "sampler", "estimator"),
    [
        (None, None, {}, {}),
        (
            1,
            0,
            {"max_execution_time": 1},
            {"max_execution_time": 1, "resilience_level": 0},
        ),
        (
            600,
            1,
            {"max_execution_time": 600},
            {"max_execution_time": 600, "resilience_level": 1},
        ),
        (
            10801,
            2,
            {"max_execution_time": 10801},
            {"max_execution_time": 10801, "resilience_level": 2},
        ),
    ],
)
def test_typed_options_target_the_correct_primitive(
    max_execution_time: int | None,
    resilience_level: int | None,
    sampler: dict[str, Any],
    estimator: dict[str, Any],
) -> None:
    """Time limits reach both primitives; resilience reaches only the estimator."""
    options = QiskitExecutionOptions(
        max_execution_time=max_execution_time, resilience_level=resilience_level
    )

    assert options.sampler_kwargs() == sampler
    assert options.estimator_kwargs() == estimator


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "60", float("nan")])
def test_execution_time_requires_a_positive_integer(value: Any) -> None:
    """Nonpositive, boolean, and noninteger time limits fail during construction."""
    with pytest.raises(ValueError, match="max_execution_time"):
        QiskitExecutionOptions(max_execution_time=value)


@pytest.mark.parametrize("value", [-1, 3, True, 1.5, "1", float("nan")])
def test_resilience_level_requires_an_integer_in_range(value: Any) -> None:
    """Only the estimator's integer levels zero through two are accepted."""
    with pytest.raises(ValueError, match="resilience_level"):
        QiskitExecutionOptions(resilience_level=value)


@pytest.mark.parametrize("field", ["sampler_options", "estimator_options"])
@pytest.mark.parametrize("value", [None, [], [("seed", 1)], 42])
def test_native_options_require_mappings(field: str, value: Any) -> None:
    """Native option inputs are mappings rather than coerced sequences or nulls."""
    with pytest.raises(TypeError, match=field):
        QiskitExecutionOptions(**{field: value})


@pytest.mark.parametrize("field", ["sampler_options", "estimator_options"])
@pytest.mark.parametrize("key", [1, None])
def test_native_options_require_string_keys(field: str, key: Any) -> None:
    """Invalid native option key types fail before SDK construction."""
    with pytest.raises(TypeError, match=field):
        QiskitExecutionOptions(**{field: {key: 1}})


@pytest.mark.parametrize("field", ["sampler_options", "estimator_options"])
@pytest.mark.parametrize("reserved", ["max_execution_time", "resilience_level"])
def test_typed_keys_cannot_be_hidden_in_native_options(
    field: str, reserved: str
) -> None:
    """Typed fields own their keys even when the corresponding field is unset."""
    with pytest.raises(ValueError, match=reserved):
        QiskitExecutionOptions(**{field: {reserved: 1}})


def test_nested_options_are_detached_from_inputs_and_converted_outputs() -> None:
    """Caller inputs and separately generated primitive mappings never alias."""
    sampler = UserDict({"environment": {"job_tags": ["sampler"]}})
    estimator = UserDict({"resilience": {"zne_noise_factors": [1, 3]}})
    options = QiskitExecutionOptions(
        max_execution_time=120,
        resilience_level=1,
        sampler_options=sampler,
        estimator_options=estimator,
    )
    sampler["environment"]["job_tags"].append("caller")
    estimator["resilience"]["zne_noise_factors"].append(5)
    first_sampler = options.sampler_kwargs()
    first_estimator = options.estimator_kwargs()
    first_sampler["environment"]["job_tags"].append("consumer")
    first_estimator["resilience"]["zne_noise_factors"].append(7)

    assert options.sampler_kwargs() == {
        "max_execution_time": 120,
        "environment": {"job_tags": ["sampler"]},
    }
    assert options.estimator_kwargs() == {
        "max_execution_time": 120,
        "resilience_level": 1,
        "resilience": {"zne_noise_factors": [1, 3]},
    }
    assert sampler == {"environment": {"job_tags": ["sampler", "caller"]}}
    assert estimator == {"resilience": {"zne_noise_factors": [1, 3, 5]}}


def test_options_construction_does_not_require_the_runtime_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qamomile can prepare native dictionaries without an installed Runtime SDK."""
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)

    options = QiskitExecutionOptions(max_execution_time=60, resilience_level=0)

    assert options.sampler_kwargs() == {"max_execution_time": 60}
    assert options.estimator_kwargs() == {
        "max_execution_time": 60,
        "resilience_level": 0,
    }
