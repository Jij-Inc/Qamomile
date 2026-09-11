"""Configure IBM Runtime execution through Qamomile-owned options."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
from typing import Any, cast


@dataclasses.dataclass(frozen=True)
class QiskitExecutionOptions:
    """Configure Runtime primitives without constructing Qiskit option objects.

    Per-request sampling shots and estimation precision remain arguments to
    the executable. Advanced mappings use Runtime's option names; the SDK
    validates their supported fields when the executor is constructed.

    Args:
        max_execution_time (int | None): Positive quantum execution time limit
            in seconds for both primitives, excluding queue time. Defaults to
            the SDK setting; does not set the local result-wait timeout.
        resilience_level (int | None): Estimator error mitigation level from
            zero through two. Defaults to the SDK setting.
        sampler_options (Mapping[str, Any]): Additional sampler settings.
            Defaults to an empty mapping. Nested values are copied.
        estimator_options (Mapping[str, Any]): Additional estimator settings.
            Defaults to an empty mapping. Nested values are copied.

    Raises:
        TypeError: If advanced options are not mappings with string keys.
        ValueError: If a numeric setting is invalid or an advanced mapping
            contains a field exposed directly by this class.

    Example:
        >>> options = QiskitExecutionOptions(
        ...     max_execution_time=300,
        ...     resilience_level=1,
        ...     sampler_options={"dynamical_decoupling": {"enable": True}},
        ... )
    """

    max_execution_time: int | None = None
    resilience_level: int | None = None
    sampler_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    estimator_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shared settings and copy caller-owned option mappings.

        Raises:
            TypeError: If advanced options are not mappings with string keys.
            ValueError: If a numeric setting is invalid or an advanced mapping
                duplicates a setting exposed directly by this class.
        """
        if self.max_execution_time is not None and (
            isinstance(self.max_execution_time, bool)
            or not isinstance(cast(object, self.max_execution_time), Integral)
            or self.max_execution_time <= 0
        ):
            raise ValueError("max_execution_time must be a positive integer")
        if self.resilience_level is not None and (
            isinstance(self.resilience_level, bool)
            or not isinstance(cast(object, self.resilience_level), Integral)
            or not 0 <= self.resilience_level <= 2
        ):
            raise ValueError("resilience_level must be an integer from zero to two")
        reserved = {"max_execution_time", "resilience_level"}
        for name in ("sampler_options", "estimator_options"):
            values = getattr(self, name)
            if not isinstance(values, Mapping) or any(
                not isinstance(key, str) for key in values
            ):
                raise TypeError(f"{name} must be a mapping with string keys")
            duplicate = reserved & values.keys()
            if duplicate:
                raise ValueError(
                    f"Set {sorted(duplicate)} directly on QiskitExecutionOptions"
                )
            object.__setattr__(self, name, deepcopy(dict(values)))
        for name in ("max_execution_time", "resilience_level"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, int(value))

    def sampler_kwargs(self) -> dict[str, Any]:
        """Build an independent Runtime sampler options dictionary.

        Returns:
            dict[str, Any]: Sampler settings with the shared execution limit.
        """
        values = deepcopy(dict(self.sampler_options))
        if self.max_execution_time is not None:
            values["max_execution_time"] = self.max_execution_time
        return values

    def estimator_kwargs(self) -> dict[str, Any]:
        """Build an independent Runtime estimator options dictionary.

        Returns:
            dict[str, Any]: Estimator settings with execution and mitigation
                settings when supplied.
        """
        values = deepcopy(dict(self.estimator_options))
        if self.max_execution_time is not None:
            values["max_execution_time"] = self.max_execution_time
        if self.resilience_level is not None:
            values["resilience_level"] = self.resilience_level
        return values
