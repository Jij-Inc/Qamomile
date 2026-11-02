"""Pure resource-coordinate projection over one selected semantic study."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from .report import ResourceStudy
from .values import ResourceValue


@runtime_checkable
class ResourceModel(Protocol):
    """A pure model that adds coordinates without changing a program."""

    @property
    def identity(self) -> str:
        """Return the stable, versioned model identity."""

        ...

    def project(
        self,
        study: ResourceStudy,
    ) -> Mapping[str, ResourceValue]:
        """Return additional metrics for one immutable study."""

        ...


def project(
    study: ResourceStudy,
    using: ResourceModel,
) -> ResourceStudy:
    """Apply one explicit model as a pure study-to-study transformation."""

    if not isinstance(study, ResourceStudy):
        raise TypeError("project expects a ResourceStudy")
    if not isinstance(using, ResourceModel):
        raise TypeError("using= must implement ResourceModel")
    if not using.identity:
        raise ValueError("resource model identity must be nonempty")
    projected = dict(using.project(study))
    if not projected:
        raise ValueError("resource model produced no metrics")
    for identifier, value in projected.items():
        if not isinstance(identifier, str) or not isinstance(value, ResourceValue):
            raise TypeError("resource model metrics must map strings to ResourceValue")
        if identifier != value.metric.identifier:
            raise ValueError("resource model metric key disagrees with its definition")
    return study.with_metrics(
        projected,
        derivation=f"model:{using.identity}",
    )


__all__ = ["ResourceModel", "project"]
