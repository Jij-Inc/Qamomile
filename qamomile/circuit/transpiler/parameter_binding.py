"""Define and validate the runtime parameter ABI for quantum segments."""

from __future__ import annotations

import dataclasses
import enum
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from qamomile.circuit.transpiler.param_keys import (
    dict_param_key,
    is_decomposable_dict_binding_key,
    normalize_dict_binding_key,
)

_INDEX_SUFFIX = re.compile(r"^(?P<base>[^[]+)(?P<suffix>(?:\[\d+\])+)$")
_INDEX = re.compile(r"\[(\d+)\]")


class ParameterContainerKind(enum.StrEnum):
    """Classify the public container that owns one backend scalar slot."""

    SCALAR = "scalar"
    ARRAY = "array"
    DICT = "dict"


def split_parameter_key(name: str) -> tuple[str, tuple[int, ...] | None]:
    """Split an emitted scalar key into its root name and array indices.

    Args:
        name (str): Backend parameter key such as ``theta`` or
            ``angles[1][0]``.

    Returns:
        tuple[str, tuple[int, ...] | None]: Root parameter name and its
            concrete index tuple, or ``None`` for a scalar parameter.
    """
    match = _INDEX_SUFFIX.fullmatch(name)
    if match is None:
        return name, None
    indices = tuple(int(index) for index in _INDEX.findall(match.group("suffix")))
    return match.group("base"), indices


@dataclasses.dataclass
class ParameterInfo:
    """Describe one scalar slot in a compiled backend parameter ABI.

    Args:
        name (str): Full scalar key, for example ``gammas[0]``.
        array_name (str): Root parameter name, for example ``gammas``.
        index (int | None): Backward-compatible one-dimensional index, or
            ``None`` for scalars and higher-rank elements.
        backend_param (Any): Backend-specific parameter object.
        source_ref (str | None): IR value UUID providing the runtime value.
            Defaults to ``None``.
        indices (tuple[int, ...] | None): Complete array index tuple, or
            ``None`` for a scalar. Defaults to ``None``.
        container_kind (ParameterContainerKind): Public parameter container
            kind. Defaults to ``SCALAR``.
    """

    name: str
    array_name: str
    index: int | None
    backend_param: Any
    source_ref: str | None = None
    indices: tuple[int, ...] | None = None
    container_kind: ParameterContainerKind = ParameterContainerKind.SCALAR

    def __post_init__(self) -> None:
        """Keep legacy one-dimensional and complete indices consistent."""
        if self.container_kind is ParameterContainerKind.DICT:
            self.index = None
            self.indices = None
            return
        if self.indices is None and self.index is not None:
            self.indices = (self.index,)
        elif self.index is None and self.indices is not None and len(self.indices) == 1:
            self.index = self.indices[0]
        if self.indices is not None:
            self.container_kind = ParameterContainerKind.ARRAY


@dataclasses.dataclass(frozen=True)
class ParameterArrayInfo:
    """Describe the shape constraints known for one runtime array.

    ``None`` dimensions remain open because the frontend annotation records
    rank but does not always provide a concrete runtime extent. A dimension
    becomes concrete only when emitted scalar slots establish a contiguous
    ABI prefix with at least two elements.

    Args:
        name (str): Root runtime parameter name.
        rank (int): Number of array dimensions.
        expected_shape (tuple[int | None, ...]): Exact known dimensions and
            ``None`` for dimensions whose extent remains open.
    """

    name: str
    rank: int
    expected_shape: tuple[int | None, ...]


@dataclasses.dataclass
class ParameterMetadata:
    """Describe every scalar slot and runtime array in a compiled segment.

    Args:
        parameters (list[ParameterInfo]): Ordered scalar backend slots.
            Defaults to an empty list.
        arrays (dict[str, ParameterArrayInfo]): Explicit runtime-array ABI
            descriptors keyed by root name. Defaults to descriptors derived
            from ``parameters`` for backward compatibility.
    """

    parameters: list[ParameterInfo] = dataclasses.field(default_factory=list)
    arrays: dict[str, ParameterArrayInfo] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive explicit array descriptors when callers omit them."""
        if not self.arrays:
            self.arrays = _derive_array_info(self.parameters)

    @classmethod
    def merge(cls, metadata: Sequence[ParameterMetadata]) -> ParameterMetadata:
        """Merge parameter manifests from multiple quantum segments.

        Args:
            metadata (Sequence[ParameterMetadata]): Segment manifests in
                execution order.

        Returns:
            ParameterMetadata: Combined manifest with first-seen scalar slot
                ordering and array descriptors derived across all segments.
        """
        parameters: list[ParameterInfo] = []
        seen: set[str] = set()
        for manifest in metadata:
            for parameter in manifest.parameters:
                if parameter.name not in seen:
                    parameters.append(parameter)
                    seen.add(parameter.name)
        return cls(parameters=parameters)

    def get_array_names(self) -> set[str]:
        """Return unique scalar and array root names.

        Returns:
            set[str]: Root name for every emitted parameter.
        """
        return {parameter.array_name for parameter in self.parameters}

    def get_param_by_name(self, name: str) -> ParameterInfo | None:
        """Find one scalar slot by its full emitted key.

        Args:
            name (str): Full backend parameter key.

        Returns:
            ParameterInfo | None: Matching slot, or ``None`` when absent.
        """
        return next(
            (parameter for parameter in self.parameters if parameter.name == name),
            None,
        )

    def get_ordered_params(self) -> list[Any]:
        """Return backend parameter objects in ABI definition order.

        Returns:
            list[Any]: Backend-specific parameter objects.
        """
        return [parameter.backend_param for parameter in self.parameters]

    def to_binding_dict(self, bindings: Mapping[str, Any]) -> dict[Any, Any]:
        """Map indexed user bindings to backend parameter objects.

        Args:
            bindings (Mapping[str, Any]): Scalar values keyed by full emitted
                parameter name.

        Returns:
            dict[Any, Any]: Backend parameter objects mapped to bound values.
        """
        return {
            parameter.backend_param: bindings[parameter.name]
            for parameter in self.parameters
            if parameter.name in bindings
        }

    def validate_required_bindings(self, indexed_bindings: Mapping[str, Any]) -> None:
        """Reject missing scalar slots in an indexed binding map.

        Args:
            indexed_bindings (Mapping[str, Any]): Flattened user bindings.

        Raises:
            ValueError: If one or more emitted scalar slots are missing.
        """
        required = {parameter.name for parameter in self.parameters}
        missing = required - indexed_bindings.keys()
        if not missing:
            return
        array_names = {
            parameter.array_name
            for parameter in self.parameters
            if parameter.name in missing
        }
        example = next(iter(array_names), "param")
        raise ValueError(
            f"Missing parameter bindings: {sorted(missing)}. "
            "For array parameters, pass the full array "
            f"(e.g., bindings={{'{example}': [...]}})"
        )

    def validate_array_shapes(
        self,
        bindings: Mapping[str, Any] | None,
    ) -> None:
        """Validate user array rank and every concrete ABI dimension.

        Args:
            bindings (Mapping[str, Any] | None): Raw public bindings before
                scalar flattening. ``None`` means no validation is needed.

        Raises:
            ValueError: If a supplied array has the wrong rank or exceeds a
                dimension whose ABI extent is known exactly.
        """
        if bindings is None:
            return
        for name, info in self.arrays.items():
            if name not in bindings:
                continue
            actual_shape = _binding_shape(bindings[name])
            if len(actual_shape) != info.rank:
                raise ValueError(
                    f"Runtime array parameter {name!r} requires rank {info.rank}, "
                    f"but received shape {actual_shape}."
                )
            overflow_dimensions = [
                dimension
                for dimension, expected in enumerate(info.expected_shape)
                if expected is not None and actual_shape[dimension] > expected
            ]
            if overflow_dimensions:
                raise ValueError(
                    "Unexpected array parameter bindings beyond the emitted shape: "
                    f"{name!r} has shape {actual_shape}, expected "
                    f"{info.expected_shape}."
                )


def flatten_user_bindings(
    bindings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Flatten public arrays and dictionaries into scalar ABI keys.

    Args:
        bindings (Mapping[str, Any] | None): Raw user bindings keyed by kernel
            parameter name.

    Returns:
        dict[str, Any]: Scalar and dictionary entries keyed by emitted ABI
            names.
    """
    if bindings is None:
        return {}
    result: dict[str, Any] = {}
    for key, value in bindings.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            _flatten_array(result, key, value)
        elif isinstance(value, dict):
            for dict_key, item in value.items():
                normalized = normalize_dict_binding_key(dict_key)
                if not is_decomposable_dict_binding_key(normalized):
                    continue
                result[dict_param_key(key, normalized)] = item
        else:
            result[key] = value
    return result


def _flatten_array(result: dict[str, Any], prefix: str, value: Any) -> None:
    """Recursively append one array binding to a scalar result map.

    Args:
        result (dict[str, Any]): Destination scalar binding map.
        prefix (str): Indexed parameter-name prefix accumulated so far.
        value (Any): Nested list, tuple, ndarray, or scalar leaf.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            result[prefix] = value.item()
            return
        for index, element in enumerate(value):
            _flatten_array(result, f"{prefix}[{index}]", element)
        return
    if isinstance(value, (list, tuple)):
        for index, element in enumerate(value):
            _flatten_array(result, f"{prefix}[{index}]", element)
        return
    result[prefix] = value


def _derive_array_info(
    parameters: Sequence[ParameterInfo],
) -> dict[str, ParameterArrayInfo]:
    """Build array descriptors from explicit scalar index tuples.

    Args:
        parameters (Sequence[ParameterInfo]): Scalar ABI slots.

    Returns:
        dict[str, ParameterArrayInfo]: Array descriptors keyed by root name.
    """
    grouped: dict[str, list[tuple[int, ...]]] = {}
    for parameter in parameters:
        if parameter.container_kind is not ParameterContainerKind.ARRAY:
            continue
        indices = parameter.indices
        if indices is None:
            root, indices = split_parameter_key(parameter.name)
            if indices is None:
                continue
            parameter.array_name = root
            parameter.indices = indices
            if len(indices) == 1:
                parameter.index = indices[0]
        grouped.setdefault(parameter.array_name, []).append(indices)

    arrays: dict[str, ParameterArrayInfo] = {}
    for name, indices in grouped.items():
        ranks = {len(index) for index in indices}
        if len(ranks) != 1:
            raise ValueError(
                f"Runtime array parameter {name!r} has inconsistent ranks: "
                f"{sorted(ranks)}."
            )
        rank = ranks.pop()
        expected_shape: tuple[int | None, ...] = (None,) * rank
        if rank == 1:
            emitted = sorted({index[0] for index in indices})
            if len(emitted) >= 2 and emitted == list(range(emitted[-1] + 1)):
                expected_shape = (emitted[-1] + 1,)
        arrays[name] = ParameterArrayInfo(
            name=name,
            rank=rank,
            expected_shape=expected_shape,
        )
    return arrays


def _binding_shape(value: Any) -> tuple[int, ...]:
    """Return the rectangular shape of one public array binding.

    Args:
        value (Any): Nested sequence, ndarray, or scalar candidate.

    Returns:
        tuple[int, ...]: Rectangular array shape; scalars have rank zero.

    Raises:
        ValueError: If nested sequences have inconsistent shapes.
    """
    if isinstance(value, np.ndarray):
        return tuple(int(dimension) for dimension in value.shape)
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    child_shapes = [_binding_shape(item) for item in value]
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise ValueError("Runtime parameter arrays must be rectangular.")
    return (len(value), *child_shapes[0])
