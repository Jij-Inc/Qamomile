"""Contract tests for the internal static-binding adapter registry."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, cast

import pytest

import qamomile.circuit as qmc
import qamomile.circuit.frontend.static_binding as static_binding_module
from qamomile.circuit.frontend.static_binding import (
    StaticBindingFieldSpec,
    StaticBindingMemberSpec,
    StaticBindingSpec,
    materialize_static_field,
    register_static_binding,
)


@qmc.qkernel
def _vector_identity(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Return one quantum vector unchanged."""
    return qubits


def _valid_spec(annotation: type[Any], type_key: str) -> StaticBindingSpec:
    """Build one valid single-vector static adapter.

    Args:
        annotation (type[Any]): Unique descriptor annotation for the test.
        type_key (str): Unique serialized type key for the test.

    Returns:
        StaticBindingSpec: Valid adapter contract.
    """
    return StaticBindingSpec(
        annotation=annotation,
        type_key=type_key,
        fields={
            "width": StaticBindingFieldSpec(
                handle_type=qmc.UInt,
                getter=lambda value: value.width,
            )
        },
        members={
            "unitary": StaticBindingMemberSpec(
                input_types={"qubits": qmc.Vector[qmc.Qubit]},
                output_types=(qmc.Vector[qmc.Qubit],),
                return_annotation=qmc.Vector[qmc.Qubit],
                getter=lambda value: value.unitary,
                qubit_width_fields={"qubits": "width"},
            )
        },
    )


def test_registration_rejects_duplicate_annotation_and_type_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both registry lookup keys remain globally unique."""
    monkeypatch.setattr(
        static_binding_module,
        "_SPECS_BY_ANNOTATION",
        dict(static_binding_module._SPECS_BY_ANNOTATION),
    )
    monkeypatch.setattr(
        static_binding_module,
        "_SPECS_BY_TYPE_KEY",
        dict(static_binding_module._SPECS_BY_TYPE_KEY),
    )

    class FirstDescriptor:
        """Unique annotation used only by this registry test."""

    class SecondDescriptor:
        """Second unique annotation used only by this registry test."""

    first = _valid_spec(FirstDescriptor, "tests.static_binding.duplicate")
    register_static_binding(first)

    with pytest.raises(ValueError, match="annotation.*already registered"):
        register_static_binding(
            dataclasses.replace(first, type_key="tests.static_binding.annotation")
        )
    with pytest.raises(ValueError, match="type key.*registered"):
        register_static_binding(
            _valid_spec(SecondDescriptor, "tests.static_binding.duplicate")
        )


def _invalid_noncallable_getter(spec: StaticBindingSpec) -> StaticBindingSpec:
    """Replace the width getter by a non-callable value.

    Args:
        spec (StaticBindingSpec): Valid source adapter.

    Returns:
        StaticBindingSpec: Invalid adapter with a non-callable getter.
    """
    return dataclasses.replace(
        spec,
        fields={
            "width": dataclasses.replace(
                spec.fields["width"],
                getter=cast(Callable[[Any], int | float], 0),
            )
        },
    )


def _invalid_field_handle(spec: StaticBindingSpec) -> StaticBindingSpec:
    """Replace the UInt field handle by an unsupported Qubit handle.

    Args:
        spec (StaticBindingSpec): Valid source adapter.

    Returns:
        StaticBindingSpec: Invalid adapter with a quantum field handle.
    """
    return dataclasses.replace(
        spec,
        fields={
            "width": dataclasses.replace(
                spec.fields["width"],
                handle_type=qmc.Qubit,
            )
        },
    )


def _invalid_name_overlap(spec: StaticBindingSpec) -> StaticBindingSpec:
    """Give a scalar field and callable member the same name.

    Args:
        spec (StaticBindingSpec): Valid source adapter.

    Returns:
        StaticBindingSpec: Invalid adapter with overlapping surface names.
    """
    return dataclasses.replace(
        spec,
        fields={"unitary": spec.fields["width"]},
    )


def _invalid_missing_width(spec: StaticBindingSpec) -> StaticBindingSpec:
    """Remove the required vector-width mapping.

    Args:
        spec (StaticBindingSpec): Valid source adapter.

    Returns:
        StaticBindingSpec: Invalid adapter without an input width.
    """
    member = dataclasses.replace(
        spec.members["unitary"],
        qubit_width_fields={},
    )
    return dataclasses.replace(spec, members={"unitary": member})


def _invalid_unknown_width(spec: StaticBindingSpec) -> StaticBindingSpec:
    """Point a vector input at an unknown scalar field.

    Args:
        spec (StaticBindingSpec): Valid source adapter.

    Returns:
        StaticBindingSpec: Invalid adapter with an unknown width field.
    """
    member = dataclasses.replace(
        spec.members["unitary"],
        qubit_width_fields={"qubits": "missing"},
    )
    return dataclasses.replace(spec, members={"unitary": member})


def _invalid_member_output(spec: StaticBindingSpec) -> StaticBindingSpec:
    """Replace the vector pass-through output by a scalar qubit.

    Args:
        spec (StaticBindingSpec): Valid source adapter.

    Returns:
        StaticBindingSpec: Invalid adapter with a non-vector output.
    """
    member = dataclasses.replace(
        spec.members["unitary"],
        output_types=(qmc.Qubit,),
        return_annotation=qmc.Qubit,
    )
    return dataclasses.replace(spec, members={"unitary": member})


def _invalid_annotation(spec: StaticBindingSpec) -> StaticBindingSpec:
    """Replace the descriptor annotation by a non-type value.

    Args:
        spec (StaticBindingSpec): Valid source adapter.

    Returns:
        StaticBindingSpec: Invalid adapter with a string annotation.
    """
    return dataclasses.replace(spec, annotation=cast(type[Any], "descriptor"))


def _invalid_type_key(spec: StaticBindingSpec) -> StaticBindingSpec:
    """Replace the stable string type key by an integer.

    Args:
        spec (StaticBindingSpec): Valid source adapter.

    Returns:
        StaticBindingSpec: Invalid adapter with a non-string type key.
    """
    return dataclasses.replace(spec, type_key=cast(str, 1))


@pytest.mark.parametrize(
    ("case", "mutate", "error_type", "message"),
    [
        (
            "noncallable_getter",
            _invalid_noncallable_getter,
            TypeError,
            "getter must be callable",
        ),
        (
            "unsupported_field",
            _invalid_field_handle,
            TypeError,
            "support only UInt and Float",
        ),
        ("name_overlap", _invalid_name_overlap, ValueError, "names overlap"),
        (
            "missing_width",
            _invalid_missing_width,
            ValueError,
            "one width field for every input",
        ),
        (
            "unknown_width",
            _invalid_unknown_width,
            ValueError,
            "unknown width field",
        ),
        (
            "member_output",
            _invalid_member_output,
            TypeError,
            "outputs must mirror",
        ),
        (
            "annotation_type",
            _invalid_annotation,
            TypeError,
            "annotation must be a concrete type",
        ),
        (
            "type_key_type",
            _invalid_type_key,
            TypeError,
            "type_key must be a string",
        ),
    ],
)
def test_registration_rejects_malformed_contracts(
    case: str,
    mutate: Callable[[StaticBindingSpec], StaticBindingSpec],
    error_type: type[Exception],
    message: str,
) -> None:
    """Malformed field and member contracts fail before registration."""

    class Descriptor:
        """Unique annotation generated for one parametrized registry case."""

    spec = mutate(_valid_spec(Descriptor, f"tests.static_binding.{case}"))

    with pytest.raises(error_type, match=message):
        register_static_binding(spec)


def test_materialization_rejects_negative_uint_field() -> None:
    """A static UInt projection preserves the handle's non-negative domain."""

    class Descriptor:
        """Descriptor carrying one invalid negative width."""

        width = -1

    spec = _valid_spec(Descriptor, "tests.static_binding.negative_uint")

    with pytest.raises(ValueError, match="must be non-negative"):
        materialize_static_field(spec, Descriptor(), "width")
