"""Stable public diagnostic transport tests."""

from __future__ import annotations

import pytest

import qamomile as qm


def test_prefixed_message_preserves_its_machine_readable_code() -> None:
    """A native-style message becomes one structured public diagnostic."""

    error = qm.ExecutionError("Q8109: parameter disagrees with its type")

    assert error.diagnostic.code == "Q8109"
    assert error.diagnostic.message == "parameter disagrees with its type"
    assert str(error) == "Q8109: parameter disagrees with its type"


def test_wrapped_public_error_does_not_acquire_a_q0000_prefix() -> None:
    """Boundary translation retains the original code without a registry."""

    source = qm.ExecutionError("Q8145: native execution failed")
    wrapped = qm.CapabilityError(str(source))

    assert wrapped.diagnostic.code == "Q8145"
    assert str(wrapped) == str(source)


@pytest.mark.parametrize(
    "message",
    (
        "ordinary explanation",
        "Q123: short code",
        "Q12345: long code",
        "q1234: lowercase prefix",
        "Q12x4: nondigit prefix",
        "Q1234:no separator",
    ),
)
def test_noncanonical_message_uses_the_generic_code(message: str) -> None:
    """Only the exact stable-code syntax is interpreted structurally."""

    error = qm.QamomileError(message)

    assert error.diagnostic == qm.Diagnostic("Q0000", message)
    assert str(error) == f"Q0000: {message}"


def test_explicit_source_diagnostic_remains_authoritative() -> None:
    """Structured source location and notes bypass message interpretation."""

    diagnostic = qm.Diagnostic(
        "Q5001",
        "unsupported source construct",
        uri="memory://example.py",
        line=3,
        column=4,
        note="rewrite the expression",
    )
    error = qm.FrontendError(diagnostic)

    assert error.diagnostic is diagnostic
    assert str(error) == (
        "memory://example.py:3:5: Q5001: unsupported source construct\n"
        "  note: rewrite the expression"
    )
