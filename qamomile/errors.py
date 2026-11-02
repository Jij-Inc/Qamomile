"""Stable diagnostic hierarchy for the Qamomile compiler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A source-oriented compiler diagnostic.

    Args:
        code: Stable machine-readable diagnostic code.
        message: Human-readable explanation.
        uri: Optional content-addressed source URI.
        line: Optional one-based source line.
        column: Optional zero-based source column.
        note: Optional actionable detail.
    """

    code: str
    message: str
    uri: str | None = None
    line: int | None = None
    column: int | None = None
    note: str | None = None

    def format(self) -> str:
        """Return the stable user-facing representation."""

        location = ""
        if self.uri is not None:
            location = self.uri
            if self.line is not None:
                location += f":{self.line}"
                if self.column is not None:
                    location += f":{self.column + 1}"
            location += ": "
        rendered = f"{location}{self.code}: {self.message}"
        if self.note:
            rendered += f"\n  note: {self.note}"
        return rendered


class QamomileError(Exception):
    """Base class for all public Qamomile failures."""

    def __init__(self, diagnostic: Diagnostic | str) -> None:
        """Create an error from a diagnostic or a simple message."""

        if isinstance(diagnostic, str):
            diagnostic = _diagnostic_from_message(diagnostic)
        self.diagnostic = diagnostic
        super().__init__(diagnostic.format())


def _diagnostic_from_message(message: str) -> Diagnostic:
    """Preserve an exact stable code carried by a boundary error message.

    Args:
        message: Plain explanation or ``Qdddd: explanation`` boundary message.

    Returns:
        A structured diagnostic retaining an exact four-digit code when present.
    """

    code = message[:5]
    if (
        len(message) >= 7
        and code.startswith("Q")
        and all("0" <= digit <= "9" for digit in code[1:])
        and message[5:7] == ": "
    ):
        return Diagnostic(code, message[7:])
    return Diagnostic("Q0000", message)


class FrontendError(QamomileError):
    """Raised when Python source is outside the Qamomile language."""


class SourceUnavailableError(FrontendError):
    """Raised when a decorator handle has no recoverable source unit."""


class TypeCheckError(QamomileError):
    """Raised when semantic value types do not compose."""


class OwnershipError(TypeCheckError):
    """Raised when a linear value or borrow violates ownership rules."""


class VerificationError(QamomileError):
    """Raised when persistent QCore invariants do not hold."""


class SerializationError(QamomileError):
    """Raised when an image or artifact cannot be decoded safely."""


class SpecializationError(QamomileError):
    """Raised when compile-time inputs cannot form a closed program."""


class CapabilityError(QamomileError):
    """Raised when a transform or target cannot preserve the program contract."""


class ArtifactError(QamomileError):
    """Raised when a target artifact violates its immutable contract."""


class ExecutionError(QamomileError):
    """Raised when an endpoint cannot execute an artifact or invocation."""
