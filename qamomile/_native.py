"""Single required import boundary for the Qamomile Rust compiler."""

from __future__ import annotations

try:
    from . import _qamomile_native as compiler
except ImportError as error:
    raise ImportError(
        "Q0001: Qamomile requires its bundled native compiler; reinstall Qamomile"
    ) from error

__all__ = ["compiler"]
