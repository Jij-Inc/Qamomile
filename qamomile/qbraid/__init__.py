"""Expose qBraid execution support for Qamomile.

The Qiskit adapter remains the artifact authority. This namespace only exposes
the live qBraid provider endpoint, and credentials or SDK objects never enter
persistent Qamomile artifacts.
"""

from ._integration import QBraidDevice as QBraidDevice

__all__ = ["QBraidDevice"]
