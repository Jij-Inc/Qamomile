"""Public surface for the Qamomile IR package.

Re-exports the contributor-facing debugging helpers from
``qamomile.circuit.ir.printer`` so callers can use the short form
``from qamomile.circuit.ir import pretty_print_block``.
"""

from qamomile.circuit.ir.printer import format_value, pretty_print_block

__all__ = ["format_value", "pretty_print_block"]
