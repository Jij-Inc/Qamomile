"""Public surface for the Qamomile IR package.

``pretty_print_block`` and ``format_value`` are re-exported lazily via
``__getattr__`` because the printer module transitively imports many other
IR modules (``operation.*``, ``value``, ...) — eager re-export at package
import time would create a circular import with the internal modules that
pull ``Value`` from ``qamomile.circuit.ir.value`` during their own
initialisation.
"""

from typing import Any
from qamomile.circuit.ir.printer import format_value, pretty_print_block

__all__ = ["format_value", "pretty_print_block"]

