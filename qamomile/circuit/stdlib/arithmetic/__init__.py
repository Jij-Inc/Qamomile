"""Expose standard-library arithmetic callables.

The package groups arithmetic primitives by implementation strategy while
keeping the original arithmetic-module import surface stable. Public callables
are re-exported here; selected internal helpers remain available for Qamomile's
algorithm implementations and focused tests.
"""

from .bitwise import (
    _phase_shift_if as _phase_shift_if,
    _xor_constant as _xor_constant,
)
from .carry_venting import (
    _dirty_const_add_extended as _dirty_const_add_extended,
)
from .constant import add_const, controlled_add_const
from .increment import (
    _apply_fixed_window_periodic_shift as _apply_fixed_window_periodic_shift,
    modular_decrement,
    modular_increment,
)
from .modular import (
    controlled_modular_add,
    controlled_modular_add_const,
    controlled_modular_add_const_modulus,
    modular_add,
    modular_add_const,
)
from .modular_multiplication import (
    _controlled_modular_add_const_modulus_dirty as _controlled_modular_add_const_modulus_dirty,
    _modmul_const_body as _modmul_const_body,
    lookup_xor,
    modmul_const,
)
from .ripple_carry import ripple_carry_add

__all__ = [
    "add_const",
    "controlled_add_const",
    "controlled_modular_add",
    "controlled_modular_add_const",
    "controlled_modular_add_const_modulus",
    "lookup_xor",
    "modular_add",
    "modular_add_const",
    "modular_decrement",
    "modular_increment",
    "modmul_const",
    "ripple_carry_add",
]
