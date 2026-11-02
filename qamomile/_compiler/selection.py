"""Required Rust specialization, cleanup, and target-lowering boundaries."""

from __future__ import annotations

from .._native import compiler as native
from ..errors import CapabilityError, SpecializationError, VerificationError
from ..ir import Module, PortableValue, Program


def prepare_selection(
    module: Module,
    entrypoint: str,
    bindings: tuple[tuple[str, PortableValue], ...],
) -> Program:
    """Select, bind, link, specialize, and restore a module in Rust.

    Args:
        module: Complete verified source module.
        entrypoint: Local or fully qualified source selector.
        bindings: Portable values explicitly supplied by the caller.

    Returns:
        A closed executable program produced by the required native compiler.

    Raises:
        SpecializationError: If native selection or specialization fails.
        VerificationError: If the source module or restored program is invalid.
        CapabilityError: If the selected target-neutral semantics are unsupported.
    """

    prepare_native = getattr(native, "prepare_program", None)
    if not callable(prepare_native):
        raise CapabilityError("Q6200: native program preparation is unavailable")
    from ..serialization import encode_selection_request

    try:
        payload = bytes(
            prepare_native(encode_selection_request(module, entrypoint, bindings))
        )
    except NotImplementedError as error:
        raise CapabilityError(str(error)) from error
    except ValueError as error:
        message = str(error)
        if message.startswith(("Q1045:", "Q1046:", "Q11", "Q40")):
            raise VerificationError(message) from error
        if message.startswith(("Q61", "Q62")):
            raise CapabilityError(message) from error
        raise SpecializationError(message) from error
    return Program.from_bytes(payload)
