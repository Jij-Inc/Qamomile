"""Diagnostics for qkernel quantum-rebind analysis."""

from __future__ import annotations

from qamomile.circuit.frontend.ast_transform import RebindSourceKind, RebindViolation


def format_rebind_violation(v: RebindViolation) -> tuple[str, str, str]:
    """Format a quantum-rebind violation for a user-facing error.

    Args:
        v (RebindViolation): Violation record produced by the AST analyzer.

    Returns:
        tuple[str, str, str]: Offending pattern, reason, and suggested fix.

    Raises:
        AssertionError: If the analyzer produced an internally inconsistent
            violation record.
    """
    target = v.target_name
    func = v.func_name
    src = v.source_name
    src_expr = v.source_expr if v.source_expr is not None else src
    match v.source_kind:
        case RebindSourceKind.DIRECT_ALIAS:
            if src is None:
                raise AssertionError(
                    f"DIRECT_ALIAS violation for '{target}' has no "
                    f"source_name; analyzer must set source_name for this kind"
                )
            pattern = f"{target} = {src_expr}"
            reason = f"an alias of a different quantum variable '{src}'"
            fix = (
                f"  - Use a new variable: new_var = {src_expr}\n"
                f"  - Remove the assignment if '{target}' is no longer needed"
            )
        case RebindSourceKind.QUANTUM_ARG:
            if src is None:
                raise AssertionError(
                    f"QUANTUM_ARG violation for '{target}' has no "
                    f"source_name; analyzer must set source_name for this kind"
                )
            pattern = (
                f"{target} = {func}({src}, ...)"
                if func
                else f"{target} = ...({src}, ...)"
            )
            reason = f"a value derived from different quantum variable '{src}'"
            self_update = (
                f"  - Use self-update: {target} = {func}({target}, ...)\n"
                if func
                else f"  - Use self-update: {target} = ...({target}, ...)\n"
            )
            new_var = (
                f"  - Use a new variable: new_var = {func}({src}, ...)"
                if func
                else f"  - Use a new variable: new_var = ...({src}, ...)"
            )
            fix = self_update + new_var
        case RebindSourceKind.FRESH_ALLOCATION:
            pattern = f"{target} = {func}(...)" if func else f"{target} = ...(...)"
            reason = "a freshly allocated quantum value"
            new_var = (
                f"  - Bind the new allocation to a new name: new_var = {func}(...)"
                if func
                else "  - Bind the new allocation to a new name: new_var = ...(...)"
            )
            fix = (
                f"{new_var}\n"
                f"  - Or remove the original '{target}' if it is no longer needed"
            )
        case RebindSourceKind.UNKNOWN_CALL:
            pattern = f"{target} = {func}(...)" if func else f"{target} = ...(...)"
            reason = (
                "a value that does not thread the original quantum variable "
                "through the call"
            )
            self_update = (
                f"  - Pass '{target}' into the call so it is self-updated: "
                f"{target} = {func}({target}, ...)\n"
                if func
                else f"  - Pass '{target}' into the call so it is self-updated\n"
            )
            fix = f"{self_update}  - Or bind the new value to a different name"
        case RebindSourceKind.CHAINED_ASSIGNMENT:
            pattern = f"{target} = ... = ..."
            reason = (
                "a chained assignment, which cannot be verified as a "
                "self-update for every target"
            )
            fix = (
                f"  - Write a separate assignment for '{target}'\n"
                f"  - Avoid chained ``a = b = expr`` when any target is "
                f"a quantum variable"
            )
        case _:
            raise AssertionError(f"unhandled RebindSourceKind: {v.source_kind!r}")
    return pattern, reason, fix
