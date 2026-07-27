---
name: docstrings
description: Qamomile's mandatory Google-style docstring convention for every function, method, and class in qamomile/ (public and private alike). Use whenever writing or editing code under qamomile/ so the docstring has the required Args/Returns/Raises sections. Enforced by /local-review (missing docstrings are P2+).
---

# Docstring Convention (MANDATORY)

All functions, methods, and classes in `qamomile/` — **public and private alike** — MUST carry a **Google-style docstring** with the appropriate sections filled in, not just a one-line summary. This is enforced by `/local-review` (missing docstrings are P2+).

Required sections, in this order:

1. **One-line summary** (imperative mood, ending with a period).
2. *(Optional)* A longer description paragraph after a blank line.
3. **`Args:`** — one entry per parameter. Use the form `name (type): description`. The type **MUST** appear in the docstring even though it is also in the signature (this is standard Google style; type annotations in the signature alone are NOT sufficient). Describe meaning, units, valid range, and default behavior.
4. **`Returns:`** — use the form `type: description`. The return type **MUST** be stated explicitly even though it is also in the signature. For tuple returns, name each element and give each its own type. Omit this section only for functions that truly return `None`.
5. **`Raises:`** — list every exception the function can raise with the condition that triggers it. Omit only if the function genuinely cannot raise.
6. *(When helpful)* **`Example:`** — a minimal runnable snippet, especially for public API surfaces and `@qkernel` building blocks. Error classes MUST include both correct and incorrect examples.

Example:

```python
def transpile(
    self,
    kernel: QKernel,
    bindings: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
) -> ExecutableProgram[T]:
    """Transpile a qkernel into a backend-specific executable program.

    Runs the full compilation pipeline (to_block → inline → partial_eval →
    analyze → plan → emit) and returns an executable bound to this backend.

    Args:
        kernel (QKernel): The `@qkernel`-decorated function to compile.
            Must be an entry-point kernel with concrete (non-symbolic)
            shapes once `bindings` are applied.
        bindings (dict[str, Any] | None): Compile-time parameter bindings,
            keyed by parameter name. Values are coerced to the parameter's
            declared handle type. Also resolves array shapes. Defaults to
            None, meaning no bindings — the kernel must then have no free
            parameters.
        parameters (list[str] | None): Names of kernel parameters to
            preserve as backend runtime parameters rather than binding at
            compile time. Each name must refer to a non-array parameter
            of the kernel. Defaults to None, meaning all unbound
            parameters are treated as compile-time fixed.

    Returns:
        ExecutableProgram[T]: Executable wrapping the backend circuit and
            the parameter metadata needed to re-bind runtime parameters.

    Raises:
        QamomileCompileError: If analyze/plan detects a dependency or shape
            violation in the IR.
        KeyError: If `bindings` or `parameters` contains a name that is not
            a kernel parameter.

    Example:
        >>> transpiler = QiskitTranspiler()
        >>> exe = transpiler.transpile(my_kernel, parameters=["theta"])
        >>> counts = exe.sample(
        ...     transpiler.executor(),
        ...     shots=1024,
        ...     bindings={"theta": 0.5},
        ... ).result()
    """
```

Additional rules:

- **Private helpers** (`_foo`, `__bar`) follow the same rule — a Google-style docstring with `Args`/`Returns`/`Raises` as applicable. The sections may be compact, but they must be present whenever they apply.
- **Tests** do NOT need `Args`/`Returns` sections — a clear 1–2 line description of **what the test verifies** is sufficient (see test philosophy).
- **`X | None` syntax** is the project standard in both signatures and docstrings — no `Optional[X]`.
- **Error-class docstrings** must include both a correct-usage example and an incorrect example that triggers the error.
