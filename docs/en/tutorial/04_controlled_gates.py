# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [tutorial]
# ---
#
# # Controlling Gates and Sub-Kernels with `qmc.control`
#
# *(intro paragraph — to be written)*

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile

# %%
import math

import qamomile.circuit as qmc

# %% [markdown]
# ## 1. The minimal example: controlled-RX
#
# The smallest useful application of `qmc.control` is wrapping a
# single built-in rotation. `qmc.rx(q, angle)` is a one-qubit
# gate; passing it to `qmc.control` produces a two-qubit
# controlled-RX.


# %%
@qmc.qkernel
def crx_demo() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    # Drive the control to |1> so the controlled rotation fires.
    c = qmc.x(c)
    crx = qmc.control(qmc.rx)
    c, t = crx(c, t, angle=math.pi)
    return qmc.measure(t)


crx_demo.draw()

# %% [markdown]
# Three things to notice at the call site:
#
# - `qmc.control(qmc.rx)` is evaluated at *decoration time*. The
#   returned `ControlledGate` (here bound to `crx`) is a reusable
#   value; you can stash it in a variable and call it multiple
#   times.
# - When you call `crx(c, t, angle=...)`, the control qubits come
#   first as positional arguments, then the targets, then any
#   classical keyword arguments. The order mirrors the wrapped
#   `qmc.rx(q, angle)` signature with one extra control prefixed.
# - The keyword name for the classical parameter is whatever the
#   wrapped function uses (`angle` for `qmc.rx`, `theta` for
#   `qmc.p`, etc.) — `qmc.control` does not rename it.

# %% [markdown]
# ## 2. Two modes at a glance
#
# *(to be written)*

# %% [markdown]
# ## 3. Patterns that work in BOTH modes
#
# *(to be written)*

# %% [markdown]
# ### 3.1 Wrapping any callable
#
# *(to be written)*

# %% [markdown]
# ### 3.2 Sub-kernel taking `Vector[Qubit]`
#
# *(to be written)*

# %% [markdown]
# ### 3.3 Default values from `@qmc.qkernel` signatures
#
# *(to be written)*

# %% [markdown]
# ### 3.4 Classical keyword arguments in any order
#
# *(to be written)*

# %% [markdown]
# ### 3.5 Controlling `U^k` with `power=`
#
# *(to be written)*

# %% [markdown]
# ## 4. Concrete-mode-only patterns
#
# *(to be written)*

# %% [markdown]
# ### 4.1 Multiple separate positional control args (CCX style)
#
# *(to be written)*

# %% [markdown]
# ### 4.2 Mixing scalar Qubit and `VectorView` controls
#
# *(to be written)*

# %% [markdown]
# ## 5. Symbolic-mode-only patterns
#
# *(to be written)*

# %% [markdown]
# ### 5.1 `num_controls = n` over a whole pool
#
# *(to be written)*

# %% [markdown]
# ### 5.2 Canonical `n - 1` multi-controlled form
#
# *(to be written)*

# %% [markdown]
# ### 5.3 Selecting a subset with `controlled_indices=`
#
# *(to be written)*

# %% [markdown]
# ### 5.4 `controlled_indices` with `UInt` entries
#
# *(to be written)*

# %% [markdown]
# ## 6. Patterns that don't work
#
# *(to be written — uses an `expect_error` helper so each rejected
# shape doubles as a regression check)*

# %% [markdown]
# ## 7. Summary
#
# *(to be written)*
