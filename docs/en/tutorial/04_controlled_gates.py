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

# %% [markdown]
# ## 1. The minimal example: controlled-RX
#
# *(to be written)*

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
