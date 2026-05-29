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
# [Tutorial 03](03_vector_slicing.ipynb) showed how `VectorView`
# slices let one helper kernel operate on a contiguous chunk of a
# larger register. This chapter covers `qmc.control`, which turns
# any Qamomile gate — a built-in like `qmc.rx`, or a user-defined
# `@qmc.qkernel` — into its *controlled* version.
#
# `qmc.control` has two modes: *concrete* (the number of control
# qubits is a Python `int`) and *symbolic* (the number is a
# `qmc.UInt` kernel parameter, or an expression that contains one,
# resolved at transpile time). Most features — `power=`, default
# args, sub-kernels that take `Vector[Qubit]`, reordering classical
# kwargs — work the same way in both modes. What differs between
# the modes is how the control arguments are passed and a small
# set of mode-specific extras, handled in later sections.

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile

# %%
import math

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import UnreturnedBorrowError
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. The minimal example: controlled-RX
#
# The simplest, most practical use of `qmc.control` is to make a
# controlled version of one of the gates Qamomile provides. For
# example, below we pass the one-qubit gate `qmc.rx(q, angle)` to
# `qmc.control` to obtain a two-qubit controlled-RX gate.
#
# To confirm the control actually takes effect, we build two
# kernels that differ only in whether the control qubit is driven
# to |1> first, transpile both to Qiskit, run them on the
# simulator, and check the target measurement. With
# `angle=math.pi`, `RX(pi)` maps |0> to |1>, so the target ends up
# |1> on every shot exactly when the control is |1>, and stays
# |0> otherwise.


# %%
@qmc.qkernel
def crx_control_off() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    # The control stays |0>, so the controlled rotation does NOT fire.
    crx = qmc.control(qmc.rx)
    c, t = crx(c, t, angle=math.pi)
    return qmc.measure(t)


@qmc.qkernel
def crx_control_on() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    # Drive the control to |1>, so the controlled rotation fires.
    c = qmc.x(c)
    crx = qmc.control(qmc.rx)
    c, t = crx(c, t, angle=math.pi)
    return qmc.measure(t)


off_counts = dict(
    transpiler.transpile(crx_control_off)
    .sample(transpiler.executor(), shots=256)
    .result()
    .results
)
on_counts = dict(
    transpiler.transpile(crx_control_on)
    .sample(transpiler.executor(), shots=256)
    .result()
    .results
)
print("control |0> ->", off_counts)
print("control |1> ->", on_counts)
# RX(pi) is deterministic here: the target is |0> on every shot
# when the control is |0>, and |1> on every shot when it is |1>.
assert off_counts == {0: 256}
assert on_counts == {1: 256}

crx_control_on.draw()

# %% [markdown]
# Three things to notice at the call site:
#
# - You can write `crx = qmc.control(qmc.rx)` either inside or
#   outside a qkernel. Either way the returned value is reusable;
#   bind it to a name and call it as many times as you like.
# - When you call `crx(c, t, angle=...)`, the control qubits come
#   first as positional arguments, then the targets, then any
#   classical keyword arguments. The order mirrors the
#   `qmc.rx(q, angle)` signature of the gate being controlled,
#   with one extra control prefixed.
# - The keyword name for the classical parameter is whatever the
#   controlled gate uses (`angle` for `qmc.rx`, `theta` for
#   `qmc.p`, etc.) — `qmc.control` does not rename it.

# %% [markdown]
# ## 2. Two modes at a glance
#
# `qmc.control` has two modes. Which one you are in is decided
# entirely by the type you pass for `num_controls`: a Python
# `int` puts you in *concrete* mode, a `qmc.UInt` handle (or any
# `UInt` expression like `n - 1`) puts you in *symbolic* mode.
# Everything else about the call follows from that choice.
#
# | Aspect | Concrete | Symbolic |
# | --- | --- | --- |
# | `num_controls=` | Python `int` (default `1`) | `qmc.UInt` handle, or any `UInt` expression |
# | Control argument(s) | one or more positional args (`Qubit`, `VectorView`, or `Vector[Qubit]`) whose qubit counts sum to `num_controls` | one positional `Vector[Qubit]` / `VectorView` *pool* (single-pool form, with optional `control_indices=`), **or** several positional args mixing `Qubit` / `VectorView` / `Vector[Qubit]` (multi-arg form, §5.5) |
# | `control_indices=` | not accepted | optional — picks which slots of the pool are active |
# | Control count resolved at | when `qmc.control(...)` is evaluated (module-load or tracing time) | transpile time (from `bindings`) |
#
# A short decision rule: reach for *concrete* mode when the
# control count is a literal you know while writing the qkernel
# and you want to name each control qubit individually. Reach for
# *symbolic* mode when the count is a kernel parameter (or an
# expression over one — `num_controls=n - 1` is the textbook
# multi-controlled form).
#
# Most of `qmc.control`'s features (`power=`, default values,
# classical-kwarg reordering, sub-kernels that take
# `Vector[Qubit]`, ...) behave identically in both modes; Section
# 3 collects those. Section 4 then shows the multi-argument
# control shapes (using concrete mode for brevity, though they
# work symbolically too) and Section 5 the symbolic-mode-specific
# features.

# %% [markdown]
# ## 3. Patterns that work in BOTH modes
#
# Each feature in this section behaves the same way under either
# mode. The cells below use concrete mode (because the code is
# shorter without a `UInt` kernel parameter in the picture), but
# the same feature is available in symbolic mode too. Symbolic
# mode accepts both the single-pool control argument shape (5.1
# – 5.4) and the multi-arg form (5.5) — the only thing that
# changes from concrete is that `num_controls` is a `UInt`
# expression and the qubit-count match is checked at transpile
# time. Sections 4 and 5 spell out the per-mode argument shapes;
# this section is about features whose *behaviour* is
# mode-agnostic.

# %% [markdown]
# ### 3.1 Controlling any callable
#
# `qmc.control` accepts either a built-in gate function (`qmc.rx`,
# `qmc.h`, `qmc.p`, ...) or any user-defined `@qmc.qkernel`.
# `qmc.control` does not care which: it looks at the callable's
# signature, extracts the quantum operands and the classical
# parameters, and emits a controlled-U around the rest. In the
# example below, `ch` controls a single primitive and `cg`
# controls a user-defined kernel body with two gates inside.


# %%
@qmc.qkernel
def _h_then_rx(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return q


@qmc.qkernel
def control_any_callable_demo() -> qmc.Vector[qmc.Bit]:
    # q[0] is the shared control; q[1] / q[2] are the two targets.
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    ch = qmc.control(qmc.h)  # built-in gate function
    q[0], q[1] = ch(q[0], q[1])
    cg = qmc.control(_h_then_rx)  # user @qmc.qkernel
    q[0], q[2] = cg(q[0], q[2], theta=math.pi / 4)
    return qmc.measure(q)


control_any_callable_demo.draw()

# %% [markdown]
# ### 3.2 Sub-kernel taking `Vector[Qubit]`
#
# A controlled kernel may take a `Vector[Qubit]` argument. The
# caller passes a `Vector` or a `VectorView` of the matching
# length; the controlled-U emit pass resolves the vector operand
# to its physical target qubits and hands the inner block to the
# backend. Qiskit emits the whole controlled block as a single
# native gate, while CUDA-Q controls each gate of the block
# individually. Not every backend can control a multi-qubit inner
# block this way — QuriParts rejects the multi-target case above
# with a clear `EmitError`, so run such kernels on Qiskit or
# CUDA-Q. Either way the call site stays the same — you do not
# have to spell out one operand per qubit.


# %%
@qmc.qkernel
def _vec_h(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    qs[0] = qmc.h(qs[0])
    qs[1] = qmc.h(qs[1])
    return qs


@qmc.qkernel
def vec_target_demo() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(3, "qs")
    qs[0] = qmc.x(qs[0])
    cg = qmc.control(_vec_h, num_controls=1)
    qs[0], qs[1:3] = cg(qs[0], qs[1:3])
    return qmc.measure(qs)


vec_target_demo.draw()

# %% [markdown]
# ### 3.3 Default values from the controlled kernel's signature
#
# When the controlled `@qmc.qkernel` declares a Python default for
# a classical parameter, callers may either omit that keyword
# (letting the default flow through) or override it positionally
# at the call site. `qmc.control` fills the missing value in via
# `inspect.Signature.bind + apply_defaults`, so the resolved value
# reaches the controlled-U just like a normal direct call. (Only
# `@qmc.qkernel` callables can carry defaults — see Section 6.7
# for what happens if you try to do the same with a plain Python
# function.)
#
# Both forms work in either mode. The cells below show the omit
# form first in concrete mode, then the same omit form repeated in
# symbolic mode to make explicit that "default in both modes" is
# real and not a concrete-only convenience.


# %%
@qmc.qkernel
def _phase(q: qmc.Qubit, theta: qmc.Float = math.pi / 2) -> qmc.Qubit:
    return qmc.rx(q, theta)


@qmc.qkernel
def default_arg_demo() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(_phase)
    c, t = cg(c, t)  # theta defaults to math.pi / 2
    return qmc.measure(t)


default_arg_demo.draw()


# %%
# Same `_phase` kernel, this time controlled with a symbolic
# `num_controls=n - 1`.  The `theta=math.pi / 2` default still
# applies even though the caller never names it.  Replace the
# omitted `theta` with a positional override at the call site
# (`cg(q[0 : n - 1], q[n - 1], math.pi / 4)`) when you want a
# different angle without switching to a kwarg.
@qmc.qkernel
def default_arg_demo_symbolic(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(n - 1):
        q[i] = qmc.x(q[i])  # drive every control slot to |1>
    cg = qmc.control(_phase, num_controls=n - 1)
    q[0 : n - 1], q[n - 1] = cg(q[0 : n - 1], q[n - 1])
    return qmc.measure(q)


default_arg_demo_symbolic.draw(n=3)

# %% [markdown]
# ### 3.4 Classical keyword arguments in any order
#
# Classical kwargs at the call site are matched by name and
# reordered to follow the controlled kernel's declared signature,
# so the same call compiled with the kwargs in either order
# produces the same circuit. The assertion at the end of the cell
# verifies that explicitly by comparing the transpiled Qiskit
# circuits character-for-character.


# %%
@qmc.qkernel
def _two_param(q: qmc.Qubit, alpha: qmc.Float, beta: qmc.Float) -> qmc.Qubit:
    q = qmc.rx(q, alpha)
    q = qmc.rz(q, beta)
    return q


@qmc.qkernel
def kwargs_in_order() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(_two_param)
    c, t = cg(c, t, alpha=0.7, beta=1.3)
    return qmc.measure(t)


@qmc.qkernel
def kwargs_reversed() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(_two_param)
    c, t = cg(c, t, beta=1.3, alpha=0.7)
    return qmc.measure(t)


exe_a = transpiler.transpile(kwargs_in_order)
exe_b = transpiler.transpile(kwargs_reversed)
assert str(exe_a.compiled_quantum[0].circuit) == str(exe_b.compiled_quantum[0].circuit)

kwargs_in_order.draw()

# %% [markdown]
# ### 3.5 Controlling `U^k` with `power=`
#
# Passing `power=k` controls the *k-th power* of the underlying
# unitary instead of `U` itself — the standard pattern in QPE,
# where the j-th register applies a controlled-`U^(2**j)`.
# `power` accepts a Python `int` (resolved at compile time) **or**
# a `qmc.UInt` handle (resolved at transpile time from
# `bindings`), and this works regardless of whether `num_controls`
# is concrete or symbolic. Both shapes are shown side by side.


# %%
@qmc.qkernel
def power_demo_concrete() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(qmc.rx)  # num_controls = 1 (concrete)
    c, t = cg(c, t, angle=math.pi / 4, power=3)  # power is a Python int
    return qmc.measure(t)


@qmc.qkernel
def power_demo_symbolic(k: qmc.UInt) -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    cg = qmc.control(qmc.rx)
    c, t = cg(c, t, angle=math.pi / 4, power=k)  # power is a UInt handle
    return qmc.measure(t)


power_demo_concrete.draw()

# %%
# Symbolic-power example needs a binding for `k` before draw / transpile.
power_demo_symbolic.draw(k=3)

# %% [markdown]
# ## 4. Multiple positional control arguments (concrete mode)
#
# The shapes in this section list each control as its own
# positional argument. They are shown here in concrete mode —
# where every control's qubit count is a literal — but they are
# **not** concrete-only: symbolic mode accepts the same
# multi-argument prefix, as §5.5 demonstrates by lifting exactly
# these shapes to a `UInt` `num_controls`. Concrete mode is just
# the shortest way to introduce them.

# %% [markdown]
# ### 4.1 Multiple separate positional control args (CCX style)
#
# With `num_controls=2`, the call site lists each control qubit
# as its own positional argument before the target. The example
# below is the canonical CCX (Toffoli): two controls `c0`, `c1`
# and a target `t`. The same pattern extends to `num_controls=3`
# (CCCX), `num_controls=4`, etc., as long as you have that many
# distinct `Qubit` handles to pass in.


# %%
@qmc.qkernel
def toffoli_demo() -> qmc.Bit:
    c0 = qmc.qubit(name="c0")
    c1 = qmc.qubit(name="c1")
    t = qmc.qubit(name="t")
    c0 = qmc.x(c0)
    c1 = qmc.x(c1)
    ccx = qmc.control(qmc.x, num_controls=2)
    c0, c1, t = ccx(c0, c1, t)
    return qmc.measure(t)


toffoli_demo.draw()

# %% [markdown]
# ### 4.2 Mixing scalar Qubit and `VectorView` controls
#
# The positional control prefix in concrete mode may freely mix
# scalar `Qubit` handles, `VectorView` slices, and whole
# `Vector[Qubit]`s, as long as the total qubit count adds up to
# `num_controls`. Here the three controls for a `num_controls=3`
# controlled-H come from `qs[0]` (a scalar `Qubit`, 1 qubit) plus
# `qs[1:3]` (a `VectorView`, 2 qubits). Symbolic mode offers the
# same freedom through its multi-arg form (§5.5).


# %%
@qmc.qkernel
def mixed_controls_demo() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(5, "qs")
    qs[0] = qmc.x(qs[0])
    qs[1] = qmc.x(qs[1])
    qs[2] = qmc.x(qs[2])
    cg = qmc.control(qmc.h, num_controls=3)
    qs[0], qs[1:3], qs[3] = cg(qs[0], qs[1:3], qs[3])
    return qmc.measure(qs)


mixed_controls_demo.draw()

# %% [markdown]
# ## 5. Symbolic-mode patterns
#
# This section covers what you get when `num_controls` is a
# `qmc.UInt` handle (or any `UInt` expression like `n - 1`): the
# number of *active* controls is decided at transpile time from
# `bindings`, rather than at the moment
# `qmc.control(..., num_controls=...)` is evaluated. Some of what
# follows is genuinely symbolic-only (a pool with a symbolic
# length, and `control_indices=`); the multi-arg shape in 5.5 is
# simply the symbolic counterpart of the concrete demos in
# Section 4.
#
# Two call-site shapes are supported for the control side:
#
# - **Single-pool form** (5.1 – 5.4): one `Vector[Qubit]` or
#   `VectorView` is passed as the control argument and the entire
#   pool — or the subset chosen by `control_indices=` — acts
#   as the active controls.
# - **Multi-arg form** (5.5): the control prefix is several
#   positional arguments (scalar `Qubit`, `VectorView` slices,
#   whole `Vector`s, or a mix) whose total qubit count equals
#   `num_controls` at transpile time.  This is what concrete mode
#   already does (Section 4.1 / 4.2), now lifted to symbolic
#   `num_controls`.
#
# A `control_indices=` keyword is available in symbolic mode
# only; it picks which slots of a single-pool argument actually
# wire in as active controls (the rest pass through untouched).
# `control_indices=` is only valid with the single-pool form;
# combining it with the multi-arg form is rejected at compose
# time.

# %% [markdown]
# ### 5.1 `num_controls = n` over a whole pool
#
# The simplest symbolic shape: `num_controls=n` with the entire
# pool (length `n`) used as the active controls. The kernel
# parameter `n` is concretised at transpile time via `bindings`.
# The controlled-gate shape itself adapts once `n` is bound; in
# this demo we bind `n=3` because the surrounding body literally
# initializes `ctrls[0]`, `ctrls[1]`, `ctrls[2]` with `qmc.x`,
# so binding `n < 3` would index out of range and binding
# `n > 3` would leave some pool slots in `|0>` (transpilation
# would still succeed, but the prepared state would not match
# the "all controls active" intent of the demo). Replace the
# fixed initializations with a loop if you want a body that
# scales with `n`.


# %%
@qmc.qkernel
def symbolic_pool(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    ctrls = qmc.qubit_array(n, "ctrls")
    tgt = qmc.qubit(name="tgt")
    ctrls[0] = qmc.x(ctrls[0])
    ctrls[1] = qmc.x(ctrls[1])
    ctrls[2] = qmc.x(ctrls[2])
    cg = qmc.control(qmc.x, num_controls=n)
    ctrls, tgt = cg(ctrls, tgt)
    return qmc.measure(ctrls)


symbolic_pool.draw(n=3)

# %% [markdown]
# ### 5.2 Canonical `n - 1` multi-controlled form
#
# A frequent shape in multi-controlled-X designs: the first
# `n - 1` qubits of a register become controls, the last one
# becomes the target. The bound on `num_controls` is the
# symbolic expression `n - 1`, and the control argument is the
# slice `qs[0:n - 1]`.


# %%
@qmc.qkernel
def mcx_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(n, "qs")
    qs[0] = qmc.x(qs[0])
    qs[1] = qmc.x(qs[1])
    qs[2] = qmc.x(qs[2])
    mcx = qmc.control(qmc.x, num_controls=n - 1)
    qs[0 : n - 1], qs[n - 1] = mcx(qs[0 : n - 1], qs[n - 1])
    return qmc.measure(qs)


mcx_demo.draw(n=4)

# %% [markdown]
# ### 5.3 Selecting a subset with `control_indices=`
#
# When the control pool is wider than the number of active
# controls you want, the `control_indices=` keyword (symbolic
# mode only) picks exactly which pool slots are wired in. The
# remaining slots are passed through untouched — they sit on the
# wires but emit no extra gate of their own. The indices do not
# have to be contiguous.
#
# In the example below the pool has 4 qubits but the three
# active controls are `pool[0]`, `pool[1]`, `pool[3]`
# (`control_indices=[0, 1, 3]`). `pool[2]` is along for the
# ride: no control dot is drawn on it, and the vertical
# connection line of the MCX skips over it.


# %%
@qmc.qkernel
def subset_pool(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    pool = qmc.qubit_array(n, "pool")
    tgt = qmc.qubit(name="tgt")
    pool[0] = qmc.x(pool[0])
    pool[1] = qmc.x(pool[1])
    pool[3] = qmc.x(pool[3])  # pool[2] left at |0> — it is the inactive slot
    cg = qmc.control(qmc.x, num_controls=k_ctrls)
    pool, tgt = cg(pool, tgt, control_indices=[0, 1, 3])
    return qmc.measure(pool)


subset_pool.draw(n=4, k_ctrls=3)

# %% [markdown]
# ### 5.4 `control_indices` with `UInt` entries
#
# Each entry inside `control_indices` may be a Python `int`
# literal, a `qmc.UInt` handle, or any arithmetic expression
# over `UInt` values. Cheap structural checks on literal `int`
# entries (rejecting `bool`, negative values, and entries that
# duplicate another literal `int` in the list) are done at
# compose time; everything else — length agreement with
# `num_controls`, range against the pool size, and any check
# that depends on a `UInt` resolving to a concrete value — is
# deferred until transpile time once `bindings` make the
# parameters concrete.
#
# Here the third active control is `pool[n - 1]` — "the last
# pool slot" expressed as `UInt` arithmetic. At `n = 4` it still
# resolves to slot 3, which leaves `pool[2]` inactive; the
# compiled circuit is the same as in 5.3 and only the way the
# index is spelled differs.


# %%
@qmc.qkernel
def subset_pool_with_uint(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    pool = qmc.qubit_array(n, "pool")
    tgt = qmc.qubit(name="tgt")
    pool[0] = qmc.x(pool[0])
    pool[1] = qmc.x(pool[1])
    pool[3] = qmc.x(pool[3])
    cg = qmc.control(qmc.x, num_controls=k_ctrls)
    pool, tgt = cg(pool, tgt, control_indices=[0, 1, n - 1])
    return qmc.measure(pool)


subset_pool_with_uint.draw(n=4, k_ctrls=3)

# %% [markdown]
# ### 5.5 Multi-arg control prefix
#
# When the controls are spread across several positional
# arguments — typically because you want some slots of a single
# `Vector` to be active controls and another slot of the *same*
# `Vector` to be the target — symbolic mode accepts the same
# multi-arg call shape concrete mode does (Section 4.1 / 4.2).
# The qubit-count sum of the control prefix args is matched
# against `num_controls` at transpile time.
#
# The kernel below is a "controlled increment": when
# `q[control_index]` is `|1>` it applies `q -> q + 1 (mod
# 2**(n-1))` to the other bits of `q`.  Each iteration takes one
# scalar from `q` as the gating control, a `VectorView` slice
# `q[0:target_idx]` as the inner controls, and `q[target_idx]`
# as the target — all from the same `q`, in disjoint slots, and
# all with `num_controls = target_idx + 1` (a symbolic expression
# in the loop variable).  Before this form was supported the
# kernel was uncomposable because symbolic mode required a single
# `Vector` argument.


# %%
@qmc.qkernel
def apply_controlled_shift_plus_one(
    q: qmc.Vector[qmc.Qubit], control_index: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    n = q.shape[0]
    for k in qmc.range(n - 1):
        target_idx = n - 2 - k
        ctrl_main = q[control_index]
        prefix = q[0:target_idx]
        tgt = q[target_idx]
        cg = qmc.control(qmc.x, num_controls=target_idx + 1)
        ctrl_main, prefix, tgt = cg(ctrl_main, prefix, tgt)
        q[control_index] = ctrl_main
        q[0:target_idx] = prefix
        q[target_idx] = tgt
    return q


@qmc.qkernel
def controlled_increment_demo(
    n: qmc.UInt, control_index: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, "q")
    q[control_index] = qmc.x(q[control_index])  # drive the gating bit to |1>
    q = apply_controlled_shift_plus_one(q, control_index)
    return qmc.measure(q)


controlled_increment_demo.draw(n=4, control_index=3)

# %% [markdown]
# A few notes on the multi-arg form:
#
# - The call-site args are split into "control prefix" and
#   "sub-kernel positional" by the controlled kernel's signature:
#   any positional parameter not provided via kwargs must arrive
#   positionally, and everything *before* that trailing block is
#   the control prefix.  In the example, `qmc.x` takes one
#   `Qubit` positional, so the last arg (`tgt`) is the target
#   and the first two (`ctrl_main`, `prefix`) are the controls.
# - The borrow tracker is satisfied as long as the slots the
#   different args reach are disjoint.  Static disjointness is
#   checked when the index bounds are literal; symbolic bounds
#   (`q[0:target_idx]` versus `q[target_idx]` versus
#   `q[control_index]`) lean on the bound predicates the tracker
#   already supports for register partitioning.
# - `control_indices=` is rejected in the multi-arg form (see
#   Section 6 for the reject case): if you need subset
#   selection, use the single-pool form (5.3 / 5.4); if you need
#   multi-arg flexibility, accept the entire prefix as active.

# %% [markdown]
# ## 6. Patterns that don't work
#
# Each cell below tries one rejected call shape and asserts the
# expected exception type with a small `expect_error` helper.
# The helper only catches the *expected* exception class; any
# other exception propagates as a normal cell error so a
# regression that changes which exception fires surfaces with a
# traceback in the notebook. Missing the exception entirely
# raises an `AssertionError`. The "Mode" column tells you which
# mode of `qmc.control` each rejection applies to.
#
# | Case | Mode | Exception |
# | --- | --- | --- |
# | 6.1 control qubit count crosses an argument boundary | concrete | `ValueError` |
# | 6.2 `control_indices=` in concrete mode | concrete | `ValueError` |
# | 6.3 symbolic-length `VectorView` in concrete | concrete | `NotImplementedError` |
# | 6.4 typo in classical kwarg | both | `TypeError` |
# | 6.5 invalid `power` (zero or `bool`) | both | `ValueError` / `TypeError` |
# | 6.6 `num_controls=0` literal | concrete | `ValueError` |
# | 6.7 plain function with a Python default | both | `TypeError` |
# | 6.8 same-pool slot reused as target | symbolic | `UnreturnedBorrowError` |
# | 6.9 multi-arg control prefix + `control_indices=` | symbolic | `ValueError` |


# %%
def expect_error(label: str, exc_type: type, body) -> None:
    """Assert that ``body`` raises an exception of ``exc_type``.

    The helper only catches the *expected* exception class.  Any
    other exception propagates out untouched so a regression that
    swaps the exception type surfaces with a normal traceback in
    the cell.  Missing the exception entirely raises an
    ``AssertionError``.
    """
    try:
        body()
    except exc_type as exc:
        print(f"[{type(exc).__name__}] {label}: {exc}")
        return
    raise AssertionError(
        f"{label}: expected {exc_type.__name__}, but no exception was raised"
    )


# %% [markdown]
# ### 6.1 Control qubit count crosses an argument boundary (concrete)
#
# Concrete mode walks the positional arguments left-to-right,
# folding each one into the control list until the running total
# reaches `num_controls`. If one of those positional arguments
# would push the count *past* `num_controls` mid-argument — the
# example below passes a 5-qubit slice when only 3 controls are
# expected — the call is rejected at compose time with
# `ValueError` so you can split the offending argument cleanly
# at the boundary.
#
# (A *too-narrow* version of the same mistake — passing fewer
# control qubits than `num_controls` — looks different: extra
# positional arguments that you meant as targets get folded into
# the control list, and `qmc.control` then complains it has no
# target left. That surface is Python's own `TypeError: missing
# a required argument`, not a controlled-U `ValueError`.)


# %%
def case_count_mismatch() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        qs = qmc.qubit_array(6, "qs")
        cg = qmc.control(qmc.x, num_controls=3)
        view, t = cg(qs[0:5], qs[5])  # 5 qubits supplied, 3 expected
        qs[0:5] = view
        return qmc.measure(qs[5])

    _ = kernel.block


expect_error("control count mismatch", ValueError, case_count_mismatch)

# %% [markdown]
# ### 6.2 `control_indices=` in concrete mode (concrete)
#
# `control_indices` makes sense only when there is a control
# *pool* to select from, which is a symbolic-mode concept.
# Supplying it alongside a concrete `num_controls` raises
# `ValueError` at compose time.


# %%
def case_control_indices_in_concrete() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(qmc.x)  # num_controls defaults to 1 (concrete)
        c, t = cg(c, t, control_indices=[0])
        return qmc.measure(t)

    _ = kernel.block


expect_error(
    "control_indices in concrete mode",
    ValueError,
    case_control_indices_in_concrete,
)

# %% [markdown]
# ### 6.3 Symbolic-length `VectorView` in concrete mode (concrete)
#
# Concrete mode must compute the qubit count of every control
# argument at compile time. A slice whose length depends on a
# `UInt` (here `qs[0:m]` for symbolic `m`) is not yet supported
# in concrete mode and raises `NotImplementedError`. The
# workaround is to switch to symbolic mode — `num_controls=m`
# with `cg(qs, t)` accepts exactly this shape (see Section 5.1).


# %%
def case_symbolic_view_in_concrete() -> None:
    @qmc.qkernel
    def kernel(m: qmc.UInt) -> qmc.Bit:
        qs = qmc.qubit_array(m, "qs")
        cg = qmc.control(qmc.x, num_controls=3)
        view, q_out = cg(qs[0:m], qs[m - 1])
        qs[0:m] = view
        qs[m - 1] = q_out
        return qmc.measure(qs[m - 1])

    _ = kernel.block


expect_error(
    "symbolic-length VectorView in concrete mode",
    NotImplementedError,
    case_symbolic_view_in_concrete,
)

# %% [markdown]
# ### 6.4 Typo in a classical keyword argument (both modes)
#
# `qmc.control` inspects the controlled kernel's signature, so an
# unknown keyword name is caught at compose time. The error
# message lists the parameters `qmc.control` actually understands.


# %%
def case_kwarg_typo() -> None:
    @qmc.qkernel
    def _gate(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
        return qmc.rx(q, angle)

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(_gate)
        c, t = cg(c, t, agnle=0.5)  # typo: agnle -> angle
        return qmc.measure(t)

    _ = kernel.block


expect_error("classical kwarg typo", TypeError, case_kwarg_typo)

# %% [markdown]
# ### 6.5 Invalid `power` (both modes)
#
# `power` must be a strictly positive integer (`int` or
# `qmc.UInt`). Zero and negative values raise `ValueError`. A
# Python `bool` is rejected as `TypeError` so that `power=True`
# does not silently mean `power=1`. The same restriction applies
# in both concrete and symbolic mode.


# %%
def case_power_zero() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(qmc.x)
        c, t = cg(c, t, power=0)
        return qmc.measure(t)

    _ = kernel.block


def case_power_bool() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(qmc.x)
        c, t = cg(c, t, power=True)
        return qmc.measure(t)

    _ = kernel.block


expect_error("power=0", ValueError, case_power_zero)
expect_error("power=True (bool)", TypeError, case_power_bool)

# %% [markdown]
# ### 6.6 `num_controls=0` literal (concrete)
#
# A controlled gate with zero controls would just be the
# underlying gate, which makes controlling it meaningless. When
# the argument is a Python `int < 1`, `qmc.control` rejects this at
# the moment it is evaluated, with `ValueError` (negative `int`
# is rejected the same way). A `qmc.UInt` handle that *resolves*
# to zero is a different story: `qmc.control` does not see the
# value at evaluation time. Whether the zero is caught later
# depends on the control-argument shape and the backend — Qiskit
# surfaces it as a validation or backend-side error, but some
# backends (QuriParts) instead emit a degenerate circuit without
# complaining. The safe rule is to bind any symbolic
# `num_controls` to a strictly positive value yourself.


# %%
def case_num_controls_zero() -> None:
    qmc.control(qmc.x, num_controls=0)


expect_error("num_controls=0", ValueError, case_num_controls_zero)

# %% [markdown]
# ### 6.7 Plain function with a Python default (both modes)
#
# When the callable passed to `qmc.control` is not a `@qmc.qkernel`
# (just a plain Python function), `qmc.control` auto-synthesises a
# kernel around it. The synthesiser cannot turn Python-side
# default values into IR-level defaults, so plain functions with
# defaults are rejected at the moment `qmc.control(...)` is
# called. The fix is to mark the function as a `@qmc.qkernel`
# (where defaults are tracked end-to-end) or to drop the default
# and pass the value explicitly at the call site.


# %%
def case_plain_fn_with_default() -> None:
    def _bad_sub(q: qmc.Qubit, theta: qmc.Float = 0.5) -> qmc.Qubit:
        return qmc.rx(q, theta)

    qmc.control(_bad_sub)


expect_error("plain function with default value", TypeError, case_plain_fn_with_default)

# %% [markdown]
# ### 6.8 Same-pool slot reused as target — single-pool form (symbolic)
#
# When using the single-pool form (`cg(pool, ...)` with
# `control_indices=`), it is tempting to also pass one of the
# pool's inactive slots as the target — e.g.
# `cg(pool, pool[2], control_indices=[0, 1, 3])` so that
# `pool[2]` becomes the target of the controlled-U. The call
# site is rejected by the linear-type borrow tracker because the
# pool is already being consumed as one argument while `pool[2]`
# is being borrowed for another, which surfaces as
# `UnreturnedBorrowError` at compose time.
#
# Workarounds (preferred order):
#
# 1. **Multi-arg symbolic form (Section 5.5).** Pass each slot or
#    sub-view as its own positional argument:
#    `cg(pool[0], pool[1], pool[3], pool[2])` (or the
#    slice/scalar mix the controlled-increment example uses).
#    Each argument is a separate borrow from `pool`, the borrow
#    tracker checks disjointness, and `num_controls` matches the
#    qubit-count sum at transpile time.
# 2. **Concrete mode (Section 4.2).** If `num_controls` is a
#    Python `int`, the same multi-arg shape works in concrete
#    mode without any symbolic plumbing.


# %%
def case_pool_slot_as_target() -> None:
    @qmc.qkernel
    def kernel(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        pool = qmc.qubit_array(n, "pool")
        cg = qmc.control(qmc.x, num_controls=k_ctrls)
        pool, q = cg(pool, pool[2], control_indices=[0, 1, 3])
        pool[2] = q
        return qmc.measure(pool)

    _ = kernel.block


expect_error(
    "same-pool slot reused as target",
    UnreturnedBorrowError,
    case_pool_slot_as_target,
)

# %% [markdown]
# ### 6.9 Multi-arg control prefix + `control_indices=` (symbolic)
#
# The two symbolic-mode features are mutually exclusive.
# `control_indices=` only makes sense over a single control
# pool (one `Vector` argument), and combining it with multiple
# positional control args raises `ValueError` at compose time
# with an explicit message.  If you need both subset selection
# and per-slot routing, choose: the single-pool form (5.3 / 5.4)
# for subset selection, or the multi-arg form (5.5) for per-slot
# routing — not both.


# %%
def case_multi_arg_with_control_indices() -> None:
    @qmc.qkernel
    def kernel(n: qmc.UInt, k: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, "q")
        ctrl_main = q[0]
        prefix = q[1:k]
        tgt = q[k]
        cg = qmc.control(qmc.x, num_controls=k + 1)
        ctrl_main, prefix, tgt = cg(ctrl_main, prefix, tgt, control_indices=[0, 1, 2])
        q[0] = ctrl_main
        q[1:k] = prefix
        q[k] = tgt
        return qmc.measure(q)

    _ = kernel.block


expect_error(
    "multi-arg + control_indices",
    ValueError,
    case_multi_arg_with_control_indices,
)

# %% [markdown]
# ## 7. Summary
#
# `qmc.control(fn, num_controls=...)` returns a reusable
# `ControlledGate`. The right mental model is a two-axis matrix
# rather than two separate APIs:
#
# - **Mode is the type of `num_controls`.** Python `int` puts
#   you in *concrete* mode; a `qmc.UInt` handle (or any `UInt`
#   expression like `n - 1`) puts you in *symbolic* mode.
# - **Most features are mode-agnostic.** Controlling any callable
#   (built-in or `@qmc.qkernel`), sub-kernels that take
#   `Vector[Qubit]`, defaults from the controlled signature,
#   classical-kwarg reordering, and `power=` all work the same
#   way in either mode (Section 3).
# - **A few features are symbolic-only.** Control pools whose
#   length is symbolic and the subset selector `control_indices=`
#   exist only in symbolic mode (Section 5). The control-argument
#   *shapes* themselves — a single pool, or several positional
#   args mixing scalar `Qubit` / `VectorView` / `Vector` — work in
#   either mode; Section 4 shows them in concrete mode and §5.5 in
#   symbolic mode.
#
# Practical decision rule: reach for *symbolic* mode whenever
# the control count is a kernel parameter or an expression over
# one — including the very common "all but one" form
# `num_controls=n - 1`. Reach for *concrete* mode when the count
# is a literal and the controls live on specific qubits you can
# name.
#
# Section 6 doubles as a regression test for the validation
# rules of both modes: every rejected shape asserts the
# expected exception type, so a change that loses (or alters) a
# check would surface immediately in the docs build.
