# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to use `BinaryModel`
#
# This tutorial walks through `BinaryModel`, the core class in Qamomile's optimization module for describing unconstrained optimization problems over binary variables. `BinaryModel` supports both `binary` (0/1) and `spin` (-1/1) variable types, and the two representations can be converted into each other through the `change_vartype` method.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import numpy as np

from qamomile.optimization.binary_model.expr import VarType, binary
from qamomile.optimization.binary_model.model import BinaryModel

# %% [markdown]
# ## Building a `BinaryModel` the low-level way with `BinaryExpr`
# Let's start with the most direct route: building a `BinaryModel` by hand from a `BinaryExpr`. In practice you probably won't reach for this path very often, but the higher-level constructors shown later all rely on `BinaryExpr` under the hood, so a quick look here will make the rest of the class much easier to understand.
#
# `BinaryExpr` represents a polynomial expression over binary variables, constant term included. Variables are tracked by integer indices — the indices do **not** need to start at zero. To create a single binary (0/1) variable, call `binary(index)`. As a running example, let's build $x_1 + 2 x_3 + 3 x_1 x_3 + 5$.
#
# > Note: The explanation below uses binary (0/1) variables, but `BinaryExpr` works just as well with spin (-1/1) variables — use `qamomile.optimization.binary_model.expr.spin` to create one. For simplicity we stop at quadratic terms, but `BinaryExpr` also supports cubic and higher-order terms: just multiply in an additional `BinaryExpr`.

# %%
x_1 = binary(1)  # BinaryExpr representing x_1
x_3 = binary(3)  # BinaryExpr representing x_3

naive_expr = x_1 + 2 * x_3 + 3 * x_1 * x_3 + 5
naive_expr

# %% [markdown]
# A `BinaryExpr` exposes the following attributes:
# - `vartype`: the variable type (spin (-1/1) or binary (0/1))
# - `constant`: the constant term
# - `coefficients`: the coefficients of each term
#
# For the expression above they look like this:

# %%
print("vartype:", naive_expr.vartype)
assert naive_expr.vartype == VarType.BINARY
print("constant:", naive_expr.constant)
assert naive_expr.constant == 5.0
print("coefficients:", naive_expr.coefficients)
assert naive_expr.coefficients == {(1,): 1.0, (3,): 2.0, (1, 3): 3.0}


# %% [markdown]
# As the output shows, `BinaryExpr.coefficients` is a dictionary whose keys are tuples of variable indices and whose values are the corresponding coefficients. For example, the linear terms for `x_1` and `x_3` map to the keys `(1,)` and `(3,)` with values `1.0` and `2.0`, and the quadratic term `3 * x_1 * x_3` maps to `(1, 3)` with value `3.0`.

# %% [markdown]
# Now let's feed that `BinaryExpr` into `BinaryModel`. The constructor accepts a `BinaryExpr` directly. The resulting model stores the objective function using the following attributes:
# - `vartype`: the variable type
# - `constant`: the constant term
# - `linear`: coefficients of the linear terms
# - `quad`: coefficients of the quadratic terms
# - `higher`: coefficients of cubic and higher-order terms
# - `coefficients`: all coefficients in one dictionary (`linear` + `quad` + `higher`)
# - `num_bits`: the number of variables

# %%
naive_model = BinaryModel(naive_expr)

print("vartype:", naive_model.vartype)
assert naive_model.vartype == VarType.BINARY
print("constant:", naive_model.constant)
assert naive_model.constant == 5.0
print("linear:", naive_model.linear)
assert naive_model.linear == {0: 1.0, 1: 2.0}
print("quad:", naive_model.quad)
assert naive_model.quad == {(0, 1): 3.0}
print("higher:", naive_model.higher)
assert naive_model.higher == {}
print("coefficients:", naive_model.coefficients)
assert naive_model.coefficients == {(0,): 1.0, (1,): 2.0, (0, 1): 3.0}

# %% [markdown]
# There is one important detail to notice. Just like `BinaryExpr.coefficients`, `BinaryModel.coefficients` uses tuples of variable indices as keys — but the indices are **not** the same as the ones in the original `BinaryExpr`. On construction, `BinaryModel` re-labels the variables so that indices form a contiguous range starting at zero. In our example:
#
# In the original `BinaryExpr`:
# - `x_1`'s linear-term key is `(1,)`
# - `x_3`'s linear-term key is `(3,)`
# - the `x_1 * x_3` quadratic-term key is `(1, 3)`
#
# In the resulting `BinaryModel`:
# - `x_1`'s linear-term key is `(0,)`
# - `x_3`'s linear-term key is `(1,)`
# - the `x_1 * x_3` quadratic-term key is `(0, 1)`
#
# To recover the mapping between the original and the re-labeled indices, `BinaryModel` exposes two dictionaries:
# - `index_new_to_origin`: re-labeled index → original `BinaryExpr` index
# - `index_origin_to_new`: original `BinaryExpr` index → re-labeled index

# %%
print("index_new_to_origin:", naive_model.index_new_to_origin)
for new_index, original_index in naive_model.index_new_to_origin.items():
    print("---")
    print(
        f"naive_model.coefficients[(new_index, )] = {naive_model.coefficients[(new_index,)]}"
    )
    print(
        f"naive_expr.coefficients[(original_index, )] = {naive_expr.coefficients[(original_index,)]}"
    )

# %% [markdown]
# As mentioned at the top of this tutorial, `BinaryModel` provides a `change_vartype` method for converting between binary and spin variables. The model we just built uses binary variables, so let's convert it to a spin-variable model. `change_vartype` takes the target variable type (`qamomile.optimization.binary_model.expr.VarType`) as its argument — for example, `change_vartype(VarType.SPIN)` converts a binary model into a spin model.

# %%
spin_naive_model = naive_model.change_vartype(VarType.SPIN)
print("vartype:", spin_naive_model.vartype)
assert spin_naive_model.vartype == VarType.SPIN
print("constant:", spin_naive_model.constant)
assert spin_naive_model.constant == 29.0 / 4.0
print("linear:", spin_naive_model.linear)
assert spin_naive_model.linear == {0: -5.0 / 4.0, 1: -7.0 / 4.0}
print("quad:", spin_naive_model.quad)
assert spin_naive_model.quad == {(0, 1): 3.0 / 4.0}
print("higher:", spin_naive_model.higher)
assert spin_naive_model.higher == {}
print("coefficients:", spin_naive_model.coefficients)
assert spin_naive_model.coefficients == {
    (0,): -5.0 / 4.0,
    (1,): -7.0 / 4.0,
    (0, 1): 3.0 / 4.0,
}

# %% [markdown]
# Binary and spin variables are related by $s = 1 - 2x$, where $s$ is the spin variable and $x$ is the binary variable. Substituting this into our expression gives:
#
# $$
# \begin{align*}
# x_1 + 2 x_3 + 3 x_1 x_3 + 5
# &= \left( \frac{1 - s_1}{2} \right) + 2 \left( \frac{1 - s_3}{2} \right) + 3 \left( \frac{1 - s_1}{2} \right) \left( \frac{1 - s_3}{2} \right) + 5 \\
# &= \frac{3}{4} s_1 s_3 - \frac{5}{4} s_1 - \frac{7}{4} s_3 + \frac{29}{4} \\
# &= 0.75 s_1 s_3 - 1.25 s_1 - 1.75 s_3 + 7.25
# \end{align*}
# $$
#
# which matches the values printed above. Also note that `BinaryModel.change_vartype` keeps the (re-labeled) indices of the original `BinaryModel` unchanged.

# %% [markdown]
# ## Building a `BinaryModel` from QUBO, HUBO, or Ising
# So far we've built `BinaryModel` objects by hand through `BinaryExpr`. In practice, users rarely keep their models in `BinaryExpr` form, and rebuilding one every time would be tedious. To make this easier, `BinaryModel` provides class methods for constructing models directly from QUBO, HUBO, and Ising representations. Let's go through them one at a time. Note that these class methods internally build a `BinaryExpr` and feed it into the `BinaryModel` constructor, so although we will use 0-origin consecutive integers for variable indices in the examples below, the class methods themselves do not require that. As mentioned above, `BinaryModel` re-labels variables and `index_new_to_origin` / `index_origin_to_new` can be used to track the mapping between the original and re-labeled indices.

# %% [markdown]
# ### Building from QUBO (`from_qubo`)
# `from_qubo` takes a `qubo` argument: a dictionary whose keys are tuples of variable indices and whose values are the corresponding coefficients. It also takes a `constant` argument for the constant term. As an example, consider the following QUBO matrix:
#
# $$
# \begin{bmatrix}
# 1 & 0.5 & 0 \\
# 0 & 2 & 1 \\
# 0 & 0 & 3
# \end{bmatrix}
# $$
#
# which encodes the objective function
#
# $$
# 1 x_0 + 2 x_1 + 3 x_2 + 0.5 x_0 x_1 + 1 x_1 x_2
# $$
#

# %%
qubo = {
    (0, 0): 1.0,
    (1, 1): 2.0,
    (2, 2): 3.0,
    (0, 1): 0.5,
    (1, 2): 1.0,
}
constant = 0.0
qubo_model = BinaryModel.from_qubo(qubo, constant)
print("vartype:", qubo_model.vartype)
assert qubo_model.vartype == VarType.BINARY
print("constant:", qubo_model.constant)
assert qubo_model.constant == 0.0
print("linear:", qubo_model.linear)
assert qubo_model.linear == {0: 1.0, 1: 2.0, 2: 3.0}
print("quad:", qubo_model.quad)
assert qubo_model.quad == {(0, 1): 0.5, (1, 2): 1.0}
print("higher:", qubo_model.higher)
assert qubo_model.higher == {}
print("coefficients:", qubo_model.coefficients)
assert qubo_model.coefficients == {
    (0,): 1.0,
    (1,): 2.0,
    (2,): 3.0,
    (0, 1): 0.5,
    (1, 2): 1.0,
}

# %% [markdown]
# ### Building from HUBO (`from_hubo`)
# `from_hubo` takes a `hubo` argument: a dictionary whose keys are tuples of variable indices and whose values are the corresponding coefficients, plus a `constant` argument for the constant term. As an example, consider the following HUBO:
#
# $$
# \begin{equation*}
# 1 x_0 + 2 x_1 + 3 x_2 + 0.5 x_0 x_1 + 1 x_1 x_2 + 0.1 x_0 x_1 x_2 \\
# \end{equation*}
# $$
#

# %%
hubo = {
    (0,): 1.0,
    (1,): 2.0,
    (2,): 3.0,
    (0, 1): 0.5,
    (1, 2): 1.0,
    (0, 1, 2): 0.1,
}
constant = 0.0
hubo_model = BinaryModel.from_hubo(hubo, constant)
print("vartype:", hubo_model.vartype)
assert hubo_model.vartype == VarType.BINARY
print("constant:", hubo_model.constant)
assert hubo_model.constant == 0.0
print("linear:", hubo_model.linear)
assert hubo_model.linear == {0: 1.0, 1: 2.0, 2: 3.0}
print("quad:", hubo_model.quad)
assert hubo_model.quad == {(0, 1): 0.5, (1, 2): 1.0}
print("higher:", hubo_model.higher)
assert hubo_model.higher == {(0, 1, 2): 0.1}
print("coefficients:", hubo_model.coefficients)
assert hubo_model.coefficients == hubo

# %% [markdown]
# ### Building from Ising (`from_ising`)
# `from_ising` takes the Ising coefficients through two separate arguments: `linear` and `quad`. `linear` is a dictionary whose keys are variable indices and whose values are the corresponding linear coefficients. `quad` is a dictionary whose keys are tuples of variable indices and whose values are the corresponding quadratic coefficients.
# > Note: higher-order Ising terms are not supported yet, but we plan to add support for them.

# %%
ising_linear = {
    0: -1.0,
    1: 2.0,
    2: -3.0,
}
ising_quad = {
    (0, 1): 0.5,
    (1, 2): -1.0,
}
constant = 0.0
ising_model = BinaryModel.from_ising(ising_linear, ising_quad, constant)
print("vartype:", ising_model.vartype)
assert ising_model.vartype == VarType.SPIN
print("constant:", ising_model.constant)
assert ising_model.constant == 0.0
print("linear:", ising_model.linear)
assert ising_model.linear == {0: -1.0, 1: 2.0, 2: -3.0}
print("quad:", ising_model.quad)
assert ising_model.quad == {(0, 1): 0.5, (1, 2): -1.0}
print("higher:", ising_model.higher)
assert ising_model.higher == {}
print("coefficients:", ising_model.coefficients)
assert ising_model.coefficients == {
    (0,): -1.0,
    (1,): 2.0,
    (2,): -3.0,
    (0, 1): 0.5,
    (1, 2): -1.0,
}

# %% [markdown]
# ## Normalization and energy evaluation
# `BinaryModel` ships with two normalization methods, `normalize_by_abs_max` and `normalize_by_rms`, and an energy evaluation method, `calc_energy`. Let's look at each of them.

# %% [markdown]
# ### Normalization
# `normalize_by_abs_max` rescales every coefficient by the largest absolute value. Let's normalize the first model we built, $x_1 + 2 x_3 + 3 x_1 x_3 + 5$ (`naive_expr`). The largest coefficient in absolute value is 3, so the normalized model becomes:
#
# $$
# \frac{1}{3} x_1 + \frac{2}{3} x_3 + 1 x_1 x_3 + \frac{5}{3}
# $$
#

# %%
normalized_model = naive_model.normalize_by_abs_max(replace=False)
print("original vartype:", naive_model.vartype)
print("normalized vartype:", normalized_model.vartype)
assert normalized_model.vartype == naive_model.vartype
print("---")
print("original constant:", naive_model.constant)
print("normalized constant:", normalized_model.constant)
assert normalized_model.constant == naive_model.constant / 3.0
print("---")
print("original linear:", naive_model.linear)
print("normalized linear:", normalized_model.linear)
assert normalized_model.linear == {0: 1.0 / 3.0, 1: 2.0 / 3.0}
print("---")
print("original quad:", naive_model.quad)
print("normalized quad:", normalized_model.quad)
assert normalized_model.quad == {(0, 1): 1.0}

# %% [markdown]
# `normalize_by_rms` rescales every coefficient by the following root-mean-square value:
#
# $$
# W = \sqrt{\frac{1}{\lvert E_2 \rvert} \sum_{i, j} (w_{(i, j)})^2 + \frac{1}{\lvert E_1 \rvert} \sum_i (w_i)^2}
# $$
#
# where $w_{(i, j)}$ is a quadratic-term coefficient, $w_i$ is a linear-term coefficient, $E_2$ is the set of quadratic terms, and $E_1$ is the set of linear terms. Let's apply it to the same `naive_expr` model. We have
# - $E_1$ = 2
# - $E_2$ = 1
# - $\sum_i (w_i)^2$ = $1^2 + 2^2$ = 5
# - $\sum_{i, j} (w_{(i, j)})^2$ = $3^2$ = 9
#
# so the RMS value is
#
# $$
# \sqrt{5 / 2 + 9 / 1} = \sqrt{2.5 + 9} = \sqrt{11.5} \approx 3.391
# $$
#
# and the normalized model becomes
#
# $$
# \frac{1}{3.391} x_1 + \frac{2}{3.391} x_3 + \frac{3}{3.391} x_1 x_3 + \frac{5}{3.391}
# \approx 0.295 x_1 + 0.590 x_3 + 0.884 x_1 x_3 + 1.475
# $$
#

# %%
normalized_model_rms = naive_model.normalize_by_rms(replace=False)
print("original vartype:", naive_model.vartype)
print("normalized vartype:", normalized_model_rms.vartype)
assert normalized_model_rms.vartype == naive_model.vartype
print("---")
print("original constant:", naive_model.constant)
print("normalized constant:", normalized_model_rms.constant)
assert normalized_model_rms.constant == naive_model.constant / np.sqrt(11.5)
print("---")
print("original linear:", naive_model.linear)
print("normalized linear:", normalized_model_rms.linear)
assert normalized_model_rms.linear == {0: 1.0 / np.sqrt(11.5), 1: 2.0 / np.sqrt(11.5)}
print("---")
print("original quad:", naive_model.quad)
print("normalized quad:", normalized_model_rms.quad)
assert normalized_model_rms.quad == {(0, 1): 3.0 / np.sqrt(11.5)}

# %% [markdown]
# ### Objective (energy) evaluation
# `calc_energy` evaluates the objective function (the "energy") for a given variable assignment. Let's compute the energy of $x_1 + 2 x_3 + 3 x_1 x_3 + 5$ (`naive_expr`) at $x_1 = 1$, $x_3 = 0$. The expected value is
#
# $$
# x_1 + 2 x_3 + 3 x_1 x_3 + 5 = 1 + 2 \cdot 0 + 3 \cdot 1 \cdot 0 + 5 = 6
# $$
#
# `calc_energy` expects a **`list[int]` in the variable order used by `BinaryModel`**. In our example, `BinaryModel` places `x_1` at index 0 and `x_3` at index 1, so we must pass `[1, 0]`. Using `BinaryModel.index_new_to_origin`, we can build that list mechanically from a solution in the original index space.

# %%
# Build a solution in the original-problem index space.
example_solution = {
    3: 0,  # x_3 = 0
    1: 1,  # x_1 = 1
}
# Convert it into a list[int] in BinaryModel's internal variable order.
solution_in_model_order = [
    example_solution[naive_model.index_new_to_origin[new_index]]
    for new_index in range(naive_model.num_bits)
]
# Evaluate the energy.
energy = naive_model.calc_energy(solution_in_model_order)
print("solution in model order:", solution_in_model_order)
assert solution_in_model_order == [1, 0]
print("energy:", energy)
assert energy == 6.0

# %% [markdown]
# Depending on the `vartype` of the target `BinaryModel`, `calc_energy` first validates that the provided values are either spin (-1/1) or binary (0/1) before computing the energy. As a result, passing spin values to a binary model (such as `naive_model`) raises an error.

# %%
# Convert the binary solution into the equivalent spin values.
example_spin_solution = [1 - 2 * solution for solution in solution_in_model_order]

try:
    energy = naive_model.calc_energy(example_spin_solution)
except ValueError as e:
    print("Raising an error here is the expected behavior.")
    print("Error:", e)


# %% [markdown]
# Of course, as long as the values match the `vartype` of the `BinaryModel`, `calc_energy` computes the energy just fine. Below we pass spin values to `spin_naive_model`. The two models use different variable types but encode the same problem, so they return the same energy.

# %%
energy_spin = spin_naive_model.calc_energy(example_spin_solution)
print("solution in model order (spin):", example_spin_solution)
assert example_spin_solution == [-1, 1]
print("energy (spin):", energy_spin)
assert energy_spin == energy

# %% [markdown]
# ## Building a `BinaryModel` from `OMMX`
# Finally, let's look at how to build a `BinaryModel` from an [`OMMX`](https://jij-inc.github.io/ommx/en/introduction.html) (Open Mathematical prograMming eXchange) instance. OMMX is an open data format — together with an SDK for manipulating it — designed to exchange mathematical optimization data between software systems and between people. Qamomile's optimization module understands OMMX directly, so if you stick to the built-in quantum algorithms you don't need to perform this conversion yourself. Still, it comes in handy when you build custom algorithms, so let's cover it here. We'll take the same model we've been using throughout this tutorial, assume it is given as an OMMX instance, and convert it into a `BinaryModel`.
#
# First, let's create the OMMX instance. To keep the example self-contained we construct it from low-level OMMX components, but in real use cases you would typically rely on an existing packaged instance or on [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html), a Python-based mathematical modeler for describing optimization problems.

# %%
# Define the model in OMMX form.
from ommx.v1 import DecisionVariable, Instance

ommx_x_1 = DecisionVariable.binary(1, name="x_1")
ommx_x_3 = DecisionVariable.binary(3, name="x_3")

instance = Instance.from_components(
    decision_variables=[ommx_x_1, ommx_x_3],
    objective=ommx_x_1 + 2 * ommx_x_3 + 3 * ommx_x_1 * ommx_x_3 + 5,
    constraints=[],
    sense=Instance.MINIMIZE,
)

# %% [markdown]
# OMMX instances provide `to_qubo` / `to_hubo` methods for extracting QUBO/HUBO coefficients. Passing the results directly to `BinaryModel.from_qubo` / `from_hubo` gives you a `BinaryModel` constructed from an OMMX instance.

# %%
qubo_from_ommx, constant = instance.to_qubo()
model_from_ommx = BinaryModel.from_qubo(qubo=qubo_from_ommx, constant=constant)
print("vartype:", model_from_ommx.vartype)
assert model_from_ommx.vartype == naive_model.vartype
print("constant:", model_from_ommx.constant)
assert model_from_ommx.constant == naive_model.constant
print("linear:", model_from_ommx.linear)
assert model_from_ommx.linear == naive_model.linear
print("quad:", model_from_ommx.quad)
assert model_from_ommx.quad == naive_model.quad
print("higher:", model_from_ommx.higher)
assert model_from_ommx.higher == naive_model.higher
print("coefficients:", model_from_ommx.coefficients)
assert model_from_ommx.coefficients == naive_model.coefficients

# %% [markdown]
# As discussion above, the variable indices in `BinaryModel` are re-labeled from the original OMMX instance, so the keys in `model_from_ommx.coefficients` do not match those in `qubo_from_ommx`. However, the mapping between the original and re-labeled indices is available through `model_from_ommx.index_new_to_origin` and `model_from_ommx.index_origin_to_new`, so you can use them to verify that the coefficients match correctly.

# %%
print("index_new_to_origin:", model_from_ommx.index_new_to_origin)
for original_index1, original_index2 in qubo_from_ommx.keys():
    print("---")
    if original_index1 == original_index2:
        new_index = model_from_ommx.index_origin_to_new[original_index1]
        print(
            f"model_from_ommx.coefficients[(new_index, )] = {model_from_ommx.coefficients[(new_index,)]}"
        )
    else:
        new_index1 = model_from_ommx.index_origin_to_new[original_index1]
        new_index2 = model_from_ommx.index_origin_to_new[original_index2]
        print(
            f"model_from_ommx.coefficients[(new_index1, new_index2)] = {model_from_ommx.coefficients[(new_index1, new_index2)]}"
        )
    print(
        f"qubo_from_ommx[(original_index, )] = {qubo_from_ommx[(original_index1, original_index2)]}"
    )

# %% [markdown]
# ## Summary
# We've covered `BinaryModel` from two angles: building it by hand with `BinaryExpr`, and building it from QUBO/HUBO/Ising representations. We also looked at normalization and energy evaluation, and at how to convert an OMMX instance into a `BinaryModel`. `BinaryModel` is a flexible class for describing unconstrained optimization problems over binary variables, and it can be built from a variety of input formats. In most cases you won't touch `BinaryExpr` directly — you'll convert from OMMX or use `from_qubo` / `from_hubo` / `from_ising` with a plain dictionary — but understanding the `BinaryExpr` route will give you a deeper appreciation of how `BinaryModel` works under the hood.

# %% [markdown]
# ## Related topics
# - [Solving MaxCut with QAOA: Building the Circuit from Scratch](./../../vqa/qaoa-maxcut): an example that creates a QUBO dictionary from a random graph built with networkx, defines a `BinaryModel` directly, and applies QAOA.
# - [QAOA for Graph Partitioning](./../../optimization/qaoa-graph-partition): an example that applies QAOA to an OMMX instance using Qamomile's `QAOAConverter`. It only touches `BinaryModel` directly for the normalization step, but it's a realistic end-to-end example based on an OMMX instance.
