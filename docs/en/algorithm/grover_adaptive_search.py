# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %% [markdown]
# ---
# tags: [algorithm, optimization, oracle-based]
# ---
#
# # Grover Adaptive Search for Combinatorial Polynomial Binary Optimization
#
# This tutorial demonstrates how to solve the **portfolio problem**
# using the Grover Adaptive Search (GAS) algorithm with Qamomile.
#
# This tutorial explains how to use the Grover Adaptive Search Method for Constrained Polynomial Binary Optimization from the paper 
# > Gilliam, Austin, et al. Quantum, 5, p428 (2021) https://doi.org/10.22331/q-2021-04-08-428.
#
# 1. Formulate the problem with [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html).
# 2. Create an instance with concrete data.
# 3. Use `GASConverter` to build the Grover Adaptive Search circuit.
# 4. Run the circuit inside a classical layer that control the number of iterations. Sample the circuit and keep the best solutions until a stopping criterion is reached.

# %% [markdown]
# ## Portfolio Problem Formulation
#
# Given $n$ assets (stocks, bonds, etc.). For each one, we decide to buy it or not (binary choice).
# We want to maximize returns while minimizing risk. The tension is:
#
# - $\mu$ tell us how profitable each asset is expected to be
# - $\Sigma$ tell us how assets move together (correlated assets amplify risk)
# - $q$ controls how much we care about risk vs. return
#
# We need to pick the best subset of assets so that the portfolio earns as much as possible without taking on too much correlated risk.
#
# \begin{equation}
# \min_{x\in \lbrace 0 , 1 \rbrace^n} \big( q x^T \Sigma x - \mu^T x \big)
# \end{equation}

# %% [markdown]
# ## Define the Problem with JijModeling

# %%
import jijmodeling as jm
import numpy as np

@jm.Problem.define("Portfolio Optimization (Unconstrained)")
def portfolio_problem(problem: jm.DecoratedProblem):
    n = problem.Length(description="Number of assets")
    q = problem.Float("q", description="Risk aversion factor")
    μ = problem.Float("μ", shape=(n,), description="Expected returns vector")
    Σ = problem.Float("Σ", shape=(n, n), description="Covariance matrix")

    x = problem.BinaryVar("x", shape=(n,), description="1 if asset i is selected")

    problem += (
        q * jm.sum(Σ[i, j] * x[i] * x[j] for i in n for j in n)
        - jm.sum(μ[i] * x[i] for i in n)
    )
portfolio_problem

# %% [markdown]
# ## Problem Instance
#
# In the following instance we have 9 assets:
#
# | Name    | Expected Return | Variance   |
# |---------|-----------------|------------|
# | Asset 1 | 22              | 12         |
# | Asset 2 | 4               | 15         |
# | Asset 3 | 19              | 10         |
# | Asset 4 | 3               | 18         |
# | Asset 5 | 23              | 14         |
# | Asset 6 | 2               | 20         |
# | Asset 7 | 5               | 11         |
# | Asset 8 | 25              | 16         |
# | Asset 9 | 3               | 13         |

# %%
q = 1
μ = np.array([22, 4, 19, 3, 23, 2, 5, 25, 3], dtype=int)
Σ = np.array([
    [12, -3,  4,  0, -2,  3,  0,  2, -1],
    [-3, 15,  0,  5,  1, -4,  2,  0,  3],
    [ 4,  0, 10, -6,  3,  2, -1,  4,  0],
    [ 0,  5, -6, 18, -4,  0,  3, -2,  5],
    [-2,  1,  3, -4, 14,  2, -3,  0,  2],
    [ 3, -4,  2,  0,  2, 20,  4, -3,  1],
    [ 0,  2, -1,  3, -3,  4, 11,  2, -2],
    [ 2,  0,  4, -2,  0, -3,  2, 16, -4],
    [-1,  3,  0,  5,  2,  1, -2, -4, 13],
], dtype=int)

# %% [markdown]
# ## Create the Instance
#
# We evaluate the JijModeling problem with the concrete data.

# %%
instance_data = {
    "n": 9,
    "q": int(q),
    "μ": μ.tolist(),
    "Σ": Σ.tolist(),
}
instance = portfolio_problem.eval(instance_data)

# %% [markdown]
# ## Set Up the GAS Converter
#
# `GASConverter` takes an OMMX instance and internally: Converts the problem to a QUBO (Quadratic Unconstrained Binary Optimization) form.
#
# The converter also have a transpile method that given the transpiler of your choosing, will build the Grover circuit that is used for the Adaptive Search.

# %%
from qamomile.optimization.gas import GASConverter
from qamomile.qiskit import QiskitTranspiler

converter = GASConverter(instance)
transpiler = QiskitTranspiler()

# %% [markdown]
# ## Visualize the Grover Circuit
#
# `GASConverter.transpile()` internally builds the sampling
# qkernel below and feeds it to the transpiler. 
# For visualization, we can use the `Transpiler.to_block` and the `MatplotlibDrawer.draw` function on the qkernel that is used inside the transpile method.

# %%
import qamomile.circuit as qmc
from qamomile.circuit.algorithm.gas import grover_algorithm
from qamomile.circuit.visualization import MatplotlibDrawer

@qmc.qkernel
def sampling_grover_algorithm(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    iters: qmc.UInt = 1
) -> qmc.Tuple[qmc.UInt, qmc.UInt]:
    q_output,q_input = grover_algorithm(
        n,
        m,
        y,
        linear,
        quad,
        iters
    )
    return qmc.measure(q_output), qmc.measure(q_input)


block = transpiler.to_block(
    sampling_grover_algorithm,
    bindings={
        "n" : converter.binary_model.num_bits,      # Number of input qubits = number of binary variables
        "m" : 9, #Number of output qubits, computed by the converter
        "y" : 0,  # Oracle threshold: marks states where f(x) < y (arbitrary value for visualization)
        "linear" : converter.binary_model.linear,   # Linear QUBO coefficients
        "quad" : converter.binary_model.quad,       # Quadratic QUBO coefficients
        "iters" : 1  # Number of Grover iterations (can be tuned)
    }
)
block = transpiler.inline(block)
MatplotlibDrawer(block).draw(fold_loops=False)

# %% [markdown]
# ### Inspecting the Building Blocks
#
# The idea of the Grover used for optimization is to, given a reference solution $y$, mark all the elements such that their functional value is lower than $y$, i.e. $f(x) < y$. This is achieved by the preparation oracle $A_y$ and the marker operator $O_y$. Because the preparation operator, prepare all input whose functional value is smaller than $y$ with a negative value, we can identify them by their Most Significant Bit (MSB) that turn to 1 in complementary two representation. Thus the marker operator is simply a $Z$ gate. Finally, the diffusion operator amplifies the amplitude of the marked states.
#
# The Grover Ansatz is made of the following component three :
#
# - $A_y$ the oracle that prepare the state $\sum \ket{x, y - f(x)}$ : Built from QFT encoding
# - $O_y = Z$ on the MSB
# - $D$ Diffusion Operator: propagate the amplitude of marked elements $f(x) < y$, built from a single multi-Ctrl-Z gates

# %%
from qamomile.circuit.algorithm.gas import apply_function_preparation_qubo

apply_function_preparation_qubo.draw(
    q_output = 9, 
    q_input = 9, 
    y = 0, 
    linear = converter.binary_model.linear, 
    quad = converter.binary_model.quad,
    inline = True,
    inline_depth = None,
    fold_loops = False
)

# %%
from qamomile.circuit.algorithm.gas import diffusion_op

diffusion_op.draw(
    q_input = 9,
    inline = True,
    inline_depth = None,
    fold_loops = False
)

# %% [markdown]
# ## Classical layer of the Adaptive Search
#
# In this section, we define the classical Adaptive Search. The Adaptive Search start from a random candidate $x$ and $y = f(x)$, then sample from the grover circuit and finally update the solution if the sample contains a better candidate. We stop the search when a stopping criterion is met, here we use several iterations without improvement as the stopping criterion.
#
# A remaining question is for how many operation to apply the Grover Search ? To handle this, the Adaptive Search use a randomized approach by sampling the number of iterations and slightly increasing the sampling space when no improvement is found.
#
#
# The following function is not part of Qamomile. It implements the classical outer loop of GAS using `converter.transpile()` and `converter.decode()` as the quantum primitives.
#
# `converter.transpile(transpiler, y=y, num_iterations=num_iterations)` builds the Grover circuit for the current threshold $y$ and Grover depth. 
#
# `exec.sample(executor, shots=256)` runs it on the backend. Even for Grover, multiple shots are needed because NISQ quantum devices are noisy. 
#
# `converter.decode(result)` maps the raw bitstring counts back to decision variable assignments.

# %%
from typing import Any
import numpy as np
import random
import ommx
from qiskit_aer import AerSimulator

def grover_adaptive_search(
    converter: Any,
    transpiler: Any,
    lamb: float,
    max_no_improvement: int = 5,
    seed: int = 900
):

    ##########################################################
    #                   Initialization                       #
    ##########################################################

    random.seed(seed)
    n = converter.binary_model.num_bits
    k = 1 #Control the upper bound of the number of Grover iterations sampled at each step
    x_int = random.randint(0, 2**n - 1)
    x = [int(b) for b in format(x_int, f"0{n}b")]
    y = converter.instance.evaluate({i: x_i for i, x_i in enumerate(x)}).objective

    current_iter = 0
    no_improvement_count = 0

    print("[GAS] Initialization")
    print(f"[GAS] n={n}, lambda={lamb}")
    print(f"[GAS] Start state: x={x}, y={y}, k={k}")

    executor = transpiler.executor(backend=AerSimulator(seed_simulator=seed,max_parallel_threads=None))

    ############################################################
    #                     Main Loop                            #
    ############################################################

    while no_improvement_count < max_no_improvement:
        
        # Sample t uniformly in {0, ..., ceil(k)-1}; avoid empty range when k == 1
        num_iterations = random.randrange(max(1, int(np.ceil(k))))
        print(
            f"\n[GAS] Iteration {current_iter + 1} | current y={y}, k={k:.6f}, "
            f"Grover iters={num_iterations}"
        )

        ####################################################
        #       Call to the Quantum Grover Circuit         #
        ####################################################

        exec = converter.transpile(transpiler,
                                y=y,
                                num_iterations=num_iterations)

        #We run the circuit several time since NISQ hardware are noisy
        job = exec.sample(executor, shots=256)
        result = job.result()
        sample_set = converter.decode(result)

        #Extract x and y from the best sample
        x_vals = sample_set.best_feasible.extract_decision_variables("x")
        candidate_x = [
            int(round(x_vals.get((i,), x_vals.get(i, 0.0))))
            for i in range(n)
        ]
        candidate_y = float(sample_set.best_feasible.objective)

        print(f"[GAS] Candidate: x={candidate_x}, y={candidate_y}")

        if candidate_y < y:
            print("[GAS] Improvement found -> accepting candidate and resetting k to 1")
            x = candidate_x
            y = candidate_y
            k = 1
            no_improvement_count = 0
        else:
            old_k = k
            k = lamb * k
            no_improvement_count += 1
            print(
                f"[GAS] No improvement -> keeping current solution and scaling "
                f"k: {old_k:.6f} -> {k:.6f}"
            )

        current_iter += 1


    print(f"\n[GAS] Finished after {current_iter} iterations. Best found: x={x}, y={y}")

    return x, y

# %% [markdown]
# ## Running the full pipeline
#
# We call the hybrid classical-quantum algorithm using Grover Search.
#
# The parameter `lamb` handle the speed at which the sample space grow for the number of iterations.
# The parameters `max_iter` and `max_no_improvement` fix the stopping criterion

# %%
x,y = grover_adaptive_search(
    converter=converter,
    transpiler=transpiler,
    lamb=1.2,
    max_no_improvement=5,
)
selected = [i+1 for i, xi in enumerate(x) if xi == 1]
print(f"Selected assets: {selected}, objective value: {y}")

# %% [markdown]
# After $10$ iterations, the algorithm stop because the solution no further improve.
#
# It return the global minimum for the portfolio problem.
#
# The optimal solution is to buy assets : $1, 5$ and $8$.
