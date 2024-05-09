# Qamomile
## What is Qamomile
Qamomile is a library that supports running quantum optimization algorithms with various quantum computation libraries.
Currently, Qamomile supports two quantum computation libraries, [Qiskit](https://www.ibm.com/quantum/qiskit) and [QURI-Part](https://quri-parts.qunasys.com/).

Qamomile stands for Quantum Algorithm for Mathematical OptiMization with jIjmodeLing Extension. It transforms mathematical models written by [JijModeling](https://www.documentation.jijzept.com/docs/jijmodeling) into Ising Hamiltonians and various other encoded Hamiltonians such as Quantum Random Access Optimization.

## Installation
The installation for qiskit is 
```bash
# jijmodeling-transpiler-quantum for qiskit
pip install "jijmodeling-transpiler-quantum[qiskit]"
```

The installation for QURI Parts is
```bash
# jijmodeling-transpiler-quantum for quri-parts
pip install "jijmodeling-transpiler-quantum[quri-parts]"
```

## Quickstart
In the following example, QAOA for the graph colouring problem is implemented using Qamomile.
```python
import jijmodeling as jm
import jijmodeling_transpiler.core as jtc
import jijmodeling_transpiler_quantum.qiskit as jt_qk

from qiskit.primitives import Estimator, Sampler
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import COBYLA

import networkx as nx

# Create Mathematical Model
# define variables
V = jm.Placeholder("V")
E = jm.Placeholder("E", ndim=2)
N = jm.Placeholder("N")
x = jm.BinaryVar("x", shape=(V, N))
n = jm.Element("i", belong_to=(0, N))
v = jm.Element("v", belong_to=(0, V))
e = jm.Element("e", belong_to=E)
# set problem
problem = jm.Problem("Graph Coloring")
# set one-hot constraint that each vertex has only one color

problem += jm.Constraint("one-color", x[v, :].sum() == 1, forall=v)
# set objective function: minimize edges whose vertices connected by edges are the same color
problem += jm.sum([n, e], x[e[0], n] * x[e[1], n])

# Create Problem Instance
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4])
G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (2, 4)])
inst_E = [list(edge) for edge in G.edges]
color_num = 3
num_nodes = G.number_of_nodes()
instance_data = {"V": num_nodes, "N": color_num, "E": inst_E}
num_qubits = num_nodes * color_num

# Transpile mathematical model to Qiskit Ising Hamiltonian
compiled_instance = jtc.compile_model(problem, instance_data)
qaoa_builder = jt_qk.qaoa.transpile_to_qaoa_ansatz(compiled_instance,normalize=False,relax_method=jtc.pubo.RelaxationMethod.SquaredPenalty)
hamiltonian, _ = qaoa_builder.get_hamiltonian(multipliers={"one-color": 1})

# Run QAOA by Qiskit
sampler = Sampler()
optimizer = COBYLA()
qaoa = QAOA(sampler, optimizer, reps=1)
result = qaoa.compute_minimum_eigenvalue(hamiltonian)

# Analyze Result
sampleset = qaoa_builder.decode_from_quasi_dist(result.eigenstate)
sampleset.feasible()
```

## Community
Join our [discord channel](https://discord.gg/Km5dKF9JjG)!