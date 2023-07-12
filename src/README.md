# JijModelingTranspiler for Qiskit

Transpiler to Qiskit operator and circuit from JijModeling.

Currently support only cost function transpiling that mean every constraint automatically convert to penalty and added to cost function.

If you want to handle constraint by alternative ansatz technique you have to set zero weight for penalty and write alternative ansatz by your hand.

## Getting started

```python
import jijmodeling as jm
import jijmodeling.transpiler as jmt
import jijtranspiler_qiskit as jtq

d = jm.Placeholder("d")
n = d.shape[0]
x = jm.Binary("x", shape=n)
i = jm.Element("i", n)

problem = jm.Problem("sample")
problem = jm.Sum(i, d[i]*x[i])
problem += jm.Constraint("onehot", x[:] == 1)

# Compile (Substitute) data to the expression
instance_data = {"d": [1, 2, 3]}
compiled_instance = jmt.core.compile_model(problem, instance_data)

# Transpile to Ising Operators for QAOA
qaoa_builder = jtq.qaoa.transpile_to_qaoa_ansatz(compiled_instance)
ising_hamiltonian, constant = qaoa_builder.get_hamiltonian()

# You can access qubit index by `.var_map`
var_map: dict[str, tuple[int, ...]] = qaoa_builder.var_map
var_map["x"][(0,)]
# -> 0
# When you create original mixer, we expect that `var_map` helps corrensponding classical variable and each qubits..
 
# Write QAOA by your hand
# ...

# Decoding
# transpiler can decode the reuslts from qiskit.
# EX. QuasiProb obj <- can get from qiskit.premitive.Sampler
#     counts obj <- can get from qiskit.Terra
sampleset = qaoa_builder.decode_from_quasi_prob(quasi_prob)
sampleset = qaoa_builder.decode_from_counts(counts)


# Transpile to QRAC Hamiltonian for QRAO
# Also transpiler support QRAO
qrao_builder = jtq.qrao.transpile_to_qrac31_hamiltonian(compiled_instance)
qrac_hamiltonian, constant = qrao_builder.get_hamiltonian()

# Write QRAO by your hand
# ...

# Decoding
# You have to rouding method by your hand.
# binary_results: list[list[int]] is a rounded result of QRAO resulits.
sampleset = qrao_builder.decode_from_binary_list(binary_results)

```

## Test

The test is written at tests/ and docstring.

```
python -m pytest tests jijtranspiler_qiskit
```

