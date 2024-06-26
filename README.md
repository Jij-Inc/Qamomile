# Qamomile

`Qamomile` is a transpiler from model written in [JijModeling]() to quantum optimization algorithms on variaous quantum platform.

- [Qiskit](#qiskit)
- [QURI-Parts](#quri-parts) 

Documentation: [https://jij-inc.github.io/Qamomile/](https://jij-inc.github.io/Qamomile/)

```mermaid
graph LR
    JijModeling --> Instance
    subgraph JijModelingTranspiler
        Instance
    end
    
    subgraph Qiskit
        QSKT[Hamiltonian]
        QSKT --> QSKTCIRC[Q Circuit] --> QSKTR[Results]
    end
    subgraph Quri-Parts
        QP[Hamiltonian]
        QP --> QPCIRC[Q Circuit] --> QURIR[Results]
    end
 
    

    subgraph Qamomile
        Instance --> Encode
        Encode{Encode}
        Encode --> QH[Quantum Hamiltonian]
    end

    QH --> QSKT
    QH --> QP
    
    subgraph Qamomile
        QSKTR --> Decode{Decode}
        QURIR --> Decode
    end
    Decode --> Solutions
```

## Qiskit

[Qiskit](https://qiskit.org/) is an open-source SDK for working with quantum computers at the level of circuits, algorithms, and application modules.

### Installation

```bash
pip install "qamomile[qiskit]"
```

### Quantum Approximate Optimization Algorithm (QAOA)

```python
import jijmodeling as jm
import jijmodeling_transpiler as jmt
import qamomile as qamo

# Create model
problem = jm.Problem("model")
...  # Modeling ...

# Compile
compiled_instance = jmt.compile_model(problem, instance_data, fixed_vars)

# Transpile to QAOA of qikit
qaoa_builder = qamo.qiskit.transpile_to_qaoa(compiled_instance)
```


## QURI-Parts

[QURI Parts](https://quri-parts.qunasys.com/) is an open source library suite for creating and executing quantum algorithms on various quantum computers and simulators.

```bash
pip install "qamomile[quri-parts]"
```

### Quantum Approximate Optimization Algorithm (QAOA)

```python
import jijmodeling as jm
import jijmodeling_transpiler as jmt
import qamomile as qamo

# Create model
problem = jm.Problem("model")
...  # Modeling ...

# Compile
compiled_instance = jmt.compile_model(problem, instance_data, fixed_vars)

# Transpile to QAOA of qikit
qaoa_builder = qamo.quri.transpile_to_qaoa(compiled_instance)
```


## Contributing

### Setup

```bash
pip install poetry
poetry install --all-extras
poetry shell
```

### Test

```bash
pytest tests
```



