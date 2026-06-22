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
# tags: [algorithm, primitive, resource-estimation]
# ---
#
# # FTQCのbuilding blockとしての量子位相推定
#
# 量子位相推定(QPE)は、多くのfault-tolerantアルゴリズムの背後にあるcontrol-flow patternです。固有状態を準備し、unitaryの制御付き冪を適用し、逆QFTで位相をdecodeします。このnotebookでは、最小のQamomile実装を示し、測定された位相を確認し、同じprecision選択をsymbolicなリソース推定へつなげます。

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import math

import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.estimator.algorithmic import estimate_qpe
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## 背景
#
# $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$のとき、QPEは$\phi$の2進桁を推定します。下の例では1量子ビットのphase gate $P(\theta)|1\rangle = e^{i\theta}|1\rangle$を使います。$\theta=\pi/2$にすると$\phi=\theta/(2\pi)=1/4$になり、3つの小数bitで正確に表せます。
#
# target量子ビットはphase gateの固有状態である$|1\rangle$に準備します。counting registerは`QFixed`値として測定するため、Qamomileは測定bit列を`Float`へdecodeします。

# %%
theta = math.pi / 2
expected_phase = theta / (2 * math.pi)
counting_qubits = 3

assert expected_phase == 0.25

# %% [markdown]
# ## 実装
#
# `qmc.qpe`はtarget固有状態、counting register、Qamomileのunitary量子カーネルを受け取ります。制御付き冪はstandard library helperの中で生成されます。Qamomileは逆QFTをcomposite operationとして保つため、backendはそれをnativeにemitするか、後でdecomposeできます。

# %%
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    q = qmc.p(q, theta)
    return q


@qmc.qkernel
def estimate_phase(theta: qmc.Float) -> qmc.Float:
    counting = qmc.qubit_array(counting_qubits, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)

    phase = qmc.qpe(target, counting, phase_gate, theta=theta)
    return qmc.measure(phase)

# %%
block = estimate_phase.build(theta=theta)
composite_types = [
    op.gate_type
    for op in block.operations
    if isinstance(op, CompositeGateOperation)
]

print([gate_type.value for gate_type in composite_types])
assert CompositeGateType.IQFT in composite_types

# %% [markdown]
# ## 結果
#
# この位相は3量子ビットのcounting registerで正確に表せるため、simulatorは1つのdecode済み値を返します。

# %%
transpiler = QiskitTranspiler()
executable = transpiler.transpile(estimate_phase, bindings={"theta": theta})
result = executable.sample(transpiler.executor(), shots=256).result()

print(result.results)
assert len(result.results) == 1
measured_phase, count = result.results[0]
assert measured_phase == sp.Float(expected_phase)
assert count == 256

# %% [markdown]
# ## リソース推定
#
# 回路実行では小さなexact exampleを確認できます。FTQC planningでより重要なのは、system sizeとprecisionに対してcostがどうscaleするかです。`estimate_qpe`はその関係を、backend circuit decompositionにcommitせずsymbolicに記録します。

# %%
resource_estimate = estimate_qpe(
    n_system=1,
    precision=counting_qubits,
    hamiltonian_norm=1,
    method="qubitization",
)

print("qubits:", resource_estimate.qubits)
print("total gates:", resource_estimate.gates.total)

assert resource_estimate.qubits == 4
assert sp.simplify(resource_estimate.gates.total - 8) == 0

# %% [markdown]
# ## Summary
#
# このnotebookでは、次のことを行いました。
#
# - `qmc.qpe`で最小のQPE量子カーネルを構築し、`QFixed` phaseを測定しました。
# - backend emission前のIRで、逆QFTが抽象的なcomposite operationとして残ることを確認しました。
# - 同じprecision parameterをsymbolicなQPEリソース推定へ接続しました。
