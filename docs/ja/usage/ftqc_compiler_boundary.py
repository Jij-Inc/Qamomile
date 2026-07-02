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
# tags: [usage, resource-estimation, circuit-compilation]
# ---
#
# # FTQCコンパイラ境界
#
# このページでは、fault-tolerantなアルゴリズム開発をQamomileのコンパイラスタックのどこに置くべきかを説明します。
# 新しいFTQCのアイデアをcircuit IR、resource-estimation workload、backend emitter機能、またはドキュメントのどこで扱うべきかを判断するための設計チェックリストです。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import math

import qamomile.circuit as qmc
import qamomile.observable as qm_o
import qamomile.resource_estimation as qre
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## 境界
#
# Qamomileには、別々だが接続された2つのFTQC surfaceがあります。
#
# | Layer | 表すべきもの | 外に置くべきもの |
# | --- | --- | --- |
# | Compiler IR | 量子プログラム。register、control flow、measurement、QPE/IQFTのような高レベルのcomposite gate | 論文固有の量子化学テーブル、factory schedule、report schema |
# | Resource workloads | アルゴリズム契約。Hamiltonian normalization、QPE precision budget、Trotter sample、block-encoding cost | backend固有の量子ビット配置や出力SDK instruction |
# | Physical lifts | code distance、cycle time、factory throughput、active volumeのような明示的なarchitecture仮定 | 隠れたhardware model選択や固定されたreport format |
#
# この分割により、IRを抽象的に保てます。
# コンパイラは、backendがloweringに必要な情報を持つまで高レベルの意味を保ちます。
# 一方でresource-estimation objectは、circuitを出力する価値があるかを判断する前に、論文規模の仮定を比較できます。

# %% [markdown]
# ## IRとしてのQPE
#
# QPEはFTQCの構成要素ですが、コンパイラには抽象的な量子プログラムとして入るべきです。
# counting register、controlled unitary、IQFT、fixed-point measurementはコンパイラの概念です。
# 必要なQPE iteration数を決めた量子化学固有のworkloadは、このIRの一部ではありません。

# %%


@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """QPEが推定するphaseを持つunitaryを適用する。"""
    return qmc.p(q, theta)


@qmc.qkernel
def phase_probe(theta: qmc.Float) -> qmc.Float:
    counting = qmc.qubit_array(3, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)

    phase = qmc.qpe(target, counting, phase_gate, theta=theta)
    return qmc.measure(phase)


def flatten_ops(ops):
    """nested control-flow bodyを含めてoperationを返す。"""
    flat = []
    for op in ops:
        flat.append(op)
        for nested in getattr(op, "nested_op_lists", lambda: [])():
            flat.extend(flatten_ops(nested))
    return flat


transpiler = QiskitTranspiler()
block = transpiler.to_block(phase_probe, parameters=["theta"])
block = transpiler.inline(block)
ops = flatten_ops(block.operations)
composites = [op for op in ops if isinstance(op, CompositeGateOperation)]

print(
    {
        "block_kind": block.kind.name,
        "composite_types": [op.gate_type for op in composites],
    }
)

assert block.kind.name == "AFFINE"
assert any(op.gate_type == CompositeGateType.IQFT for op in composites)

# %% [markdown]
# 上のテストはコンパイラ境界を直接確認しています。
# inliningによって通常の量子カーネル呼び出しは取り除かれますが、QPE内部のIQFTはcomposite operationのまま残ります。
# native supportを持つbackendはそのまま出力でき、別のbackendはemit時にdecompositionを使えます。
#
# これが望ましいFTQCコンパイラの形です。
# IRは「これはIQFTである」と表し、「あるbackend固有のprimitive gate列」を表しません。

# %% [markdown]
# ## Resource ContractとしてのAlgorithm Workload
#
# Resource estimationは別の問いから始まります。
# 近年の論文では、よりよいHamiltonian representation、小さいeffective normalization、active-volume schedulingの改善が主張されることがあります。
# そのような主張は、まずsymbolic resource quantityとして表すべきです。

# %%
hamiltonian = qre.summarize_pauli_hamiltonian(4 * qm_o.Z(0) + 2 * qm_o.X(0) * qm_o.X(1))

block_encoding = qre.BlockEncodingResource(
    system_qubits=hamiltonian.n_qubits,
    normalization=hamiltonian.lambda_norm,
    prepare_cost_toffoli=8,
    select_cost_toffoli=20,
    reflection_cost_toffoli=4,
    ancilla_qubits=1,
    name="toy block encoding",
)
workload = qre.HamiltonianQPEWorkload.from_block_encoding(
    hamiltonian,
    block_encoding,
    qpe_register_qubits=3,
)
logical = qre.estimate_qubitized_qpe_resources_from_workload(workload, precision=1)
values = qre.resource_values_from_estimate(logical)

for quantity in (
    qre.ResourceQuantity.LAMBDA_NORM,
    qre.ResourceQuantity.WALK_COST_TOFFOLI,
    qre.ResourceQuantity.QPE_ITERATIONS,
    qre.ResourceQuantity.NON_CLIFFORD_COUNT,
):
    spec = qre.describe_resource_quantity(quantity)
    print({"quantity": spec.quantity.value, "category": spec.category.value})

assert values["logical_qubits"] == 6
assert math.isclose(float(values["qpe_iterations"]), 6.0, rel_tol=0.0, abs_tol=1e-12)
assert math.isclose(
    float(values["non_clifford_count"]), 240.0, rel_tol=0.0, abs_tol=1e-12
)

# %% [markdown]
# これらの値はbackend instructionではありません。
# backend circuit representationを選ぶ前に比較できる、review可能な仮定です。
# そのためQamomileでは、`HamiltonianQPEWorkload`、`TrotterQPEWorkload`、`BlockEncodingResource`を`qamomile.circuit.ir`の外に置きます。

# %% [markdown]
# ## 明示的な仮定としてのPhysical Lift
#
# physical estimateは、どのarchitecture仮定を使ったかを示すべきです。
# 下の小さなsurface-code modelは、それらの仮定が見えるように意図的にsymbolicな形を残しています。

# %%
surface_code = qre.SurfaceCodeCostModel(
    code_distance=5,
    physical_cycle_time_seconds=1e-6,
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=2,
    physical_qubits_per_factory=1000,
    factory_cycles_per_non_clifford=4,
)
physical = qre.estimate_physical_resources(logical, surface_code)
physical_values = physical.resource_values()

for name in (
    "physical_qubits",
    "runtime_seconds",
    "depth_limited_runtime_seconds",
    "non_clifford_limited_runtime_seconds",
):
    print({name: physical_values[name]})

assert physical_values["code_distance"] == 5
assert physical_values["physical_qubits"] == 2300
assert float(physical_values["non_clifford_limited_runtime_seconds"]) > 0

# %% [markdown]
# physical liftはcompiler IRを変更しません。
# 名前付きmodelのもとでlogical estimateに価格を付けます。
# 将来のPRでより豊かなsurface-code modelやfactory modelを追加する場合は、report formatを追加する前に、その仮定をcanonical quantityとして公開すべきです。

# %% [markdown]
# ## Contributor Checklist
#
# FTQCアルゴリズムやresource-estimation機能を追加するときは、次のチェックリストを使います。
#
# - 機能が量子プログラムの意味を変える場合は、抽象的なIR operation、composite gate、frontend helperを追加または再利用する。
# - 機能が論文レベルのアルゴリズムコストを変える場合は、resource workloadとcanonical quantityを追加または再利用する。
# - 機能がhardware pricingを変える場合は、明示的なphysical lift modelを追加または再利用する。
# - 機能が結果のreview方法だけを変える場合は、基礎となるquantityが安定するまで、docsまたは後続のreport layerに置く。
#
# ## まとめ
#
# このnotebookでは、次のことを学びました。
#
# - QPEは抽象的な量子構造としてコンパイラに属する。
# - FTQC量子化学の主張は、まずresource quantityとworkload contractとして表すべきである。
# - Physical estimateでは、architecture仮定をreport schemaの背後に隠さず見える形に保つべきである。
