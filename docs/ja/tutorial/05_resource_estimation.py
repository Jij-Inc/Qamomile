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
# tags: [tutorial, resource-estimation]
# ---
#
# # リソース推定
#
# 量子カーネルを実機で実行する前に、必要なリソース（量子ビット数、ゲート数等）を把握しておきたい場合や、そもそも定義した量子カーネルを実行するために必要なリソースを知りたい場合があります。Qamomileの`estimate_resources()`は**量子カーネルを実行せずに**リソース推定が可能です。具体的な（パラメータ固定の）量子カーネルにも、シンボリック（パラメータ付き）な量子カーネルにも対応しています。
#
# この章では以下を扱います：
#
# - 固定量子カーネルの基本的なリソース推定
# - パラメータ付き量子カーネルのシンボリックなリソース推定
# - `ResourceEstimate`フィールドリファレンス
# - `.substitute()`によるスケーリング分析
# - FTQC向けの論理リソースと物理リソースproxyの比較

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import qamomile.circuit as qmc
import qamomile.observable as qm_o
import qamomile.resource_estimation as qre
import sympy as sp

# %% [markdown]
# ## 固定量子カーネルのリソース推定
#
# パラメータを持たない量子カーネルに対しては、`estimate_resources()`は具体的な数値を返します。


# %%
@qmc.qkernel
def fixed_circuit() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")

    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1], q[2] = qmc.cx(q[1], q[2])

    return qmc.measure(q)


# %%
fixed_circuit.draw()

# %%
est = fixed_circuit.estimate_resources()
print("qubits:", est.qubits)
assert est.qubits == 3
print("total gates:", est.gates.total)
assert est.gates.total == 3
print("single-qubit gates:", est.gates.single_qubit)
assert est.gates.single_qubit == 1
print("two-qubit gates:", est.gates.two_qubit)
assert est.gates.two_qubit == 2

# %% [markdown]
# ## シンボリックなリソース推定
#
# 量子カーネルに未バインドのパラメータ（例：`n: qmc.UInt`）がある場合、`estimate_resources()`は**SymPy式**を返します。特定の値を選ばなくてもコストのスケーリングが分かります。


# %%
@qmc.qkernel
def scalable_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    q = qmc.h(q)
    q = qmc.ry(q, theta)

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
scalable_circuit.draw(n=4, fold_loops=False)

# %%
est = scalable_circuit.estimate_resources()
print("qubits:", est.qubits)
assert str(est.qubits) == "n"
print("total gates:", est.gates.total)
assert str(est.gates.total) == "3*n - 1"
print("single-qubit gates:", est.gates.single_qubit)
assert str(est.gates.single_qubit) == "2*n"
print("two-qubit gates:", est.gates.two_qubit)
assert str(est.gates.two_qubit) == "n - 1"
print("rotation gates:", est.gates.rotation_gates)
assert str(est.gates.rotation_gates) == "n"
print("parameters:", est.parameters)
assert set(est.parameters.keys()) == {"n"}

# %% [markdown]
# 出力には、量子ビット数を表す`n`や総ゲート数を表す`3*n - 1`のようなSymPy式が含まれます。これらは近似ではなく厳密な値です。

# %% [markdown]
# ## `ResourceEstimate`フィールドリファレンス
#
# | フィールド | 説明 |
# |-------|------------|
# | `est.qubits` | 論理量子ビット数 |
# | `est.gates.total` | 総ゲート数 |
# | `est.gates.single_qubit` | 単一量子ビットゲート数 |
# | `est.gates.two_qubit` | 2量子ビットゲート数 |
# | `est.gates.multi_qubit` | 多量子ビットゲート数（3量子ビット以上） |
# | `est.gates.t_gates` | Tゲート数 |
# | `est.gates.clifford_gates` | Cliffordゲート数 |
# | `est.gates.rotation_gates` | 回転ゲート数 |
# | `est.gates.oracle_calls` | オラクル呼び出し回数（名前別の辞書） |
# | `est.parameters` | シンボル名からSymPyシンボルへの辞書 |
#
# すべてのフィールドはSymPy式です。固定量子カーネルの場合は通常の整数に評価されます。

# %% [markdown]
# ## `.substitute()`によるスケーリング分析
#
# シンボリック式は*数式*を示してくれますが、特定のサイズでの具体的な数値も確認したい場合`.substitute()`で評価できます：

# %%
for n_val in [4, 8, 16, 32]:
    c = est.substitute(n=n_val)
    print(
        f"n={n_val:2d}: {int(c.gates.total):>3} gates total, {int(c.gates.two_qubit):>2} two-qubit"
    )
    assert int(c.gates.total) == 3 * n_val - 1
    assert int(c.gates.two_qubit) == n_val - 1

# %% [markdown]
# ## FTQCのコスト要因を比較する
#
# Fault-tolerantアルゴリズムは、backend circuitへloweringする前に比較することがよくあります。Qamomileではこの層を分けて扱います。`qamomile.resource_estimation`を使うと、Hamiltonianの記述、アルゴリズムレベルの論理リソース推定、canonical quantityによる比較、architecture modelを通した物理リソースproxyへの変換を順に扱えます。
#
# この小さな例では、各候補をblock-encoding contractとして表します。このcontractには、Hamiltonian normalization、PREPARE/SELECT/reflectionのコスト、ancilla footprint、QPE readout registerのサイズ、任意のrepresentation errorを記録します。これらのquantityがあれば、backend circuitに固定せずにHamiltonian QPE workloadを作れます。
#
# :::{note}
# [symmetry-compressed double factorization](https://arxiv.org/abs/2403.03502)や[unitary weight concentration](https://arxiv.org/abs/2603.22778)のような近年の量子化学リソース推定では、Hamiltonian normalization、representation error、walk operatorのコスト、Toffoli数、論理量子ビット数、runtime、space-time volumeなどを通してアルゴリズムを比較します。このチュートリアルはこれらの論文の再現ではありません。そのような比較を組み立てるために必要なQamomileのresource quantityを示します。
# :::

# %%
hamiltonian = 4 * qm_o.Z(0) + 3 * qm_o.Z(1) + 2 * qm_o.X(0) * qm_o.X(1)
summary = qre.summarize_pauli_hamiltonian(hamiltonian)

baseline_block = qre.BlockEncodingResource(
    system_qubits=summary.n_qubits,
    normalization=summary.lambda_norm,
    prepare_cost_toffoli=20,
    select_cost_toffoli=70,
    reflection_cost_toffoli=10,
    ancilla_qubits=1,
    name="sparse Pauli LCU",
)
candidate_block = qre.BlockEncodingResource(
    system_qubits=summary.n_qubits,
    normalization=sp.Rational(2, 5) * summary.lambda_norm,
    prepare_cost_toffoli=15,
    select_cost_toffoli=45,
    reflection_cost_toffoli=5,
    ancilla_qubits=2,
    name="compressed factorization",
)

baseline_workload = qre.HamiltonianQPEWorkload.from_block_encoding(
    summary,
    baseline_block,
    representation=qre.HamiltonianRepresentation.SPARSE_PAULI_LCU,
    qpe_register_qubits=2,
    description="sparse Pauli LCU",
)
candidate_workload = qre.HamiltonianQPEWorkload.from_block_encoding(
    summary,
    candidate_block,
    representation=qre.HamiltonianRepresentation.SYMMETRY_COMPRESSED_DF,
    second_factor_rank=4,
    qpe_register_qubits=2,
    representation_error=sp.Rational(1, 10),
    description="compressed factorization",
)

baseline_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    baseline_workload,
    precision=1,
)
candidate_logical = qre.estimate_qubitized_qpe_resources_from_workload(
    candidate_workload,
    precision=1,
)

logical_rows = qre.compare_resource_values(
    baseline_logical,
    candidate_logical,
    quantities=(
        qre.ResourceQuantity.QPE_ITERATIONS,
        qre.ResourceQuantity.NON_CLIFFORD_COUNT,
        qre.ResourceQuantity.LOGICAL_QUBITS,
    ),
)
for row in logical_rows:
    print(row.to_dict())

assert (
    candidate_logical.gates.oracle_calls["qpe_iterations"]
    < baseline_logical.gates.oracle_calls["qpe_iterations"]
)
assert candidate_logical.gates.multi_qubit < baseline_logical.gates.multi_qubit
assert candidate_workload.qpe_register_qubits == 2
assert candidate_workload.algorithmic_precision(1) == sp.Rational(9, 10)

# %% [markdown]
# `compare_resource_values()`は論理`ResourceEstimate`オブジェクトを直接受け取れます。物理リソースproxyが必要な場合は、コンパクトなarchitecture modelを渡します。次の推定はhardware designではありません。同じsurface-code風の仮定のもとで候補を比較するための一貫した方法です。

# %%
surface_code = qre.SurfaceCodeCostModel(
    code_distance=5,
    physical_cycle_time_seconds=1e-6,
    physical_qubits_per_logical_factor=2,
    logical_cycle_factor=3,
    factory_count=1,
    physical_qubits_per_factory=1000,
    factory_cycles_per_non_clifford=4,
)

baseline_physical = qre.estimate_physical_resources(baseline_logical, surface_code)
candidate_physical = qre.estimate_physical_resources(candidate_logical, surface_code)

physical_rows = qre.compare_resource_values(
    baseline_physical,
    candidate_physical,
    quantities=(
        qre.ResourceQuantity.PHYSICAL_QUBITS,
        qre.ResourceQuantity.RUNTIME_SECONDS,
        qre.ResourceQuantity.PHYSICAL_QUBIT_SECONDS,
    ),
)
for row in physical_rows:
    print(row.to_dict())

assert candidate_physical.runtime_seconds < baseline_physical.runtime_seconds
assert (
    candidate_physical.resource_values()["physical_qubit_seconds"]
    < baseline_physical.resource_values()["physical_qubit_seconds"]
)

# %% [markdown]
# ## まとめ
#
# - `estimate_resources()`は実行せずに量子ビット数とゲートコストを算出します。
# - パラメータ付き量子カーネルでは、結果は厳密なスケーリングを示すSymPy式になります。
# - `.substitute(n=...)`で特定のサイズに代入し、実行可能性を確認できます。
# - `qamomile.resource_estimation`を使うと、FTQCアルゴリズム候補をcanonicalな論理リソースと物理リソースquantityで比較できます。
#
# **次へ**：[実行モデル](06_execution_models.ipynb) — `sample()`と`run()`、オブザーバブル、ビット順序について。
