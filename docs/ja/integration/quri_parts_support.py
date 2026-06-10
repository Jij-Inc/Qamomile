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
# tags: [integration, optimization, variational]
# ---
#
# # QURI Parts サポート
#
# このページでは、具体的な最適化問題を通して、Qamomile の [QURI Parts](https://quri-parts.qunasys.com/) バックエンドを紹介します。
# 小さな MaxCut インスタンスを `BinaryModel.from_ising` で Ising 問題として表し、QAOA アンザッツを `@qkernel` として直接書きます。
# そのカーネルを `QuriPartsTranspiler` / `QuriPartsExecutor` で実行する流れを見ていきましょう。
# `QuriPartsExecutor` は、既定で高速な C++ 製状態ベクトルシミュレータ [Qulacs](https://docs.qulacs.org/) を使うため、以下の例は追加設定なしでローカル CPU 上で実行できます。

# %%
# 最新の Qamomile を QURI Parts オプション付きで pip からインストールします。
# # !pip install "qamomile[quri_parts]"

# %%
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.optimization.binary_model import BinaryModel
from qamomile.quri_parts import QuriPartsExecutor, QuriPartsTranspiler
from qamomile.quri_parts.observable import hamiltonian_to_quri_operator

# %% [markdown]
# ## MaxCut 問題
#
# QURI Parts 連携の説明に集中するため、[MaxCut に対する QAOA チュートリアル](../algorithm/qaoa_maxcut.ipynb)と同じ 5 ノードの小さなグラフを使います。
# $\sum_{(i,j) \in E}(1 - s_i s_j)/2$ を最大化することは、定数項を除けば、反強磁性 Ising ハミルトニアン $H_C = \sum_{(i,j) \in E} s_i s_j$ を*最小化*することと同じです。
# 重みなし MaxCut ではすべての $J_{ij} = 1$、$h_i = 0$ なので、これらの係数をそのまま `BinaryModel.from_ising` に渡します。
# ここで作るモデルは、QAOA カーネルに渡す `quad` / `linear` 辞書と、測定結果をスピン値 $(+1 / -1)$ に戻すヘルパーを持つ問題コンテナとして使います。

# %%
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
num_nodes = G.number_of_nodes()

ising_quad: dict[tuple[int, int], float] = {
    tuple(sorted((i, j))): 1.0 for i, j in G.edges()
}
ising_linear: dict[int, float] = {}
spin_model = BinaryModel.from_ising(linear=ising_linear, quad=ising_quad)
# 問題の構造はグラフから一意に決まります。重みなし MaxCut では、quad 項は辺と
# 1 対 1 に対応し、linear 項は存在しません。`BinaryModel.from_ising` が将来
# 壊れた場合に docs テストで検知できるよう、ここで assert で確認しておきます。
assert len(spin_model.quad) == G.number_of_edges()
assert len(spin_model.linear) == 0

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(5, 4))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"MaxCut graph: {num_nodes} nodes, {G.number_of_edges()} edges")
plt.show()

# %% [markdown]
# ## `@qkernel` による QAOA アンザッツの構築
#
# QAOA アンザッツを小さな `@qkernel` として直接書きます。
# レシピは[MaxCut に対する QAOA チュートリアル](../algorithm/qaoa_maxcut.ipynb)と同じです。一様重ね合わせから始め、$p$ 回のコスト層とミキサー層を適用し、最後に計算基底で測定します。
#
# :::{tip}
# Qamomile の回転ゲートは $e^{-i\theta/2}$ という規約に従います。
# そのため、$1/2$ 係数の扱いはコスト層とミキサー層で少し異なります。
# ミキサー層では `rx` に $2\beta$ を渡すので、$1/2$ が打ち消され、教科書通りの $e^{-i\beta X}$ になります。
# 一方、コスト層では `rzz` に $J_{ij} \cdot \gamma$ を渡すため、$1/2$ は残ります。
# この係数の違いは変分パラメータ $\gamma$ に吸収しています。つまり、ここで使う $\gamma$ は教科書の QAOA の $\gamma$ の 2 倍に相当します。
# :::


# %%
@qmc.qkernel
def superposition(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return q


@qmc.qkernel
def cost_layer(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    for (i, j), Jij in quad.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=Jij * gamma)
    for i, hi in linear.items():
        q[i] = qmc.rz(q[i], angle=hi * gamma)
    return q


@qmc.qkernel
def mixer_layer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=2.0 * beta)
    return q


@qmc.qkernel
def qaoa_ansatz(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = superposition(n)
    for layer in qmc.range(p):
        q = cost_layer(quad, linear, q, gammas[layer])
        q = mixer_layer(q, betas[layer])
    return qmc.measure(q)


# %% [markdown]
# `qaoa_ansatz.draw(...)` で Qamomile の回路図を描画できます。
# 問題の構造を決める引数 (`p`、`quad`、`linear`、`n`) には具体値を渡し、層の形が見えるようにします。
# 一方、`gammas` / `betas` はシンボリックなパラメータとして残します。

# %%
p = 3  # QAOA 層数
qaoa_ansatz.draw(
    p=p,
    quad=spin_model.quad,
    linear=spin_model.linear,
    n=num_nodes,
)

# %% [markdown]
# ## QURI Parts へのトランスパイル
#
# `QuriPartsTranspiler` は、他のバックエンドと同じように `transpile()` で使えます。
# 問題の構造を決める引数は `bindings` で固定し、`gammas` / `betas` はランタイムパラメータとして残します。

# %%
transpiler = QuriPartsTranspiler()
# `seed` を渡すと Qulacs sampler が再現可能になります。同じ seed と回路で `sample(...)` を 2 回呼ぶと、まったく同じショットカウントが得られます。
# 非決定的なサンプリングにしたい場合は、この引数を省略する（または `seed=None` を渡す）だけです。
executor = QuriPartsExecutor(seed=42)

executable = transpiler.transpile(
    qaoa_ansatz,
    bindings={
        "p": p,
        "quad": spin_model.quad,
        "linear": spin_model.linear,
        "n": num_nodes,
    },
    parameters=["gammas", "betas"],
)

# %% [markdown]
# `executable.get_first_circuit()` で内部の QURI Parts 回路を取り出せます。
# 取り出した回路は QURI Parts の `LinearMappedParametricQuantumCircuit` であり、$2p$ 個の QAOA 角度 (`gammas[0..p-1]`、`betas[0..p-1]`) が名前付きランタイムパラメータとして残っています。
# `type(...)` とパラメータ数で確認し、さらに QURI Parts 組み込みの `draw_circuit` で回路そのものを描画してみましょう。

# %%
from quri_parts.circuit.utils.circuit_drawer import (  # type: ignore[import-not-found]
    draw_circuit,
)

quri_circuit = executable.get_first_circuit()
assert (
    quri_circuit is not None
)  # transpile() はここで必ず 1 つの量子セグメントを生成する
# `qubit_count` と `parameter_count` は問題設定から一意に決まります。
# 量子ビット数はグラフのノード数と一致し、ランタイムパラメータ数は層ごとに
# (gamma | beta) の組が 1 つずつ、合計 2p になります。QuriParts の emit
# パスに回帰が起きた場合に docs テストで検知できるよう assert します。
assert quri_circuit.qubit_count == num_nodes
assert quri_circuit.parameter_count == 2 * p
print(type(quri_circuit).__name__)
print("qubit_count    :", quri_circuit.qubit_count)
print("parameter_count:", quri_circuit.parameter_count)

draw_circuit(quri_circuit, line_length=200)

# %% [markdown]
# 各ランタイムパラメータは、実行時まで未バインドのまま残ります。
# そのため、`gammas` / `betas` のバインドは QURI Parts 側での回路の作り直しではなく、パラメータ値の更新として扱われます。
# Ising 係数、量子ビット数、層数といった問題構造はコンパイル時に固定され、ランタイム入力として残るのは変分角度だけです。

# %% [markdown]
# ## `QuriPartsExecutor` による QAOA サンプリング
#
# `executable.sample(executor, bindings=..., shots=...)` は `SampleJob` を返します。
# `.result()` で得られる `SampleResult` は、`BinaryModel.decode_from_sampleresult` でスピン変数 $(+1 / -1)$ の `BinarySampleSet` にデコードできます。
# これにより、追加の変換なしでカット辺を数えられます。
# `QuriPartsExecutor()` は、デフォルトでは Qulacs の状態ベクトルシミュレータ上で動作します。

# %%
rng = np.random.default_rng(42)
init_params = rng.uniform(-np.pi / 2, np.pi / 2, 2 * p)
init_gammas = list(init_params[:p])
init_betas = list(init_params[p:])
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2000
maxiter = 20 if docs_test_mode else 100

sample_result = executable.sample(
    executor,
    bindings={"gammas": init_gammas, "betas": init_betas},
    shots=sample_shots,
).result()

decoded = spin_model.decode_from_sampleresult(sample_result)
print(f"Mean energy at random init: {decoded.energy_mean():+.4f}")

# %% [markdown]
# ## QAOA パラメータの最適化
#
# 同じ `executable` を異なる `(gammas, betas)` で繰り返し呼び出すのが、QAOA の最適化ループの基本形です。
# `transpiler.transpile()` を 1 回呼び、その後は `executable.sample()` を何度も呼び出します。
# この例では、サンプリングとデコードの処理を `cost_fn()` として定義し、SciPy の `minimize` 関数で最適化します。
# 古典オプティマイザは `(gammas, betas)` を更新しながら、サンプリングされた Ising エネルギーの平均を下げていきます。
# 各反復では、同じ `executable` と `QuriPartsExecutor` を再利用します。

# %%
cost_history: list[float] = []


def cost_fn(params: np.ndarray) -> float:
    result = executable.sample(
        executor,
        bindings={"gammas": list(params[:p]), "betas": list(params[p:])},
        shots=sample_shots,
    ).result()
    energy = spin_model.decode_from_sampleresult(result).energy_mean()
    cost_history.append(energy)
    return energy


res = minimize(cost_fn, init_params, method="COBYLA", options={"maxiter": maxiter})

opt_gammas = list(res.x[:p])
opt_betas = list(res.x[p:])
print(f"Optimized mean energy: {res.fun:+.4f}")
print(f"Optimal gammas       : {[round(float(v), 4) for v in opt_gammas]}")
print(f"Optimal betas        : {[round(float(v), 4) for v in opt_betas]}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Cost (mean Ising energy)")
plt.title("QAOA optimization on Qulacs via `QuriPartsExecutor`")
plt.tight_layout()
plt.show()

# %% [markdown]
# 上の `QuriPartsExecutor` は `seed=42` で構築しているため Qulacs sampler は再現可能で、このページを再実行しても同じ最適化の軌跡と最終エネルギーが得られます。元の非決定的な挙動に戻すには `seed` 引数を外してください。
# この 5 ノードグラフ上の $H_C$ の基底状態エネルギー付近までは収束するはずです。
# ここで得た最適パラメータ (`opt_gammas`、`opt_betas`) を、以降の例でも使います。

# %% [markdown]
# ## 期待値計算: 未バインド回路とバインド済み回路の違い
#
# `QuriPartsExecutor.estimate_expectation(circuit, hamiltonian, param_values)` は、QURI Parts で期待値を計算するためのメソッドです。
# 渡された**回路の状態**に応じて、内部で QURI Parts の 2 種類の estimator を使い分けます。
#
# - **未バインドのパラメトリック回路**: `transpile()` が生成した直後の回路は、パラメータをまだ自由変数として保持しています。
#   QURI Parts の `apply_circuit` はこれを `ParametricCircuitQuantumState` でラップし、executor は QURI Parts の**パラメトリックな estimator** を呼び出します。
#   この経路では、評価時に `param_values` の値でパラメータが束縛されます。
# - **バインド済みの回路、もしくは最初からパラメータを持たない回路**: 例えば `circuit.bind_parameters([...])` を呼んでパラメータを具体的な数値に固定すると、同じ `apply_circuit` は今度は `GeneralCircuitQuantumState` を返します。
#   executor はこの場合、QURI Parts の**非パラメトリックな estimator** を呼び出し、引数の `param_values` は使われません。
#
# この違いを知っておくと、計算コストを見積もりやすくなります。
# 同じ回路を異なるパラメータで何度も評価する最適化ループでは、毎回の回路コピーを省けるパラメトリックな estimator が向いています。
# 逆に、パラメータがすでに具体的な数値に固定されている場合は、パラメトリック回路用の処理が不要な非パラメトリック estimator のほうが効率的です。
#
# QURI Parts は回路レベルでは `measure` を何もしない命令として扱います。
# そのため、`transpiler.transpile(qaoa_ansatz, ...)` が出力するパラメトリック回路は、そのまま QAOA の出力状態 $|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle$ を準備する回路として使えます。
# これをコストハミルトニアンと一緒に `estimate_expectation` に渡せば、サンプリングノイズを含まない $\langle H_C \rangle$ を計算できます。
# QAOA の最適化でも、同じ回路を保ったまま `executable.sample()` とデコードの組み合わせを `executor.estimate(circuit, hamiltonian, params=...)` に置き換えられます。
#
# 2 つの経路を直接試すため、まず Qamomile の `Hamiltonian` として $H_C = \sum_{(i,j) \in E} Z_i Z_j$ を組み立てます。
# それを QURI Parts の演算子に変換し、各回路に対して `estimate_expectation` を呼び出します。

# %%
cost_hamiltonian = qm_o.Hamiltonian()
for (i, j), Jij in spin_model.quad.items():
    cost_hamiltonian.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Z, i), qm_o.PauliOperator(qm_o.Pauli.Z, j)),
        Jij,
    )
for i, hi in spin_model.linear.items():
    cost_hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, i),), hi)

quri_H = hamiltonian_to_quri_operator(cost_hamiltonian)

# transpiler.transpile() 直後の、まだバインドされていないパラメトリック回路
unbound_circuit = executable.get_first_circuit()
assert unbound_circuit is not None
print(f"unbound type           : {type(unbound_circuit).__name__}")
print(f"unbound parameter_count: {unbound_circuit.parameter_count}")

# QURI Parts はランタイムパラメータを「回路に登録された順序のフラットなリスト」
# として要求します。登録順は回路を出力したときの初出順で決まるため、QAOA では
# gammas[0], betas[0], gammas[1], betas[1], ... と層ごとに交互の順になります。
# 「全 gammas のあとに全 betas」という順序ではない点に注意してください。
# 順序を推測しなくて済むよう、executable から登録順を読み取り、
# 名前で値を引いてフラットなリストに整えます。
named_values = {f"gammas[{i}]": opt_gammas[i] for i in range(p)}
named_values.update({f"betas[{i}]": opt_betas[i] for i in range(p)})
flat_params = [named_values[name] for name in executable.parameter_names]
# ランタイムパラメータは 2p 個の QAOA 角度のみ。QuriPartsTranspiler の
# パラメータ登録方法が将来変わった場合に検知できるよう assert します。
assert len(executable.parameter_names) == 2 * p
assert len(flat_params) == 2 * p
print(f"circuit parameter order: {executable.parameter_names}")

# QURI Parts 標準のバインド処理で、同じ数値を手動で束縛します。
bound_circuit = unbound_circuit.bind_parameters(flat_params)
print(f"bound   type           : {type(bound_circuit).__name__}")

# 経路 1: 未バインド回路 → パラメトリック estimator。param_values の値が使われます。
energy_unbound = executor.estimate_expectation(unbound_circuit, quri_H, flat_params)

# 経路 2: バインド済み回路 → 非パラメトリック estimator。param_values は無視されます。
energy_bound = executor.estimate_expectation(bound_circuit, quri_H, [])

print(f"parametric  estimator: {energy_unbound:+.10f}")
print(f"non-param.  estimator: {energy_bound:+.10f}")
assert np.isclose(energy_unbound, energy_bound, atol=1e-10)

# %% [markdown]
# 両方の経路は数値精度の範囲で一致します。
# 同じ QAOA 状態を、同じ Ising コストハミルトニアンに対して評価しているためです。
# また、最適化後パラメータでのこのノイズなし期待値は、先ほど出力した標本平均エネルギーともショットノイズの範囲で一致するはずです。
# この経路の切り替えは Qamomile の `executor.estimate()` インターフェース内部に隠れているので、通常は意識する必要はありません。
# `estimate_expectation` を直接呼び出すのは、QURI Parts の回路を自分で扱う場合に限られます。
#
# `executor.estimate(circuit, hamiltonian, params=...)` は、これより 1 段抽象度の高いメソッドです。
# `qamomile.observable.Hamiltonian` を直接受け取り、内部で自動変換してから `estimate_expectation` に処理を委ねます。

# %%
energy_via_estimate = executor.estimate(
    unbound_circuit, cost_hamiltonian, params=flat_params
)
print(f"executor.estimate     : {energy_via_estimate:+.10f}")
assert np.isclose(energy_via_estimate, energy_unbound, atol=1e-10)

# %% [markdown]
# ## sampler と estimator の差し替え
#
# `QuriPartsExecutor()` は、初回利用時に既定の Qulacs 状態ベクトル sampler とパラメトリック estimator を遅延生成します。
# 別の QURI Parts バックエンドに差し替えたい場合は、`QuriPartsTranspiler.executor(sampler=..., estimator=...)` 経由で sampler や estimator を渡すか、`QuriPartsExecutor(sampler=..., estimator=...)` を直接インスタンス化します。
# 差し替えた executor は、上で使った `executor` の位置にそのまま当てはめられます。
# sampler を変えても、カーネルをトランスパイルし直す必要はありません。
# executable が回路を持ち、executor がシミュレーション基盤を持つ、という役割分担になっているためです。
#
# 具体例として、QURI Parts の Qulacs 用 `NoiseSimulator` を使ったノイズ込み sampler を構築します。
# そして、**同じ**最適化済みパラメータに対して、ノイズなし版とノイズあり版の標本平均エネルギーを比較します。
# すべてのゲートに脱分極ノイズがかかれば、ノイズあり側の平均エネルギーはノイズなしの値からずれるはずです。
# これにより、差し替えた sampler が実際に使われていることを確認できます。

# %%
from quri_parts.circuit.noise import (  # type: ignore[import-not-found]
    DepolarizingNoise,
    NoiseModel,
)
from quri_parts.qulacs.sampler import (  # type: ignore[import-not-found]
    create_qulacs_noisesimulator_sampler,
)

noise_model = NoiseModel([DepolarizingNoise(error_prob=0.02)])
noisy_sampler = create_qulacs_noisesimulator_sampler(noise_model)
noisy_executor = transpiler.executor(sampler=noisy_sampler)

clean_result = executable.sample(
    executor,
    bindings={"gammas": opt_gammas, "betas": opt_betas},
    shots=sample_shots,
).result()
noisy_result = executable.sample(
    noisy_executor,
    bindings={"gammas": opt_gammas, "betas": opt_betas},
    shots=sample_shots,
).result()

clean_decoded = spin_model.decode_from_sampleresult(clean_result)
noisy_decoded = spin_model.decode_from_sampleresult(noisy_result)
clean_energy = clean_decoded.energy_mean()
noisy_energy = noisy_decoded.energy_mean()
print(f"noiseless sampler mean energy: {clean_energy:+.4f}")
print(f"noisy     sampler mean energy: {noisy_energy:+.4f}")

# %% [markdown]
# どちらのsampler実行もshot countを返すので、サンプルされたエネルギー分布を直接比較できます。
# 各subplotの縦線は、標本平均エネルギーを表します。

# %%


def energy_distribution(decoded_samples):
    counts: Counter[float] = Counter()
    for energy, occ in zip(decoded_samples.energy, decoded_samples.num_occurrences):
        counts[energy] += occ
    energies = sorted(counts.keys())
    return energies, [counts[energy] for energy in energies]


fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, decoded_samples, mean_energy, title, color in [
    (axes[0], clean_decoded, clean_energy, "Noiseless sampler", "#2696EB"),
    (
        axes[1],
        noisy_decoded,
        noisy_energy,
        "NoiseSimulator sampler",
        "#FF8A3D",
    ),
]:
    energies, counts = energy_distribution(decoded_samples)
    ax.bar(energies, counts, width=0.6, color=color)
    ax.axvline(
        mean_energy,
        color="#2B2B2B",
        linestyle="--",
        linewidth=1.5,
        label=f"mean = {mean_energy:+.3f}",
    )
    ax.set_xticks(energies)
    ax.set_title(title)
    ax.set_xlabel("Ising energy")
    ax.legend()

axes[0].set_ylabel("Frequency")
fig.suptitle("Sampled energy distributions by QURI Parts sampler")
fig.tight_layout()
plt.show()

# %% [markdown]
# 脱分極ノイズは QAOA 状態を最大混合状態へ近づけます。
# その極限ではすべてのスピン配置が等確率になり、$H_C$ の平均値は $0$ に収束します。
# そのため、ノイズあり側の平均エネルギーは、ノイズなしよりも 0 に近い、つまり高い値になります。
# `error_prob` を大きくしたり、ノイズチャネルを追加したりすれば、ノイズあり側のエネルギーはさらに $0$ へ近づきます。
# 逆に `error_prob=0.0` にすれば、ショットノイズの範囲でノイズなしの値に戻ります。
# リモートデバイス、密度行列シミュレータ、確率的な状態ベクトル sampler など、他の QURI Parts sampler に差し替える場合も同じやり方で動作します。
# どの場合でも、カーネルの再トランスパイルは不要です。

# %% [markdown]
# ## まとめ
#
# - `QuriPartsTranspiler().transpile(kernel, bindings=..., parameters=[...])` はカーネルを QURI Parts の `LinearMappedParametricQuantumCircuit` に変換し、QURI Parts ネイティブの `draw_circuit` でそのまま確認できます。
# - `QuriPartsExecutor` は、既定の Qulacs 状態ベクトルシミュレータ上で、QAOA 形式のサンプリングを行う `executable.sample()` と、ノイズなしの期待値計算を行う `executor.estimate(...)` の両方をサポートします。
# - `estimate_expectation` は、渡された回路にフリーパラメータが残っているかどうかに応じて、QURI Parts のパラメトリック estimator と非パラメトリック estimator を切り替えます。通常は `executor.estimate()` を使えば、この切り替えを意識せずに済みます。
# - QURI Parts の `NoiseSimulator` ベースの sampler など、独自の sampler / estimator は `transpiler.executor(...)` 経由で差し替えられます。カーネルをトランスパイルし直す必要はありません。

# %% [markdown]
# ### 関連項目
#
# - [CUDA-Qサポート](cudaq_support.ipynb)では、同じMaxCut QAOAの流れをCUDA-Qバックエンドで扱います。
