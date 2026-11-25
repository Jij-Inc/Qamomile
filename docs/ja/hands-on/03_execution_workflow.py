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
# # 実行の流れ
#
# このノートブックでは、小さなVQE（変分量子固有値ソルバー）の例を使って、qkernelを対応する量子SDKや中間表現（IR）に変換し、その実行先で動かす手順を学びます。以下では、実行先にQiskitを使います。
#
# 対象のハミルトニアン$H$に対し、VQEはパラメータ付き量子回路$\ket{\psi(\theta)}$を繰り返し実行し、その結果を使ってパラメータを更新します。ハミルトニアンのエネルギー（期待値）$\bra{\psi(\theta)}H\ket{\psi(\theta)}$が低くなるパラメータ$\theta$を探します。[Peruzzoら（2014年）](https://doi.org/10.1038/ncomms5213)を参照してください。
#
# このノートブックでは、qkernelから期待値を求め、ビット列を測定する方法に焦点を当てます。パラメータの最適化手法の詳細には立ち入りません。

# %%
# Qamomileとこのノートブックで使うパッケージをインストールします。
# # !pip install "qamomile[qiskit,visualization]" matplotlib scipy

# %%
from __future__ import annotations

import math

import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

import qamomile as qm
from qamomile.qiskit import bind_circuit, load_circuit

# %% [markdown]
# ## 1. 実行先を選ぶ
#
# 実行するtargetに合わせてqkernelをコンパイルします。この例ではQiskit Aerシミュレータを実行先に選び、結果を再現できるよう、実行時に乱数シードを指定します。

# %%
endpoint = qm.qiskit.Aer()


# %% [markdown]
# ## 2. パラメータ付きqkernelを定義する
#
# まずパラメータ付きqkernelを定義します。$n$量子ビットの状態を準備し、各量子ビットに$R_Y(\theta_i)$を適用します。
#
# $$
# \ket{\psi_n(\theta)}
# = \bigotimes_{i} R_Y(\theta_i)\ket{0}
# = \bigotimes_{i}\left( \cos\frac{\theta_i}{2}\ket{0} +\sin\frac{\theta_i}{2}\ket{1} \right)
# $$
#
# 次に、この量子状態を測定するqkernelを定義します。測定には`measure()`を使います。


# %%
@qm.qkernel
def parameterized_state(
    n: qm.Const[qm.Int],
    thetas: qm.Param[qm.Vector[qm.Real, n]],  # noqa: F821
) -> qm.Q[qm.Bits[n]]:  # noqa: F821
    """Prepare n qubits with a separate rotation angle for each qubit without measuring them."""
    q = qm.qubits(n)
    for i in range(n):
        q = qm.ry(q, thetas[i], target=i)
    return q


@qm.qkernel
def sample_state(
    n: qm.Const[qm.Int],
    thetas: qm.Param[qm.Vector[qm.Real, n]],  # noqa: F821
) -> qm.Shot[qm.Bits[n]]:  # noqa: F821
    """Measure every qubit of the variational state in the computational basis."""
    q = parameterized_state(n=n, thetas=thetas)
    return qm.measure(q)


# %% [markdown]
# `sample_state`をQiskitで実行できる形式にコンパイルします。実行するqkernelは古典データを入力として受け取り、古典データを返します。
#
# ここでは$n = 1$とします。
#
# $$
# \ket{\psi_1} = \cos\frac{\theta}{2}\ket{0} +\sin\frac{\theta}{2}\ket{1}
# $$
#
# コンパイル前に`qm.specialize`で量子ビット数`n`を固定します。`thetas`には`qm.Param`を付けて実行時パラメータとして残します。これにより、実行のたびに値を変更できます。

# %%
sample_program = qm.specialize(sample_state, n=1)
sample_executable = qm.compile(sample_program, to=endpoint.target)
qiskit_circuit = load_circuit(sample_executable)
print(f"Qubits: {qiskit_circuit.num_qubits}")
print(
    "Runtime parameters:", [slot.name for slot in sample_executable.parameter_abi.slots]
)
qiskit_circuit.draw()

# %% [markdown]
# コンパイルしたQiskitの量子回路を実行します。`qm.run()`を使って量子ビットを測定し、ビット列を得ます。実行時に`parameters`を通して`thetas`を渡します。まず初期角度を$\pi / 2$にして実行します。
#
# `qm.run()`は実行記録を返し、その`.result`に測定結果が含まれます。

# %%
shots = 1024
initial_thetas = [math.pi / 2]

initial_parameters = {"thetas": initial_thetas}
sample_run = qm.run(
    sample_executable,
    on=endpoint,
    shots=shots,
    seed=42,
    parameters=initial_parameters,
)

print(type(sample_run))

# %%
initial_sample = sample_run.result
initial_probabilities = {
    bits: count / initial_sample.shots for bits, count in initial_sample.counts.items()
}

print("Initial counts:", dict(initial_sample.counts))
print("Initial probabilities:", initial_probabilities)
assert sum(initial_sample.counts.values()) == shots

# %% [markdown]
# `qm.run()`の実行結果に含まれる`counts`は、ビット列をその出現回数に対応付けます。各出現回数をショット数で割ると、測定結果から求めた確率が得られます。

# %% [markdown]
# ## 3. エネルギーを評価する
#
# 次に、定義した量子状態のパラメータを更新し、与えられたハミルトニアンの期待値を下げます。ローカルのシミュレータで正確な期待値を計算するため、コンパイル済み回路から末尾の測定を取り除いたコピーを使います。

# %%
state_circuit = qiskit_circuit.remove_final_measurements(inplace=False)

# %% [markdown]
# ハミルトニアンはパウリ演算子の和です。状態ベクトルから計算するため、Qiskitの`SparsePauliOp`でハミルトニアンを表します。

# %%
hamiltonian = SparsePauliOp.from_list([("Z", -1.0), ("X", -1.0)])
print(f"Hamiltonian: {hamiltonian}")

# %% [markdown]
# 状態準備回路に同じ実行時パラメータを割り当て、シミュレータで求めた状態ベクトルを使ってハミルトニアンの期待値を計算します。


# %%
def run_state(theta_values):
    """Evaluate the Hamiltonian on the shared variational state."""
    invocation = qm.Invocation.create(parameters={"thetas": tuple(theta_values)})
    circuit = bind_circuit(state_circuit, sample_executable, invocation)
    return float(Statevector(circuit).expectation_value(hamiltonian).real)


# %%
initial_energy = run_state(initial_thetas)
print(f"Initial energy: {initial_energy:+.6f}")


# %% [markdown]
# ## 4. コンパイル済み回路のパラメータを更新する
#
# パラメータを変えながら`run_state`を繰り返し呼び出し、期待値を下げます。評価のたびに、同じコンパイル済み回路に新しいパラメータを割り当てます。パラメータの更新にはSciPyの`minimize`を使います。

# %%
history = []


def energy(theta_values):
    """Bind one trial angle and return the energy to the optimizer."""
    value = run_state(theta_values)
    history.append((theta_values, value))
    return value


optimized = minimize(
    energy,
    x0=initial_thetas,
    method="COBYLA",
    options={"maxiter": 50, "rhobeg": 0.5, "tol": 1e-6},
)
optimized_parameters = {"thetas": tuple(map(float, optimized.x))}

print(f"Optimized thetas: {optimized_parameters['thetas']}")
print(f"Energy: {initial_energy:+.6f} -> {optimized.fun:+.6f}")
print("Energy evaluations:", optimized.nfev)
print("Optimizer status:", optimized.message)
assert math.isclose(optimized.fun, -math.sqrt(2), abs_tol=1e-6)

# %% [markdown]
# この回路では$\langle Z\rangle=\cos\theta$と$\langle X\rangle=\sin\theta$なので、解析的に次の結果が得られます。
#
# $$
# \bra{\psi_1(\theta)} H \ket{\psi_1(\theta)}
# =-\cos\theta-\sin\theta,\qquad \theta_\star=\pi/4\pmod{2\pi},\qquad E_\star=-\sqrt2
# $$
#
# この結果から、`minimize`が最適なパラメータを求められたことがわかります。

# %% [markdown]
# ## 5. 最適化の過程と結果を比較する
#
# 最後に、最適化の途中でエネルギーがどう変化するかを確認し、最適化したパラメータで得られる確率分布を、初期値で得られる分布と比較します。`sample_state`はコンパイル済みなので、同じオブジェクトを使って`qm.run()`を呼び出せます。

# %%
optimized_sample = qm.run(
    sample_executable,
    on=endpoint,
    shots=shots,
    seed=42,
    parameters=optimized_parameters,
).result
optimized_probabilities = {
    bits: count / optimized_sample.shots
    for bits, count in optimized_sample.counts.items()
}
print("Optimized counts:", dict(optimized_sample.counts))
print("Optimized probabilities:", optimized_probabilities)

# %% [markdown]
# 得られたパラメータ$\theta_* = \pi / 4$を使うと、解析的に次の状態が得られます。
#
# $$
# \ket{\psi_1(\pi / 4)} = \cos\frac{\pi}{8}\ket{0} +\sin\frac{\pi}{8}\ket{1}
# $$
#
# 測定結果は、この状態から期待される結果と整合しています。
#
# これらの結果を以下に描画します。

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(
    range(1, len(history) + 1),
    [value for _, value in history],
    marker=".",
    color="#2673B8",
    label="Evaluated energy",
)
axes[0].axhline(-math.sqrt(2), color="#555555", linestyle="--", label="Exact minimum")
axes[0].set(xlabel="Energy evaluation", ylabel="Energy", title="Variational loop")
axes[0].legend()

axes[1].bar(
    [-0.18, 0.82],
    [initial_probabilities.get((bit,), 0.0) for bit in (0, 1)],
    width=0.36,
    label="Before",
    color="#2673B8",
)
axes[1].bar(
    [0.18, 1.18],
    [optimized_probabilities.get((bit,), 0.0) for bit in (0, 1)],
    width=0.36,
    label="After",
    color="#E58B38",
)
axes[1].set(
    xticks=[0, 1],
    xticklabels=["0", "1"],
    ylim=(0, 1),
    xlabel="Measurement outcome",
    ylabel="Measured probability",
    title=f"Before and after ({shots} shots each)",
)
axes[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## まとめ
#
# 小さなVQEの例を通して、量子回路のコンパイルと実行を学びました。`Const`と`qm.specialize`で回路の構造を固定し、`Param`で角度を実行時に変更できるようにしました。また、Qiskitの状態ベクトルから正確な期待値を計算し、`qm.measure()`と`qm.run()`で量子ビットを測定しました。
