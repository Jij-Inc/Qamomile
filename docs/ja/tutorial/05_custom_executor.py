# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # カスタムExecutorの実装：クラウドバックエンド連携
#
# このチュートリアルでは、Qamomileでカスタム量子Executorを実装する方法を学びます。
# Executorをカスタマイズすることで、IBM QuantumやAWS Braketなどの
# クラウド量子デバイスに回路を実行できます。
#
# ## このチュートリアルで学ぶこと
# - QuantumExecutorの役割と構造
# - 最小限のカスタムExecutorの作り方
# - IBM Quantumクラウドへの接続方法
# - パラメータバインディングの実装
# - 期待値計算（estimate）の実装

# %%
import qamomile.circuit as qm
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## 1. QuantumExecutorとは
#
# **QuantumExecutor**は、Qamomileのコンパイル済み回路を実際の量子バックエンドで
# 実行するためのインターフェースです。
#
# Qamomileのパイプラインは以下のようになっています：
#
# ```
# @qm.qkernel (Python関数)
#     ↓ transpile()
# ExecutableProgram (コンパイル済みプログラム)
#     ↓ sample() / run()
# QuantumExecutor が回路を実行
#     ↓
# 結果 (ビット列カウント)
# ```
#
# 標準の`QiskitTranspiler`は`AerSimulator`を使用しますが、
# カスタムExecutorを作ることで任意のバックエンドを使用できます。

# %% [markdown]
# ## 2. QuantumExecutorの基本構造
#
# `QuantumExecutor`は3つのメソッドを持つ抽象基底クラスです：
#
# | メソッド | 必須/オプション | 説明 |
# |---------|---------------|------|
# | `execute()` | **必須** | 回路を実行してビット列カウントを返す |
# | `bind_parameters()` | オプション | パラメトリック回路のパラメータをバインド |
# | `estimate()` | オプション | オブザーバブルの期待値を計算 |
#
# 最も重要なのは`execute()`メソッドです。これだけ実装すれば動作します。

# %% [markdown]
# ### execute()メソッドの仕様
#
# ```python
# def execute(self, circuit: T, shots: int) -> dict[str, int]:
#     """
#     Args:
#         circuit: バックエンド固有の量子回路
#         shots: 測定回数
#
#     Returns:
#         ビット列からカウントへの辞書
#         例: {"00": 512, "11": 512}
#     """
# ```
#
# **重要**: 戻り値のビット列はbig-endianフォーマットです。
# - "011" は qubit[2]=0, qubit[1]=1, qubit[0]=1 を意味します
# - 左端が最も大きいインデックスの量子ビット

# %% [markdown]
# ## 3. 最小限のカスタムExecutorを作る
#
# まずは`execute()`のみを実装した最小限のExecutorを作りましょう。

# %%
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qiskit import QuantumCircuit


class MySimpleExecutor(QuantumExecutor[QuantumCircuit]):
    """最小限のカスタムExecutor

    AerSimulatorを使ったシンプルな実装です。
    """

    def __init__(self):
        """AerSimulatorバックエンドで初期化"""
        from qiskit_aer import AerSimulator

        self.backend = AerSimulator()

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """回路を実行してビット列カウントを返す

        Args:
            circuit: Qiskit QuantumCircuit
            shots: 測定回数

        Returns:
            ビット列カウントの辞書 (例: {"00": 512, "11": 512})
        """
        from qiskit import transpile

        # 測定がなければ追加
        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        # バックエンド向けにトランスパイル
        transpiled = transpile(circuit, self.backend)

        # 実行
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()


# %% [markdown]
# ### カスタムExecutorのテスト
#
# 作成したExecutorを使って、Bell状態を生成してみましょう。

# %%
@qm.qkernel
def bell_state() -> tuple[qm.Bit, qm.Bit]:
    """Bell状態を生成"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    return qm.measure(q0), qm.measure(q1)


# トランスパイル
transpiler = QiskitTranspiler()
executable = transpiler.transpile(bell_state)

# カスタムExecutorで実行
my_executor = MySimpleExecutor()
job = executable.sample(my_executor, shots=1000)
result = job.result()

print("=== カスタムExecutorでBell状態を生成 ===")
for value, count in result.results:
    print(f"  {value}: {count}回")

# %% [markdown]
# 正しくBell状態（|00⟩と|11⟩がほぼ同数）が生成されていることを確認できます。

# %% [markdown]
# ## 4. IBM Quantumクラウド連携
#
# 次に、IBM Quantum Platformのクラウドバックエンドに接続するExecutorを作ります。
#
# ### 事前準備
#
# 1. [IBM Quantum](https://quantum.ibm.com/)でアカウントを作成
# 2. APIトークンを取得
# 3. `qiskit-ibm-runtime`をインストール
#
# ```bash
# pip install qiskit-ibm-runtime
# ```
#
# 4. APIトークンを設定
#
# ```python
# from qiskit_ibm_runtime import QiskitRuntimeService
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
# ```

# %%
# IBM Quantum Executor の実装例
# 注意: 実際に実行するにはIBM Quantumのアカウントが必要です


class IBMQuantumExecutor(QuantumExecutor[QuantumCircuit]):
    """IBM Quantum Platform用のカスタムExecutor

    実際のIBM量子デバイスまたはシミュレータで回路を実行します。
    """

    def __init__(
        self,
        backend_name: str = "ibm_brisbane",
        channel: str = "ibm_quantum",
    ):
        """IBM Quantum Serviceに接続

        Args:
            backend_name: 使用するバックエンド名
                - "ibm_brisbane": 127量子ビットデバイス
                - "ibm_sherbrooke": 127量子ビットデバイス
                - "ibmq_qasm_simulator": クラウドシミュレータ
            channel: チャンネル ("ibm_quantum" または "ibm_cloud")
        """
        # 実際に使用する場合はコメントを外してください
        # from qiskit_ibm_runtime import QiskitRuntimeService
        # self.service = QiskitRuntimeService(channel=channel)
        # self.backend_name = backend_name
        self.service = None
        self.backend_name = backend_name
        self._channel = channel

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """IBM Quantumで回路を実行

        SamplerV2 Primitiveを使用して実行します。
        """
        if self.service is None:
            raise RuntimeError(
                "IBM Quantum Service が設定されていません。\n"
                "QiskitRuntimeService.save_account() でトークンを設定してください。"
            )

        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # バックエンドを取得
        backend = self.service.backend(self.backend_name)

        # バックエンド向けにトランスパイル（最適化レベル1）
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        transpiled = pm.run(circuit)

        # SamplerV2で実行
        sampler = Sampler(backend)
        job = sampler.run([transpiled], shots=shots)
        result = job.result()

        # 結果をdict[str, int]形式に変換
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        return counts


# %% [markdown]
# ### 使用例（シミュレーション）
#
# IBM Quantumのアカウントがない場合でも、以下のようにローカルでテストできます：

# %%
# ローカルシミュレータで動作確認
print("=== IBM Quantum Executor（ローカルモック）===")
print("注意: 実際のIBM Quantumへの接続にはアカウント設定が必要です")
print()

# ローカルAerSimulatorで代用してテスト
local_executor = MySimpleExecutor()
job = executable.sample(local_executor, shots=1000)
result = job.result()
print("ローカルシミュレーション結果:")
for value, count in result.results:
    print(f"  {value}: {count}回")

# %% [markdown]
# ## 5. パラメータバインディングの実装
#
# QAOA等のパラメトリック回路を使う場合、`bind_parameters()`メソッドの実装が必要です。
#
# ### ParameterMetadataの構造
#
# `bind_parameters()`は3つの引数を受け取ります：
#
# ```python
# def bind_parameters(
#     self,
#     circuit: T,                          # パラメータ化された回路
#     bindings: dict[str, Any],            # パラメータ名→値のマッピング
#     parameter_metadata: ParameterMetadata # パラメータのメタデータ
# ) -> T:
# ```
#
# `bindings`は以下のような形式です：
# ```python
# {
#     "gammas[0]": 0.1,
#     "gammas[1]": 0.2,
#     "betas[0]": 0.3,
#     "betas[1]": 0.4
# }
# ```

# %% [markdown]
# ### ParameterMetadataのヘルパーメソッド
#
# `ParameterMetadata`には便利なヘルパーメソッドがあります：
#
# - `to_binding_dict(bindings)`: Qiskit形式のバインディング辞書に変換
# - `get_ordered_params()`: パラメータを順序付きリストで取得（QURI Parts向け）

# %%
from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
from typing import Any


class MyParametricExecutor(QuantumExecutor[QuantumCircuit]):
    """パラメータバインディング対応のExecutor"""

    def __init__(self):
        from qiskit_aer import AerSimulator

        self.backend = AerSimulator()

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        from qiskit import transpile

        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        transpiled = transpile(circuit, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()

    def bind_parameters(
        self,
        circuit: QuantumCircuit,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> QuantumCircuit:
        """パラメータをバインド

        ParameterMetadata.to_binding_dict()を使うと、
        バックエンド固有のパラメータオブジェクトへのマッピングを
        簡単に作成できます。
        """
        # ヘルパーメソッドでQiskit形式に変換
        qiskit_bindings = parameter_metadata.to_binding_dict(bindings)
        return circuit.assign_parameters(qiskit_bindings)


# %% [markdown]
# ### パラメトリック回路のテスト

# %%
@qm.qkernel
def parametric_circuit(theta: qm.Float) -> qm.Bit:
    """パラメータ化された回路"""
    q = qm.qubit(name="q")
    q = qm.h(q)
    q = qm.rz(q, theta)
    q = qm.h(q)
    return qm.measure(q)


# パラメータを保持してトランスパイル
executable_param = transpiler.transpile(parametric_circuit, parameters=["theta"])

# パラメトリックExecutorで実行
param_executor = MyParametricExecutor()

print("=== パラメトリック回路のテスト ===")
print()

for theta_val in [0.0, 1.57, 3.14]:  # 0, π/2, π
    job = executable_param.sample(
        param_executor, shots=1000, bindings={"theta": theta_val}
    )
    result = job.result()
    print(f"theta = {theta_val:.2f}:")
    for value, count in result.results:
        print(f"  {value}: {count}回")
    print()

# %% [markdown]
# ## 6. 期待値計算（estimate）の実装
#
# QAOAなどの変分アルゴリズムでは、ハミルトニアンの期待値を計算する必要があります。
# `estimate()`メソッドを実装することで、最適化ループで使用できます。
#
# ### estimate()メソッドの仕様
#
# ```python
# def estimate(
#     self,
#     circuit: T,              # 状態準備回路
#     observable: Observable,  # 測定するオブザーバブル
#     params: Sequence[float] | None = None  # パラメータ値
# ) -> float:
#     """期待値 <ψ|H|ψ> を計算"""
# ```

# %%
from qamomile.circuit.observable import Observable
from typing import Sequence


class MyFullExecutor(QuantumExecutor[QuantumCircuit]):
    """全機能を実装したカスタムExecutor

    - execute(): 回路実行
    - bind_parameters(): パラメータバインディング
    - estimate(): 期待値計算
    """

    def __init__(self):
        from qiskit_aer import AerSimulator

        self.backend = AerSimulator()
        self._estimator = None

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        from qiskit import transpile

        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        transpiled = transpile(circuit, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()

    def bind_parameters(
        self,
        circuit: QuantumCircuit,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> QuantumCircuit:
        qiskit_bindings = parameter_metadata.to_binding_dict(bindings)
        return circuit.assign_parameters(qiskit_bindings)

    def estimate(
        self,
        circuit: QuantumCircuit,
        observable: Observable,
        params: Sequence[float] | None = None,
    ) -> float:
        """オブザーバブルの期待値を計算

        Qiskit Estimatorプリミティブを使用します。
        """
        from qiskit.primitives import Estimator
        from qamomile.qiskit.observable import to_sparse_pauli_op

        if self._estimator is None:
            self._estimator = Estimator()

        # ObservableをQiskit形式に変換
        sparse_pauli_op = to_sparse_pauli_op(observable.hamiltonian)

        # パラメータ値を設定
        param_values = list(params) if params is not None else []

        # 期待値を計算
        job = self._estimator.run([(circuit, sparse_pauli_op, param_values)])
        result = job.result()

        return float(result[0].data.evs)


# %% [markdown]
# ## 7. まとめ
#
# このチュートリアルでは、カスタムQuantumExecutorの実装方法を学びました。
#
# ### 実装の3段階
#
# | レベル | 実装するメソッド | 用途 |
# |-------|----------------|------|
# | **基本** | `execute()` | 単純な回路実行 |
# | **中級** | + `bind_parameters()` | パラメトリック回路 |
# | **上級** | + `estimate()` | 変分アルゴリズム（QAOA等） |
#
# ### 実装のポイント
#
# 1. **execute()**: ビット列カウント`dict[str, int]`を返す（big-endianフォーマット）
# 2. **bind_parameters()**: `ParameterMetadata.to_binding_dict()`を活用
# 3. **estimate()**: バックエンドのEstimatorプリミティブを使用
#
# ### コード例のまとめ
#
# ```python
# from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
# from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
#
# class MyExecutor(QuantumExecutor[QuantumCircuit]):
#     def __init__(self):
#         self.backend = ...  # バックエンドを初期化
#
#     def execute(self, circuit, shots):
#         # 回路を実行してビット列カウントを返す
#         ...
#         return {"00": 512, "11": 512}
#
#     def bind_parameters(self, circuit, bindings, metadata):
#         # metadata.to_binding_dict()でバックエンド形式に変換
#         return circuit.assign_parameters(metadata.to_binding_dict(bindings))
#
#     def estimate(self, circuit, observable, params):
#         # Estimatorプリミティブで期待値を計算
#         ...
#         return expectation_value
# ```
#
# ### 次のステップ
#
# - **qaoa.py**: QAOAアルゴリズムでカスタムExecutorを活用
# - **qpe.py**: 量子位相推定アルゴリズム
