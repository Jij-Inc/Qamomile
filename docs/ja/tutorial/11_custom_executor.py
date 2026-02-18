# ---
# jupyter:
#   jupytext:
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
# このチュートリアルでは、QamomileでカスタムのQuantum Executorを実装する方法を解説します。
# Executorをカスタマイズすることで、IBM QuantumやAWS Braketなどの
# クラウド量子デバイス上で回路を実行できるようになります。
#
# ## このチュートリアルで学ぶこと
# - QuantumExecutorの役割と構造
# - 最小限のカスタムExecutorの作成方法
# - IBM Quantumクラウドへの接続方法
# - パラメータバインディングの実装方法
# - 期待値計算（estimate）の実装方法

# %%
from qiskit import QuantumCircuit

import qamomile.circuit as qmc

# %% [markdown]
# ## 1. QuantumExecutorとは
#
# **QuantumExecutor**は、Qamomileでトランスパイルされた回路を
# 実際の量子バックエンド上で実行するためのインターフェースです。
#
# Qamomileのパイプラインは以下のようになっています：
#
# ```
# @qmc.qkernel (Python関数)
#     ↓ transpile()
# ExecutableProgram (トランスパイル済みプログラム)
#     ↓ sample() / run()
# QuantumExecutor が回路を実行
#     ↓
# Results (ビット列のカウント)
# ```
#
# 標準の `QiskitTranspiler` は `AerSimulator` を使用しますが、
# カスタムExecutorを作成することで任意のバックエンドを利用できます。
# %% [markdown]
# ## 2. QuantumExecutorの基本構造
#
# `QuantumExecutor` は3つのメソッドを持つ抽象基底クラスです：
#
# | メソッド | 必須/任意 | 説明 |
# |---------|----------|------|
# | `execute()` | **必須** | 回路を実行してビット列のカウントを返す |
# | `bind_parameters()` | 任意 | パラメトリック回路にパラメータをバインドする |
# | `estimate()` | 任意 | オブザーバブルの期待値を計算する |
#
# 最も重要なメソッドは `execute()` です。このメソッドだけ実装すれば十分です。
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
# **重要**: 返されるビット列はビッグエンディアン形式です。
# - "011" は qubit[2]=0, qubit[1]=1, qubit[0]=1 を意味します
# - 最も左のビットが最も大きいインデックスの量子ビットです
#
# これは[01_introduction](01_introduction.ipynb)で説明されている
# 量子ビットの順序規約と一致しています。
# %% [markdown]
# ## 3. 最小限のカスタムExecutorの作成
#
# `execute()` のみを実装した最小限のExecutorを作成してみましょう。
# %%
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qamomile.qiskit import QiskitTranspiler


class MySimpleExecutor(QuantumExecutor[QuantumCircuit]):
    """Minimal custom Executor

    A simple implementation using AerSimulator.
    """

    def __init__(self):
        """Initialize with AerSimulator backend"""
        from qiskit_aer import AerSimulator

        self.backend = AerSimulator()

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts

        Args:
            circuit: Qiskit QuantumCircuit
            shots: Number of measurements

        Returns:
            Dictionary of bitstring counts (e.g., {"00": 512, "11": 512})
        """
        from qiskit import transpile

        # Add measurements if none exist
        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        # Transpile for backend
        transpiled = transpile(circuit, self.backend)

        # Execute
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()


# %% [markdown]
# ### カスタムExecutorのテスト
#
# 作成したExecutorを使ってベル状態を生成してみましょう。


# %%
@qmc.qkernel
def bell_state() -> tuple[qmc.Bit, qmc.Bit]:
    """Generate Bell state"""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)


bell_state.draw()

# %%
# Transpile
transpiler = QiskitTranspiler()
executable = transpiler.transpile(bell_state)

# Execute with custom Executor
my_executor = MySimpleExecutor()
job = executable.sample(my_executor, shots=1000)
result = job.result()

print("=== Bell State Generated with Custom Executor ===")
for value, count in result.results:
    print(f"  {value}: {count} times")

# %% [markdown]
# 適切なベル状態（|00⟩ と |11⟩ がほぼ同数）が生成されていることを確認できます。

# %% [markdown]
# ## 4. IBM Quantumクラウドとの連携
#
# 次に、IBM Quantum Platformのクラウドバックエンドに接続するExecutorを作成します。
#
# ### 前提条件
#
# 1. [IBM Quantum](https://quantum.ibm.com/)でアカウントを作成する
# 2. APIトークンを取得する
# 3. `qiskit-ibm-runtime` をインストールする
#
# ```bash
# pip install qiskit-ibm-runtime
# ```
#
# 4. APIトークンを設定する
#
# ```python
# from qiskit_ibm_runtime import QiskitRuntimeService
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
# ```

# %%
# IBM Quantum Executor implementation example
# Note: Requires an IBM Quantum account to actually execute


class IBMQuantumExecutor(QuantumExecutor[QuantumCircuit]):
    """Custom Executor for IBM Quantum Platform

    Executes circuits on actual IBM quantum devices or simulators.
    """

    def __init__(
        self,
        backend_name: str = "ibm_brisbane",
        channel: str = "ibm_quantum",
    ):
        """Connect to IBM Quantum Service

        Args:
            backend_name: Backend name to use
                - "ibm_brisbane": 127-qubit device
                - "ibm_sherbrooke": 127-qubit device
                - "ibmq_qasm_simulator": Cloud simulator
            channel: Channel ("ibm_quantum" or "ibm_cloud")
        """
        from qiskit_ibm_runtime import QiskitRuntimeService

        self.service = QiskitRuntimeService(channel=channel)
        self.backend_name = backend_name

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """Execute circuit on IBM Quantum

        Uses the SamplerV2 Primitive for execution.
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        # Get backend
        backend = self.service.backend(self.backend_name)

        # Transpile for backend (optimization level 1)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        transpiled = pm.run(circuit)

        # Execute with SamplerV2
        sampler = Sampler(backend)
        job = sampler.run([transpiled], shots=shots)
        result = job.result()

        # Convert result to dict[str, int] format
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        return counts


# %% [markdown]
# ### IBM Quantum上での実行
#
# IBM Quantumの認証情報が設定されている場合、Executorはクラウドに接続して
# 実際のハードウェア上で回路を実行します。認証情報が利用できない場合は、
# ローカルシミュレータにフォールバックします。

# %%
try:
    ibm_executor = IBMQuantumExecutor(backend_name="ibm_brisbane")
    job = executable.sample(ibm_executor, shots=1000)
    result = job.result()
    print("=== IBM Quantum Results ===")
    for value, count in result.results:
        print(f"  {value}: {count} times")
except Exception as e:
    print(f"IBM Quantum not available: {e}")
    print()
    print("To use IBM Quantum, configure your credentials:")
    print("  from qiskit_ibm_runtime import QiskitRuntimeService")
    print(
        '  QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")'
    )
    print()
    print("Running on local simulator instead:")
    local_executor = MySimpleExecutor()
    job = executable.sample(local_executor, shots=1000)
    result = job.result()
    for value, count in result.results:
        print(f"  {value}: {count} times")

# %% [markdown]
# ## 5. パラメータバインディングの実装
#
# QAOAのようなパラメトリック回路では、`bind_parameters()` メソッドの実装が必要です。
#
# ### ParameterMetadataの構造
#
# `bind_parameters()` は3つの引数を受け取ります：
#
# ```python
# def bind_parameters(
#     self,
#     circuit: T,                          # パラメータ付き回路
#     bindings: dict[str, Any],            # パラメータ名 → 値のマッピング
#     parameter_metadata: ParameterMetadata # パラメータのメタデータ
# ) -> T:
# ```
#
# `bindings` は以下のような形式です：
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
# `ParameterMetadata` は便利なヘルパーメソッドを提供しています：
#
# - `to_binding_dict(bindings)`: Qiskit形式のバインディング辞書に変換
# - `get_ordered_params()`: パラメータを順序付きリストとして取得（QURI Parts用）

# %%
from typing import Any

from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata


class MyParametricExecutor(QuantumExecutor[QuantumCircuit]):
    """Executor with parameter binding support"""

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
        """Bind parameters

        Using ParameterMetadata.to_binding_dict(), you can
        easily create mappings to backend-specific parameter objects.
        """
        # Convert to Qiskit format using helper method
        qiskit_bindings = parameter_metadata.to_binding_dict(bindings)
        return circuit.assign_parameters(qiskit_bindings)


# %% [markdown]
# ### パラメトリック回路のテスト


# %%
@qmc.qkernel
def parametric_circuit(theta: qmc.Float) -> qmc.Bit:
    """Parameterized circuit"""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    q = qmc.rz(q, theta)
    q = qmc.h(q)
    return qmc.measure(q)


parametric_circuit.draw()

# %%
# Transpile while preserving parameters
executable_param = transpiler.transpile(parametric_circuit, parameters=["theta"])

# Execute with parametric Executor
param_executor = MyParametricExecutor()

print("=== Parametric Circuit Test ===")
print()

for theta_val in [0.0, 1.57, 3.14]:  # 0, π/2, π
    job = executable_param.sample(
        param_executor, shots=1000, bindings={"theta": theta_val}
    )
    result = job.result()
    print(f"theta = {theta_val:.2f}:")
    for value, count in result.results:
        print(f"  {value}: {count} times")
    print()

# %% [markdown]
# ## 6. 期待値計算（estimate）の実装
#
# QAOAのような変分アルゴリズムでは、ハミルトニアンの期待値計算が必要です。
# `estimate()` メソッドを実装することで、最適化ループで使用できるようになります。
#
# ### estimate()メソッドの仕様
#
# ```python
# def estimate(
#     self,
#     circuit: T,              # 状態準備回路
#     hamiltonian: qm_o.Hamiltonian,  # 測定するハミルトニアン
#     params: Sequence[float] | None = None  # パラメータ値
# ) -> float:
#     """期待値 <ψ|H|ψ> を計算する"""
# ```

# %%
from typing import Sequence

import qamomile.observable as qm_o


class MyFullExecutor(QuantumExecutor[QuantumCircuit]):
    """Custom Executor with full functionality

    - execute(): Circuit execution
    - bind_parameters(): Parameter binding
    - estimate(): Expectation value calculation
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
        hamiltonian: qm_o.Hamiltonian,
        params: Sequence[float] | None = None,
    ) -> float:
        """Calculate Hamiltonian expectation value

        Uses the Qiskit StatevectorEstimator primitive.
        """
        from qiskit.primitives import StatevectorEstimator

        from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op

        if self._estimator is None:
            self._estimator = StatevectorEstimator()

        # Convert Hamiltonian to Qiskit format
        sparse_pauli_op = hamiltonian_to_sparse_pauli_op(hamiltonian)

        # Calculate expectation value
        job = self._estimator.run([(circuit, sparse_pauli_op)])
        result = job.result()

        return float(result[0].data.evs)


# %% [markdown]
# ### 期待値計算のテスト
#
# `estimate()` メソッドを簡単なハミルトニアンでテストしてみましょう。
# $H = Z_0 + 0.5 \cdot Z_0 Z_1$ を作成し、ベル状態に対して $\langle\psi|H|\psi\rangle$ を計算します。
#
# Qamomileでは、qkernel内で `qmc.expval()` を使用して期待値を計算します。
# ハミルトニアンは `Observable` パラメータとしてバインディング経由で渡されます。

# %%
# Create a simple Hamiltonian: H = Z0 + 0.5 * Z0*Z1
hamiltonian = qm_o.Z(0) + 0.5 * qm_o.Z(0) * qm_o.Z(1)

print("Hamiltonian:", hamiltonian)


# %%
# Define a circuit that prepares a Bell state and computes expval
@qmc.qkernel
def bell_expval(H: qmc.Observable) -> qmc.Float:
    """Prepare a Bell state and compute <ψ|H|ψ>"""
    q = qmc.qubit_array(2, name="q")
    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    return qmc.expval(q, H)


bell_expval.draw()

# %%
# Transpile with the Hamiltonian bound
executable_expval = transpiler.transpile(bell_expval, bindings={"H": hamiltonian})

# Calculate expectation value with our custom executor
full_executor = MyFullExecutor()
job_expval = executable_expval.run(full_executor)
expectation = job_expval.result()

print("=== Expectation Value Calculation ===")
print("  Hamiltonian: Z0 + 0.5 * Z0*Z1")
print("  State: Bell state (|00⟩ + |11⟩)/√2")
print(f"  <ψ|H|ψ> = {expectation:.4f}")
print()
print("  Expected: For Bell state, <Z0> = 0, <Z0*Z1> = 1")
print("  So <H> = 0 + 0.5 * 1 = 0.5")

# %% [markdown]
# ベル状態における $Z_0$ の期待値は0です（$|0\rangle$ と $|1\rangle$ が等確率であるため）。
# 一方、$Z_0 Z_1$ の期待値は1です（両方の量子ビットが常に相関しているため）。
# したがって $\langle H \rangle = 0 + 0.5 \times 1 = 0.5$ となります。

# %% [markdown]
# ## 7. まとめ
#
# このチュートリアルでは、カスタムQuantumExecutorの実装方法を学びました。
#
# ### 3段階の実装レベル
#
# | レベル | 実装するメソッド | 用途 |
# |--------|-----------------|------|
# | **基本** | `execute()` | 単純な回路実行 |
# | **中級** | + `bind_parameters()` | パラメトリック回路 |
# | **上級** | + `estimate()` | 変分アルゴリズム（QAOAなど） |
#
# ### 実装のポイント
#
# 1. **execute()**: ビット列のカウントを `dict[str, int]` で返す（ビッグエンディアン形式）
# 2. **bind_parameters()**: `ParameterMetadata.to_binding_dict()` を活用する
# 3. **estimate()**: バックエンドのEstimatorプリミティブを利用する
#
# ### コード例のまとめ
#
# ```python
# from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
# from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
#
# class MyExecutor(QuantumExecutor[QuantumCircuit]):
#     def __init__(self):
#         self.backend = ...  # Initialize backend
#
#     def execute(self, circuit, shots):
#         # Execute circuit and return bitstring counts
#         ...
#         return {"00": 512, "11": 512}
#
#     def bind_parameters(self, circuit, bindings, metadata):
#         # Convert to backend format with metadata.to_binding_dict()
#         return circuit.assign_parameters(metadata.to_binding_dict(bindings))
#
#     def estimate(self, circuit, observable, params):
#         # Calculate expectation value with Estimator primitive
#         ...
#         return expectation_value
# ```
#
# ### 次のステップ
#
# - QAOAを本番のコンバータで使う方法は[最適化セクション](../optimization/qaoa.ipynb)を参照してください
# - QPEや標準ライブラリ関数については[05_stdlib](05_stdlib.ipynb)を参照してください

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **QuantumExecutorの役割と構造** — `QuantumExecutor[T]` は `execute()`、`bind_parameters()`、`estimate()` を持つ抽象基底クラスであり、トランスパイル済みプログラムとバックエンドを橋渡しします。
# - **最小限のカスタムExecutorの作成方法** — `execute()` のみを実装し、`dict[str, int]`（ビッグエンディアンのビット列カウント）を返すだけで、任意のバックエンドで回路を実行できます。
# - **IBM Quantumクラウドへの接続方法** — `qiskit-ibm-runtime` と `SamplerV2` を使用して、実際のIBM量子デバイスに回路を送信できます。
# - **パラメータバインディングの実装方法** — `ParameterMetadata.to_binding_dict()` を活用して `bind_parameters()` を実装することで、再トランスパイルなしにパラメトリック回路をサポートできます。
# - **期待値計算（estimate）の実装方法** — Estimatorプリミティブを用いて `estimate()` を実装することで、変分アルゴリズムに必要な $\langle\psi|H|\psi\rangle$ を計算できます。
