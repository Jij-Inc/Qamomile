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
# tags: [integration]
# ---
#
# # Amazon Braketサポート
#
# このページでは、Qamomileの量子カーネルをAmazon Braketネイティブ回路へ
# トランスパイルし、ローカルシミュレータでサンプリングと期待値計算を実行します。
# リモート実行時も、同じexecutor境界へAWSデバイスを渡せます。

# %%
# Amazon Braket対応を含む最新のQamomileをインストールします。
# # !pip install "qamomile[braket]"

# %%
import math

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.braket import BraketExecutionOptions, BraketTranspiler

# %% [markdown]
# ## Qamomileで回路を作る
#
# ランタイムパラメータはBraket回路内にシンボリックなまま残ります。executorは
# Braketネイティブの`inputs` APIで値を送り、provider側のコンパイルと
# パラメータ処理を維持します。

# %%
@qmc.qkernel
def parameterized_bell(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Prepare and measure a parameterized Bell-like state.

    Args:
        theta (qmc.Float): Rotation angle.

    Returns:
        qmc.Vector[qmc.Bit]: Two measured output bits.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.ry(q[0], theta)
    q[0], q[1] = qmc.cx(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def plus_expectation(observable: qmc.Observable) -> qmc.Float:
    """Prepare a plus state and evaluate an observable.

    Args:
        observable (qmc.Observable): Observable to evaluate.

    Returns:
        qmc.Float: Observable expectation value.
    """
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.h(q[0])
    return qmc.expval(q, observable)

# %% [markdown]
# ## Braketへトランスパイルする
#
# `BraketTranspiler`はネイティブな`braket.circuits.Circuit`を生成します。
# 終端測定はQamomile側の結果マッピングとして保持され、executorがBraketの
# 測定量子ビット順を一貫した形へ正規化します。

# %%
transpiler = BraketTranspiler()
executable = transpiler.transpile(parameterized_bell, parameters=["theta"])
braket_circuit = executable.quantum_circuit

assert type(braket_circuit).__module__.startswith("braket.")
assert {str(parameter) for parameter in braket_circuit.parameters} == {"theta"}
print(braket_circuit)

# %% [markdown]
# ## ローカルで実行する
#
# デバイスを省略すると、executorはBraketの`LocalSimulator`を作成します。

# %%
executor = transpiler.executor()
sample_job = executable.sample(
    executor,
    shots=128,
    bindings={"theta": math.pi},
)
sample = sample_job.result()

assert sample.results == [((1, 1), 128)]
print(sample.results)

# %%
energy_program = transpiler.transpile(
    plus_expectation,
    bindings={"observable": qm_o.X(0)},
)
energy = energy_program.run(executor).result()

assert abs(energy - 1.0) < 1e-10
print("<X> =", energy)

# %% [markdown]
# ## AWSデバイスを使う
#
# `AwsDevice`を`transpiler.executor(device)`へ渡すと、同じjob APIを
# リモートでも利用できます。submit後は結果取得を待たずにjobが返り、正規化された
# status、キャンセル、native task、task ARN参照を利用できます。
#
# ```python
# from braket.aws import AwsDevice
#
# device = AwsDevice("your-device-arn")
# options = BraketExecutionOptions(
#     s3_destination_folder=("your-bucket", "qamomile-results"),
#     poll_timeout_seconds=600,
#     poll_interval_seconds=2,
#     batch_max_retries=0,
# )
# aws_executor = transpiler.executor(device, options=options)
# job = executable.sample(aws_executor, shots=1_000, bindings={"theta": 0.4})
# print(job.status(), job.raw_status())
# references = job.references()
#
# # 別processでexecutorを再作成し、raw task resultを復元します。
# restored = aws_executor.restore(references[0])
# counts = restored.result()
#
# # キャンセルはbest effortで、task参照は失われません。
# job.cancel()
# ```
#
# retryは追加の課金taskを作る可能性があるため、batch再送はデフォルトで無効です。
# 意図的に再送する場合だけ`batch_max_retries`を設定してください。
# `poll_timeout_seconds`はproviderのpolling制限と、ローカルで結果を待つデフォルト時間を設定します。`result(timeout=...)`を明示すると、このデフォルト値を上書きします。timeout時もremote taskはキャンセルされません。非同期待機には`await job.result_async()`を利用できます。
#
# リモートタスクにはAWS認証情報とS3出力先が必要で、費用が発生する場合があります。
# そのため、このページではローカルシミュレータだけを実行します。

# %% [markdown]
# ## 制約
#
# 現在のBraketエンジンは静的なgate-model回路を対象としています。測定結果に依存する`if`や`while`のcontrol flow、mid-circuit reset、およびそれらの操作を必要とするアルゴリズムには対応していません。これには、modular算術やShorアルゴリズムのうちresetに依存する実行経路が含まれます。そのようなプログラムは静的回路へ書き換えるか、必要なdynamic-circuit primitiveを提供するエンジンを選択してください。

# %% [markdown]
# ## まとめ
#
# - `BraketTranspiler`はBraketネイティブ回路とフリーパラメータを生成します。
# - `BraketExecutor`は結果取得を待たず、サンプリングとHamiltonian期待値計算を
#   native taskとしてsubmitします。
# - jobはstatus、キャンセル、native task、復元可能なAWS task参照を公開し、
#   batch retryは明示的かつ回数制限付きです。
# - `AwsDevice`を注入すれば、Qamomileカーネルを再コンパイルせず実行先を切り替えられます。
