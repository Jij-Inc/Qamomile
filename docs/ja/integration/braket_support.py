# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
# # Amazon Braket Support
#
# QamomileはAmazon Braketをrealization境界に保ちます。ユーザーは1つの型付きQamomile programを書き、`Braket` targetがportable OpenQASM 3をemitし、`BraketLocal`が分離されたlocal-simulator workerで実行します。`BraketDevice` endpointを使うと、AWS sessionやdevice objectをQCoreへ入れずに、互換性のあるartifactをmanaged deviceへsubmitできます。
#
# このページでは、Braketのdynamic OpenQASM routeをstatic logical-circuit統合と区別するため、measurement feedbackを使います。source-levelの意味、linear ownership、resource、result shapeを所有するのは引き続きRustです。

# %% [markdown]
# ## インストール
#
# ```bash
# pip install "qamomile[braket]"
# ```

# %%
# Install the latest Qamomile and its Amazon Braket integration through pip!
# # !pip install "qamomile[braket]"

# %%
import math

import qamomile as qm

from qamomile.persistence import deserialize, serialize
from qamomile.braket import Braket, BraketLocal, load_openqasm

# %% [markdown]
# ## Qamomileで構築する
#
# runtime parameterはemitted Braket program内でsymbolicなまま保持されます。
# endpointはinvocation時に値を渡すため、provider側のcompileとparameter handlingを
# 維持しながら、1つのimmutable artifactを再利用できます。


# %%
@qm.qkernel
def parameterized_bell(
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare and measure a parameterized Bell-like state.

    Args:
        theta (qm.Param[qm.Real]): Rotation angle in radians.

    Returns:
        qm.Shot[qm.Bits[2]]: Two measured output bits.
    """

    state = qm.qubits(2)
    state = qm.ry(state, theta, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


# %% [markdown]
# ## Braketへtranspileする
#
# `Braket` targetはportableなOpenQASM 3をemitします。terminal measurementは
# Qamomileのresult ABIに保持され、endpointがBraketのmeasured-qubit orderを
# 一貫してnormalizeできます。

# %%
bell_program = qm.specialize(parameterized_bell)
bell_artifact = qm.compile(bell_program, to=Braket())
bell_source = load_openqasm(bell_artifact)

assert bell_artifact.format == "braket.openqasm3.v1"
assert tuple(slot.name for slot in bell_artifact.parameter_abi.slots) == ("theta",)
print(bell_source)
assert "input float[64] p0;" in bell_source

# %% [markdown]
# ## localで実行する
#
# `BraketLocal`はisolated worker内でBraketのlocal simulatorを作ります。
# 同じartifactを異なる`theta`で繰り返しinvokeできます。

# %%
bell_result = qm.run(
    bell_artifact,
    on=BraketLocal(),
    parameters={"theta": math.pi},
    shots=32,
    seed=0,
).result

assert bell_result.counts == {(1, 1): 32}
print(bell_result.counts)

# %% [markdown]
# ## 型付きfeedback programを1つ書く
#
# `measure_z`はshot-time Booleanと同じaffine量子状態を返します。その後、通常のPython `if`構文で2番目の量子ビットに対する操作を選択します。両branchは所有された状態を1つだけ返します。実行時のmeasurement eventは3つですが、最後のmeasurementが返す公開resultは2 bitです。


# %%
@qm.qkernel
def correlated_feedback() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Correlate two output bits through measurement feedback.

    Returns:
        qm.Shot[qm.Bits[2]]: Two equal bits in Qamomile source order.
    """

    state = qm.qubits(2)
    state = qm.h(state, target=0)
    flag, state = qm.measure_z(state, 0)
    if flag:
        state = qm.x(state, target=1)
    else:
        state = qm.z(state, target=1)
    return qm.measure(state)


program = correlated_feedback.specialize()
study = qm.resources.logical(program)

assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (2, 2)
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (2, 2)

# %% [markdown]
# ## 一度コンパイルして実行対象を保持する
#
# `qm.compile`は、Braket OpenQASM Program、SDKオブジェクトの所有者、コンパイラが導出したABIとリソース情報を保持する`CompiledProgram`を返します。通常の実行では保存・復元を行いません。以下の`serialize`と`deserialize`は、必要な場合だけ`qamomile.persistence`で保存する例です。復元時には新しい実行対象を生成するため、Pythonオブジェクトの等値ではなくコンパイル識別子を比較します。

# %%
target = Braket()
artifact = qm.compile(program, to=target)
restored = deserialize(serialize(artifact))

assert artifact.format == "braket.openqasm3.v1"
assert restored.execution_identity == artifact.execution_identity
assert artifact.execution_identity == qm.compile(program, to=target).execution_identity
assert artifact.result_abi.width == 2

# %% [markdown]
# `load_openqasm`は明示的なinspection helperであり、別のcompilerやexecution pathではありません。非公開のfeedback measurementは`c[0]`を使い、最後の2つのmeasurementは残りのnative bitを使います。result ABIは宣言された2 bit値だけを保持し、source順序を所有します。

# %%
source = load_openqasm(restored)

print(source)

assert source.startswith("OPENQASM 3.0;")
assert "bit[3] c;" in source
assert "if (c[0])" in source
assert "c[1] = measure" in source
assert "c[2] = measure" in source

# %% [markdown]
# ## 分離されたlocal simulatorで実行する
#
# `BraketLocal`がSDK simulatorを構築するのはworker process内だけです。endpointはコンパイル済みOpenQASMを実行し、Braket nativeのq0-first resultを1回だけ正規化して、共通の型付き`SampleResult`を返します。このlocal profileはsimulator seedの再現性を約束しないため、portable seedを明示的に0にします。

# %%
record = qm.run(
    restored,
    on=BraketLocal(),
    shots=32,
    seed=0,
)

print(record.result.counts)

assert record.artifact is restored
assert sum(record.result.counts.values()) == 32
assert set(record.result.counts) <= {(0, 0), (1, 1)}
assert all(left == right for left, right in record.result.counts)

# %% [markdown]
# ## 同じlocal endpointでexpectation valueを評価する
#
# samplingはBraket専用の特別なAPIではありません。状態を返すsourceはQamomile共通のexpectation contractで評価でき、observable measurement workは同じOpenQASM targetへloweringされます。plus stateに対する$X$の厳密なexpectationは1です。


# %%
@qm.qkernel
def plus_state() -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
    """Prepare a one-qubit plus state.

    Returns:
        qm.Q[qm.Bits[1]]: Owned plus state.
    """

    return qm.h(qm.qubits(1), target=0)


plus_x = qm.expectation(
    plus_state,
    qm.X(0),
    on=BraketLocal(),
    shots=8,
    seed=0,
)
print("<X> =", plus_x.value)
assert plus_x.value == 1.0

# %% [markdown]
# ## 互換性のあるprogramをmanaged deviceへrouteする
#
# live deviceはendpoint内だけに残ります。`BraketDevice.target`はimmutableなdevice capabilityとcalibration snapshotをartifact fingerprintへ反映しますが、SDK object、credential、taskはpersistent program dataへ入りません。managed deviceはlocalのdynamic-control capabilityやportable seed contractを約束しません。そのためQamomileはsubmit前にcapabilityを確認し、requestを暗黙に弱めず`seed=0`を要求します。
#
# ```python
# from braket.aws import AwsDevice
# from qamomile.braket import BraketDevice
#
# device = AwsDevice("your-device-arn")
# endpoint = BraketDevice(
#     device,
#     poll_timeout_seconds=600,
#     poll_interval_seconds=2,
# )
# job = qm.submit(your_static_program, on=endpoint, shots=100, seed=0)
# print(job.job_id, job.status)
# result = job.result()
# print(result.counts)
# job.cancel()  # best effort; the task reference remains available
# ```
#
# remote submissionには設定済みAWS credentialが必要で、費用が発生する場合があるため、このページでは実行しません。local endpointとmanaged endpointは同じ`run`と型付きresult contractに従います。

# %% [markdown]
# ## 制限事項
#
# local OpenQASM simulatorは上で示したdynamic feedbackを受け付けます。managed deviceは固有のimmutable capability snapshotをadvertiseし、defaultの`BraketDevice` contractはdynamic branchとresetを除外するため、そのprogramはsubmission前のcapability checkで失敗します。remote taskにはAWS credentialが必要で、costが発生する場合があり、portable random seedも保証されません。device選択、polling limit、cancellation、retry policyはquantum programへ埋め込まずendpoint境界に置きます。

# %% [markdown]
# ## まとめ
#
# - QCoreはfeedback programの唯一のpersistentな意味であり続けます。
# - `Braket`はSDK objectを保存せず、deterministicなOpenQASM 3をemitします。
# - shot-time controlはBraketのemission境界でのみloweringされます。
# - `BraketLocal`はlocal simulationを分離し、native bit順序を正規化します。
# - `BraketDevice`はlive device、credential、task、capability snapshotをendpoint境界に保ちます。
# - 実行ページは最大2量子ビットで、32 sampling shotsとboundedな8-shot expectation evaluationだけを使います。
