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
# # QURI Partsサポート
#
# 2量子ビットの回路をQulacsシミュレーターで実行し、同じ回路を[QURI Parts](https://quri-sdk.qunasys.com/docs/tutorials/quri-parts/)経由でIQM Emeraldの実機に送信する例を紹介します{cite:p}`quri-parts-ja`。
#
# :::{note}
# このページの「対応」「非対応」は、Qamomile経由で利用できるかどうかを示します。QURI Partsの仕様や個々のデバイスの対応範囲とは異なる場合があります。
# :::

# %% [markdown]
# ## 準備
#
# Qulacsシミュレーターを使うため、QURI Parts用の追加パッケージを含めてQamomileをインストールします。
#
# ```bash
# pip install "qamomile[quri_parts]"
# ```

# %%
# # %pip install "qamomile[quri_parts]"

# %%
import math
import os

import qamomile as qm
from qamomile.quri_parts import Qulacs

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
shots = 1 if docs_test_mode else 32
local = Qulacs()

# %% [markdown]
# 実機を使う場合は、Amazon Braketを利用できるAWS認証情報を設定し、次のパッケージをインストールしてインポートします。
#
# ```bash
# pip install quri-parts-braket
# ```
#
# ```python
# from braket.aws import AwsDevice
# from braket.devices import Devices
# from quri_parts.braket.backend import BraketSamplingBackend
# from quri_parts.core.sampling import create_sampler_from_sampling_backend
#
# from qamomile.quri_parts import QuriPartsSampler
# ```

# %% [markdown]
# ## QURI Partsでできること
#
# `Qulacs`は手元のシミュレーターで回路を実行します。`QuriPartsSampler`は、回路と実行回数をQURI Partsのサンプラー（測定結果ごとの回数を返す関数）に渡すQamomileのクラスです。どのシミュレーターや実機を使うかは、渡すサンプラーで決まります。
#
# ✅は対応、✗は非対応です。
#
# | 機能 | `Qulacs` | `QuriPartsSampler` |
# | --- | :---: | :---: |
# | 実行時に回転角を指定する | ✅ | ✅ |
# | 量子ビットを測定せずに捨てる（`qm.discard`） | ✅ | ✅ |
# | 中間測定する（`qm.measure_z`, `qm.measure_x`, `qm.measure_y`） | ✗ | ✗ |
# | 中間測定の結果に応じて同じ回路内で分岐する（`if`） | ✗ | ✗ |
# | 中間測定の結果で、同じ回路内の繰り返し条件（`while`）や回数（`for`）を決める | ✗ | ✗ |
# | 回路の途中で量子ビットをリセットして再利用する（`qm.reset`） | ✗ | ✗ |
# | 中間測定の結果から決まる添字を使って古典配列の要素を選ぶ | ✗ | ✗ |
# | 回路内で現在の試行番号を読む（`qm.get_current_shot`） | ✗ | ✗ |
#
# どちらも実行ごとに回転角を指定でき、回路の最後に測定します。

# %% [markdown]
# ## ローカルシミュレーション
#
# ### 実行時に回転角を指定する
#
# 1つ目の量子ビットを`theta`ラジアンだけ回転させます。次にCNOTゲートを適用します。この回路では、最後に両方の量子ビットをまとめて測定します。`theta`を`qm.Param`で宣言すると、実行のたびに角度を指定できます。


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
# ローカルシミュレーター向けに回路を一度だけコンパイルします。`theta`に2通りの角度を指定し、それぞれ`shots`回実行します。

# %%
bell_program = qm.specialize(parameterized_bell)
bell_artifact = qm.compile(bell_program, to=local.target)

for theta, expected_bits in [(0.0, (0, 0)), (math.pi, (1, 1))]:
    result = qm.run(
        bell_artifact, on=local, parameters={"theta": theta}, shots=shots
    ).result
    counts = result.counts
    print(f"theta = {theta:.3f}", counts)
    assert counts == {expected_bits: shots}

# %% [markdown]
# `counts` から、測定したビットの組ごとの出現回数を読み取れます。角度が0なら毎回両方が0、$\pi$なら毎回両方が1です。

# %% [markdown]
# ### 量子ビットを測定せずに捨てる（`qm.discard`）
#
# 2つのビットが異なるかどうかだけを調べます。CNOTゲートを使うと、2つの値が等しければ0、異なれば1が最初の量子ビットに残ります。`qm.split_at()`で2つに分け、`qm.discard()`で2つ目を手放し、最初の量子ビットだけを測定します。この例の入力は`(1, 1)`なので、結果は0です。


# %%
@qm.qkernel
def parity_only() -> qm.Shot[qm.Bits[1]]:  # type: ignore[valid-type]
    """Return the parity of two bits and discard the unused qubit."""
    state = qm.prepare((1, 1))
    state = qm.cx(state, control=1, target=0)
    parity, unused = qm.split_at(state, 1)
    qm.discard(unused)
    return qm.measure(parity)


parity_counts = qm.run(parity_only, on=local, shots=shots).result.counts
print(parity_counts)
assert parity_counts == {(0,): shots}

# %% [markdown]
# ## 実機での実行
#
# 同じ`bell_program`をAmazon Braket経由でIQM Emeraldに送信します。この例ではAWSアカウントを使用し、実行料金がかかります。サンプラーは`qm.submit()`の呼び出し中に測定結果を待ちます。`job.job_id`は完了したQamomileのジョブを識別するIDです。
#
# ```python
# device = AwsDevice(Devices.IQM.Emerald)
# sampler = create_sampler_from_sampling_backend(BraketSamplingBackend(device))
# endpoint = QuriPartsSampler(sampler, label="iqm-emerald")
# artifact = qm.compile(bell_program, to=endpoint.target)
# job = qm.submit(artifact, on=endpoint, parameters={"theta": math.pi}, shots=100)
# print(job.job_id)
# print(job.result().counts)
# ```
