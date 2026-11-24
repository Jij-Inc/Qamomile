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
# # qBraidサポート
#
# 2量子ビットの回路を準備し、[qBraid](https://docs.qbraid.com/)経由で送信する例を紹介します{cite:p}`qbraid-ja`。この例で使うIQM Emeraldはクラウド上の実機です。
#
# :::{note}
# このページの「対応」「非対応」は、Qamomile経由で利用できるかどうかを示します。qBraidの仕様や個々のデバイスの対応範囲とは異なる場合があります。
# :::

# %% [markdown]
# ## 準備
#
# qBraid用の追加パッケージを含めてQamomileをインストールします。回路の準備と確認はqBraidにログインする前に試せます。
#
# ```bash
# pip install "qamomile[qbraid]"
# ```

# %%
# # %pip install "qamomile[qbraid]"

# %%
# ruff: noqa: F401 -- The hardware example uses math and QBraidDevice.
import math

import qamomile as qm
from qamomile.qbraid import QBraidDevice
from qamomile.qiskit import Qiskit, load_circuit

# %% [markdown]
# 1つ目の量子ビットを`theta`ラジアンだけ回転させ、CNOTゲートを適用して両方の量子ビットを測定します。`theta`を`qm.Param`で宣言すると、送信時に角度を指定できます。


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


bell_program = qm.specialize(parameterized_bell)

# %% [markdown]
# qBraidにはQiskitの回路を渡します。`bell_program`をコンパイルし、量子ビットが2つあり、実行時に指定する角度が1つあることを手元で確認します。

# %%
artifact = qm.compile(bell_program, to=Qiskit())
circuit = load_circuit(artifact)
print(circuit.draw(output="text"))
assert circuit.num_qubits == 2
assert circuit.num_parameters == 1

# %% [markdown]
# ## qBraidでできること
#
# この表に示すQamomileの対応範囲は、`QBraidDevice`の接続先を変えても同じです。
#
# ✅は対応、✗は非対応です。
#
# | 機能 | `QBraidDevice` |
# | --- | :---: |
# | 実行時に回転角を指定する | ✅ |
# | 量子ビットを測定せずに捨てる（`qm.discard`） | ✅ |
# | 中間測定する（`qm.measure_z`, `qm.measure_x`, `qm.measure_y`） | ✗ |
# | 中間測定の結果に応じて同じ回路内で分岐する（`if`） | ✗ |
# | 中間測定の結果で、同じ回路内の繰り返し条件（`while`）や回数（`for`）を決める | ✗ |
# | 回路の途中で量子ビットをリセットして再利用する（`qm.reset`） | ✗ |
# | 中間測定の結果から決まる添字を使って古典配列の要素を選ぶ | ✗ |
# | 回路内で現在の試行番号を読む（`qm.get_current_shot`） | ✗ |
#
# Qamomileは回転角を設定してから回路を送信します。`QBraidDevice`で測定できるのは回路の最後です。

# %% [markdown]
# ## 実機での実行
#
# qBraidのAPIキー（認証用の鍵）を設定し、アカウントからIQM Emeraldを利用できることを確認します。送信にはqBraidのクレジットを使います。次の例では同じ`bell_program`の回転角に$\pi$ラジアンを指定します。`job.job_id`で送信したジョブのIDを確認し、`job.result()`で測定結果を待ちます。
#
# ```python
# endpoint = QBraidDevice.from_device_id("aws:iqm:qpu:emerald")
# artifact = qm.compile(bell_program, to=endpoint.target)
# job = qm.submit(artifact, on=endpoint, parameters={"theta": math.pi}, shots=100)
# print(job.job_id)
# print(job.result().counts)
# ```
