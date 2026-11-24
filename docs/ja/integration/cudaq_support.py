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
# # CUDA-Qサポート
#
# 2量子ビットの回路を[CUDA-Q](https://nvidia.github.io/cuda-quantum/0.14.0/using/install/local_installation.html)のローカルCPUシミュレーターで実行し、Qamomile経由で使える機能を確認します{cite:p}`cudaq-ja`。
#
# :::{note}
# このページの「対応」「非対応」は、Qamomile経由で利用できるかどうかを示します。CUDA-Qの仕様や個々のデバイスの対応範囲とは異なる場合があります。
# :::

# %% [markdown]
# ## 準備
#
# CUDA-Qに対応する追加パッケージを指定して、Qamomileをインストールします。Appleシリコン搭載のMacとCUDA 13を使うLinuxでは`cudaq-cu13`を、CUDA 12を使うLinuxでは`cudaq-cu12`を指定します。
#
# ```bash
# pip install "qamomile[cudaq-cu13]"
# ```

# %%
# # %pip install "qamomile[cudaq-cu13]"

# %%
import math
import os

import qamomile as qm
from qamomile.cudaq import CudaqLocal
from qamomile.persistence import export_payload  # noqa: F401

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
shots = 1 if docs_test_mode else 32
local = CudaqLocal(runtime_target="qpp-cpu")

# %% [markdown]
# ## CUDA-Qでできること
#
# ローカルシミュレーターで利用できる機能を✅、利用できない機能を✗で示します。
#
# | 機能 | `CudaqLocal`（`qpp-cpu`） |
# | --- | :---: |
# | 実行時に回転角を指定する | ✅ |
# | 量子ビットを測定せずに捨てる（`qm.discard`） | ✅ |
# | 中間測定する（`qm.measure_z`, `qm.measure_x`, `qm.measure_y`） | ✅ |
# | 中間測定の結果に応じて同じ回路内で分岐する（`if`） | ✅ |
# | 中間測定の結果で、同じ回路内の繰り返し条件（`while`）や回数（`for`）を決める | ✗ |
# | 回路の途中で量子ビットをリセットして再利用する（`qm.reset`） | ✅ |
# | 中間測定の結果から決まる添字を使って古典配列の要素を選ぶ | ✗ |
# | 回路内で現在の試行番号を読む（`qm.get_current_shot`） | ✗ |

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
# ### 中間測定する（`qm.measure_z`, `qm.measure_x`, `qm.measure_y`）
#
# 1に準備した量子ビットを`qm.measure_z(state, target=0)`で測定すると、測定値と測定後の量子ビットが返ります。その量子ビットをXゲートで反転してからもう一度測定すると、2回の測定結果は毎回`(1, 0)`になります。


# %%
@qm.qkernel
def measure_then_flip() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    """Measure the same qubit before and after a flip."""
    state = qm.prepare((1,))
    before, state = qm.measure_z(state, target=0)
    state = qm.x(state, target=0)
    return bool(before), bool(qm.measure(state)[0])


measurement_counts = qm.run(measure_then_flip, on=local, shots=shots).result.counts
print(measurement_counts)
assert measurement_counts == {(1, 0): shots}

# %% [markdown]
# ### 中間測定の結果に応じて同じ回路内で分岐する（`if`）
#
# まず、1つ目の量子ビットを、測ると0と1のどちらも出る状態にします。`measure_z`は測定結果のビット（`flag`）と測定後の状態を返します。`flag`が1のときだけ、初めは0だった2つ目の量子ビットを反転します。これで最後に測定する2つのビットが一致します。


# %%
@qm.qkernel
def correlated_feedback() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Correlate two output bits through measurement feedback.

    Returns:
        qm.Shot[qm.Bits[2]]: Two equal bits in Qamomile source order.
    """

    state = qm.qubits(2)
    state = qm.h(state, target=0)
    flag, state = qm.measure_z(state, target=0)
    if flag:
        state = qm.x(state, target=1)
    return qm.measure(state)


counts = qm.run(correlated_feedback, on=local, shots=shots).result.counts
print(counts)
assert set(counts) <= {(0, 0), (1, 1)}
assert sum(counts.values()) == shots

# %% [markdown]
# それぞれの回数は実行するたびに変わり得ます。

# %% [markdown]
# ### 回路の途中で量子ビットをリセットして再利用する（`qm.reset`）
#
# 最初の3つの量子ビットに`010`を用意し、隣り合う2つの値が異なるかを4つ目の量子ビットで調べます。2回のCNOTゲートで、調べる組に1があるたびに4つ目を反転します。初期値は0なので、左の`01`を調べると1になります。`qm.reset()`で0に戻してから右の`10`を調べると、こちらも1です。


# %%
@qm.qkernel
def check_neighbors() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    """Use one auxiliary qubit to compare two neighboring pairs."""
    state = qm.prepare((0, 1, 0, 0))
    state = qm.cx(state, control=0, target=3)
    state = qm.cx(state, control=1, target=3)
    left, state = qm.measure_z(state, target=3)

    data, probe = qm.split_at(state, 3)
    probe = qm.reset(probe)
    state = qm.concat(data, probe)

    state = qm.cx(state, control=1, target=3)
    state = qm.cx(state, control=2, target=3)
    right, state = qm.measure_z(state, target=3)
    qm.discard(state)
    return bool(left), bool(right)


checks = qm.run(check_neighbors, on=local, shots=shots).result.counts
print(checks)
assert checks == {(1, 1): shots}

# %% [markdown]
# ## 実機での実行
#
# QamomileにはCUDA-Qのローカルシミュレーター用の実行先があります。CUDA-Qの実機に送信するための実行先はありません。Qamomileの外で実機を使う準備として、コンパイル済みのベル回路からCUDA-QのPythonソースを取り出して確認できます。
#
# ```python
# print(export_payload(bell_artifact).decode("utf-8"))
# ```
