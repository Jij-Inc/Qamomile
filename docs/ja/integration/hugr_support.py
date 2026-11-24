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
# # HUGRサポート
#
# 2量子ビットの回路を、量子プログラムの中間表現である[HUGR](https://github.com/Quantinuum/hugr)にコンパイルします{cite:p}`hugr-ja`。Seleneを使ってローカルで実行し、同じ回路をQuantinuumの実機Helios-1に送信する方法を紹介します。
#
# :::{note}
# このページの「対応」「非対応」は、Qamomile経由で利用できるかどうかを示します。HUGRの仕様や個々のデバイスの対応範囲とは異なる場合があります。
# :::

# %% [markdown]
# ## 準備
#
# HUGR対応のQamomileをインストールします。ローカルでの実行にはシミュレータのSeleneを使います。
#
# ```bash
# pip install "qamomile[hugr]"
# ```

# %%
# # %pip install "qamomile[hugr]"

# %%
import math
import os

import qnexus as qnx  # noqa: F401 -- Used in the hardware example.

import qamomile as qm
from qamomile.hugr import Helios, HeliosOptions, Selene  # noqa: F401

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
shots = 1 if docs_test_mode else 32
local = Selene()

# %% [markdown]
# ## HUGRでできること
#
# ✅は対応、✗は非対応です。
#
# | 機能 | `Selene` | `Helios`（Helios-1） |
# | --- | :---: | :---: |
# | 実行時に回転角を指定する | ✅ | ✅ |
# | 量子ビットを測定せずに捨てる（`qm.discard`） | ✅ | ✅ |
# | 中間測定する（`qm.measure_z`, `qm.measure_x`, `qm.measure_y`） | ✅ | ✅ |
# | 中間測定の結果に応じて同じ回路内で分岐する（`if`） | ✅ | ✅ |
# | 中間測定の結果で、同じ回路内の繰り返し条件（`while`）や回数（`for`）を決める | ✗ | ✗ |
# | 回路の途中で量子ビットをリセットして再利用する（`qm.reset`） | ✅ | ✅ |
# | 中間測定の結果から決まる添字を使って古典配列の要素を選ぶ | ✅ | ✅ |
# | 回路内で現在の試行番号を読む（`qm.get_current_shot`） | ✅ | ✗ |
#
# SeleneとHelios-1では、X、Y、Zのどの基底でも中間測定ができます。どちらも、繰り返し条件と回数は回路の実行前に確定している必要があります。

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
# ### 中間測定の結果から決まる添字を使って古典配列の要素を選ぶ
#
# 1つ目の測定値を添字にして、2つの角度から2つ目の量子ビットの回転角を選びます。1を測定すると$\pi$だけ回転するので、結果は毎回`(1, 1)`です。


# %%
@qm.qkernel
def select_rotation() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Use a measured index to select the rotation on the second qubit."""
    state = qm.prepare((1, 0))
    flag, state = qm.measure_z(state, target=0)
    index = 1 if flag else 0
    angles = [0.0, qm.pi]
    state = qm.ry(state, angles[index], target=1)
    return qm.measure(state)


lookup_counts = qm.run(select_rotation, on=local, shots=shots).result.counts
print(lookup_counts)
assert lookup_counts == {(1, 1): shots}

# %% [markdown]
# ### 回路内で現在の試行番号を読む（`qm.get_current_shot`）
#
# 現在の試行番号が0かどうかを返します。最初の試行だけが1、残りは0です。32回実行すると1が1回、0が31回です。


# %%
@qm.qkernel
def first_shot() -> qm.Shot[qm.Bool]:
    """Return whether this is the first shot."""
    return qm.get_current_shot() == 0


shot_counts = qm.run(first_shot, on=local, shots=shots).result.counts
print(shot_counts)
assert shot_counts[(1,)] == 1
assert shot_counts.get((0,), 0) == shots - 1

# %% [markdown]
# ## 実機での実行
#
# 同じ`bell_program`を実機Helios-1に送信するには、実機の利用権限があるアカウントでQuantinuum Nexusにログインします。`"your-project"`を利用するNexusプロジェクトの名前に置き換えてください。`job.job_id`でジョブIDを確認し、`job.result()`で完了を待って測定結果を取得します。
#
# ```python
# qnx.login()
# project = qnx.projects.get(name="your-project")
# endpoint = Helios(HeliosOptions(project=project, system_name="Helios-1"))
# artifact = qm.compile(bell_program, to=endpoint.target)
# job = qm.submit(artifact, on=endpoint, parameters={"theta": math.pi}, shots=100)
# print(job.job_id)
# print(job.result().counts)
# ```
