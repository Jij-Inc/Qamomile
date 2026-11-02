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
# # HUGRサポート
#
# Qamomileでベル状態の回路を作り、HUGR（量子プログラムを表す中間形式）へコンパイルしてSeleneでローカル実行します。
#
# `hugr`オプションを付けてQamomileをインストールしてください。

# %%
# HUGRサポートを含めてQamomileをインストールします。
# # !pip install "qamomile[hugr]"

# %%
import qamomile as qm
from qamomile.hugr import Hugr, Selene

# %% [markdown]
# ## Qamomileで回路を作る
#
# 2つの量子ビットをもつれさせて測定します。戻り値には2つの測定ビットが入ります。


# %%
@qm.qkernel
def bell() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


program = qm.specialize(bell)
artifact = qm.compile(program, to=Hugr())

# %% [markdown]
# `qm.specialize(bell)`で、変換先を選ぶ前にプログラムを検証します。`qm.compile(program, to=Hugr())`で、次の手順に使うコンパイル済みのHUGRプログラムを作成します。
#
# ## ローカルで確認する
#
# Seleneで100ショット実行します。ショットは量子プログラムの1回の実行です。ノイズのないシミュレーションでは、2つの測定ビットは常に一致します。出現回数のキーは、測定結果を表す`0`と`1`のタプル（値を順番に並べた組）です。

# %%
shots = 100
record = qm.run(artifact, on=Selene(), shots=shots, seed=7)
local_result = record.result
counts = dict(local_result.counts)

assert local_result.shots == shots
assert sum(counts.values()) == shots
assert set(counts) <= {(0, 0), (1, 1)}
print(counts)

# %% [markdown]
# ## まとめ
#
# - `qm.compile(program, to=Hugr())`で回路をHUGRへコンパイルします。
# - `qm.run(artifact, on=Selene(), shots=100)`で回路をローカル実行し、`record.result.counts`から測定結果の出現回数を取得します。
