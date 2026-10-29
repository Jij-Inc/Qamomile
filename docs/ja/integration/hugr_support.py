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
# Qamomileでベル状態の回路を作り、HUGR（量子プログラムを表す中間形式）へ変換してSeleneでローカル実行します。同じプログラムをNexus経由でQuantinuum H2実機へ送信する手順も紹介します。
#
# Python 3.11または3.12を使用し、`hugr`オプションを付けてインストールしてください。実機での実行には、NexusアカウントとH2-1へのアクセス権が必要です。
#
# :::{note}
# Heliosでの動作は未確認です。
# :::

# %%
# Install Qamomile with HUGR support.
# # !pip install "qamomile[hugr]"

# %%
import qamomile.circuit as qmc
from qamomile.hugr import HugrExecutor, HugrTranspiler

# %% [markdown]
# ## Qamomileで回路を作る
#
# 2つの量子ビットをもつれさせて測定します。戻り値には2つの測定ビットが入ります。


# %%
@qmc.qkernel
def bell() -> tuple[qmc.Bit, qmc.Bit]:
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.h(left)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


program = HugrTranspiler().transpile(bell)

# %% [markdown]
# ## ローカルで確認する
#
# Seleneで100ショット実行します。ノイズのないシミュレーションでは、2つの測定ビットは常に一致します。`False`は0、`True`は1に対応します。

# %%
shots = 100
local_result = program.sample(HugrExecutor("selene"), shots=shots).result()
counts = dict(local_result.results)

assert sum(counts.values()) == shots
assert set(counts) <= {(False, False), (True, True)}
print(counts)

# %% [markdown]
# ## H2実機で実行する
#
# `system_name="H2-1"`でH2実機を選択します。`project`はNexusのプロジェクト名で、存在しなければQamomileが作成します。初回の送信時に認証が始まり、必要に応じてブラウザーでログインします。
#
# :::{note}
# 次のコードは実機にジョブを送信し、利用枠を消費します。費用が発生する場合があります。送信する際はコードセルにコピーして実行してください。
# :::
#
# ```python
# from qamomile.hugr import NexusExecutionOptions
#
# executor = HugrExecutor(
#     "nexus",
#     options=NexusExecutionOptions(
#         system_name="H2-1",
#         project="qamomile-demo",
#     ),
# )
# job = program.sample(executor, shots=100)
# print(job.status())
# ```
#
# 送信前のプログラムの変換と検証はQamomileが行います。Nexus経由でH2エミュレーターを使う場合は、`system_name="H2-1E"`に変更してください。
#
# `job.result()`で実行完了を待ち、測定結果ごとの出現回数を取得します。実機ではノイズにより、2つのビットが一致しない結果も現れることがあります。
#
# ```python
# result = job.result()
# print(dict(result.results))
# ```

# %% [markdown]
# ## get_current_shotを使う
#
# ショット（量子プログラムの1回の実行）の番号は、`@qmc.qkernel`で定義した関数内の`qmc.get_current_shot()`で取得できます。同じショット内で繰り返し呼び出しても、同じ番号を返します。
#
# Quantinuum H2へNexus経由でネイティブのプログラムを送る場合、デバイス側のショット番号は1-origin（最初のショットが1）です。一方、Seleneは0-origin（最初のショットが0）です。Qamomileはこの違いを吸収するため、`qmc.get_current_shot()`はSeleneとH2のどちらでも0-originになります。H2向けに自分で1を引く必要はありません。
#
# 次の例は、ショット番号が0のときだけ`qmc.x`（0と1を反転する操作）を適用します。ノイズのないSeleneで4ショット実行すると、最初の測定結果は`True`、残り3回は`False`となり、出現回数は`{True: 1, False: 3}`になります。


# %%
@qmc.qkernel
def first_shot_only() -> qmc.Bit:
    q = qmc.qubit("q")
    if qmc.get_current_shot() == 0:
        q = qmc.x(q)
    return qmc.measure(q)


shot_program = HugrTranspiler().transpile(first_shot_only)
shot_result = shot_program.sample(HugrExecutor("selene"), shots=4).result()
shot_counts = dict(shot_result.results)

assert shot_counts == {True: 1, False: 3}
print(shot_counts)

# %% [markdown]
# H2実機で試す場合は、前の節のH2用`executor`を再利用します。次の例も実機へ送信するため、利用枠を消費し、費用が発生する場合があります。理想的には同じ出現回数になりますが、実機ではノイズにより測定結果が変わることがあります。
#
# ```python
# shot_job = shot_program.sample(executor, shots=4)
# print(dict(shot_job.result().results))
# ```

# %% [markdown]
# ## まとめ
#
# - `HugrExecutor("selene")`で回路をローカル実行して確認できます。
# - 同じプログラムを`HugrExecutor("nexus")`でH2に送信し、`job.result()`で測定結果の出現回数を取得できます。
# - `qmc.get_current_shot()`のショット番号は、SeleneとH2のどちらでも0-originです。
