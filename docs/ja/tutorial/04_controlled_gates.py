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
# tags: [tutorial]
# ---
#
# # `qmc.control`によるゲートと量子カーネルの制御
#
# `qmc.control`を使うと、Qamomileの任意のゲート(`qmc.rx`のようなビルトイン関数や、ユーザが書いた`@qmc.qkernel`)の制御版を作れます。
#
# `qmc.control`には2つのモードがあります。*concrete mode*は制御qubitの数をPythonの`int`で与え、*symbolic mode*は`qmc.UInt`の量子カーネルパラメータ(あるいはそれを含む式)で与えてtranspile時に解決します。`power=`、デフォルト引数、`Vector[Qubit]`を取る量子カーネル、古典kwargの並び替えなど大半の機能は両モードで同じ挙動です。モードによって違うのは制御引数の渡し方と一部の追加機能だけで、以降のセクションで分けて扱います。

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile

# %%
import math

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import UnreturnedBorrowError
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# (cg-1)=
# ## 1. 最小例: controlled-RX
#
# `qmc.control`の最も簡単かつ実用的な使い方は、Qamomileで用意されている1つのゲートを制御化することです。例えば以下では、1qubitゲートの`qmc.rx(q, angle)`を`qmc.control`に渡して、2qubitのcontrolled-RXゲートを得ています。


# %%
# 制御RXゲートを定義します。
crx = qmc.control(qmc.rx)


# %%
@qmc.qkernel
def crx_control_off() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    # 制御は|0>のままなので、制御回転は発火しません。
    c, t = crx(c, t, angle=math.pi)
    return qmc.measure(t)


crx_control_off.draw()


# %%
@qmc.qkernel
def crx_control_on() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    # 制御を|1>に立てるので、制御回転が発火します。
    c = qmc.x(c)
    c, t = crx(c, t, angle=math.pi)
    return qmc.measure(t)


crx_control_on.draw()

# %% [markdown]
# 制御が実際に効いていることを確かめるために、両方の量子カーネルをQiskitにtranspileしてsimulatorで実行し、targetの測定結果を確認します。`angle=math.pi`では`RX(pi)`が|0>を|1>に写すので、制御が|1>のときだけtargetは全shotで|1>になり、それ以外では|0>のままになります。

# %%
off_counts = dict(
    transpiler.transpile(crx_control_off)
    .sample(transpiler.executor(), shots=256)
    .result()
    .results
)
on_counts = dict(
    transpiler.transpile(crx_control_on)
    .sample(transpiler.executor(), shots=256)
    .result()
    .results
)
print("control |0> ->", off_counts)
assert off_counts == {0: 256}
print("control |1> ->", on_counts)
assert on_counts == {1: 256}

# %% [markdown]
# ポイントとして、
#
# - `crx = qmc.control(qmc.rx)`はqkernelの中でも外でもどちらに書いてもかまいません。返ってきたものは再利用可能な値なので、変数に置いて何度でも呼び出せます。
# - `crx(c, t, angle=...)`を呼ぶと、まず制御qubitがpositional引数として並び、次にtarget、最後に古典keyword引数が続きます。順序は制御化する対象の`qmc.rx(q, angle)`シグネチャを踏襲しつつ、先頭に制御を加えた形です。
# - 古典パラメータのkeyword名は制御化する対象の関数の名前をそのまま使います(`qmc.rx`なら`angle`、`qmc.p`なら`theta`など)。`qmc.control`が改名することはありません。

# %% [markdown]
# (cg-2)=
# ## 2. 2つのモードの概要
#
# `qmc.control`には2つのモードがあります。どちらかは`num_controls`に渡す型だけで決まります。Pythonの`int`なら*concrete mode*、`qmc.UInt`ハンドル(あるいは`n - 1`のような`UInt`式)なら*symbolic mode*です。その他の挙動はすべてこの選択から決まります。
#
# | 項目 | Concrete | Symbolic |
# | --- | --- | --- |
# | `num_controls=` | Pythonの`int`(デフォルト`1`) | `qmc.UInt`ハンドル、または`UInt`式 |
# | 制御引数 | 合計qubit数が`num_controls`に一致する1つ以上のpositional引数(`Qubit`、`VectorView`、`Vector[Qubit]`) | 1つのpositionalな`Vector[Qubit]` / `VectorView`の*pool*(single-pool形、任意で`control_indices`)、**または**`Qubit` / `VectorView` / `Vector[Qubit]`を混ぜた複数のpositional引数 |
# | `control_indices` | 受け付けない | 任意。poolのどの量子ビットがactiveかを指定 |
# | 制御数が解決される時点 | `qmc.control(...)`が評価された時(module load時かtracing時) | transpile時(`bindings`から) |
#
# `qmc.control`のほとんどの機能(`power=`、デフォルト値、古典kwargの並び替え、`Vector[Qubit]`を受け取る量子カーネル、multi-argの制御引数形など)は両モードで同じ挙動を示します。これらは[](#cg-3)でまとめます。[](#cg-4)はconcrete modeを必要とする唯一の形を、[](#cg-5)はsymbolic mode固有の機能を扱います。

# %% [markdown]
# (cg-3)=
# ## 3. 両モードで動作するパターン
#
# 本セクションの各機能は、どちらのモードでも同じ挙動を示します。以下では原則concrete modeを使いますが、同じ機能がsymbolic modeでも利用可能です。concrete modeと違うのは`num_controls`が`UInt`式であることと、qubit数の一致がtranspile時にチェックされることだけです。symbolic専用の引数形は[](#cg-5)で扱います。

# %% [markdown]
# (cg-3-1)=
# ### 3.1 任意のcallableを制御化
#
# `qmc.control`はビルトインのゲート関数(`qmc.rx`、`qmc.h`、`qmc.p`など)も、ユーザ定義の`@qmc.qkernel`も同様に受け付けます。以下の例では、`qmc.h`を制御化した`ch`とユーザ定義の`_h_then_rx`を制御化した`cg`の2つの制御演算を含む量子カーネルを扱います。


# %%
@qmc.qkernel
def _h_then_rx(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return q


ch = qmc.control(qmc.h)
cg = qmc.control(_h_then_rx)


# %%
@qmc.qkernel
def control_any_callable_demo() -> qmc.Vector[qmc.Bit]:
    # q[0]は共通の制御。q[1] / q[2]は2つのtarget。
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[0], q[1] = ch(q[0], q[1])
    q[0], q[2] = cg(q[0], q[2], theta=math.pi / 4)
    return qmc.measure(q)


control_any_callable_demo.draw()

# %% [markdown]
# (cg-3-2)=
# ### 3.2 `Vector[Qubit]`を受け取る量子カーネル
#
# 制御化する対象の量子カーネルは引数として`Vector[Qubit]`を取ることができます。呼び出し側は長さの一致する`Vector`または`VectorView`を渡します。


# %%
@qmc.qkernel
def _vec_h(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    qs[0] = qmc.h(qs[0])
    qs[1] = qmc.h(qs[1])
    return qs


cg = qmc.control(_vec_h, num_controls=1)


# %%
@qmc.qkernel
def vec_target_demo() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(3, "qs")
    qs[0] = qmc.x(qs[0])
    qs[0], qs[1:3] = cg(qs[0], qs[1:3])
    return qmc.measure(qs)


vec_target_demo.draw()

# %% [markdown]
# (cg-3-3)=
# ### 3.3 制御化する対象量子カーネルのシグネチャ由来のデフォルト値
#
# 制御化する対象の量子カーネルが古典パラメータにPythonのデフォルト値を宣言している場合、呼び出し側ではそのkeywordを省略することも可能です。


# %%
@qmc.qkernel
def _phase(q: qmc.Qubit, theta: qmc.Float = math.pi / 2) -> qmc.Qubit:
    return qmc.rx(q, theta)


cg = qmc.control(_phase)


# %%
@qmc.qkernel
def default_arg_demo() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    c, t = cg(c, t)  # thetaはデフォルトのmath.pi / 2が入る
    return qmc.measure(t)


default_arg_demo.draw()


# %%
# 同じ`_phase`量子カーネルを、今度はsymbolicな`num_controls=n - 1`で制御化
# します。呼び出し側が`theta`を名指ししなくても`theta=math.pi / 2`の
# デフォルトはそのまま適用されます。別の角度を使いたいがkwargには
# 切り替えたくない場合は、省略した`theta`をcallsiteのpositional上書き
# (`cg(q[0 : n - 1], q[n - 1], math.pi / 4)`)に置き換えます。
@qmc.qkernel
def default_arg_demo_symbolic(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, "q")
    q[0 : n - 1] = qmc.x(q[0 : n - 1])  # 制御量子ビットを全て|1>にする
    cg = qmc.control(_phase, num_controls=n - 1)  # 制御量子ビット数をsymbolicに指定
    q[0 : n - 1], q[n - 1] = cg(q[0 : n - 1], q[n - 1])
    return qmc.measure(q)


default_arg_demo_symbolic.draw(n=3, fold_loops=False)

# %% [markdown]
# (cg-3-4)=
# ### 3.4 `power=`で`U^k`を制御
#
# `power=k`を渡すと、`U`そのものではなく*k乗*の`U^k`が制御されます。`power`はPythonの`int`(コンパイル時に解決)も`qmc.UInt`ハンドル(`bindings`からtranspile時に解決)も受け取り、`num_controls`がconcreteかsymbolicかに関係なく動作します。


# %%
cg = qmc.control(qmc.rx)  # num_controls = 1 (concrete)


# %%
@qmc.qkernel
def power_demo_concrete() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)
    c, t = cg(c, t, angle=math.pi / 4, power=3)  # powerはPythonのint
    return qmc.measure(t)


power_demo_concrete.draw()

# %% [markdown]
# (cg-3-5)=
# ### 3.5 制御引数を別々のpositionalで渡す(CCXスタイル)
#
# `num_controls=2`にすると、呼び出し側では各制御qubitをそれぞれ独立したpositional引数としてtargetの前に並べます。以下は典型的なCCX(Toffoli)で、2つの制御`c0`、`c1`と1つのtarget`t`を渡しています。同じパターンは`num_controls=3`(CCCX)や`num_controls=4`にも拡張でき、渡したい`Qubit`が`num_controls`で指定した数だけあれば成立します。


# %%
ccx = qmc.control(qmc.x, num_controls=2)


# %%
@qmc.qkernel
def toffoli_demo() -> qmc.Bit:
    c0 = qmc.qubit(name="c0")
    c1 = qmc.qubit(name="c1")
    t = qmc.qubit(name="t")
    c0 = qmc.x(c0)
    c1 = qmc.x(c1)
    c0, c1, t = ccx(c0, c1, t)
    return qmc.measure(t)


toffoli_demo.draw()

# %% [markdown]
# (cg-3-6)=
# ### 3.6 scalar Qubitと`VectorView`の制御を混ぜる
#
# positional引数で渡す制御量子ビットは、合計qubit数が`num_controls`と一致する限り、scalarな`Qubit`、`VectorView`、`Vector[Qubit]`を自由に混ぜられます。以下では`num_controls=3`のcontrolled-Hに対し、3つの制御を`qs[0]`(scalar `Qubit`、1qubit)と`qs[1:3]`(`VectorView`、2qubit)で渡しています。


# %%
cg = qmc.control(qmc.h, num_controls=3)


# %%
@qmc.qkernel
def mixed_controls_demo() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(5, "qs")
    qs[0] = qmc.x(qs[0])
    qs[1] = qmc.x(qs[1])
    qs[2] = qmc.x(qs[2])
    qs[0], qs[1:3], qs[3] = cg(qs[0], qs[1:3], qs[3])
    return qmc.measure(qs)


mixed_controls_demo.draw()

# %% [markdown]
# (cg-4)=
# ## 4. concrete専用: 単一のscalar制御
#
# 制御引数の形はほぼすべて両モードで動き([](#cg-3))、symbolic modeはさらに固有の機能を持ちます([](#cg-5))。concrete modeを*必要とする*唯一の形は、単一のscalar `Qubit`を制御にするケースです。symbolic modeでは単一の制御引数はpoolの形と解釈され`Vector` / `VectorView`が要求されます。そもそも制御数が1に固定されている制御をsymbolicにする理由がないためです。これは[](#cg-1)の最小controlled-RXとまったく同じ形で、以下のcontrolled-X(CNOT)も同じ単一scalar制御の形です。


# %%
cx = qmc.control(qmc.x)  # num_controlsはデフォルトの1(concrete)


# %%
@qmc.qkernel
def cnot_demo() -> qmc.Bit:
    c = qmc.qubit(name="c")
    t = qmc.qubit(name="t")
    c = qmc.x(c)  # 制御を|1>に立ててXを発火させる
    c, t = cx(c, t)
    return qmc.measure(t)


cnot_demo.draw()

# %% [markdown]
# (cg-5)=
# ## 5. Symbolic modeのパターン
#
# 本セクションは`num_controls`が`qmc.UInt`ハンドル(または`n - 1`のような`UInt`式)のときのパターン、symbolic modeについてみていきます。制御量子ビットの数は`qmc.control(..., num_controls=...)`の評価時ではなく、`bindings`からtranspile時に決まります。また、[](#cg-5-5)では[](#cg-3-5) / [](#cg-3-6)のsymbolic版を確認します。
#
# 制御量子ビットの渡し方としては以下の2種類があります。
#
# - **Single-poolの形**([](#cg-5-1) – [](#cg-5-4)): 制御引数として`Vector[Qubit]`または`VectorView`を1つ渡し、pool全体、もしくは`control_indices`で選んだsubsetがactiveな制御として使用されます。
# - **Multi-argの形**([](#cg-5-5)): 制御prefixが複数のpositional引数(scalar`Qubit`、`VectorView`、`Vector[Qubit]`、またはこれらの組み合わせ)で、qubit数の合計が`num_controls`と一致するよう渡します。concrete modeで見た([](#cg-3-5) / [](#cg-3-6))を、symbolicな`num_controls`に持ち上げたものです。
#
# `control_indices`keywordはsymbolic mode専用で、single-poolの引数のどの量子ビットがactiveな制御として実際に配線されるかを指定します(残りはそのまま素通りします)。`control_indices`はsingle-poolの形でのみ有効で、multi-argの形と組み合わせるとrejectされます。

# %% [markdown]
# (cg-5-1)=
# ### 5.1 `num_controls = n`でpool全体を制御に
#
# 最もシンプルなsymbolicの形として`num_controls=n`としてpool(長さ`n`)全体を制御として使います。パラメータ`n`は`bindings`からtranspile時に具体化されます。


# %%
@qmc.qkernel
def symbolic_pool(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    ctrls = qmc.qubit_array(n, "ctrls")
    tgt = qmc.qubit(name="tgt")
    ctrls = qmc.x(ctrls)  # 制御を全て|1>にする
    cg = qmc.control(qmc.x, num_controls=n)  # シンボリックなnを制御数に指定
    ctrls, tgt = cg(ctrls, tgt)
    return qmc.measure(ctrls)


symbolic_pool.draw(n=3, fold_loops=False)

# %% [markdown]
# (cg-5-2)=
# ### 5.2 `n - 1`の典型的なmulti-controlled形
#
# multi-controlled-X設計で頻出する形で、レジスタの最初の`n - 1`qubitを制御に、最後の1qubitをtargetにします。`num_controls`の値はsymbolic式の`n - 1`で、制御引数はスライス`qs[0:n - 1]`です。


# %%
@qmc.qkernel
def mcx_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(n, "qs")
    qs[0 : n - 1] = qmc.x(qs[0 : n - 1])  # 制御部分を全て|1>にする
    mcx = qmc.control(qmc.x, num_controls=n - 1)
    qs[0 : n - 1], qs[n - 1] = mcx(qs[0 : n - 1], qs[n - 1])
    return qmc.measure(qs)


mcx_demo.draw(n=4, fold_loops=False)

# %% [markdown]
# (cg-5-3)=
# ### 5.3 `control_indices`でsubsetを選ぶ
#
# 制御poolがactiveな制御数より広い場合、`control_indices`keyword(symbolic mode専用)でpoolのどの量子ビットを制御として使うかを指定します。残りの量子ビットには触りません。indexは連続である必要はありません。


# %%
@qmc.qkernel
def subset_pool(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    pool = qmc.qubit_array(n, "pool")
    tgt = qmc.qubit(name="tgt")
    pool[0] = qmc.x(pool[0])
    pool[1] = qmc.x(pool[1])
    pool[3] = qmc.x(pool[3])  # pool[2]は|0>のまま。これがinactiveな量子ビット。
    cg = qmc.control(qmc.x, num_controls=k_ctrls)
    pool, tgt = cg(pool, tgt, control_indices=[0, 1, 3])
    return qmc.measure(pool)


subset_pool.draw(n=4, k_ctrls=3)

# %% [markdown]
# (cg-5-4)=
# ### 5.4 `control_indices`に`UInt`式を含める
#
# `control_indices`の各エントリはPythonの`int`リテラル、`qmc.UInt`でも`UInt`値による算術式のいずれでも構いません。リテラル`int`エントリに対する軽い構造チェック(`bool`、負値、リテラル`int`同士の重複の拒否)はcompose時に行われますが、それ以外、すなわち`num_controls`との長さ整合、pool sizeに対する範囲、`UInt`の値解決を必要とするチェックはtranspile時、`bindings`からパラメータが具体化されてからに先送りされます。


# %%
@qmc.qkernel
def subset_pool_with_uint(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    pool = qmc.qubit_array(n, "pool")
    tgt = qmc.qubit(name="tgt")
    pool[0] = qmc.x(pool[0])
    pool[1] = qmc.x(pool[1])
    pool[3] = qmc.x(pool[3])
    cg = qmc.control(qmc.x, num_controls=k_ctrls)
    pool, tgt = cg(pool, tgt, control_indices=[0, 1, n - 1])
    return qmc.measure(pool)


subset_pool_with_uint.draw(n=4, k_ctrls=3)

# %% [markdown]
# (cg-5-5)=
# ### 5.5 Multi-argの制御prefix
#
# 制御を複数のpositional引数に分けたい場合、(典型的には「同じ`Vector`のいくつかの量子ビットをactiveな制御に、別の量子ビットをtargetにしたい」場合)symbolic modeでもconcrete modeと同じmulti-argの形([](#cg-3-5) / [](#cg-3-6))が使えます。同じ`Vector[Qubit]`から複数の量子ビットを取り出しても、互いにdisjoint(重ならない)な量子ビットであれば制御prefixに並べられます。制御prefixの各引数のqubit数の合計が、transpile時に`num_controls`と照合されます。
#
# なお、`control_indices`はmulti-argの形では使えません([](#cg-6)のreject caseを参照)。subset選択が必要ならsingle-poolの形([](#cg-5-3) / [](#cg-5-4))、multi-argの自由度が必要ならprefix全体をactiveとして使うかのどちらかを選んでください。


# %%
@qmc.qkernel
def controlled_increment_demo(
    n: qmc.UInt, control_index: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, "q")
    q[control_index] = qmc.x(q[control_index])
    n = q.shape[0]
    for k in qmc.range(n - 1):
        target_idx = n - 2 - k
        ctrl_main = q[control_index]
        prefix = q[0:target_idx]
        tgt = q[target_idx]
        cg = qmc.control(qmc.x, num_controls=target_idx + 1)
        ctrl_main, prefix, tgt = cg(ctrl_main, prefix, tgt)
        q[control_index] = ctrl_main
        q[0:target_idx] = prefix
        q[target_idx] = tgt
    return qmc.measure(q)


controlled_increment_demo.draw(n=4, control_index=3, fold_loops=False)

# %% [markdown]
# (cg-6)=
# ## 6. 動かないパターン
#
# 本章ではrejectされる呼び出し形を1つずつ見ていきます。
#
# | ケース | モード | 例外 |
# | --- | --- | --- |
# | 6.1 制御qubit数が引数境界をまたぐ | concrete | `ValueError` |
# | 6.2 concrete modeで`control_indices` | concrete | `ValueError` |
# | 6.3 concrete modeでsymbolic長の`VectorView` | concrete | `NotImplementedError` |
# | 6.4 同じpool量子ビットをtargetに再利用 | symbolic | `UnreturnedBorrowError` |
# | 6.5 multi-arg制御prefix + `control_indices` | symbolic | `ValueError` |
# | 6.6 symbolic modeで単一scalar制御 | symbolic | `ValueError` |


# %%
def expect_error(label: str, exc_type: type, body) -> None:
    """``body``が``exc_type``の例外を投げることをassertします。

    ヘルパは*想定*の例外クラスだけをcatchします。それ以外の
    例外はそのまま伝播するので、別の例外型に変わってしまうような
    regressionはnotebook上で通常のtracebackとして見えます。
    例外が一度も発生しなかった場合は``AssertionError``を投げます。
    """
    try:
        body()
    except exc_type as exc:
        print(f"[{type(exc).__name__}] {label}: {exc}")
        return
    raise AssertionError(
        f"{label}: expected {exc_type.__name__}, but no exception was raised"
    )


# %% [markdown]
# (cg-6-1)=
# ### 6.1 制御qubit数が引数境界をまたぐ (concrete)
#
# concrete modeはpositional引数を順に確認して、各引数を制御リストに畳み込むということを累計が`num_controls`に達するまで続けます。`VectorView`や`Vector`が与えられ、そこまでの累計の量子ビット数が`num_controls`を超えるような場合にはエラーが起きます。


# %%
def case_count_mismatch() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        qs = qmc.qubit_array(6, "qs")
        cg = qmc.control(qmc.x, num_controls=3)
        view, t = cg(qs[0:5], qs[5])  # 5qubit渡しているが3expected
        qs[0:5] = view
        return qmc.measure(qs[5])

    _ = kernel.block


expect_error("control count mismatch", ValueError, case_count_mismatch)

# %% [markdown]
# (cg-6-2)=
# ### 6.2 concrete modeで`control_indices` (concrete)
#
# `control_indices`は選択元となる制御*pool*がある時にだけ意味を持つsymbolic modeの専用の引数です。concreteな`num_controls`と一緒に渡すと、compose時に`ValueError`になります。


# %%
def case_control_indices_in_concrete() -> None:
    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(qmc.x)  # num_controlsはデフォルトの1 (concrete)
        c, t = cg(c, t, control_indices=[0])
        return qmc.measure(t)

    _ = kernel.block


expect_error(
    "control_indices in concrete mode",
    ValueError,
    case_control_indices_in_concrete,
)

# %% [markdown]
# (cg-6-3)=
# ### 6.3 concrete modeでsymbolic長の`VectorView` (concrete)
#
# concrete modeは各制御引数のqubit数をコンパイル時に決定する必要があります。長さが`UInt`に依存するスライスはconcrete modeでは未対応で、`NotImplementedError`になります。


# %%
def case_symbolic_view_in_concrete() -> None:
    @qmc.qkernel
    def kernel(m: qmc.UInt) -> qmc.Bit:
        qs = qmc.qubit_array(m, "qs")
        cg = qmc.control(qmc.x, num_controls=3)
        view, q_out = cg(qs[0:m], qs[m - 1])
        qs[0:m] = view
        qs[m - 1] = q_out
        return qmc.measure(qs[m - 1])

    _ = kernel.block


expect_error(
    "symbolic-length VectorView in concrete mode",
    NotImplementedError,
    case_symbolic_view_in_concrete,
)

# %% [markdown]
# (cg-6-4)=
# ### 6.4 同じpool量子ビットをtargetに再利用 — single-poolの形 (symbolic)
#
# single-poolの形(`cg(pool, ...)`に`control_indices`を組み合わせる場合)で、pool内のinactiveな量子ビットを取り出してtargetとして渡したくなることがあります。例えば`cg(pool, pool[2], control_indices=[0, 1, 3])`として`pool[2]`をcontrolled-Uのtargetにする、といった形です。この呼び出しはlinear typeのborrow trackerによってrejectされます。poolが1引数として消費されている最中に`pool[2]`が別引数として借り出されるため、compose時に`UnreturnedBorrowError`として表面化します。
#
# Workaround(推奨順):
#
# 1. **Multi-arg symbolicの形([](#cg-5-5))。** 各量子ビットまたはsub-viewを別々のpositional引数として渡します。`cg(pool[0], pool[1], pool[3], pool[2])`(またはcontrolled-incrementの例のようにscalar / sliceを混ぜる形)。各引数は`pool`からの別borrowで、borrow trackerがdisjointnessをcheckし、`num_controls`はtranspile時にqubit数の合計と照合されます。
# 2. **Concrete mode([](#cg-3-6))。** `num_controls`がPythonの`int`なら、同じmulti-argの形がsymbolicの仕組みなしでそのまま動きます。


# %%
def case_pool_slot_as_target() -> None:
    @qmc.qkernel
    def kernel(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        pool = qmc.qubit_array(n, "pool")
        cg = qmc.control(qmc.x, num_controls=k_ctrls)
        pool, q = cg(pool, pool[2], control_indices=[0, 1, 3])
        pool[2] = q
        return qmc.measure(pool)

    _ = kernel.block


expect_error(
    "same-pool slot reused as target",
    UnreturnedBorrowError,
    case_pool_slot_as_target,
)

# %% [markdown]
# (cg-6-5)=
# ### 6.5 Multi-arg制御prefix + `control_indices` (symbolic)
#
# symbolic modeの2つの機能は相互排他です。`control_indices`は単一の制御pool(`Vector`引数1つ)に対してのみ意味を持ち、複数のpositional制御引数と組み合わせるとcompose時に`ValueError`がraiseされます。


# %%
def case_multi_arg_with_control_indices() -> None:
    @qmc.qkernel
    def kernel(n: qmc.UInt, k: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, "q")
        ctrl_main = q[0]
        prefix = q[1:k]
        tgt = q[k]
        cg = qmc.control(qmc.x, num_controls=k + 1)
        ctrl_main, prefix, tgt = cg(ctrl_main, prefix, tgt, control_indices=[0, 1, 2])
        q[0] = ctrl_main
        q[1:k] = prefix
        q[k] = tgt
        return qmc.measure(q)

    _ = kernel.block


expect_error(
    "multi-arg + control_indices",
    ValueError,
    case_multi_arg_with_control_indices,
)

# %% [markdown]
# (cg-6-6)=
# ### 6.6 symbolic modeで単一scalar制御 (symbolic)
#
# 単一のscalar `Qubit`制御は、concrete modeが必要になる唯一の形です。symbolic modeでは単一の制御引数はsingle-poolの形と解釈され、`Vector` / `VectorView`が要求されます。


# %%
def case_single_scalar_control_symbolic() -> None:
    @qmc.qkernel
    def kernel(n: qmc.UInt) -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        cg = qmc.control(qmc.rx, num_controls=n)
        c, t = cg(c, t, angle=math.pi)
        return qmc.measure(t)

    _ = kernel.block


expect_error(
    "single scalar control in symbolic mode",
    ValueError,
    case_single_scalar_control_symbolic,
)

# %% [markdown]
# (cg-7)=
# ## 7. まとめ
#
# `qmc.control(fn, num_controls=...)`を使うことでQamomileのビルトインゲートやユーザ定義の量子カーネルを制御化することができます。`qmc.control`には二つのモードがあり、そのモードは`num_controls`の型で決まります。Pythonの`int`であれば*concrete mode*、`qmc.UInt`(または`n - 1`のような`UInt`式)なら*symbolic mode*です。
