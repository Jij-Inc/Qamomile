---
slug: integration
---

# インテグレーション

Qamomileを外部ライブラリや量子プラットフォームと組み合わせて使うときのノートです。

::::{grid} 1 1 1 1

:::{card}
:header: **Amazon Braketサポート**
:link: braket_support
Braketネイティブ回路へトランスパイルし、ローカルまたはAWS上で実行します。
:::

:::{card}
:header: **CUDA-Qサポート**
:link: cudaq_support
MaxCut QAOAをCUDA-Qへトランスパイルし、サンプリングと期待値計算を実行します。
:::

:::{card}
:header: **OMMX Quantum Benchmarksの活用: Qamomileによる量子アルゴリズムの実装とベンチマーク**
:link: ommx_quantum_benchmarks_qaoa
OMMX Quantum Benchmarksから取得したLABSインスタンス上でQAOAを動かし、SCIPと比較します。
:::

:::{card}
:header: **qBraid サポート**
:link: qbraid_executor
qBraid対応デバイスでQiskit回路を実行します。
:::

:::{card}
:header: **Qiskit サポート**
:link: qiskit_support
Qiskitへトランスパイルし、ローカルシミュレータで実行し、Qiskitネイティブ回路機能を確認します。
:::

:::{card}
:header: **QURI Parts サポート**
:link: quri_parts_support
QURI Partsへトランスパイルし、Qulacs状態ベクトルシミュレータで実行します。
:::

::::

## HUGRで量子整数を測定する

`hugr` extraをインストールすると、`HugrTranspiler`で幅が0〜64ビットに確定したレジスタの`qmc.measure(qmc.cast(register, qmc.QInt))`を扱えます。QIntレジスタは、量子カーネルを直接呼び出す際の引数や戻り値にも使えます。`HugrTranspiler().transpile(kernel, bindings=...)`で`run()`と`sample()`を備えた`HugrExecutable`を取得し、`HugrExecutor(target="selene")`でローカル実行します。HUGRグラフと入出力情報を`CompiledProgram`として取得する場合は、`compile()`を使います。

測定はレジスタを消費して`UInt`を返します。ビット0が最下位ビットで、整数値は`bit[i] * 2**i`の総和です。スライスでは先頭の量子ビットがビット0になります。`run()`はPythonの`int`を返し、`sample()`は復元した整数の出現回数を集計します。対応するタプルや辞書の戻り値でも整数型を維持します。空のレジスタの値は`0`です。整数への復元にはHUGRグラフ内の整数演算を使い、`Float`への変換を介さずにビット63や`2**64 - 1`を含む64ビットすべてを保持します。

レジスタ幅を決める引数は、トランスパイル時の`bindings`で指定してください。`bindings`の指定前にプログラムをシリアライズすることもできます。未確定の幅と引数の対応は量子カーネルの直接呼び出しを通じて保持され、復元したプログラムのトランスパイル時に幅が確定します。

:::{note}
現在のHUGRの`UInt`は64ビットのため、`compile()`と`transpile()`は幅が未確定のQIntや65ビット以上のQIntに対して`EmitError`を送出します。幅が未確定であることと、空のレジスタであることは異なります。この制限はHUGR固有であり、一般のQIntの整数復元や他のエンジンに同じ制限が加わるわけではありません。64量子ビットのグラフをトランスパイルできても、ローカルの状態ベクトルシミュレータで実行するには大きなメモリが必要になる場合があります。

実行時の条件分岐をまたいでQIntレジスタを合流させる処理には現在対応しておらず、`EmitError`を送出します。`bindings`によってトランスパイル時に分岐が確定する場合は対応しています。

QIntを`control()`や`select()`の対象として渡すことには対応しておらず、入力ハンドルを消費する前に`TypeError`を送出します。QInt引数は量子カーネルの直接呼び出しで渡してください。
:::
