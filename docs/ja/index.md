# Qamomile ドキュメントへようこそ

Qamomile は量子プログラミング SDK です。型付き Python 関数で量子回路を記述し、Qiskit・QuriParts などのバックエンドで実行できます。

> **注意**: Qamomile は現在もアクティブに開発中であり、リリース間で破壊的変更が加わる可能性があります。

> **不具合報告**: 不具合を見つけた場合は、[GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new) でお知らせいただければ幸いです。

## チュートリアル

1. [はじめての量子カーネル](tutorial/01_your_first_quantum_kernel) — カーネルの定義・可視化・実行、アフィンルール
2. [パラメータ付きカーネル](tutorial/02_parameterized_kernels) — 構造パラメータとランタイムパラメータ、バインド/スイープパターン
3. [リソース推定](tutorial/03_resource_estimation) — シンボリックなコスト分析、ゲート内訳、スケーリング分析
4. [実行モデル](tutorial/04_execution_models) — `sample()` と `run()`、オブザーバブル、ビット順序
5. [古典フローパターン](tutorial/05_classical_flow_patterns) — ループ、スパースデータ、条件分岐
6. [再利用パターン](tutorial/06_reuse_patterns) — ヘルパーカーネル、コンポジットゲート、スタブ

## インストール

```bash
pip install qamomile
```

## クイック例

```python
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

@qmc.qkernel
def bell_state() -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)

transpiler = QiskitTranspiler()
exe = transpiler.transpile(bell_state)
result = exe.sample(transpiler.executor(), shots=1000).result()

for outcome, count in result.results:
    print(f"  {outcome}: {count}")
```

## リンク

- [GitHub リポジトリ](https://github.com/Jij-Inc/Qamomile)
- [API リファレンス](api/index.md)
