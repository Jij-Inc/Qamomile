---
slug: release-notes
---

# リリースノート

- [v0.12.7](v0_12_7) — `Dict[K, Float]`の係数を`d[key]`添字付きでruntime parameterとして扱う、`UInt`の`%`演算子、これまでsilentなmiscompileだったパターンを拒否する一連の新しいコンパイル時エラー(`QubitRebindError`、`QubitConsumedError`、`bool` / 型不一致引数への`TypeError`)
- [v0.12.6](v0_12_6) — `qmc.inverse`による回路の逆操作(ビルトインゲート・`@qkernel`・QFT/IQFT)、計算基底レジスタの算術`qmc.modular_increment` / `qmc.modular_decrement`、job型・結果型を`qamomile.circuit`からimport可能に
- [v0.12.5](v0_12_5) — Pauli Correlation Encoding用の`PCEConverter` / `PCEEncoder`、QURI Partsサンプリングへのseed指定、測定済み`Vector[Bit]`を使う条件分岐の修正、`Vector`要素の`qmc.expval`修正
- [v0.12.4](v0_12_4) — `qmc.controlled`を`qmc.control`に改名しsymbolic modeの表現力を強化、`BinaryModel.from_higher_ising`による高次Isingモデルからの構築をサポート、`Float`ハンドルへの単項マイナス`-`を追加
- [v0.12.3](v0_12_3) — Python風の`Vector`スライシング、Pauli-Hamiltonian同士の`commutator(a, b)`、`computational_basis_state`アルゴリズムヘルパー
- [v0.12.2](v0_12_2) — Möttönen振幅エンコーディング、サンプルベースの部分対角化(QSCI)、`qmc.controlled`が組み込みゲートを受け取れる、`BinaryModel`に対する`LocalSearch`、ドキュメントを`tutorial/` / `algorithm/` / `usage/` / `integration/`に再編
- [v0.12.1](v0_12_1) — 単一量子ビットゲートの`Vector[Qubit]`へのブロードキャスト、サブ`@qkernel`呼び出しでのスカラーリテラル昇格、QURI Partsの記号的パラメータ算術の修正
- [v0.12.0](v0_12_0) — Suzuki–Trotter時間発展、`qamomile.linalg`、自己再帰`@qkernel`、量子最適化コンバーターのOMMX `SampleSet`出力対応
- [v0.11.1](v0_11_1) — Python 3.11サポート
- [v0.11.0](v0_11_0) — パラメトリックVector QAOAの堅牢化、`cx_entangling_layer`、コンパイラコアの整理
- [v0.10.0](v0_10_0) — 回路プログラミング層をゼロから再構築
