# Ryzen AI NPUで動くd1-2B

[LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B)を、AMD Ryzen AI 300シリーズのNPUで実行できます。言語モデルの主な投影演算と画像のニューラルネットワーク全体をNPUで処理し、PyTorch・transformers・torchvisionを使いません。

## インストールと実行

必要な環境はWindows 11、NPUドライバー、Ryzen AI Software 1.8.0です。ONNX RuntimeはRyzen AI Softwareに含まれるものを使い、PyPI版で置き換えないでください。

```bash
conda create --clone ryzen-ai-1.8.0 -n d1npu2
conda activate d1npu2
git clone https://github.com/tomhda/d1-2b-ryzen-ai-npu.git
cd d1-2b-ryzen-ai-npu
pip install -e ".[download]"
python -m d1npu2 download --to C:/models/d1-2B-ryzen-ai-npu
python -m d1npu2 screen-check --model C:/models/d1-2B-ryzen-ai-npu --image screenshot.png
```

モデルのダウンロード容量は約2.4GBです。初めて画像を処理すると、お使いのNPUに必要な実行用データをPC上に自動生成します。約10分かかる場合がありますが、以降の実行では保存済みのデータを自動的に再利用します。そのまま同じコマンドで利用を続けられます。

## 質問と画像の指定

`noul`（はい・いいえ）、`choice`（選択式）、`score`（段階評価）の質問に対応します。1回の推論に最大12問をまとめ、入力長の上限は4096トークンです。画像は1枚まで指定でき、640×384へ自動でサイズを調整します。音声・動画・抽出型の質問には対応していません。

独自の質問は`examples/questions.json`を参考に作成し、次のように指定します。

```bash
python -m d1npu2 decide --model C:/models/d1-2B-ryzen-ai-npu --state @message.txt --questions questions.json
```

`message.txt`はUTF-8で保存してください。画像を加える場合は`--image screenshot.png`、結果をファイルに保存する場合は`--output answers.json`を指定します。NPUへの処理割り当てを確認する場合は`--profile-dir profiles`を指定できます。

## 実行用データの保存先と再生成

初回に生成する実行用データは、ダウンロードしたモデルとは別に、標準で`%LOCALAPPDATA%/d1npu2`へ保存されます。保存先を変える場合は`--cache-dir`を指定してください。

NPU本体・ドライバー・Ryzen AI Softwareを変更した場合は、`--cache-dir`で空の保存先を指定すると、その環境に合った実行用データを生成できます。生成データが不完全、または互換性がないというエラーが表示された場合も同じ操作で再生成してください。モデルを再ダウンロードする必要はありません。エラー時に既存の生成データを自動削除することはありません。

## 処理範囲と実測

画像のニューラルネットワーク全体と言語モデルの主な投影演算143個をNPUで処理します。埋め込み、言語モデルの注意機構・畳み込み、画像の前処理にはCPU処理が残ります。NPU使用率100%を意味するものではありません。

1プロセスで1モデルを使用し、複数の要求は順番に処理してください。実行エンジンの終了処理で問題が起こる場合があるため、コマンドは結果をすべて書き出してからプロセスを終了します。Pythonから直接利用する場合の終了方法は[英語版の使用例](../../README.md#use)を参照してください。

24判定の比較試験では、PC全体の物理RAM増分は推論後が2B 4.279GB・3B 4.460GB、推論中が2B 4.265GB・3B 4.427GBでした。測定環境はRyzen AI 7 PRO 350、32GB、Windows 11、Ryzen AI Software 1.8.0です。同じ2画像を3周処理し、実行順を反転させた4ペアを測定しました。両方ともPyTorch・transformers・torchvisionを使わない構成です。

数値は、物理メモリ総量から利用可能量を引いた値の、実行前6秒間の中央値からの増分です。ほかのアプリケーションを起動したまま測定したPC全体の変化であり、モデル専有量や、このコマンドの起動・初回生成にかかる使用量ではありません。GBは10億バイトです。

同一重みのCPUモデルとの別比較では、48判定中47判定が一致し、最大確率差は0.03363でした。限られた試験での結果であり、サンプルの3問設定の精度を評価した数値ではありません。

## ライセンス

この実行プログラムはMIT、モデルと元モデルのコードはLFM Open License v1.0です。年商1,000万米ドル以上の法人の商用利用には同ライセンスの制限があります。全文はモデル付属のLICENSEを確認してください。[英語版](../../README.md)
