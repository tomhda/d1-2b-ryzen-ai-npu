# Ryzen AI NPUで動くd1-2B

[LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B)を、AMD Ryzen AI 300シリーズのNPUで実行します。言語モデルの主な投影演算と画像NN全体をNPU化し、PyTorch・transformers・torchvisionを読み込みません。

必要環境はWindows 11、NPU driver、Ryzen AI Software 1.8.0です。SDKのONNX Runtimeを使い、PyPI版で置き換えないでください。

```bash
conda create --clone ryzen-ai-1.8.0 -n d1npu2
conda activate d1npu2
git clone https://github.com/tomhda/d1-2b-ryzen-ai-npu.git
cd d1-2b-ryzen-ai-npu
pip install -e ".[download]"
python -m d1npu2 download --to C:/models/d1-2B-ryzen-ai-npu
python -m d1npu2 screen-check --model C:/models/d1-2B-ryzen-ai-npu --image screenshot.png
```

モデルは約2.4GBです。`noul`・`choice`・`score`に対応し、1回のforwardへ最大12問をまとめます。contextは4096、画像は1枚・640×384固定です。音声・動画・抽出型質問には対応していません。独自の質問は`examples/questions.json`を参考にし、`decide --state @message.txt --questions questions.json`で指定します。画像は`--image`、結果ファイルは`--output`、配置検証は`--profile-dir`で指定できます。

最初の画像ではnative ONNXからNPU向けにcompileし、約10分かかる場合があります。生成先は標準で`%LOCALAPPDATA%/d1npu2`、変更する場合は`--cache-dir`を使います。次のプロセスからはローカルEPContextを直接loadします。EPContextのONNXだけでなく、外部bin・compiler artifactsを含むcache全体が必要です。個人環境のpathを含むため、別PCへコピーせず再生成してください。不完全・重み不一致のcacheは削除せずエラーにします。

画像は単一NPU graph、言語の主な投影143個はNPUです。埋め込み、言語Attention/Conv、画像前処理にはCPU処理が残ります。NPU使用率100%を意味しません。1プロセスで1モデルを使用し、要求は直列に処理してください。CLIはnative providerの終了処理の問題を避けるため、結果をflush後にプロセスを終了します。

別の24判定ベンチマークでは、PC全体の物理RAM増分は2B 4.279GB・軽量3B 4.460GB、推論中は2B 4.265GB・3B 4.427GBでした。Ryzen AI 7 PRO 350 / 32GB / Windows 11 / SDK 1.8.0、同じ2画像×3周を順序反転4ペアで測定しています。背景appを維持したtotal−availableの増分で、モデル専有量や本CLIの起動・初回compileの実測値ではありません。同一重みCPUとの別比較は48判定中47一致、最大確率差0.03363です。サンプル3問presetの精度評価ではありません。

独自wrapperはMIT、モデルと元checkpointのコードはLFM Open License v1.0です。年商1,000万米ドル以上の法人の商用利用には同licenseの制限があります。全文はモデル付属のLICENSEを確認してください。[English README](../../README.md)
