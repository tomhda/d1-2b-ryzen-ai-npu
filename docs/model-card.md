---
library_name: onnxruntime
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model:
- LiquidAI/d1-2B
- LiquidAI/d1-2B-ONNX
base_model_relation: quantized
tags:
- liquid
- lfm3
- onnx
- ryzen-ai
- npu
- decision
language:
- en
- ja
---

# d1-2B for AMD Ryzen AI NPU

Converted [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) for Ryzen AI 300-series (XDNA 2), using Ryzen AI Software 1.8.0 on Windows 11. The framework-free runtime and instructions are in [tomhda/d1-2b-ryzen-ai-npu](https://github.com/tomhda/d1-2b-ryzen-ai-npu).

## Use

Install the runtime into a clone of the Ryzen AI Software environment. Do not replace its ONNX Runtime with the PyPI package.

```bash
conda create --clone ryzen-ai-1.8.0 -n d1npu2
conda activate d1npu2
git clone https://github.com/tomhda/d1-2b-ryzen-ai-npu.git
cd d1-2b-ryzen-ai-npu
pip install -e ".[download]"
python -m d1npu2 download --to C:/models/d1-2B-ryzen-ai-npu
python -m d1npu2 screen-check --model C:/models/d1-2B-ryzen-ai-npu --image screenshot.png
```

When you process an image for the first time, the application automatically generates the execution data needed for your NPU. This can take about 10 minutes. Later runs automatically reuse the saved data. The downloaded model and the generated execution data are stored separately; no additional setup is needed for normal use. See the GitHub instructions if you need to change the storage location or regenerate data after an environment change.

## Format and scope

- About 2.4 GB of model files, with checksums in `model-manifest.json`.
- 4-bit language dense projections, 8-bit embedding, FP32 attention gates; 4096-token runtime context.
- Vision: floating-point ONNX with automatic input-image resizing to 640×384, decomposed attention/rotary and an equivalent 2048×2 projector split preserving the original GELU attributes.
- One image and text; `noul`, `choice`, `score` questions; up to 12 questions per packed forward.
- Language's main 143 projection kernels and the complete vision NN run on the NPU. Embedding, language Attention/Conv and preprocessing retain CPU work. This is not a claim of 100% NPU utilization.
- Audio, video and extraction question types are not supported by this runtime.

The original prompt, media, calibration and packing source are included. No PyTorch, transformers or torchvision import is needed. The model is a decision model, not a text-generating chat model.

## Validation

A 48-decision same-weight CPU comparison agreed on 47 decisions, with a maximum probability difference of 0.03363. This limited test is not a general accuracy guarantee or the accuracy of the example screen preset.

A separate two-screenshot / 24-question / three-pass comparison used profiling-off fresh processes in four counterbalanced pairs. Both the 2B and 3B configurations ran without PyTorch, transformers or torchvision and used the same SDK. Whole-PC physical RAM increase after inference was 4.279 GB for 2B and 4.460 GB for 3B; during inference it was 4.265 GB and 4.427 GB. Hardware: Ryzen AI 7 PRO 350, 32 GB, Windows 11, Ryzen AI Software 1.8.0. These system total-minus-available changes include background variation, not dedicated model allocation. They do not measure this packaged CLI's startup or first compilation.

## License and attribution

Model files and upstream checkpoint source use the LFM Open License v1.0 (`LICENSE`). Converted and restructured files are identified in `NOTICE`. The model license restricts commercial use by legal entities at or above USD 10 million annual revenue; read the complete license. The independent runtime wrapper has a separate MIT license. AMD SDK DLLs are not redistributed. This project is not affiliated with Liquid AI or AMD.

## 日本語

AMD Ryzen AI 300シリーズのNPUでd1-2Bを利用できます。Windows 11とRyzen AI Software 1.8.0を用意し、上記の手順で実行プログラムとモデルをインストールしてください。PyTorch・transformers・torchvisionは不要です。

初めて画像を処理すると、お使いのNPUに必要な実行用データをPC上に自動生成します。初回は約10分かかる場合があります。このデータはモデルとは別に保存され、以降の実行では自動的に再利用されます。

画像は1枚まで指定でき、640×384へ自動でサイズを調整します。入力長の上限は4096トークンで、`noul`（はい・いいえ）・`choice`（選択式）・`score`（段階評価）の質問に対応します。主な投影演算と画像のニューラルネットワーク全体をNPUで処理し、埋め込み、言語モデルの注意機構・畳み込み、画像の前処理にはCPU処理が残ります。

モデルはLFM Open License v1.0、実行プログラムはMITです。詳しい使い方は[日本語の説明](https://github.com/tomhda/d1-2b-ryzen-ai-npu/blob/main/docs/ja/README.md)を参照してください。
