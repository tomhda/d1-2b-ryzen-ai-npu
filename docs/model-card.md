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

The first image compiles the native ONNX vision graph and may take about 10 minutes. The runtime creates a machine-local EPContext; subsequent processes load it directly. No machine-specific vision cache or EPContext is distributed here. Keep the entire local cache together and regenerate it on another machine.

## Format and scope

- About 2.4 GB of model files, with checksums in `model-manifest.json`.
- 4-bit language dense projections, 8-bit embedding, FP32 attention gates; 4096-token runtime context.
- Vision: native floating-point ONNX, fixed 640×384, decomposed attention/rotary and equivalent 2048×2 projector split preserving the original GELU attributes.
- One image and text; `noul`, `choice`, `score` questions; up to 12 questions per packed forward.
- Language's main 143 projection kernels and the complete vision NN run on the NPU. Embedding, language Attention/Conv and preprocessing retain CPU work. This is not a claim of 100% NPU utilization.
- Audio, video and extraction question types are not supported by this runtime.

The original prompt, media, calibration and packing source are included. No PyTorch, transformers or torchvision import is needed. The model is a decision model, not a text-generating chat model.

## Validation

A 48-decision same-weight CPU comparison agreed on 47 decisions, with a maximum probability difference of 0.03363. This limited test is not a general accuracy guarantee or the accuracy of the example screen preset.

A separate two-screenshot / 24-question / three-pass comparison used profiling-off fresh processes in four counterbalanced pairs. Both 2B and the local 3B comparator excluded PyTorch/transformers/torchvision and used the same SDK. Whole-PC physical RAM increase after inference was 4.279 GB for 2B and 4.460 GB for 3B; during inference it was 4.265 GB and 4.427 GB. Hardware: Ryzen AI 7 PRO 350, 32 GB, Windows 11, Ryzen AI Software 1.8.0. These system total-minus-available changes include background variation, not dedicated model allocation. They do not measure this packaged CLI's startup or first compilation.

## License and attribution

Model files and upstream checkpoint source use the LFM Open License v1.0 (`LICENSE`). Converted and restructured files are identified in `NOTICE`. The model license restricts commercial use by legal entities at or above USD 10 million annual revenue; read the complete license. The independent runtime wrapper has a separate MIT license. AMD SDK DLLs are not redistributed. This project is not affiliated with Liquid AI or AMD.

## 日本語

AMD Ryzen AI 300シリーズのNPU向けd1-2Bです。Windows 11・Ryzen AI Software 1.8.0の環境で、上記GitHubの`d1npu2` runtimeを使用します。重いML frameworkは不要です。画像は1枚・640×384、contextは4096、noul/choice/scoreに対応します。最初の画像で約10分のcompileが必要な場合があり、次回からは生成したローカルEPContextを直接loadします。cache全体を保持し、別PCでは再生成してください。主な投影演算と画像NNはNPU、埋め込み・言語Attention/Conv・前処理にはCPUが残ります。モデルはLFM Open License v1.0、独自runtimeはMITです。詳細はGitHubの日本語READMEを参照してください。
