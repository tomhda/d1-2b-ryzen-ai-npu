# d1-2B on the Ryzen AI NPU

Framework-free runtime for [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) on AMD Ryzen AI 300-series processors (XDNA 2). The language model's main dense projections and the complete vision neural network run on the NPU. Converted model: [Tomoali/d1-2B-ryzen-ai-npu](https://huggingface.co/Tomoali/d1-2B-ryzen-ai-npu).

[日本語](docs/ja/README.md)

d1 returns calibrated probabilities for typed questions without generating text. This runtime supports `noul` (yes/no), `choice` and `score`, with text and at most one image. It packs up to 12 questions per forward pass, uses a 4096-token context and automatically resizes input images to **640×384**. Audio, video and extraction question types are not supported.

## Requirements and installation

- Windows 11, Ryzen AI 300-series NPU and its driver.
- [Ryzen AI Software 1.8.0](https://ryzenai.docs.amd.com/). Its environment supplies ONNX Runtime with `RyzenAILightExecutionProvider` and `VitisAIExecutionProvider`.

```bash
conda create --clone ryzen-ai-1.8.0 -n d1npu2
conda activate d1npu2
git clone https://github.com/tomhda/d1-2b-ryzen-ai-npu.git
cd d1-2b-ryzen-ai-npu
pip install -e ".[download]"
python -m d1npu2 download --to C:/models/d1-2B-ryzen-ai-npu
```

The model download is about 2.4 GB. The package uses NumPy, Pillow and tokenizers; it does not require PyTorch, transformers or torchvision. Do not replace Ryzen AI's ONNX Runtime with the PyPI `onnxruntime` package. The separate module name `d1npu2` allows installation alongside the 3B runtime.

## Use

```bash
python -m d1npu2 screen-check --model C:/models/d1-2B-ryzen-ai-npu --image screenshot.png
python -m d1npu2 decide --model C:/models/d1-2B-ryzen-ai-npu --state "Please refund the duplicate charge." --questions examples/questions.json
```

For your own questions, use the JSON form in [examples/questions.json](examples/questions.json). `--state @message.txt` reads UTF-8 text. Add `--image` to a `decide` request. `--output answers.json` writes the same JSON that is printed to stdout. Provider diagnostics go to stderr. The three-question screen preset is a generic example, not an accuracy-tuned benchmark.

```python
from d1npu2 import D1

model = D1("C:/models/d1-2B-ryzen-ai-npu")
answers = model.decide("Inspect the screenshot.", {
    "login": {"type": "noul", "instructions": "Is a sign-in form visible?"}
}, image="screenshot.png")
```

Use one model instance per process and serialize requests. Native providers can fail during Python shutdown; the CLI flushes its result and exits with `os._exit`. Embedded applications should account for the same native-provider limitation.

In a standalone Python script, after writing all results, use `sys.stdout.flush(); sys.stderr.flush(); os._exit(0)` (import `sys` and `os` first) to avoid native interpreter teardown.

### First image setup

When you process an image for the first time, the application automatically generates the data needed to run the vision model on your NPU. This can take about 10 minutes. Later runs automatically reuse the saved data, so you can continue using the same commands.

The downloaded model and this generated execution data are stored separately. The default location for the generated data is `%LOCALAPPDATA%/d1npu2`. Use `--cache-dir` if you want to choose its location. Use `--ryzen-ai-dir` if Ryzen AI Software is installed outside `C:/Program Files/RyzenAI/1.8.0`.

#### If the environment changes or setup fails

The generated data depends on the NPU, its driver and Ryzen AI Software. After changing these, select an empty directory with `--cache-dir` to generate data for the new environment. Use the same option if the application reports incomplete or incompatible execution data. You do not need to download the model again. Existing generated files are retained when an error is reported.

## NPU execution and measurements

- Vision: one NPU graph, with attention/rotary decomposition and an equivalent 2048×2 projector split. Original GELU attributes are preserved.
- Language: 143 main projection kernels converted for the NPU. Dense weights are 4-bit, embedding weights 8-bit, attention gates FP32. Embeddings, language Attention/Conv and image preprocessing retain CPU work. This does not mean 100% NPU utilization.
- `--profile-dir profiles` records execution-provider placement and rejects observed vision CPU fallback or missing language NPU projection events.

On a Ryzen AI 7 PRO 350 / 32 GB / Windows 11 / Ryzen AI Software 1.8.0, a separate benchmark used two screenshots, 24 questions (two sets of 12), three passes, profiling off and four counterbalanced pairs. Both model configurations ran without PyTorch, transformers or torchvision and used the same SDK. The 3B figures apply to its framework-free configuration, not its transformers-based configuration:

| Whole-PC physical RAM increase | 2B | Framework-free 3B |
|---|---:|---:|
| After inference, median | 4.279 GB | 4.460 GB |
| During inference, median | 4.265 GB | 4.427 GB |

The change is system physical total minus available, relative to a six-second pre-launch baseline. Background applications were retained; these are whole-PC changes, not dedicated model allocations. GB means 10⁹ bytes. The 2B increase was about 0.16–0.18 GB lower in this protocol. These figures do not measure the packaged CLI's startup or first compilation.

On a separate 48-decision comparison against the same-weight CPU model, 47 decisions agreed and the largest probability difference was 0.03363. This is limited validation, not a general accuracy guarantee or a measurement of the example screen preset.

## Model layout and assembly

The Hugging Face model contains native `vision/`, converted `language/`, the tokenizer, original prompt/media/calibration/packing source, licenses and file checksums. NPU execution data is generated during use rather than included in the download. SDK DLLs must be installed separately.

[conversion/assemble.py](conversion/assemble.py) assembles **already converted** language and vision files into this layout. It is not a complete upstream-to-NPU conversion pipeline. The runtime weights preserve the converted architecture, quantization and 4096-token context.

## License

The independent wrapper is MIT licensed ([LICENSE](LICENSE), [NOTICE](NOTICE)). Model files and upstream checkpoint code are under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-2B/blob/main/LICENSE), distributed with the model. Its commercial-use limitation applies to legal entities at or above USD 10 million annual revenue; see the complete license. This project is not affiliated with Liquid AI or AMD.
