# Models

None of these ship with PixelFlux. They total roughly two gigabytes, and none of them is ours to
redistribute. Every feature they power degrades on its own: a missing model means that one feature
is absent and says so, and nothing else changes.

Put the files in `models/` — either the repository's own `models/` folder or
`%LOCALAPPDATA%\PixelFlux\models`, both of which are searched — and restart.

| File | Model | Licence | Size | Powers |
| --- | --- | --- | --- | --- |
| `face_yunet_2023mar.onnx` | YuNet | Apache-2.0 | 232 KB | Finding faces |
| `face_sface_2021dec.onnx` | SFace | Apache-2.0 | 38 MB | "Who else looks like this" |
| `yolo11n-seg.onnx` | YOLO11n-seg | **AGPL-3.0** | 11 MB | Outlining and labelling objects |
| `clip_vision_model.onnx`, `clip_text_model.onnx`, `clip_vocab.json`, `clip_merges.txt` | CLIP ViT-B/32 | MIT | ~580 MB | Search by meaning |
| `qwen3vl/` | Qwen3-VL-2B-Instruct, ONNX Runtime GenAI build | Apache-2.0 | ~1.44 GB | Writing a description of each photograph |

## Where to get them

**YuNet and SFace** come from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo), under
`models/face_detection_yunet` and `models/face_recognition_sface`. Both Apache-2.0. YuNet is small
enough to ship and is only a download because keeping every model in one place is simpler to
explain than one exception.

**YOLO11n-seg** comes from the [Ultralytics](https://github.com/ultralytics/ultralytics) assets
release, exported to ONNX. **It is AGPL-3.0**, which is the reason it is fetched rather than
bundled — see the note below.

**CLIP ViT-B/32** is the ONNX export published by the transformers.js project
([Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32)). MIT. You
need both encoders plus the two tokenizer files; PixelFlux implements the byte-pair tokenizer
itself rather than depending on a tokenizer runtime.

**Qwen3-VL-2B-Instruct** is
[onnx-community/Qwen3-VL-2B-Instruct-ONNX](https://huggingface.co/onnx-community/Qwen3-VL-2B-Instruct-ONNX),
specifically the `onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32/` folder, copied into
`models/qwen3vl/`. Apache-2.0. What made this the right model was not its benchmark scores but that
it ships an **ONNX Runtime GenAI** package — the autoregressive loop, KV cache, chat template and
image tokens are all handled by the runtime rather than hand-rolled.

## On the AGPL model

`yolo11n-seg.onnx` is AGPL-3.0. PixelFlux itself is MIT, and the two coexist because **the model is
never distributed with the software**: you download it, and it stays on your machine.

If you distribute a build of PixelFlux with that model file included, or run a modified PixelFlux
as a network service, the AGPL's terms apply to what you distribute or serve. If you are doing
either and would rather not, the segmentation stage is optional — every other feature works without
it, and the code path already treats an absent model as normal.

This is not legal advice. If it matters to you commercially, read the licence or replace the model
with a permissively licensed segmenter; the `ISegmenter` interface exists partly so that is a
contained change.

## Adding a different model

`ISegmenter`, `IFaceDetector`, `IFaceRecognizer`, `IImageTextEmbedder` and `IPhotoDescriber` are
the seams. Each reports `IsAvailable` and a `ModelVersion`, and the second one matters: it is
written beside every result, so installing a better model makes the affected work outstanding again
automatically, with no migration and no manual reset.

Anything whose output would differ belongs in that version string. The caption blend weight is part
of the embedder's, which was discovered the hard way — changing the weight without changing the
version silently reused every cached vector.

## Execution providers

`models/providers/` is where execution provider libraries go, if you have one. The stock ONNX
Runtime NuGet package has no provider compiled in at all; `python tools/fetch_runtime.py` extracts
a build that does. `pixelflux accel` reports what the runtime can see.

Whether an accelerator helps is per model and per machine — measured numbers for a Snapdragon X2
are in `ComputeBackend.MeasuredDefaults`, where face work runs on the graphics processor and
everything else does not, because everything else measured slower there.
