# Minimal DA3 → reconstruction evidence (API prompt)

One concrete, opt-in ComfyUI workflow for [img2threejs evidence v0](https://github.com/kvnloo/img2threejs/pull/16). This **does not** build or score a mesh; it exports bounded evidence for a separate, code-only procedural reconstruction loop.

## Prerequisites

- A current ComfyUI with built-in `LoadDA3Model` and `DA3Inference` nodes. See the [official DA3 guide](https://docs.comfy.org/tutorials/utility/depth-anything-3).
- Download [DA3 Base](https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/depth_anything_3_base.safetensors) into `ComfyUI/models/geometry_estimation/depth_anything_3_base.safetensors`. The example uses mono mode, resolution 504, and does not require a separate DA3 extension.
- Place **two same-size files** in `ComfyUI/input/`:
  - `reference.png`: the reference image.
  - `foreground-mask.png`: a white subject silhouette on a black background. The red channel is read as the mask; no automatic segmentation or placeholder all-foreground mask is used.

To evaluate generated reference art, change the exporter node's `reference_state` from `measured` to `generated`. Never present generated data as measured.

## Queue the API-format workflow

Start ComfyUI locally. From the ComfyUI repository root, run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from urllib.request import Request, urlopen

path = Path("custom_nodes/reconstruction_evidence/examples/da3-evidence-api.json")
prompt = json.loads(path.read_text(encoding="utf-8"))
payload = json.dumps({"prompt": prompt}).encode("utf-8")
request = Request("http://127.0.0.1:8188/prompt", data=payload,
                  headers={"Content-Type": "application/json"}, method="POST")
with urlopen(request, timeout=30) as response:
    print(response.read().decode("utf-8"))
PY
```

This returns a `prompt_id`, **not** a completed inference receipt. Wait for that prompt to finish in ComfyUI and confirm it produced `ComfyUI/output/reconstruction_evidence/<bundle-id>/manifest.json`. The bundle includes a reference PNG, *inferred* normalized 16-bit ordinal depth, the caller-supplied foreground mask, and hashes/provenance. Keep artifacts local unless cleared for sharing.

## Consumer validation and comparison

With the [img2threejs evidence experiment](https://github.com/kvnloo/img2threejs/pull/16) checked out, validate the manifest:

```bash
python3 forge/stage1_intake/reconstruction_evidence.py --manifest /path/to/ComfyUI/output/reconstruction_evidence/BUNDLE/manifest.json --json
```

Do not claim depth is metric ground truth. Before giving any external evidence a fitting-score weight, run the img2threejs receipt-only **shadow-parity** comparison under identical parameters, camera, evaluator, seed and render budgets. A positive silhouette weight is a separate flagged research experiment, not a default change.

**Scope:** this JSON is a ComfyUI `/prompt` API graph, *not* an interactive canvas/workflow export. Runtime execution and fidelity gains still require a real checkpoint, reference image, real model inference, fixed-camera renders, and paired A/B evidence.
