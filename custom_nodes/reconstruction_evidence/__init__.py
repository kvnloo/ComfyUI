from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import folder_paths


_ALLOWED_REFERENCE_STATES = ("measured", "generated")


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("._-")
    return (label[:48] or "bundle")


def _safe_model_label(value: str) -> str:
    normalized = str(value).strip().replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip("._-")
    return (label[:128] or "unspecified")


def _workflow_shape_value(value: Any, key: str | None = None) -> Any:
    if key == "class_type" and isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): _workflow_shape_value(value[k], str(k)) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        if (
            len(value) == 2
            and isinstance(value[0], (str, int))
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
        ):
            return ["link", str(value[0]), value[1]]
        return [_workflow_shape_value(item) for item in value]
    if value is None:
        return "<null>"
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, (int, float)):
        return "<number>"
    if isinstance(value, str):
        return "<string>"
    return f"<{type(value).__name__}>"


def _workflow_shape_hash(prompt: Any) -> str | None:
    if not isinstance(prompt, dict):
        return None
    shaped = _workflow_shape_value(prompt)
    payload = json.dumps(shaped, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _single_image(images: Any, field: str) -> np.ndarray:
    try:
        array = images.detach().cpu().float().numpy()
    except AttributeError as exc:
        raise ValueError(f"{field} must be a tensor-like IMAGE") from exc
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[-1] < 3:
        raise ValueError(f"{field} must contain exactly one HxWx3 IMAGE")
    return np.clip(array[0, :, :, :3], 0.0, 1.0)


def _single_mask(mask: Any, field: str) -> np.ndarray:
    try:
        array = mask.detach().cpu().float().numpy()
    except AttributeError as exc:
        raise ValueError(f"{field} must be a tensor-like MASK") from exc
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 4 and array.shape[0] == 1 and array.shape[-1] == 1:
        array = array[0, :, :, 0]
    if array.ndim != 2:
        raise ValueError(f"{field} must contain exactly one HxW MASK")
    return np.clip(array, 0.0, 1.0)


def _single_depth(geometry: Any) -> tuple[np.ndarray, dict[str, float]]:
    if not isinstance(geometry, dict) or "depth" not in geometry or "image" not in geometry:
        raise ValueError("da3_geometry must contain depth and image")
    try:
        depth = geometry["depth"].detach().cpu().float().numpy()
    except AttributeError as exc:
        raise ValueError("da3_geometry.depth must be a tensor") from exc
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError("v0 evidence bundles require exactly one DA3 depth frame")
    depth = depth[0]
    if not np.isfinite(depth).all():
        raise ValueError("da3_geometry.depth contains non-finite values")
    minimum = float(depth.min())
    maximum = float(depth.max())
    if maximum > minimum:
        normalized = (depth - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(depth, dtype=np.float32)
    return np.clip(normalized, 0.0, 1.0), {"min": minimum, "max": maximum}


def _png_rgb(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB").save(
        buffer, format="PNG", compress_level=4
    )
    return buffer.getvalue()


def _png_mask(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="L").save(
        buffer, format="PNG", compress_level=4
    )
    return buffer.getvalue()


def _png_depth_u16(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.rint(array * 65535.0).astype(np.uint16), mode="I;16").save(
        buffer, format="PNG", compress_level=4
    )
    return buffer.getvalue()


def _artifact(
    role: str,
    state: str,
    relative_path: str,
    payload: bytes,
    width: int,
    height: int,
    encoding: str,
    provenance: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "role": role,
        "state": state,
        "relativePath": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dimensions": {"width": int(width), "height": int(height)},
        "encoding": encoding,
        "provenance": provenance,
    }
    item.update(extra)
    return item


class ReconstructionEvidenceBundle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "da3_geometry": ("DA3_GEOMETRY",),
                "foreground_mask": ("MASK",),
                "reference_state": (_ALLOWED_REFERENCE_STATES,),
                "bundle_label": ("STRING", {"default": "reference"}),
                "model_label": ("STRING", {"default": "unspecified"}),
            },
            "optional": {
                "material_mask": ("MASK",),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_path",)
    FUNCTION = "save_bundle"
    OUTPUT_NODE = True
    CATEGORY = "image/geometry estimation"
    DESCRIPTION = (
        "Writes a bounded, provenance-labelled reconstruction evidence bundle for external evaluators. "
        "Raw DA3 depth is monotonically normalized to 16-bit PNG for ordinal use; no prompt text is persisted."
    )

    def save_bundle(
        self,
        da3_geometry,
        foreground_mask,
        reference_state="measured",
        bundle_label="reference",
        model_label="unspecified",
        material_mask=None,
        prompt=None,
    ):
        if reference_state not in _ALLOWED_REFERENCE_STATES:
            raise ValueError(f"unsupported reference_state: {reference_state}")

        reference = _single_image(da3_geometry["image"], "da3_geometry.image")
        depth, source_range = _single_depth(da3_geometry)
        foreground = _single_mask(foreground_mask, "foreground_mask")
        material = _single_mask(material_mask, "material_mask") if material_mask is not None else None

        height, width = reference.shape[:2]
        expected_shape = (height, width)
        for field, array in (("depth", depth), ("foreground_mask", foreground), ("material_mask", material)):
            if array is not None and array.shape != expected_shape:
                raise ValueError(
                    f"{field} dimensions {array.shape[1]}x{array.shape[0]} do not match "
                    f"reference dimensions {width}x{height}; v0 does not resample evidence"
                )

        reference_png = _png_rgb(reference)
        depth_png = _png_depth_u16(depth)
        foreground_png = _png_mask(foreground)
        material_png = _png_mask(material) if material is not None else None

        workflow_shape_hash = _workflow_shape_hash(prompt)
        model_label = _safe_model_label(model_label)
        digest = hashlib.sha256()
        digest.update(model_label.encode("utf-8"))
        for payload in (reference_png, depth_png, foreground_png, material_png or b""):
            digest.update(payload)
        if workflow_shape_hash:
            digest.update(workflow_shape_hash.encode("ascii"))

        bundle_id = f"{_safe_label(bundle_label)}-{digest.hexdigest()[:16]}"
        output_root = Path(folder_paths.get_output_directory()).resolve()
        bundle_root = output_root / "reconstruction_evidence" / bundle_id
        bundle_root.mkdir(parents=True, exist_ok=True)

        payloads = {
            "reference.png": reference_png,
            "depth.png": depth_png,
            "foreground-mask.png": foreground_png,
        }
        if material_png is not None:
            payloads["material-mask-00.png"] = material_png
        for name, payload in payloads.items():
            (bundle_root / name).write_bytes(payload)

        depth_provenance = {
            "sourceType": "DA3_GEOMETRY",
            "mode": str(da3_geometry.get("mode", "unknown")),
            "confidenceAvailable": "confidence" in da3_geometry,
            "skyMaskAvailable": "sky" in da3_geometry,
        }
        artifacts = [
            _artifact(
                "reference",
                reference_state,
                "reference.png",
                reference_png,
                width,
                height,
                "rgb8-png",
                {"sourceType": "DA3_GEOMETRY.image"},
            ),
            _artifact(
                "depth",
                "inferred",
                "depth.png",
                depth_png,
                width,
                height,
                "relative-linear-u16-png",
                depth_provenance,
                sourceRange=source_range,
                ordering="lower-value-nearer",
            ),
            _artifact(
                "foreground-mask",
                "inferred",
                "foreground-mask.png",
                foreground_png,
                width,
                height,
                "mask-u8-png-white-is-1",
                {"sourceType": "MASK"},
            ),
        ]
        if material_png is not None:
            artifacts.append(
                _artifact(
                    "material-mask",
                    "inferred",
                    "material-mask-00.png",
                    material_png,
                    width,
                    height,
                    "mask-u8-png-white-is-1",
                    {"sourceType": "MASK"},
                    index=0,
                )
            )

        producer: dict[str, Any] = {
            "runtime": "ComfyUI",
            "node": "ReconstructionEvidenceBundle",
            "modelLabel": model_label,
        }
        if workflow_shape_hash:
            producer["workflowShapeSha256"] = workflow_shape_hash

        manifest = {
            "schemaVersion": 1,
            "kind": "reconstruction-evidence",
            "bundleId": bundle_id,
            "producer": producer,
            "artifacts": artifacts,
            "limitations": [
                "Depth is inferred evidence and is normalized for ordinal comparison; metric scale is not preserved.",
                "Masks are external evidence and do not grant authority to rewrite authored geometry or material state.",
            ],
        }
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        relative_manifest = manifest_path.relative_to(output_root).as_posix()
        return {
            "ui": {"text": [relative_manifest]},
            "result": (relative_manifest,),
        }


NODE_CLASS_MAPPINGS = {
    "ReconstructionEvidenceBundle": ReconstructionEvidenceBundle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReconstructionEvidenceBundle": "Save Reconstruction Evidence Bundle",
}
