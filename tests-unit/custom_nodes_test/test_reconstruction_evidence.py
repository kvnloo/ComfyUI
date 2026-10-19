import hashlib
import json
from pathlib import Path

import torch
import pytest

from custom_nodes.reconstruction_evidence import (
    ReconstructionEvidenceBundle,
    _safe_label,
    _safe_model_label,
    _workflow_shape_hash,
)


def test_safe_label_cannot_create_paths():
    assert _safe_label("../../weird / name") == "weird-name"


def test_model_label_drops_local_paths():
    assert _safe_model_label("/home/user/models/da3-base.safetensors") == "da3-base.safetensors"
    assert _safe_model_label(r"C:\\models\\da3-base.safetensors") == "da3-base.safetensors"


def test_workflow_shape_hash_ignores_literal_prompt_values():
    left = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "TOP-SECRET-A", "clip": ["2", 0]},
        }
    }
    right = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "TOTALLY-DIFFERENT", "clip": ["2", 0]},
        }
    }

    assert _workflow_shape_hash(left) == _workflow_shape_hash(right)


def test_bundle_writes_relative_hashed_artifacts_without_prompt_text(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "custom_nodes.reconstruction_evidence.folder_paths.get_output_directory",
        lambda: str(tmp_path),
    )
    geometry = {
        "depth": torch.tensor([[[1.0, 2.0], [3.0, 5.0]]]),
        "image": torch.tensor(
            [[
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            ]]
        ),
        "mode": "mono",
        "confidence": torch.ones((1, 2, 2)),
    }
    foreground = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
    material = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    prompt = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "TOP-SECRET-PROMPT", "clip": ["2", 0]},
        }
    }

    result = ReconstructionEvidenceBundle().save_bundle(
        geometry,
        foreground,
        reference_state="measured",
        bundle_label="../chair reference",
        model_label="/home/user/models/da3-base.safetensors",
        material_mask=material,
        prompt=prompt,
    )

    relative_manifest = result["result"][0]
    assert not Path(relative_manifest).is_absolute()
    manifest_path = tmp_path / relative_manifest
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert "TOP-SECRET-PROMPT" not in manifest_text
    assert manifest["kind"] == "reconstruction-evidence"
    assert manifest["schemaVersion"] == 1
    assert manifest["producer"]["workflowShapeSha256"] == _workflow_shape_hash(prompt)
    assert manifest["producer"]["modelLabel"] == "da3-base.safetensors"
    assert "/home/user/models" not in manifest_text

    by_role = {}
    for artifact in manifest["artifacts"]:
        by_role.setdefault(artifact["role"], []).append(artifact)
        artifact_path = manifest_path.parent / artifact["relativePath"]
        assert artifact_path.is_file()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]
        assert artifact["dimensions"] == {"width": 2, "height": 2}

    assert by_role["reference"][0]["state"] == "measured"
    assert by_role["depth"][0]["state"] == "inferred"
    assert by_role["depth"][0]["sourceRange"] == {"min": 1.0, "max": 5.0}
    assert by_role["depth"][0]["ordering"] == "lower-value-nearer"
    assert by_role["foreground-mask"][0]["encoding"] == "mask-u8-png-white-is-1"
    assert len(by_role["material-mask"]) == 1


def test_bundle_rejects_dimension_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "custom_nodes.reconstruction_evidence.folder_paths.get_output_directory",
        lambda: str(tmp_path),
    )
    geometry = {
        "depth": torch.ones((1, 2, 2)),
        "image": torch.ones((1, 2, 2, 3)),
        "mode": "mono",
    }
    foreground = torch.ones((1, 3, 2))

    try:
        ReconstructionEvidenceBundle().save_bundle(geometry, foreground)
    except ValueError as exc:
        assert "do not match" in str(exc)
    else:
        raise AssertionError("expected dimension mismatch to fail")


@pytest.mark.parametrize("field", ["image", "foreground_mask", "material_mask"])
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_bundle_rejects_non_finite_image_and_masks(tmp_path, monkeypatch, field, non_finite):
    monkeypatch.setattr(
        "custom_nodes.reconstruction_evidence.folder_paths.get_output_directory",
        lambda: str(tmp_path),
    )
    geometry = {
        "depth": torch.ones((1, 2, 2)),
        "image": torch.ones((1, 2, 2, 3)),
        "mode": "mono",
    }
    foreground = torch.ones((1, 2, 2))
    material = torch.ones((1, 2, 2))
    if field == "image":
        geometry["image"][0, 0, 0, 0] = non_finite
    elif field == "foreground_mask":
        foreground[0, 0, 0] = non_finite
    else:
        material[0, 0, 0] = non_finite

    with pytest.raises(ValueError, match=f"{field} contains non-finite values"):
        ReconstructionEvidenceBundle().save_bundle(
            geometry, foreground, material_mask=material
        )
    assert not (tmp_path / "reconstruction_evidence").exists()


def test_bundle_rejects_non_finite_depth(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "custom_nodes.reconstruction_evidence.folder_paths.get_output_directory",
        lambda: str(tmp_path),
    )
    geometry = {
        "depth": torch.tensor([[[1.0, float("nan")], [2.0, 3.0]]]),
        "image": torch.ones((1, 2, 2, 3)),
        "mode": "mono",
    }
    foreground = torch.ones((1, 2, 2))

    try:
        ReconstructionEvidenceBundle().save_bundle(geometry, foreground)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("expected non-finite depth to fail")


def test_da3_api_example_connects_real_model_and_explicit_foreground_mask():
    root = Path(__file__).resolve().parents[2]
    example = root / "custom_nodes/reconstruction_evidence/examples/da3-evidence-api.json"
    graph = json.loads(example.read_text(encoding="utf-8"))

    assert graph["1"]["class_type"] == "LoadImage"
    assert graph["2"]["class_type"] == "LoadImage"
    assert graph["2"]["inputs"]["image"] == "foreground-mask.png"
    assert graph["3"]["class_type"] == "ImageToMask"
    assert graph["3"]["inputs"] == {"image": ["2", 0], "channel": "red"}
    assert graph["4"]["class_type"] == "LoadDA3Model"
    assert graph["4"]["inputs"]["model_name"] == "depth_anything_3_base.safetensors"
    assert graph["5"]["class_type"] == "DA3Inference"
    assert graph["5"]["inputs"]["da3_model"] == ["4", 0]
    assert graph["5"]["inputs"]["image"] == ["1", 0]
    assert graph["5"]["inputs"]["mode"] == {"mode": "mono"}
    assert graph["6"]["class_type"] == "ReconstructionEvidenceBundle"
    assert graph["6"]["inputs"]["da3_geometry"] == ["5", 0]
    assert graph["6"]["inputs"]["foreground_mask"] == ["3", 0]
    assert graph["6"]["inputs"]["model_label"] == graph["4"]["inputs"]["model_name"]
    assert all(node["class_type"] != "SolidMask" for node in graph.values())
