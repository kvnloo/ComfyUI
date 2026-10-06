import hashlib
import json
from pathlib import Path

import torch

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
