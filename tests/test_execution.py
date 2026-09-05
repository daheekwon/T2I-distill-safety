from __future__ import annotations

from pathlib import Path

from t2i_distill.config import load_config
from t2i_distill.execution import audit_labels, audit_manifest, split_manifest
from t2i_distill.io import write_csv, write_jsonl
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS


def _job(idx: int, *, model_id: str = "sdxl_base", device: str = "cuda:2") -> dict[str, object]:
    return {
        "job_id": f"job-{idx}",
        "model_id": model_id,
        "display_name": model_id,
        "family": "sdxl",
        "runner": "diffusers_sdxl_base",
        "hf_model": "model",
        "base_model": "",
        "checkpoint": "",
        "num_inference_steps": 1,
        "guidance_scale": 1.0,
        "device": device,
        "benchmark": "grade",
        "condition_type": "broad",
        "image_prompt_uid": f"prompt-{idx}",
        "eval_uids": [f"eval-{idx}"],
        "seed": idx,
        "prompt_text": "a simple object",
        "output_path": str(Path("outputs") / f"image-{idx}.png"),
    }


def test_split_manifest_and_audit_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_job(idx) for idx in range(5)])

    index = split_manifest(manifest, tmp_path / "shards", jobs_per_shard=2)
    assert index["shard_count"] == 3
    assert [shard["jobs"] for shard in index["shards"]] == [2, 2, 1]

    audit = audit_manifest(manifest, expected_device="cuda:2")
    assert audit["jobs"] == 5
    assert audit["jobs_by_device"] == {"cuda:2": 5}
    assert audit["duplicate_job_ids"] == 0
    assert audit["missing_required_fields"] == {}


def _label_row(label_uid: str, model_id: str, *, pass_score: str, quality_score: str = "4") -> dict[str, object]:
    row = {field: "" for field in LABEL_TEMPLATE_FIELDS}
    row.update(
        {
            "label_uid": label_uid,
            "job_id": f"job-{label_uid}",
            "eval_uid": "eval-a",
            "model_id": model_id,
            "benchmark": "dimcim",
            "axis_id": "dimcim/chair/color",
            "axis_name": "chair:color",
            "condition_type": "explicit",
            "complexity_level": "atomic",
            "image_prompt_uid": "prompt-a",
            "seed": "0",
            "prompt_text": "a red chair",
            "image_path": "outputs/image.png",
            "target_behavior": "red",
            "candidate_behaviors": ["red", "blue"],
            "pass_score": pass_score,
            "quality_score": quality_score,
            "evaluator": "test",
        }
    )
    return row


def test_audit_labels_required_signals_and_pair_coverage(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    write_csv(
        labels,
        [
            _label_row("a", "sdxl_base", pass_score="1"),
            _label_row("b", "sdxl_lightning_4step", pass_score="1"),
            _label_row("c", "sd35_large", pass_score=""),
        ],
        LABEL_TEMPLATE_FIELDS,
    )
    config = load_config(Path("config/experiment.json"))
    audit = audit_labels(labels, config=config)
    assert audit["rows"] == 3
    assert audit["filled_rows"] == 2
    assert audit["missing_required_signal_by_column"] == {"pass_score": 1}
    assert audit["pair_coverage"]["sdxl_base__sdxl_lightning_4step"]["matched_labeled_cells"] == 1



def test_split_manifest_respects_model_boundaries(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            _job(0, model_id="model-a"),
            _job(1, model_id="model-a"),
            _job(2, model_id="model-b"),
            _job(3, model_id="model-b"),
        ],
    )
    index = split_manifest(manifest, tmp_path / "boundary_shards", jobs_per_shard=10, respect_model_boundaries=True)
    assert index["respect_model_boundaries"] is True
    assert index["shard_count"] == 2
    assert index["shards"][0]["jobs_by_model"] == {"model-a": 2}
    assert index["shards"][1]["jobs_by_model"] == {"model-b": 2}
