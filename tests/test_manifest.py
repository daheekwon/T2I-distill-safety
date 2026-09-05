from __future__ import annotations

from pathlib import Path

from t2i_distill.config import load_config
from t2i_distill.io import read_jsonl, write_jsonl
from t2i_distill.manifest import build_generation_manifest, build_label_template


def test_manifest_uses_cuda_2_and_dedupes_prompt_images(tmp_path: Path) -> None:
    config = load_config(Path("config/experiment.json"))
    bank = tmp_path / "bank.jsonl"
    write_jsonl(
        bank,
        [
            {
                "eval_uid": "eval-a",
                "image_prompt_uid": "prompt-a",
                "benchmark": "grade",
                "condition_type": "broad",
                "prompt_text": "a plain object",
                "requires_valid_lineage": False,
                "axis_id": "grade/object/color",
                "axis_name": "color",
                "complexity_level": "atomic",
                "target_behavior": "",
                "candidate_behaviors": ["red", "blue"],
                "evaluator": "test",
            },
            {
                "eval_uid": "eval-b",
                "image_prompt_uid": "prompt-a",
                "benchmark": "grade",
                "condition_type": "broad",
                "prompt_text": "a plain object",
                "requires_valid_lineage": False,
                "axis_id": "grade/object/shape",
                "axis_name": "shape",
                "complexity_level": "atomic",
                "target_behavior": "",
                "candidate_behaviors": ["round", "square"],
                "evaluator": "test",
            },
        ],
    )
    manifest = tmp_path / "manifest.jsonl"
    summary = build_generation_manifest(config, bank, manifest, model_scope="primary")
    assert summary["jobs"] == 6 * 32
    assert summary["jobs_by_device"] == {"cuda:2": 192}
    jobs = read_jsonl(manifest)
    assert [job["model_id"] for job in jobs[:32]] == ["sdxl_base"] * 32
    assert jobs[31]["model_id"] == "sdxl_base"
    assert jobs[32]["model_id"] == "sdxl_lightning_4step"

    prompt_major = tmp_path / "manifest_prompt_major.jsonl"
    build_generation_manifest(config, bank, prompt_major, model_scope="primary", job_order="prompt_major")
    prompt_jobs = read_jsonl(prompt_major)
    assert prompt_jobs[0]["model_id"] == "sdxl_base"
    assert prompt_jobs[32]["model_id"] == "sdxl_lightning_4step"

    labels = tmp_path / "labels.csv"
    label_summary = build_label_template(bank, manifest, labels)
    assert label_summary["label_rows"] == 6 * 32 * 2
