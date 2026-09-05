from __future__ import annotations

from pathlib import Path

from t2i_distill.io import read_json, read_jsonl, write_jsonl
from t2i_distill.preflight import build_preflight_package


def test_build_preflight_package_selects_conditions_and_seed_zero(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    write_jsonl(
        bank,
        [
            _record("eval-a", "prompt-a", "grade", "broad", "grade/a"),
            _record("eval-b", "prompt-b", "dimcim", "explicit", "dimcim/b"),
            _record("eval-c", "prompt-c", "hub", "unlearning_retain", "hub/c", gated=True),
        ],
    )
    manifest = tmp_path / "manifest.jsonl"
    jobs = []
    for uid in ["prompt-a", "prompt-b", "prompt-c"]:
        for model in ["sdxl_base", "sdxl_lightning_4step"]:
            for seed in [0, 1]:
                jobs.append(_job(uid, model, seed))
    write_jsonl(manifest, jobs)

    summary = build_preflight_package(bank, manifest, tmp_path / "preflight", prompts_per_condition=1, seeds={0}, jobs_per_shard=3)
    assert summary["selected_condition_count"] == 2
    assert summary["generation_manifest"]["jobs"] == 4
    assert summary["label_template"]["label_rows"] == 4
    assert summary["shards"]["shard_count"] == 2
    assert summary["evaluator_batches"]["rows_selected"] == 4
    assert summary["evaluator_preview"]["written"] == 4
    assert (tmp_path / "preflight" / "evaluator_summary.csv").exists()
    assert (tmp_path / "preflight" / "evaluator_batch_index.csv").exists()
    assert read_json(tmp_path / "preflight" / "preflight_summary.json")["selected_unique_image_prompts"] == 2
    assert len(read_jsonl(tmp_path / "preflight" / "generation_manifest.jsonl")) == 4
    preview = read_jsonl(tmp_path / "preflight" / "evaluator_batch_preview.jsonl")
    assert "Quality score rubric" in preview[0]["evaluation_instruction"]
    assert "teacher/student identity" in preview[0]["evaluation_instruction"]


def _record(eval_uid: str, image_uid: str, benchmark: str, condition: str, axis_id: str, gated: bool = False) -> dict[str, object]:
    return {
        "eval_uid": eval_uid,
        "image_prompt_uid": image_uid,
        "benchmark": benchmark,
        "source": "test",
        "source_path": "test.jsonl",
        "source_row_id": eval_uid,
        "prompt_text": "a prompt",
        "axis_id": axis_id,
        "axis_name": axis_id,
        "condition_type": condition,
        "analysis_layers": [],
        "complexity_level": "atomic",
        "target_behavior": "",
        "candidate_behaviors": [],
        "evaluator": "test",
        "requires_valid_lineage": gated,
        "metadata": {},
    }


def _job(image_uid: str, model_id: str, seed: int) -> dict[str, object]:
    benchmark = "grade" if image_uid == "prompt-a" else "dimcim" if image_uid == "prompt-b" else "hub"
    condition = "broad" if image_uid == "prompt-a" else "explicit" if image_uid == "prompt-b" else "unlearning_retain"
    return {
        "job_id": f"{image_uid}-{model_id}-{seed}",
        "model_id": model_id,
        "runner": "test",
        "hf_model": "test",
        "device": "cuda:2",
        "benchmark": benchmark,
        "condition_type": condition,
        "image_prompt_uid": image_uid,
        "seed": seed,
        "prompt_text": "a prompt",
        "output_path": f"outputs/{image_uid}-{model_id}-{seed}.png",
    }
