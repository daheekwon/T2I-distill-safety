from __future__ import annotations

from pathlib import Path

from t2i_distill.io import read_csv, read_json, read_jsonl, write_jsonl
from t2i_distill.mats20 import GPU0_MODELS, GPU2_MODELS, build_mats20_package


MODELS = list(GPU0_MODELS) + list(GPU2_MODELS)


def test_build_mats20_package_splits_gpu_manifests_and_filters_prompts(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    rows = [
        _record("mem-trigger", "membench", "memorization_trigger", "membench/sd1_mem345", "sd1_mem345"),
        _record("mem-control", "membench", "memorization_control", "membench/group_nomem500", "group_nomem500"),
        _record("overt-benign", "overt", "benign_overrefusal", "overt/privacy_public", "privacy_public"),
        _record("overt-unsafe", "overt", "unsafe_safety", "overt/privacy_public", "privacy_public", source_row_id="pair-0:unsafe_safety"),
        _record(
            "overt-counter",
            "overt",
            "safety_counterfactual_benign",
            "overt/privacy_public",
            "privacy_public",
            source_row_id="pair-0:safety_counterfactual_benign",
        ),
        _record("native-safety", "t2isafety", "safety_native", "t2isafety/public_figures", "public figures"),
        _record("bias-default", "t2isafety", "social_bias_default", "t2isafety/fairness", "fairness_demographic_distribution", metadata={"category": "public figures"}),
        _record("risky", "t2i_riskyprompt", "risky_safety", "t2i_riskyprompt/Copyright_Infringement/LOGO", "LOGO"),
        _record("grade", "grade", "broad", "grade/default", "default"),
        _record("dimcim", "dimcim", "broad", "dimcim/default", "default"),
        _record(
            "blocked",
            "t2isafety",
            "safety_native",
            "t2isafety/sexual",
            "sexual",
            prompt="unsafe preteen sexualized request",
        ),
    ]
    write_jsonl(bank, rows)

    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            _job(row["image_prompt_uid"], row["benchmark"], row["condition_type"], model, seed)
            for row in rows
            for model in MODELS
            for seed in (0, 1)
        ],
    )

    summary = build_mats20_package(bank, manifest, tmp_path / "mats20", jobs_per_shard=10)

    assert summary["prompt_selection"]["selected_unique_image_prompts"] == 10
    assert "blocked" not in {
        row["image_prompt_uid"] for row in read_jsonl(tmp_path / "mats20" / "prompt_bank.jsonl")
    }

    canonical_jobs = read_jsonl(tmp_path / "mats20" / "generation_manifest.jsonl")
    gpu0_jobs = read_jsonl(tmp_path / "mats20" / "generation_manifest_gpu0.jsonl")
    gpu2_jobs = read_jsonl(tmp_path / "mats20" / "generation_manifest_gpu2.jsonl")

    assert len(canonical_jobs) == 84
    assert len(gpu0_jobs) == 28
    assert len(gpu2_jobs) == 56
    assert {job["device"] for job in canonical_jobs} == {"cuda:2"}
    assert {job["device"] for job in gpu0_jobs} == {"cuda:0"}
    assert {job["device"] for job in gpu2_jobs} == {"cuda:2"}
    assert {job["model_id"] for job in gpu0_jobs} == set(GPU0_MODELS)
    assert {job["model_id"] for job in gpu2_jobs} == set(GPU2_MODELS)
    assert all(job["output_path"].startswith("outputs/mats20/images/") for job in canonical_jobs)
    assert {job["output_path"] for job in gpu0_jobs}.issubset({job["output_path"] for job in canonical_jobs})

    assert len(read_csv(tmp_path / "mats20" / "label_template.csv")) == 84
    assert read_json(tmp_path / "mats20" / "mats20_summary.json")["evaluator_batches"]["rows_selected"] == 84
    assert (tmp_path / "mats20" / "selection_plan.md").exists()


def _record(
    uid: str,
    benchmark: str,
    condition: str,
    axis_id: str,
    category: str,
    *,
    source_row_id: str | None = None,
    prompt: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = {"category": category} if metadata is None else metadata
    return {
        "eval_uid": f"eval-{uid}",
        "image_prompt_uid": uid,
        "benchmark": benchmark,
        "source": "test",
        "source_path": "test.jsonl",
        "source_row_id": source_row_id or uid,
        "prompt_text": prompt or f"a neutral prompt for {uid}",
        "axis_id": axis_id,
        "axis_name": category,
        "condition_type": condition,
        "analysis_layers": [],
        "complexity_level": "atomic",
        "target_behavior": "",
        "candidate_behaviors": ["yes", "no"],
        "evaluator": "test",
        "requires_valid_lineage": False,
        "metadata": metadata,
    }


def _job(
    image_uid: str,
    benchmark: str,
    condition: str,
    model_id: str,
    seed: int,
) -> dict[str, object]:
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
        "prompt_text": f"a neutral prompt for {image_uid}",
        "output_path": f"outputs/images/{model_id}/{benchmark}/{image_uid}__seed{seed}.png",
    }
