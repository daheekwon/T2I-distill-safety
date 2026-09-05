from __future__ import annotations

from pathlib import Path

from t2i_distill.design_audit import audit_design_matrix
from t2i_distill.io import write_csv, write_jsonl
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS


def test_design_audit_checks_manifest_label_join_and_rq_pairs(tmp_path: Path) -> None:
    config = _config()
    prompt_bank = tmp_path / "prompts.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    labels = tmp_path / "labels.csv"
    write_jsonl(
        prompt_bank,
        [
            _prompt("grade:e0", "grade", "broad", "grade/color", "", ["red", "blue"]),
            _prompt("dim:e1", "dimcim", "broad", "dimcim/chair/color", "", ["red", "blue"]),
            _prompt("dim:e2", "dimcim", "explicit", "dimcim/chair/color", "red", ["red"]),
            _prompt("risk:e3", "t2i_riskyprompt", "risky_safety", "risk/privacy", "", []),
            _prompt("mem:e4", "membench", "memorization_trigger", "mem/trigger", "", []),
            _prompt("mem:e5", "membench", "memorization_control", "mem/control", "", []),
        ],
    )
    jobs = []
    label_rows = []
    for model_id in ["teacher", "student"]:
        for idx, (benchmark, condition, image_uid, eval_uid) in enumerate(
            [
                ("grade", "broad", "img-grade", "grade:e0"),
                ("dimcim", "broad", "img-dim-broad", "dim:e1"),
                ("dimcim", "explicit", "img-dim-explicit", "dim:e2"),
                ("t2i_riskyprompt", "risky_safety", "img-risk", "risk:e3"),
                ("membench", "memorization_trigger", "img-mem-t", "mem:e4"),
                ("membench", "memorization_control", "img-mem-c", "mem:e5"),
            ]
        ):
            job_id = f"job-{model_id}-{idx}"
            jobs.append(
                {
                    "job_id": job_id,
                    "model_id": model_id,
                    "runner": "unit",
                    "hf_model": "unit/model",
                    "device": "cuda:2",
                    "benchmark": benchmark,
                    "condition_type": condition,
                    "image_prompt_uid": image_uid,
                    "seed": 0,
                    "prompt_text": "prompt",
                    "output_path": f"outputs/{model_id}/{idx}.png",
                    "eval_uids": [eval_uid],
                }
            )
            label_rows.append(_label(job_id, eval_uid, model_id, benchmark, condition, image_uid, f"outputs/{model_id}/{idx}.png"))
    write_jsonl(manifest, jobs)
    write_csv(labels, label_rows, LABEL_TEMPLATE_FIELDS)

    audit = audit_design_matrix(
        config=config,
        prompt_bank_path=prompt_bank,
        manifest_path=manifest,
        labels_path=labels,
        shard_index_path=None,
    )

    assert audit["gates"]["manifest_label_join"]["pass"] is True
    assert audit["gates"]["label_uid_integrity"]["pass"] is True
    assert audit["rq_design_summary"]["rq1_sharpening_fit"]["planned_matched_cells"] == 2
    assert audit["rq3_dissociation_bridge"]["axis_target_bridge_count"] == 1
    assert audit["gates"]["safety_memorization_coverage"]["pass"] is False


def test_design_audit_catches_label_uid_mismatch(tmp_path: Path) -> None:
    config = _config(benchmarks=["grade"])
    prompt_bank = tmp_path / "prompts.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    labels = tmp_path / "labels.csv"
    write_jsonl(prompt_bank, [_prompt("grade:e0", "grade", "broad", "grade/color", "", ["red", "blue"])])
    write_jsonl(
        manifest,
        [
            {
                "job_id": "job-teacher-0",
                "model_id": "teacher",
                "runner": "unit",
                "hf_model": "unit/model",
                "device": "cuda:2",
                "benchmark": "grade",
                "condition_type": "broad",
                "image_prompt_uid": "img-grade",
                "seed": 0,
                "prompt_text": "prompt",
                "output_path": "outputs/teacher/0.png",
                "eval_uids": ["grade:e0"],
            }
        ],
    )
    row = _label("job-teacher-0", "grade:e0", "teacher", "grade", "broad", "img-grade", "outputs/teacher/0.png")
    row["label_uid"] = "wrong"
    write_csv(labels, [row], LABEL_TEMPLATE_FIELDS)

    audit = audit_design_matrix(config=config, prompt_bank_path=prompt_bank, manifest_path=manifest, labels_path=labels, shard_index_path=None)

    assert audit["gates"]["label_uid_integrity"]["pass"] is False
    assert audit["label_design"]["label_uid_mismatches"] == 1


def test_design_audit_can_scope_required_rqs_for_safety_memorization_plan(tmp_path: Path) -> None:
    config = _config(benchmarks=["overt", "t2isafety", "t2i_riskyprompt", "membench"])
    config["analysis"]["required_primary_rqs"] = ["safety_inheritance", "memorization_retention"]
    config["analysis"]["require_rq3_dissociation_bridge"] = False
    for benchmark, condition in [
        ("overt", "unsafe_safety"),
        ("overt", "benign_overrefusal"),
        ("overt", "safety_counterfactual_benign"),
        ("t2isafety", "safety_native"),
        ("t2isafety", "social_bias_default"),
    ]:
        config["benchmark_plan"][benchmark]["seed_keys"][condition] = "single"
    prompt_bank = tmp_path / "prompts.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    labels = tmp_path / "labels.csv"
    conditions = [
        ("overt", "unsafe_safety", "overt:unsafe"),
        ("overt", "benign_overrefusal", "overt:benign"),
        ("overt", "safety_counterfactual_benign", "overt:counter"),
        ("t2isafety", "safety_native", "safe:native"),
        ("t2isafety", "social_bias_default", "safe:bias"),
        ("t2i_riskyprompt", "risky_safety", "risk:risky"),
        ("membench", "memorization_trigger", "mem:trigger"),
        ("membench", "memorization_control", "mem:control"),
    ]
    write_jsonl(prompt_bank, [_prompt(eval_uid, benchmark, condition, f"{benchmark}/{condition}", "", []) for benchmark, condition, eval_uid in conditions])
    jobs = []
    label_rows = []
    for model_id in ["teacher", "student"]:
        for idx, (benchmark, condition, eval_uid) in enumerate(conditions):
            job_id = f"job-{model_id}-{idx}"
            image_uid = f"img-{eval_uid}"
            image_path = f"outputs/{model_id}/{idx}.png"
            jobs.append(
                {
                    "job_id": job_id,
                    "model_id": model_id,
                    "runner": "unit",
                    "hf_model": "unit/model",
                    "device": "cuda:2",
                    "benchmark": benchmark,
                    "condition_type": condition,
                    "image_prompt_uid": image_uid,
                    "seed": 0,
                    "prompt_text": "prompt",
                    "output_path": image_path,
                    "eval_uids": [eval_uid],
                }
            )
            label_rows.append(_label(job_id, eval_uid, model_id, benchmark, condition, image_uid, image_path))
    write_jsonl(manifest, jobs)
    write_csv(labels, label_rows, LABEL_TEMPLATE_FIELDS)

    audit = audit_design_matrix(config=config, prompt_bank_path=prompt_bank, manifest_path=manifest, labels_path=labels, shard_index_path=None)

    assert audit["gates"]["rq_pair_design"]["pass"] is True
    assert audit["gates"]["rq3_dissociation_bridge"]["pass"] is True
    assert audit["gates"]["safety_memorization_coverage"]["pass"] is True
    assert audit["overall_pass"] is True


def _config(benchmarks: list[str] | None = None) -> dict[str, object]:
    benchmark_names = benchmarks or ["grade", "dimcim", "t2i_riskyprompt", "membench"]
    plans = {
        name: {"seed_keys": {"broad": "single", "explicit": "single", "risky_safety": "single", "memorization_trigger": "single", "memorization_control": "single"}, "source_paths": []}
        for name in benchmark_names
    }
    return {
        "default_device": "cuda:2",
        "seed_sets": {"single": [0]},
        "benchmark_plan": plans,
        "analysis": {
            "fit_sharpening_on": ["grade", "dimcim"],
            "transfer_alpha_to": ["t2i_riskyprompt", "membench"],
        },
        "model_pairs": [
            {
                "comparison_id": "teacher__student",
                "family": "unit",
                "priority": "primary",
                "teacher": {"model_id": "teacher"},
                "student": {"model_id": "student"},
            }
        ],
    }


def _prompt(eval_uid: str, benchmark: str, condition: str, axis_id: str, target: str, candidates: list[str]) -> dict[str, object]:
    return {
        "eval_uid": eval_uid,
        "benchmark": benchmark,
        "condition_type": condition,
        "axis_id": axis_id,
        "axis_name": axis_id,
        "complexity_level": "atomic",
        "image_prompt_uid": f"img-{eval_uid}",
        "prompt_text": "prompt",
        "target_behavior": target,
        "candidate_behaviors": candidates,
        "source": "unit",
        "source_path": "",
        "requires_valid_lineage": False,
    }


def _label(job_id: str, eval_uid: str, model_id: str, benchmark: str, condition: str, image_uid: str, image_path: str) -> dict[str, object]:
    from t2i_distill.io import stable_hash

    row = {field: "" for field in LABEL_TEMPLATE_FIELDS}
    row.update(
        {
            "label_uid": stable_hash(job_id, eval_uid),
            "job_id": job_id,
            "eval_uid": eval_uid,
            "model_id": model_id,
            "benchmark": benchmark,
            "axis_id": "dimcim/chair/color" if benchmark == "dimcim" else f"{benchmark}/axis",
            "axis_name": "axis",
            "condition_type": condition,
            "complexity_level": "atomic",
            "image_prompt_uid": image_uid,
            "seed": "0",
            "prompt_text": "prompt",
            "image_path": image_path,
            "target_behavior": "red" if condition == "explicit" else "",
            "candidate_behaviors": ["red", "blue"],
            "evaluator": "unit",
        }
    )
    return row
