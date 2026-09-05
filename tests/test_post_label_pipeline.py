from __future__ import annotations

import json
import subprocess
from pathlib import Path

from t2i_distill.io import write_csv
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS


def test_run_post_label_pipeline_on_filled_fixture(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    rows = []
    for seed in range(4):
        rows.append(_row(f"t{seed}", "sdxl_base", "grade", "broad", "grade/color", f"prompt-{seed}", str(seed), semantic_label="red"))
        rows.append(_row(f"s{seed}", "sdxl_lightning_4step", "grade", "broad", "grade/color", f"prompt-{seed}", str(seed), semantic_label="red"))
    write_csv(labels, rows, LABEL_TEMPLATE_FIELDS)
    out = tmp_path / "post"
    result = subprocess.run(
        [
            "/DATA0/dahee/diversity/.miniconda3/envs/diversity/bin/python",
            "scripts/run_post_label_pipeline.py",
            "--labels",
            str(labels),
            "--manifest",
            "data/preflight/generation_manifest.jsonl",
            "--output-dir",
            str(out),
            "--allow-partial-rq",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(result.stdout)
    assert payload["labels_path"] == str(labels)
    assert (out / "design_matrix.json").exists()
    assert (out / "statistical_power.json").exists()
    assert (out / "label_quality.json").exists()
    assert (out / "experiment_state.json").exists()
    assert (out / "rq_readiness.json").exists()
    assert (out / "analysis" / "distribution_summary.csv").exists()


def _row(
    label_uid: str,
    model_id: str,
    benchmark: str,
    condition: str,
    axis_id: str,
    image_uid: str,
    seed: str,
    *,
    semantic_label: str = "",
    pass_score: str = "",
    quality_score: str = "4",
) -> dict[str, object]:
    row = {field: "" for field in LABEL_TEMPLATE_FIELDS}
    row.update(
        {
            "label_uid": label_uid,
            "job_id": f"job-{label_uid}",
            "eval_uid": f"eval-{axis_id}",
            "model_id": model_id,
            "benchmark": benchmark,
            "axis_id": axis_id,
            "axis_name": axis_id,
            "condition_type": condition,
            "complexity_level": "atomic",
            "image_prompt_uid": image_uid,
            "seed": seed,
            "prompt_text": "a prompt",
            "image_path": "outputs/image.png",
            "target_behavior": "red",
            "candidate_behaviors": ["red", "blue"],
            "semantic_label": semantic_label,
            "pass_score": pass_score,
            "quality_score": quality_score,
            "evaluator": "unit",
        }
    )
    return row
