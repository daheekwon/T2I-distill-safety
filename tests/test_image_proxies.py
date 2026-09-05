
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from t2i_distill.image_proxies import analyze_image, auto_label_image_proxies
from t2i_distill.io import stable_hash, write_csv, write_json
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS


def test_analyze_image_flags_blank_as_low_quality_refusal(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    varied = tmp_path / "varied.png"
    Image.new("RGB", (512, 512), "white").save(blank)
    img = Image.new("RGB", (512, 512), "black")
    for x in range(512):
        for y in range(512):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    img.save(varied)

    blank_features = analyze_image(blank)
    varied_features = analyze_image(varied)

    assert blank_features["refusal_proxy"] == "true"
    assert float(blank_features["quality_score"]) < 3.0
    assert varied_features["readable"] == "true"
    assert float(varied_features["quality_score"]) >= float(blank_features["quality_score"])


def test_auto_label_image_proxies_fills_quality_refusal_and_phash_memorization(tmp_path: Path) -> None:
    generated = tmp_path / "generated.png"
    ref = tmp_path / "ref.png"
    safety = tmp_path / "safety.png"
    Image.new("RGB", (512, 512), (10, 120, 200)).save(generated)
    Image.new("RGB", (512, 512), (10, 120, 200)).save(ref)
    Image.new("RGB", (512, 512), "white").save(safety)
    labels = tmp_path / "labels.csv"
    rows = [
        _row("job-mem", "mem:e0", "teacher", "membench", "memorization_trigger", str(generated)),
        _row("job-safe", "safe:e0", "teacher", "overt", "benign_overrefusal", str(safety)),
    ]
    write_csv(labels, rows, LABEL_TEMPLATE_FIELDS)
    reference_index = tmp_path / "refs.json"
    write_json(
        reference_index,
        {
            "summary": {},
            "records": [
                {
                    "eval_uid": "mem:e0",
                    "image_prompt_uid": "img-mem:e0",
                    "condition_type": "memorization_trigger",
                    "source_row_id": "0",
                    "url_index": 0,
                    "url": "https://example.test/ref.png",
                    "status": "ok",
                    "path": str(ref),
                    "bytes": ref.stat().st_size,
                    "width": 512,
                    "height": 512,
                    "error": "",
                }
            ],
        },
    )

    summary = auto_label_image_proxies(
        labels_path=labels,
        output_labels_path=tmp_path / "auto_labels.csv",
        output_updates_path=tmp_path / "updates.csv",
        feature_csv_path=tmp_path / "features.csv",
        summary_json_path=tmp_path / "summary.json",
        copy_scores_csv_path=tmp_path / "copy.csv",
        reference_index_path=reference_index,
        sscd_checkpoint=None,
        device="cpu",
        batch_size=2,
    )

    assert summary["updates_rows"] == 2
    with (tmp_path / "auto_labels.csv").open("r", encoding="utf-8", newline="") as handle:
        out = {row["label_uid"]: row for row in csv.DictReader(handle)}
    assert float(out[stable_hash("job-mem", "mem:e0")]["memorization_score"]) >= 0.5
    assert out[stable_hash("job-safe", "safe:e0")]["refusal"] == "true"
    assert out[stable_hash("job-safe", "safe:e0")]["unsafe"] == ""


def test_auto_label_image_proxies_keeps_false_refusal_false(tmp_path: Path) -> None:
    image = tmp_path / "varied.png"
    img = Image.new("RGB", (512, 512), "black")
    for x in range(512):
        for y in range(512):
            img.putpixel((x, y), (x % 256, y % 256, (x * 2 + y) % 256))
    img.save(image)
    labels = tmp_path / "labels.csv"
    write_csv(labels, [_row("job-safe", "safe:e1", "teacher", "overt", "unsafe_safety", str(image))], LABEL_TEMPLATE_FIELDS)

    auto_label_image_proxies(
        labels_path=labels,
        output_labels_path=tmp_path / "auto_labels.csv",
        output_updates_path=tmp_path / "updates.csv",
        feature_csv_path=tmp_path / "features.csv",
        summary_json_path=tmp_path / "summary.json",
        copy_scores_csv_path=tmp_path / "copy.csv",
        reference_index_path=None,
        sscd_checkpoint=None,
        device="cpu",
        batch_size=2,
    )

    with (tmp_path / "auto_labels.csv").open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["refusal"] == "false"
    assert row["unsafe"] == ""


def _row(job_id: str, eval_uid: str, model_id: str, benchmark: str, condition: str, image_path: str) -> dict[str, object]:
    row = {field: "" for field in LABEL_TEMPLATE_FIELDS}
    row.update(
        {
            "label_uid": stable_hash(job_id, eval_uid),
            "job_id": job_id,
            "eval_uid": eval_uid,
            "model_id": model_id,
            "benchmark": benchmark,
            "axis_id": f"{benchmark}/{condition}",
            "axis_name": condition,
            "condition_type": condition,
            "complexity_level": "unit",
            "image_prompt_uid": f"img-{eval_uid}",
            "seed": "0",
            "prompt_text": "prompt",
            "image_path": image_path,
            "target_behavior": "not_memorized_reference",
            "candidate_behaviors": [],
            "evaluator": "unit",
        }
    )
    return row
