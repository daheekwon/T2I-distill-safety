#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.image_proxies import _embed_images_sscd  # noqa: E402


PAIR_DEFS = [
    ("sdxl_base__sdxl_lightning_4step", "sdxl", "sdxl_base", "sdxl_lightning_4step"),
    ("sd35_large__sd35_large_turbo", "sd35", "sd35_large", "sd35_large_turbo"),
    ("flux2_klein_base_4b__flux2_klein_4b", "flux2_klein", "flux2_klein_base_4b", "flux2_klein_4b"),
]

MATRIX_FIELDS = [
    "label_uid",
    "model_id",
    "condition_type",
    "image_eval_uid",
    "reference_eval_uid",
    "is_target_reference",
    "seed",
    "image_path",
    "reference_count",
    "best_reference_path",
    "raw_sscd_similarity",
]

EFFECT_FIELDS = [
    "comparison_id",
    "family",
    "eval_uid",
    "paired_n_trigger",
    "paired_n_control",
    "teacher_trigger_mean_sscd",
    "teacher_control_mean_sscd",
    "teacher_trigger_effect",
    "student_trigger_mean_sscd",
    "student_control_mean_sscd",
    "student_trigger_effect",
    "trigger_effect_delta_D",
]

CONTROL_FIELDS = [
    "eval_uid",
    "reference_path",
    "control_type",
    "control_path",
    "raw_sscd_similarity",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MemBench trigger-control effects with raw SSCD and target-reference specificity.")
    parser.add_argument("--labels", default="data/mats20_aug/balanced_1h/filled_labels_qwen_proxy_sscd.csv")
    parser.add_argument("--reference-index", default="data/mats20/membench_reference_index_recovered.json")
    parser.add_argument("--output-dir", default="results/mats20_deep/diagnostics/membench_trigger_effect")
    parser.add_argument("--sscd-checkpoint", default="/DATA0/dahee/.cache/torch/hub/checkpoints/sscd_disc_mixup.torchscript.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in _read_csv(Path(args.labels)) if row.get("benchmark") == "membench"]
    refs_by_eval = _load_refs(Path(args.reference_index))
    eval_uids = sorted({row["eval_uid"] for row in rows if refs_by_eval.get(row["eval_uid"])})
    refs_by_eval = {eval_uid: refs_by_eval[eval_uid] for eval_uid in eval_uids}
    if not eval_uids:
        raise SystemExit("No MemBench rows with available references.")

    transformed_paths = _write_positive_control_images(refs_by_eval, outdir / "positive_controls")
    all_paths = sorted(
        {
            *[row["image_path"] for row in rows if row.get("image_path")],
            *[path for paths in refs_by_eval.values() for path in paths],
            *transformed_paths,
        }
    )
    embeddings = _embed_images_sscd([Path(path) for path in all_paths], Path(args.sscd_checkpoint), device=args.device, batch_size=args.batch_size)
    matrix_rows = _specificity_matrix(rows, refs_by_eval, embeddings)
    effect_rows = _trigger_effect_rows(rows, matrix_rows)
    control_rows = _positive_control_rows(refs_by_eval, transformed_paths, embeddings)

    _write_csv(outdir / "reference_specificity_matrix.csv", matrix_rows, MATRIX_FIELDS)
    _write_csv(outdir / "trigger_effect_by_item.csv", effect_rows, EFFECT_FIELDS)
    _write_csv(outdir / "detector_positive_controls.csv", control_rows, CONTROL_FIELDS)
    summary = {
        "labels": args.labels,
        "reference_index": args.reference_index,
        "membench_rows": len(rows),
        "eval_uids": eval_uids,
        "reference_eval_uids": len(eval_uids),
        "matrix_rows": len(matrix_rows),
        "trigger_effect_rows": len(effect_rows),
        "positive_control_rows": len(control_rows),
        "max_abs_trigger_effect_delta_D": max((abs(float(row["trigger_effect_delta_D"])) for row in effect_rows), default=0.0),
        "target_top1_rate": _target_top1_rate(matrix_rows),
        "outputs": {
            "matrix": str(outdir / "reference_specificity_matrix.csv"),
            "trigger_effect": str(outdir / "trigger_effect_by_item.csv"),
            "positive_controls": str(outdir / "detector_positive_controls.csv"),
            "report_md": str(outdir / "membench_trigger_effect_report.md"),
        },
    }
    (outdir / "membench_trigger_effect_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (outdir / "membench_trigger_effect_report.md").write_text(_report(summary, effect_rows, control_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _specificity_matrix(rows: list[dict[str, str]], refs_by_eval: dict[str, list[str]], embeddings: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        image_path = row.get("image_path", "")
        image_embedding = embeddings.get(image_path)
        if image_embedding is None:
            continue
        for ref_eval_uid, ref_paths in refs_by_eval.items():
            values: list[tuple[float, str]] = []
            for ref_path in ref_paths:
                ref_embedding = embeddings.get(ref_path)
                if ref_embedding is not None:
                    values.append((float(np.dot(image_embedding, ref_embedding)), ref_path))
            if not values:
                continue
            best, best_path = max(values, key=lambda item: item[0])
            out.append(
                {
                    "label_uid": row["label_uid"],
                    "model_id": row["model_id"],
                    "condition_type": row["condition_type"],
                    "image_eval_uid": row["eval_uid"],
                    "reference_eval_uid": ref_eval_uid,
                    "is_target_reference": str(row["eval_uid"] == ref_eval_uid).lower(),
                    "seed": row.get("seed", ""),
                    "image_path": image_path,
                    "reference_count": len(ref_paths),
                    "best_reference_path": best_path,
                    "raw_sscd_similarity": f"{best:.6f}",
                }
            )
    return out


def _trigger_effect_rows(labels: list[dict[str, str]], matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in matrix_rows:
        if str(row["is_target_reference"]) != "true":
            continue
        target_scores[(row["eval_uid"] if "eval_uid" in row else row["image_eval_uid"], row["model_id"], row["condition_type"])].append(float(row["raw_sscd_similarity"]))
    out: list[dict[str, Any]] = []
    for comparison_id, family, teacher, student in PAIR_DEFS:
        for eval_uid in sorted({row["eval_uid"] for row in labels}):
            teacher_trigger = target_scores.get((eval_uid, teacher, "memorization_trigger"), [])
            teacher_control = target_scores.get((eval_uid, teacher, "memorization_control"), [])
            student_trigger = target_scores.get((eval_uid, student, "memorization_trigger"), [])
            student_control = target_scores.get((eval_uid, student, "memorization_control"), [])
            if not (teacher_trigger and teacher_control and student_trigger and student_control):
                continue
            t_trigger = _mean(teacher_trigger)
            t_control = _mean(teacher_control)
            s_trigger = _mean(student_trigger)
            s_control = _mean(student_control)
            t_effect = t_trigger - t_control
            s_effect = s_trigger - s_control
            out.append(
                {
                    "comparison_id": comparison_id,
                    "family": family,
                    "eval_uid": eval_uid,
                    "paired_n_trigger": min(len(teacher_trigger), len(student_trigger)),
                    "paired_n_control": min(len(teacher_control), len(student_control)),
                    "teacher_trigger_mean_sscd": f"{t_trigger:.6f}",
                    "teacher_control_mean_sscd": f"{t_control:.6f}",
                    "teacher_trigger_effect": f"{t_effect:.6f}",
                    "student_trigger_mean_sscd": f"{s_trigger:.6f}",
                    "student_control_mean_sscd": f"{s_control:.6f}",
                    "student_trigger_effect": f"{s_effect:.6f}",
                    "trigger_effect_delta_D": f"{s_effect - t_effect:.6f}",
                }
            )
    return out


def _write_positive_control_images(refs_by_eval: dict[str, list[str]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for eval_uid, paths in refs_by_eval.items():
        source = Path(paths[0])
        try:
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                jpeg = output_dir / f"{eval_uid.replace(':', '_')}_jpeg_q85.jpg"
                resized = output_dir / f"{eval_uid.replace(':', '_')}_resize512.jpg"
                rgb.save(jpeg, quality=85)
                rgb.resize((512, 512), Image.Resampling.LANCZOS).save(resized, quality=92)
            out.extend([str(jpeg), str(resized)])
        except Exception:
            continue
    return out


def _positive_control_rows(refs_by_eval: dict[str, list[str]], transformed_paths: list[str], embeddings: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    by_eval_prefix: dict[str, list[str]] = defaultdict(list)
    for path in transformed_paths:
        stem = Path(path).stem
        for eval_uid in refs_by_eval:
            if stem.startswith(eval_uid.replace(":", "_")):
                by_eval_prefix[eval_uid].append(path)
    out: list[dict[str, Any]] = []
    for eval_uid, ref_paths in refs_by_eval.items():
        source = ref_paths[0]
        source_embedding = embeddings.get(source)
        if source_embedding is None:
            continue
        out.append(
            {
                "eval_uid": eval_uid,
                "reference_path": source,
                "control_type": "self",
                "control_path": source,
                "raw_sscd_similarity": "1.000000",
            }
        )
        for transformed in sorted(by_eval_prefix.get(eval_uid, [])):
            emb = embeddings.get(transformed)
            if emb is None:
                continue
            out.append(
                {
                    "eval_uid": eval_uid,
                    "reference_path": source,
                    "control_type": Path(transformed).stem.rsplit("_", 2)[-2],
                    "control_path": transformed,
                    "raw_sscd_similarity": f"{float(np.dot(source_embedding, emb)):.6f}",
                }
            )
    return out


def _target_top1_rate(matrix_rows: list[dict[str, Any]]) -> float:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix_rows:
        by_label[row["label_uid"]].append(row)
    if not by_label:
        return 0.0
    top1 = 0
    for rows in by_label.values():
        best = max(rows, key=lambda row: float(row["raw_sscd_similarity"]))
        top1 += str(best["is_target_reference"]) == "true"
    return top1 / len(by_label)


def _load_refs(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", payload if isinstance(payload, list) else [])
    out: dict[str, list[str]] = defaultdict(list)
    for record in records:
        status = str(record.get("status", ""))
        ref_path = str(record.get("path", ""))
        if status.startswith("ok") and ref_path and Path(ref_path).exists():
            out[str(record.get("eval_uid", ""))].append(ref_path)
    return {key: sorted(set(values)) for key, values in out.items()}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _report(summary: dict[str, Any], effect_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> str:
    lines = ["# MemBench Trigger Effect Diagnostic", ""]
    lines.append(f"- MemBench rows: {summary['membench_rows']}")
    lines.append(f"- reference-backed eval_uids: {summary['reference_eval_uids']}")
    lines.append(f"- target top-1 reference rate: {summary['target_top1_rate']:.3f}")
    if control_rows:
        non_self = [float(row["raw_sscd_similarity"]) for row in control_rows if row["control_type"] != "self"]
        if non_self:
            lines.append(f"- positive control transformed-reference mean SSCD: {_mean(non_self):.3f}")
    lines.append("")
    lines.append("| pair | eval_uid | teacher effect | student effect | D |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in effect_rows:
        lines.append(
            f"| `{row['comparison_id']}` | `{row['eval_uid']}` | "
            f"{float(row['teacher_trigger_effect']):.4f} | {float(row['student_trigger_effect']):.4f} | {float(row['trigger_effect_delta_D']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
