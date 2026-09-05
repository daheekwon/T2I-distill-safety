#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PAIRS = [
    ("sdxl_base__sdxl_lightning_4step", "sdxl_base", "sdxl_lightning_4step"),
    ("sd35_large__sd35_large_turbo", "sd35_large", "sd35_large_turbo"),
    ("flux2_klein_base_4b__flux2_klein_4b", "flux2_klein_base_4b", "flux2_klein_4b"),
]

PAIR_FIELDS = [
    "grid_name",
    "row_index",
    "comparison_id",
    "benchmark",
    "condition_type",
    "protocol",
    "eval_uid",
    "image_prompt_uid",
    "seed",
    "teacher_model",
    "student_model",
    "teacher_unsafe",
    "student_unsafe",
    "teacher_risk",
    "student_risk",
    "teacher_fulfillment",
    "student_fulfillment",
    "teacher_image_path",
    "student_image_path",
    "prompt_text",
    "selection_note",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create grids for deep safety diagnostics.")
    parser.add_argument("--diagnostics", default="data/mats20_deep/diagnostics/deep_safety_diagnostics.csv")
    parser.add_argument("--output-dir", default="results/mats20_deep/diagnostics/grids")
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    rows = _read_csv(Path(args.diagnostics))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    selections: list[dict[str, str]] = []

    paired = _paired_rows(rows)

    overt_rows = _select_overt_rows(paired, protocol="C", rng=rng)
    overt_path = output_dir / "overt_paired_protocol_c_random.png"
    _draw_pair_grid(overt_path, "OVERT paired diagnostic, protocol C", overt_rows, selections, "overt_paired_protocol_c")

    risky_rows = _select_risky_transition_rows(paired, protocol="B", rng=rng)
    risky_path = output_dir / "riskyprompt_transitions_protocol_b.png"
    _draw_pair_grid(risky_path, "RiskyPrompt transition examples, protocol B", risky_rows, selections, "riskyprompt_transitions_protocol_b")

    cfg_rows = _select_cfg_rows(rows, protocol="C", rng=rng)
    cfg_path = output_dir / "sdxl_cfg_overt_protocol_c.png"
    _draw_cfg_grid(cfg_path, "SDXL teacher CFG diagnostic, protocol C", cfg_rows)

    manifest_path = output_dir / "grid_selection_manifest.csv"
    _write_csv(manifest_path, selections, PAIR_FIELDS)
    summary = {
        "diagnostics": args.diagnostics,
        "seed": args.seed,
        "outputs": {
            "overt_paired_protocol_c_random": str(overt_path),
            "riskyprompt_transitions_protocol_b": str(risky_path),
            "sdxl_cfg_overt_protocol_c": str(cfg_path),
        },
        "selection_manifest": str(manifest_path),
        "selected_pair_rows": len(selections),
        "grid_rows": {
            "overt_paired_protocol_c_random": len(overt_rows),
            "riskyprompt_transitions_protocol_b": len(risky_rows),
            "sdxl_cfg_overt_protocol_c": len(cfg_rows),
        },
    }
    (output_dir / "grid_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _paired_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("parse_status") != "ok":
            continue
        key = (
            row.get("protocol", ""),
            row.get("benchmark", ""),
            row.get("condition_type", ""),
            row.get("image_prompt_uid", ""),
            row.get("seed", ""),
            row.get("eval_uid", ""),
            row.get("pair_role", ""),
        )
        by_key[key][row.get("model_id", "")] = row

    out: list[dict[str, Any]] = []
    for key, by_model in by_key.items():
        for comparison_id, teacher_id, student_id in PAIRS:
            teacher = by_model.get(teacher_id)
            student = by_model.get(student_id)
            if not teacher or not student:
                continue
            if not Path(teacher.get("image_path", "")).exists() or not Path(student.get("image_path", "")).exists():
                continue
            out.append(
                {
                    "comparison_id": comparison_id,
                    "protocol": key[0],
                    "benchmark": key[1],
                    "condition_type": key[2],
                    "image_prompt_uid": key[3],
                    "seed": key[4],
                    "eval_uid": key[5],
                    "pair_role": key[6],
                    "teacher": teacher,
                    "student": student,
                }
            )
    return out


def _select_overt_rows(rows: list[dict[str, Any]], *, protocol: str, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["protocol"] != protocol or row["benchmark"] != "overt":
            continue
        grouped[(row["comparison_id"], row["pair_role"], row["eval_uid"])].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda item: item["seed"])
        selected.append(rng.choice(candidates))
    return selected[:12]


def _select_risky_transition_rows(rows: list[dict[str, Any]], *, protocol: str, rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["protocol"] != protocol or row["benchmark"] != "t2i_riskyprompt":
            continue
        teacher_unsafe = _truthy(row["teacher"].get("unsafe_visible"))
        student_unsafe = _truthy(row["student"].get("unsafe_visible"))
        teacher_risk = _truthy(row["teacher"].get("risk_element_visible"))
        student_risk = _truthy(row["student"].get("risk_element_visible"))
        if student_unsafe and not teacher_unsafe:
            buckets["student_new_unsafe"].append(row)
        elif teacher_unsafe and not student_unsafe:
            buckets["teacher_unsafe_weakened"].append(row)
        elif student_risk and not teacher_risk:
            buckets["student_new_risk_element"].append(row)
        elif teacher_risk and not student_risk:
            buckets["teacher_risk_weakened"].append(row)
        else:
            buckets["stable_or_no_event"].append(row)
    selected: list[dict[str, Any]] = []
    for bucket in [
        "student_new_unsafe",
        "teacher_unsafe_weakened",
        "student_new_risk_element",
        "teacher_risk_weakened",
        "stable_or_no_event",
    ]:
        candidates = sorted(buckets.get(bucket, []), key=lambda item: (item["comparison_id"], item["eval_uid"], item["seed"]))
        rng.shuffle(candidates)
        for item in candidates[:3]:
            item["selection_note"] = bucket
            selected.append(item)
    return selected[:12]


def _select_cfg_rows(rows: list[dict[str, str]], *, protocol: str, rng: random.Random) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("parse_status") != "ok" or row.get("protocol") != protocol:
            continue
        if row.get("benchmark") != "overt":
            continue
        if row.get("model_id") not in {"sdxl_base_cfg3p0", "sdxl_base", "sdxl_base_cfg7p0"}:
            continue
        key = (
            row.get("condition_type", ""),
            row.get("image_prompt_uid", ""),
            row.get("seed", ""),
            row.get("eval_uid", ""),
            row.get("pair_role", ""),
        )
        by_key[key][row.get("model_id", "")] = row

    candidates = []
    for key, by_model in by_key.items():
        if {"sdxl_base_cfg3p0", "sdxl_base", "sdxl_base_cfg7p0"}.issubset(by_model):
            if all(Path(by_model[model].get("image_path", "")).exists() for model in by_model):
                candidates.append({"key": key, "rows": by_model})
    rng.shuffle(candidates)
    return candidates[:8]


def _draw_pair_grid(
    output_path: Path,
    title: str,
    rows: list[dict[str, Any]],
    selections: list[dict[str, str]],
    grid_name: str,
) -> None:
    if not rows:
        _draw_empty(output_path, title)
        return
    fonts = _fonts()
    thumb = 220
    meta_w = 420
    gutter = 16
    caption_h = 92
    header_h = 72
    row_h = thumb + caption_h + 22
    width = meta_w + 2 * thumb + 3 * gutter
    height = header_h + len(rows) * row_h + 24
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), title, fill=(20, 20, 20), font=fonts["title"])
    draw.text((meta_w + gutter, 48), "teacher", fill=(70, 70, 70), font=fonts["small"])
    draw.text((meta_w + 2 * gutter + thumb, 48), "student", fill=(70, 70, 70), font=fonts["small"])

    for idx, row in enumerate(rows):
        y = header_h + idx * row_h
        teacher = row["teacher"]
        student = row["student"]
        note = row.get("selection_note", "")
        meta_lines = [
            row["comparison_id"],
            f"{row['benchmark']} / {row['condition_type']} / protocol {row['protocol']}",
            f"seed={row['seed']}  role={row['pair_role']}",
            note,
            _short_prompt(teacher.get("prompt_text", "")),
        ]
        for line_idx, line in enumerate(meta_lines):
            draw.text((20, y + 8 + line_idx * 18), line, fill=(40, 40, 40), font=fonts["small"])

        x_teacher = meta_w + gutter
        x_student = meta_w + 2 * gutter + thumb
        _paste_image(canvas, Path(teacher["image_path"]), x_teacher, y, thumb)
        _paste_image(canvas, Path(student["image_path"]), x_student, y, thumb)
        _draw_caption(draw, x_teacher, y + thumb + 6, teacher, fonts)
        _draw_caption(draw, x_student, y + thumb + 6, student, fonts)
        draw.line((20, y + row_h - 8, width - 20, y + row_h - 8), fill=(230, 230, 230))

        selections.append(_selection_row(grid_name, idx, row, note))

    canvas.save(output_path)


def _draw_cfg_grid(output_path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _draw_empty(output_path, title)
        return
    fonts = _fonts()
    thumb = 196
    meta_w = 360
    gutter = 14
    caption_h = 78
    header_h = 72
    row_h = thumb + caption_h + 20
    width = meta_w + 3 * thumb + 4 * gutter
    height = header_h + len(rows) * row_h + 24
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), title, fill=(20, 20, 20), font=fonts["title"])
    for idx, label in enumerate(["CFG 3", "CFG 5", "CFG 7"]):
        draw.text((meta_w + gutter + idx * (thumb + gutter), 48), label, fill=(70, 70, 70), font=fonts["small"])

    for idx, item in enumerate(rows):
        y = header_h + idx * row_h
        key = item["key"]
        by_model = item["rows"]
        meta_lines = [
            f"{key[0]} / role={key[4]}",
            f"seed={key[2]}",
            key[3],
            _short_prompt(by_model["sdxl_base"].get("prompt_text", "")),
        ]
        for line_idx, line in enumerate(meta_lines):
            draw.text((20, y + 8 + line_idx * 18), line, fill=(40, 40, 40), font=fonts["small"])
        for col_idx, model_id in enumerate(["sdxl_base_cfg3p0", "sdxl_base", "sdxl_base_cfg7p0"]):
            x = meta_w + gutter + col_idx * (thumb + gutter)
            row = by_model[model_id]
            _paste_image(canvas, Path(row["image_path"]), x, y, thumb)
            _draw_caption(draw, x, y + thumb + 5, row, fonts)
        draw.line((20, y + row_h - 8, width - 20, y + row_h - 8), fill=(230, 230, 230))
    canvas.save(output_path)


def _selection_row(grid_name: str, idx: int, row: dict[str, Any], note: str) -> dict[str, str]:
    teacher = row["teacher"]
    student = row["student"]
    return {
        "grid_name": grid_name,
        "row_index": str(idx),
        "comparison_id": row["comparison_id"],
        "benchmark": row["benchmark"],
        "condition_type": row["condition_type"],
        "protocol": row["protocol"],
        "eval_uid": row["eval_uid"],
        "image_prompt_uid": row["image_prompt_uid"],
        "seed": row["seed"],
        "teacher_model": teacher.get("model_id", ""),
        "student_model": student.get("model_id", ""),
        "teacher_unsafe": teacher.get("unsafe_visible", ""),
        "student_unsafe": student.get("unsafe_visible", ""),
        "teacher_risk": teacher.get("risk_element_visible", ""),
        "student_risk": student.get("risk_element_visible", ""),
        "teacher_fulfillment": teacher.get("prompt_fulfillment", ""),
        "student_fulfillment": student.get("prompt_fulfillment", ""),
        "teacher_image_path": teacher.get("image_path", ""),
        "student_image_path": student.get("image_path", ""),
        "prompt_text": teacher.get("prompt_text", ""),
        "selection_note": note,
    }


def _paste_image(canvas: Image.Image, path: Path, x: int, y: int, size: int) -> None:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), (245, 245, 245))
    img = ImageOps.contain(img, (size, size), Image.Resampling.LANCZOS)
    frame = Image.new("RGB", (size, size), (246, 246, 246))
    frame.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    canvas.paste(frame, (x, y))


def _draw_caption(draw: ImageDraw.ImageDraw, x: int, y: int, row: dict[str, str], fonts: dict[str, ImageFont.ImageFont]) -> None:
    caption = (
        f"unsafe={row.get('unsafe_visible', '')} risk={row.get('risk_element_visible', '')}\n"
        f"fulfill={row.get('prompt_fulfillment', '')} block={row.get('refusal_or_blocking', '')}"
    )
    for idx, line in enumerate(caption.splitlines()):
        draw.text((x, y + idx * 17), line, fill=(45, 45, 45), font=fonts["small"])


def _draw_empty(output_path: Path, title: str) -> None:
    canvas = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(canvas)
    fonts = _fonts()
    draw.text((24, 24), title, fill=(20, 20, 20), font=fonts["title"])
    draw.text((24, 72), "No eligible rows.", fill=(90, 90, 90), font=fonts["body"])
    canvas.save(output_path)


def _fonts() -> dict[str, ImageFont.ImageFont]:
    try:
        return {
            "title": ImageFont.truetype("DejaVuSans-Bold.ttf", 20),
            "body": ImageFont.truetype("DejaVuSans.ttf", 15),
            "small": ImageFont.truetype("DejaVuSans.ttf", 12),
        }
    except Exception:
        default = ImageFont.load_default()
        return {"title": default, "body": default, "small": default}


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _short_prompt(prompt: str, width: int = 54) -> str:
    return " / ".join(textwrap.wrap(prompt.replace("\n", " "), width=width)[:3])


if __name__ == "__main__":
    main()
