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

PAIR = ("flux2_klein_base_4b", "flux2_klein_4b")

FIELDS = [
    "grid_name",
    "row_index",
    "selection_note",
    "eval_type",
    "expression",
    "scene_id",
    "condition_type",
    "seed",
    "teacher_risk_relation",
    "student_risk_relation",
    "teacher_target_action",
    "student_target_action",
    "teacher_visible_unsafe",
    "student_visible_unsafe",
    "teacher_image_path",
    "student_image_path",
    "prompt_text",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create teacher/student grids for relation intervention switches.")
    parser.add_argument("--input", default="data/mats20_deep/relation_intervention/flux_relation_intervention_recheck.csv")
    parser.add_argument("--output-dir", default="results/mats20_deep/relation_intervention/grids")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-rows", type=int, default=10)
    args = parser.parse_args()

    rows = [r for r in read_csv(Path(args.input)) if r.get("parse_status") == "ok" and r.get("control_type") == "actual"]
    pairs = paired_rows(rows, eval_type="predicate")
    selected = select_rows(pairs, random.Random(args.seed), args.max_rows)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid_path = outdir / "relation_switch_examples.png"
    manifest_path = outdir / "relation_switch_examples_manifest.csv"
    draw_grid(grid_path, selected)
    write_csv(manifest_path, manifest_rows(selected), FIELDS)
    summary = {
        "input": args.input,
        "selected_rows": len(selected),
        "grid": str(grid_path),
        "manifest": str(manifest_path),
    }
    (outdir / "grid_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def paired_rows(rows: list[dict[str, str]], *, eval_type: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("eval_type") != eval_type:
            continue
        key = (row.get("image_prompt_uid", ""), row.get("seed", ""))
        by_key[key][row.get("model_id", "")] = row
    teacher_id, student_id = PAIR
    out = []
    for by_model in by_key.values():
        teacher = by_model.get(teacher_id)
        student = by_model.get(student_id)
        if not teacher or not student:
            continue
        if not Path(teacher.get("image_path", "")).exists() or not Path(student.get("image_path", "")).exists():
            continue
        out.append(
            {
                "teacher": teacher,
                "student": student,
                "condition_type": teacher.get("condition_type", ""),
                "expression": teacher.get("selection_bucket", ""),
                "scene_id": teacher.get("pair_id", ""),
                "seed": teacher.get("seed", ""),
                "prompt_text": teacher.get("prompt_text", ""),
                "selection_note": transition_label(teacher, student),
            }
        )
    return out


def transition_label(teacher: dict[str, str], student: dict[str, str]) -> str:
    if teacher.get("condition_type") == "risky_relation":
        t = is_yes(teacher.get("risk_relation_visible"))
        s = is_yes(student.get("risk_relation_visible"))
        if not t and s:
            return "student_new_risky_relation"
        if t and not s:
            return "teacher_risky_relation_weakened"
        if t and s:
            return "both_risky_relation"
        return "neither_risky_relation"
    t = is_yes(teacher.get("target_action_visible"))
    s = is_yes(student.get("target_action_visible"))
    if not t and s:
        return "student_new_benign_action"
    if t and not s:
        return "teacher_benign_action_weakened"
    if t and s:
        return "both_benign_action"
    return "neither_benign_action"


def select_rows(rows: list[dict[str, Any]], rng: random.Random, max_rows: int) -> list[dict[str, Any]]:
    priority = [
        "student_new_risky_relation",
        "teacher_risky_relation_weakened",
        "student_new_benign_action",
        "both_risky_relation",
        "neither_risky_relation",
        "both_benign_action",
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["selection_note"]].append(row)
    selected = []
    for name in priority:
        candidates = sorted(buckets.get(name, []), key=lambda r: (r["expression"], r["scene_id"], int(r["seed"] or 0)))
        rng.shuffle(candidates)
        take = 3 if name in {"student_new_risky_relation", "teacher_risky_relation_weakened"} else 2
        selected.extend(candidates[:take])
    return selected[:max_rows]


def draw_grid(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fonts = fonts_for_grid()
    thumb = 224
    meta_w = 470
    gutter = 16
    header_h = 78
    row_h = thumb + 32
    width = meta_w + 2 * thumb + 3 * gutter
    height = header_h + max(len(rows), 1) * row_h + 24
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), "FLUX relation intervention switch examples", fill=(20, 20, 20), font=fonts["title"])
    draw.text((meta_w + gutter, 52), "teacher: base 4B", fill=(70, 70, 70), font=fonts["small"])
    draw.text((meta_w + 2 * gutter + thumb, 52), "student: distilled 4B", fill=(70, 70, 70), font=fonts["small"])
    if not rows:
        draw.text((20, header_h), "No paired rows available.", fill=(120, 30, 30), font=fonts["body"])
        canvas.save(output_path)
        return
    for idx, row in enumerate(rows):
        y = header_h + idx * row_h
        teacher = row["teacher"]
        student = row["student"]
        meta_lines = [
            row["selection_note"],
            f"{row['condition_type']} | {row['expression']} | {row['scene_id']} | seed {row['seed']}",
            f"risk {teacher.get('risk_relation_visible')} -> {student.get('risk_relation_visible')} | action {teacher.get('target_action_visible')} -> {student.get('target_action_visible')}",
            textwrap.shorten(row["prompt_text"], width=120, placeholder="..."),
        ]
        draw.multiline_text((20, y + 4), "\n".join(wrap_lines(meta_lines, width=60)), fill=(35, 35, 35), font=fonts["body"], spacing=4)
        draw_image(canvas, teacher.get("image_path", ""), (meta_w + gutter, y), thumb)
        draw_image(canvas, student.get("image_path", ""), (meta_w + 2 * gutter + thumb, y), thumb)
    canvas.save(output_path)


def manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for idx, row in enumerate(rows):
        teacher = row["teacher"]
        student = row["student"]
        out.append(
            {
                "grid_name": "relation_switch_examples",
                "row_index": str(idx),
                "selection_note": row["selection_note"],
                "eval_type": teacher.get("eval_type", ""),
                "expression": row["expression"],
                "scene_id": row["scene_id"],
                "condition_type": row["condition_type"],
                "seed": row["seed"],
                "teacher_risk_relation": teacher.get("risk_relation_visible", ""),
                "student_risk_relation": student.get("risk_relation_visible", ""),
                "teacher_target_action": teacher.get("target_action_visible", ""),
                "student_target_action": student.get("target_action_visible", ""),
                "teacher_visible_unsafe": teacher.get("visible_unsafe", ""),
                "student_visible_unsafe": student.get("visible_unsafe", ""),
                "teacher_image_path": teacher.get("image_path", ""),
                "student_image_path": student.get("image_path", ""),
                "prompt_text": row["prompt_text"],
            }
        )
    return out


def draw_image(canvas: Image.Image, path: str, xy: tuple[int, int], size: int) -> None:
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (size, size), (235, 235, 235))
    thumb = ImageOps.contain(image, (size, size))
    background = Image.new("RGB", (size, size), (248, 248, 248))
    background.paste(thumb, ((size - thumb.width) // 2, (size - thumb.height) // 2))
    canvas.paste(background, xy)


def wrap_lines(lines: list[str], width: int) -> list[str]:
    out = []
    for line in lines:
        wrapped = textwrap.wrap(line, width=width) or [""]
        out.extend(wrapped)
    return out


def fonts_for_grid() -> dict[str, ImageFont.ImageFont]:
    def load(size: int) -> ImageFont.ImageFont:
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        return ImageFont.load_default()

    return {"title": load(22), "body": load(14), "small": load(13)}


def is_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
