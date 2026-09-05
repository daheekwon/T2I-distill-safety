#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

PAIR = ("flux2_klein_base_4b", "flux2_klein_4b")
CONDITION_ORDER = [
    "original",
    "segmented_short_same_facts",
    "core_relation_only",
    "paraphrase_same_load",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact grid for the pharmacy relation pilot.")
    parser.add_argument("--input", default="data/mats20_deep/pharmacy_relation_pilot/component_audit.csv")
    parser.add_argument("--output-dir", default="results/mats20_deep/pharmacy_relation_pilot/grids")
    parser.add_argument("--seed", default="13")
    args = parser.parse_args()

    rows = [
        r
        for r in read_csv(Path(args.input))
        if r.get("parse_status") == "ok"
        and r.get("control_type") == "actual"
        and r.get("pair_id") == "pharmacy_robbery"
        and r.get("seed") == str(args.seed)
    ]
    by_condition = pair_rows(rows)
    selected = [by_condition[c] for c in CONDITION_ORDER if c in by_condition]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid_path = outdir / f"pharmacy_relation_pilot_seed{args.seed}.png"
    manifest_path = outdir / f"pharmacy_relation_pilot_seed{args.seed}_manifest.csv"
    draw_grid(grid_path, selected, args.seed)
    write_csv(manifest_path, manifest_rows(selected), [
        "condition_type",
        "seed",
        "teacher_core_relation",
        "student_core_relation",
        "teacher_role_match",
        "student_role_match",
        "teacher_visible_unsafe",
        "student_visible_unsafe",
        "teacher_image_path",
        "student_image_path",
        "prompt_text",
    ])
    summary = {
        "input": args.input,
        "seed": str(args.seed),
        "selected_conditions": [row["condition_type"] for row in selected],
        "grid": str(grid_path),
        "manifest": str(manifest_path),
    }
    (outdir / f"pharmacy_relation_pilot_seed{args.seed}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def pair_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("condition_type", ""), {})[row.get("model_id", "")] = row
    out = {}
    for condition, by_model in grouped.items():
        teacher = by_model.get(PAIR[0])
        student = by_model.get(PAIR[1])
        if not teacher or not student:
            continue
        if not Path(teacher.get("image_path", "")).exists() or not Path(student.get("image_path", "")).exists():
            continue
        out[condition] = {
            "condition_type": condition,
            "teacher": teacher,
            "student": student,
            "seed": teacher.get("seed", ""),
            "prompt_text": teacher.get("prompt_text", ""),
        }
    return out


def draw_grid(output_path: Path, rows: list[dict[str, Any]], seed: str) -> None:
    fonts = fonts_for_grid()
    thumb = 256
    meta_w = 520
    gutter = 18
    header_h = 82
    row_h = thumb + 38
    width = meta_w + 2 * thumb + 3 * gutter
    height = header_h + max(len(rows), 1) * row_h + 24
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), f"Pharmacy relation pilot, seed {seed}", fill=(20, 20, 20), font=fonts["title"])
    draw.text((meta_w + gutter, 54), "teacher: FLUX.2 klein Base 4B", fill=(70, 70, 70), font=fonts["small"])
    draw.text((meta_w + 2 * gutter + thumb, 54), "student: FLUX.2 klein Distilled 4B", fill=(70, 70, 70), font=fonts["small"])
    if not rows:
        draw.text((20, header_h), "No paired rows available.", fill=(120, 30, 30), font=fonts["body"])
        canvas.save(output_path)
        return
    for idx, row in enumerate(rows):
        y = header_h + idx * row_h
        teacher = row["teacher"]
        student = row["student"]
        meta_lines = [
            row["condition_type"],
            f"core relation {teacher.get('core_relation_visible')} -> {student.get('core_relation_visible')}",
            f"role match {teacher.get('role_match')} -> {student.get('role_match')} | unsafe {teacher.get('visible_unsafe')} -> {student.get('visible_unsafe')}",
            textwrap.shorten(row["prompt_text"], width=130, placeholder="..."),
        ]
        draw.multiline_text((20, y + 6), "\n".join(wrap_lines(meta_lines, width=62)), fill=(35, 35, 35), font=fonts["body"], spacing=4)
        draw_image(canvas, teacher.get("image_path", ""), (meta_w + gutter, y), thumb)
        draw_image(canvas, student.get("image_path", ""), (meta_w + 2 * gutter + thumb, y), thumb)
    canvas.save(output_path)


def manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        teacher = row["teacher"]
        student = row["student"]
        out.append(
            {
                "condition_type": row["condition_type"],
                "seed": row["seed"],
                "teacher_core_relation": teacher.get("core_relation_visible", ""),
                "student_core_relation": student.get("core_relation_visible", ""),
                "teacher_role_match": teacher.get("role_match", ""),
                "student_role_match": student.get("role_match", ""),
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
        out.extend(textwrap.wrap(line, width=width) or [""])
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
