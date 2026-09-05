from __future__ import annotations

import random
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import selected_pairs
from .io import ensure_parent, read_csv, read_json, write_csv, write_json
from .safety_mem_analysis import safety_behavior

GRID_SELECTION_FIELDS = [
    "grid_name",
    "row_index",
    "selection_mode",
    "comparison_id",
    "benchmark",
    "condition_type",
    "axis_id",
    "eval_uid",
    "image_prompt_uid",
    "seed",
    "teacher_model",
    "student_model",
    "teacher_behavior",
    "student_behavior",
    "teacher_quality",
    "student_quality",
    "teacher_memorization_score",
    "student_memorization_score",
    "reference_path",
    "teacher_image_path",
    "student_image_path",
    "prompt_text",
    "notes",
]

SAFETY_CONDITIONS = {
    "benign_overrefusal",
    "unsafe_safety",
    "safety_counterfactual_benign",
    "risky_safety",
    "safety_native",
}
MEM_CONDITIONS = {"memorization_trigger", "memorization_control"}


def make_report_grids(
    *,
    labels_path: Path,
    config_path: Path,
    output_dir: Path,
    copy_scores_path: Path | None = None,
    seed: int = 20260905,
    model_scope: str = "primary",
) -> dict[str, Any]:
    rows = read_csv(labels_path)
    config = read_json(config_path)
    copy_by_label_uid = _copy_rows_by_label_uid(copy_scores_path)
    paired = _paired_rows(rows, config, model_scope=model_scope)
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    selections: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}

    safety_random = _select_random_safety_rows(paired, rng)
    outputs["random_safety_stratified"] = str(output_dir / "random_safety_stratified.png")
    _draw_pair_grid(
        output_path=Path(outputs["random_safety_stratified"]),
        title="Random Stratified Safety Pairs",
        rows=safety_random,
        mode="random_stratified",
        include_reference=False,
        copy_by_label_uid=copy_by_label_uid,
        selections=selections,
    )

    safety_events = _select_safety_event_rows(paired, rng)
    outputs["safety_transition_examples"] = str(output_dir / "safety_transition_examples.png")
    _draw_pair_grid(
        output_path=Path(outputs["safety_transition_examples"]),
        title="Safety Transition Examples",
        rows=safety_events,
        mode="event_random",
        include_reference=False,
        copy_by_label_uid=copy_by_label_uid,
        selections=selections,
    )

    mem_trigger = _select_mem_rows(paired, rng, condition="memorization_trigger", per_pair=2)
    outputs["membench_trigger_random_with_reference"] = str(output_dir / "membench_trigger_random_with_reference.png")
    _draw_pair_grid(
        output_path=Path(outputs["membench_trigger_random_with_reference"]),
        title="Random MemBench Trigger Pairs With Reference",
        rows=mem_trigger,
        mode="random_reference_available",
        include_reference=True,
        copy_by_label_uid=copy_by_label_uid,
        selections=selections,
    )

    mem_control = _select_mem_rows(paired, rng, condition="memorization_control", per_pair=2)
    outputs["membench_control_random"] = str(output_dir / "membench_control_random.png")
    _draw_pair_grid(
        output_path=Path(outputs["membench_control_random"]),
        title="Random MemBench Control Pairs",
        rows=mem_control,
        mode="random_control",
        include_reference=False,
        copy_by_label_uid=copy_by_label_uid,
        selections=selections,
    )

    attr_random = _select_attribute_rows(rows, rng, per_benchmark=6)
    outputs["attribute_sanity_random"] = str(output_dir / "attribute_sanity_random.png")
    _draw_single_image_grid(
        output_path=Path(outputs["attribute_sanity_random"]),
        title="Random Attribute/Sanity Labels",
        rows=attr_random,
        selections=selections,
    )

    selection_path = output_dir / "grid_selection_manifest.csv"
    write_csv(selection_path, selections, GRID_SELECTION_FIELDS)
    summary = {
        "labels_path": str(labels_path),
        "config_path": str(config_path),
        "copy_scores_path": str(copy_scores_path or ""),
        "seed": seed,
        "model_scope": model_scope,
        "outputs": outputs,
        "selection_manifest": str(selection_path),
        "selected_rows": len(selections),
        "grid_rows": {
            "random_safety_stratified": len(safety_random),
            "safety_transition_examples": len(safety_events),
            "membench_trigger_random_with_reference": len(mem_trigger),
            "membench_control_random": len(mem_control),
            "attribute_sanity_random": len(attr_random),
        },
    }
    write_json(output_dir / "grid_summary.json", summary)
    return summary


def _paired_rows(rows: list[dict[str, str]], config: dict[str, Any], *, model_scope: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (
            row.get("benchmark", ""),
            row.get("axis_id", ""),
            row.get("condition_type", ""),
            row.get("image_prompt_uid", ""),
            row.get("seed", ""),
            row.get("eval_uid", ""),
        )
        by_key[key][row.get("model_id", "")] = row

    out: list[dict[str, Any]] = []
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        for key, by_model in by_key.items():
            if teacher not in by_model or student not in by_model:
                continue
            left = by_model[teacher]
            right = by_model[student]
            out.append(
                {
                    "comparison_id": pair["comparison_id"],
                    "family": pair["family"],
                    "teacher": left,
                    "student": right,
                    "benchmark": key[0],
                    "axis_id": key[1],
                    "condition_type": key[2],
                    "image_prompt_uid": key[3],
                    "seed": key[4],
                    "eval_uid": key[5],
                }
            )
    return out


def _select_random_safety_rows(paired: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in paired:
        if item["condition_type"] in SAFETY_CONDITIONS and _has_image_pair(item):
            grouped[(item["comparison_id"], item["benchmark"], item["condition_type"])].append(item)
    out = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=_sort_key)
        out.append(rng.choice(candidates))
    return out


def _select_safety_event_rows(paired: list[dict[str, Any]], rng: random.Random, max_rows: int = 12) -> list[dict[str, Any]]:
    priority = [
        "student_new_unsafe",
        "teacher_unsafe_weakened",
        "preserved_unsafe",
        "student_new_refusal",
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in paired:
        if item["condition_type"] not in SAFETY_CONDITIONS or not _has_image_pair(item):
            continue
        event = _safety_transition_name(item)
        if event:
            buckets[event].append(item)
    out = []
    for event in priority:
        candidates = sorted(buckets.get(event, []), key=_sort_key)
        rng.shuffle(candidates)
        out.extend(candidates[: max(1, max_rows // max(len(priority), 1))])
    if len(out) < max_rows:
        remaining = [item for event in priority for item in buckets.get(event, []) if item not in out]
        rng.shuffle(remaining)
        out.extend(remaining[: max_rows - len(out)])
    return out[:max_rows]


def _select_mem_rows(paired: list[dict[str, Any]], rng: random.Random, *, condition: str, per_pair: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in paired:
        if item["benchmark"] != "membench" or item["condition_type"] != condition or not _has_image_pair(item):
            continue
        if condition == "memorization_trigger":
            if not item["teacher"].get("memorization_score") or not item["student"].get("memorization_score"):
                continue
        grouped[item["comparison_id"]].append(item)
    out = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=_sort_key)
        rng.shuffle(candidates)
        out.extend(candidates[:per_pair])
    return out


def _select_attribute_rows(rows: list[dict[str, str]], rng: random.Random, *, per_benchmark: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("benchmark") not in {"grade", "dimcim", "t2isafety"}:
            continue
        if row.get("benchmark") == "t2isafety" and row.get("condition_type") != "social_bias_default":
            continue
        if not row.get("semantic_label") or not Path(row.get("image_path", "")).exists():
            continue
        grouped[row["benchmark"]].append(row)
    out = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda row: (row.get("model_id", ""), row.get("eval_uid", ""), row.get("seed", "")))
        rng.shuffle(candidates)
        out.extend(candidates[:per_benchmark])
    return out


def _draw_pair_grid(
    *,
    output_path: Path,
    title: str,
    rows: list[dict[str, Any]],
    mode: str,
    include_reference: bool,
    copy_by_label_uid: dict[str, dict[str, str]],
    selections: list[dict[str, Any]],
) -> None:
    if not rows:
        _draw_empty(output_path, title)
        return
    fonts = _fonts()
    thumb = 224
    gutter = 18
    caption_h = 86
    meta_w = 420
    columns = ["reference", "teacher", "student"] if include_reference else ["teacher", "student"]
    cell_w = thumb + 14
    row_h = thumb + caption_h + 28
    header_h = 84
    width = meta_w + len(columns) * cell_w + (len(columns) + 1) * gutter
    height = header_h + len(rows) * row_h + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), title, fill=(25, 25, 25), font=fonts["title"])
    draw.text((24, 52), f"selection: {mode}; rows: {len(rows)}", fill=(90, 90, 90), font=fonts["small"])

    for idx, item in enumerate(rows):
        y = header_h + idx * row_h
        teacher = item["teacher"]
        student = item["student"]
        transition = _transition_label(item)
        meta = [
            item["comparison_id"],
            f"{item['benchmark']} / {item['condition_type']}",
            _short_axis(item["axis_id"]),
            f"seed={item['seed']}  {transition}",
        ]
        _draw_wrapped(draw, "\n".join(meta), (24, y + 10), meta_w - 36, fonts["meta"], fill=(30, 30, 30), line_spacing=4)
        x = meta_w + gutter
        ref_path = ""
        if include_reference:
            ref_path = _reference_path_for_pair(item, copy_by_label_uid)
            _paste_thumb(canvas, ref_path, (x, y + 8), thumb)
            _draw_cell_caption(draw, "Reference", ref_path, "", (x, y + thumb + 14), cell_w, fonts)
            x += cell_w + gutter
        for role, row in [("Teacher", teacher), ("Student", student)]:
            _paste_thumb(canvas, row.get("image_path", ""), (x, y + 8), thumb)
            label = _row_label(row)
            score = _row_score(row)
            _draw_cell_caption(draw, role, label, score, (x, y + thumb + 14), cell_w, fonts)
            x += cell_w + gutter
        selections.append(_selection_row(title, idx, mode, item, ref_path))
    ensure_parent(output_path)
    canvas.save(output_path)


def _draw_single_image_grid(
    *,
    output_path: Path,
    title: str,
    rows: list[dict[str, str]],
    selections: list[dict[str, Any]],
) -> None:
    if not rows:
        _draw_empty(output_path, title)
        return
    fonts = _fonts()
    thumb = 196
    cols = 3
    cell_w = 280
    cell_h = 292
    header_h = 84
    width = cols * cell_w + 48
    height = header_h + ((len(rows) + cols - 1) // cols) * cell_h + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), title, fill=(25, 25, 25), font=fonts["title"])
    draw.text((24, 52), f"selection: random_stratified; rows: {len(rows)}", fill=(90, 90, 90), font=fonts["small"])
    for idx, row in enumerate(rows):
        col = idx % cols
        grid_row = idx // cols
        x = 24 + col * cell_w
        y = header_h + grid_row * cell_h
        _paste_thumb(canvas, row.get("image_path", ""), (x, y), thumb)
        text = "\n".join(
            [
                row.get("model_id", ""),
                f"{row.get('benchmark','')} / {_short_axis(row.get('axis_id',''))}",
                f"label={row.get('semantic_label','')}",
                f"q={row.get('quality_score','')}",
            ]
        )
        _draw_wrapped(draw, text, (x, y + thumb + 10), cell_w - 20, fonts["small"], fill=(40, 40, 40), line_spacing=3)
        selections.append(
            {
                "grid_name": title,
                "row_index": idx,
                "selection_mode": "random_attribute",
                "comparison_id": "",
                "benchmark": row.get("benchmark", ""),
                "condition_type": row.get("condition_type", ""),
                "axis_id": row.get("axis_id", ""),
                "eval_uid": row.get("eval_uid", ""),
                "image_prompt_uid": row.get("image_prompt_uid", ""),
                "seed": row.get("seed", ""),
                "teacher_model": "",
                "student_model": row.get("model_id", ""),
                "teacher_behavior": "",
                "student_behavior": row.get("semantic_label", ""),
                "teacher_quality": "",
                "student_quality": row.get("quality_score", ""),
                "teacher_memorization_score": "",
                "student_memorization_score": "",
                "reference_path": "",
                "teacher_image_path": "",
                "student_image_path": row.get("image_path", ""),
                "prompt_text": row.get("prompt_text", ""),
                "notes": row.get("notes", ""),
            }
        )
    ensure_parent(output_path)
    canvas.save(output_path)


def _draw_empty(output_path: Path, title: str) -> None:
    fonts = _fonts()
    canvas = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 24), title, fill=(25, 25, 25), font=fonts["title"])
    draw.text((24, 78), "No eligible rows found.", fill=(90, 90, 90), font=fonts["meta"])
    ensure_parent(output_path)
    canvas.save(output_path)


def _paste_thumb(canvas: Image.Image, path_value: str, xy: tuple[int, int], size: int) -> None:
    path = Path(path_value)
    draw = ImageDraw.Draw(canvas)
    box = (xy[0], xy[1], xy[0] + size, xy[1] + size)
    if not path.exists():
        draw.rectangle(box, outline=(180, 180, 180), width=2)
        draw.text((xy[0] + 12, xy[1] + size // 2 - 8), "missing", fill=(120, 120, 120), font=_fonts()["small"])
        return
    try:
        with Image.open(path) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    except Exception:
        draw.rectangle(box, outline=(180, 60, 60), width=2)
        draw.text((xy[0] + 12, xy[1] + size // 2 - 8), "unreadable", fill=(160, 60, 60), font=_fonts()["small"])
        return
    canvas.paste(thumb, xy)
    draw.rectangle(box, outline=(210, 210, 210), width=1)


def _draw_cell_caption(
    draw: ImageDraw.ImageDraw,
    role: str,
    label: str,
    score: str,
    xy: tuple[int, int],
    width: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    draw.text(xy, role, fill=(20, 20, 20), font=fonts["meta"])
    lines = [line for line in [label, score] if line]
    _draw_wrapped(draw, "\n".join(lines), (xy[0], xy[1] + 24), width - 8, fonts["small"], fill=(70, 70, 70), line_spacing=3)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int,
    font: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int],
    line_spacing: int,
) -> None:
    x, y = xy
    for raw_line in text.splitlines():
        wrapped = _wrap_by_pixels(draw, raw_line, width, font) or [""]
        for line in wrapped:
            draw.text((x, y), line, fill=fill, font=font)
            bbox = draw.textbbox((x, y), line or "Ag", font=font)
            y += bbox[3] - bbox[1] + line_spacing


def _wrap_by_pixels(draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.ImageFont) -> list[str]:
    text = str(text)
    if not text:
        return []
    words = textwrap.wrap(text, width=44, break_long_words=True, break_on_hyphens=False)
    out: list[str] = []
    for chunk in words:
        line = ""
        for word in chunk.split(" "):
            candidate = word if not line else f"{line} {word}"
            if draw.textlength(candidate, font=font) <= width:
                line = candidate
            else:
                if line:
                    out.append(line)
                line = word
        if line:
            out.append(line)
    return out


def _fonts() -> dict[str, ImageFont.ImageFont]:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    try:
        return {
            "title": ImageFont.truetype(str(bold_path), 28),
            "meta": ImageFont.truetype(str(bold_path), 14),
            "small": ImageFont.truetype(str(font_path), 12),
        }
    except OSError:
        font = ImageFont.load_default()
        return {"title": font, "meta": font, "small": font}


def _copy_rows_by_label_uid(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row.get("label_uid", ""): row for row in read_csv(path)}


def _has_image_pair(item: dict[str, Any]) -> bool:
    return Path(item["teacher"].get("image_path", "")).exists() and Path(item["student"].get("image_path", "")).exists()


def _sort_key(item: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        item.get("comparison_id", ""),
        item.get("benchmark", ""),
        item.get("condition_type", ""),
        item.get("axis_id", ""),
        item.get("eval_uid", ""),
        item.get("seed", ""),
    )


def _safety_transition_name(item: dict[str, Any]) -> str:
    teacher = safety_behavior(item["teacher"])
    student = safety_behavior(item["student"])
    if teacher != "unsafe" and student == "unsafe":
        return "student_new_unsafe"
    if teacher == "unsafe" and student != "unsafe":
        return "teacher_unsafe_weakened"
    if teacher == "unsafe" and student == "unsafe":
        return "preserved_unsafe"
    if teacher != "refusal" and student == "refusal":
        return "student_new_refusal"
    if teacher == "refusal" and student != "refusal":
        return "teacher_refusal_released"
    return ""


def _transition_label(item: dict[str, Any]) -> str:
    if item["condition_type"] in SAFETY_CONDITIONS:
        name = _safety_transition_name(item)
        return name or f"{safety_behavior(item['teacher'])}->{safety_behavior(item['student'])}"
    if item["condition_type"] in MEM_CONDITIONS:
        return f"mem {item['teacher'].get('memorization_score','')}->{item['student'].get('memorization_score','')}"
    return ""


def _row_label(row: dict[str, str]) -> str:
    if row.get("unsafe") or row.get("refusal"):
        return f"unsafe={row.get('unsafe','')} refusal={row.get('refusal','')}"
    if row.get("memorization_score"):
        return "copy_score=" + row.get("memorization_score", "")
    if row.get("semantic_label"):
        return "label=" + row.get("semantic_label", "")
    return ""


def _row_score(row: dict[str, str]) -> str:
    parts = []
    if row.get("quality_score"):
        parts.append("q=" + row["quality_score"])
    if row.get("semantic_label_confidence"):
        parts.append("conf=" + row["semantic_label_confidence"])
    return " ".join(parts)


def _reference_path_for_pair(item: dict[str, Any], copy_by_label_uid: dict[str, dict[str, str]]) -> str:
    for row in (item["teacher"], item["student"]):
        copy = copy_by_label_uid.get(row.get("label_uid", ""))
        if copy and copy.get("best_reference_path"):
            return copy["best_reference_path"]
    return ""


def _short_axis(axis_id: str, max_len: int = 58) -> str:
    axis_id = str(axis_id)
    if len(axis_id) <= max_len:
        return axis_id
    return axis_id[: max_len - 1] + "."


def _selection_row(title: str, idx: int, mode: str, item: dict[str, Any], ref_path: str) -> dict[str, Any]:
    teacher = item["teacher"]
    student = item["student"]
    return {
        "grid_name": title,
        "row_index": idx,
        "selection_mode": mode,
        "comparison_id": item.get("comparison_id", ""),
        "benchmark": item.get("benchmark", ""),
        "condition_type": item.get("condition_type", ""),
        "axis_id": item.get("axis_id", ""),
        "eval_uid": item.get("eval_uid", ""),
        "image_prompt_uid": item.get("image_prompt_uid", ""),
        "seed": item.get("seed", ""),
        "teacher_model": teacher.get("model_id", ""),
        "student_model": student.get("model_id", ""),
        "teacher_behavior": safety_behavior(teacher) if item.get("condition_type") in SAFETY_CONDITIONS else "",
        "student_behavior": safety_behavior(student) if item.get("condition_type") in SAFETY_CONDITIONS else "",
        "teacher_quality": teacher.get("quality_score", ""),
        "student_quality": student.get("quality_score", ""),
        "teacher_memorization_score": teacher.get("memorization_score", ""),
        "student_memorization_score": student.get("memorization_score", ""),
        "reference_path": ref_path,
        "teacher_image_path": teacher.get("image_path", ""),
        "student_image_path": student.get("image_path", ""),
        "prompt_text": teacher.get("prompt_text", ""),
        "notes": _transition_label(item),
    }
