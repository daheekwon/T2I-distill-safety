#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_PAIR = ("flux2_klein_base_4b__flux2_klein_4b", "flux2_klein_base_4b", "flux2_klein_4b")

OUTCOMES = [
    "scene_visible",
    "object_bundle_visible",
    "actor_visible",
    "target_visible",
    "core_action_visible",
    "core_relation_visible",
    "object_to_person_relation_visible",
    "role_match",
    "target_role_match",
    "additional_conditions_visible",
    "illegal_or_harmful_context_visible",
    "visible_unsafe",
]

PAIR_FIELDS = [
    "comparison_id",
    "benchmark",
    "prompt_id",
    "condition_type",
    "pair_id",
    "selection_bucket",
    "outcome",
    "paired_n",
    "teacher_yes",
    "student_yes",
    "teacher_rate",
    "student_rate",
    "delta_student_minus_teacher",
    "both_yes",
    "teacher_yes_student_no",
    "teacher_no_student_yes",
    "both_no",
    "teacher_unclear",
    "student_unclear",
    "prompt_text",
]

CONTROL_FIELDS = [
    "benchmark",
    "control_type",
    "rows",
    "scene_yes",
    "object_bundle_yes",
    "core_relation_yes",
    "role_match_yes",
    "visible_unsafe_yes",
    "unrelated_yes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze component-level relation audit outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Relation Component Audit")
    parser.add_argument("--teacher", default=DEFAULT_PAIR[1])
    parser.add_argument("--student", default=DEFAULT_PAIR[2])
    parser.add_argument("--comparison-id", default=None)
    args = parser.parse_args()

    rows = [recompute_additional_conditions(r) for r in read_csv(Path(args.input)) if r.get("parse_status") == "ok"]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pair = (args.comparison_id or f"{args.teacher}__{args.student}", args.teacher, args.student)
    pair_rows = aggregate_pairs(rows, pair)
    control_rows = aggregate_controls(rows)
    write_csv(outdir / "component_pair_counts.csv", pair_rows, PAIR_FIELDS)
    write_csv(outdir / "component_control_summary.csv", control_rows, CONTROL_FIELDS)
    summary = {
        "input": args.input,
        "title": args.title,
        "rows_ok": len(rows),
        "actual_rows_ok": sum(r.get("control_type") == "actual" for r in rows),
        "control_rows_ok": sum(r.get("control_type") != "actual" for r in rows),
        "pair_count_rows": len(pair_rows),
        "control_rows": len(control_rows),
        "outputs": {
            "component_pair_counts": str(outdir / "component_pair_counts.csv"),
            "component_control_summary": str(outdir / "component_control_summary.csv"),
            "report_md": str(outdir / "component_audit_report.md"),
        },
    }
    (outdir / "component_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (outdir / "component_audit_report.md").write_text(report(summary, pair_rows, control_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def aggregate_pairs(rows: list[dict[str, str]], pair: tuple[str, str, str]) -> list[dict[str, Any]]:
    actual = [r for r in rows if r.get("control_type") == "actual"]
    keys = sorted({(r.get("benchmark", ""), r.get("image_prompt_uid", ""), r.get("condition_type", ""), r.get("pair_id", ""), r.get("selection_bucket", "")) for r in actual})
    out = []
    for benchmark, prompt_id, condition, pair_id, bucket in keys:
        pairs = paired(actual, benchmark, prompt_id, condition, pair)
        if not pairs:
            continue
        prompt_text = pairs[0][0].get("prompt_text", "")
        for outcome in OUTCOMES:
            out.append(count_row(benchmark, prompt_id, condition, pair_id, bucket, outcome, pairs, prompt_text, pair))
    return sorted(out, key=lambda r: (r["benchmark"], r["pair_id"], r["selection_bucket"], r["outcome"]))


def aggregate_controls(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("control_type") == "actual":
            continue
        grouped[(row.get("benchmark", ""), row.get("control_type", ""))].append(row)
    out = []
    for (benchmark, control_type), items in sorted(grouped.items()):
        out.append(
            {
                "benchmark": benchmark,
                "control_type": control_type,
                "rows": len(items),
                "scene_yes": sum(is_yes(r.get("scene_visible")) for r in items),
                "object_bundle_yes": sum(is_yes(r.get("object_bundle_visible")) for r in items),
                "core_relation_yes": sum(is_yes(r.get("core_relation_visible")) for r in items),
                "role_match_yes": sum(is_yes(r.get("role_match")) for r in items),
                "visible_unsafe_yes": sum(is_yes(r.get("visible_unsafe")) for r in items),
                "unrelated_yes": sum(is_yes(r.get("unrelated_output")) for r in items),
            }
        )
    return out


def paired(
    rows: list[dict[str, str]],
    benchmark: str,
    prompt_id: str,
    condition: str,
    pair: tuple[str, str, str],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    by_key: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("benchmark") != benchmark:
            continue
        if row.get("image_prompt_uid") != prompt_id:
            continue
        if row.get("condition_type") != condition:
            continue
        key = (row.get("image_prompt_uid", ""), row.get("eval_uid", ""), row.get("seed", ""))
        by_key[key][row.get("model_id", "")] = row
    _, teacher, student = pair
    return [(models[teacher], models[student]) for models in by_key.values() if teacher in models and student in models]


def count_row(
    benchmark: str,
    prompt_id: str,
    condition: str,
    pair_id: str,
    bucket: str,
    outcome: str,
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    prompt_text: str,
    pair: tuple[str, str, str],
) -> dict[str, Any]:
    both_yes = teacher_yes_student_no = teacher_no_student_yes = both_no = 0
    teacher_unclear = student_unclear = 0
    for teacher, student in pairs:
        t_yes = is_yes(teacher.get(outcome))
        s_yes = is_yes(student.get(outcome))
        teacher_unclear += is_unclear(teacher.get(outcome))
        student_unclear += is_unclear(student.get(outcome))
        if t_yes and s_yes:
            both_yes += 1
        elif t_yes and not s_yes:
            teacher_yes_student_no += 1
        elif not t_yes and s_yes:
            teacher_no_student_yes += 1
        else:
            both_no += 1
    n = len(pairs)
    teacher_yes = both_yes + teacher_yes_student_no
    student_yes = both_yes + teacher_no_student_yes
    return {
        "comparison_id": pair[0],
        "benchmark": benchmark,
        "prompt_id": prompt_id,
        "condition_type": condition,
        "pair_id": pair_id,
        "selection_bucket": bucket,
        "outcome": outcome,
        "paired_n": n,
        "teacher_yes": teacher_yes,
        "student_yes": student_yes,
        "teacher_rate": fmt(teacher_yes / n),
        "student_rate": fmt(student_yes / n),
        "delta_student_minus_teacher": fmt((student_yes - teacher_yes) / n),
        "both_yes": both_yes,
        "teacher_yes_student_no": teacher_yes_student_no,
        "teacher_no_student_yes": teacher_no_student_yes,
        "both_no": both_no,
        "teacher_unclear": teacher_unclear,
        "student_unclear": student_unclear,
        "prompt_text": prompt_text,
    }


def report(summary: dict[str, Any], rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    lines = [
        f"# {summary['title']}",
        "",
        f"- rows ok: {summary['rows_ok']}",
        f"- actual rows ok: {summary['actual_rows_ok']}",
        f"- control rows ok: {summary['control_rows_ok']}",
        "",
        "## Core Component Counts",
        "| benchmark | prompt/condition | outcome | n | teacher->student | delta | switches T yes/S no | switches T no/S yes | unclear T/S |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    important = {
        "object_bundle_visible",
        "core_action_visible",
        "core_relation_visible",
        "object_to_person_relation_visible",
        "role_match",
        "target_role_match",
        "additional_conditions_visible",
        "illegal_or_harmful_context_visible",
        "visible_unsafe",
    }
    for row in rows:
        if row["outcome"] not in important:
            continue
        prompt_label = row["pair_id"] or row["prompt_id"]
        if row["selection_bucket"]:
            prompt_label = f"{prompt_label}/{row['selection_bucket']}"
        lines.append(
            "| "
            f"{row['benchmark']} | {prompt_label} | `{row['outcome']}` | {row['paired_n']} | "
            f"{row['teacher_yes']}->{row['student_yes']} | {row['delta_student_minus_teacher']} | "
            f"{row['teacher_yes_student_no']} | {row['teacher_no_student_yes']} | "
            f"{row['teacher_unclear']}/{row['student_unclear']} |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "| benchmark | control | rows | scene yes | object yes | core relation yes | role match yes | unsafe yes | unrelated yes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in controls:
        lines.append(
            "| "
            f"{row['benchmark']} | `{row['control_type']}` | {row['rows']} | {row['scene_yes']} | "
            f"{row['object_bundle_yes']} | {row['core_relation_yes']} | {row['role_match_yes']} | "
            f"{row['visible_unsafe_yes']} | {row['unrelated_yes']} |"
        )
    lines.append("")
    lines.append("Interpretation rule: object presence, physical relation, role match, and extra-condition fulfillment should not be collapsed into a single action score.")
    lines.append("")
    return "\n".join(lines)


def is_yes(value: Any) -> bool:
    return str(value).strip().lower() == "yes"


def recompute_additional_conditions(row: dict[str, str]) -> dict[str, str]:
    row = dict(row)
    fields = [
        "additional_condition_1_visible",
        "additional_condition_2_visible",
        "additional_condition_3_visible",
    ]
    values = [str(row.get(field, "")).strip().lower() for field in fields]
    applicable = [value for value in values if value and value not in {"not_applicable", "not applicable", "n/a", "na"}]
    if not applicable:
        row["additional_conditions_visible"] = "not_applicable"
    elif any(value == "no" for value in applicable):
        row["additional_conditions_visible"] = "no"
    elif all(value == "yes" for value in applicable):
        row["additional_conditions_visible"] = "yes"
    else:
        row["additional_conditions_visible"] = "unclear"
    return row


def is_unclear(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"unclear", "unknown", "ambiguous"} else 0


def fmt(value: float) -> str:
    return f"{value:.3f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
