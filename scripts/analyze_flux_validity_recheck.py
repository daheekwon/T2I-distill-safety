#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PAIR = ("flux2_klein_base_4b__flux2_klein_4b", "flux2_klein_base_4b", "flux2_klein_4b")
COUNT_FIELDS = [
    "comparison_id","eval_type","control_type","benchmark","condition_type","seed_scope","paired_n",
    "teacher_visible_unsafe_yes","student_visible_unsafe_yes","delta_visible_unsafe_yes",
    "teacher_risk_relation_yes","student_risk_relation_yes","delta_risk_relation_yes",
    "teacher_target_action_yes","student_target_action_yes","delta_target_action_yes",
    "teacher_scene_yes","student_scene_yes","delta_scene_yes",
    "teacher_refusal_yes","student_refusal_yes","delta_refusal_yes",
]
PROMPT_FIELDS = [
    "eval_type","benchmark","condition_type","seed_scope","image_prompt_uid","axis_id","prompt_text","paired_n",
    "teacher_visible_unsafe_yes","student_visible_unsafe_yes","delta_visible_unsafe_yes",
    "teacher_risk_relation_yes","student_risk_relation_yes","delta_risk_relation_yes",
    "teacher_target_action_yes","student_target_action_yes","delta_target_action_yes",
]
CONTROL_FIELDS = ["eval_type","control_type","rows","visible_unsafe_yes","risk_relation_yes","target_action_yes","scene_yes","unrelated_yes","refusal_yes","ambiguous_unknown_or_yes"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FLUX example-free validity recheck outputs.")
    parser.add_argument("--input", default="data/mats20_deep/diagnostics/flux_validity_recheck.csv")
    parser.add_argument("--output-dir", default="results/mats20_deep/diagnostics/flux_validity_recheck")
    args = parser.parse_args()
    rows = [r for r in read_csv(Path(args.input)) if r.get("parse_status") == "ok"]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    count_rows = aggregate_counts(rows)
    prompt_rows = prompt_concentration(rows)
    control_rows = control_summary(rows)
    write_csv(outdir / "flux_pair_counts.csv", count_rows, COUNT_FIELDS)
    write_csv(outdir / "flux_prompt_concentration.csv", prompt_rows, PROMPT_FIELDS)
    write_csv(outdir / "flux_control_summary.csv", control_rows, CONTROL_FIELDS)
    summary = {
        "input": args.input,
        "rows_ok": len(rows),
        "pair_count_rows": len(count_rows),
        "prompt_rows": len(prompt_rows),
        "control_rows": len(control_rows),
        "outputs": {
            "pair_counts": str(outdir / "flux_pair_counts.csv"),
            "prompt_concentration": str(outdir / "flux_prompt_concentration.csv"),
            "control_summary": str(outdir / "flux_control_summary.csv"),
            "report_md": str(outdir / "flux_validity_recheck_report.md"),
        },
    }
    (outdir / "flux_validity_recheck_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (outdir / "flux_validity_recheck_report.md").write_text(report(summary, count_rows, prompt_rows, control_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def aggregate_counts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for seed_scope, exclude in (("all", set()), ("seed1_2_3", {"0"})):
        for control_type in sorted({r["control_type"] for r in rows}):
            if control_type != "actual":
                continue
            for eval_type in sorted({r["eval_type"] for r in rows}):
                for benchmark, condition in sorted({(r["benchmark"], r["condition_type"]) for r in rows if r["control_type"] == control_type}):
                    pairs = paired(rows, eval_type, control_type, benchmark, condition, exclude)
                    if pairs:
                        out.append(count_row(eval_type, control_type, benchmark, condition, seed_scope, pairs))
    return out


def prompt_concentration(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for seed_scope, exclude in (("all", set()), ("seed1_2_3", {"0"})):
        for eval_type in sorted({r["eval_type"] for r in rows}):
            keys = sorted({(r["benchmark"], r["condition_type"], r["image_prompt_uid"]) for r in rows if r["control_type"] == "actual"})
            for benchmark, condition, image_prompt_uid in keys:
                pairs = paired(rows, eval_type, "actual", benchmark, condition, exclude, image_prompt_uid=image_prompt_uid)
                if not pairs:
                    continue
                base = count_row(eval_type, "actual", benchmark, condition, seed_scope, pairs)
                example = pairs[0][0]
                out.append({
                    "eval_type": eval_type,
                    "benchmark": benchmark,
                    "condition_type": condition,
                    "seed_scope": seed_scope,
                    "image_prompt_uid": image_prompt_uid,
                    "axis_id": example.get("axis_id", ""),
                    "prompt_text": example.get("prompt_text", ""),
                    "paired_n": base["paired_n"],
                    "teacher_visible_unsafe_yes": base["teacher_visible_unsafe_yes"],
                    "student_visible_unsafe_yes": base["student_visible_unsafe_yes"],
                    "delta_visible_unsafe_yes": base["delta_visible_unsafe_yes"],
                    "teacher_risk_relation_yes": base["teacher_risk_relation_yes"],
                    "student_risk_relation_yes": base["student_risk_relation_yes"],
                    "delta_risk_relation_yes": base["delta_risk_relation_yes"],
                    "teacher_target_action_yes": base["teacher_target_action_yes"],
                    "student_target_action_yes": base["student_target_action_yes"],
                    "delta_target_action_yes": base["delta_target_action_yes"],
                })
    return sorted(out, key=lambda r: (r["seed_scope"], r["eval_type"], -abs(float(r["delta_visible_unsafe_yes"])), -abs(float(r["delta_risk_relation_yes"])), r["image_prompt_uid"]))


def paired(rows: list[dict[str, str]], eval_type: str, control_type: str, benchmark: str, condition: str, exclude_seeds: set[str], image_prompt_uid: str | None = None) -> list[tuple[dict[str, str], dict[str, str]]]:
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["eval_type"] != eval_type or row["control_type"] != control_type or row["benchmark"] != benchmark or row["condition_type"] != condition:
            continue
        if row.get("seed") in exclude_seeds:
            continue
        if image_prompt_uid is not None and row.get("image_prompt_uid") != image_prompt_uid:
            continue
        key = (row["image_prompt_uid"], row["eval_uid"], row["seed"], row["control_type"])
        by_key[key][row["model_id"]] = row
    out = []
    _, teacher, student = PAIR
    for models in by_key.values():
        if teacher in models and student in models:
            out.append((models[teacher], models[student]))
    return out


def count_row(eval_type: str, control_type: str, benchmark: str, condition: str, seed_scope: str, pairs: list[tuple[dict[str, str], dict[str, str]]]) -> dict[str, Any]:
    def counts(field: str) -> tuple[int, int, float, float, float]:
        t = sum(is_yes(a.get(field)) for a, _ in pairs)
        s = sum(is_yes(b.get(field)) for _, b in pairs)
        n = len(pairs)
        return t, s, t / n, s / n, (s - t) / n
    tu, su, _, _, du = counts("visible_unsafe")
    tr, sr, _, _, dr = counts("risk_relation_visible")
    ta, sa, _, _, da = counts("target_action_visible")
    ts, ss, _, _, ds = counts("scene_core_visible")
    tf, sf, _, _, df = counts("refusal_or_blocking")
    return {
        "comparison_id": PAIR[0],
        "eval_type": eval_type,
        "control_type": control_type,
        "benchmark": benchmark,
        "condition_type": condition,
        "seed_scope": seed_scope,
        "paired_n": len(pairs),
        "teacher_visible_unsafe_yes": tu,
        "student_visible_unsafe_yes": su,
        "delta_visible_unsafe_yes": du,
        "teacher_risk_relation_yes": tr,
        "student_risk_relation_yes": sr,
        "delta_risk_relation_yes": dr,
        "teacher_target_action_yes": ta,
        "student_target_action_yes": sa,
        "delta_target_action_yes": da,
        "teacher_scene_yes": ts,
        "student_scene_yes": ss,
        "delta_scene_yes": ds,
        "teacher_refusal_yes": tf,
        "student_refusal_yes": sf,
        "delta_refusal_yes": df,
    }


def control_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["control_type"] == "actual":
            continue
        grouped[(row["eval_type"], row["control_type"])].append(row)
    for (eval_type, control_type), items in sorted(grouped.items()):
        out.append({
            "eval_type": eval_type,
            "control_type": control_type,
            "rows": len(items),
            "visible_unsafe_yes": sum(is_yes(r.get("visible_unsafe")) for r in items),
            "risk_relation_yes": sum(is_yes(r.get("risk_relation_visible")) for r in items),
            "target_action_yes": sum(is_yes(r.get("target_action_visible")) for r in items),
            "scene_yes": sum(is_yes(r.get("scene_core_visible")) for r in items),
            "unrelated_yes": sum(is_yes(r.get("unrelated_output")) for r in items),
            "refusal_yes": sum(is_yes(r.get("refusal_or_blocking")) for r in items),
            "ambiguous_unknown_or_yes": sum(r.get("ambiguous", "").lower() in {"yes", "unknown"} for r in items),
        })
    return out


def report(summary: dict[str, Any], counts: list[dict[str, Any]], prompts: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    lines = ["# FLUX Validity Recheck Report", ""]
    lines.append(f"- rows ok: {summary['rows_ok']}")
    lines.append("- evaluator prompt: no filled JSON example answers; tri-state visual predicates use yes/no/unknown/not_applicable.")
    lines.append("- target claim: whether FLUX distilled 4B more often implements visible risky relations than FLUX base 4B on selected diagnostics.")
    lines.append("")
    lines.append("## Pair counts")
    lines.append("| eval | subset | seeds | n | unsafe teacher->student | risk relation teacher->student | target action teacher->student | scene teacher->student |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for r in counts:
        subset = f"{r['benchmark']}/{r['condition_type']}"
        lines.append(f"| `{r['eval_type']}` | {subset} | {r['seed_scope']} | {r['paired_n']} | {r['teacher_visible_unsafe_yes']} -> {r['student_visible_unsafe_yes']} | {r['teacher_risk_relation_yes']} -> {r['student_risk_relation_yes']} | {r['teacher_target_action_yes']} -> {r['student_target_action_yes']} | {r['teacher_scene_yes']} -> {r['student_scene_yes']} |")
    lines.append("")
    lines.append("## Controls")
    lines.append("| eval | control | rows | unsafe yes | risk relation yes | target action yes | scene yes | unrelated yes |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in controls:
        lines.append(f"| `{r['eval_type']}` | `{r['control_type']}` | {r['rows']} | {r['visible_unsafe_yes']} | {r['risk_relation_yes']} | {r['target_action_yes']} | {r['scene_yes']} | {r['unrelated_yes']} |")
    lines.append("")
    lines.append("## Largest selected-prompt contributors")
    lines.append("| eval | seeds | subset | prompt | unsafe T->S | risk T->S | action T->S |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    filtered = [p for p in prompts if p["seed_scope"] == "seed1_2_3" and (abs(float(p["delta_visible_unsafe_yes"])) > 0 or abs(float(p["delta_risk_relation_yes"])) > 0)]
    for p in filtered[:12]:
        prompt = str(p["prompt_text"]).replace("|", " ")[:120]
        subset = f"{p['benchmark']}/{p['condition_type']}"
        lines.append(f"| `{p['eval_type']}` | {p['seed_scope']} | {subset} | {prompt} | {p['teacher_visible_unsafe_yes']} -> {p['student_visible_unsafe_yes']} | {p['teacher_risk_relation_yes']} -> {p['student_risk_relation_yes']} | {p['teacher_target_action_yes']} -> {p['student_target_action_yes']} |")
    lines.append("")
    lines.append("Interpretation rule: if the FLUX increase remains under example-free prompts and controls do not blindly mark prompt-only risks as visible, it becomes evidence for a concrete risky-relation implementation difference in the selected FLUX diagnostics. It still remains a selected-prompt diagnostic, not a population benchmark estimate.")
    lines.append("")
    return "\n".join(lines)


def is_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
