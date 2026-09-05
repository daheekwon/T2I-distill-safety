#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PAIR_DEFS = [
    ("sdxl_base__sdxl_lightning_4step", "sdxl", "sdxl_base", "sdxl_lightning_4step"),
    ("sd35_large__sd35_large_turbo", "sd35", "sd35_large", "sd35_large_turbo"),
    ("flux2_klein_base_4b__flux2_klein_4b", "flux2_klein", "flux2_klein_base_4b", "flux2_klein_4b"),
]
BOOL_FIELDS = [
    "common_scene_present",
    "main_subject_present",
    "key_object_present",
    "action_relation_present",
    "risk_element_visible",
    "refusal_or_blocking",
    "unsafe_visible",
    "unsafe_contextual",
    "unrelated_output",
    "unclear",
]
PAIR_FIELDS = [
    "comparison_id","family","benchmark","condition_type","protocol","unsafe_outcome_field","paired_n",
    "teacher_unsafe_count","student_unsafe_count","teacher_unsafe_rate","student_unsafe_rate","delta_unsafe",
    "teacher_risk_count","student_risk_count","teacher_risk_rate","student_risk_rate","delta_risk",
    "student_new_unsafe","weakened_unsafe","preserved_unsafe","both_safe",
]
FIELD_FIELDS = ["field","protocol","rows","true_count","false_count","blank_count","unique_values","most_common_value","most_common_count"]
PROTO_FIELDS = ["benchmark","model_id","protocol_left","protocol_right","outcome_field","paired_n","disagreement_count","disagreement_rate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected deep diagnostic aggregation using a fixed unsafe_visible outcome.")
    parser.add_argument("--diagnostics", default="data/mats20_deep/diagnostics/deep_safety_diagnostics.csv")
    parser.add_argument("--output-dir", default="results/mats20_deep/diagnostics/corrected_analysis")
    args = parser.parse_args()
    rows = [r for r in read_csv(Path(args.diagnostics)) if r.get("parse_status") == "ok"]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pair_rows = pair_counts(rows)
    field_rows = field_degeneracy(rows)
    proto_visible = protocol_disagreement(rows, "unsafe_visible")
    proto_contextual = protocol_disagreement(rows, "unsafe_contextual")
    write_csv(outdir / "teacher_student_counts_unsafe_visible.csv", pair_rows, PAIR_FIELDS)
    write_csv(outdir / "field_degeneracy.csv", field_rows, FIELD_FIELDS)
    write_csv(outdir / "protocol_disagreement_unsafe_visible.csv", proto_visible, PROTO_FIELDS)
    write_csv(outdir / "protocol_disagreement_unsafe_contextual.csv", proto_contextual, PROTO_FIELDS)
    summary = {
        "diagnostics": args.diagnostics,
        "rows_ok": len(rows),
        "unsafe_outcome_field": "unsafe_visible",
        "pair_rows": len(pair_rows),
        "field_degeneracy_rows": len(field_rows),
        "max_abs_delta_unsafe_visible": max((abs(float(r["delta_unsafe"])) for r in pair_rows), default=0.0),
        "max_protocol_disagreement_unsafe_visible": max((float(r["disagreement_rate"]) for r in proto_visible), default=0.0),
        "outputs": {
            "pair_counts": str(outdir / "teacher_student_counts_unsafe_visible.csv"),
            "field_degeneracy": str(outdir / "field_degeneracy.csv"),
            "protocol_visible": str(outdir / "protocol_disagreement_unsafe_visible.csv"),
            "protocol_contextual": str(outdir / "protocol_disagreement_unsafe_contextual.csv"),
            "report_md": str(outdir / "corrected_deep_diagnostics_report.md"),
        },
    }
    (outdir / "corrected_deep_diagnostics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (outdir / "corrected_deep_diagnostics_report.md").write_text(report(summary, pair_rows, field_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def pair_counts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["model_id"].startswith("sdxl_base_cfg"):
            continue
        key = (row["benchmark"], row["condition_type"], row["protocol"], row["image_prompt_uid"], row["eval_uid"], row["seed"])
        by_key[key][row["model_id"]] = row
    grouped: dict[tuple[str, str, str, str, str], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for comparison_id, family, teacher, student in PAIR_DEFS:
        for key, models in by_key.items():
            if teacher in models and student in models:
                benchmark, condition, protocol, *_ = key
                grouped[(comparison_id, family, benchmark, condition, protocol)].append((models[teacher], models[student]))
    out: list[dict[str, Any]] = []
    for (comparison_id, family, benchmark, condition, protocol), pairs in sorted(grouped.items()):
        t_unsafe = [truthy(t.get("unsafe_visible")) for t, _ in pairs]
        s_unsafe = [truthy(s.get("unsafe_visible")) for _, s in pairs]
        t_risk = [truthy(t.get("risk_element_visible")) for t, _ in pairs]
        s_risk = [truthy(s.get("risk_element_visible")) for _, s in pairs]
        transitions = Counter(zip(t_unsafe, s_unsafe, strict=True))
        n = len(pairs)
        out.append({
            "comparison_id": comparison_id,
            "family": family,
            "benchmark": benchmark,
            "condition_type": condition,
            "protocol": protocol,
            "unsafe_outcome_field": "unsafe_visible",
            "paired_n": n,
            "teacher_unsafe_count": sum(t_unsafe),
            "student_unsafe_count": sum(s_unsafe),
            "teacher_unsafe_rate": mean(t_unsafe),
            "student_unsafe_rate": mean(s_unsafe),
            "delta_unsafe": mean(s_unsafe) - mean(t_unsafe),
            "teacher_risk_count": sum(t_risk),
            "student_risk_count": sum(s_risk),
            "teacher_risk_rate": mean(t_risk),
            "student_risk_rate": mean(s_risk),
            "delta_risk": mean(s_risk) - mean(t_risk),
            "student_new_unsafe": transitions[(False, True)],
            "weakened_unsafe": transitions[(True, False)],
            "preserved_unsafe": transitions[(True, True)],
            "both_safe": transitions[(False, False)],
        })
    return out


def field_degeneracy(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    fields = BOOL_FIELDS + ["quality_score", "prompt_fulfillment"]
    for protocol in sorted({r.get("protocol", "") for r in rows}):
        sub = [r for r in rows if r.get("protocol") == protocol]
        for field in fields:
            values = [r.get(field, "") for r in sub]
            counts = Counter(values)
            most_value, most_count = counts.most_common(1)[0] if counts else ("", 0)
            out.append({
                "field": field,
                "protocol": protocol,
                "rows": len(values),
                "true_count": sum(truthy(v) for v in values),
                "false_count": sum(str(v).strip().lower() == "false" for v in values),
                "blank_count": sum(str(v).strip() == "" for v in values),
                "unique_values": len(counts),
                "most_common_value": most_value,
                "most_common_count": most_count,
            })
    return out


def protocol_disagreement(rows: list[dict[str, str]], field: str) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_label[row["label_uid"]][row["protocol"]] = row
    grouped: dict[tuple[str, str, str, str], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for by_protocol in by_label.values():
        for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
            if left in by_protocol and right in by_protocol:
                row = by_protocol[left]
                grouped[(row["benchmark"], row["model_id"], left, right)].append((by_protocol[left], by_protocol[right]))
    out = []
    for (benchmark, model_id, left, right), pairs in sorted(grouped.items()):
        disagreements = [truthy(a.get(field)) != truthy(b.get(field)) for a, b in pairs]
        out.append({
            "benchmark": benchmark,
            "model_id": model_id,
            "protocol_left": left,
            "protocol_right": right,
            "outcome_field": field,
            "paired_n": len(pairs),
            "disagreement_count": sum(disagreements),
            "disagreement_rate": mean(disagreements),
        })
    return out


def report(summary: dict[str, Any], pair_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> str:
    lines = ["# Corrected Deep Diagnostics Report", ""]
    lines.append(f"- rows ok: {summary['rows_ok']}")
    lines.append("- unsafe outcome used for pair/protocol summaries: `unsafe_visible` for A, B, and C.")
    lines.append("- `unsafe_contextual` is kept separate; it is not mixed into the headline unsafe rate.")
    lines.append(f"- max |delta unsafe_visible|: {summary['max_abs_delta_unsafe_visible']:.3f}")
    lines.append("")
    lines.append("## Teacher-student visible unsafe counts")
    lines.append("| subset | lineage | A | B | C | n per protocol |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for benchmark, condition, label in (("overt", "unsafe_safety", "OVERT-inspired risky"), ("t2i_riskyprompt", "risky_safety", "RiskyPrompt")):
        for comparison_id, family, _, _ in PAIR_DEFS:
            cells = {r["protocol"]: r for r in pair_rows if r["comparison_id"] == comparison_id and r["benchmark"] == benchmark and r["condition_type"] == condition}
            if not cells:
                continue
            def cell(proto: str) -> str:
                r = cells.get(proto)
                return "" if not r else f"{r['teacher_unsafe_count']} -> {r['student_unsafe_count']}"
            n = next(iter(cells.values()))["paired_n"]
            lines.append(f"| {label} | `{comparison_id}` | {cell('A')} | {cell('B')} | {cell('C')} | {n} |")
    lines.append("")
    lines.append("## Degenerate fields to treat cautiously")
    lines.append("| field | protocol | rows | most common | count | unique values |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: |")
    flagged = [r for r in field_rows if int(r["most_common_count"]) >= max(1, int(r["rows"]) - 1)]
    for r in flagged:
        lines.append(f"| `{r['field']}` | `{r['protocol']}` | {r['rows']} | `{r['most_common_value']}` | {r['most_common_count']} | {r['unique_values']} |")
    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("These corrected counts keep the strongest FLUX safety signal visible, but they do not validate the old scene/action/fulfillment decomposition. Fields that are constant or nearly constant should be treated as evaluator diagnostics, not evidence for preserved scene understanding.")
    lines.append("")
    return "\n".join(lines)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def mean(values) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


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
