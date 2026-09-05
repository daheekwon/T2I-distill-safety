#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


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

DECOMP_FIELDS = [
    "benchmark",
    "condition_type",
    "pair_id",
    "pair_role",
    "protocol",
    "model_id",
    "rows",
    "mean_quality",
    "mean_prompt_fulfillment",
    *[f"rate_{field}" for field in BOOL_FIELDS],
]

PAIR_DELTA_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "condition_type",
    "protocol",
    "paired_n",
    "teacher_risk_rate",
    "student_risk_rate",
    "delta_risk",
    "teacher_scene_rate",
    "student_scene_rate",
    "delta_scene",
    "teacher_action_rate",
    "student_action_rate",
    "delta_action",
    "teacher_prompt_fulfillment",
    "student_prompt_fulfillment",
    "delta_prompt_fulfillment",
    "teacher_unsafe_rate",
    "student_unsafe_rate",
    "delta_unsafe",
    "student_new_risk",
    "weakened_risk",
]

PROTOCOL_FIELDS = [
    "benchmark",
    "model_id",
    "protocol_left",
    "protocol_right",
    "paired_n",
    "unsafe_disagreement_rate",
    "risk_element_disagreement_rate",
    "scene_disagreement_rate",
    "fulfillment_mean_abs_delta",
]

CFG_FIELDS = [
    "protocol",
    "condition_type",
    "pair_role",
    "cfg_value",
    "rows",
    "risk_rate",
    "scene_rate",
    "action_rate",
    "unsafe_rate",
    "mean_prompt_fulfillment",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze deep Qwen safety diagnostics: decomposition, protocol sensitivity, and CFG.")
    parser.add_argument("--diagnostics", default="data/mats20_deep/diagnostics/deep_safety_diagnostics.csv")
    parser.add_argument("--output-dir", default="results/mats20_deep/diagnostics/deep_safety_analysis")
    args = parser.parse_args()

    rows = [row for row in _read_csv(Path(args.diagnostics)) if row.get("parse_status") == "ok"]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    decomp = _decomposition(rows)
    pair_deltas = _pair_deltas(rows)
    protocol = _protocol_sensitivity(rows)
    cfg = _cfg_sensitivity(rows)

    _write_csv(outdir / "behavior_decomposition_rates.csv", decomp, DECOMP_FIELDS)
    _write_csv(outdir / "teacher_student_deep_pair_deltas.csv", pair_deltas, PAIR_DELTA_FIELDS)
    _write_csv(outdir / "protocol_sensitivity.csv", protocol, PROTOCOL_FIELDS)
    _write_csv(outdir / "sdxl_cfg_sensitivity.csv", cfg, CFG_FIELDS)
    figures = _write_figures(outdir, decomp, pair_deltas, cfg)
    summary = {
        "diagnostics": args.diagnostics,
        "input_rows_ok": len(rows),
        "decomposition_rows": len(decomp),
        "pair_delta_rows": len(pair_deltas),
        "protocol_sensitivity_rows": len(protocol),
        "cfg_rows": len(cfg),
        "max_abs_deep_delta_unsafe": max((abs(float(row["delta_unsafe"])) for row in pair_deltas), default=0.0),
        "max_abs_delta_risk_element": max((abs(float(row["delta_risk"])) for row in pair_deltas), default=0.0),
        "max_protocol_unsafe_disagreement": max((float(row["unsafe_disagreement_rate"]) for row in protocol), default=0.0),
        "outputs": {
            "behavior_decomposition_rates": str(outdir / "behavior_decomposition_rates.csv"),
            "teacher_student_deep_pair_deltas": str(outdir / "teacher_student_deep_pair_deltas.csv"),
            "protocol_sensitivity": str(outdir / "protocol_sensitivity.csv"),
            "sdxl_cfg_sensitivity": str(outdir / "sdxl_cfg_sensitivity.csv"),
            "report_md": str(outdir / "deep_safety_analysis_report.md"),
            "figures": figures,
        },
    }
    (outdir / "deep_safety_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (outdir / "deep_safety_analysis_report.md").write_text(_report(summary, pair_deltas, protocol, cfg), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _decomposition(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["benchmark"],
                row["condition_type"],
                row["pair_id"],
                row["pair_role"],
                row["protocol"],
                row["model_id"],
            )
        ].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        benchmark, condition, pair_id, pair_role, protocol, model_id = key
        base = {
            "benchmark": benchmark,
            "condition_type": condition,
            "pair_id": pair_id,
            "pair_role": pair_role,
            "protocol": protocol,
            "model_id": model_id,
            "rows": len(items),
            "mean_quality": _mean_float(items, "quality_score"),
            "mean_prompt_fulfillment": _mean_float(items, "prompt_fulfillment"),
        }
        for field in BOOL_FIELDS:
            base[f"rate_{field}"] = _mean_bool(items, field)
        out.append(base)
    return out


def _pair_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["model_id"].startswith("sdxl_base_cfg"):
            continue
        key = (row["benchmark"], row["condition_type"], row["protocol"], row["image_prompt_uid"], row["eval_uid"], row["seed"])
        by_key[key][row["model_id"]] = row
    grouped: dict[tuple[str, str, str, str, str], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for comparison_id, family, teacher, student in PAIR_DEFS:
        for key, by_model in by_key.items():
            if teacher in by_model and student in by_model:
                benchmark, condition, protocol, *_ = key
                grouped[(comparison_id, family, benchmark, condition, protocol)].append((by_model[teacher], by_model[student]))
    out = []
    for (comparison_id, family, benchmark, condition, protocol), pairs in sorted(grouped.items()):
        if not pairs:
            continue
        n = len(pairs)
        t_risk = [_truthy(t["risk_element_visible"]) for t, _ in pairs]
        s_risk = [_truthy(s["risk_element_visible"]) for _, s in pairs]
        row = {
            "comparison_id": comparison_id,
            "family": family,
            "benchmark": benchmark,
            "condition_type": condition,
            "protocol": protocol,
            "paired_n": n,
            "teacher_risk_rate": _mean(t_risk),
            "student_risk_rate": _mean(s_risk),
            "delta_risk": _mean(s_risk) - _mean(t_risk),
            "teacher_scene_rate": _mean([_truthy(t["common_scene_present"]) for t, _ in pairs]),
            "student_scene_rate": _mean([_truthy(s["common_scene_present"]) for _, s in pairs]),
            "delta_scene": _mean([_truthy(s["common_scene_present"]) for _, s in pairs]) - _mean([_truthy(t["common_scene_present"]) for t, _ in pairs]),
            "teacher_action_rate": _mean([_truthy(t["action_relation_present"]) for t, _ in pairs]),
            "student_action_rate": _mean([_truthy(s["action_relation_present"]) for _, s in pairs]),
            "delta_action": _mean([_truthy(s["action_relation_present"]) for _, s in pairs]) - _mean([_truthy(t["action_relation_present"]) for t, _ in pairs]),
            "teacher_prompt_fulfillment": _mean([_float(t["prompt_fulfillment"]) for t, _ in pairs]),
            "student_prompt_fulfillment": _mean([_float(s["prompt_fulfillment"]) for _, s in pairs]),
            "delta_prompt_fulfillment": _mean([_float(s["prompt_fulfillment"]) for _, s in pairs]) - _mean([_float(t["prompt_fulfillment"]) for t, _ in pairs]),
            "teacher_unsafe_rate": _mean([_unsafe(t) for t, _ in pairs]),
            "student_unsafe_rate": _mean([_unsafe(s) for _, s in pairs]),
            "delta_unsafe": _mean([_unsafe(s) for _, s in pairs]) - _mean([_unsafe(t) for t, _ in pairs]),
            "student_new_risk": sum((not t) and s for t, s in zip(t_risk, s_risk, strict=True)),
            "weakened_risk": sum(t and (not s) for t, s in zip(t_risk, s_risk, strict=True)),
        }
        out.append(row)
    return out


def _protocol_sensitivity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
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
        n = len(pairs)
        out.append(
            {
                "benchmark": benchmark,
                "model_id": model_id,
                "protocol_left": left,
                "protocol_right": right,
                "paired_n": n,
                "unsafe_disagreement_rate": _mean([_unsafe(a) != _unsafe(b) for a, b in pairs]),
                "risk_element_disagreement_rate": _mean([_truthy(a["risk_element_visible"]) != _truthy(b["risk_element_visible"]) for a, b in pairs]),
                "scene_disagreement_rate": _mean([_truthy(a["common_scene_present"]) != _truthy(b["common_scene_present"]) for a, b in pairs]),
                "fulfillment_mean_abs_delta": _mean([abs(_float(a["prompt_fulfillment"]) - _float(b["prompt_fulfillment"])) for a, b in pairs]),
            }
        )
    return out


def _cfg_sensitivity(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    cfg_rows = [row for row in rows if row["benchmark"] == "overt" and (row["model_id"] == "sdxl_base" or row["model_id"].startswith("sdxl_base_cfg"))]
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in cfg_rows:
        cfg = row.get("cfg_value") or ("5" if row["model_id"] == "sdxl_base" else "")
        grouped[(row["protocol"], row["condition_type"], row["pair_role"], cfg)].append(row)
    out = []
    for (protocol, condition, pair_role, cfg), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], float(item[0][3] or 0))):
        out.append(
            {
                "protocol": protocol,
                "condition_type": condition,
                "pair_role": pair_role,
                "cfg_value": cfg,
                "rows": len(items),
                "risk_rate": _mean_bool(items, "risk_element_visible"),
                "scene_rate": _mean_bool(items, "common_scene_present"),
                "action_rate": _mean_bool(items, "action_relation_present"),
                "unsafe_rate": _mean([_unsafe(row) for row in items]),
                "mean_prompt_fulfillment": _mean_float(items, "prompt_fulfillment"),
            }
        )
    return out


def _write_figures(outdir: Path, decomp: list[dict[str, Any]], pair_deltas: list[dict[str, Any]], cfg: list[dict[str, Any]]) -> dict[str, str]:
    figures: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return figures

    overt_c = [row for row in decomp if row["benchmark"] == "overt" and row["protocol"] == "C" and row["model_id"] in {"sdxl_base", "sdxl_lightning_4step", "sd35_large", "sd35_large_turbo", "flux2_klein_base_4b", "flux2_klein_4b"}]
    if overt_c:
        labels = []
        risk = []
        scene = []
        action = []
        fulfill = []
        for row in sorted(overt_c, key=lambda r: (r["condition_type"], r["model_id"])):
            labels.append(f"{row['condition_type'].replace('_', ' ')}\n{row['model_id'].replace('_', ' ')}")
            risk.append(float(row["rate_risk_element_visible"]))
            scene.append(float(row["rate_common_scene_present"]))
            action.append(float(row["rate_action_relation_present"]))
            fulfill.append(float(row["mean_prompt_fulfillment"]))
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5))
        x = range(len(labels))
        width = 0.2
        ax.bar([i - 1.5 * width for i in x], scene, width, label="scene")
        ax.bar([i - 0.5 * width for i in x], action, width, label="action")
        ax.bar([i + 0.5 * width for i in x], risk, width, label="risk")
        ax.bar([i + 1.5 * width for i in x], fulfill, width, label="fulfillment")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("rate / score")
        ax.set_title("OVERT derived pair behavior decomposition (Protocol C)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.legend(ncols=4, fontsize=8)
        fig.tight_layout()
        path = outdir / "overt_behavior_decomposition_protocol_c.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figures["overt_behavior_decomposition_protocol_c"] = str(path)

    if pair_deltas:
        labels = [f"{row['comparison_id'].split('__')[-1]}\n{row['benchmark']} {row['protocol']}" for row in pair_deltas]
        values = [float(row["delta_unsafe"]) for row in pair_deltas]
        colors = ["#b94e48" if value > 0 else "#3a6ea5" for value in values]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.45), 4.5))
        ax.bar(range(len(labels)), values, color=colors)
        ax.axhline(0, color="#222", linewidth=0.8)
        ax.set_ylabel("student - teacher unsafe")
        ax.set_title("Deep diagnostic unsafe deltas by protocol")
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        fig.tight_layout()
        path = outdir / "protocol_unsafe_delta_sensitivity.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figures["protocol_unsafe_delta_sensitivity"] = str(path)

    cfg_c = [row for row in cfg if row["protocol"] == "C" and row["condition_type"] == "unsafe_safety"]
    if cfg_c:
        labels = [f"{row['pair_role']} cfg{row['cfg_value']}" for row in cfg_c]
        values = [float(row["risk_rate"]) for row in cfg_c]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(labels)), values, color="#6f8f72")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("risk element rate")
        ax.set_title("SDXL teacher CFG sensitivity on risky OVERT prompts")
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        path = outdir / "sdxl_cfg_risk_sensitivity_protocol_c.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figures["sdxl_cfg_risk_sensitivity_protocol_c"] = str(path)
    return figures


def _report(summary: dict[str, Any], pair_deltas: list[dict[str, Any]], protocol: list[dict[str, Any]], cfg: list[dict[str, Any]]) -> str:
    lines = ["# Deep Safety Diagnostic Report", ""]
    lines.append(f"- ok diagnostic rows: {summary['input_rows_ok']}")
    lines.append(f"- max abs delta unsafe: {summary['max_abs_deep_delta_unsafe']:.3f}")
    lines.append(f"- max abs delta risk element: {summary['max_abs_delta_risk_element']:.3f}")
    lines.append(f"- max protocol unsafe disagreement: {summary['max_protocol_unsafe_disagreement']:.3f}")
    lines.append("")
    lines.append("## Teacher-Student Deltas")
    lines.append("| pair | benchmark | condition | protocol | n | delta risk | delta scene | delta action | delta fulfillment | delta unsafe |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(pair_deltas, key=lambda r: abs(float(r["delta_unsafe"])), reverse=True):
        lines.append(
            f"| `{row['comparison_id']}` | `{row['benchmark']}` | `{row['condition_type']}` | `{row['protocol']}` | {row['paired_n']} | "
            f"{float(row['delta_risk']):.3f} | {float(row['delta_scene']):.3f} | {float(row['delta_action']):.3f} | "
            f"{float(row['delta_prompt_fulfillment']):.3f} | {float(row['delta_unsafe']):.3f} |"
        )
    lines.append("")
    lines.append("## Protocol Sensitivity")
    lines.append("| benchmark | model | protocols | n | unsafe disagreement | risk disagreement | scene disagreement | fulfillment abs delta |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(protocol, key=lambda r: float(r["unsafe_disagreement_rate"]), reverse=True)[:24]:
        lines.append(
            f"| `{row['benchmark']}` | `{row['model_id']}` | `{row['protocol_left']}/{row['protocol_right']}` | {row['paired_n']} | "
            f"{float(row['unsafe_disagreement_rate']):.3f} | {float(row['risk_element_disagreement_rate']):.3f} | "
            f"{float(row['scene_disagreement_rate']):.3f} | {float(row['fulfillment_mean_abs_delta']):.3f} |"
        )
    if cfg:
        lines.append("")
        lines.append("## SDXL CFG Sensitivity")
        lines.append("| protocol | condition | role | cfg | n | risk | scene | action | unsafe | fulfillment |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in cfg:
            lines.append(
                f"| `{row['protocol']}` | `{row['condition_type']}` | `{row['pair_role']}` | {row['cfg_value']} | {row['rows']} | "
                f"{float(row['risk_rate']):.3f} | {float(row['scene_rate']):.3f} | {float(row['action_rate']):.3f} | "
                f"{float(row['unsafe_rate']):.3f} | {float(row['mean_prompt_fulfillment']):.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _unsafe(row: dict[str, str]) -> bool:
    field = "unsafe_contextual" if row["protocol"] == "C" else "unsafe_visible"
    return _truthy(row.get(field, ""))


def _mean_bool(rows: Iterable[dict[str, str]], field: str) -> float:
    values = [_truthy(row.get(field, "")) for row in rows]
    return _mean(values)


def _mean_float(rows: Iterable[dict[str, str]], field: str) -> float:
    return _mean([_float(row.get(field, "")) for row in rows])


def _mean(values: Iterable[float | bool]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _float(value: str) -> float:
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
