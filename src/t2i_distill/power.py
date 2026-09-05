from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .io import read_json, write_csv, write_json

POWER_FIELDS = [
    "rq",
    "comparison_id",
    "family",
    "benchmark",
    "planned_matched_cells",
    "ci_half_width_p50",
    "two_proportion_mde_nominal",
    "two_proportion_mde_bonferroni",
    "design_resolution",
]

RQ_POWER_FIELDS = [
    "rq",
    "benchmark_tests",
    "min_planned_matched_cells",
    "median_planned_matched_cells",
    "max_planned_matched_cells",
    "worst_nominal_mde",
    "worst_bonferroni_mde",
    "underpowered_cells",
]


def audit_statistical_power(
    design_audit: dict[str, Any],
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    baseline_rate: float = 0.50,
    min_matched_cells: int = 100,
    included_rqs: set[str] | None = None,
) -> dict[str, Any]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < target_power < 1:
        raise ValueError("target_power must be between 0 and 1")
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1")

    cells = _benchmark_power_cells(design_audit, included_rqs=included_rqs)
    test_count = max(len(cells), 1)
    nominal_alpha = alpha
    bonferroni_alpha = alpha / test_count
    for cell in cells:
        n = int(cell["planned_matched_cells"])
        cell["ci_half_width_p50"] = _one_proportion_half_width(n, alpha=nominal_alpha, p=baseline_rate)
        cell["two_proportion_mde_nominal"] = _two_proportion_mde(n, alpha=nominal_alpha, target_power=target_power, p=baseline_rate)
        cell["two_proportion_mde_bonferroni"] = _two_proportion_mde(n, alpha=bonferroni_alpha, target_power=target_power, p=baseline_rate)
        cell["design_resolution"] = _resolution_label(float(cell["two_proportion_mde_nominal"]), n, min_matched_cells)

    rq_rows = _rq_power_summary(cells, min_matched_cells=min_matched_cells)
    gates = {
        "design_audit_passed": {
            "pass": bool(design_audit.get("overall_pass")),
            "detail": "Statistical power audit assumes the design matrix audit has passed.",
        },
        "minimum_matched_cells": {
            "pass": all(int(cell["planned_matched_cells"]) >= min_matched_cells for cell in cells),
            "detail": f"Every RQ/pair/benchmark cell should have at least {min_matched_cells} planned matched cells for stable reporting.",
        },
        "inferential_cells_present": {
            "pass": len(cells) > 0,
            "detail": f"benchmark-level inferential cells={len(cells)}",
        },
    }
    return {
        "design_audit_path": design_audit.get("source_path", ""),
        "model_scope": design_audit.get("model_scope", ""),
        "alpha": alpha,
        "target_power": target_power,
        "baseline_rate": baseline_rate,
        "min_matched_cells": min_matched_cells,
        "included_rqs": sorted(included_rqs) if included_rqs else [],
        "benchmark_test_count": len(cells),
        "bonferroni_alpha": bonferroni_alpha,
        "overall_pass": all(gate["pass"] for gate in gates.values()),
        "gates": gates,
        "power_rows": cells,
        "rq_power_summary": rq_rows,
    }


def audit_statistical_power_from_path(
    design_audit_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    design = read_json(design_audit_path)
    design["source_path"] = str(design_audit_path)
    return audit_statistical_power(design, **kwargs)


def write_power_audit_reports(audit: dict[str, Any], output_json: Path, output_markdown: Path | None = None, output_csv: Path | None = None) -> None:
    write_json(output_json, audit)
    if output_csv is not None:
        write_csv(output_csv, audit["power_rows"], POWER_FIELDS)
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_power_audit_markdown(audit), encoding="utf-8")


def render_power_audit_markdown(audit: dict[str, Any]) -> str:
    lines = ["# Statistical Power Audit", ""]
    lines.append(f"- design audit: `{audit['design_audit_path']}`")
    lines.append(f"- benchmark-level tests: {audit['benchmark_test_count']}")
    lines.append(f"- alpha: {audit['alpha']}")
    lines.append(f"- bonferroni alpha: {audit['bonferroni_alpha']:.6g}")
    lines.append(f"- target power: {audit['target_power']}")
    lines.append(f"- baseline rate for conservative MDE: {audit['baseline_rate']}")
    lines.append(f"- overall_pass: {audit['overall_pass']}")
    lines.append("")
    lines.append("## Gates")
    for name, gate in sorted(audit["gates"].items()):
        status = "PASS" if gate["pass"] else "FAIL"
        lines.append(f"- {status} `{name}`: {gate['detail']}")
    lines.append("")
    lines.append("## RQ Summary")
    lines.append("| rq | tests | min n | median n | max n | worst nominal MDE | worst Bonferroni MDE | underpowered |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in audit["rq_power_summary"]:
        lines.append(
            f"| `{row['rq']}` | {row['benchmark_tests']} | {row['min_planned_matched_cells']} | {row['median_planned_matched_cells']} | {row['max_planned_matched_cells']} | {row['worst_nominal_mde']:.4f} | {row['worst_bonferroni_mde']:.4f} | {row['underpowered_cells']} |"
        )
    lines.append("")
    lines.append("## Benchmark Cells")
    lines.append("| rq | comparison_id | benchmark | n | CI half-width | nominal MDE | Bonferroni MDE | resolution |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in audit["power_rows"]:
        lines.append(
            f"| `{row['rq']}` | `{row['comparison_id']}` | `{row['benchmark']}` | {row['planned_matched_cells']} | {row['ci_half_width_p50']:.4f} | {row['two_proportion_mde_nominal']:.4f} | {row['two_proportion_mde_bonferroni']:.4f} | {row['design_resolution']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _benchmark_power_cells(design_audit: dict[str, Any], *, included_rqs: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for rq_row in design_audit.get("rq_pair_design_rows", []):
        rq = str(rq_row.get("rq", ""))
        if rq == "unlearning_optional":
            continue
        if included_rqs is not None and rq not in included_rqs:
            continue
        for benchmark, count in sorted((rq_row.get("planned_matched_cells_by_benchmark") or {}).items()):
            if int(count) <= 0:
                continue
            rows.append(
                {
                    "rq": rq,
                    "comparison_id": str(rq_row.get("comparison_id", "")),
                    "family": str(rq_row.get("family", "")),
                    "benchmark": str(benchmark),
                    "planned_matched_cells": int(count),
                }
            )
    return rows


def _rq_power_summary(rows: list[dict[str, Any]], *, min_matched_cells: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["rq"]), []).append(row)
    out = []
    for rq, items in sorted(grouped.items()):
        ns = sorted(int(row["planned_matched_cells"]) for row in items)
        out.append(
            {
                "rq": rq,
                "benchmark_tests": len(items),
                "min_planned_matched_cells": min(ns),
                "median_planned_matched_cells": _median(ns),
                "max_planned_matched_cells": max(ns),
                "worst_nominal_mde": max(float(row["two_proportion_mde_nominal"]) for row in items),
                "worst_bonferroni_mde": max(float(row["two_proportion_mde_bonferroni"]) for row in items),
                "underpowered_cells": sum(int(row["planned_matched_cells"]) < min_matched_cells for row in items),
            }
        )
    return out


def _one_proportion_half_width(n: int, *, alpha: float, p: float) -> float:
    if n <= 0:
        return math.nan
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return z * math.sqrt(p * (1 - p) / n)


def _two_proportion_mde(n: int, *, alpha: float, target_power: float, p: float) -> float:
    if n <= 0:
        return math.nan
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(target_power)
    return (z_alpha + z_power) * math.sqrt(2 * p * (1 - p) / n)


def _resolution_label(mde: float, n: int, min_matched_cells: int) -> str:
    if n < min_matched_cells:
        return "underpowered"
    if mde <= 0.02:
        return "fine"
    if mde <= 0.05:
        return "moderate"
    return "coarse"


def _median(values: list[int]) -> float:
    if not values:
        return math.nan
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2
