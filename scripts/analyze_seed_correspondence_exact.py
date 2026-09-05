#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PAIR_DEFS = [
    ("sdxl_base__sdxl_lightning_4step", "sdxl", "sdxl_base", "sdxl_lightning_4step"),
    ("sd35_large__sd35_large_turbo", "sd35", "sd35_large", "sd35_large_turbo"),
    ("flux2_klein_base_4b__flux2_klein_4b", "flux2_klein", "flux2_klein_base_4b", "flux2_klein_4b"),
]
AGG_FIELDS = [
    "comparison_id","family","protocol","benchmark","seed_scope","prompts","informative_prompts","n_images",
    "teacher_positive","student_positive","observed_overlap","max_possible_overlap","expected_overlap",
    "exact_p_ge_observed","student_new_unsafe","weakened_unsafe","both_safe","preserved_unsafe",
]
PROMPT_FIELDS = [
    "comparison_id","family","protocol","benchmark","seed_scope","image_prompt_uid","selection_bucket","axis_id","prompt_text","n_seeds",
    "teacher_positive","student_positive","observed_overlap","max_possible_overlap","expected_overlap","exact_p_ge_observed",
    "student_new_unsafe","weakened_unsafe","both_safe","preserved_unsafe",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact seed correspondence analysis with prompt-level fixed marginals.")
    parser.add_argument("--diagnostics", default="data/mats20_deep/diagnostics/deep_safety_diagnostics.csv")
    parser.add_argument("--output-dir", default="results/mats20_deep/diagnostics/seed_correspondence_exact")
    parser.add_argument("--benchmark", default="t2i_riskyprompt")
    parser.add_argument("--protocols", default="A,B,C")
    args = parser.parse_args()
    rows = [r for r in read_csv(Path(args.diagnostics)) if r.get("parse_status") == "ok" and r.get("benchmark") == args.benchmark and r.get("protocol") in split(args.protocols)]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_prompt, all_agg = analyze(rows, seed_scope="all", exclude_seeds=set())
    new_prompt, new_agg = analyze(rows, seed_scope="seed1_2_3", exclude_seeds={"0"})
    prompt_rows = all_prompt + new_prompt
    agg_rows = all_agg + new_agg
    write_csv(outdir / "seed_correspondence_exact_by_prompt.csv", prompt_rows, PROMPT_FIELDS)
    write_csv(outdir / "seed_correspondence_exact_aggregate.csv", agg_rows, AGG_FIELDS)
    write_csv(outdir / "flux_prompt_concentration_seed1_2_3.csv", flux_concentration(new_prompt), PROMPT_FIELDS)
    summary = {
        "diagnostics": args.diagnostics,
        "benchmark": args.benchmark,
        "input_rows": len(rows),
        "aggregate_rows": len(agg_rows),
        "prompt_rows": len(prompt_rows),
        "unsafe_outcome_field": "unsafe_visible",
        "outputs": {
            "aggregate": str(outdir / "seed_correspondence_exact_aggregate.csv"),
            "by_prompt": str(outdir / "seed_correspondence_exact_by_prompt.csv"),
            "flux_concentration": str(outdir / "flux_prompt_concentration_seed1_2_3.csv"),
            "report_md": str(outdir / "seed_correspondence_exact_report.md"),
        },
    }
    (outdir / "seed_correspondence_exact_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (outdir / "seed_correspondence_exact_report.md").write_text(report(summary, agg_rows, new_prompt), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def analyze(rows: list[dict[str, str]], *, seed_scope: str, exclude_seeds: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt: dict[tuple[str, str, str], dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("seed") in exclude_seeds:
            continue
        key = (row["protocol"], row["benchmark"], row["image_prompt_uid"])
        by_prompt[key][row["seed"]][row["model_id"]] = row
    prompt_rows: list[dict[str, Any]] = []
    for (protocol, benchmark, image_prompt_uid), by_seed in sorted(by_prompt.items()):
        for comparison_id, family, teacher, student in PAIR_DEFS:
            pairs: list[tuple[bool, bool, dict[str, str]]] = []
            for seed, models in sorted(by_seed.items(), key=lambda item: int(item[0])):
                if teacher in models and student in models:
                    pairs.append((truthy(models[teacher].get("unsafe_visible")), truthy(models[student].get("unsafe_visible")), models[teacher]))
            if len(pairs) < 2:
                continue
            prompt_rows.append(prompt_summary(comparison_id, family, protocol, benchmark, image_prompt_uid, seed_scope, pairs))
    agg_rows = aggregate(prompt_rows, seed_scope)
    return prompt_rows, agg_rows


def prompt_summary(comparison_id: str, family: str, protocol: str, benchmark: str, image_prompt_uid: str, seed_scope: str, pairs: list[tuple[bool, bool, dict[str, str]]]) -> dict[str, Any]:
    n = len(pairs)
    teacher_count = sum(t for t, _, _ in pairs)
    student_count = sum(s for _, s, _ in pairs)
    overlap = sum(t and s for t, s, _ in pairs)
    dist = hypergeom_overlap_dist(n, teacher_count, student_count)
    expected = sum(k * p for k, p in dist.items())
    p_ge = sum(p for k, p in dist.items() if k >= overlap)
    transitions = Counter((t, s) for t, s, _ in pairs)
    example = pairs[0][2]
    return {
        "comparison_id": comparison_id,
        "family": family,
        "protocol": protocol,
        "benchmark": benchmark,
        "seed_scope": seed_scope,
        "image_prompt_uid": image_prompt_uid,
        "selection_bucket": example.get("selection_bucket", ""),
        "axis_id": example.get("axis_id", ""),
        "prompt_text": example.get("prompt_text", ""),
        "n_seeds": n,
        "teacher_positive": teacher_count,
        "student_positive": student_count,
        "observed_overlap": overlap,
        "max_possible_overlap": min(teacher_count, student_count),
        "expected_overlap": expected,
        "exact_p_ge_observed": p_ge,
        "student_new_unsafe": transitions[(False, True)],
        "weakened_unsafe": transitions[(True, False)],
        "both_safe": transitions[(False, False)],
        "preserved_unsafe": transitions[(True, True)],
    }


def aggregate(prompt_rows: list[dict[str, Any]], seed_scope: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        grouped[(row["comparison_id"], row["family"], row["protocol"], row["benchmark"])].append(row)
    out = []
    for (comparison_id, family, protocol, benchmark), items in sorted(grouped.items()):
        total_dist = {0: 1.0}
        for item in items:
            dist = hypergeom_overlap_dist(int(item["n_seeds"]), int(item["teacher_positive"]), int(item["student_positive"]))
            total_dist = convolve(total_dist, dist)
        observed = sum(int(i["observed_overlap"]) for i in items)
        p_ge = sum(p for k, p in total_dist.items() if k >= observed)
        teacher_pos = sum(int(i["teacher_positive"]) for i in items)
        student_pos = sum(int(i["student_positive"]) for i in items)
        out.append({
            "comparison_id": comparison_id,
            "family": family,
            "protocol": protocol,
            "benchmark": benchmark,
            "seed_scope": seed_scope,
            "prompts": len(items),
            "informative_prompts": sum((int(i["teacher_positive"]) > 0 or int(i["student_positive"]) > 0) for i in items),
            "n_images": sum(int(i["n_seeds"]) for i in items),
            "teacher_positive": teacher_pos,
            "student_positive": student_pos,
            "observed_overlap": observed,
            "max_possible_overlap": sum(int(i["max_possible_overlap"]) for i in items),
            "expected_overlap": sum(float(i["expected_overlap"]) for i in items),
            "exact_p_ge_observed": p_ge,
            "student_new_unsafe": sum(int(i["student_new_unsafe"]) for i in items),
            "weakened_unsafe": sum(int(i["weakened_unsafe"]) for i in items),
            "both_safe": sum(int(i["both_safe"]) for i in items),
            "preserved_unsafe": sum(int(i["preserved_unsafe"]) for i in items),
        })
    return out


def flux_concentration(prompt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in prompt_rows if r["family"] == "flux2_klein" and int(r["student_new_unsafe"]) + int(r["preserved_unsafe"]) > 0]
    return sorted(rows, key=lambda r: (r["protocol"], -(int(r["student_new_unsafe"]) + int(r["preserved_unsafe"])), r["image_prompt_uid"]))


def hypergeom_overlap_dist(n: int, a: int, b: int) -> dict[int, float]:
    denom = math.comb(n, b) if 0 <= b <= n else 0
    if denom == 0:
        return {0: 1.0}
    lo = max(0, a + b - n)
    hi = min(a, b)
    return {k: math.comb(a, k) * math.comb(n - a, b - k) / denom for k in range(lo, hi + 1)}


def convolve(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for ka, pa in a.items():
        for kb, pb in b.items():
            out[ka + kb] += pa * pb
    return dict(out)


def report(summary: dict[str, Any], agg_rows: list[dict[str, Any]], prompt_rows: list[dict[str, Any]]) -> str:
    lines = ["# Exact Seed Correspondence Report", ""]
    lines.append(f"- input rows: {summary['input_rows']}")
    lines.append("- outcome: `unsafe_visible` for all protocols.")
    lines.append("- shuffle null: exact prompt-level fixed teacher/student positive counts, convolved across prompts.")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("| lineage | protocol | seeds | prompts | informative | teacher+ | student+ | overlap | max overlap | expected | p_ge | new | weakened |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in agg_rows:
        lines.append(f"| `{r['comparison_id']}` | `{r['protocol']}` | {r['seed_scope']} | {r['prompts']} | {r['informative_prompts']} | {r['teacher_positive']} | {r['student_positive']} | {r['observed_overlap']} | {r['max_possible_overlap']} | {float(r['expected_overlap']):.3f} | {float(r['exact_p_ge_observed']):.3f} | {r['student_new_unsafe']} | {r['weakened_unsafe']} |")
    lines.append("")
    lines.append("## FLUX contributing prompts, seed1_2_3")
    lines.append("| protocol | prompt | teacher+ | student+ | overlap | new | p_ge |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for r in flux_concentration([p for p in prompt_rows if p["seed_scope"] == "seed1_2_3"]):
        prompt = str(r["prompt_text"]).replace("|", " ")[:120]
        lines.append(f"| `{r['protocol']}` | {prompt} | {r['teacher_positive']} | {r['student_positive']} | {r['observed_overlap']} | {r['student_new_unsafe']} | {float(r['exact_p_ge_observed']):.3f} |")
    lines.append("")
    lines.append("Interpretation: rare positives make both 'strong correspondence' and 'no correspondence' underdetermined. Report teacher-positive preservation and student-new positives separately.")
    lines.append("")
    return "\n".join(lines)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


if __name__ == "__main__":
    main()
