#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAIR_DEFS = [
    ("sdxl_base__sdxl_lightning_4step", "sdxl", "sdxl_base", "sdxl_lightning_4step"),
    ("sd35_large__sd35_large_turbo", "sd35", "sd35_large", "sd35_large_turbo"),
    ("flux2_klein_base_4b__flux2_klein_4b", "flux2_klein", "flux2_klein_base_4b", "flux2_klein_4b"),
]

OUT_FIELDS = [
    "comparison_id",
    "family",
    "protocol",
    "benchmark",
    "image_prompt_uid",
    "selection_bucket",
    "seeds",
    "teacher_unsafe_rate",
    "student_unsafe_rate",
    "delta_unsafe",
    "same_seed_joint_unsafe_rate",
    "shuffled_joint_unsafe_rate_mean",
    "shuffled_joint_unsafe_rate_sd",
    "same_minus_shuffled_joint",
    "same_seed_agreement_rate",
    "preserved_unsafe",
    "weakened_unsafe",
    "student_new_unsafe",
    "both_safe",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate unsafe-rate change from same-seed correspondence preservation.")
    parser.add_argument("--diagnostics", default="data/mats20_deep/diagnostics/deep_safety_diagnostics.csv")
    parser.add_argument("--output-dir", default="results/mats20_deep/diagnostics/seed_correspondence")
    parser.add_argument("--protocols", default="A,B,C")
    parser.add_argument("--benchmark", default="t2i_riskyprompt")
    parser.add_argument("--shuffle-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    rows = [
        row
        for row in _read_csv(Path(args.diagnostics))
        if row.get("parse_status") == "ok"
        and row.get("benchmark") == args.benchmark
        and row.get("protocol") in _split(args.protocols)
    ]
    prompt_rows = _prompt_level(rows, shuffle_iters=args.shuffle_iters, rng=random.Random(args.seed))
    aggregate_rows = _aggregate(prompt_rows)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(outdir / "seed_correspondence_by_prompt.csv", prompt_rows, OUT_FIELDS)
    _write_csv(outdir / "seed_correspondence_aggregate.csv", aggregate_rows, OUT_FIELDS)
    summary = {
        "diagnostics": args.diagnostics,
        "input_rows": len(rows),
        "prompt_rows": len(prompt_rows),
        "aggregate_rows": len(aggregate_rows),
        "benchmark": args.benchmark,
        "shuffle_iters": args.shuffle_iters,
        "max_abs_delta_unsafe": max((abs(float(row["delta_unsafe"])) for row in aggregate_rows), default=0.0),
        "max_abs_same_minus_shuffled_joint": max((abs(float(row["same_minus_shuffled_joint"])) for row in aggregate_rows), default=0.0),
        "outputs": {
            "by_prompt": str(outdir / "seed_correspondence_by_prompt.csv"),
            "aggregate": str(outdir / "seed_correspondence_aggregate.csv"),
            "report_md": str(outdir / "seed_correspondence_report.md"),
        },
    }
    (outdir / "seed_correspondence_report.md").write_text(_report(summary, aggregate_rows), encoding="utf-8")
    (outdir / "seed_correspondence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _prompt_level(rows: list[dict[str, str]], *, shuffle_iters: int, rng: random.Random) -> list[dict[str, Any]]:
    by_prompt: dict[tuple[str, str, str], dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (row["protocol"], row["benchmark"], row["image_prompt_uid"])
        by_prompt[key][row["seed"]][row["model_id"]] = row
    out: list[dict[str, Any]] = []
    for (protocol, benchmark, image_prompt_uid), by_seed in sorted(by_prompt.items()):
        selection_bucket = ""
        for pair_id, family, teacher, student in PAIR_DEFS:
            paired: list[tuple[bool, bool]] = []
            seeds: list[str] = []
            for seed, by_model in sorted(by_seed.items(), key=lambda item: int(item[0])):
                if teacher not in by_model or student not in by_model:
                    continue
                teacher_unsafe = _unsafe(by_model[teacher], protocol)
                student_unsafe = _unsafe(by_model[student], protocol)
                paired.append((teacher_unsafe, student_unsafe))
                seeds.append(seed)
                selection_bucket = by_model[teacher].get("selection_bucket", selection_bucket)
            if len(paired) < 2:
                continue
            out.append(_row(pair_id, family, protocol, benchmark, image_prompt_uid, selection_bucket, seeds, paired, shuffle_iters, rng))
    return out


def _aggregate(prompt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        grouped[(row["comparison_id"], row["family"], row["protocol"], row["benchmark"])].append(row)
    out: list[dict[str, Any]] = []
    for (comparison_id, family, protocol, benchmark), items in sorted(grouped.items()):
        weights = [int(row["seeds"]) for row in items]
        total = sum(weights)
        merged = {
            "comparison_id": comparison_id,
            "family": family,
            "protocol": protocol,
            "benchmark": benchmark,
            "image_prompt_uid": "__aggregate__",
            "selection_bucket": "__all__",
            "seeds": total,
        }
        for field in (
            "teacher_unsafe_rate",
            "student_unsafe_rate",
            "delta_unsafe",
            "same_seed_joint_unsafe_rate",
            "shuffled_joint_unsafe_rate_mean",
            "shuffled_joint_unsafe_rate_sd",
            "same_minus_shuffled_joint",
            "same_seed_agreement_rate",
        ):
            merged[field] = _weighted_mean(items, field, weights)
        for field in ("preserved_unsafe", "weakened_unsafe", "student_new_unsafe", "both_safe"):
            merged[field] = sum(int(row[field]) for row in items)
        out.append(merged)
    return out


def _row(
    comparison_id: str,
    family: str,
    protocol: str,
    benchmark: str,
    image_prompt_uid: str,
    selection_bucket: str,
    seeds: list[str],
    paired: list[tuple[bool, bool]],
    shuffle_iters: int,
    rng: random.Random,
) -> dict[str, Any]:
    n = len(paired)
    transitions = Counter(paired)
    teacher_rate = sum(t for t, _ in paired) / n
    student_rate = sum(s for _, s in paired) / n
    same_joint = sum(t and s for t, s in paired) / n
    shuffled = _shuffle_joint_rates(paired, shuffle_iters, rng)
    shuffled_mean = sum(shuffled) / len(shuffled) if shuffled else 0.0
    shuffled_sd = (sum((x - shuffled_mean) ** 2 for x in shuffled) / len(shuffled)) ** 0.5 if shuffled else 0.0
    return {
        "comparison_id": comparison_id,
        "family": family,
        "protocol": protocol,
        "benchmark": benchmark,
        "image_prompt_uid": image_prompt_uid,
        "selection_bucket": selection_bucket,
        "seeds": n,
        "teacher_unsafe_rate": teacher_rate,
        "student_unsafe_rate": student_rate,
        "delta_unsafe": student_rate - teacher_rate,
        "same_seed_joint_unsafe_rate": same_joint,
        "shuffled_joint_unsafe_rate_mean": shuffled_mean,
        "shuffled_joint_unsafe_rate_sd": shuffled_sd,
        "same_minus_shuffled_joint": same_joint - shuffled_mean,
        "same_seed_agreement_rate": sum(t == s for t, s in paired) / n,
        "preserved_unsafe": transitions[(True, True)],
        "weakened_unsafe": transitions[(True, False)],
        "student_new_unsafe": transitions[(False, True)],
        "both_safe": transitions[(False, False)],
    }


def _shuffle_joint_rates(paired: list[tuple[bool, bool]], iterations: int, rng: random.Random) -> list[float]:
    teachers = [teacher for teacher, _ in paired]
    students = [student for _, student in paired]
    if len(paired) <= 1:
        return []
    out: list[float] = []
    for _ in range(iterations):
        shuffled = students[:]
        rng.shuffle(shuffled)
        out.append(sum(t and s for t, s in zip(teachers, shuffled, strict=True)) / len(paired))
    return out


def _unsafe(row: dict[str, str], protocol: str) -> bool:
    field = "unsafe_contextual" if protocol == "C" else "unsafe_visible"
    return _truthy(row.get(field, ""))


def _weighted_mean(rows: list[dict[str, Any]], field: str, weights: list[int]) -> float:
    total = sum(weights)
    return sum(float(row[field]) * weight for row, weight in zip(rows, weights, strict=True)) / total if total else 0.0


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


def _split(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def _report(summary: dict[str, Any], aggregate_rows: list[dict[str, Any]]) -> str:
    lines = ["# Seed Correspondence Diagnostic", ""]
    lines.append(f"- input rows: {summary['input_rows']}")
    lines.append(f"- prompt-level rows: {summary['prompt_rows']}")
    lines.append(f"- shuffle iterations: {summary['shuffle_iters']}")
    lines.append("")
    lines.append("| pair | protocol | seeds | teacher unsafe | student unsafe | delta | same joint | shuffled joint | same-shuffled | new unsafe | weakened |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in aggregate_rows:
        lines.append(
            f"| `{row['comparison_id']}` | `{row['protocol']}` | {row['seeds']} | "
            f"{float(row['teacher_unsafe_rate']):.3f} | {float(row['student_unsafe_rate']):.3f} | {float(row['delta_unsafe']):.3f} | "
            f"{float(row['same_seed_joint_unsafe_rate']):.3f} | {float(row['shuffled_joint_unsafe_rate_mean']):.3f} | "
            f"{float(row['same_minus_shuffled_joint']):.3f} | {row['student_new_unsafe']} | {row['weakened_unsafe']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
