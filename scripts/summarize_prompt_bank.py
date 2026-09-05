#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from typing import Any

from t2i_distill.config import load_config, unique_models
from t2i_distill.io import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark/axis coverage in the inheritance prompt bank.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--prompt-bank", default="data/prompts/inheritance_prompt_bank.jsonl")
    parser.add_argument("--benchmark-output", default="data/prompts/prompt_benchmark_summary.csv")
    parser.add_argument("--axis-output", default="data/prompts/prompt_axis_summary.csv")
    parser.add_argument("--model-scope", choices=["primary", "all"], default="primary")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    model_count = len(unique_models(cfg, args.model_scope))
    rows = list(iter_jsonl(Path(args.prompt_bank)))

    benchmark_rows = _benchmark_summary(rows, cfg, model_count)
    axis_rows = _axis_summary(rows, cfg, model_count)
    _write_csv(Path(args.benchmark_output), benchmark_rows)
    _write_csv(Path(args.axis_output), axis_rows)
    print({
        "benchmarks": len(benchmark_rows),
        "axes": len(axis_rows),
        "model_count": model_count,
        "benchmark_output": args.benchmark_output,
        "axis_output": args.axis_output,
    })


def _benchmark_summary(rows: list[dict[str, Any]], cfg: dict[str, Any], model_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)
    out = []
    for benchmark in sorted(grouped):
        br = grouped[benchmark]
        cond_counter = collections.Counter(row["condition_type"] for row in br)
        generation_jobs = 0
        eval_rows = 0
        seed_fragments = []
        for condition in sorted(cond_counter):
            cr = [row for row in br if row["condition_type"] == condition]
            seed_key, seed_count = _seed_info(cfg, benchmark, condition)
            generation_jobs += len({row["image_prompt_uid"] for row in cr}) * seed_count * model_count
            eval_rows += len(cr) * seed_count * model_count
            seed_fragments.append(f"{condition}:{seed_key or 'unspecified'}={seed_count}")
        out.append({
            "benchmark": benchmark,
            "records": len(br),
            "unique_image_prompts": len({row["image_prompt_uid"] for row in br}),
            "axis_count": len({row["axis_id"] for row in br}),
            "conditions": ";".join(f"{name}:{count}" for name, count in sorted(cond_counter.items())),
            "seed_policy": ";".join(seed_fragments),
            "primary_generation_jobs_attributed": generation_jobs,
            "primary_eval_rows_attributed": eval_rows,
        })
    return out


def _axis_summary(rows: list[dict[str, Any]], cfg: dict[str, Any], model_count: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        key = (row["benchmark"], row["condition_type"], row["axis_id"], row["axis_name"])
        grouped[key].append(row)
    out = []
    for (benchmark, condition, axis_id, axis_name), ar in sorted(grouped.items()):
        seed_key, seed_count = _seed_info(cfg, benchmark, condition)
        image_count = len({row["image_prompt_uid"] for row in ar})
        out.append({
            "benchmark": benchmark,
            "condition_type": condition,
            "axis_id": axis_id,
            "axis_name": axis_name,
            "records": len(ar),
            "unique_image_prompts": image_count,
            "seed_key": seed_key,
            "seeds_per_image": seed_count,
            "primary_generation_jobs_attributed": image_count * seed_count * model_count,
            "primary_eval_rows_attributed": len(ar) * seed_count * model_count,
            "evaluator": _join_unique(row.get("evaluator", "") for row in ar),
            "complexity_level": _join_unique(row.get("complexity_level", "") for row in ar),
        })
    return out


def _seed_info(cfg: dict[str, Any], benchmark: str, condition: str) -> tuple[str, int]:
    plan = cfg["benchmark_plan"].get(benchmark, {})
    seed_key = plan.get("seed_keys", {}).get(condition, plan.get("seed_keys", {}).get("benchmark_native", ""))
    if not seed_key:
        return "", 1
    return seed_key, len(cfg["seed_sets"][seed_key])


def _join_unique(values) -> str:
    return ";".join(sorted({str(value) for value in values if value}))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
