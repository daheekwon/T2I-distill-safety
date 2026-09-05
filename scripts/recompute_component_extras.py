#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

FIELDS = [
    "source",
    "component_uid",
    "control_type",
    "model_id",
    "benchmark",
    "condition_type",
    "pair_id",
    "seed",
    "original_additional_conditions_visible",
    "recomputed_additional_conditions_visible",
    "additional_condition_1_visible",
    "additional_condition_2_visible",
    "additional_condition_3_visible",
    "changed",
    "image_path",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute additional_conditions_visible from item-level fields.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "existing_flux=data/mats20_deep/relation_component_audit/existing_flux_riskyprompt_components_strict.csv",
            "pharmacy_pilot=data/mats20_deep/pharmacy_relation_pilot/component_audit_strict.csv",
            "sdxl=data/mats20_deep/relation_component_audit/sdxl_riskyprompt_components_strict.csv",
            "sd35=data/mats20_deep/relation_component_audit/sd35_riskyprompt_components_strict.csv",
        ],
    )
    parser.add_argument("--output", default="data/mats20_deep/relation_component_audit/component_extras_recomputed.csv")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for spec in args.inputs:
        source, path = split_spec(spec)
        for row in read_csv(Path(path)):
            if row.get("parse_status") != "ok" or row.get("control_type") != "actual":
                continue
            original = row.get("additional_conditions_visible", "")
            recomputed = recompute(row)
            out = {field: row.get(field, "") for field in FIELDS}
            out["source"] = source
            out["original_additional_conditions_visible"] = original
            out["recomputed_additional_conditions_visible"] = recomputed
            out["changed"] = str(original != recomputed).lower()
            rows.append(out)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows, FIELDS)
    changed = sum(row["changed"] == "true" for row in rows)
    print({"rows": len(rows), "changed": changed, "output": str(output)})


def split_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, spec
    source, path = spec.split("=", 1)
    return source, path


def recompute(row: dict[str, str]) -> str:
    values = [
        str(row.get("additional_condition_1_visible", "")).strip().lower(),
        str(row.get("additional_condition_2_visible", "")).strip().lower(),
        str(row.get("additional_condition_3_visible", "")).strip().lower(),
    ]
    applicable = [v for v in values if v and v not in {"not_applicable", "not applicable", "n/a", "na"}]
    if not applicable:
        return "not_applicable"
    if any(v == "no" for v in applicable):
        return "no"
    if all(v == "yes" for v in applicable):
        return "yes"
    return "unclear"


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
