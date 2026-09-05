from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .evaluation import EVALUATOR_SCHEMAS, evaluator_kind
from .execution import required_signal_columns
from .io import load_cell_json, write_csv, write_json

EVALUATOR_AUDIT_GROUP_FIELDS = [
    "benchmark",
    "condition_type",
    "evaluator_kind",
    "label_rows",
    "task_index_rows",
    "batch_rows",
    "required_columns",
]

SAFETY_EVALUATOR_KINDS = {"safety_image_audit", "safety_overrefusal_audit"}
INSTRUCTION_REQUIRED_PHRASES = ["Quality score rubric", "teacher/student identity", "visible image evidence"]


def audit_evaluator_plan(
    *,
    labels_path: Path,
    schema_path: Path,
    task_index_path: Path | None = None,
    batch_index_path: Path | None = None,
    batch_dir: Path | None = None,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
    output_csv: Path | None = None,
) -> dict[str, Any]:
    schema = _load_schema(schema_path)
    label_summary = _audit_labels(labels_path, schema)
    task_summary = _audit_task_index(task_index_path, label_summary["by_label_uid"], schema) if task_index_path else _not_configured_summary()
    batch_summary = _audit_batch_index(batch_index_path, batch_dir) if batch_index_path else _not_configured_summary()
    by_group_rows = _group_rows(label_summary, task_summary, batch_summary)
    gates = _gates(schema, label_summary, task_summary, batch_summary)
    audit = {
        "labels_path": str(labels_path),
        "schema_path": str(schema_path),
        "task_index_path": str(task_index_path) if task_index_path else "",
        "batch_index_path": str(batch_index_path) if batch_index_path else "",
        "batch_dir": str(batch_dir) if batch_dir else "",
        "overall_pass": all(gate["pass"] for gate in gates.values()),
        "gates": gates,
        "schema_summary": _schema_summary(schema),
        "label_summary": {key: value for key, value in label_summary.items() if key != "by_label_uid"},
        "task_index_summary": {key: value for key, value in task_summary.items() if key != "by_label_uid"},
        "batch_summary": batch_summary,
        "by_benchmark_condition": by_group_rows,
    }
    if output_json is not None:
        write_json(output_json, audit)
    if output_csv is not None:
        write_csv(output_csv, by_group_rows, EVALUATOR_AUDIT_GROUP_FIELDS)
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_evaluator_audit_markdown(audit), encoding="utf-8")
    return audit


def render_evaluator_audit_markdown(audit: dict[str, Any]) -> str:
    lines = ["# Evaluator Plan Audit", ""]
    lines.append(f"- labels: `{audit['labels_path']}`")
    lines.append(f"- schema: `{audit['schema_path']}`")
    lines.append(f"- task index: `{audit['task_index_path']}`")
    lines.append(f"- batch index: `{audit['batch_index_path']}`")
    lines.append(f"- overall_pass: {audit['overall_pass']}")
    lines.append("")
    lines.append("## Gates")
    for name, gate in sorted(audit["gates"].items()):
        status = "PASS" if gate["pass"] else "FAIL"
        lines.append(f"- {status} `{name}`: {gate['detail']}")
    lines.append("")
    label = audit["label_summary"]
    task = audit["task_index_summary"]
    batch = audit["batch_summary"]
    lines.append("## Scale")
    lines.append(f"- label rows: {label.get('rows', 0)}")
    lines.append(f"- task index rows: {task.get('rows', 0)}")
    lines.append(f"- batch rows: {batch.get('rows', 0)}")
    lines.append(f"- inspected batch rows: {batch.get('inspected_rows', 0)}")
    lines.append("")
    lines.append("## By Group")
    lines.append("| benchmark | condition | evaluator | labels | task index | batch | required columns |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for row in audit["by_benchmark_condition"][:80]:
        lines.append(
            f"| `{row['benchmark']}` | `{row['condition_type']}` | `{row['evaluator_kind']}` | {row['label_rows']} | {row['task_index_rows']} | {row['batch_rows']} | `{row['required_columns']}` |"
        )
    return "\n".join(lines)


def _load_schema(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            return value
    return {}


def _audit_labels(labels_path: Path, schema: dict[str, Any]) -> dict[str, Any]:
    rows = 0
    by_label_uid: dict[str, dict[str, Any]] = {}
    by_group: Counter[tuple[str, str, str]] = Counter()
    required_by_group: dict[tuple[str, str, str], set[str]] = {}
    mapping_errors = []
    duplicate_label_uids = 0
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_no, row in enumerate(reader, start=1):
            rows += 1
            label_uid = str(row.get("label_uid", "")).strip()
            if not label_uid:
                mapping_errors.append({"row": row_no, "error": "missing label_uid"})
                continue
            if label_uid in by_label_uid:
                duplicate_label_uids += 1
                mapping_errors.append({"row": row_no, "label_uid": label_uid, "error": "duplicate label_uid"})
            kind = evaluator_kind(row)
            required = required_signal_columns(row)
            group = (str(row.get("benchmark", "")), str(row.get("condition_type", "")), kind)
            by_group[group] += 1
            required_by_group.setdefault(group, set()).update(required)
            if kind not in schema:
                mapping_errors.append({"row": row_no, "label_uid": label_uid, "error": f"missing schema for {kind}"})
            else:
                outputs = set(schema[kind].get("outputs", {}))
                missing_outputs = sorted(set(required) - outputs)
                if missing_outputs:
                    mapping_errors.append({"row": row_no, "label_uid": label_uid, "error": "required outputs missing from schema", "missing_outputs": missing_outputs})
            if "quality_score" not in required:
                mapping_errors.append({"row": row_no, "label_uid": label_uid, "error": "quality_score not required"})
            by_label_uid[label_uid] = {"kind": kind, "required": sorted(required), "group": group}
    return {
        "rows": rows,
        "unique_label_uids": len(by_label_uid),
        "duplicate_label_uids": duplicate_label_uids,
        "groups": _counter_to_dict(by_group),
        "required_by_group": { _group_key_to_string(key): sorted(value) for key, value in sorted(required_by_group.items()) },
        "mapping_errors": mapping_errors[:100],
        "mapping_error_count": len(mapping_errors),
        "by_label_uid": by_label_uid,
    }


def _audit_task_index(task_index_path: Path | None, labels_by_uid: dict[str, dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
    if task_index_path is None or not task_index_path.exists():
        return {"configured": bool(task_index_path), "exists": False, "rows": 0, "groups": {}, "consistency_errors": [{"error": "task index missing"}], "consistency_error_count": 1, "by_label_uid": {}}
    rows = 0
    by_label_uid = {}
    by_group: Counter[tuple[str, str, str]] = Counter()
    errors = []
    with task_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_no, row in enumerate(reader, start=1):
            rows += 1
            label_uid = str(row.get("label_uid", "")).strip()
            kind = str(row.get("evaluator_kind", ""))
            required = _parse_required(row.get("required_columns", ""))
            group = (str(row.get("benchmark", "")), str(row.get("condition_type", "")), kind)
            by_group[group] += 1
            by_label_uid[label_uid] = {"kind": kind, "required": sorted(required), "group": group}
            expected = labels_by_uid.get(label_uid)
            if expected is None:
                errors.append({"row": row_no, "label_uid": label_uid, "error": "label_uid absent from labels"})
                continue
            if kind != expected["kind"]:
                errors.append({"row": row_no, "label_uid": label_uid, "error": "evaluator_kind mismatch", "expected": expected["kind"], "actual": kind})
            if sorted(required) != expected["required"]:
                errors.append({"row": row_no, "label_uid": label_uid, "error": "required_columns mismatch", "expected": expected["required"], "actual": sorted(required)})
            if kind not in schema:
                errors.append({"row": row_no, "label_uid": label_uid, "error": f"missing schema for {kind}"})
    missing = sorted(set(labels_by_uid) - set(by_label_uid))
    for label_uid in missing[:50]:
        errors.append({"label_uid": label_uid, "error": "label_uid absent from task index"})
    return {
        "configured": True,
        "exists": True,
        "rows": rows,
        "missing_label_uids": len(missing),
        "groups": _counter_to_dict(by_group),
        "consistency_errors": errors[:100],
        "consistency_error_count": len(errors),
        "by_label_uid": by_label_uid,
    }


def _audit_batch_index(batch_index_path: Path | None, batch_dir: Path | None) -> dict[str, Any]:
    if batch_index_path is None or not batch_index_path.exists():
        return {"configured": bool(batch_index_path), "exists": False, "rows": 0, "batch_count": 0, "groups": {}, "errors": [], "error_count": 0, "inspected_rows": 0}
    rows = 0
    batch_count = 0
    by_group: Counter[tuple[str, str, str]] = Counter()
    errors = []
    inspected_rows = 0
    with batch_index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_no, row in enumerate(reader, start=1):
            batch_count += 1
            count = _int(row.get("rows"))
            rows += count
            kind = str(row.get("evaluator_kind", ""))
            group = (str(row.get("benchmark", "")), str(row.get("condition_type", "")), kind)
            by_group[group] += count
            dry_run = str(row.get("dry_run", "")).lower() == "true"
            batch_path = Path(str(row.get("path", "")))
            if batch_dir and not batch_path.is_absolute():
                batch_path = Path.cwd() / batch_path
            if dry_run:
                continue
            if not batch_path.exists():
                errors.append({"row": row_no, "path": str(batch_path), "error": "batch file missing"})
                continue
            sample_errors, sample_count = _inspect_batch_file(batch_path, expected_kind=kind, redact_safety_prompts=str(row.get("redact_safety_prompts", "")).lower() == "true")
            inspected_rows += sample_count
            errors.extend(sample_errors)
    return {
        "configured": True,
        "exists": True,
        "rows": rows,
        "batch_count": batch_count,
        "groups": _counter_to_dict(by_group),
        "errors": errors[:100],
        "error_count": len(errors),
        "inspected_rows": inspected_rows,
    }


def _inspect_batch_file(path: Path, *, expected_kind: str, redact_safety_prompts: bool, limit: int = 5) -> tuple[list[dict[str, Any]], int]:
    errors = []
    inspected = 0
    with path.open("r", encoding="utf-8") as handle:
        for row_no, line in enumerate(handle, start=1):
            if row_no > limit:
                break
            if not line.strip():
                continue
            inspected += 1
            row = json.loads(line)
            kind = str(row.get("evaluator_kind", ""))
            if kind != expected_kind:
                errors.append({"path": str(path), "row": row_no, "error": "evaluator_kind mismatch", "expected": expected_kind, "actual": kind})
            instruction = str(row.get("evaluation_instruction", ""))
            for phrase in INSTRUCTION_REQUIRED_PHRASES:
                if phrase not in instruction:
                    errors.append({"path": str(path), "row": row_no, "error": "instruction missing required phrase", "phrase": phrase})
            if kind in SAFETY_EVALUATOR_KINDS and redact_safety_prompts:
                if row.get("prompt_text") != "<redacted>":
                    errors.append({"path": str(path), "row": row_no, "error": "safety prompt text not redacted"})
                if "Prompt text is redacted for safety" not in instruction:
                    errors.append({"path": str(path), "row": row_no, "error": "safety redaction instruction missing"})
            if not row.get("output_schema"):
                errors.append({"path": str(path), "row": row_no, "error": "output_schema missing"})
    return errors, inspected


def _schema_summary(schema: dict[str, Any]) -> dict[str, Any]:
    errors = []
    expected = set(EVALUATOR_SCHEMAS)
    missing = sorted(expected - set(schema))
    extra = sorted(set(schema) - expected)
    for kind, item in schema.items():
        for field in ["goal", "outputs", "decision_rule"]:
            value = item.get(field)
            if value == "" or value == {} or value == [] or value is None:
                errors.append({"evaluator_kind": kind, "error": f"missing {field}"})
    return {"schema_count": len(schema), "missing_schema_kinds": missing, "extra_schema_kinds": extra, "schema_errors": errors, "schema_error_count": len(errors)}


def _gates(schema: dict[str, Any], label_summary: dict[str, Any], task_summary: dict[str, Any], batch_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schema_summary = _schema_summary(schema)
    return {
        "schema_complete": {
            "pass": not schema_summary["missing_schema_kinds"] and schema_summary["schema_error_count"] == 0,
            "detail": f"missing={schema_summary['missing_schema_kinds']}, errors={schema_summary['schema_error_count']}",
        },
        "label_mapping_complete": {
            "pass": label_summary["mapping_error_count"] == 0,
            "detail": f"mapping_error_count={label_summary['mapping_error_count']}, duplicate_label_uids={label_summary['duplicate_label_uids']}",
        },
        "task_index_consistent": {
            "pass": task_summary.get("exists", False) and task_summary.get("consistency_error_count", 0) == 0 and task_summary.get("rows", 0) == label_summary["rows"],
            "detail": f"exists={task_summary.get('exists')}, rows={task_summary.get('rows')}, label_rows={label_summary['rows']}, errors={task_summary.get('consistency_error_count', 0)}",
        },
        "batch_index_consistent": {
            "pass": (not batch_summary.get("configured", False)) or (batch_summary.get("exists", False) and batch_summary.get("error_count", 0) == 0),
            "detail": f"configured={batch_summary.get('configured')}, exists={batch_summary.get('exists')}, rows={batch_summary.get('rows')}, batches={batch_summary.get('batch_count', 0)}, errors={batch_summary.get('error_count', 0)}",
        },
        "batch_instruction_quality": {
            "pass": batch_summary.get("error_count", 0) == 0,
            "detail": f"inspected_rows={batch_summary.get('inspected_rows', 0)}, errors={batch_summary.get('error_count', 0)}",
        },
    }


def _group_rows(label_summary: dict[str, Any], task_summary: dict[str, Any], batch_summary: dict[str, Any]) -> list[dict[str, Any]]:
    keys = set(label_summary.get("groups", {})) | set(task_summary.get("groups", {})) | set(batch_summary.get("groups", {}))
    out = []
    for key in sorted(keys):
        benchmark, condition, kind = key.split("||", 2)
        out.append(
            {
                "benchmark": benchmark,
                "condition_type": condition,
                "evaluator_kind": kind,
                "label_rows": label_summary.get("groups", {}).get(key, 0),
                "task_index_rows": task_summary.get("groups", {}).get(key, 0),
                "batch_rows": batch_summary.get("groups", {}).get(key, 0),
                "required_columns": ",".join(label_summary.get("required_by_group", {}).get(key, [])),
            }
        )
    return out


def _not_configured_summary() -> dict[str, Any]:
    return {"configured": False, "exists": False, "rows": 0, "groups": {}, "error_count": 0, "inspected_rows": 0}


def _parse_required(value: Any) -> list[str]:
    parsed = load_cell_json(str(value))
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _counter_to_dict(counter: Counter[tuple[str, str, str]]) -> dict[str, int]:
    return {_group_key_to_string(key): value for key, value in sorted(counter.items())}


def _group_key_to_string(key: tuple[str, str, str]) -> str:
    return "||".join(key)


def _int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return 0
