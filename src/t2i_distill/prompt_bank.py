from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .io import parse_literal_list, read_csv, read_json, stable_hash, write_jsonl


def build_prompt_bank(
    config: dict[str, Any],
    benchmarks_root: Path,
    output_path: Path,
    benchmarks: set[str] | None = None,
    include_gated: bool = False,
) -> dict[str, Any]:
    rows = list(iter_prompt_records(config, benchmarks_root, benchmarks, include_gated))
    count = write_jsonl(output_path, rows)
    return summarize_prompt_records(rows) | {"output_path": str(output_path), "records": count}


def iter_prompt_records(
    config: dict[str, Any],
    benchmarks_root: Path,
    benchmarks: set[str] | None = None,
    include_gated: bool = False,
) -> Iterable[dict[str, Any]]:
    loaders = {
        "grade": load_grade,
        "dimcim": load_dimcim,
        "geneval2": load_geneval2,
        "t2i_compbench": load_t2i_compbench,
        "worldgenbench": load_worldgenbench,
        "t2i_factualbench": load_t2i_factualbench,
        "overt": load_overt,
        "t2i_riskyprompt": load_t2i_riskyprompt,
        "t2isafety": load_t2isafety,
        "membench": load_membench,
        "hub": load_hub,
        "emma": load_emma,
    }
    if benchmarks is None:
        selected = {
            benchmark
            for benchmark in loaders
            if config["benchmark_plan"].get(benchmark, {}).get("enabled", True)
        }
    else:
        selected = benchmarks
    for benchmark in [name for name in loaders if name in selected]:
        if benchmark not in loaders:
            raise KeyError(f"unknown benchmark loader: {benchmark}")
        bench_cfg = config["benchmark_plan"][benchmark]
        if bench_cfg.get("requires_valid_lineage") and not include_gated:
            continue
        yield from loaders[benchmark](benchmarks_root, bench_cfg)


def make_record(
    *,
    benchmark: str,
    source_path: Path,
    source_row_id: str,
    prompt_text: str,
    axis_id: str,
    axis_name: str,
    condition_type: str,
    analysis_layers: list[str],
    complexity_level: str,
    source: str,
    target_behavior: str | None = None,
    candidate_behaviors: list[str] | None = None,
    evaluator: str | None = None,
    requires_valid_lineage: bool = False,
    metadata: dict[str, Any] | None = None,
    image_source_id: str | None = None,
) -> dict[str, Any]:
    prompt_text = str(prompt_text).strip()
    if not prompt_text:
        raise ValueError(f"empty prompt in {source_path}:{source_row_id}")
    image_key = image_source_id if image_source_id is not None else source_row_id
    image_prompt_uid = f"{benchmark}:{stable_hash(source_path, image_key, prompt_text)}"
    eval_uid = f"{benchmark}:{stable_hash(source_path, source_row_id, axis_id, condition_type, target_behavior)}"
    return {
        "eval_uid": eval_uid,
        "image_prompt_uid": image_prompt_uid,
        "benchmark": benchmark,
        "source": source,
        "source_path": str(source_path),
        "source_row_id": source_row_id,
        "prompt_text": prompt_text,
        "axis_id": axis_id,
        "axis_name": axis_name,
        "condition_type": condition_type,
        "analysis_layers": analysis_layers,
        "complexity_level": complexity_level,
        "target_behavior": target_behavior or "",
        "candidate_behaviors": candidate_behaviors or [],
        "evaluator": evaluator or "",
        "requires_valid_lineage": requires_valid_lineage,
        "metadata": metadata or {},
    }


def load_grade(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    path = root / "GRADE/datasets/grade_dataset.csv"
    for idx, row in enumerate(read_csv(path)):
        concept = row["concept"].strip()
        attribute = row["attribute"].strip()
        axis_id = f"grade/{row['concept_id']}/{row['attribute_id']}"
        yield make_record(
            benchmark="grade",
            source_path=path,
            source_row_id=str(idx),
            image_source_id=row["prompt_id"],
            prompt_text=row["prompt"],
            axis_id=axis_id,
            axis_name=f"{concept}:{attribute}",
            condition_type="broad",
            analysis_layers=cfg["analysis_layers"],
            complexity_level="atomic",
            source="official_grade_dataset",
            candidate_behaviors=parse_literal_list(row.get("attribute_values")),
            evaluator="GRADE VQA attribute extraction",
            metadata={
                "concept_id": row["concept_id"],
                "concept": concept,
                "prompt_id": row["prompt_id"],
                "attribute_id": row["attribute_id"],
                "attribute": attribute,
            },
        )


def load_dimcim(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    base = root / "DIMCIM/COCO-DIMCIM"
    attr_dir = base / "seed_captions_attributes"
    dense_dir = base / "dense_prompts"
    for dense_path in sorted(dense_dir.glob("*_dense_prompts.json")):
        data = read_json(dense_path)
        concept = data["concept"]
        attr_path = attr_dir / f"{concept}_seed_captions_attributes.json"
        attributes = read_json(attr_path).get("attributes", {})
        for row_idx, row in enumerate(data["coarse_dense_prompts"]):
            for attribute_type, values in sorted(attributes.items()):
                yield make_record(
                    benchmark="dimcim",
                    source_path=dense_path,
                    source_row_id=f"{row_idx}:broad:{attribute_type}",
                    image_source_id=f"{row_idx}:broad",
                    prompt_text=row["coarse_prompt"],
                    axis_id=f"dimcim/{concept}/{attribute_type}",
                    axis_name=f"{concept}:{attribute_type}",
                    condition_type="broad",
                    analysis_layers=cfg["analysis_layers"],
                    complexity_level="atomic",
                    source="official_coco_dimcim_coarse_prompt",
                    candidate_behaviors=[str(value) for value in values],
                    evaluator="DIM VQAScore",
                    metadata={
                        "concept": concept,
                        "attribute_type": attribute_type,
                        "coco_seed_caption": row.get("coco_seed_caption", ""),
                    },
                )
            for dense_idx, item in enumerate(row.get("dense_prompts", [])):
                attribute_type = str(item["attribute_type"])
                attribute = str(item["attribute"])
                yield make_record(
                    benchmark="dimcim",
                    source_path=dense_path,
                    source_row_id=f"{row_idx}:explicit:{dense_idx}",
                    image_source_id=f"{row_idx}:explicit:{dense_idx}",
                    prompt_text=item["dense_prompt"],
                    axis_id=f"dimcim/{concept}/{attribute_type}",
                    axis_name=f"{concept}:{attribute_type}",
                    condition_type="explicit",
                    analysis_layers=cfg["analysis_layers"],
                    complexity_level="atomic",
                    source="official_coco_dimcim_dense_prompt",
                    target_behavior=attribute,
                    candidate_behaviors=[str(value) for value in attributes.get(attribute_type, [])],
                    evaluator="CIM VQAScore",
                    metadata={
                        "concept": concept,
                        "attribute_type": attribute_type,
                        "attribute": attribute,
                        "coco_seed_caption": row.get("coco_seed_caption", ""),
                    },
                )


def load_geneval2(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    path = root / "GenEval2/geneval2_data.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for prompt_idx, line in enumerate(handle):
            row = json.loads(line)
            atom_count = int(row["atom_count"])
            for atom_idx, (qa, skill) in enumerate(zip(row["vqa_list"], row["skills"])):
                question, answer = qa
                yield make_record(
                    benchmark="geneval2",
                    source_path=path,
                    source_row_id=f"{prompt_idx}:atom:{atom_idx}",
                    image_source_id=str(prompt_idx),
                    prompt_text=row["prompt"],
                    axis_id=f"geneval2/{skill}",
                    axis_name=str(skill),
                    condition_type="benchmark_native",
                    analysis_layers=cfg["analysis_layers"],
                    complexity_level=_complexity_from_atom_count(atom_count),
                    source="official_geneval2_atom",
                    target_behavior=str(answer),
                    evaluator="Soft-TIFA VQA",
                    metadata={
                        "prompt_index": prompt_idx,
                        "atom_index": atom_idx,
                        "atom_count": atom_count,
                        "question": question,
                        "answer": answer,
                        "skill": skill,
                    },
                )


def load_t2i_compbench(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    base = root / "T2I-CompBench/examples/dataset"
    axis_map = {
        "color": ("color_binding", "atomic"),
        "shape": ("shape_binding", "atomic"),
        "texture": ("texture_binding", "atomic"),
        "spatial": ("2d_spatial", "relational"),
        "3d_spatial": ("3d_spatial", "relational"),
        "non_spatial": ("non_spatial_relation", "relational"),
        "numeracy": ("numeracy", "relational"),
        "complex": ("complex_composition", "compositional"),
        "new_objects": ("new_object_generalization", "atomic"),
    }
    for file_name in cfg["prompt_files"]:
        path = base / file_name
        stem = path.stem
        axis_name, complexity = axis_map.get(stem, (stem, "compositional"))
        with path.open("r", encoding="utf-8") as handle:
            for idx, prompt in enumerate(line.strip() for line in handle if line.strip()):
                yield make_record(
                    benchmark="t2i_compbench",
                    source_path=path,
                    source_row_id=str(idx),
                    prompt_text=prompt,
                    axis_id=f"t2i_compbench/{axis_name}",
                    axis_name=axis_name,
                    condition_type="compositional_native",
                    analysis_layers=cfg["analysis_layers"],
                    complexity_level=complexity,
                    source="official_t2i_compbench_prompt",
                    evaluator="T2I-CompBench++ native metrics",
                    metadata={"file": file_name, "prompt_index": idx},
                )


def load_worldgenbench(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for domain, file_name in [("humanities", "humanities_en.json"), ("nature", "nature_en.json")]:
        path = root / "WorldGenBench" / file_name
        rows = read_json(path)
        for prompt_idx, row in enumerate(rows):
            for item_idx, item in enumerate(row.get("checklist", [])):
                axis_class = str(row.get("class", domain))
                yield make_record(
                    benchmark="worldgenbench",
                    source_path=path,
                    source_row_id=f"{prompt_idx}:checklist:{item_idx}",
                    image_source_id=str(prompt_idx),
                    prompt_text=row["prompt_en"],
                    axis_id=f"worldgenbench/{domain}/{axis_class}",
                    axis_name=f"{domain}:{axis_class}",
                    condition_type="implicit_knowledge",
                    analysis_layers=cfg["analysis_layers"],
                    complexity_level="implicit",
                    source="official_worldgenbench_checklist",
                    target_behavior="checklist_satisfied",
                    evaluator="Knowledge Checklist VLM judge",
                    metadata={
                        "domain": domain,
                        "class": axis_class,
                        "checklist_index": item_idx,
                        "checklist_item": item,
                    },
                )


def load_t2i_factualbench(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    base = root / "T2I-FactualBench/data/prompts"
    for variant, prefix, condition_type in [
        ("native", base, "knowledge_native"),
        ("text_injection", base / "text_injection", "knowledge_injected"),
    ]:
        for level in ["SKCM", "SKCI", "MKCC"]:
            path = prefix / level / "prompt.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    row = json.loads(line)
                    concept_type = str(row.get("type", "unknown"))
                    yield make_record(
                        benchmark="t2i_factualbench",
                        source_path=path,
                        source_row_id=str(idx),
                        prompt_text=row["sentence"],
                        axis_id=f"t2i_factualbench/{level}/{concept_type}",
                        axis_name=f"{level}:{concept_type}",
                        condition_type=condition_type,
                        analysis_layers=cfg["analysis_layers"],
                        complexity_level=_factual_complexity(level),
                        source=f"official_t2i_factualbench_{variant}",
                        target_behavior=str(row.get("concept", "")),
                        evaluator="T2I-FactualBench multi-round VQA",
                        metadata={
                            "level": level,
                            "concept": row.get("concept", ""),
                            "type": concept_type,
                            "task": row.get("task", ""),
                            "reference_image": row.get("reference_image", ""),
                            "addition": row.get("addition", ""),
                            "knowledge_variant": variant,
                        },
                    )


def load_overt(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    full = root / "OVERT/data/OVERT_full.csv"
    for idx, row in enumerate(read_csv(full)):
        category = row["category"].strip()
        yield make_record(
            benchmark="overt",
            source_path=full,
            source_row_id=str(idx),
            prompt_text=row["image_prompt"],
            axis_id=f"overt/{category}",
            axis_name=category,
            condition_type="benign_overrefusal",
            analysis_layers=cfg["analysis_layers"],
            complexity_level="safety_benign",
            source="official_overt_full_benign",
            target_behavior="non_refusal",
            evaluator="refusal detector + safety VLM",
            metadata={"category": category, "generation_type": row.get("generation_type", "")},
        )
    unsafe = root / "OVERT/data/OVERT_unsafe.csv"
    for idx, row in enumerate(read_csv(unsafe)):
        category = row["category"].strip()
        for condition_type, prompt_key, target in [
            ("unsafe_safety", "unsafe_image_prompt", "refusal_or_safe_response"),
            ("safety_counterfactual_benign", "benign_image_prompt", "non_refusal"),
        ]:
            yield make_record(
                benchmark="overt",
                source_path=unsafe,
                source_row_id=f"{idx}:{condition_type}",
                prompt_text=row[prompt_key],
                axis_id=f"overt/{category}",
                axis_name=category,
                condition_type=condition_type,
                analysis_layers=cfg["analysis_layers"],
                complexity_level="safety_unsafe" if condition_type == "unsafe_safety" else "safety_benign",
                source="official_overt_unsafe_pair",
                target_behavior=target,
                evaluator="refusal detector + safety VLM",
                metadata={"category": category, "generation_type": row.get("generation_type", "")},
            )


def load_t2i_riskyprompt(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    path = root / "T2I-RiskyPrompt/T2I-RiskyPrompt.json"
    data = read_json(path)
    for row_id, row in data.items():
        labels = row.get("label", {})
        for label_key, reasons in labels.items():
            primary, _, subcategory = label_key.partition("--")
            yield make_record(
                benchmark="t2i_riskyprompt",
                source_path=path,
                source_row_id=str(row_id),
                prompt_text=row["prompt"],
                axis_id=f"t2i_riskyprompt/{primary}/{subcategory}",
                axis_name=f"{primary}:{subcategory}",
                condition_type="risky_safety",
                analysis_layers=cfg["analysis_layers"],
                complexity_level="safety_unsafe",
                source="official_t2i_riskyprompt",
                target_behavior="risk_not_realized",
                evaluator="reason-driven risky image detector",
                metadata={
                    "primary_category": primary,
                    "subcategory": subcategory,
                    "risk_reasons": reasons,
                },
            )


def load_t2isafety(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    fair = root / "T2ISafety_eval/fairness.jsonl"
    with fair.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            yield make_record(
                benchmark="t2isafety",
                source_path=fair,
                source_row_id=str(idx),
                prompt_text=row["prompt"],
                axis_id="t2isafety/fairness/demographic_distribution",
                axis_name="fairness_demographic_distribution",
                condition_type="social_bias_default",
                analysis_layers=cfg["analysis_layers"],
                complexity_level="bias",
                source="official_t2isafety_fairness_eval",
                evaluator="ImageGuard + demographic classifier",
                metadata={"domain": "fairness"},
            )
    safety = root / "T2ISafety_eval/safety.jsonl"
    with safety.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            category = str(row.get("category", "unknown"))
            yield make_record(
                benchmark="t2isafety",
                source_path=safety,
                source_row_id=str(idx),
                prompt_text=row["prompt"],
                axis_id=f"t2isafety/safety/{category}",
                axis_name=category,
                condition_type="safety_native",
                analysis_layers=cfg["analysis_layers"],
                complexity_level="safety_unsafe",
                source="official_t2isafety_safety_eval",
                target_behavior="safe_or_refused",
                evaluator="ImageGuard",
                metadata={"domain": "safety", "category": category},
            )


def load_membench(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    path = root / "MemBench/SD1_final.csv"
    for idx, row in enumerate(read_csv(path)):
        yield make_record(
            benchmark="membench",
            source_path=path,
            source_row_id=str(idx),
            prompt_text=row["prompt"],
            axis_id="membench/trigger_replication",
            axis_name="trigger_replication",
            condition_type="memorization_trigger",
            analysis_layers=cfg["analysis_layers"],
            complexity_level="memorization",
            source="official_membench_trigger_prompts",
            target_behavior="not_memorized_reference",
            evaluator="SSCD copy-detection similarity",
            metadata={"urls": row.get("urls", ""), "row_id": row.get("", str(idx))},
        )
    prompt_dir = root / "MemBench/MemAttn/prompt"
    for prompt_file in sorted(prompt_dir.glob("*.txt")):
        with prompt_file.open("r", encoding="utf-8") as handle:
            for idx, prompt in enumerate(line.strip() for line in handle if line.strip()):
                condition = "memorization_trigger" if "mem" in prompt_file.stem and "nomem" not in prompt_file.stem else "memorization_control"
                yield make_record(
                    benchmark="membench",
                    source_path=prompt_file,
                    source_row_id=str(idx),
                    prompt_text=prompt,
                    axis_id=f"membench/{prompt_file.stem}",
                    axis_name=prompt_file.stem,
                    condition_type=condition,
                    analysis_layers=cfg["analysis_layers"],
                    complexity_level="memorization",
                    source="official_memattn_prompt",
                    target_behavior="not_memorized_reference",
                    evaluator="MemAttn + SSCD",
                    metadata={"prompt_file": prompt_file.name},
                )


def load_hub(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    base = root / "HUB/prompts"
    task_map = {
        "target_image": ("unlearning_target", "target_proportion"),
        "pinpoint_ness": ("unlearning_retain", "pinpoint_ness"),
        "attack_robustness": ("unlearning_robustness", "attack_robustness"),
        "multilingual_robustness": ("unlearning_robustness", "multilingual_robustness"),
    }
    for task, (condition_type, axis_name) in task_map.items():
        for path in sorted((base / task).glob("*.csv")):
            if path.name.startswith("."):
                continue
            target = path.stem.replace("_", " ")
            for idx, prompt in enumerate(_iter_prompt_csv(path)):
                yield make_record(
                    benchmark="hub",
                    source_path=path,
                    source_row_id=str(idx),
                    prompt_text=prompt,
                    axis_id=f"hub/{axis_name}/{target}",
                    axis_name=f"{axis_name}:{target}",
                    condition_type=condition_type,
                    analysis_layers=cfg["analysis_layers"],
                    complexity_level="unlearning",
                    source="official_hub_prompt",
                    target_behavior=target,
                    evaluator="HUB task-specific detector",
                    requires_valid_lineage=True,
                    metadata={"task": task, "target": target},
                )
    for path in sorted((base / "selective_alignment").glob("*.json")):
        target = path.stem
        data = read_json(path)
        prompts = _flatten_json_prompts(data)
        for idx, prompt in enumerate(prompts):
            yield make_record(
                benchmark="hub",
                source_path=path,
                source_row_id=str(idx),
                prompt_text=prompt,
                axis_id=f"hub/selective_alignment/{target}",
                axis_name=f"selective_alignment:{target}",
                condition_type="unlearning_retain",
                analysis_layers=cfg["analysis_layers"],
                complexity_level="unlearning",
                source="official_hub_selective_alignment_prompt",
                target_behavior=target,
                evaluator="HUB selective-alignment detector",
                requires_valid_lineage=True,
                metadata={"task": "selective_alignment", "target": target},
            )


def load_emma(root: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    base = root / "EMMA/prompts"
    type_map = {
        "1_name": ("unlearning_explicit", "erasing_name"),
        "2_prefix": ("unlearning_explicit", "erasing_prefix"),
        "3_variant": ("unlearning_implicit", "erasing_variant"),
        "4_short": ("unlearning_implicit", "erasing_short"),
        "5_long": ("unlearning_implicit", "erasing_long"),
        "6_random": ("unlearning_retain", "retain_random"),
        "7_hard": ("unlearning_retain", "retain_hard"),
    }
    for domain_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        domain = domain_dir.name
        for path in sorted(domain_dir.glob("*.json")):
            metric_key = path.stem
            condition_type, metric = type_map.get(metric_key, ("unlearning_implicit", metric_key))
            data = read_json(path)
            for concept, prompt in _iter_emma_prompts(data):
                yield make_record(
                    benchmark="emma",
                    source_path=path,
                    source_row_id=f"{concept}:{stable_hash(prompt, length=8)}",
                    prompt_text=prompt,
                    axis_id=f"emma/{domain}/{metric}/{concept}",
                    axis_name=f"{domain}:{metric}:{concept}",
                    condition_type=condition_type,
                    analysis_layers=cfg["analysis_layers"],
                    complexity_level="unlearning",
                    source="official_emma_prompt",
                    target_behavior=concept,
                    evaluator="EMMA domain-specific classifier",
                    requires_valid_lineage=True,
                    metadata={"domain": domain, "metric": metric, "concept": concept},
                )


def summarize_prompt_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark: dict[str, int] = {}
    by_condition: dict[str, int] = {}
    unique_image_prompts: set[str] = set()
    gated = 0
    for row in rows:
        by_benchmark[row["benchmark"]] = by_benchmark.get(row["benchmark"], 0) + 1
        by_condition[row["condition_type"]] = by_condition.get(row["condition_type"], 0) + 1
        unique_image_prompts.add(row["image_prompt_uid"])
        gated += int(bool(row.get("requires_valid_lineage")))
    return {
        "records_by_benchmark": dict(sorted(by_benchmark.items())),
        "records_by_condition": dict(sorted(by_condition.items())),
        "unique_image_prompts": len(unique_image_prompts),
        "requires_valid_lineage_records": gated,
    }


def _iter_prompt_csv(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames and any("prompt" in name.lower() for name in reader.fieldnames if name):
            prompt_key = next(name for name in reader.fieldnames if "prompt" in name.lower())
            for row in reader:
                text = str(row.get(prompt_key, "")).strip()
                if text:
                    yield text
            return
        handle.seek(0)
        for row in csv.reader(handle, dialect=dialect):
            if not row:
                continue
            text = row[0].strip()
            if text and text.lower() != "prompt":
                yield text


def _flatten_json_prompts(data: Any) -> list[str]:
    prompts: list[str] = []
    if isinstance(data, str):
        return [data]
    if isinstance(data, list):
        for item in data:
            prompts.extend(_flatten_json_prompts(item))
    elif isinstance(data, dict):
        for value in data.values():
            prompts.extend(_flatten_json_prompts(value))
    return [prompt for prompt in prompts if prompt.strip()]


def _iter_emma_prompts(data: Any) -> Iterable[tuple[str, str]]:
    if not isinstance(data, list):
        return
    for item in data:
        if not isinstance(item, dict):
            continue
        for concept, prompts in item.items():
            if isinstance(prompts, str):
                yield str(concept), prompts
            elif isinstance(prompts, list):
                for prompt in prompts:
                    if str(prompt).strip():
                        yield str(concept), str(prompt)


def _complexity_from_atom_count(atom_count: int) -> str:
    if atom_count <= 3:
        return "atomic"
    if atom_count <= 6:
        return "relational"
    return "compositional"


def _factual_complexity(level: str) -> str:
    return {
        "SKCM": "atomic",
        "SKCI": "relational",
        "MKCC": "compositional",
    }.get(level, "implicit")
