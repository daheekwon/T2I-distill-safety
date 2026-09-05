from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import read_json

DEFAULT_CONFIG = Path("config/experiment.json")


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(Path(path))
    if "default_device" not in config:
        raise ValueError("config must define default_device")
    if "model_pairs" not in config:
        raise ValueError("config must define model_pairs")
    if "benchmark_plan" not in config:
        raise ValueError("config must define benchmark_plan")
    return config


def selected_pairs(config: dict[str, Any], model_scope: str = "primary") -> list[dict[str, Any]]:
    if model_scope not in {"primary", "secondary", "all"}:
        raise ValueError("model_scope must be one of: primary, secondary, all")
    pairs = config["model_pairs"]
    if model_scope == "all":
        return list(pairs)
    return [pair for pair in pairs if pair.get("priority") == model_scope]


def unique_models(config: dict[str, Any], model_scope: str = "primary") -> list[dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for pair in selected_pairs(config, model_scope=model_scope):
        for role in ("teacher", "student"):
            model = dict(pair[role])
            model["family"] = pair["family"]
            model["role_hint"] = role
            model["priority"] = pair["priority"]
            models.setdefault(model["model_id"], model)
    return list(models.values())


def iter_pair_models(config: dict[str, Any], model_scope: str = "primary") -> Iterable[tuple[dict[str, Any], str, dict[str, Any]]]:
    for pair in selected_pairs(config, model_scope=model_scope):
        yield pair, "teacher", pair["teacher"]
        yield pair, "student", pair["student"]


def benchmark_config(config: dict[str, Any], benchmark: str) -> dict[str, Any]:
    try:
        return config["benchmark_plan"][benchmark]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark: {benchmark}") from exc


def seed_values(config: dict[str, Any], seed_key: str) -> list[int]:
    try:
        return list(config["seed_sets"][seed_key])
    except KeyError as exc:
        raise KeyError(f"unknown seed set: {seed_key}") from exc


def seeds_for_condition(config: dict[str, Any], benchmark: str, condition_type: str) -> list[int]:
    bench = benchmark_config(config, benchmark)
    seed_key = bench.get("seed_keys", {}).get(condition_type)
    if seed_key is None:
        seed_key = "matched_4"
    return seed_values(config, seed_key)
