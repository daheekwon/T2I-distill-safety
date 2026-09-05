#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config, unique_models
from t2i_distill.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Check configured model backends without downloading model weights.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--model-scope", choices=["primary", "secondary", "all"], default="primary")
    parser.add_argument("--online", action="store_true", help="Use Hugging Face API metadata checks. Does not download weights.")
    parser.add_argument("--output", default="results/audit/model_backend_check.json")
    args = parser.parse_args()

    summary = check_model_backends(load_config(Path(args.config)), model_scope=args.model_scope, online=args.online)
    write_json(Path(args.output), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["valid"]:
        raise SystemExit(1)


def check_model_backends(config: dict[str, Any], *, model_scope: str, online: bool = False) -> dict[str, Any]:
    try:
        import diffusers
        diffusers_error = ""
    except Exception as exc:
        diffusers = None
        diffusers_error = repr(exc)
    try:
        import torch
        torch_error = ""
    except Exception as exc:
        torch = None
        torch_error = repr(exc)

    models = unique_models(config, model_scope=model_scope)
    rows = []
    valid = diffusers is not None and torch is not None
    for model in models:
        row = _check_model(model, diffusers=diffusers, online=online)
        rows.append(row)
        valid = valid and row["valid"]
    return {
        "model_scope": model_scope,
        "default_device": config.get("default_device"),
        "diffusers_available": diffusers is not None,
        "diffusers_version": getattr(diffusers, "__version__", "") if diffusers is not None else "",
        "diffusers_error": diffusers_error,
        "torch_available": torch is not None,
        "torch_version": getattr(torch, "__version__", "") if torch is not None else "",
        "torch_error": torch_error,
        "online_metadata_check": online,
        "models": rows,
        "valid": valid,
    }


def _check_model(model: dict[str, Any], *, diffusers: Any, online: bool) -> dict[str, Any]:
    runner = str(model.get("runner", ""))
    checks = []
    if runner == "diffusers_sdxl_base":
        checks.append(_has_attr(diffusers, "StableDiffusionXLPipeline"))
    elif runner == "diffusers_sdxl_lightning_unet":
        checks.extend([_has_attr(diffusers, "StableDiffusionXLPipeline"), _has_attr(diffusers, "UNet2DConditionModel"), _has_attr(diffusers, "EulerDiscreteScheduler")])
        checks.append({"name": "checkpoint_configured", "pass": bool(model.get("checkpoint")), "detail": str(model.get("checkpoint", ""))})
    elif runner == "diffusers_sd3":
        checks.append(_has_attr(diffusers, "DiffusionPipeline"))
    elif runner == "diffusers_flux2_klein":
        checks.append(_has_attr(diffusers, "Flux2KleinPipeline"))
    else:
        checks.append({"name": "known_runner", "pass": False, "detail": runner})
    if online:
        checks.extend(_online_model_checks(model))
    return {
        "model_id": model.get("model_id", ""),
        "display_name": model.get("display_name", ""),
        "runner": runner,
        "hf_model": model.get("hf_model", ""),
        "device": model.get("device", ""),
        "num_inference_steps": model.get("num_inference_steps"),
        "guidance_scale": model.get("guidance_scale"),
        "checks": checks,
        "valid": all(item["pass"] for item in checks),
    }


def _has_attr(module: Any, name: str) -> dict[str, Any]:
    return {"name": f"diffusers_has_{name}", "pass": module is not None and hasattr(module, name), "detail": ""}


def _online_model_checks(model: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return [{"name": "huggingface_hub_available", "pass": False, "detail": repr(exc)}]
    api = HfApi()
    repo_id = str(model.get("hf_model", ""))
    checks = []
    try:
        info = api.model_info(repo_id, files_metadata=False)
        siblings = {s.rfilename for s in getattr(info, "siblings", []) or []}
        checks.append({"name": "hf_model_exists", "pass": True, "detail": repo_id})
        checkpoint = str(model.get("checkpoint", ""))
        if checkpoint:
            checks.append({"name": "hf_checkpoint_exists", "pass": checkpoint in siblings, "detail": checkpoint})
    except Exception as exc:
        checks.append({"name": "hf_model_exists", "pass": False, "detail": f"{repo_id}: {exc!r}"})
    return checks


if __name__ == "__main__":
    main()
