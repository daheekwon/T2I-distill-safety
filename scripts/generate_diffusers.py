#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from a manifest with Diffusers backends.")
    parser.add_argument("--manifest", default="data/manifests/generation_manifest.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based inclusive job index within the manifest.")
    parser.add_argument("--end-index", type=int, default=None, help="Zero-based exclusive job index within the manifest.")
    parser.add_argument("--device", default="", help="Override every job device, e.g. cuda:2.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-failures", type=int, default=0, help="With --continue-on-error, stop after this many failures. 0 means no cap.")
    parser.add_argument("--failure-log", default="results/generation/failures.jsonl")
    parser.add_argument("--success-log", default="")
    parser.add_argument("--status-every", type=int, default=25)
    parser.add_argument("--max-loaded-pipelines", type=int, default=1, help="Limit simultaneously cached pipelines to avoid VRAM accumulation.")
    args = parser.parse_args()

    if args.dry_run:
        preview = _preview_jobs(
            Path(args.manifest),
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
            device_override=args.device,
        )
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return

    try:
        import torch
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install torch, pillow, diffusers, transformers, accelerate, safetensors, and huggingface_hub first.") from exc

    if args.max_loaded_pipelines <= 0:
        raise ValueError("--max-loaded-pipelines must be positive")
    cache: dict[tuple[str, str], object] = {}
    cache_order: list[tuple[str, str]] = []
    completed = 0
    skipped = 0
    failures = 0
    jobs_seen = 0
    for local_idx, job in enumerate(
        _iter_jobs(Path(args.manifest), limit=args.limit, start_index=args.start_index, end_index=args.end_index),
        start=1,
    ):
        jobs_seen += 1
        if args.device:
            job["device"] = args.device
        output_path = Path(str(job["output_path"]))
        if args.skip_existing and _looks_complete_image(output_path):
            skipped += 1
            continue
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cache_key = (str(job["model_id"]), str(job["device"]))
            pipe = cache.get(cache_key)
            if pipe is None:
                _evict_if_needed(cache, cache_order, args.max_loaded_pipelines, torch)
                pipe = _load_pipeline(job, torch)
                cache[cache_key] = pipe
                cache_order.append(cache_key)
            generator = torch.Generator(device=str(job["device"])).manual_seed(int(job["seed"]))
            with torch.inference_mode():
                image = pipe(
                    prompt=str(job["prompt_text"]),
                    num_inference_steps=int(job["num_inference_steps"]),
                    guidance_scale=float(job["guidance_scale"]),
                    generator=generator,
                ).images[0]
            image.save(output_path)
            completed += 1
            if args.success_log:
                _append_jsonl(Path(args.success_log), _event(job, status="done", output_path=str(output_path)))
            if args.status_every > 0 and completed % args.status_every == 0:
                print(json.dumps({"completed": completed, "skipped": skipped, "failures": failures, "last_job_id": job["job_id"]}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failures += 1
            failure = _event(job, status="failed", error=repr(exc), traceback=traceback.format_exc())
            _append_jsonl(Path(args.failure_log), failure)
            print(json.dumps(_redact_prompt(failure), ensure_ascii=False), file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise
            if args.max_failures and failures >= args.max_failures:
                raise SystemExit(f"stopping after {failures} generation failures") from exc
    print(json.dumps({"completed": completed, "skipped": skipped, "failures": failures, "jobs_seen": jobs_seen}, ensure_ascii=False, indent=2, sort_keys=True))


def _preview_jobs(
    path: Path,
    *,
    limit: int | None,
    start_index: int,
    end_index: int | None,
    device_override: str,
) -> dict[str, Any]:
    first = None
    last = None
    count = 0
    by_model: dict[str, int] = {}
    model_switches = 0
    previous_model = None
    for job in _iter_jobs(path, limit=limit, start_index=start_index, end_index=end_index):
        if device_override:
            job["device"] = device_override
        if first is None:
            first = dict(job)
        last = dict(job)
        model_id = str(job.get("model_id", ""))
        by_model[model_id] = by_model.get(model_id, 0) + 1
        if previous_model is not None and model_id != previous_model:
            model_switches += 1
        previous_model = model_id
        count += 1
    return {
        "jobs": count,
        "jobs_by_model": dict(sorted(by_model.items())),
        "model_switches": model_switches,
        "first_job": _redact_prompt(first) if first else None,
        "last_job": _redact_prompt(last) if last else None,
    }


def _read_jobs(path: Path, limit: int | None, start_index: int = 0, end_index: int | None = None) -> list[dict[str, Any]]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index < start_index:
        raise ValueError("end_index must be >= start_index")
    jobs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx < start_index:
                continue
            if end_index is not None and line_idx >= end_index:
                break
            if not line.strip():
                continue
            jobs.append(json.loads(line))
            if limit is not None and len(jobs) >= limit:
                break
    return jobs


def _iter_jobs(path: Path, limit: int | None, start_index: int = 0, end_index: int | None = None) -> Iterable[dict[str, Any]]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index < start_index:
        raise ValueError("end_index must be >= start_index")
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx < start_index:
                continue
            if end_index is not None and line_idx >= end_index:
                break
            if not line.strip():
                continue
            yield json.loads(line)
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def _evict_if_needed(cache: dict[tuple[str, str], object], cache_order: list[tuple[str, str]], max_loaded: int, torch: Any) -> None:
    while len(cache_order) >= max_loaded:
        old_key = cache_order.pop(0)
        cache.pop(old_key, None)
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_pipeline(job: dict[str, Any], torch: Any) -> object:
    runner = str(job["runner"])
    device = str(job["device"])
    dtype = torch.bfloat16 if "flux2" in runner or "sd3" in runner else torch.float16
    if runner == "diffusers_sdxl_lightning_unet":
        return _load_sdxl_lightning(job, torch, dtype, device)
    if runner == "diffusers_sdxl_base":
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(str(job["hf_model"]), torch_dtype=dtype, variant="fp16")
        return pipe.to(device)
    if runner == "diffusers_sd3":
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(str(job["hf_model"]), torch_dtype=dtype)
        return pipe.to(device)
    if runner == "diffusers_flux2_klein":
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError:
            from diffusers import DiffusionPipeline as Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(str(job["hf_model"]), torch_dtype=dtype)
        return pipe.to(device)
    raise ValueError(f"unsupported runner: {runner}")


def _load_sdxl_lightning(job: dict[str, Any], torch: Any, dtype: Any, device: str) -> object:
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    base = str(job["base_model"])
    repo = str(job["hf_model"])
    checkpoint = str(job["checkpoint"])
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, dtype)
    unet.load_state_dict(load_file(hf_hub_download(repo, checkpoint), device=device))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=dtype, variant="fp16").to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe


def _redact_prompt(job: dict[str, Any]) -> dict[str, Any]:
    row = dict(job)
    prompt = str(row.get("prompt_text", ""))
    row["prompt_text"] = "<redacted>"
    row["prompt_chars"] = len(prompt)
    if "traceback" in row:
        row["traceback"] = "<redacted traceback; see failure log>"
    return row


def _event(job: dict[str, Any], **extra: Any) -> dict[str, Any]:
    row = _redact_prompt(job)
    row.update(extra)
    return row


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _looks_complete_image(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


if __name__ == "__main__":
    main()
