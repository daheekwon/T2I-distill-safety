# T2I Distillation Behavioral Inheritance

This repository implements a reproducible experiment scaffold for studying what is inherited when diffusion models are distilled.

The default study compares:

- `SDXL Base -> SDXL-Lightning 4-step`
- `SD3.5 Large -> SD3.5 Large Turbo`
- `FLUX.2 [klein] 4B Base -> FLUX.2 [klein] 4B`

All generation jobs default to `cuda:2`.

## Build The Study Artifacts

```bash
python scripts/record_benchmark_versions.py
python scripts/check_model_backends.py --online
python scripts/build_prompt_bank.py
python scripts/summarize_prompt_bank.py
python scripts/build_generation_manifest.py --model-scope primary
python scripts/build_label_template.py
python scripts/build_evaluator_plan.py
python scripts/plan_evaluator_batches.py --dry-run
python scripts/audit_evaluator_plan.py
python scripts/audit_design_matrix.py
python scripts/audit_statistical_power.py
python scripts/audit_label_quality.py
```

The primary manifest is model-major by default, so generation runs one model for long contiguous blocks instead of reloading models prompt by prompt. The generation runner streams JSONL and keeps only one pipeline loaded by default.

Split the 1.2M-job manifest into resumable, single-model shards:

```bash
python scripts/split_manifest.py --jobs-per-shard 10000 --respect-model-boundaries
```

Run a safe generation preview:

```bash
python scripts/generate_diffusers.py --manifest data/manifests/shards_primary/shard_00000.jsonl --dry-run --limit 1
```

Run a shard on GPU 2 with resume/error logs:

```bash
python scripts/generate_diffusers.py \
  --manifest data/manifests/shards_primary/shard_00000.jsonl \
  --device cuda:2 \
  --skip-existing \
  --continue-on-error \
  --max-loaded-pipelines 1 \
  --success-log results/generation/shard_00000_success.jsonl \
  --failure-log results/generation/shard_00000_failures.jsonl
```

For unattended primary generation, use the queue runner. It selects pending/partial shards, waits for the requested GPU or GPUs to fall below the memory/utilization thresholds, keeps a lock file while active, and writes both the selected plan and final shard result to `results/generation/run_plan.json`.

```bash
python scripts/run_generation_queue.py --dry-run --max-shards 2 --run-plan-output results/generation/run_plan_dry_run.json
python scripts/run_generation_queue.py --max-shards 1 --gpu 2 --max-memory-mib 8000 --max-utilization 15
```

To use GPUs 0 and 1 opportunistically alongside other workloads, use one parent-managed multi-GPU queue so shards are assigned once and do not race each other. Conservative thresholds keep the runner waiting while those GPUs are actively saturated.

```bash
python scripts/run_generation_queue.py --gpus 0,1 --dry-run --max-shards 4 --run-plan-output results/generation/run_plan_gpus_0_1_dry_run.json
python scripts/run_generation_queue.py --gpus 0,1 --max-shards 2 --max-memory-mib 22000 --max-utilization 40 --run-plan-output results/generation/run_plan_gpus_0_1.json
```

Use `--max-shards 0` only when you intentionally want to run every matching pending/partial shard.

Audit readiness and generation progress at any point:

```bash
python scripts/audit_design_matrix.py
python scripts/audit_statistical_power.py
python scripts/audit_label_quality.py
python scripts/audit_evaluator_plan.py
python scripts/audit_experiment_state.py
python scripts/audit_rq_readiness.py
python scripts/summarize_generation_progress.py
```


## Preflight Package

The preflight package samples one image prompt per benchmark condition from the full prompt bank. It is only an end-to-end systems check; it does not replace the full experiment.

```bash
python scripts/build_preflight_package.py --prompts-per-condition 1 --seeds 0
python scripts/generate_diffusers.py --manifest data/preflight/generation_manifest.jsonl --dry-run --limit 2
```

Wait for GPU 2 before launching preflight generation:

```bash
python scripts/wait_for_gpu_and_generate.py \
  --gpu 2 \
  --max-memory-mib 8000 \
  --max-utilization 15 \
  --manifest data/preflight/generation_manifest.jsonl \
  --success-log results/generation/preflight_success.jsonl \
  --failure-log results/generation/preflight_failures.jsonl
```

## Evaluation Loop

Plan full evaluator batches without writing large JSONL files:

```bash
python scripts/plan_evaluator_batches.py --dry-run --batch-size 5000
```

Export evaluator input batches after images exist. Use `--only-existing-images` for real labeling runs.

```bash
python scripts/plan_evaluator_batches.py \
  --labels data/labels/label_template.csv \
  --output-dir data/evaluation/batches_existing \
  --batch-index data/evaluation/batch_index_existing.csv \
  --batch-size 5000 \
  --only-existing-images
```

For a narrow one-off batch, use:

```bash
python scripts/export_evaluator_batch.py \
  --labels data/labels/label_template.csv \
  --output data/evaluation/batches/grade_attribute_batch.jsonl \
  --benchmarks grade \
  --only-existing-images
```

Validate and merge evaluator updates:

```bash
python scripts/validate_label_updates.py results/evaluator_updates/*.jsonl --fail-on-invalid
python scripts/merge_label_updates.py results/evaluator_updates/*.jsonl --output data/labels/filled_labels.csv --strict
python scripts/audit_label_quality.py results/evaluator_updates/*.jsonl --labels data/labels/filled_labels.csv
python scripts/audit_experiment_state.py --labels data/labels/filled_labels.csv
python scripts/audit_rq_readiness.py --labels data/labels/filled_labels.csv
```

After image generation and VLM/manual labeling, run the integrated post-label pipeline. It writes design, power, readiness, analysis, and claim-level report artifacts into one result directory:

```bash
python scripts/run_post_label_pipeline.py \
  --labels data/labels/filled_labels.csv \
  --output-dir results/post_label_pipeline \
  --strict-design \
  --strict-power \
  --strict-label-quality
```

For analysis only, use `python scripts/analyze_labels.py --labels data/labels/filled_labels.csv --model-scope primary`. Core analysis outputs include distribution shift, paired outcome tests, sharpening fits/transfers, residual transport with uncertainty, preference/capability dissociation, survival features, frequency-controlled survival regression, and quality balance tables. To rebuild only the claim-level interpretation report from an existing result directory, use `python scripts/build_claim_report.py --result-dir results/post_label_pipeline`.

## Public Snapshot Scope

This public repository contains the reusable experiment code, configuration, and command-line pipeline. Generated data, benchmark downloads, results, private notes, application drafts, and local validation artifacts are intentionally excluded.
