# Diffusion Distillation Behavioral Inheritance Protocol

## 핵심 질문

이 연구는 distillation을 단순한 가속/압축이 아니라 teacher behavior를 선택적으로 보존, 강화, 약화, 재배치하는 operator로 본다. 중심 질문은 다음과 같다.

> Distillation은 teacher의 behavior를 균등하게 보존하는가, 아니면 공통된 선택 규칙에 따라 어떤 behavior는 강화하고 다른 behavior는 약화시키는가?

## 모델 우선순위

Primary 비교는 세 개다.

| comparison_id | teacher | student | 목적 |
| --- | --- | --- | --- |
| `sdxl_base__sdxl_lightning_4step` | SDXL Base | SDXL-Lightning 4-step | SDXL 대표 distilled pair |
| `sd35_large__sd35_large_turbo` | SD3.5 Large | SD3.5 Large Turbo | architecture/family external validation |
| `flux2_klein_base_4b__flux2_klein_4b` | FLUX.2 [klein] 4B Base | FLUX.2 [klein] 4B Distilled | 4B-scale third family validation |

SDXL-Lightning 2-step과 8-step은 secondary dose-response 분석용이다. 기본 manifest는 primary만 포함한다. 모든 generation job의 기본 device는 `cuda:2`다.

## Benchmark Scope

GenEval2는 2026-09-04 사용자 지시에 따라 기본 실험에서 제외했다. relation/composition/knowledge 역할은 T2I-CompBench++, WorldGenBench, T2I-FactualBench로 충분히 덮는다.

| benchmark | 역할 | 포함 방식 |
| --- | --- | --- |
| GRADE | generic sharpening law fit | 공식 2,430 prompt rows 전체 |
| DIMCIM | default vs explicit accessibility | 930 coarse prompts와 14,641 dense prompts 전체 |
| T2I-CompBench++ | composition/remapping stress test | full txt prompt files 전체 |
| WorldGenBench | implicit world knowledge | humanities/nature 1,072 prompts의 checklist items |
| T2I-FactualBench | factual knowledge hierarchy | SKCM, SKCI, MKCC native 및 text-injection prompts |
| OVERT | over-refusal and safety-utility tradeoff | benign full, unsafe, paired benign counterfactual |
| T2I-RiskyPrompt | risky prompt safety inheritance | 6 primary / 14 fine-grained categories |
| T2ISafety | social bias and safety category | fairness 236 prompts, safety 2,450 prompts |
| MemBench | memorization retention | SD1 trigger CSV와 MemAttn prompt files |
| HUB | unlearning retention/collateral loss | valid lineage가 있을 때만 gated |
| EMMA | explicit/implicit concept-erasure robustness | valid lineage가 있을 때만 gated |

## RQ별 산출물

1. RQ1: Generic sharpening law
   - GRADE와 DIMCIM broad condition에서 `alpha`를 fit한다.
   - 같은 `alpha`를 다른 benchmark에 refit 없이 적용해 transfer error를 측정한다.
   - 산출물: `sharpening_fits.csv`, `sharpening_transfer.csv`.

2. RQ2: Structured remapping
   - 같은 prompt/seed/eval cell의 teacher label과 student label을 매칭한다.
   - `Mij = P(YS=j | YT=i,c)`와 `Rij = Mij - pS(j|c)`를 계산하고, residual별 근사 CI, z/p-value, Bonferroni/BH-FDR 보정값을 기록한다.
   - 산출물: `residual_transport.csv`, `paired_outcome_tests.csv`.

3. RQ3: What is lost?
   - broad prompt에서 감소한 behavior가 explicit or diagnostic condition에서 회복되는지 본다.
   - broad drop은 크고 explicit drop이 작으면 preference shift다.
   - broad drop과 explicit drop이 모두 크면 capability/information loss다.
   - 산출물: `preference_capability.csv`.

4. RQ4: Survival principle
   - teacher frequency, complexity level, benchmark domain이 retention ratio를 얼마나 설명하는지 본다.
   - 핵심 검정은 frequency를 통제한 뒤에도 atomic > relational/compositional > implicit hierarchy가 남는지다.
   - 산출물: `survival_features.csv`, `survival_regression.csv`, `claim_evidence_matrix.json`.

## 실행 순서

```bash
python scripts/record_benchmark_versions.py
python scripts/check_model_backends.py --online
python scripts/build_prompt_bank.py
python scripts/summarize_prompt_bank.py
python scripts/build_generation_manifest.py --model-scope primary --job-order model_major
python scripts/build_label_template.py
python scripts/build_evaluator_plan.py
python scripts/plan_evaluator_batches.py --dry-run --batch-size 5000
python scripts/audit_evaluator_plan.py
python scripts/audit_design_matrix.py
python scripts/audit_statistical_power.py
python scripts/audit_label_quality.py
python scripts/generate_diffusers.py --manifest data/manifests/generation_manifest.jsonl --dry-run --limit 1
```

라벨링/VLM 평가가 끝난 뒤:

```bash
python scripts/analyze_labels.py --labels data/labels/filled_labels.csv --model-scope primary
```

HUB/EMMA는 valid unlearning lineage가 있을 때만 다음처럼 포함한다.

```bash
python scripts/build_prompt_bank.py --include-gated
python scripts/build_generation_manifest.py --include-gated --model-scope primary
```


## 실행/Audit 계층

대규모 실험은 전체 manifest를 한 번에 돌리지 않고 shard 단위로 실행한다. 현재 primary manifest는 1,226,094 generation jobs이며 10,000 jobs 단위, model boundary 보존 기준으로 126개 shard를 만든다. Manifest는 기본적으로 model-major order다. 즉, 같은 모델의 job을 길게 이어 실행해 모델 로딩 횟수와 VRAM churn을 줄인다.

```bash
python scripts/split_manifest.py --manifest data/manifests/generation_manifest.jsonl --output-dir data/manifests/shards_primary --jobs-per-shard 10000 --respect-model-boundaries
```

각 shard는 다음처럼 재개 가능하게 실행한다. runner는 JSONL을 streaming으로 읽고 기본적으로 pipeline cache를 1개로 제한한다. `--skip-existing`는 이미 생성된 이미지를 건너뛰고, `--continue-on-error`는 일부 model/prompt 실패를 failure log에 남긴 뒤 다음 job으로 진행한다.

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

장시간 primary generation은 queue runner로 운용한다. 이 runner는 pending/partial shard만 고르고, 지정한 GPU 또는 GPU들이 `--max-memory-mib`와 `--max-utilization` threshold 아래로 내려갈 때까지 기다린 뒤 shard를 실행한다. 실행 중에는 lock file을 두고, `--run-plan-output`에는 선택된 shard 계획과 최종 `completed_shards`/`failed_shards`를 모두 기록한다.

```bash
python scripts/run_generation_queue.py --dry-run --max-shards 2 --run-plan-output results/generation/run_plan_dry_run.json
python scripts/run_generation_queue.py --max-shards 1 --gpu 2 --max-memory-mib 8000 --max-utilization 15
```

GPU 0/1을 다른 작업과 함께 보조 worker로 쓸 때는 별도 queue를 두 개 띄우지 않고 하나의 parent-managed multi-GPU queue를 사용한다. 이렇게 해야 shard 중복 실행을 피할 수 있다. 현재처럼 0/1이 이미 VRAM을 쓰고 compute util이 높은 상태에서는 conservative threshold로 두어, GPU가 실제로 한가해질 때만 진입하게 한다.

```bash
python scripts/run_generation_queue.py --gpus 0,1 --dry-run --max-shards 4 --run-plan-output results/generation/run_plan_gpus_0_1_dry_run.json
python scripts/run_generation_queue.py --gpus 0,1 --max-shards 2 --max-memory-mib 22000 --max-utilization 40 --run-plan-output results/generation/run_plan_gpus_0_1.json
```

`--max-shards 0`은 조건에 맞는 모든 pending/partial shard를 실행한다는 뜻이므로, 전체 generation을 의도할 때만 사용한다.

실험 상태와 shard 진행률은 항상 다음 gate로 확인한다.

```bash
python scripts/audit_design_matrix.py
python scripts/audit_statistical_power.py
python scripts/audit_label_quality.py
python scripts/audit_evaluator_plan.py
python scripts/audit_experiment_state.py
python scripts/audit_rq_readiness.py
python scripts/summarize_generation_progress.py
```

실제 image 파일 존재 여부까지 확인하려면 다음처럼 sample 또는 full image audit를 별도로 실행한다.

```bash
python scripts/audit_experiment_state.py --check-images --image-limit 1000 --output-json results/audit/experiment_state_image_sample.json --output-md results/audit/experiment_state_image_sample.md
```

현재 template 상태에서는 `geneval2_excluded`, `manifest_device`, `manifest_unique_ids`, `primary_model_balance`, `label_schema`가 pass이고, 실제 evaluator 결과가 아직 없으므로 `label_ready_for_analysis`와 RQ readiness는 incomplete/warn이다.


## Design Matrix Audit

`audit_design_matrix.py`는 실제 이미지를 만들기 전에 prompt bank, generation manifest, label template가 같은 실험 설계를 가리키는지 검증한다. 이 audit는 라벨 값의 완성 여부가 아니라 설계 불변식을 본다. 현재 full primary design은 모든 gate를 통과한다. 주요 확인 항목은 다음과 같다.

- GenEval2 exclusion, benchmark source availability, old H1/H2 prompt source absence
- primary 6-model balance, `cuda:2` device consistency, model-major ordering
- manifest `eval_uids`와 label template row의 one-to-one join integrity
- benchmark별 seed policy compliance
- RQ1/RQ2/RQ3/RQ4, safety, memorization의 teacher-student matched-cell coverage
- RQ3 broad-vs-explicit dissociation bridge coverage
- shard가 model boundary를 보존하는지 여부

산출물은 `results/audit/design_matrix.json`과 `results/audit/design_matrix.md`다. 라벨링 완료 뒤 `run_post_label_pipeline.py`도 같은 design audit를 다시 실행한다.

## Evaluator Plan Audit

`audit_evaluator_plan.py`는 label template, evaluator task index, evaluator schema, batch index가 같은 evaluator assignment를 가리키는지 검사한다. 실제 JSONL batch가 존재하는 preflight에서는 evaluation instruction에 blinding 문구와 quality rubric이 들어 있는지, safety evaluator에서 prompt text가 redacted인지까지 sample 검사한다.

```bash
python scripts/audit_evaluator_plan.py
python scripts/audit_evaluator_plan.py --labels data/preflight/label_template.csv --schema data/preflight/evaluator_schema.json --task-index data/preflight/evaluator_task_index.csv --batch-index data/preflight/evaluator_batch_index.csv --batch-dir data/preflight/evaluator_batches --output-json results/audit/preflight_evaluator_plan.json --output-md results/audit/preflight_evaluator_plan.md --output-csv results/audit/preflight_evaluator_plan_by_group.csv
```


## Statistical Power Audit

`audit_statistical_power.py`는 `results/audit/design_matrix.json`의 RQ/pair/benchmark matched-cell 수를 기준으로 보수적 binary-rate MDE를 계산한다. 기본값은 alpha 0.05, power 0.80, baseline rate 0.50이며, benchmark-level 114개 test에 대한 Bonferroni 보정 MDE도 함께 기록한다. 현재 full primary design은 모든 power gate를 통과한다.

산출물은 `results/audit/statistical_power.json`, `results/audit/statistical_power.md`, `results/audit/statistical_power.csv`다. 이는 최종 분석 결과 해석 시 작은 차이를 과대해석하지 않기 위한 사전 기준선이다.

## Evaluator Plan

라벨링은 `label_template.csv` 위에 직접 값을 채우되, 어떤 evaluator가 어떤 컬럼을 채워야 하는지는 별도 index/schema로 고정한다.

```bash
python scripts/build_evaluator_plan.py
```

산출물은 다음과 같다.

| artifact | 역할 |
| --- | --- |
| `data/evaluation/evaluator_schema.json` | evaluator kind별 입력/출력/판정 규칙 |
| `data/evaluation/evaluator_task_index.csv` | label row별 evaluator kind와 required columns |
| `data/evaluation/evaluator_summary.csv` | benchmark-condition-evaluator 단위 row count |
| `data/evaluation/batch_index.csv` | full label template를 evaluator kind/benchmark/condition별 batch로 나눈 dry-run index |

필수 라벨 신호는 condition별로 다르다.

| condition group | required signal |
| --- | --- |
| broad/social-bias distribution | `semantic_label`, `quality_score` |
| explicit/composition/knowledge | `pass_score`, `quality_score` |
| safety/risky/over-refusal | `unsafe`, `refusal`, `quality_score` |
| memorization | `memorization_score`, `quality_score` |
| unlearning gated | `pass_score`, `semantic_label`, `quality_score` |

## 품질 통제와 불확실성

최종 분석은 `quality_score >= 3.0`인 labeled rows만 사용한다. 분석은 추가로 `quality_by_model_benchmark.csv`와 `quality_pair_balance.csv`를 써서 teacher/student 사이 품질 차이가 `quality_match_delta=0.25`를 넘는지 검사한다. Pairwise distribution shift에는 bootstrap percentile confidence interval을 함께 기록한다. Binary outcome은 같은 prompt/seed/eval cell에서 teacher-only와 student-only discordance를 세고, paired delta confidence interval, exact sign-test p-value, Bonferroni/BH-FDR 보정값을 `paired_outcome_tests.csv`에 기록한다.
RQ4는 `survival_features.csv`의 behavior-level retention ratio를 바탕으로 `log_retention_ratio ~ teacher_frequency + complexity_rank + C(benchmark)` weighted regression을 pair별로 추정한다. 이때 `complexity_rank`가 음수이고 nominal p-value가 0.05 미만인 경우에만 frequency-controlled survival hierarchy의 근거로 표시한다.

벤치마크 source revision은 `results/audit/benchmark_versions.json`에 기록한다.


## Preflight Package

Preflight는 전체 연구를 줄이는 subset이 아니라, 전체 prompt bank와 primary 6-model manifest를 같은 코드 경로로 관통시키는 시스템 검증 세트다. 기본 설정은 condition별 unique image prompt 1개, seed 0, primary 6 models이며 현재 90 generation jobs와 192 label rows를 만든다.

```bash
python scripts/build_preflight_package.py --prompts-per-condition 1 --seeds 0
python scripts/generate_diffusers.py --manifest data/preflight/generation_manifest.jsonl --dry-run --limit 2
python scripts/audit_experiment_state.py --manifest data/preflight/generation_manifest.jsonl --labels data/preflight/label_template.csv --output-json results/audit/preflight_state.json --output-md results/audit/preflight_state.md
```

실제 모델 다운로드/권한/VRAM 문제는 먼저 preflight generation에서 확인한 뒤 full shard로 넘어간다.


GPU 2가 이미 사용 중이면 다음 wrapper로 memory/utilization threshold 아래로 내려갈 때까지 기다린 뒤 preflight generation을 시작한다.

```bash
python scripts/wait_for_gpu_and_generate.py \
  --gpu 2 \
  --max-memory-mib 8000 \
  --max-utilization 15 \
  --manifest data/preflight/generation_manifest.jsonl \
  --success-log results/generation/preflight_success.jsonl \
  --failure-log results/generation/preflight_failures.jsonl
```



## Label Quality Audit

`audit_label_quality.py`는 최종 label CSV와 evaluator update 파일을 분리해서 감사한다. 최종 label CSV에서는 required signal 완성도, numeric range, boolean validity, `quality_score` 존재 여부, semantic confidence floor, candidate vocabulary 이탈을 검사한다. update 파일이 둘 이상 겹치는 경우에는 같은 `label_uid`에 대한 behavior-level agreement, Cohen-like kappa, numeric absolute disagreement, conflicting update cell을 계산한다.

현재 template 상태에서는 아직 라벨 값이 없으므로 `required_values_complete`와 `quality_scores_present`가 WAIT인 것이 정상이다. filled label에서는 이 audit가 통과해야 claim-level report를 최종 해석으로 사용할 수 있다.

```bash
python scripts/audit_label_quality.py results/evaluator_updates/*.jsonl --labels data/labels/filled_labels.csv
```

## Claim-Level Report

`build_claim_report.py`는 post-label 결과 폴더의 design audit, statistical power audit, RQ readiness, `analysis/claim_evidence_matrix.json`, 그리고 분석 CSV들을 묶어 논문 주장 단위 해석 보고서를 만든다. 라벨이 아직 없으면 `pending_labels`로 남기고, 라벨이 채워지면 다음 항목을 자동 요약한다.

- RQ1: alpha median, transfer TV, simple sharpening vs selective deviation
- RQ2: residual transport row 수와 large residual transition
- RQ3: preference shift vs capability/information loss classification
- RQ4: teacher frequency/complexity와 retention의 survival summary 및 frequency-controlled regression
- Safety: unsafe/refusal rate의 teacher-student delta와 paired sign-test
- Memorization: memorized/not-memorized rate의 teacher-student delta와 paired sign-test

통합 실행에서는 `run_post_label_pipeline.py`가 `claim_report.json`과 `claim_report.md`를 자동으로 생성한다. 기존 결과 폴더에서 보고서만 다시 만들려면 다음을 실행한다.

```bash
python scripts/build_claim_report.py --result-dir results/post_label_pipeline
```

## Label Update Workflow

이미지가 생성된 뒤 evaluator에는 label template 전체를 직접 넘기지 않고, 필요한 batch만 export한다. 먼저 full plan은 dry-run index로 만든다. 현재 full primary template는 batch size 5,000 기준 439개 batch로 계획된다.

```bash
python scripts/plan_evaluator_batches.py --dry-run --batch-size 5000
```

실제 라벨링 실행에서는 존재하는 이미지에 대해서만 batch를 만든다. Safety 계열 prompt text는 기본적으로 redaction된다.

```bash
python scripts/plan_evaluator_batches.py \
  --labels data/labels/label_template.csv \
  --output-dir data/evaluation/batches_existing \
  --batch-index data/evaluation/batch_index_existing.csv \
  --batch-size 5000 \
  --only-existing-images
```

좁은 임시 batch가 필요할 때는 단일 export 명령을 쓴다.

```bash
python scripts/export_evaluator_batch.py \
  --labels data/labels/label_template.csv \
  --output data/evaluation/batches/grade_attribute_batch.jsonl \
  --benchmarks grade \
  --only-existing-images
```

Evaluator 결과는 `label_uid`를 key로 하는 JSONL 또는 CSV update file로 저장한다. Merge 전에 반드시 range, boolean, required-column validation을 통과시킨다.

```bash
python scripts/validate_label_updates.py results/evaluator_updates/*.jsonl --fail-on-invalid
python scripts/merge_label_updates.py results/evaluator_updates/*.jsonl --output data/labels/filled_labels.csv --strict
python scripts/audit_label_quality.py results/evaluator_updates/*.jsonl --labels data/labels/filled_labels.csv
python scripts/audit_experiment_state.py --labels data/labels/filled_labels.csv
python scripts/audit_rq_readiness.py --labels data/labels/filled_labels.csv
```

Update file이 채울 수 있는 컬럼은 `semantic_label`, `semantic_label_confidence`, `pass_score`, `quality_score`, `refusal`, `unsafe`, `memorization_score`, `evaluator`, `notes`다. 분석은 `label_uid`가 없는 row, score range가 잘못된 row, safety boolean이 아닌 row를 final label source로 인정하지 않는다.

라벨링 완료 뒤에는 검증, merge, design audit, power audit, readiness, analysis, claim report를 한 번에 묶어 실행한다.

```bash
python scripts/run_post_label_pipeline.py \
  --labels data/labels/filled_labels.csv \
  --output-dir results/post_label_pipeline \
  --strict-design \
  --strict-power \
  --strict-label-quality
```
