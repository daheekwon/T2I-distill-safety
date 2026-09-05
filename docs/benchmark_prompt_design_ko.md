# Benchmark/Axis Prompt Design for Distillation Inheritance

## 반영한 지시

- 기존 diversity 프로젝트의 H1/H2 prompt는 사용하지 않는다. 이 실험은 `./benchmarks`에 받은 각 benchmark의 native prompt와, benchmark가 정의한 diagnostic axis를 기준으로 독립 prompt bank를 만든다.
- GenEval2는 2026-09-04 사용자 지시에 따라 현재 primary 실험에서 제외한다.
- PoC subset이 아니라 현재 접근 가능한 benchmark prompt를 전부 포함한다. 단, HUB/EMMA는 teacher-student unlearning lineage가 valid할 때만 gated로 포함한다.
- 기본 generation device는 `cuda:2`다.
- primary model set은 6개 image generator다: SDXL Base, SDXL-Lightning 4-step, SD3.5 Large, SD3.5 Large Turbo, FLUX.2 [klein] 4B Base, FLUX.2 [klein] 4B Distilled.

## 카운트 정의

- `record`: 하나의 prompt가 하나의 평가 축/라벨에 연결된 단위다. GRADE, DIMCIM, WorldGenBench처럼 한 이미지 prompt가 여러 질문/checklist/attribute로 평가되면 record 수가 image prompt 수보다 크다.
- `unique_image_prompts`: 실제 image generation에 쓰는 중복 제거 prompt 수다.
- `primary_generation_jobs_attributed`: 각 benchmark/condition 안에서 `unique_image_prompts x seeds x 6 models`로 계산한 생성 job 수다. 서로 다른 axis가 같은 image prompt를 공유하는 경우 axis 단위 합은 전체 manifest job 수와 다를 수 있다.
- `primary_eval_rows_attributed`: `records x seeds x 6 models`로 계산한 label/evaluation row 수다.

## 전체 산출물

| artifact | 역할 |
| --- | --- |
| `data/prompts/inheritance_prompt_bank.jsonl` | benchmark별 prompt, axis, condition, evaluator metadata를 합친 master prompt bank |
| `data/prompts/prompt_benchmark_summary.csv` | benchmark 단위 record/image prompt/axis/seed/job 요약 |
| `data/prompts/prompt_axis_summary.csv` | 1,577개 axis-condition row 전체 요약 |
| `data/manifests/generation_manifest.jsonl` | primary 6-model generation job manifest, 전부 `cuda:2` |
| `data/labels/label_template.csv` | image 생성 후 VLM/classifier/metric 결과를 채울 label template |

## Benchmark별 설계

| benchmark | 주 평가 의도 | prompt/axis 구성 | seed 정책 | primary 생성/eval 규모 |
| --- | --- | --- | --- | --- |
| GRADE | broad prompt에서 teacher의 attribute distribution이 student로 어떻게 sharpening되는지 추정 | 2,430 records, 600 unique image prompts, 405 concept-attribute axes. 각 prompt는 object-level broad prompt로 생성하고, attribute question별 분포를 평가한다. | `distribution_32` | 115,200 generation jobs, 466,560 eval rows |
| DIMCIM | default preference와 explicit accessibility를 분리 | 19,136 records, 15,571 unique image prompts, 145 concept-attribute axes. broad condition은 930 coarse prompts를 여러 attribute 축으로 평가하고, explicit condition은 14,641 dense prompts로 해당 attribute를 직접 지정한다. | broad `distribution_16`, explicit `matched_4` | 440,664 generation jobs, 782,904 eval rows |
| T2I-CompBench++ | composition, relation, binding failure가 distillation 뒤 어디서 커지는지 확인 | 7,961 records/prompts, 9 axes: color, shape, texture, 2D spatial, 3D spatial, non-spatial relation, numeracy, complex composition, new object generalization. | `matched_4` | 191,064 generation/eval rows |
| WorldGenBench | implicit world knowledge/checklist survival | 10,340 checklist records, 1,072 unique image prompts, 582 domain/topic axes. humanities/nature prompt를 생성하고 checklist item 단위 pass/fail을 평가한다. | `matched_4` | 25,728 generation jobs, 248,160 eval rows |
| T2I-FactualBench | factual knowledge가 native vs text-injection에서 어떻게 유지되는지 비교 | 6,058 records/prompts, 118 axes. SKCM, SKCI, MKCC를 native prompt와 text-injection prompt로 각각 평가한다. | native/injected 모두 `matched_4` | 145,392 generation/eval rows |
| OVERT | benign over-refusal, unsafe compliance, safety-utility tradeoff | 8,170 records/prompts, 10 axes. benign full prompt, unsafe prompt, unsafe pair의 benign counterfactual을 분리한다. | `single_full` | 49,020 generation/eval rows |
| T2I-RiskyPrompt | risk category별 safety behavior inheritance | 7,853 label records, 6,433 unique image prompts, 14 fine-grained safety axes. multi-label prompt는 하나의 image prompt에 여러 safety label record를 연결한다. | `single_full` | 38,598 generation jobs, 47,118 eval rows |
| T2ISafety | fairness default와 safety category response 비교 | 2,686 records/prompts, 10 axes. fairness prompt는 distribution shift로 보고, safety prompt는 category별 safe/refusal behavior로 본다. | fairness `distribution_16`, safety `single_full` | 37,356 generation/eval rows |
| MemBench | memorization trigger가 distilled model에 얼마나 남는지 측정 | 4,064 records/prompts, 4 axes. SD1 trigger CSV와 MemAttn trigger/control prompt를 분리한다. | trigger `memorization_8`, control `matched_4` | 183,072 generation/eval rows |
| HUB | unlearning target retention, collateral loss, attack/multilingual robustness | benchmark는 받아두었지만 current primary prompt bank에는 포함하지 않는다. teacher-student valid lineage가 확인될 때 `--include-gated`로 활성화한다. | gated `unlearning_4` | current primary: 0 |
| EMMA | explicit/implicit concept erasure와 retain-set bias control | benchmark는 받아두었지만 current primary prompt bank에는 포함하지 않는다. valid erasure lineage가 확인될 때 `--include-gated`로 활성화한다. | gated `unlearning_4` | current primary: 0 |
| GenEval2 | 제외 | 사용하지 않는다. T2I-CompBench++, WorldGenBench, T2I-FactualBench가 composition/relation/knowledge 역할을 대체한다. | none | 0 |

## 축별 평가 방식

| 축군 | 포함 benchmark | 핵심 label column |
| --- | --- | --- |
| Default distribution / diversity | GRADE, DIMCIM broad, T2ISafety fairness | `semantic_label`, `quality_score` |
| Explicit attribute accessibility | DIMCIM explicit | `pass_score`, `semantic_label`, `quality_score` |
| Compositional binding / relation | T2I-CompBench++ | `pass_score`, `semantic_label`, `quality_score` |
| Implicit/factual knowledge | WorldGenBench, T2I-FactualBench | `pass_score`, `semantic_label`, `quality_score` |
| Safety and over-refusal | OVERT, T2I-RiskyPrompt, T2ISafety safety | `unsafe`, `refusal`, `pass_score`, `quality_score` |
| Memorization | MemBench | `memorization_score`, `pass_score`, `quality_score` |
| Unlearning lineage, optional | HUB, EMMA | `pass_score`, `semantic_label`, `quality_score` |

## 분석 연결

- RQ1 generic sharpening law: GRADE와 DIMCIM broad에서 teacher distribution `pT`와 student distribution `pS`를 만들고 `pS ~= normalize(pT ** alpha)`를 fit한다.
- RQ2 structured remapping: 같은 prompt/seed/evaluator cell에서 teacher label to student label transition matrix, residual transport, binary outcome paired sign-test를 계산한다.
- RQ3 preference vs capability loss: DIMCIM broad/explicit, factual native/injected, safety benign/unsafe counterfactual을 paired diagnostic으로 사용한다.
- RQ4 survival principle: teacher frequency, complexity level, benchmark domain을 feature로 두고 retention ratio를 설명하며, weighted regression으로 frequency-controlled complexity effect를 검정한다.

## 실행 명령

```bash
python scripts/build_prompt_bank.py
python scripts/summarize_prompt_bank.py
python scripts/build_generation_manifest.py --model-scope primary --job-order model_major
python scripts/build_label_template.py
python scripts/generate_diffusers.py --manifest data/manifests/generation_manifest.jsonl --dry-run --limit 1
```

실제 생성은 dry-run을 빼고 실행한다. label template가 채워진 뒤에는 다음을 실행한다.

```bash
python scripts/analyze_labels.py --labels data/labels/filled_labels.csv --model-scope primary
```
