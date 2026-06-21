# Chapter 4 Method Ablation Protocol

## Purpose

The supplementary experiments isolate three sources of the Chapter 4 model
gain while keeping the formal diagnosis protocol unchanged.

| Experiment | Controlled comparison | Scientific question |
|---|---|---|
| DeepAttn depth | L1 / L2 / L3 / L4 | Does iterative path-guided refinement improve localization? |
| Path prior | full / distance-only / content-only at L3 | Do hydraulic path-feature values contribute beyond content attention? |
| Temporal encoder | Hydraulic-Inverse-GRU / Hydraulic-Inverse-LSTM | Is the single-layer result mainly caused by the temporal encoder choice? |

`content-only` removes all path-feature values from the attention logits but
retains the reachability mask. It does not introduce attention between
physically disconnected node pairs.

## Frozen protocol

```text
dataset       = ie420_plus_normal20_v1
defect matrix = formal_conservative420_seed42
layout        = degree_N25
features      = raw_plus_residual
lambda_loc    = 0.5
lambda_kd     = 0
split         = scenario
epochs        = 25
seed screening= 42
confirm seeds = 7 / 42 / 123
```

Persistent, fulltime, legacy and seedset10 data are not used.

## Execution

Seed-42 screening:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "E:\11.16\script2_new\chapter4_diagnosis_model\scripts\start_ch4_method_ablation_detached.ps1" `
  -Mode seed42
```

Three-seed confirmation after reviewing the screening:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "E:\11.16\script2_new\chapter4_diagnosis_model\scripts\start_ch4_method_ablation_detached.ps1" `
  -Mode confirm
```

`confirm` runs seeds 7 and 123 for L1/L2/L3/L4, L3 content-only and
Hydraulic-Inverse-LSTM. The distance-only variant remains a seed-42 mechanism
probe and is not expanded by default.

## Interpretation rules

- Depth contribution is supported only if the multi-layer variants improve
  consistently over the shallow variants on MRR and Top-1/Top-3.
- Path-prior contribution is supported only if `full` improves over
  `content-only`; `distance-only` indicates whether distance channels explain
  most of that gain.
- The LSTM control is not a replacement main model. It only tests whether the
  single-layer comparison is dominated by GRU/LSTM encoder choice.
- Seed-42 screening results are not described as multi-seed stability evidence.
- If an expected mechanism is not supported, the method claim must be narrowed
  rather than selecting only favorable metrics.
