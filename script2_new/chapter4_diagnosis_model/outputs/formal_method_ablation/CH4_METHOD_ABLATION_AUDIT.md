# Chapter 4 Method Ablation Audit

## Frozen protocol

- dataset: `ie420_plus_normal20_v1`
- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N25.json`
- features: `raw_plus_residual`
- lambda_loc: `0.5`
- split: `scenario`
- KD: disabled
- epochs: `25`

## Mechanism controls

- depth: DeepAttn L1/L2/L3/L4 with full path prior
- path prior: full/distance_only/content_only at DeepAttn L3
- temporal encoder: Hydraulic-Inverse-LSTM with full path prior
- content_only retains the reachability mask but removes all path-feature values from attention logits

## Status

- complete: `True`
- valid: `19/19`
- invalid: `0`
- missing: `0`

Depth, content-only path ablation and the LSTM encoder control use
seeds 7/42/123. Distance-only is retained as a seed-42 mechanism probe.