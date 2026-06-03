# Chapter 4 Formal Model Comparison Audit

## Frozen Protocol

- dataset: `ie420_plus_normal20_v1`
- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N25.json`
- feature_set: `raw_plus_residual`
- lambda_loc: `0.5`
- lambda_kd: `0.0`
- lambda_active_kd: `0.0`
- split_mode: `scenario`
- epochs: `25`
- seeds: `7, 42, 123`

## Audit Result

- complete: `True`
- valid metrics files: `15/15`
- missing files: `0`
- invalid files: `0`

## Notes

- Smoke metrics are excluded from the formal summary.
- Existing deep-attention checkpoints are reused and are not overwritten.
- The historical old-protocol comparison is archived only after a complete audit.
