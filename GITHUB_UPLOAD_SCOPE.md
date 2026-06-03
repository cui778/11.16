# 11.16 Research Package Upload Scope

This repository is a curated research-code and evidence package for the SWMM-based I/E defect diagnosis project. It is not a full mirror of the local `E:\11.16` workspace.

## What This Repository Is For

Use this repository to review and continue:

- SWMM / PySWMM data generation scripts and protocol notes.
- Chapter 3 data-generation evidence tables.
- Chapter 4 diagnosis-model experiment scripts and formal result summaries.
- Chapter 5 monitoring-layout optimization scripts and formal result summaries.
- Lightweight JSON / CSV / Markdown files needed for thesis writing and PPT reconstruction.

The companion writing repository is:

- `https://github.com/cui778/thesis_writing_repo`

For writing and PPT work, use `thesis_writing_repo` as the main source of chapter text, figure source tables, evidence maps, and protocol notes. Use this repository only when code-level or experiment-provenance context is needed.

## What Is Intentionally Excluded

The local workspace contains many large or temporary artifacts that should not be uploaded to GitHub:

- Raw SWMM generated datasets, especially `*.parquet` node time-series files.
- Model checkpoints, including `*.pth`, `*.pt`, and `*.ckpt`.
- Large simulation outputs, temporary logs, PID files, and cache folders.
- Original project documents, PDFs, Word files, and full raw engineering material.
- Historical exploratory outputs that are not part of the current formal evidence package.

These files stay on the local machine. If a future reader needs to reproduce training from scratch, rebuild the datasets locally from the documented scripts and protocol files rather than pulling raw data from GitHub.

## Current Formal Protocol

The current thesis protocol is:

- Chapter 3: SWMM-based IE420 time-gated defect scenarios plus normal operating scenarios.
- Chapter 4: `ie420_plus_normal20_v1`, `raw_plus_residual`, `lambda_loc=0.5`, `degree_N25`, `scenario split`.
- Chapter 5: same fixed diagnosis protocol as Chapter 4, with the monitoring node set `S` changed by layout method.

The formal writing side keeps the final text and figure source data under:

- `thesis_writing_repo/chapters`
- `thesis_writing_repo/figures`
- `thesis_writing_repo/notes`
- `thesis_writing_repo/docs`

## How Web GPT Should Use This Repository

When using this repository from the web interface:

1. Read this file first.
2. Read `script2_new/chapter3_data_generation/README.md`.
3. Read `script2_new/chapter4_diagnosis_model/README.md`.
4. Read `script2_new/chapter5_layout_optimization/README.md`.
5. Use `script2_new/experiments/tables/` and each chapter's `outputs/` folders for numeric evidence.
6. Do not infer missing conclusions from old exploratory files unless they are explicitly marked as formal evidence.

For PPT or thesis text, cross-check with `thesis_writing_repo` before writing final language.
