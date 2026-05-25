# Chapter 5 Visualizations

This directory stores visualization scripts dedicated to Chapter 5 layout optimization.

Current scripts:

- `plot_layout_comparison.py`
  - Plot spatial comparison of candidate nodes and selected monitor layouts.
  - Default comparison: `degree / candidate_observability / identifiability_driven`
- `plot_split_generalization_comparison.py`
  - Plot `scenario split` vs `node_holdout` performance comparison for Chapter 5 layouts.
  - Default input:
    - `outputs/ch5_split_generalization_comparison_N25_seed42.csv`
  - Default output:
    - `figures/ch5_split_generalization_comparison_N25_seed42.png`

Generated figures are stored in:

- `E:\11.16\script2_new\chapter5_layout_optimization\figures\`
