# CH5 Overlap-Controlled Probe at N=25

## Purpose

This probe was run before a full budget sweep to answer a more basic question:

- if budget is fixed
- and candidate-monitor overlap is also fixed

does layout structure still affect localization performance?

## Fixed-overlap setting

Budget:

- `N=25`

Overlap anchor:

- use the existing `degree N25` overlap count as the fixed target
- target overlap = `12`

New probe layouts:

- [monitor_nodes_overlap_controlled_candidate_focus_N25.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts/overlap_controlled_candidate_focus/monitor_nodes_overlap_controlled_candidate_focus_N25.json)
- [monitor_nodes_overlap_controlled_balanced_N25.json](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts/overlap_controlled_balanced/monitor_nodes_overlap_controlled_balanced_N25.json)

Static summary:

- [overlap_controlled_probe_summary.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/overlap_controlled_probe_summary.csv)

## Static layout comparison

At the same overlap count `12`:

- `degree`: `direct=12`, `near=21`, `far=17`, `mean_hop=3.48`
- `overlap_controlled_candidate_focus`: `direct=12`, `near=38`, `far=0`, `mean_hop=0.98`
- `overlap_controlled_balanced`: `direct=12`, `near=37`, `far=1`, `mean_hop=1.04`

This confirms that fixed overlap does not imply fixed observability quality.

## Scenario-split results

Result table:

- [ch5_overlap_controlled_probe_scenario_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_overlap_controlled_probe_scenario_N25_seed42.csv)

Main numbers:

- `degree`: `MRR=0.7933`, `Top-1=0.6721`, overlap `12`
- `overlap_controlled_candidate_focus`: `MRR=0.8991`, `Top-1=0.8285`, overlap `12`
- `overlap_controlled_balanced`: `MRR=0.8866`, `Top-1=0.7946`, overlap `12`
- `candidate_observability`: `MRR=0.9097`, `Top-1=0.8413`, overlap `15`

Interpretation:

- under the same overlap count, both new layouts still clearly outperform `degree`
- therefore the Chapter-5 gain cannot be explained only by increasing `S ∩ C`
- layout structure still matters

## Node-holdout results

Result table:

- [ch5_overlap_controlled_probe_node_holdout_N25_seed42.csv](/E:/11.16/script2_new/chapter5_layout_optimization/outputs/ch5_overlap_controlled_probe_node_holdout_N25_seed42.csv)

Main numbers:

- `degree`: `MRR=0.3655`, `Top-1=0.1545`, overlap `12`
- `overlap_controlled_candidate_focus`: `MRR=0.2416`, `Top-1=0.0957`, overlap `12`
- `overlap_controlled_balanced`: `MRR=0.2473`, `Top-1=0.0829`, overlap `12`
- `candidate_observability`: `MRR=0.2936`, `Top-1=0.0987`, overlap `15`

Interpretation:

- the fixed-overlap probe methods do not beat `degree` under node holdout
- this means the new layouts improve same-task scenario performance, but still do not solve unseen-node generalization
- so the current Chapter-5 narrative should be:
  - overlap is not the whole story
  - but stronger candidate-side observability alone is still not sufficient for node generalization

## Current conclusion for the mainline

The probe has already answered one key scientific doubt:

- performance gains are not only overlap gains

But it also exposed the next problem:

- the Chapter-5 task-aware layouts remain too tied to the current candidate structure when tested on unseen-node generalization

So the next priority should be:

1. keep this overlap-controlled result as a formal ablation
2. add stronger global/generalization constraints to the main method
3. only then decide whether to expand to the full budget sweep
