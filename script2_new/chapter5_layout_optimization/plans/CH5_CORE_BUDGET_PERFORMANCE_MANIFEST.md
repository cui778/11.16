# CH5 core budget performance manifest

- Methods: Degree / Cand-Obs / Two-stage v1 / v0_2 clean / v2_2 clean
- Budgets: 5, 10, 15, 20, 25
- Diagnosis seeds: 7, 42, 123
- Split mode: scenario
- Row count: 75

## ch5_core_degree_N5_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N5.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N5_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N5.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_degree_N5_scenario_s7
```

## ch5_core_degree_N5_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N5.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N5_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N5.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_degree_N5_scenario_s42
```

## ch5_core_degree_N5_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N5.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N5_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N5.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_degree_N5_scenario_s123
```

## ch5_core_candidate_observability_N5_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N5.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N5_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N5.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_candidate_observability_N5_scenario_s7
```

## ch5_core_candidate_observability_N5_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N5.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N5_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N5.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_candidate_observability_N5_scenario_s42
```

## ch5_core_candidate_observability_N5_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N5.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N5_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N5.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_candidate_observability_N5_scenario_s123
```

## ch5_core_two_stage_balanced_layout_v1_N5_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N5.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N5_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N5.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_two_stage_balanced_layout_v1_N5_scenario_s7
```

## ch5_core_two_stage_balanced_layout_v1_N5_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N5.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N5_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N5.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_two_stage_balanced_layout_v1_N5_scenario_s42
```

## ch5_core_two_stage_balanced_layout_v1_N5_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N5.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N5_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N5.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_two_stage_balanced_layout_v1_N5_scenario_s123
```

## ch5_core_v0_2_clean_scenario_N5_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N5_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N5_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N5_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v0_2_clean_scenario_N5_scenario_s7
```

## ch5_core_v0_2_clean_scenario_N5_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N5_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N5_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N5_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v0_2_clean_scenario_N5_scenario_s42
```

## ch5_core_v0_2_clean_scenario_N5_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N5_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N5_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N5_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v0_2_clean_scenario_N5_scenario_s123
```

## ch5_core_v2_2_clean_generalization_N5_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N5_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N5_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N5_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v2_2_clean_generalization_N5_scenario_s7
```

## ch5_core_v2_2_clean_generalization_N5_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N5_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N5_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N5_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v2_2_clean_generalization_N5_scenario_s42
```

## ch5_core_v2_2_clean_generalization_N5_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N5_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N5_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N5\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N5_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v2_2_clean_generalization_N5_scenario_s123
```

## ch5_core_degree_N10_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N10.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N10_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N10.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_degree_N10_scenario_s7
```

## ch5_core_degree_N10_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N10.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N10_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N10.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_degree_N10_scenario_s42
```

## ch5_core_degree_N10_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N10.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N10_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N10.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_degree_N10_scenario_s123
```

## ch5_core_candidate_observability_N10_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N10.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N10_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N10.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_candidate_observability_N10_scenario_s7
```

## ch5_core_candidate_observability_N10_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N10.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N10_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N10.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_candidate_observability_N10_scenario_s42
```

## ch5_core_candidate_observability_N10_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N10.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N10_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N10.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_candidate_observability_N10_scenario_s123
```

## ch5_core_two_stage_balanced_layout_v1_N10_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N10.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N10_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N10.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_two_stage_balanced_layout_v1_N10_scenario_s7
```

## ch5_core_two_stage_balanced_layout_v1_N10_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N10.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N10_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N10.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_two_stage_balanced_layout_v1_N10_scenario_s42
```

## ch5_core_two_stage_balanced_layout_v1_N10_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N10.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N10_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N10.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_two_stage_balanced_layout_v1_N10_scenario_s123
```

## ch5_core_v0_2_clean_scenario_N10_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N10_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N10_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N10_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v0_2_clean_scenario_N10_scenario_s7
```

## ch5_core_v0_2_clean_scenario_N10_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N10_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N10_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N10_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v0_2_clean_scenario_N10_scenario_s42
```

## ch5_core_v0_2_clean_scenario_N10_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N10_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N10_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N10_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v0_2_clean_scenario_N10_scenario_s123
```

## ch5_core_v2_2_clean_generalization_N10_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N10_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N10_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N10_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v2_2_clean_generalization_N10_scenario_s7
```

## ch5_core_v2_2_clean_generalization_N10_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N10_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N10_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N10_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v2_2_clean_generalization_N10_scenario_s42
```

## ch5_core_v2_2_clean_generalization_N10_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N10_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N10_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N10\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N10_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v2_2_clean_generalization_N10_scenario_s123
```

## ch5_core_degree_N15_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N15.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N15_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N15.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_degree_N15_scenario_s7
```

## ch5_core_degree_N15_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N15.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N15_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N15.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_degree_N15_scenario_s42
```

## ch5_core_degree_N15_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N15.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N15_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N15.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_degree_N15_scenario_s123
```

## ch5_core_candidate_observability_N15_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N15.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N15_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N15.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_candidate_observability_N15_scenario_s7
```

## ch5_core_candidate_observability_N15_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N15.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N15_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N15.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_candidate_observability_N15_scenario_s42
```

## ch5_core_candidate_observability_N15_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N15.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N15_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N15.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_candidate_observability_N15_scenario_s123
```

## ch5_core_two_stage_balanced_layout_v1_N15_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N15.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N15_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N15.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_two_stage_balanced_layout_v1_N15_scenario_s7
```

## ch5_core_two_stage_balanced_layout_v1_N15_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N15.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N15_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N15.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_two_stage_balanced_layout_v1_N15_scenario_s42
```

## ch5_core_two_stage_balanced_layout_v1_N15_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N15.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N15_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N15.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_two_stage_balanced_layout_v1_N15_scenario_s123
```

## ch5_core_v0_2_clean_scenario_N15_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N15_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N15_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N15_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v0_2_clean_scenario_N15_scenario_s7
```

## ch5_core_v0_2_clean_scenario_N15_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N15_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N15_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N15_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v0_2_clean_scenario_N15_scenario_s42
```

## ch5_core_v0_2_clean_scenario_N15_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N15_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N15_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N15_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v0_2_clean_scenario_N15_scenario_s123
```

## ch5_core_v2_2_clean_generalization_N15_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N15_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N15_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N15_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v2_2_clean_generalization_N15_scenario_s7
```

## ch5_core_v2_2_clean_generalization_N15_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N15_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N15_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N15_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v2_2_clean_generalization_N15_scenario_s42
```

## ch5_core_v2_2_clean_generalization_N15_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N15_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N15_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N15\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N15_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v2_2_clean_generalization_N15_scenario_s123
```

## ch5_core_degree_N20_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N20.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N20_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N20.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_degree_N20_scenario_s7
```

## ch5_core_degree_N20_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N20.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N20_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N20.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_degree_N20_scenario_s42
```

## ch5_core_degree_N20_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N20.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N20_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N20.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_degree_N20_scenario_s123
```

## ch5_core_candidate_observability_N20_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N20.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N20_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N20.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_candidate_observability_N20_scenario_s7
```

## ch5_core_candidate_observability_N20_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N20.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N20_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N20.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_candidate_observability_N20_scenario_s42
```

## ch5_core_candidate_observability_N20_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N20.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N20_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N20.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_candidate_observability_N20_scenario_s123
```

## ch5_core_two_stage_balanced_layout_v1_N20_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N20.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N20_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N20.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_two_stage_balanced_layout_v1_N20_scenario_s7
```

## ch5_core_two_stage_balanced_layout_v1_N20_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N20.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N20_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N20.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_two_stage_balanced_layout_v1_N20_scenario_s42
```

## ch5_core_two_stage_balanced_layout_v1_N20_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N20.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N20_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N20.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_two_stage_balanced_layout_v1_N20_scenario_s123
```

## ch5_core_v0_2_clean_scenario_N20_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N20_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N20_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N20_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v0_2_clean_scenario_N20_scenario_s7
```

## ch5_core_v0_2_clean_scenario_N20_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N20_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N20_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N20_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v0_2_clean_scenario_N20_scenario_s42
```

## ch5_core_v0_2_clean_scenario_N20_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N20_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N20_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N20_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v0_2_clean_scenario_N20_scenario_s123
```

## ch5_core_v2_2_clean_generalization_N20_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N20_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N20_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N20_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v2_2_clean_generalization_N20_scenario_s7
```

## ch5_core_v2_2_clean_generalization_N20_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N20_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N20_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N20_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v2_2_clean_generalization_N20_scenario_s42
```

## ch5_core_v2_2_clean_generalization_N20_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N20_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N20_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N20\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N20_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v2_2_clean_generalization_N20_scenario_s123
```

## ch5_core_degree_N25_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N25.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N25_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N25.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_degree_N25_scenario_s7
```

## ch5_core_degree_N25_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N25.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N25_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N25.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_degree_N25_scenario_s42
```

## ch5_core_degree_N25_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N25.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_degree_N25_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\degree\monitor_nodes_degree_N25.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_degree_N25_scenario_s123
```

## ch5_core_candidate_observability_N25_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N25.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N25_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N25.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_candidate_observability_N25_scenario_s7
```

## ch5_core_candidate_observability_N25_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N25.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N25_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N25.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_candidate_observability_N25_scenario_s42
```

## ch5_core_candidate_observability_N25_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N25.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_candidate_observability_N25_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\candidate_observability\monitor_nodes_candidate_observability_N25.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_candidate_observability_N25_scenario_s123
```

## ch5_core_two_stage_balanced_layout_v1_N25_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N25.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N25_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N25.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_two_stage_balanced_layout_v1_N25_scenario_s7
```

## ch5_core_two_stage_balanced_layout_v1_N25_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N25.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N25_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N25.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_two_stage_balanced_layout_v1_N25_scenario_s42
```

## ch5_core_two_stage_balanced_layout_v1_N25_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N25.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_two_stage_balanced_layout_v1_N25_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layouts\two_stage_balanced_layout_v1\monitor_nodes_two_stage_balanced_layout_v1_N25.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_two_stage_balanced_layout_v1_N25_scenario_s123
```

## ch5_core_v0_2_clean_scenario_N25_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N25_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N25_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N25_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v0_2_clean_scenario_N25_scenario_s7
```

## ch5_core_v0_2_clean_scenario_N25_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N25_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N25_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N25_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v0_2_clean_scenario_N25_scenario_s42
```

## ch5_core_v0_2_clean_scenario_N25_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N25_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v0_2_clean_scenario_N25_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v0_2_clean_scenario\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v0_2_clean_scenario_N25_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v0_2_clean_scenario_N25_scenario_s123
```

## ch5_core_v2_2_clean_generalization_N25_scenario_s7

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N25_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N25_scenario_s7.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N25_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 7 --output-tag ch5_core_v2_2_clean_generalization_N25_scenario_s7
```

## ch5_core_v2_2_clean_generalization_N25_scenario_s42

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N25_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N25_scenario_s42.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N25_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 42 --output-tag ch5_core_v2_2_clean_generalization_N25_scenario_s42
```

## ch5_core_v2_2_clean_generalization_N25_scenario_s123

- layout: `E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N25_layoutseed42.json`
- metrics: `E:\11.16\script2_new\outputs\reports\last_run_metrics_ch5_core_v2_2_clean_generalization_N25_scenario_s123.json`

```powershell
conda run -n swmm_gpu python "E:\11.16\script2_new\scripts\train_privileged_teacher_student.py" --student-monitors "E:\11.16\script2_new\chapter5_layout_optimization\outputs\layout_stability\layout_files\v2_2_clean_generalization\N25\layout_seed_42\monitor_nodes_learnable_layout_network_v2_2_clean_generalization_N25_layoutseed42.json" --teacher-subdir time_gated_full_ie_v4_formal_conservative420_seed42 --student-model-type hydraulic_inverse_deepattn --num-epochs 25 --lambda-kd 0 --lambda-active-kd 0 --split-mode scenario --seed 123 --output-tag ch5_core_v2_2_clean_generalization_N25_scenario_s123
```
