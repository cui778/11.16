"""
Sequential budget sweep: 4 methods x 4 budgets = 16 runs.
Runs ONE training at a time to avoid memory issues.
"""
import gc
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "D:/conda3/envs/swmm_gpu/python.exe"
SCRIPT = "E:/11.16/script2_new/scripts/train_privileged_teacher_student.py"

LAYOUTS = "E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts"

# (layout_file, output_tag)
RUNS = [
    # Degree
    (f"{LAYOUTS}/degree/monitor_nodes_degree_N5.json",
     "48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N5_s42"),
    (f"{LAYOUTS}/degree/monitor_nodes_degree_N10.json",
     "48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N10_s42"),
    (f"{LAYOUTS}/degree/monitor_nodes_degree_N15.json",
     "48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N15_s42"),
    (f"{LAYOUTS}/degree/monitor_nodes_degree_N20.json",
     "48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N20_s42"),
    # Betweenness
    (f"{LAYOUTS}/betweenness/monitor_nodes_betweenness_N5.json",
     "48h_control_ie420_normal20_raw_plus_residual_loc0p5_betweenness_N5_s42"),
    (f"{LAYOUTS}/betweenness/monitor_nodes_betweenness_N10.json",
     "48h_control_ie420_normal20_raw_plus_residual_loc0p5_betweenness_N10_s42"),
    (f"{LAYOUTS}/betweenness/monitor_nodes_betweenness_N15.json",
     "48h_control_ie420_normal20_raw_plus_residual_loc0p5_betweenness_N15_s42"),
    (f"{LAYOUTS}/betweenness/monitor_nodes_betweenness_N20.json",
     "48h_control_ie420_normal20_raw_plus_residual_loc0p5_betweenness_N20_s42"),
    # Two-stage v1
    (f"{LAYOUTS}/two_stage_balanced_layout_v1/monitor_nodes_two_stage_balanced_layout_v1_N5.json",
     "ch5_fixed_two_stage_balanced_layout_v1_N5_normal20_rawres_loc0p5_s42"),
    (f"{LAYOUTS}/two_stage_balanced_layout_v1/monitor_nodes_two_stage_balanced_layout_v1_N10.json",
     "ch5_fixed_two_stage_balanced_layout_v1_N10_normal20_rawres_loc0p5_s42"),
    (f"{LAYOUTS}/two_stage_balanced_layout_v1/monitor_nodes_two_stage_balanced_layout_v1_N15.json",
     "ch5_fixed_two_stage_balanced_layout_v1_N15_normal20_rawres_loc0p5_s42"),
    (f"{LAYOUTS}/two_stage_balanced_layout_v1/monitor_nodes_two_stage_balanced_layout_v1_N20.json",
     "ch5_fixed_two_stage_balanced_layout_v1_N20_normal20_rawres_loc0p5_s42"),
    # Embedding-Guided
    (f"{LAYOUTS}/embedding_guided_clean_fixed/monitor_nodes_embedding_guided_clean_N5.json",
     "ch5_fixed_embedding_guided_clean_new_N5_normal20_rawres_loc0p5_s42"),
    (f"{LAYOUTS}/embedding_guided_clean_fixed/monitor_nodes_embedding_guided_clean_N10.json",
     "ch5_fixed_embedding_guided_clean_new_N10_normal20_rawres_loc0p5_s42"),
    (f"{LAYOUTS}/embedding_guided_clean_fixed/monitor_nodes_embedding_guided_clean_N15.json",
     "ch5_fixed_embedding_guided_clean_new_N15_normal20_rawres_loc0p5_s42"),
    (f"{LAYOUTS}/embedding_guided_clean_fixed/monitor_nodes_embedding_guided_clean_N20.json",
     "ch5_fixed_embedding_guided_clean_new_N20_normal20_rawres_loc0p5_s42"),
]

COMMON = [
    "--seed", "42",
    "--teacher-subdir", "ie420_plus_normal20_v1",
    "--split-mode", "scenario",
    "--feature-set", "raw_plus_residual",
    "--lambda-loc", "0.5",
    "--student-model-type", "hydraulic_inverse_deepattn",
    "--lambda-kd", "0",
    "--lambda-active-kd", "0",
]

REPORTS = Path("E:/11.16/script2_new/outputs/reports")

def main():
    total = len(RUNS)
    for i, (layout, tag) in enumerate(RUNS):
        metrics_file = REPORTS / f"last_run_metrics_{tag}.json"
        if metrics_file.exists():
            print(f"[{i+1}/{total}] SKIP (already done): {tag}")
            continue

        print(f"\n[{i+1}/{total}] Training: {tag}")
        cmd = [PYTHON, SCRIPT,
               "--student-monitors", layout,
               "--output-tag", tag] + COMMON
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
        else:
            print(f"  DONE: {tag}")
        # Release memory between runs
        gc.collect()
        time.sleep(60)

    print("\n=== ALL BUDGET RUNS COMPLETE ===")

if __name__ == "__main__":
    main()
