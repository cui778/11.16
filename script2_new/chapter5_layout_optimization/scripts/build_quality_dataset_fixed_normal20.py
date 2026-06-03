"""
构建新固定协议下的布局质量数据集。
用于 Node-Feedback / Surrogate-Search 的布局生成训练。

输入：纯规则方法的新协议训练结果（不依赖旧数据）
输出：layout_quality_dataset_wide_fixed_normal20_v1.csv
"""
import json
import csv
import numpy as np
from pathlib import Path

ROOT = Path(r"E:\11.16\script2_new")
LAYOUTS_DIR = ROOT / "chapter5_layout_optimization" / "outputs" / "layouts"
REPORTS_DIR = ROOT / "outputs" / "reports"
CHECKPOINTS_DIR = ROOT / "outputs" / "model_checkpoints"
GRAPH_NPZ = ROOT / "input_1" / "graph_path_features.npz"
NODE_LIST_FILE = ROOT / "input_1" / "node_list.json"
CANDIDATE_FILE = ROOT / "input_1" / "candidate_nodes_new.json"
OUTPUT = ROOT / "chapter5_layout_optimization" / "outputs" / "structural_innovation" / "layout_quality_dataset_wide_fixed_normal20_v1.csv"

INF_HOP = 999

# 纯规则方法定义（不依赖训练结果）
METHODS = [
    {
        "method": "degree",
        "strategy": "degree",
        "layout_file": LAYOUTS_DIR / "degree" / "monitor_nodes_degree_N25.json",
        "seeds": [7, 42, 123],
        "tag_tpl": "48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s{}",
    },
    {
        "method": "betweenness",
        "strategy": "betweenness",
        "layout_file": LAYOUTS_DIR / "betweenness" / "monitor_nodes_betweenness_N25.json",
        "seeds": [7, 42, 123],
        "tag_tpl": "48h_control_ie420_normal20_raw_plus_residual_loc0p5_betweenness_N25_s{}",
    },
    {
        "method": "candidate_observability",
        "strategy": "candidate_observability",
        "layout_file": LAYOUTS_DIR / "candidate_observability" / "monitor_nodes_candidate_observability_N25.json",
        "seeds": [7, 42, 123],
        "tag_tpl": "ch5_fixed_candidate_observability_N25_normal20_rawres_loc0p5_s{}",
    },
    {
        "method": "two_stage_balanced_layout_v1",
        "strategy": "two_stage_balanced_layout_v1",
        "layout_file": LAYOUTS_DIR / "two_stage_balanced_layout_v1" / "monitor_nodes_two_stage_balanced_layout_v1_N25.json",
        "seeds": [7, 42, 123],
        "tag_tpl": "ch5_fixed_two_stage_balanced_layout_v1_N25_normal20_rawres_loc0p5_s{}",
    },
]


def sym_hop(shortest, i, j):
    a = int(shortest[i, j])
    b = int(shortest[j, j]) if i == j else int(shortest[j, i])
    vals = [x for x in (a, b) if x < INF_HOP]
    return min(vals) if vals else INF_HOP


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_layout_features(monitor_nodes, node_list, candidate_nodes, shortest):
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    cand_set = set(candidate_nodes)
    monitor_set = set(monitor_nodes)
    monitor_indices = [node_to_idx[m] for m in monitor_nodes if m in node_to_idx]
    candidate_indices = [node_to_idx[c] for c in candidate_nodes if c in node_to_idx]

    # direct/near/far
    best_hops = []
    for ci in candidate_indices:
        hops = [sym_hop(shortest, ci, mi) for mi in monitor_indices]
        best_hops.append(min(hops))
    best_hops_arr = np.array(best_hops, dtype=np.int32)
    finite = best_hops_arr[best_hops_arr < INF_HOP]

    direct = int(np.sum(finite == 0))
    near = int(np.sum((finite >= 1) & (finite <= 2)))
    far = int(np.sum(finite > 2))
    mean_hop = float(finite.mean()) if finite.size > 0 else float(INF_HOP)
    max_hop = int(finite.max()) if finite.size > 0 else 0

    # overlap
    overlap_count = len(monitor_set & cand_set)

    # monitor dispersion (mean hop between monitors)
    monitor_hops = []
    for i, mi in enumerate(monitor_indices):
        for mj in monitor_indices[i + 1:]:
            h = sym_hop(shortest, mi, mj)
            if h < INF_HOP:
                monitor_hops.append(h)
    monitor_dispersion = float(np.mean(monitor_hops)) if monitor_hops else 0.0

    # global mean hop (all nodes to nearest monitor)
    all_hops = []
    for ni in range(len(node_list)):
        hops = [sym_hop(shortest, ni, mi) for mi in monitor_indices]
        all_hops.append(min(hops))
    all_hops_arr = np.array([h for h in all_hops if h < INF_HOP], dtype=np.int32)
    global_mean_hop = float(all_hops_arr.mean()) if all_hops_arr.size > 0 else float(INF_HOP)

    return {
        "direct": direct,
        "near": near,
        "far": far,
        "mean_hop": mean_hop,
        "max_hop": max_hop,
        "overlap_count": overlap_count,
        "monitor_dispersion_mean_hop": monitor_dispersion,
        "global_mean_hop": global_mean_hop,
    }


def main():
    shortest = np.load(str(GRAPH_NPZ))["shortest_dist"]
    node_list = load_json(NODE_LIST_FILE)
    candidate_nodes = load_json(CANDIDATE_FILE)["candidate_nodes"]

    rows = []
    for m in METHODS:
        layout_data = load_json(m["layout_file"])
        monitor_nodes = layout_data["monitor_nodes"]
        struct = compute_layout_features(monitor_nodes, node_list, candidate_nodes, shortest)

        for seed in m["seeds"]:
            tag = m["tag_tpl"].format(seed)
            metrics_path = REPORTS_DIR / f"last_run_metrics_{tag}.json"
            history_path = CHECKPOINTS_DIR / f"training_history_{tag}.csv"

            if not metrics_path.exists():
                print(f"[SKIP] {tag} metrics not found")
                continue

            metrics = load_json(metrics_path)

            # Load validation metrics from training history
            val_metrics = {}
            if history_path.exists():
                with open(history_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    hist_rows = list(reader)
                    if hist_rows:
                        last = hist_rows[-1]
                        for k in ["val_mrr", "val_top1", "val_top3", "val_top5",
                                  "val_active_f1", "val_normal_window_fpr",
                                  "val_scene_recall", "val_scene_fpr", "val_scene_f1"]:
                            if k in last:
                                val_metrics[k] = float(last[k])

            row = {
                "method": m["method"],
                "strategy": m["strategy"],
                "layout_file": str(m["layout_file"]),
                "metrics_file": str(metrics_path),
                "budget": 25,
                "n": 25,
                "diagnosis_seed": seed,
                "protocol": "IE420+normal20",
                "feature_set": "raw_plus_residual",
                "lambda_loc": 0.5,
                # Structural features
                **struct,
                # Test metrics
                "scenario_mrr": metrics.get("mrr"),
                "scenario_top1": metrics.get("topk_recall_1"),
                "scenario_top3": metrics.get("topk_recall_3"),
                "scenario_top5": metrics.get("topk_recall_5"),
                "scenario_event_top1": metrics.get("event_level_top1"),
                "scenario_event_top3": metrics.get("event_level_top3"),
                "scenario_event_top5": metrics.get("event_level_top5"),
                "scenario_active_period_recall": metrics.get("active_period_recall"),
                # Validation metrics
                **val_metrics,
            }
            rows.append(row)
            print(f"[OK] {m['method']} seed={seed} MRR={metrics.get('mrr', 0):.4f}")

    # Write CSV
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[OK] Written {len(rows)} rows to {OUTPUT}")
    else:
        print("[WARN] No rows written")


if __name__ == "__main__":
    main()
