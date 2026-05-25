#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取两次或多次训练的 last_run_metrics_*.json，输出对比表。
用法:
  # 两两对比
  python scripts/compare_model_runs.py gru_gcn_time_gated hydraulic_inverse_time_gated
  python scripts/compare_model_runs.py gru_gcn_time_gated hydraulic_inverse_time_gated --append-log

  # 三（或多）个 run 一张表
  python scripts/compare_model_runs.py gru_gcn_node_holdout gru_only_node_holdout hydraulic_inverse_node_holdout
"""
from pathlib import Path
import json
import argparse
from datetime import datetime

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = SCRIPT_ROOT / "outputs" / "reports"


def load_metrics(run_name_or_path):
    """run_name 如 gru_gcn_time_gated，或直接传 json 路径"""
    p = Path(run_name_or_path)
    if p.suffix == ".json" and p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = p.stem.replace("last_run_metrics_", "") if p.stem.startswith("last_run_metrics_") else p.stem
        return data, name
    fpath = REPORTS_DIR / f"last_run_metrics_{run_name_or_path}.json"
    if not fpath.exists():
        raise FileNotFoundError(f"未找到: {fpath}\n请先跑完对应训练并带 --run-name {run_name_or_path}")
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f), run_name_or_path


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _get_val(m, key, key_sub=None):
    if key_sub is not None:
        return (m.get(key) or {}).get(key_sub)
    return m.get(key)


def run_two(m_baseline, name_baseline, m_current, name_current, append_log):
    """两 run 对比（原逻辑）"""
    rows = [["指标", "基线 " + name_baseline, "当前 " + name_current, "变化"]]

    def row(label, key_b, key_c=None, key_b_sub=None, key_c_sub=None):
        vb = _get_val(m_baseline, key_b, key_b_sub)
        vc = _get_val(m_current, key_c or key_b, key_c_sub or key_b_sub)
        if vb is not None and vc is not None and isinstance(vb, (int, float)) and isinstance(vc, (int, float)):
            delta = vc - vb
            change = f"{delta:+.4f}" if isinstance(delta, float) else f"{delta:+d}"
        else:
            change = "—"
        rows.append([label, fmt(vb), fmt(vc), change])

    row("整体 MRR", "mrr")
    row("整体 Top-1", "top1")
    row("整体 Top-3", "top3")
    row("整体 Top-5", "top5")
    row("见过节点 Top-1", "seen_nodes", "seen_nodes", "top1", "top1")
    row("见过节点 MRR", "seen_nodes", "seen_nodes", "mrr", "mrr")
    row("未见过节点 Top-1", "unseen_nodes", "unseen_nodes", "top1", "top1")
    row("未见过节点 MRR", "unseen_nodes", "unseen_nodes", "mrr", "mrr")
    for t in ("I", "E", "P"):
        btop = (m_baseline.get("by_type_top1") or {}).get(t)
        ctop = (m_current.get("by_type_top1") or {}).get(t)
        change = f"{ctop - btop:+.4f}" if (btop is not None and ctop is not None) else "—"
        rows.append([f"  {t} 类 Top-1", fmt(btop), fmt(ctop), change])

    col_widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    sep = " | "
    print("\n" + "=" * (sum(col_widths) + 3 * len(sep)))
    print("对比结果（基线 vs 当前）")
    print("=" * (sum(col_widths) + 3 * len(sep)))
    for i, r in enumerate(rows):
        print(sep.join(r[j].ljust(col_widths[j]) for j in range(len(r))))
        if i == 0:
            print("-" * (sum(col_widths) + 3 * len(sep)))
    print("=" * (sum(col_widths) + 3 * len(sep)))
    print(f"\n基线: {REPORTS_DIR / f'last_run_metrics_{name_baseline}.json'}")
    print(f"当前: {REPORTS_DIR / f'last_run_metrics_{name_current}.json'}")

    if append_log:
        log_path = REPORTS_DIR / "STEP_RESULTS_LOG.md"
        if log_path.exists():
            block = ["", "### 对比结果（自动追加）", f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
                     "| 指标 | 基线 | 当前 | 变化 |", "|------|------|------|------|"]
            for r in rows[1:]:
                block.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
            block.append("")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(block))
            print(f"\n[OK] 已追加到 {log_path}")


def run_multi(metrics_list, names, append_log):
    """多 run 一张表（无变化列）"""
    rows = [["指标"] + list(names)]
    keys = [
        ("整体 MRR", "mrr", None),
        ("整体 Top-1", "top1", None),
        ("整体 Top-3", "top3", None),
        ("整体 Top-5", "top5", None),
        ("见过节点 Top-1", "seen_nodes", "top1"),
        ("见过节点 MRR", "seen_nodes", "mrr"),
        ("未见过节点 Top-1", "unseen_nodes", "top1"),
        ("未见过节点 MRR", "unseen_nodes", "mrr"),
    ]
    for label, key, sub in keys:
        row = [label]
        for m in metrics_list:
            v = _get_val(m, key, sub)
            row.append(fmt(v))
        rows.append(row)
    for t in ("I", "E", "P"):
        row = [f"  {t} 类 Top-1"]
        for m in metrics_list:
            v = (m.get("by_type_top1") or {}).get(t)
            row.append(fmt(v))
        rows.append(row)

    col_widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    sep = " | "
    print("\n" + "=" * (sum(col_widths) + (len(rows[0]) - 1) * len(sep)))
    print("对比结果（多 run）")
    print("=" * (sum(col_widths) + (len(rows[0]) - 1) * len(sep)))
    for i, r in enumerate(rows):
        print(sep.join(r[j].ljust(col_widths[j]) for j in range(len(r))))
        if i == 0:
            print("-" * (sum(col_widths) + (len(rows[0]) - 1) * len(sep)))
    print("=" * (sum(col_widths) + (len(rows[0]) - 1) * len(sep)))

    if append_log:
        log_path = REPORTS_DIR / "STEP_RESULTS_LOG.md"
        if log_path.exists():
            header = "| 指标 | " + " | ".join(names) + " |"
            sep_row = "|------|" + "|".join(["------"] * len(names)) + "|"
            block = ["", "### 对比结果（多 run，自动追加）", f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "", header, sep_row]
            for r in rows[1:]:
                block.append("| " + " | ".join(r) + " |")
            block.append("")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(block))
            print(f"\n[OK] 已追加到 {log_path}")


def main():
    ap = argparse.ArgumentParser(description="对比 last_run_metrics 结果（2 个或 3+ 个 run）")
    ap.add_argument("runs", nargs="*", help="run-name 或 json 路径，2 个为两两对比，3+ 个为多列表")
    ap.add_argument("--append-log", action="store_true", help="将对比表追加到 STEP_RESULTS_LOG.md")
    args = ap.parse_args()

    if len(args.runs) < 2:
        # 默认：node_holdout 三模型
        args.runs = ["gru_gcn_node_holdout", "gru_only_node_holdout", "hydraulic_inverse_node_holdout"]
        print("未指定 run，使用默认: " + " ".join(args.runs))

    loaded = [load_metrics(r) for r in args.runs]
    metrics_list = [x[0] for x in loaded]
    names = [x[1] for x in loaded]

    if len(metrics_list) == 2:
        run_two(metrics_list[0], names[0], metrics_list[1], names[1], args.append_log)
    else:
        run_multi(metrics_list, names, args.append_log)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
