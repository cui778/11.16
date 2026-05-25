"""
统一缺陷矩阵生成器

功能:
1) 生成缺陷矩阵 (I/E/P)
2) 可选: 生成时间门控字段(start_hour, duration_h)并支持多样化策略

说明:
- 已内置 RealisticDefectMatrixGenerator，不依赖其他脚本
- 默认输出包含 start_hour / duration_h
- 如需“无时间门控”模式，可用 --time-mode fixed
"""

import argparse
import json
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
from pyswmm import Simulation, Nodes, Links


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealisticDefectMatrixGenerator:
    """真实场景缺陷矩阵生成器"""

    def __init__(self, inp_path, json_path=None, candidate_nodes_file=None):
        self.inp_path = inp_path
        self.json_path = json_path or inp_path.parent / 'parsed_inp_data.json'
        self.candidate_nodes_file = candidate_nodes_file
        self.node_baseline_flows = {}
        self.link_baseline_flows = {}
        self.link_properties = {}

    def calculate_baseline(self, sample_interval=100):
        logger.info(f"开始baseline仿真: {self.inp_path}")

        try:
            with Simulation(str(self.inp_path)) as sim:
                nodes = Nodes(sim)
                links = Links(sim)

                logger.info("收集节点和管道ID...")
                node_ids = []
                for node in nodes:
                    try:
                        node_ids.append(node.nodeid)
                    except:
                        pass

                link_ids = []
                for link in links:
                    try:
                        link_ids.append(str(link.linkid))
                    except:
                        pass

                logger.info(f"  节点数: {len(node_ids)}, 管道数: {len(link_ids)}")

                step_count = 0
                extract_count = 0

                logger.info("开始baseline仿真...")

                for step in sim:
                    step_count += 1

                    if step_count % sample_interval != 0:
                        continue

                    extract_count += 1

                    for node_id in node_ids:
                        try:
                            node_obj = nodes[node_id]
                            flow = getattr(node_obj, 'total_inflow', 0.0)
                            if node_id not in self.node_baseline_flows:
                                self.node_baseline_flows[node_id] = {
                                    'values': [],
                                    'sum': 0.0,
                                    'count': 0
                                }

                            self.node_baseline_flows[node_id]['values'].append(flow)
                            self.node_baseline_flows[node_id]['sum'] += flow
                            self.node_baseline_flows[node_id]['count'] += 1
                        except:
                            continue

                    for link_id in link_ids:
                        try:
                            link_obj = links[link_id]
                            flow = getattr(link_obj, 'flow', 0.0)
                            if link_id not in self.link_baseline_flows:
                                self.link_baseline_flows[link_id] = {
                                    'values': [],
                                    'sum': 0.0,
                                    'count': 0
                                }

                            self.link_baseline_flows[link_id]['values'].append(flow)
                            self.link_baseline_flows[link_id]['sum'] += flow
                            self.link_baseline_flows[link_id]['count'] += 1
                        except:
                            continue

                    if extract_count % 10 == 0:
                        logger.info(f"  已提取 {extract_count} 个时间步")

            logger.info("计算baseline统计量...")

            for node_id, data in self.node_baseline_flows.items():
                if data['count'] > 0:
                    values = data['values']
                    self.node_baseline_flows[node_id] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }

            for link_id, data in self.link_baseline_flows.items():
                if data['count'] > 0:
                    values = data['values']
                    self.link_baseline_flows[link_id] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }

            self._collect_link_properties()

            logger.info("✓ Baseline计算完成")
            logger.info(f"  节点: {len(self.node_baseline_flows)}个")
            logger.info(f"  管道: {len(self.link_baseline_flows)}个")

            return True

        except Exception as e:
            logger.error(f"✗ Baseline计算失败: {e}")
            return False

    def _collect_link_properties(self):
        try:
            with Simulation(str(self.inp_path)) as sim:
                links = Links(sim)

                for link in links:
                    try:
                        link_id = str(link.linkid)
                        # PySWMM Link 没有 from_node/to_node，正确属性是 connections=(inlet, outlet)
                        # 或分别用 inlet_node / outlet_node
                        connections = getattr(link, 'connections', None)
                        if connections and len(connections) == 2:
                            from_node = str(connections[0])
                            to_node = str(connections[1])
                        else:
                            from_node = getattr(link, 'inlet_node', None)
                            to_node = getattr(link, 'outlet_node', None)
                            if from_node is not None:
                                from_node = str(from_node)
                            if to_node is not None:
                                to_node = str(to_node)
                        length = getattr(link, 'length', 0.0)

                        self.link_properties[link_id] = {
                            'from_node': from_node,
                            'to_node': to_node,
                            'length': length
                        }
                    except:
                        continue

        except Exception as e:
            logger.warning(f"收集管道属性失败: {e}")

    def generate_diverse_defect_matrix(
            self,
            num_scenarios=100,
            k_def=None,
            defect_type_weights=None,
            intensity_choices=None,
            time_choices=None,
            duration_choices=None,
            intensity_range=(10, 50),
            time_range=(12, 120),
            duration_range=(6, 24),
            manual_selection=None,
            output_csv=None
    ):
        logger.info("\n" + "=" * 80)
        logger.info("步骤2: 生成多样化缺陷矩阵")
        logger.info("=" * 80)

        random.seed(42)
        np.random.seed(42)

        if not self.node_baseline_flows:
            logger.error("✗ 请先运行 calculate_baseline()")
            return None

        if defect_type_weights is None:
            defect_type_weights = {'I': 0.4, 'E': 0.4, 'P': 0.2}

        candidates = self._select_defect_candidates(manual_selection, k_def)

        logger.info(f"\n缺陷候选位置:")
        logger.info(f"  I类型(渗入): {len(candidates['I'])}个节点")
        logger.info(f"  E类型(渗漏): {len(candidates['E'])}条管道")
        logger.info(f"  P类型(点源): {len(candidates['P'])}个节点")

        rows = []

        type_counts = {
            'I': int(num_scenarios * defect_type_weights['I']),
            'E': int(num_scenarios * defect_type_weights['E']),
            'P': int(num_scenarios * defect_type_weights['P'])
        }
        type_counts['I'] += num_scenarios - sum(type_counts.values())

        logger.info("\n场景分布:")
        for dtype, count in type_counts.items():
            logger.info(f"  {dtype}类型: {count}个场景")

        if not candidates['E'] and type_counts.get('E', 0) > 0:
            raise ValueError(
                "E类型管道候选为空（没有管道的 to_node 落在候选节点集 C 中），"
                "无法生成 E 类缺陷场景。\n"
                "请检查：\n"
                "  1. build_candidates.py 生成的候选节点集是否合理\n"
                "  2. 管道属性中 to_node 的字段名/类型是否与候选节点 ID 一致\n"
                "  3. 或者调整 defect_type_weights 将 E 类权重设为 0"
            )

        defect_id = 1

        for defect_type, count in type_counts.items():
            logger.info(f"\n生成{defect_type}类型缺陷 ({count}个场景):")

            for i in range(count):
                if defect_type == 'E':
                    item = random.choice(candidates['E'])
                    node_id = self.link_properties[item]['to_node']
                    link_id = item
                    base_flow = self.link_baseline_flows[item]['mean']
                else:
                    item = random.choice(candidates[defect_type])
                    node_id = item
                    link_id = None
                    base_flow = self.node_baseline_flows[item]['mean']

                if intensity_choices:
                    intensity = random.choice(intensity_choices)
                else:
                    intensity = random.randint(*intensity_range)

                if time_choices:
                    start_hour = random.choice(time_choices)
                else:
                    start_hour = random.randint(*time_range)

                if duration_choices:
                    duration = random.choice(duration_choices)
                else:
                    duration = random.randint(*duration_range)

                if defect_type == "I":
                    flow = round(intensity * base_flow * 0.01, 4)
                    BODf = round(random.uniform(intensity * 3, intensity * 7), 2)
                    NH4 = round(random.uniform(intensity * 0.2, intensity * 0.5), 2)
                    DO = round(random.uniform(intensity * 0.05, intensity * 0.15), 2)

                elif defect_type == "E":
                    flow = round(-intensity * base_flow * 0.01, 4)
                    BODf = 0.0
                    NH4 = 0.0
                    DO = 0.0

                elif defect_type == "P":
                    flow = round(intensity * base_flow * 0.002, 4)
                    BODf = round(random.uniform(intensity * 20, intensity * 40), 2)
                    NH4 = round(random.uniform(intensity * 3, intensity * 7), 2)
                    DO = round(random.uniform(intensity * 0.3, intensity * 0.7), 2)

                rows.append({
                    "defect_id": defect_id,
                    "defect_type": defect_type,
                    "node_id": node_id,
                    "link_id": link_id,
                    "intensity_pct": intensity,
                    "start_hour": start_hour,
                    "duration_h": duration,
                    "flow": flow,
                    "BODf": BODf,
                    "NH4": NH4,
                    "DO": DO,
                    "baseline_flow": round(base_flow, 4)
                })

                if (i + 1) % 20 == 0 or i == count - 1:
                    logger.info(f"  已生成 {i + 1}/{count} 个场景")

                defect_id += 1

        df = pd.DataFrame(rows)
        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')

        logger.info(f"\n✓ 缺陷矩阵已生成: {len(df)}")
        self._print_matrix_stats(df)

        return df

    def _select_defect_candidates(self, manual_selection=None, k_def=None):
        if manual_selection:
            return manual_selection

        candidates = {'I': [], 'E': [], 'P': []}

        candidate_path = Path(self.candidate_nodes_file) if getattr(self, 'candidate_nodes_file', None) else (
            Path(self.inp_path).resolve().parent.parent / "input_1" / "candidate_nodes_new.json"
        )

        candidate_nodes = None
        if candidate_path.exists():
            logger.info(f"✓ 使用候选节点文件: {candidate_path}")
            with open(candidate_path, 'r', encoding='utf-8') as f:
                cdata = json.load(f)
            raw_candidates = set(cdata.get('candidate_nodes', []))

            baseline_nodes = set(self.node_baseline_flows.keys())
            candidate_nodes = list(raw_candidates & baseline_nodes)

            if not candidate_nodes:
                logger.warning("⚠️ 候选节点池与baseline统计无交集，将退回全网筛选")
                candidate_nodes = None
            else:
                logger.info(f"✓ 加载候选节点池: {len(candidate_nodes)}个节点")
                logger.info(f"   原始候选节点: {len(raw_candidates)}个")
                logger.info(f"   与baseline交集: {len(candidate_nodes)}个")
        else:
            logger.warning(f"⚠️ 未找到候选节点文件: {candidate_path}，将退回全网筛选")

        if candidate_nodes is not None:
            if k_def is not None and k_def > 0:
                if k_def > len(candidate_nodes):
                    logger.warning(f"⚠️ k_def({k_def}) > 候选节点池大小({len(candidate_nodes)})，将使用所有候选节点")
                    actual_defect_nodes = list(candidate_nodes)
                else:
                    actual_defect_nodes = random.sample(list(candidate_nodes), k_def)
                    logger.info(f"✓ 从{len(candidate_nodes)}个候选节点中随机选择了{k_def}个作为实际缺陷位置")
                    logger.info(
                        f"  实际缺陷节点: {actual_defect_nodes[:10]}{'...' if len(actual_defect_nodes) > 10 else ''}")

                candidates['I'] = actual_defect_nodes
                candidates['P'] = actual_defect_nodes
            else:
                candidates['I'] = list(candidate_nodes)
                candidates['P'] = list(candidate_nodes)
        else:
            node_flows = [(nid, stats['mean']) for nid, stats in self.node_baseline_flows.items()]
            node_flows.sort(key=lambda x: x[1], reverse=True)
            top_half = max(1, len(node_flows) // 2)
            candidates['I'] = [nid for nid, _ in node_flows[:top_half]]
            candidates['P'] = list(self.node_baseline_flows.keys())

        valid_links = []
        for lid, stats in self.link_baseline_flows.items():
            length = self.link_properties.get(lid, {}).get('length', 0)
            flow = stats['mean']
            score = length * flow

            to_node = self.link_properties.get(lid, {}).get('to_node')

            if candidate_nodes is not None:
                if to_node and to_node in candidate_nodes:
                    valid_links.append((lid, score))
            else:
                valid_links.append((lid, score))

        if valid_links:
            valid_links.sort(key=lambda x: x[1], reverse=True)
            candidates['E'] = [lid for lid, _ in valid_links]

            if candidate_nodes is not None:
                logger.info(f"✓ E类型管道: {len(candidates['E'])}个，全部to_node在候选节点中")
        else:
            logger.warning("⚠️ 没有找到符合条件的E类型管道")
            candidates['E'] = []

        return candidates

    def _print_matrix_stats(self, df):
        logger.info("\n缺陷矩阵统计:")
        logger.info("=" * 80)

        logger.info("\n类型分布:")
        for dtype in ['I', 'E', 'P']:
            count = len(df[df['defect_type'] == dtype])
            pct = count / len(df) * 100
            logger.info(f"  {dtype}类型: {count}个 ({pct:.1f}%)")

        logger.info("\n强度分布:")
        logger.info(f"  最小: {df['intensity_pct'].min()}%")
        logger.info(f"  最大: {df['intensity_pct'].max()}%")
        logger.info(f"  平均: {df['intensity_pct'].mean():.1f}%")
        logger.info(f"  中位数: {df['intensity_pct'].median():.1f}%")

        logger.info("\n时间分布:")
        logger.info(f"  开始时间: {df['start_hour'].min()}-{df['start_hour'].max()}小时")
        logger.info(f"  持续时间: {df['duration_h'].min()}-{df['duration_h'].max()}小时")

        logger.info("\n流量范围:")
        for dtype in ['I', 'E', 'P']:
            df_type = df[df['defect_type'] == dtype]
            if len(df_type) > 0:
                logger.info(f"  {dtype}类型: [{df_type['flow'].min():.4f}, {df_type['flow'].max():.4f}] L/s")

    def save_baseline_stats(self, output_path):
        stats = {
            'nodes': self.node_baseline_flows,
            'links': self.link_baseline_flows,
            'link_properties': self.link_properties
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Baseline统计已保存: {output_path}")

    def load_baseline_stats(self, input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        self.node_baseline_flows = stats['nodes']
        self.link_baseline_flows = stats['links']
        self.link_properties = stats.get('link_properties', {})

        logger.info(f"✓ 已加载baseline统计: {input_path}")
        logger.info(f"  节点: {len(self.node_baseline_flows)}个")
        logger.info(f"  管道: {len(self.link_baseline_flows)}个")


def apply_time_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """根据策略生成时间字段"""
    n_scenarios = len(df)

    if strategy == "fixed":
        start_hours = np.full(n_scenarios, 12)
        durations = np.full(n_scenarios, 24)
    elif strategy == "diverse":
        start_hours = np.random.choice(range(0, 24), size=n_scenarios, replace=True)
        durations = np.random.choice([6, 12, 18, 24], size=n_scenarios, replace=True)
    elif strategy == "realistic":
        hour_weights = np.array([
            0.5, 0.3, 0.2, 0.1, 0.1, 0.2,
            0.8, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 0.8,
            0.6, 0.4, 0.3, 0.2, 0.1, 0.5
        ], dtype=float)
        hour_weights = hour_weights / hour_weights.sum()
        start_hours = np.random.choice(list(range(24)), size=n_scenarios, replace=True, p=hour_weights)
        durations = np.random.choice([4, 6, 8, 12, 16, 24], size=n_scenarios, replace=True,
                                     p=[0.1, 0.2, 0.2, 0.3, 0.1, 0.1])
    elif strategy == "layered":
        time_slots = [(0, 6), (6, 12), (12, 18), (18, 24)]
        start_hours = []
        durations = []
        for i in range(n_scenarios):
            slot = time_slots[i % len(time_slots)]
            start_hours.append(np.random.randint(slot[0], slot[1]))
            durations.append(np.random.choice([6, 12, 18, 24]))
        start_hours = np.array(start_hours)
        durations = np.array(durations)
    else:
        raise ValueError(f"Unsupported time strategy: {strategy}")

    df = df.copy()
    df["start_hour"] = start_hours
    df["duration_h"] = durations
    return df


def interactive_main():
    """交互式主函数：让用户选择时间模式和输出文件名"""
    print("=" * 60)
    print("统一缺陷矩阵生成器 - 交互模式")
    print("=" * 60)

    # 时间模式选择
    print("\n请选择时间模式:")
    print("  1. fixed      - 固定时间 (12:00开始, 持续24h)")
    print("  2. diverse    - 多样化时间 (全天均匀分布)")
    print("  3. realistic  - 真实故障分布 (白天多, 夜间少)")
    print("  4. layered    - 分层分布 (各时间段均匀覆盖)")
    
    mode_map = {"1": "fixed", "2": "diverse", "3": "realistic", "4": "layered"}
    choice = input("\n请输入选择 (1-4): ").strip()
    time_mode = mode_map.get(choice, "diverse")

    # 输出文件名
    default_name = f"defect_matrix_{time_mode}.csv"
    output_name = input(f"\n输出文件名 (默认: {default_name}): ").strip()
    if not output_name:
        output_name = default_name

    # 场景数
    num_scenarios = input("场景数 (默认: 100): ").strip()
    num_scenarios = int(num_scenarios) if num_scenarios.isdigit() else 100

    print(f"\n配置确认:")
    print(f"  时间模式: {time_mode}")
    print(f"  输出文件: {output_name}")
    print(f"  场景数: {num_scenarios}")

    confirm = input("\n确认执行? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 调用生成函数
    generate_matrix(
        output_name=output_name,
        time_mode=time_mode,
        num_scenarios=num_scenarios
    )


def generate_matrix(output_name="defect_matrix_diverse.csv",
                   time_mode="diverse",
                   num_scenarios=100,
                   k_def=40,
                   seed=42):
    """实际生成矩阵的函数（路径从 config 读取）"""
    import sys
    from pathlib import Path as P
    _root = P(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from config import Config
    cfg = Config()
    base_inp = Path(cfg.inp_file)
    json_path = Path(cfg.parsed_inp_data_file)
    baseline_stats = Path(cfg.baseline_flow_stats_file)
    output_dir = Path(cfg.input_dir_new)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    print(f"\n[INFO] 开始生成缺陷矩阵...")
    print(f"[INFO] 时间模式: {time_mode}")
    print(f"[INFO] 场景数: {num_scenarios}")

    generator = RealisticDefectMatrixGenerator(base_inp, json_path, candidate_nodes_file=cfg.candidate_nodes_file)
    generator.load_baseline_stats(baseline_stats)

    df = generator.generate_diverse_defect_matrix(
        num_scenarios=num_scenarios,
        k_def=k_def,
        defect_type_weights={"I": 0.4, "E": 0.4, "P": 0.2},
        intensity_choices=[30, 40, 50],
        time_range=(12, 120),
        duration_range=(6, 24),
        manual_selection=None,
        output_csv=None,
    )

    df = apply_time_strategy(df, time_mode)

    output_path = output_dir / output_name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 缺陷矩阵已输出: {output_path}")
    print(f"[OK] 时间策略: {time_mode}, 场景数: {len(df)}")


def main():
    import sys
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from config import Config
    cfg = Config()

    parser = argparse.ArgumentParser(description="统一缺陷矩阵生成器")
    parser.add_argument("--base-inp", default=cfg.inp_file)
    parser.add_argument("--json-path", default=cfg.parsed_inp_data_file)
    parser.add_argument("--baseline-stats", default=cfg.baseline_flow_stats_file)
    parser.add_argument("--output-dir", default=cfg.input_dir_new)
    parser.add_argument("--output-name", default="defect_matrix_base.csv")
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--k-def", type=int, default=40)
    parser.add_argument("--time-mode", choices=["fixed", "diverse", "realistic", "layered"], default="diverse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-only", action="store_true",
                        help="仅跑无缺陷 SWMM 仿真并保存 baseline_flow_stats.json，不生成缺陷矩阵")
    args = parser.parse_args()

    np.random.seed(args.seed)

    base_inp = Path(args.base_inp)
    json_path = Path(args.json_path)
    baseline_stats = Path(args.baseline_stats)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = RealisticDefectMatrixGenerator(base_inp, json_path, candidate_nodes_file=cfg.candidate_nodes_file)

    if args.baseline_only:
        logger.info("仅生成 baseline 统计（无缺陷仿真）...")
        baseline_stats.parent.mkdir(parents=True, exist_ok=True)
        ok = generator.calculate_baseline(sample_interval=100)
        if ok:
            generator.save_baseline_stats(str(baseline_stats))
            print(f"[OK] 已保存: {baseline_stats}")
        else:
            raise RuntimeError("baseline 仿真失败")
        return

    generator.load_baseline_stats(baseline_stats)

    df = generator.generate_diverse_defect_matrix(
        num_scenarios=args.num_scenarios,
        k_def=args.k_def,
        defect_type_weights={"I": 0.4, "E": 0.4, "P": 0.2},
        intensity_choices=[30, 40, 50],
        time_range=(12, 120),
        duration_range=(6, 24),
        manual_selection=None,
        output_csv=None,
    )

    df = apply_time_strategy(df, args.time_mode)

    output_path = output_dir / args.output_name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 缺陷矩阵已输出: {output_path}")
    print(f"[OK] 时间策略: {args.time_mode}, 场景数: {len(df)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # 命令行模式
        main()
    else:
        # 交互模式
        interactive_main()
