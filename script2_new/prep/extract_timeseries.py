"""
修复版时序数据提取 - 使用固定时间间隔采样

核心改变：
  ❌ 旧版：每100步采样 → 时间不对齐
  ✅ 新版：每10分钟采样 → 所有场景时间完全对齐

这样确保：
  - Baseline 00:10:00 采样
  - 缺陷场景 00:10:00 采样
  - 时间完全一致，残差计算无空值
"""

from pyswmm import Simulation, Nodes, Links
import argparse
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_scenario_fixed_time(inp_path, defect_id, defect_type, defect_node, defect_flow,
                                   sample_interval_minutes=10, key_nodes_only=False,
                                   defect_start_time=None, defect_end_time=None, defect_duration_minutes=None,
                                   monitor_node_ids=None):
    """
    使用固定时间间隔采样（而不是固定步数）
    ✅ P1-1 新增：支持时间门控缺陷注入

    参数：
        inp_path：INP 文件路径
        defect_id：缺陷ID
        defect_type：缺陷类型
        defect_node：缺陷节点ID
        defect_flow：缺陷流量
        sample_interval_minutes：采样间隔（分钟）
        key_nodes_only：是否只提取关键节点
        defect_start_time：缺陷开始时间 (datetime) - P1-1新增
        defect_end_time：缺陷结束时间 (datetime) - P1-1新增
        defect_duration_minutes：缺陷持续时间（分钟）- P1-1新增

    返回：
        node_data, link_data
    """
    node_records = []
    link_records = []

    try:
        with Simulation(inp_path) as sim:
            nodes = Nodes(sim)
            links = Links(sim)

            # 🔧 关键修复1：预先收集所有节点和管道ID
            logger.info("  收集节点和管道ID...")
            node_ids = []
            for node in nodes:
                try:
                    node_ids.append(str(node.nodeid))
                except:
                    pass

            link_ids = []
            for link in links:
                try:
                    link_ids.append(str(link.linkid))
                except:
                    pass

            logger.info(f"  节点数: {len(node_ids)}, 管道数: {len(link_ids)}")

            if monitor_node_ids is not None:
                # Only keep the fixed monitor-node set.
                keep = set(monitor_node_ids)
                node_ids = [n for n in node_ids if n in keep]
                logger.info(f"  仅提取监测节点: {len(node_ids)} 个")
            elif key_nodes_only and defect_node and defect_node in node_ids:
                node_ids = [defect_node]
                logger.info(f"  仅提取关键节点: {defect_node}")

            # 🔧 关键修复2：基于时间采样而不是步数
            start_time = sim.start_time
            end_time = sim.end_time

            # 生成目标采样时间列表
            target_times = []
            current_target = start_time + timedelta(minutes=sample_interval_minutes)
            while current_target <= end_time:
                target_times.append(current_target)
                current_target += timedelta(minutes=sample_interval_minutes)

            logger.info(f"  目标采样点: {len(target_times)} 个 (间隔 {sample_interval_minutes} 分钟)")
            logger.info(f"  时间范围: {start_time} → {end_time}")

            # ✅ P1-1 新增：时间门控缺陷注入逻辑
            # 确定缺陷起止时间
            if defect_start_time and defect_end_time:
                # 直接使用指定的时间范围
                actual_start_time = defect_start_time
                actual_end_time = defect_end_time
                logger.info(f"  缺陷时间窗口: {actual_start_time} → {actual_end_time}")
            elif defect_start_time and defect_duration_minutes:
                # 从开始时间+持续时间计算结束时间
                actual_start_time = defect_start_time
                actual_end_time = defect_start_time + timedelta(minutes=defect_duration_minutes)
                logger.info(
                    f"  缺陷时间窗口: {actual_start_time} → {actual_end_time} (持续{defect_duration_minutes}分钟)")
            else:
                # 默认：整个仿真期间都有缺陷
                actual_start_time = start_time
                actual_end_time = end_time
                logger.info(f"  缺陷时间窗口: 全程 (默认行为)")

            # 验证时间窗口合理性
            if actual_start_time < start_time:
                actual_start_time = start_time
                logger.warning(f"  缺陷开始时间早于仿真开始时间，调整为: {actual_start_time}")
            if actual_end_time > end_time:
                actual_end_time = end_time
                logger.warning(f"  缺陷结束时间晚于仿真结束时间，调整为: {actual_end_time}")

            target_idx = 0
            step_count = 0
            extract_count = 0
            defect_active = False
            defect_injected = False

            for step in sim:
                step_count += 1

                # ✅ P1-1 时间门控缺陷注入
                # 检查是否在缺陷时间窗口内
                was_defect_active = defect_active
                defect_active = actual_start_time <= sim.current_time <= actual_end_time

                # 缺陷状态变化时的处理
                if defect_node and defect_node in node_ids:
                    if defect_active and not was_defect_active:
                        # 缺陷开始：开始注入
                        try:
                            nodes[defect_node].generated_inflow(defect_flow)
                            defect_injected = True
                            logger.info(
                                f"  开始注入缺陷: 时间={sim.current_time}, 节点={defect_node}, 流量={defect_flow}")
                        except Exception as e:
                            logger.warning(f"  注入缺陷失败: {e}")
                    elif not defect_active and was_defect_active:
                        # 缺陷结束：停止注入（设为0）
                        try:
                            nodes[defect_node].generated_inflow(0)
                            logger.info(f"  停止注入缺陷: 时间={sim.current_time}, 节点={defect_node}")
                        except Exception as e:
                            logger.warning(f"  停止缺陷注入失败: {e}")
                    elif defect_active and defect_injected and (step_count % 100 == 0):
                        # 缺陷持续期间：定期重新注入（确保持续生效）
                        try:
                            nodes[defect_node].generated_inflow(defect_flow)
                        except Exception as e:
                            pass  # 静默处理，避免日志刷屏

                # 检查是否到达目标时间
                # 如果当前时间 >= 下一个目标时间，则采样
                if target_idx < len(target_times) and sim.current_time >= target_times[target_idx]:
                    # 提取节点数据
                    nodes_extracted = 0
                    for node_id in node_ids:
                        try:
                            node = nodes[node_id]

                            node_record = {
                                'datetime': target_times[target_idx],  # 使用目标时间而非current_time
                                'scenario_id': defect_id,
                                'defect_type': defect_type,
                                'node_id': node_id,
                                'depth': float(node.depth) if hasattr(node, 'depth') else 0,
                                'head': float(node.head) if hasattr(node, 'head') else 0,
                                'volume': float(node.volume) if hasattr(node, 'volume') else 0,
                                'lateral_inflow': float(node.lateral_inflow) if hasattr(node, 'lateral_inflow') else 0,
                                'total_inflow': float(node.total_inflow) if hasattr(node, 'total_inflow') else 0,
                                'total_outflow': float(node.total_outflow) if hasattr(node, 'total_outflow') else 0,
                                'flooding': float(node.flooding) if hasattr(node, 'flooding') else 0,
                            }

                            # 水质数据
                            try:
                                if hasattr(node, 'pollut_quality'):
                                    for pollutant in ['BODf', 'BODs', 'NH4', 'NO3', 'DO', 'TSSs']:
                                        try:
                                            conc = node.pollut_quality.get(pollutant, None)
                                            node_record[f'pollut_{pollutant}'] = float(conc) if conc is not None else 0
                                        except:
                                            node_record[f'pollut_{pollutant}'] = 0
                            except:
                                pass

                            node_records.append(node_record)
                            nodes_extracted += 1
                        except Exception as e:
                            pass

                    # 提取管道数据
                    links_extracted = 0
                    for link_id in link_ids:
                        try:
                            link = links[link_id]

                            link_record = {
                                'datetime': target_times[target_idx],
                                'scenario_id': defect_id,
                                'defect_type': defect_type,
                                'link_id': link_id,
                                'flow': float(link.flow) if hasattr(link, 'flow') else 0,
                                'depth': float(link.depth) if hasattr(link, 'depth') else 0,
                                'velocity': float(link.velocity) if hasattr(link, 'velocity') else 0,
                            }

                            link_records.append(link_record)
                            links_extracted += 1
                        except:
                            pass

                    extract_count += 1

                    # 第一次采样时显示详细信息
                    if extract_count == 1:
                        logger.info(f"  首次采样: 时间={target_times[target_idx]}, "
                                    f"节点={nodes_extracted}/{len(node_ids)}, "
                                    f"管道={links_extracted}/{len(link_ids)}")

                    # 移动到下一个目标时间
                    target_idx += 1

                    # 进度显示
                    if extract_count % 20 == 0:
                        logger.info(f"  采样进度: {extract_count}/{len(target_times)} "
                                    f"({extract_count / len(target_times) * 100:.1f}%), "
                                    f"仿真步数: {step_count}")

            logger.info(f"  完成: 总步数={step_count}, 采样次数={extract_count}/{len(target_times)}, "
                        f"节点记录={len(node_records)}, 管道记录={len(link_records)}")

            # 检查是否所有目标时间都被采样
            if extract_count < len(target_times):
                logger.warning(f"  ⚠️  未完成所有采样: {extract_count}/{len(target_times)}")
                logger.warning(f"  可能原因: 仿真提前结束或采样间隔太小")

        df_nodes = pd.DataFrame(node_records) if len(node_records) > 0 else pd.DataFrame()
        df_links = pd.DataFrame(link_records) if len(link_records) > 0 else pd.DataFrame()

        return df_nodes, df_links

    except Exception as e:
        logger.error(f"  仿真失败: {e}")


def simulate_and_extract_timeseries_fixed_time(base_inp_path, defect_csv_path,
                                               output_dir="simulation_data_fixed_time",
                                               sample_interval_minutes=10,
                                               max_scenarios=None,
                                               key_nodes_only=False,
                                               time_gated=False,
                                               monitor_node_ids=None,
                                               monitor_nodes_file_path=None):
    """
    使用固定时间间隔提取所有场景的数据

    参数：
        base_inp_path：基线 INP 文件路径
        defect_csv_path：缺陷矩阵 CSV 文件路径
        output_dir：输出目录
        sample_interval_minutes：采样间隔（分钟）
        max_scenarios：最多处理多少个场景
        key_nodes_only：是否只提取关键节点
        time_gated：是否启用时间门控（默认：False）
    """

    os.makedirs(output_dir, exist_ok=True)

    # 读取缺陷矩阵
    df_defects = pd.read_csv(defect_csv_path)
    df_defects = df_defects.dropna(subset=['node_id'])

    if max_scenarios:
        df_defects = df_defects.head(max_scenarios)
        logger.info(f"⚠️  测试模式: 只处理前 {max_scenarios} 个场景")

    if len(df_defects) == 0:
        logger.warning("⚠️  没有有效的缺陷场景")
        return None, None, None

    logger.info(f"开始模拟并提取数据: {len(df_defects)} 个缺陷场景")
    logger.info(f"采样间隔: 每 {sample_interval_minutes} 分钟")
    if monitor_node_ids is not None:
        logger.info(f"仅提取监测节点: {len(monitor_node_ids)} 个")
    logger.info("=" * 80)

    all_node_data = []
    scenario_summaries = []
    extraction_stats = {
        'total_scenarios': len(df_defects) + 1,
        'successful': 0,
        'failed': 0,
        'total_node_records': 0,
        'total_link_records': 0
    }

    temp_inp = base_inp_path.replace('.inp', '_temp.inp')

    # 1. 基线场景
    logger.info("\n[基线场景] 运行无缺陷仿真...")
    baseline_node_data, baseline_link_data = run_single_scenario_fixed_time(
        inp_path=base_inp_path,
        defect_id=0,
        defect_type="BASELINE",
        defect_node=None,
        defect_flow=0,
        sample_interval_minutes=sample_interval_minutes,
        key_nodes_only=False,
        monitor_node_ids=monitor_node_ids
    )

    if baseline_node_data is not None and len(baseline_node_data) > 0:
        all_node_data.append(baseline_node_data)
        scenario_summaries.append({
            'scenario_id': 0,
            'defect_type': 'BASELINE',
            'node_id': 'N/A',
            'defect_flow': 0,
            'total_records': len(baseline_node_data),
            'unique_times': baseline_node_data['datetime'].nunique(),
            'status': 'Completed'
        })
        extraction_stats['successful'] += 1
        extraction_stats['total_node_records'] += len(baseline_node_data)
        logger.info(f"✓ 基线场景完成: {len(baseline_node_data)} 条节点记录, "
                    f"{baseline_node_data['datetime'].nunique()} 个唯一时间点")
    else:
        logger.warning("⚠️  基线场景未提取到数据")
        extraction_stats['failed'] += 1
        return None, None, None

    for idx, row in df_defects.iterrows():
        defect_id = int(row["defect_id"])
        defect_type = row["defect_type"]
        node_id = str(row["node_id"])
        flow = row["flow"]

        if time_gated:
            # 时间门控参数解析
            start_hour = float(row.get("start_hour", 0))  # 默认从0小时开始
            duration_h = float(row.get("duration_h", 24))  # 默认持续24小时

            # 修复时间基准：匹配SWMM仿真时间（2025年）
            start_time = datetime(2025, 1, 1, int(round(start_hour)), 0, 0)  # 修复：使用正确的年份
            duration_minutes = int(duration_h * 60)
        else:
            start_hour = None
            duration_h = None
            start_time = None
            duration_minutes = None

        logger.info(f"\n[场景 {idx + 1}/{len(df_defects)}] 缺陷ID={defect_id}, "
                    f"Type={defect_type}, Node={node_id}, Flow={flow}")
        if time_gated:
            logger.info(f"  时间门控: {start_hour}:00 开始，持续 {duration_h} 小时")
        else:
            logger.info("  时间门控: 关闭（全程注入）")

        shutil.copy(base_inp_path, temp_inp)

        try:
            node_data, link_data = run_single_scenario_fixed_time(
                inp_path=temp_inp,
                defect_id=defect_id,
                defect_type=defect_type,
                defect_node=node_id,
                defect_flow=flow,
                sample_interval_minutes=sample_interval_minutes,
                key_nodes_only=key_nodes_only,
                defect_start_time=start_time,
                defect_duration_minutes=duration_minutes,
                monitor_node_ids=monitor_node_ids
            )

            if node_data is not None and len(node_data) > 0:
                all_node_data.append(node_data)
                scenario_summaries.append({
                    'scenario_id': defect_id,
                    'defect_type': defect_type,
                    'node_id': node_id,
                    'defect_flow': flow,
                    'total_records': len(node_data),
                    'unique_times': node_data['datetime'].nunique(),
                    'status': 'Completed'
                })
                extraction_stats['successful'] += 1
                extraction_stats['total_node_records'] += len(node_data)
                logger.info(f"✓ 提取了 {len(node_data)} 条节点记录, "
                            f"{node_data['datetime'].nunique()} 个唯一时间点")
            else:
                logger.warning(f"⚠️  未提取到数据")
                extraction_stats['failed'] += 1

        except Exception as e:
            logger.error(f"✗ 错误: {e}")
            extraction_stats['failed'] += 1
            continue

        finally:
            if os.path.exists(temp_inp):
                try:
                    os.remove(temp_inp)
                except:
                    pass

    # 3. 合并数据
    logger.info("\n" + "=" * 80)
    logger.info("汇总并保存数据...")

    if len(all_node_data) > 0:
        df_nodes_all = pd.concat(all_node_data, ignore_index=True)

        # 保存
        node_output_parquet = os.path.join(output_dir, "node_timeseries.parquet")
        df_nodes_all.to_parquet(node_output_parquet, index=False, compression='gzip')
        logger.info(f"✓ 节点数据已保存: {node_output_parquet}")

        # 验证时间对齐
        logger.info(f"\n时间对齐验证:")
        logger.info(f"  总记录数: {len(df_nodes_all):,}")
        logger.info(f"  节点数: {df_nodes_all['node_id'].nunique()}")
        logger.info(f"  场景数: {df_nodes_all['scenario_id'].nunique()}")

        # 检查每个场景的时间点数
        time_counts = df_nodes_all.groupby('scenario_id')['datetime'].nunique()
        logger.info(f"\n每个场景的时间点数:")
        logger.info(f"  最小: {time_counts.min()}")
        logger.info(f"  最大: {time_counts.max()}")
        logger.info(f"  平均: {time_counts.mean():.1f}")

        if time_counts.min() == time_counts.max():
            logger.info(f"  ✓ 所有场景的时间点数完全一致!")
        else:
            logger.warning(f"  ⚠️  场景间时间点数不一致，差异: {time_counts.max() - time_counts.min()}")

        # 保存场景汇总
        df_summary = pd.DataFrame(scenario_summaries)
        summary_output = os.path.join(output_dir, "scenario_summary.csv")
        df_summary.to_csv(summary_output, index=False, encoding='utf-8-sig')
        logger.info(f"✓ 场景汇总已保存: {summary_output}")

        manifest = {
            "base_inp_path": os.path.normpath(base_inp_path),
            "defect_csv_path": os.path.normpath(defect_csv_path),
            "output_dir": os.path.normpath(output_dir),
            "sample_interval_minutes": int(sample_interval_minutes),
            "max_scenarios": None if max_scenarios is None else int(max_scenarios),
            "time_gated": bool(time_gated),
            "key_nodes_only": bool(key_nodes_only),
            "strict_monitor_only": monitor_node_ids is not None,
            "monitor_node_count": 0 if monitor_node_ids is None else int(len(monitor_node_ids)),
            "monitor_nodes_file": "" if not monitor_nodes_file_path else os.path.normpath(str(monitor_nodes_file_path)),
            "node_timeseries_file": os.path.normpath(node_output_parquet),
            "scenario_summary_file": os.path.normpath(summary_output),
            "scenario_count": int(df_nodes_all['scenario_id'].nunique()),
            "unique_node_count": int(df_nodes_all['node_id'].nunique()),
        }
        manifest_output = os.path.join(output_dir, "dataset_manifest.json")
        with open(manifest_output, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"[OK] dataset manifest 宸蹭繚瀛? {manifest_output}")

        return df_nodes_all, None, df_summary
    else:
        logger.warning("⚠️  没有成功提取任何数据")
        return None, None, None


if __name__ == "__main__":
    import sys
    from pathlib import Path as P
    _root = P(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from config import Config
    _cfg = Config()

    parser = argparse.ArgumentParser(description="固定时间间隔采样 - 时序数据提取")
    parser.add_argument("--base-inp", default=_cfg.inp_file)
    parser.add_argument("--defect-csv", default="", help="默认从 config.defect_matrix_file 读取")
    parser.add_argument("--output-dir", default=_cfg.training_data_dir)
    parser.add_argument("--sample-interval-minutes", type=int, default=10)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--key-nodes-only", action="store_true")
    parser.add_argument("--time-gated", action="store_true",
                        help="启用时间门控（使用start_hour/duration_h）")
    parser.add_argument("--monitor-nodes-file", default="",
                        help="监测节点 JSON 文件路径（含 monitor_nodes 字段）；"
                             "指定后只提取监测节点的时序数据，大幅减少数据量。"
                             "默认从 config.monitor_nodes_file 读取（若文件存在）。")
    parser.add_argument("--full-nodes", action="store_true",
                        help="强制提取全网节点（用于 observability 等需完整影响范围的场景）")
    args = parser.parse_args()

    defect_csv_path = args.defect_csv or _cfg.defect_matrix_file
    output_dir = args.output_dir or _cfg.training_data_dir

    # 解析监测节点
    import json as _json
    monitor_node_ids = None
    if args.full_nodes:
        print("  节点范围: 全网（--full-nodes）")
    else:
        monitor_file = args.monitor_nodes_file or _cfg.monitor_nodes_file
        if monitor_file and P(monitor_file).exists():
            with open(monitor_file, 'r', encoding='utf-8') as _f:
                _mdata = _json.load(_f)
            monitor_node_ids = [str(n) for n in _mdata.get('monitor_nodes', _mdata.get('selected_nodes', []))]
            print(f"  监测节点文件: {monitor_file}")
            print(f"  监测节点数: {len(monitor_node_ids)}")
        elif args.monitor_nodes_file:
            print(f"  ⚠️  监测节点文件不存在: {args.monitor_nodes_file}，将提取全网节点")

    print("=" * 80)
    print("固定时间间隔采样 - 时序数据提取")
    print("=" * 80)
    print("\n核心改进:")
    print("  ❌ 旧版: 每100步采样 → 时间不对齐")
    print("  ✅ 新版: 每10分钟采样 → 时间完全对齐")
    print("\n当前配置:")
    print(f"  缺陷矩阵: {defect_csv_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  采样间隔: {args.sample_interval_minutes} 分钟")
    print(f"  处理场景: {args.max_scenarios if args.max_scenarios else '全部'}")
    print(f"  时间门控: {'开启' if args.time_gated else '关闭（全程注入）'}")
    print(f"  节点范围: {'监测节点 ' + str(len(monitor_node_ids)) + ' 个' if monitor_node_ids else '全网节点'}")

    node_data, _, summary = simulate_and_extract_timeseries_fixed_time(
        base_inp_path=args.base_inp,
        defect_csv_path=defect_csv_path,
        output_dir=output_dir,
        sample_interval_minutes=args.sample_interval_minutes,
        max_scenarios=args.max_scenarios,
        key_nodes_only=args.key_nodes_only,
        time_gated=args.time_gated,
        monitor_node_ids=monitor_node_ids,
        monitor_nodes_file_path=(monitor_file if monitor_node_ids is not None else "")
    )

    if node_data is not None:
        print("\n" + "=" * 80)
        print("✓ 提取完成！")
        print(f"数据位置: {output_dir}/")
        print("  - node_timeseries.parquet: 节点时序数据（时间完全对齐）")
        print("  - scenario_summary.csv: 场景汇总")
        print("=" * 80)
