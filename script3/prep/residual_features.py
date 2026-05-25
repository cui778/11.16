"""
BASELINE 特征工程模块 - 在原始时序层面进行

核心思想：
. 从原始时序数据中提取 BASELINE 场景
2. 对所有缺陷场景计算残差特征：residual[t,n,f] = defect[t,n,f] - baseline[t,n,f]
3. 返回处理后的时序数据
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineFeatureEngineer:
    """在原始时序层面进行 BASELINE 特征工程"""
    
    def __init__(self, baseline_scenario_id: int = 0):
        """
        初始化
        
        Args:
            baseline_scenario_id: BASELINE 场景的 ID（通常为 0 或 'BASELINE'）
        """
        self.baseline_scenario_id = baseline_scenario_id
        self.baseline_df = None
        self.is_fitted = False
    
    def fit(self, node_timeseries_df: pd.DataFrame) -> Dict:
        """
        从原始时序数据中提取 BASELINE
        
        Args:
            node_timeseries_df: 原始时序数据 DataFrame
                列：['datetime', 'scenario_id', 'node_id', 'depth', 'total_inflow', ...]
        
        Returns:
            baseline_stats: 统计量字典
        """
        print("\n[特征工程] 从原始时序数据提取 BASELINE...")
        
        baseline_df = node_timeseries_df.copy()
        
        # ✅ 增强的BASELINE匹配逻辑（多层次回退）
        mask = None
        
        # 尝试1：直接整数匹配
        if isinstance(self.baseline_scenario_id, int):
            mask = baseline_df['scenario_id'] == self.baseline_scenario_id
            if mask.sum() > 0:
                print(f"  [匹配方式1] 整数匹配成功: scenario_id == {self.baseline_scenario_id}")
        
        # 尝试2：类型转换后匹配
        if mask is None or mask.sum() == 0:
            baseline_df['scenario_id_str'] = baseline_df['scenario_id'].astype(str)
            mask = baseline_df['scenario_id_str'] == str(self.baseline_scenario_id)
            if mask.sum() > 0:
                print(f"  [匹配方式2] 字符串匹配成功: scenario_id == '{self.baseline_scenario_id}'")
        
        # 尝试3：回退到 'BASELINE' 字符串
        if mask is None or mask.sum() == 0:
            mask = baseline_df['scenario_id'].astype(str).str.upper() == 'BASELINE'
            if mask.sum() > 0:
                print(f"  [匹配方式3] BASELINE字符串匹配成功")
        
        # 尝试4：回退到 scenario_id == 0
        if mask is None or mask.sum() == 0:
            mask = baseline_df['scenario_id'] == 0
            if mask.sum() > 0:
                print(f"  [匹配方式4] 整数0匹配成功")
        
        baseline_df = baseline_df[mask].copy() if mask is not None else baseline_df.iloc[0:0].copy()
        
        if len(baseline_df) == 0:
            logger.error(f"❌ 未找到 BASELINE 数据!")
            logger.error(f"   期望 scenario_id: {self.baseline_scenario_id}")
            logger.error(f"   实际 scenario_id 值: {node_timeseries_df['scenario_id'].unique()[:10]}")
            return None
        
        print(f"  ✅ 提取 BASELINE 数据: {len(baseline_df)} 条记录")
        
        # 获取特征列（排除 datetime, scenario_id, node_id, defect_type, time_step 等辅助列）
        feature_cols = [
            col
            for col in baseline_df.columns
            if col not in ['datetime', 'scenario_id', 'node_id', 'defect_type', 'time_step']
        ]
        
        # ✅ 确保特征列都是数值类型，排除非数值列
        valid_feature_cols = []
        for col in feature_cols:
            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
            # 检查是否成功转换为数值
            if baseline_df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                valid_feature_cols.append(col)
        
        feature_cols = valid_feature_cols
        
        # 保存 BASELINE 数据（保持原始格式：time_step, datetime, node_id, features）
        # 注意：time_step 仅用于对齐，不作为特征参与残差计算
        self.baseline_df = baseline_df[['time_step', 'datetime', 'node_id'] + feature_cols].copy()
        self.feature_cols = feature_cols
        self.is_fitted = True
        
        baseline_stats = {
            'baseline_df': self.baseline_df,
            'feature_cols': feature_cols
        }
        
        print(f"  ✅ BASELINE 数据提取完成:")
        print(f"     记录数: {len(self.baseline_df)}")
        print(f"     特征: {feature_cols}")
        
        return baseline_stats
    
    def transform(self, node_timeseries_df: pd.DataFrame, 
                  baseline_stats: Optional[Dict] = None) -> pd.DataFrame:
        """
        对时序数据应用特征工程（计算残差）
        
        residual[t, n, f] = defect[t, n, f] - baseline[t, n, f]
        
        Args:
            node_timeseries_df: 原始时序数据
            baseline_stats: BASELINE 统计量（如果为 None，使用 fit 的结果）
        
        Returns:
            处理后的时序数据（特征值替换为残差）
        """
        if baseline_stats is None:
            if not self.is_fitted:
                logger.warning("⚠️ 特征工程器未拟合，跳过转换")
                return node_timeseries_df
            baseline_stats = {
                'baseline_df': self.baseline_df,
                'feature_cols': self.feature_cols
            }
        
        print("\n[特征工程] 应用残差特征...")
        
        result_df = node_timeseries_df.copy()
        baseline_df = baseline_stats['baseline_df']
        feature_cols = baseline_stats['feature_cols']
        
        # ✅ 确保特征列都是数值类型
        for col in feature_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
        
        # 为每个特征计算残差与相对残差
        # 残差: residual = defect - baseline
        # 相对残差: residual_rel = residual / (|baseline| + eps)，便于弱信号在不同量级节点上可比
        rel_eps = 1e-6
        rel_clip = 10.0  # 避免 baseline≈0 时爆炸
        for feat_col in feature_cols:
            residual_col = f"{feat_col}_residual"
            rel_col = f"{feat_col}_residual_rel"

            merged = result_df[['time_step', 'node_id', feat_col]].merge(
                baseline_df[['time_step', 'node_id', feat_col]].rename(
                    columns={feat_col: f"{feat_col}_baseline"}
                ),
                on=['time_step', 'node_id'],
                how='left',
            )

            # 计算残差
            residual_vals = merged[feat_col] - merged[f"{feat_col}_baseline"]
            result_df[residual_col] = residual_vals
            # 相对残差：residual / (|baseline| + eps)，并裁剪防止极端值
            baseline_abs = np.abs(merged[f"{feat_col}_baseline"]) + rel_eps
            result_df[rel_col] = np.clip(residual_vals / baseline_abs, -rel_clip, rel_clip)
        
        # ✅ 保留原始特征列（不删除），这样 22 脚本可以继续使用
        # 输出格式：..., depth_residual, depth_residual_rel, ...
        
        print(f"  ✅ 特征工程完成: 添加了 {len(feature_cols)} 个残差 + {len(feature_cols)} 个相对残差特征列")
        print(f"     残差特征列: {[f'{col}_residual' for col in feature_cols]}")
        print(f"     相对残差特征列: {[f'{col}_residual_rel' for col in feature_cols]}")
        
        return result_df


def apply_baseline_feature_engineering(
    node_timeseries_file: str,
    baseline_scenario_id: int = 0
) -> Tuple[pd.DataFrame, Dict]:
    """便利函数：一步完成特征工程

    这里增加一个关键改动：
    - 在每个 scenario 内按 datetime 排序，生成一个整数时间步序号 time_step
    - 残差对齐时使用 (time_step, node_id) 而不是 (datetime, node_id)，
      避免因为秒级时间差导致大量无法对齐。

    Args:
        node_timeseries_file: 原始时序数据文件路径
        baseline_scenario_id: BASELINE 场景 ID

    Returns:
        (处理后的时序数据, 统计量字典)
    """
    print("=" * 80)
    print("[特征工程] BASELINE 特征工程")
    print("=" * 80)

    # 加载数据
    print("\n[步骤1] 加载原始时序数据...")
    if node_timeseries_file.endswith('.parquet'):
        node_timeseries_df = pd.read_parquet(node_timeseries_file)
    else:
        node_timeseries_df = pd.read_csv(node_timeseries_file)

    node_timeseries_df['datetime'] = pd.to_datetime(node_timeseries_df['datetime'])

    # 为每个 scenario 生成 time_step 序号：同一 scenario 内按 datetime 排序，从 0 开始递增
    node_timeseries_df = node_timeseries_df.sort_values(['scenario_id', 'datetime', 'node_id'])
    node_timeseries_df['time_step'] = (
        node_timeseries_df.groupby('scenario_id')['datetime']
        .rank(method='dense')
        .astype(int) - 1
    )

    print(f"  ✅ 加载完成: {len(node_timeseries_df)} 条记录 (已生成 time_step 序号)")
    
    # 拟合 BASELINE
    print("\n[步骤2] 拟合 BASELINE...")
    engineer = BaselineFeatureEngineer(baseline_scenario_id=baseline_scenario_id)
    baseline_stats = engineer.fit(node_timeseries_df)
    
    if baseline_stats is None:
        logger.warning("⚠️ BASELINE 拟合失败，返回原始数据")
        return node_timeseries_df, None
    
    # 应用特征工程
    print("\n[步骤3] 应用特征工程...")
    result_df = engineer.transform(node_timeseries_df, baseline_stats)
    
    print("\n" + "=" * 80)
    print("✅ 特征工程完成")
    print("=" * 80)
    
    return result_df, baseline_stats


def apply_baseline_feature_engineering_generic(
    timeseries_file: str,
    id_col: str,
    baseline_scenario_id: int = 0,
):
    if timeseries_file.endswith('.parquet'):
        df = pd.read_parquet(timeseries_file)
    else:
        df = pd.read_csv(timeseries_file)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['scenario_id', 'datetime', id_col])
    df['time_step'] = (
        df.groupby('scenario_id')['datetime']
        .rank(method='dense')
        .astype(int) - 1
    )

    baseline_df = df[df['scenario_id'].astype(str) == str(baseline_scenario_id)].copy()
    if len(baseline_df) == 0:
        baseline_df = df[df['scenario_id'] == baseline_scenario_id].copy()
    if len(baseline_df) == 0:
        return None, None

    feature_cols = [
        col for col in df.columns
        if col not in ['datetime', 'scenario_id', id_col, 'defect_type', 'time_step']
    ]
    valid_feature_cols = []
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
        if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            valid_feature_cols.append(col)
    feature_cols = valid_feature_cols

    rel_eps = 1e-6
    rel_clip = 10.0
    result_df = df.copy()
    baseline_df = baseline_df[['time_step', id_col] + feature_cols].copy()

    for feat_col in feature_cols:
        residual_col = f"{feat_col}_residual"
        rel_col = f"{feat_col}_residual_rel"
        merged = result_df[['time_step', id_col, feat_col]].merge(
            baseline_df[['time_step', id_col, feat_col]].rename(columns={feat_col: f"{feat_col}_baseline"}),
            on=['time_step', id_col],
            how='left',
        )
        residual_vals = merged[feat_col] - merged[f"{feat_col}_baseline"]
        result_df[residual_col] = residual_vals
        baseline_abs = np.abs(merged[f"{feat_col}_baseline"]) + rel_eps
        result_df[rel_col] = np.clip(residual_vals / baseline_abs, -rel_clip, rel_clip)

    stats = {'feature_cols': feature_cols, 'id_col': id_col}
    return result_df, stats


if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from config import Config
    cfg = Config()
    import os

    parser = argparse.ArgumentParser(description="BASELINE 残差特征工程")
    parser.add_argument("--input-dir", default="",
                       help="输入目录（含 node_timeseries.parquet）；默认用 config.training_data_dir")
    parser.add_argument("--output-dir", default="",
                       help="输出目录（写 node_timeseries_with_residuals.parquet）；默认与 input-dir 相同")
    args = parser.parse_args()

    data_dir = args.input_dir or cfg.training_data_dir
    out_dir = args.output_dir or data_dir
    input_file = os.path.join(data_dir, "node_timeseries.parquet")
    output_path = os.path.join(out_dir, "node_timeseries_with_residuals.parquet")

    print(f"  输入: {input_file}")
    print(f"  输出: {output_path}")

    result_df, baseline_stats = apply_baseline_feature_engineering(
        node_timeseries_file=input_file,
        baseline_scenario_id=0
    )
    if result_df is None:
        raise SystemExit("BASELINE 拟合失败")
    print("\n结果数据形状:", result_df.shape)
    print("结果列:", result_df.columns.tolist())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n保存处理后的时序数据到: {output_path}")
    result_df.to_parquet(output_path, compression='gzip', index=False)
    print("✅ 保存完成")
