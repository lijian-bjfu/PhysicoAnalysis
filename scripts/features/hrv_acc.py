import pandas as pd
import numpy as np
from typing import Optional

import sys
from pathlib import Path

# --- project-root bootstrap ---
_p = Path(__file__).resolve()
for _ in range(6):
    if (_p.parent / "settings.py").exists():
        sys.path.insert(0, str(_p.parent)); break
    _p = _p.parent
# --- end bootstrap ---
from settings import PARAMS


def _stdcol_acc(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """标准化加速度列名为 ['t_s','ax','ay','az']；若列不足返回 None。"""
    if df is None or df.empty:
        return None
    acc = df.copy()
    # 时间列：既支持 't_s' 也支持 'time_s'
    if 't_s' in acc.columns:
        pass
    elif 'time_s' in acc.columns:
        acc = acc.rename(columns={'time_s': 't_s'})
    else:
        return None
    # 三轴列
    if {'ax', 'ay', 'az'}.issubset(acc.columns):
        cols = ['ax', 'ay', 'az']
    elif {'value_x', 'value_y', 'value_z'}.issubset(acc.columns):
        acc = acc.rename(columns={'value_x': 'ax', 'value_y': 'ay', 'value_z': 'az'})
        cols = ['ax', 'ay', 'az']
    else:
        return None
    return acc[['t_s'] + cols].astype(float)


def features_segment(
    rr_df: Optional[pd.DataFrame] = None,
    resp_df: Optional[pd.DataFrame] = None,
    *,
    acc_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    计算单窗口加速度**两项协变量**（用于统计模型的时间变化协变量）。
    为兼容主流程保留统一接口：
      - 推荐：传 acc_df=...
      - 兼容：传 resp_df=...（旧统一接口的第二参数）或 rr_df=...

    输入列（任选其一）：
      - time_s/t_s, value_x/value_y/value_z  或  t_s, ax/ay/az

    输出列（仅两项）：
      - acc_enmo_mean    : ENMO 均值，ENMO = max(||a|| - acc_g, 0)
      - acc_motion_frac  : ENMO 超过阈值的时间占比（阈值见 PARAMS.acc_enmo_thresh）

    仅依赖以下参数（来自 settings.PARAMS）：
      - acc_g            : 1 g 的数值（与原始单位一致，mg 系统常设 1000.0）
      - acc_enmo_thresh  : ENMO 阈值（mg），用于计算占比
    """
    # 解析数据来源：优先 acc_df，其次 resp_df，最后 rr_df（为兼容旧调用）
    src = acc_df if acc_df is not None else (resp_df if resp_df is not None else rr_df)
    acc = _stdcol_acc(src)

    if acc is None or acc.empty:
        return pd.DataFrame([{
            # 以下两个值与 settings signal_features 的值一致
            # 去掉重力后的整体动作强度。静止时接近 0；随动作幅度增大而升高
            'acc_enmo_mean': np.nan,
            # 这一窗里，有多少比例的时间处在“明显在动”的状态。0 表示几乎全程静止；1 表示几乎全程超过阈值地在动
            'acc_motion_frac': np.nan
        }])

    # 参数（单位以你的数据为准；默认 mg 系统，1g≈1000）
    g_val = float(PARAMS.get('acc_g', 1000.0))
    enmo_thr = float(PARAMS.get('acc_enmo_thresh', 30.0))

    ax = acc['ax'].to_numpy()
    ay = acc['ay'].to_numpy()
    az = acc['az'].to_numpy()

    # 向量模与 ENMO
    vm = np.sqrt(ax*ax + ay*ay + az*az)
    enmo = np.maximum(vm - g_val, 0.0)

    out = {
        'acc_enmo_mean': float(np.nanmean(enmo)) if enmo.size else np.nan,
        'acc_motion_frac': float(np.mean(enmo > enmo_thr)) if enmo.size else np.nan,
    }
    return pd.DataFrame([out])


def features_segment_acc(acc_df: pd.DataFrame) -> pd.DataFrame:
    """语义明确的便捷别名。等价于 features_segment(acc_df=acc_df)。"""
    return features_segment(acc_df=acc_df)


if __name__ == '__main__':
    print('features_segment(acc_df=...) -> DataFrame[1x2] (acc_enmo_mean, acc_motion_frac)')