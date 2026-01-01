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

"""
时域心率变异特征（单段 RR）

设计原则（与 hrv_freq 对齐）：
- 仅计算单段 RR 的时域指标；不负责切窗、质量门槛、批处理与文件读写。
- 与 hrv_freq 保持一致的调用签名：features_segment(rr_df, resp_df=None)
  （resp_df 为占位参数，时域不会使用，但保留以便在外部循环中统一调用。）
- 所有可调口径从 settings.PARAMS 读取（单一真相源）。

输入
- rr_df: DataFrame，列 ['t_s','rr_ms']，已完成伪迹处理后的 NN 间期（毫秒）。
- resp_df: 可选；忽略，仅用于与频域模块的签名保持一致。

输出
- 单行 DataFrame，列：
  ['hr_bpm','rmssd_ms','sdnn_ms','pnn50_pct','sd1_ms','sd2_ms']

注意
- pNNx 的门槛 x 从 PARAMS['pnn_threshold_ms'] 读取（默认 50.0）。
- SDNN 的方差自由度从 PARAMS['sdnn_ddof'] 读取（默认 1，为样本标准差）。
"""

# --------- 基本指标函数 ---------

def _rmssd_ms(rr_ms: np.ndarray) -> float:
    if rr_ms.size < 2:
        return np.nan
    diff = np.diff(rr_ms)
    return float(np.sqrt(np.mean(diff ** 2)))


def _sdnn_ms(rr_ms: np.ndarray, ddof: int) -> float:
    if rr_ms.size < 2:
        return np.nan
    return float(np.std(rr_ms, ddof=ddof))


def _pnn_pct(rr_ms: np.ndarray, threshold_ms: float) -> float:
    if rr_ms.size < 2:
        return np.nan
    diff = np.abs(np.diff(rr_ms))
    return float(100.0 * (diff > threshold_ms).mean())


def _hr_bpm(rr_ms: np.ndarray) -> float:
    if rr_ms.size == 0:
        return np.nan
    mean_rr = float(np.mean(rr_ms))
    if mean_rr <= 0:
        return np.nan
    return 60000.0 / mean_rr


def _poincare_sd1_sd2(rr_ms: np.ndarray) -> tuple[float, float]:
    if rr_ms.size < 2:
        return (np.nan, np.nan)
    # 与 SDNN 保持一致：使用样本方差 ddof=1（可由 PARAMS['sdnn_ddof'] 控制）
    ddof = int(PARAMS.get('sdnn_ddof', 1))
    diff = np.diff(rr_ms)
    var_diff = np.var(diff, ddof=ddof)
    var_rr   = np.var(rr_ms, ddof=ddof)
    # 数值稳定性保护：极少数浮点误差下被开方项可能出现微负，截断到 0
    sd1_term = var_diff / 2.0
    sd2_term = 2.0 * var_rr - (var_diff / 2.0)
    sd1 = float(np.sqrt(sd1_term if sd1_term > 0.0 else 0.0))
    sd2 = float(np.sqrt(sd2_term if sd2_term > 0.0 else 0.0))
    return (sd1, sd2)


# --------- 公共 API（与 hrv_freq 一致的签名） ---------

def features_segment(rr_df: pd.DataFrame, resp_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    计算单段 RR 的时域特征。

    参数
    rr_df : DataFrame，列 ['t_s','rr_ms']；已去伪迹的 NN 间期（毫秒）。
    resp_df : 可选；占位符，与频域模块签名保持一致，时域不使用。

    返回
    单行 DataFrame：['hr_bpm','rmssd_ms','sdnn_ms','pnn50_pct','sd1_ms','sd2_ms']。
    """
    # 固定为 pNN50，避免口径歧义
    pnn_threshold = 50.0  # 固定为 pNN50，避免口径歧义
    sdnn_ddof = int(PARAMS.get('sdnn_ddof', 1))  # 1=样本标准差

    # 取 RR 数组
    rr = rr_df.get('rr_ms')
    if rr is None:
        # 容错：列名不对时直接返回 NaN 行
        return pd.DataFrame([{
            'hr_bpm': np.nan,
            'rmssd_ms': np.nan,
            'sdnn_ms': np.nan,
            'pnn50_pct': np.nan,
            'sd1_ms': np.nan,
            'sd2_ms': np.nan,
        }])
    rr = rr.to_numpy(dtype=float)

    # 计算指标
    mean_hr = _hr_bpm(rr)
    rmssd = _rmssd_ms(rr)
    sdnn = _sdnn_ms(rr, ddof=sdnn_ddof)
    pnn = _pnn_pct(rr, threshold_ms=pnn_threshold)
    sd1, sd2 = _poincare_sd1_sd2(rr)

    out = pd.DataFrame([{
        'hr_bpm': mean_hr,
        'rmssd_ms': rmssd,
        'sdnn_ms': sdnn,
        'pnn50_pct': pnn,
        'sd1_ms': sd1,
        'sd2_ms': sd2,
    }])
    return out


if __name__ == '__main__':
    # 轻量 smoke test 占位：避免被当作脚本直接跑
    print('features_segment(rr_df, resp_df=None) -> DataFrame[1 x 6]')