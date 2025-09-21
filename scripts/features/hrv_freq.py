# 心率变异的频域特征（单段 RR；ln LF 与 ln HF）
#
# 设计原则（面向艺术疗愈/游戏体验等心理生理研究）：
# - 本模块只做“单段 RR”的频域特征提取，不负责切窗、批处理、质量门槛等上游逻辑；
# - 支持两种 HF 定义：固定带宽（0.15–0.40 Hz）或基于呼吸峰的“个体化 HF”（呼吸峰±半径，且限制在0.15–0.40 Hz内）；
# - 输入：
#     RR 段 DataFrame，列名 ['t_s','rr_ms']（已完成伪迹处理的 NN 间期；时间单位秒，RR 单位毫秒）；
#     可选：与该 RR 段对齐的呼吸 DataFrame，列名 ['t_s','resp']；
# - 输出：单行 DataFrame，列名 ['hf_log_ms2','lf_log_ms2','hf_band_used','hf_center_hz']；
#         若 log_power=False，则返回 ['hf_ms2','lf_ms2', ...]（未取对数）。

import pandas as pd
import numpy as np
from typing import Optional, Tuple

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
from scipy.signal import welch, get_window

# --------- helpers ---------

def _interpolate_rr(rr_ms: np.ndarray, fs_interp: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate irregular RR (ms) to evenly sampled tachogram (ms) at fs_interp (Hz)."""
    if rr_ms.size < 3:
        return np.array([]), np.array([])
    t = np.cumsum(rr_ms) / 1000.0
    t -= t[0]
    # uniform time grid
    t_uniform = np.arange(0.0, t[-1], 1.0 / fs_interp)
    if t_uniform.size < 16:
        return np.array([]), np.array([])
    rr_interp = np.interp(t_uniform, t, rr_ms)
    return t_uniform, rr_interp

def _welch_psd(rr_interp_ms: np.ndarray, fs_interp: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD of mean-removed tachogram (ms). One-sided PSD in (ms^2/Hz)."""
    if rr_interp_ms.size < 16:
        return np.array([]), np.array([])
    x = rr_interp_ms - np.mean(rr_interp_ms)
    # 256 points ≈ 64 s at 4 Hz; cap at series length
    nperseg = min(len(x), 256)
    f, pxx = welch(x, fs=fs_interp, nperseg=nperseg, window=get_window('hann', nperseg), detrend='constant')
    return f, pxx

def _band_power(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    """Integrate PSD within [lo, hi] Hz using trapezoidal rule; returns ms^2."""
    if f.size == 0 or pxx.size == 0:
        return np.nan
    mask = (f >= lo) & (f <= hi)
    if not np.any(mask):
        return np.nan
    return np.trapz(pxx[mask], f[mask])

def _estimate_resp_peak(resp_df: pd.DataFrame, lo: float = 0.1, hi: float = 0.5) -> Optional[float]:
    """Estimate respiration peak frequency (Hz) for a window from a respiration signal segment."""
    if resp_df is None or resp_df.empty:
        return None
    # Resp input expected columns: ['t_s','resp'] with near-uniform sampling
    t = resp_df['t_s'].to_numpy()
    x = resp_df['resp'].to_numpy()
    if t.size < 32:
        return None
    fs = 1.0 / np.median(np.diff(t))
    # Welch PSD
    nperseg = min(len(x), 512)
    f, pxx = welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    band = (f >= lo) & (f <= hi)
    if not np.any(band):
        return None
    idx = np.argmax(pxx[band])
    f_peaks = f[band]
    return float(f_peaks[idx])

# --------- 公共 API ---------

def features_segment(rr_df: pd.DataFrame,
                     resp_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    计算单段 RR 的频域特征：LF 与 HF（默认返回 ln(ms²)）。

    参数（切窗与质量控制请在上游完成；本函数只做频谱计算）：
    rr_df : DataFrame，列 ['t_s','rr_ms']。已去伪迹的 NN 间期；t_s 为秒，rr_ms 为毫秒。
    resp_df : （可选）DataFrame，列 ['t_s','resp']。与 rr_df 对齐的该段呼吸信号，用于个体化 HF；若缺失则使用固定 HF 带宽。

    说明：RR 间期序列本身**没有采样率**；本函数会从 PARAMS 读取 `fs_interp`（默认 4 Hz），
    仅用于将不等间隔 RR 插值到**等间隔的心动间期序列**，以便进行 Welch 频谱估计。这是频域 HRV 的常规做法。

    返回：
    单行 DataFrame：
      当 log_power=True：['hf_log_ms2','lf_log_ms2','hf_band_used','hf_center_hz']；
      当 log_power=False：['hf_ms2','lf_ms2','hf_band_used','hf_center_hz']。
    """
    # Read all knobs from PARAMS (单一真相源)
    fs_interp = float(PARAMS.get("fs_interp", 4.0))
    hf_fixed_lo, hf_fixed_hi = PARAMS.get("hf_band", (0.15, 0.40))
    use_individual = bool(PARAMS.get("use_individual_hf", False))
    rad = float(PARAMS.get("hf_band_radius_hz", 0.05))
    do_log = bool(PARAMS.get("log_power", True))

    # Check rr_df size
    if rr_df.empty or rr_df["rr_ms"].size < 3:
        # Not enough data to compute
        if do_log:
            return pd.DataFrame([{"hf_log_ms2": np.nan, "lf_log_ms2": np.nan, "hf_band_used": "fixed", "hf_center_hz": np.nan}])
        else:
            return pd.DataFrame([{"hf_ms2": np.nan, "lf_ms2": np.nan, "hf_band_used": "fixed", "hf_center_hz": np.nan}])

    # Interpolate RR tachogram
    _, rr_interp = _interpolate_rr(rr_df["rr_ms"].to_numpy(dtype=float), fs_interp=fs_interp)
    f, pxx = _welch_psd(rr_interp, fs_interp=fs_interp)

    band_used = "fixed"
    hf_center = np.nan

    # HF band selection
    if use_individual and (resp_df is not None) and (resp_df.shape[0] > 0):
        f0 = _estimate_resp_peak(resp_df)
        if f0 is not None and (hf_fixed_lo <= f0 <= hf_fixed_hi):
            lo = max(hf_fixed_lo, f0 - rad)
            hi = min(hf_fixed_hi, f0 + rad)
            band_used = "individual"
            hf_center = float(f0)
        else:
            lo, hi = hf_fixed_lo, hf_fixed_hi
    else:
        lo, hi = hf_fixed_lo, hf_fixed_hi

    hf_power = _band_power(f, pxx, lo, hi)
    lf_power = _band_power(f, pxx, 0.04, 0.15)

    # log-transform; guard against non-positive
    if do_log:
        hf_val = np.log(hf_power) if (hf_power is not None and hf_power > 0) else np.nan
        lf_val = np.log(lf_power) if (lf_power is not None and lf_power > 0) else np.nan
        out = {
            "hf_log_ms2": hf_val,
            "lf_log_ms2": lf_val,
            "hf_band_used": band_used,
            "hf_center_hz": hf_center
        }
    else:
        out = {
            "hf_ms2": hf_power,
            "lf_ms2": lf_power,
            "hf_band_used": band_used,
            "hf_center_hz": hf_center
        }

    return pd.DataFrame([out])

if __name__ == "__main__":
    import inspect
    print("Function signature for features_segment:")
    print(inspect.signature(features_segment))
